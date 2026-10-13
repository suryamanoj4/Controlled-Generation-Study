"""Linear steering operators: h' = h + alpha * (W h + b)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Steerer:
    """A (possibly rank-limited) linear steering operator plus bias channel.

    Vector steerers (CAA) have W=None; closed-form/trained operators carry a dense W.
    `alpha_unit` is the effect-norm calibration scale (see steering/calibration.py).
    """

    name: str
    b: np.ndarray
    W: Optional[np.ndarray] = None
    alpha_unit: float = 1.0
    meta: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    # ---------------------------------------------------------- application
    @property
    def d(self) -> int:
        return int(self.b.shape[0])

    def displacement(self, X: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.W is None:
            D = np.broadcast_to(self.b[None, :], (X.shape[0], self.d))
        else:
            D = X @ self.W.T + self.b[None, :]
        return (alpha * D).astype(np.float32)

    def apply(self, X: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return (np.asarray(X, dtype=np.float32) + self.displacement(X, alpha)).astype(np.float32)

    def torch_fn(self, alpha: float = 1.0):
        """Return a function h -> h' operating on torch tensors (with cached weights)."""
        import torch

        cache: dict = {}

        def fn(h: "torch.Tensor"):
            dev = h.device
            if cache.get("dev") is not dev:
                cache["W"] = (
                    torch.as_tensor(self.W, dtype=torch.float32, device=dev)
                    if self.W is not None
                    else None
                )
                cache["b"] = torch.as_tensor(self.b, dtype=torch.float32, device=dev)
                cache["dev"] = dev
            dtype = h.dtype
            hf = h.float()
            if cache["W"] is None:
                out = hf + alpha * cache["b"]
            else:
                out = hf + alpha * (hf @ cache["W"].T + cache["b"])
            return out.to(dtype)

        return fn

    # ---------------------------------------------------------- serialization
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"b": self.b.astype(np.float32), "alpha_unit": np.float64(self.alpha_unit)}
        if self.W is not None:
            arrays["W"] = self.W.astype(np.float32)
        for key, value in self.extra.items():
            arrays[f"extra__{key}"] = np.asarray(value)
        np.savez(path, meta=json.dumps({"name": self.name, **self.meta}), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Steerer":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            name = meta.pop("name")
            extra = {}
            for key in z.files:
                if key.startswith("extra__"):
                    extra[key[len("extra__") :]] = z[key]
            return cls(
                name=name,
                b=z["b"].astype(np.float32),
                W=z["W"].astype(np.float32) if "W" in z.files else None,
                alpha_unit=float(z["alpha_unit"]),
                meta=meta,
                extra=extra,
            )


def no_op_steerer(d: int, name: str = "E") -> Steerer:
    return Steerer(name=name, b=np.zeros(d, dtype=np.float32), W=None, meta={"kind": "identity"})
