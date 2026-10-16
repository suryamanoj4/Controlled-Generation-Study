"""Model loading, layer resolution, hooks, activation extraction, generation."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """dtype policy: 'auto' = bf16 when the GPU supports it, else fp32.

    BLOOM-family models are bf16-native; fp16 inference overflows on some GPUs
    (e.g., T4), producing inf/NaN hidden states and sampling failures.
    """
    if name == "auto":
        if device.type == "cuda" and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float32
    dtype = DTYPES.get(name, torch.float32)
    if device.type == "cpu":
        return torch.float32
    return dtype


# ---------------------------------------------------------------- loading

def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(cfg) -> tuple:
    name = cfg.model.name
    device = resolve_device(getattr(cfg.model, "device", "auto"))
    dtype = _resolve_dtype(str(getattr(cfg.model, "dtype", "auto")), device)
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:  # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only generation
    return model, tokenizer, device


# ---------------------------------------------------------------- layer plumbing

def block_list(model) -> torch.nn.ModuleList:
    """Find the decoder block list, across common architectures."""
    candidates = [
        ("transformer", "h"),        # BLOOM, GPT-2
        ("model", "layers"),         # Llama, Qwen, Mistral
        ("gpt_neox", "layers"),      # Pythia
        ("transformer", "blocks"),   # MPT
        ("model", "decoder", "layers"),
    ]
    for path in candidates:
        obj = model
        ok = True
        for part in path:
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and hasattr(obj, "__len__") and len(obj) > 0:
            return obj
    raise ValueError("Could not locate decoder blocks for this model architecture")


def resolve_layer(model, layer: Optional[int] = None, fraction: float = 0.5) -> int:
    n = len(block_list(model))
    idx = int(layer) if layer is not None else int(round(fraction * n))
    return max(0, min(n - 1, idx))


def _first_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported module output type: {type(output)}")


def _replace_first(output, new_tensor):
    if isinstance(output, torch.Tensor):
        return new_tensor
    if isinstance(output, (tuple, list)):
        return type(output)([new_tensor, *list(output)[1:]])
    raise TypeError(f"Unsupported module output type: {type(output)}")


@contextmanager
def capture_module_output(module, store: list):
    """Append the output hidden states of `module` to `store` for each forward pass."""

    def hook(_mod, _inputs, output):
        store.append(_first_tensor(output).detach())

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def steer_module(module, fn: Callable[[torch.Tensor], torch.Tensor]):
    """Apply `fn` to the output hidden states of `module` in every forward pass."""

    def hook(_mod, _inputs, output):
        return _replace_first(output, fn(_first_tensor(output)))

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# ---------------------------------------------------------------- extraction

def _iter_batches(items: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def extract_doc_vectors(
    model,
    tokenizer,
    texts: Sequence[str],
    layer_idx: int,
    batch_size: int = 8,
    max_length: int = 256,
    device: Optional[torch.device] = None,
    exclude_special: bool = True,
) -> np.ndarray:
    """Mean-pooled residual-stream vectors after block `layer_idx`, one per document."""
    device = device or next(model.parameters()).device
    module = block_list(model)[layer_idx]
    special_ids = {
        tid
        for tid in (
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        )
        if tid is not None
    }
    out = []
    with torch.no_grad():
        for batch in _iter_batches(list(texts), batch_size):
            enc = tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            store: list = []
            with capture_module_output(module, store):
                model(**enc)
            h = store[0].float()  # (B, T, d)
            mask = enc["attention_mask"].bool()
            if exclude_special:
                for tid in special_ids:
                    mask = mask & (enc["input_ids"] != tid)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
            out.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


# ---------------------------------------------------------------- generation

def generate_texts(
    model,
    tokenizer,
    prompts: Sequence[str],
    layer_idx: Optional[int] = None,
    steer_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 0.95,
    batch_size: int = 8,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> list[str]:
    """Free generation; if `steer_fn` is given, it is applied at `layer_idx` each step."""
    device = device or next(model.parameters()).device
    torch.manual_seed(seed)
    module = block_list(model)[layer_idx] if (steer_fn is not None and layer_idx is not None) else None
    generations: list[str] = []
    with torch.no_grad():
        for batch in _iter_batches(list(prompts), batch_size):
            enc = tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)
            ctx = steer_module(module, steer_fn) if module is not None else _null_ctx()
            with ctx:
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            generations.extend(
                tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            )
    return generations


@contextmanager
def _null_ctx():
    yield


# ---------------------------------------------------------------- sanity

def identity_check(model, tokenizer, layer_idx: int, prompts: Sequence[str], device=None) -> float:
    """Max |logits| difference between an identity hook and no hook (should be ~0)."""
    device = device or next(model.parameters()).device
    enc = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True).to(device)
    module = block_list(model)[layer_idx]
    with torch.no_grad():
        base = model(**enc).logits.float()
        with steer_module(module, lambda h: h):
            hooked = model(**enc).logits.float()
    return float((base - hooked).abs().max().item())
