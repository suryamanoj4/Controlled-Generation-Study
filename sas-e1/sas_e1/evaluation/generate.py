"""Steered and baseline generation for one experimental condition."""
from __future__ import annotations

from typing import Optional, Sequence

from ..models import generate_texts
from ..steering.operators import Steerer


def generate_condition(
    model,
    tokenizer,
    prompts: Sequence[str],
    steerer: Optional[Steerer] = None,
    layer_idx: Optional[int] = None,
    alpha: float = 0.0,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 0.95,
    batch_size: int = 8,
    seed: int = 0,
    device=None,
) -> list[str]:
    """Generate continuations; `steerer=None` means unsteered baseline."""
    steer_fn = None
    if steerer is not None and alpha != 0.0:
        steer_fn = steerer.torch_fn(alpha=alpha)
    return generate_texts(
        model,
        tokenizer,
        prompts,
        layer_idx=layer_idx,
        steer_fn=steer_fn,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
