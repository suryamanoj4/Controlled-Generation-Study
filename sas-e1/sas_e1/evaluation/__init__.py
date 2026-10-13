"""Evaluation: classifiers, steered generation, coherence metrics, optional judge."""
from .classifiers import EmbeddingLRClassifier, TfidfStyleClassifier, evaluate_classifier
from .metrics import (
    perplexity,
    strength,
    distinct_2,
    repetition_4,
    content_similarity_pairs,
    compute_sfwc,
    alpha_target,
    alpha_break,
)

__all__ = [
    "EmbeddingLRClassifier",
    "TfidfStyleClassifier",
    "evaluate_classifier",
    "generate_condition",
    "perplexity",
    "strength",
    "distinct_2",
    "repetition_4",
    "content_similarity_pairs",
    "compute_sfwc",
    "alpha_target",
    "alpha_break",
]


def __getattr__(name: str):
    # lazy: pulling in generate_condition imports torch-heavy models.py
    if name == "generate_condition":
        from .generate import generate_condition

        return generate_condition
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")