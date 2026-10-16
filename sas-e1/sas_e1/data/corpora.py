"""Corpus loading, splits, and the matched-contrast protocol (design doc §4.2)."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from ..config import get
from ..utils import mtld, read_jsonl, ttr

_FILE_FOR = {
    "sst2_positive": "sst2_positive.jsonl",
    "sst2_negative": "sst2_negative.jsonl",
    "civil_toxic": "civil_toxic.jsonl",
    "civil_benign": "civil_benign.jsonl",
    "wikipedia": "wikipedia.jsonl",
    "moral_stories": "moral_stories.jsonl",
    "shakespeare": "shakespeare.jsonl",
    "early_modern_drama": "early_modern_drama.jsonl",
    "neutral_wikipedia": "neutral_wikipedia.jsonl",
    "wikisource_premchand": "wikisource_premchand.jsonl",
    "wikisource_prasad": "wikisource_prasad.jsonl",
    "wikisource_gurajada": "wikisource_gurajada.jsonl",
    "wikisource_veeresalingam": "wikisource_veeresalingam.jsonl",
    "hindi_sent_positive": "hindi_sent_positive.jsonl",
    "hindi_sent_negative": "hindi_sent_negative.jsonl",
    "telugu_sent_positive": "telugu_sent_positive.jsonl",
    "telugu_sent_negative": "telugu_sent_negative.jsonl",
    "toxigen_toxic": "toxigen_toxic.jsonl",
    "toxigen_benign": "toxigen_benign.jsonl",
}


def load_pool(cfg, name: str) -> list[dict]:
    path = Path(get(cfg, "paths.data_processed", "data/processed")) / _FILE_FOR[name]
    if not path.exists():
        raise FileNotFoundError(f"missing corpus {path} — run scripts/download_assets.py first")
    return read_jsonl(path)


def attach_token_counts(docs: list[dict], tokenizer, max_length: int = 256) -> list[dict]:
    """Add exact tokenizer token counts (truncated at max_length) to each doc."""
    for doc in docs:
        ids = tokenizer(doc["text"], add_special_tokens=False)["input_ids"]
        doc["n_tokens"] = min(len(ids), max_length)
    return docs


# ---------------------------------------------------------------- splits

def _shuffle(seq: list, seed: int) -> list:
    rng = random.Random(seed)
    out = list(seq)
    rng.shuffle(out)
    return out


def split_pool(
    pos: list[dict],
    neg: list[dict],
    n_pool: int,
    n_clf_train: int,
    n_clf_eval: int,
    seed: int = 0,
) -> dict:
    """Reserve disjoint classifier sets; return an oversized steering pool per side.

    The pool is intentionally larger than the final stats size because matching drops
    short/long documents afterwards. Keys: pool_pos/pool_neg + clf_*_train/_eval.
    """
    out: dict[str, list[dict]] = {}
    for label, docs in (("pos", pos), ("neg", neg)):
        docs = _shuffle(docs, seed + (0 if label == "pos" else 1))
        total = len(docs)
        need = n_pool + n_clf_train + n_clf_eval
        if total < need:
            print(f"[split] WARNING: {label} has {total} docs, wanted {need} — proportional split")
            scale = total / need
            p = max(1, int(n_pool * scale))
            t = max(1, int(n_clf_train * scale))
            e = total - p - t
            if e < 1:
                p = max(1, total - 2)
                t = 1 if total >= 2 else 0
                e = total - p - t
        else:
            p, t, e = n_pool, n_clf_train, n_clf_eval
        out[f"pool_{label}"] = docs[:p]
        out[f"clf_{label}_train"] = docs[p : p + t]
        out[f"clf_{label}_eval"] = docs[p + t : p + t + e]
    return out


def split_pool_by_source(
    pos: list[dict],
    neg: list[dict],
    n_pool: int,
    n_clf_train: int,
    n_clf_eval: int,
    seed: int = 0,
) -> dict:
    """Like `split_pool`, but whole `source` groups (e.g., plays) stay together.

    `n_pool` is advisory here: all groups not reserved for the classifier sets go
    into the steering pool, and matching/trimming happens afterwards.
    """
    out: dict[str, list[dict]] = {}
    for label, docs in (("pos", pos), ("neg", neg)):
        groups: dict[str, list[dict]] = {}
        for d in docs:
            groups.setdefault(d.get("source", "unknown"), []).append(d)
        keys = _shuffle(list(groups), seed + (0 if label == "pos" else 1))
        n_groups = len(keys)
        n_eval_groups = min(2, max(1, n_groups // 3))
        n_train_groups = min(2, max(1, (n_groups - n_eval_groups) // 3))
        eval_keys = keys[:n_eval_groups]
        train_keys = keys[n_eval_groups : n_eval_groups + n_train_groups]
        pool_keys = keys[n_eval_groups + n_train_groups :]
        out[f"pool_{label}"] = [d for k in pool_keys for d in groups[k]]
        out[f"clf_{label}_train"] = [d for k in train_keys for d in groups[k]]
        out[f"clf_{label}_eval"] = [d for k in eval_keys for d in groups[k]]
    return out


# ---------------------------------------------------------------- matching

def match_corpora(
    pos: list[dict],
    neg: list[dict],
    seed: int = 0,
    max_len_percentile: float = 90.0,
    min_tokens: int = 4,
) -> tuple[list[dict], list[dict], dict]:
    """Equalize token volume and length distributions between positive and contrast.

    Steps: drop docs below `min_tokens` or above the min of both sides' percentile caps;
    then greedily remove docs from the longer side until totals are within one doc.
    """
    rng = random.Random(seed)

    def _clean(docs):
        lens = np.array([d["n_tokens"] for d in docs], dtype=np.float64)
        cap = float(np.percentile(lens, max_len_percentile)) if lens.size else 0.0
        return [d for d in docs if min_tokens <= d["n_tokens"] <= cap], lens

    pos_c, pos_lens = _clean(pos)
    neg_c, neg_lens = _clean(neg)
    cap = min(
        float(np.percentile([d["n_tokens"] for d in pos_c], max_len_percentile)) if pos_c else 0,
        float(np.percentile([d["n_tokens"] for d in neg_c], max_len_percentile)) if neg_c else 0,
    )
    pos_c = [d for d in pos_c if d["n_tokens"] <= cap]
    neg_c = [d for d in neg_c if d["n_tokens"] <= cap]

    def _total(docs):
        return int(sum(d["n_tokens"] for d in docs))

    pos_c, neg_c = _shuffle(pos_c, seed), _shuffle(neg_c, seed + 1)
    t_pos, t_neg = _total(pos_c), _total(neg_c)
    while t_pos != t_neg and pos_c and neg_c:
        if t_pos > t_neg:
            doc = pos_c.pop()
            t_pos -= doc["n_tokens"]
            if abs(t_pos - t_neg) > abs((t_pos + doc["n_tokens"]) - t_neg):
                pos_c.append(doc)
                t_pos += doc["n_tokens"]
                break
        else:
            doc = neg_c.pop()
            t_neg -= doc["n_tokens"]
            if abs(t_neg - t_pos) > abs((t_neg + doc["n_tokens"]) - t_pos):
                neg_c.append(doc)
                t_neg += doc["n_tokens"]
                break

    def _stats(docs, lens):
        words = " ".join(d["text"] for d in docs).split()
        return {
            "n_docs": len(docs),
            "n_tokens": int(sum(d["n_tokens"] for d in docs)),
            "mean_doc_tokens": float(np.mean([d["n_tokens"] for d in docs])) if docs else 0.0,
            "median_doc_tokens": float(np.median([d["n_tokens"] for d in docs])) if docs else 0.0,
            "ttr": ttr(words),
            "mtld": mtld(words),
        }

    ps, ns = _stats(pos_c, pos_lens), _stats(neg_c, neg_lens)
    try:
        from scipy import stats  # optional: numpy-only smoke environments
    except ImportError:
        stats = None
    if stats is not None:
        ks = stats.ks_2samp([d["n_tokens"] for d in pos_c], [d["n_tokens"] for d in neg_c])
        ks_stat, ks_pvalue = float(ks.statistic), float(ks.pvalue)
    else:
        ks_stat = ks_pvalue = None
    diagnostics = {
        "positive": ps,
        "contrast": ns,
        "token_ratio": ps["n_tokens"] / ns["n_tokens"] if ns["n_tokens"] else None,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "ttr_delta": abs(ps["ttr"] - ns["ttr"]),
        "cap_tokens": cap,
    }
    return pos_c, neg_c, diagnostics


# ---------------------------------------------------------------- attribute assembly

def attribute_corpora(cfg, attr_key: str, tokenizer, seed: int | None = None) -> dict:
    """Load + split + match all corpora needed for one attribute.

    Order: reserve disjoint classifier sets from the raw pools first, then match an
    oversized steering pool (matching drops short/long docs), then take stat/fit.
    """
    attr = getattr(cfg.attributes, attr_key)
    seed = int(get(cfg, "seed", 0)) if seed is None else seed
    max_length = int(get(cfg, "steering.max_length", 256))
    pos = attach_token_counts(load_pool(cfg, attr.positive), tokenizer, max_length)
    neg = attach_token_counts(load_pool(cfg, attr.contrast), tokenizer, max_length)

    n_stat = int(getattr(attr, "n_stat_docs", None) or get(cfg, "steering.n_stat_docs", 2000))
    n_fit = int(getattr(attr, "n_fit_docs", None) or get(cfg, "steering.n_fit_docs", 400))
    n_clf_train = int(getattr(attr, "n_clf_train", None) or get(cfg, "evaluation.classifier.n_clf_train", 1500))
    n_clf_eval = int(getattr(attr, "n_clf_eval", None) or get(cfg, "evaluation.classifier.n_clf_eval", 800))
    n_pool = max(3 * n_stat, n_stat + 600)

    splitter = split_pool_by_source if attr_key == "shakespeare" else split_pool
    splits = splitter(pos, neg, n_pool, n_clf_train, n_clf_eval, seed=seed)

    pos_m, neg_m, diag = match_corpora(
        splits["pool_pos"], splits["pool_neg"], seed=seed,
        max_len_percentile=float(get(cfg, "matching.max_len_percentile", 90.0)),
        min_tokens=int(get(cfg, "matching.min_tokens", 4)),
    )
    diag["pre_match_pool"] = {"pos": len(splits["pool_pos"]), "neg": len(splits["pool_neg"])}
    diag["requested_stat_docs"] = n_stat
    if len(pos_m) < n_stat:
        print(f"[corpora:{attr_key}] WARNING: only {len(pos_m)} matched positive docs (< {n_stat})")
    if len(neg_m) < n_stat:
        print(f"[corpora:{attr_key}] WARNING: only {len(neg_m)} matched contrast docs (< {n_stat})")
    pos_m, neg_m = _shuffle(pos_m, seed + 7), _shuffle(neg_m, seed + 8)

    splits["stat_pos"], splits["stat_neg"] = pos_m[:n_stat], neg_m[:n_stat]
    splits["fit_pos"] = splits["stat_pos"][:n_fit]
    splits["fit_neg"] = splits["stat_neg"][:n_fit]
    splits.pop("pool_pos", None)
    splits.pop("pool_neg", None)
    splits["diagnostics"] = diag
    splits["attr_key"] = attr_key
    splits["kind"] = getattr(attr, "kind", "style")
    splits["level"] = getattr(attr, "level", "L1")
    return splits
