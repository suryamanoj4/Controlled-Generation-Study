"""Tests for corpus splitting and the matching protocol."""
import pytest

from sas_e1.data.corpora import match_corpora, split_pool, split_pool_by_source


def _docs(n, tokens=50, n_sources=3):
    return [
        {"text": "word " * tokens, "source": f"src{i % n_sources}", "n_tokens": tokens}
        for i in range(n)
    ]


def test_split_pool_reserves_disjoint_classifier_sets():
    pos, neg = _docs(5000), _docs(5000)
    out = split_pool(pos, neg, n_pool=1800, n_clf_train=1500, n_clf_eval=800, seed=0)
    assert len(out["pool_pos"]) == 1800
    assert len(out["clf_pos_train"]) == 1500
    assert len(out["clf_pos_eval"]) == 800
    pool_ids = {id(d) for d in out["pool_pos"]}
    assert pool_ids.isdisjoint({id(d) for d in out["clf_pos_train"]})
    assert pool_ids.isdisjoint({id(d) for d in out["clf_pos_eval"]})


def test_split_pool_by_source_keeps_groups_together():
    pos, neg = _docs(300, n_sources=6), _docs(300, n_sources=6)
    out = split_pool_by_source(pos, neg, n_pool=200, n_clf_train=100, n_clf_eval=50, seed=0)
    eval_srcs = {d["source"] for d in out["clf_pos_eval"]}
    pool_srcs = {d["source"] for d in out["pool_pos"]}
    train_srcs = {d["source"] for d in out["clf_pos_train"]}
    assert eval_srcs.isdisjoint(pool_srcs)
    assert eval_srcs.isdisjoint(train_srcs)
    assert train_srcs.isdisjoint(pool_srcs)


def test_match_corpora_drops_short_docs_and_equalizes():
    pos = [{"text": "x", "n_tokens": t} for t in [2, 50, 60, 80, 90]]
    neg = [{"text": "y", "n_tokens": t} for t in [3, 55, 65, 70, 95]]
    p, n, diag = match_corpora(pos, neg, seed=0, min_tokens=4)
    assert all(d["n_tokens"] >= 4 for d in p + n)
    assert diag["positive"]["n_docs"] == len(p)
    assert {"ks_stat", "ttr_delta", "token_ratio"} <= set(diag)
