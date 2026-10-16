"""Tests for target-driven download sizing (fetch only what the split needs)."""
from pathlib import Path

from sas_e1.config import load_config
from sas_e1.data.download import _corpus_target_docs

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "lean.yaml"


def test_targets_match_split_needs():
    cfg = load_config(CONFIG)
    # EN: pool 3000 + classifier 800/400 = 4200, +15% margin
    assert _corpus_target_docs(cfg, "sst2_positive") == int(4200 * 1.15)
    assert _corpus_target_docs(cfg, "sst2_negative") == int(4200 * 1.15)
    assert _corpus_target_docs(cfg, "shakespeare") == int(4200 * 1.15)
    assert _corpus_target_docs(cfg, "wikipedia") == int(4200 * 1.15)
    # Hindi story leg (per-attribute override): pool 1200 + 120/80 = 1400
    assert _corpus_target_docs(cfg, "wikisource_premchand") == int(1400 * 1.15)
    assert _corpus_target_docs(cfg, "wikisource_prasad") == int(1400 * 1.15)
    # Telugu story leg: pool 1200 + 80/50 = 1330
    assert _corpus_target_docs(cfg, "wikisource_gurajada") == int(1330 * 1.15)
    assert _corpus_target_docs(cfg, "wikisource_veeresalingam") == int(1330 * 1.15)
    # Indic sentiment: pool 1200 + 400/250 = 1850
    assert _corpus_target_docs(cfg, "hindi_sent_positive") == int(1850 * 1.15)


def test_target_none_for_unreferenced_corpus():
    cfg = load_config(CONFIG)
    assert _corpus_target_docs(cfg, "toxigen_toxic") is None
