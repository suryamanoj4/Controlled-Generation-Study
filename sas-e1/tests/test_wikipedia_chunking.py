"""Tests for the Wikipedia contrast chunker (fable length-matching fix)."""
from sas_e1.data.download import _chunk_wikipedia_lead


def _sentence(i: int) -> str:
    return f"Sentence number {i} contains exactly eight words in total."


def test_chunks_cover_text_within_bounds():
    text = " ".join(_sentence(i) for i in range(20))  # 20 sentences, ~9 words each
    total = len(text.split())
    chunks = _chunk_wikipedia_lead(text, min_words=20, max_words=55)
    assert len(chunks) >= 3
    sizes = [len(c.split()) for c in chunks]
    assert all(20 <= s <= 55 for s in sizes)
    assert sum(sizes) >= 0.9 * total  # nothing meaningful dropped


def test_short_text_dropped():
    assert _chunk_wikipedia_lead("Only nine words are in this text here.", min_words=20) == []


def test_giant_sentence_hard_wrapped():
    text = "word " * 120  # one 120-word "sentence", no punctuation
    chunks = _chunk_wikipedia_lead(text, min_words=20, max_words=55)
    assert chunks
    assert all(len(c.split()) <= 55 for c in chunks)


def test_short_sentences_are_packed():
    # fable-length target: several short sentences should merge into one doc
    text = " ".join(f"w{i}x" for i in range(20)) + ". Tiny tail words here now ok."
    chunks = _chunk_wikipedia_lead(text, min_words=20, max_words=55)
    assert len(chunks) >= 1
    assert all(20 <= len(c.split()) <= 55 for c in chunks)
