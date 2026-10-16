"""Asset downloads: HF models + datasets, Project Gutenberg corpora.

All functions are idempotent: existing files are reused unless `force=True`.
Licenses verified 2026-10-13 (design doc v1.2 §4.1). Project Gutenberg text is stripped
of the PG header/footer before writing.
"""
from __future__ import annotations

import random
import re
import time
import zlib
from html.parser import HTMLParser
from pathlib import Path

from ..config import get
from ..utils import ensure_dir, read_jsonl, save_json, write_jsonl

# ---------------------------------------------------------------- Gutenberg

# Curated, era/genre-matched sources: Shakespeare plays vs other early-modern drama.
# IDs verified against Project Gutenberg on 2026-10-15 (direct pg{id}.txt URLs return 200
# with the expected titles). Gutendex search remains only as a fallback for works whose
# id is not pinned — the live API is rate-limited and frequently times out.
GUTENBERG_WORKS = [
    # --- positive: Shakespeare plays (label=1)
    {"key": "hamlet", "title": "Hamlet", "author": "shakespeare", "group": "pos", "id": 1524,
     "query": "hamlet william shakespeare"},
    {"key": "macbeth", "title": "Macbeth", "author": "shakespeare", "group": "pos", "id": 1533,
     "query": "macbeth william shakespeare"},
    {"key": "romeo_and_juliet", "title": "Romeo and Juliet", "author": "shakespeare", "group": "pos",
     "id": 1513, "query": "romeo and juliet william shakespeare"},
    {"key": "othello", "title": "Othello", "author": "shakespeare", "group": "pos", "id": 1531,
     "query": "othello william shakespeare"},
    {"key": "king_lear", "title": "King Lear", "author": "shakespeare", "group": "pos", "id": 1532,
     "query": "king lear william shakespeare"},
    {"key": "the_tempest", "title": "The Tempest", "author": "shakespeare", "group": "pos", "id": 1540,
     "query": "the tempest william shakespeare"},
    {"key": "midsummer", "title": "A Midsummer Night's Dream", "author": "shakespeare", "group": "pos",
     "id": 1514, "query": "a midsummer nights dream william shakespeare"},
    {"key": "julius_caesar", "title": "Julius Caesar", "author": "shakespeare", "group": "pos",
     "id": 1522, "query": "julius caesar william shakespeare"},
    {"key": "much_ado", "title": "Much Ado About Nothing", "author": "shakespeare", "group": "pos",
     "id": 1519, "query": "much ado about nothing william shakespeare"},
    {"key": "twelfth_night", "title": "Twelfth Night", "author": "shakespeare", "group": "pos",
     "id": 1526, "query": "twelfth night william shakespeare"},
    {"key": "merchant_of_venice", "title": "The Merchant of Venice", "author": "shakespeare",
     "group": "pos", "id": 1515, "query": "the merchant of venice william shakespeare"},
    {"key": "henry_v", "title": "King Henry V", "author": "shakespeare", "group": "pos", "id": 1521,
     "query": "henry v william shakespeare"},
    # --- contrast: other early-modern drama (label=0)
    {"key": "doctor_faustus", "title": "The Tragical History of Doctor Faustus", "author": "marlowe",
     "group": "neg", "id": 779, "query": "doctor faustus christopher marlowe"},
    {"key": "tamburlaine", "title": "Tamburlaine the Great — Part 1", "author": "marlowe",
     "group": "neg", "id": 1094, "query": "tamburlaine christopher marlowe"},
    {"key": "jew_of_malta", "title": "The Jew of Malta", "author": "marlowe", "group": "neg",
     "id": 901, "query": "the jew of malta christopher marlowe"},
    {"key": "the_alchemist", "title": "The Alchemist", "author": "jonson", "group": "neg", "id": 4081,
     "query": "the alchemist ben jonson"},
    {"key": "volpone", "title": "Volpone; Or, The Fox", "author": "jonson", "group": "neg", "id": 4039,
     "query": "volpone ben jonson"},
    {"key": "duchess_of_malfi", "title": "The Duchess of Malfi", "author": "webster", "group": "neg",
     "id": 2232, "query": "the duchess of malfi john webster"},
    {"key": "white_devil", "title": "The White Devil", "author": "webster", "group": "neg",
     "id": 12915, "query": "the white devil john webster"},
]
_AUTHOR_MATCH = {
    "shakespeare": "shakespeare",
    "marlowe": "marlowe",
    "jonson": "jonson",
    "webster": "webster",
}


def _pg_url(book_id: int) -> str:
    return f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"

_GUTENDEX = "https://gutendex.com/books"
_START_RE = re.compile(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
_END_RE = re.compile(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)


def resolve_gutenberg(query: str, author_key: str, sleep: float = 0.5,
                      attempts: int = 3) -> dict | None:
    """Resolve a title query via Gutendex (fallback path; retries transient failures)."""
    import requests

    results: list = []
    for attempt in range(attempts):
        try:
            resp = requests.get(_GUTENDEX, params={"search": query}, timeout=60)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            break
        except Exception as exc:  # network/API issues are non-fatal
            print(f"[gutenberg] lookup failed ({attempt + 1}/{attempts}) for '{query}': {exc}")
            if attempt < attempts - 1:
                time.sleep(2.0 * (2 ** attempt))
    time.sleep(sleep)

    want = _AUTHOR_MATCH[author_key]
    for book in results:
        authors = " ".join(a.get("name", "").lower() for a in book.get("authors", []))
        if want not in authors:
            continue
        formats = book.get("formats", {})
        url = None
        for key in ("text/plain; charset=utf-8", "text/plain; charset=us-ascii", "text/plain"):
            if key in formats:
                url = formats[key]
                break
        if url:
            return {"id": book["id"], "title": book.get("title", query), "url": url}
    return None


def _http_get_text(url: str, attempts: int = 3) -> str | None:
    import requests

    for attempt in range(attempts):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            print(f"[gutenberg] download failed ({attempt + 1}/{attempts}) {url}: {exc}")
            if attempt < attempts - 1:
                time.sleep(2.0 * (2 ** attempt))
    return None


def strip_gutenberg(text: str) -> str:
    start = _START_RE.search(text)
    if start:
        text = text[start.end():]
    end = _END_RE.search(text)
    if end:
        text = text[: end.start()]
    return text.strip()


def _chunk_by_paragraphs(text: str, target_words: int = 150, min_words: int = 20) -> list[str]:
    """Split text into stable ~target-word documents (paragraph/sentence aware)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        if len(para.split()) <= int(target_words * 1.6):
            if len(para.split()) >= min_words:
                units.append(para)
            continue
        # long paragraph: split on sentence boundaries (incl. Devanagari danda)
        sentences = re.split(r"(?<=[।॥.!?])\s+", para)
        buf: list[str] = []
        for sent in sentences:
            buf.append(sent)
            if len(" ".join(buf).split()) >= target_words:
                units.append(" ".join(buf))
                buf = []
        if buf and len(" ".join(buf).split()) >= min_words:
            units.append(" ".join(buf))

    docs: list[str] = []
    buf, count = [], 0
    for unit in units:
        words = len(unit.split())
        if count + words > target_words and buf:
            docs.append("\n\n".join(buf))
            buf, count = [], 0
        buf.append(unit)
        count += words
    if buf:
        docs.append("\n\n".join(buf))
    return docs


def _load_or_fetch_work(cfg, work: dict, sleep: float, force: bool) -> tuple[list[dict], dict]:
    """Fetch one work (per-work cache) and return (rows, resolution info)."""
    cache_dir = ensure_dir(Path(get(cfg, "paths.data_raw", "data/raw")) / "gutenberg")
    cache_path = cache_dir / f"{work['key']}.jsonl"
    info = {"key": work["key"], "title": work["title"], "group": work["group"], "id": work.get("id")}
    if cache_path.exists() and not force:
        info["status"] = "cached"
        return read_jsonl(cache_path), info

    book = {"id": work["id"], "title": work["title"]} if work.get("id") else resolve_gutenberg(
        work["query"], work["author"], sleep=sleep
    )
    if not book:
        print(f"[gutenberg] {work['key']}: FAILED to resolve — will retry next run")
        info["status"] = "failed_resolve"
        return [], info

    text = _http_get_text(_pg_url(book["id"]))
    if not text:
        print(f"[gutenberg] {work['key']}: FAILED to download — will retry next run")
        info["status"] = "failed_download"
        return [], info

    docs = _chunk_by_paragraphs(strip_gutenberg(text))
    rows = [
        {"text": doc, "source": book["title"], "source_id": book["id"],
         "label": 1 if work["group"] == "pos" else 0}
        for doc in docs
    ]
    write_jsonl(cache_path, rows)
    info.update({"status": "fetched", "id": book["id"], "title": book["title"], "n_docs": len(rows)})
    print(f"[gutenberg] {book['title']} -> {len(rows)} docs")
    return rows, info


def prepare_gutenberg(cfg, force: bool = False) -> dict:
    """Shakespeare plays vs other early-modern drama (Marlowe/Jonson/Webster).

    Works with pinned ids download directly from Project Gutenberg (no Gutendex search);
    successful works are cached per work, so re-runs only retry the ones that failed.
    """
    out_dir = ensure_dir(Path(get(cfg, "paths.data_processed", "data/processed")))
    manifest_path = out_dir / "gutenberg_manifest.json"
    pos_path, neg_path = out_dir / "shakespeare.jsonl", out_dir / "early_modern_drama.jsonl"
    sleep = float(get(cfg, "download.gutenberg_sleep", 0.5))
    raw_dir = Path(get(cfg, "paths.data_raw", "data/raw")) / "gutenberg"
    rng = random.Random(int(get(cfg, "seed", 0)) + 17)

    rows_by_group: dict[str, list[dict]] = {"pos": [], "neg": []}
    resolved: list[dict] = []
    skipped_works: list[str] = []
    for group, corpus_name in (("pos", "shakespeare"), ("neg", "early_modern_drama")):
        works = [w for w in GUTENBERG_WORKS if w["group"] == group]
        rng.shuffle(works)  # seeded random order: title coverage varies with seed
        target = _corpus_target_docs(cfg, corpus_name)
        processed = 0
        for work in works:
            cached = (raw_dir / f"{work['key']}.jsonl").exists()
            if (not cached and target is not None
                    and len(rows_by_group[group]) >= target and processed >= 5):
                skipped_works.append(work["key"])
                continue
            rows, info = _load_or_fetch_work(cfg, work, sleep, force)
            if not str(info.get("status", "")).startswith("failed"):
                processed += 1
            rows_by_group[group].extend(rows)
            resolved.append(info)
        if skipped_works:
            print(f"[gutenberg] {group}: target reached — skipped work(s): {', '.join(skipped_works)}")
    rows_pos, rows_neg = rows_by_group["pos"], rows_by_group["neg"]

    write_jsonl(pos_path, rows_pos)
    write_jsonl(neg_path, rows_neg)
    save_json(manifest_path, {"resolved": resolved, "skipped": skipped_works,
                              "n_pos": len(rows_pos), "n_neg": len(rows_neg)})
    failed = [r["key"] for r in resolved if str(r.get("status", "")).startswith("failed")]
    if failed:
        print(f"[gutenberg] WARNING: {len(failed)} work(s) failed ({', '.join(failed)}) — re-run to retry")
    if len(rows_pos) < 300 or len(rows_neg) < 150:
        print("[gutenberg] WARNING: fewer docs than expected — check gutenberg_manifest.json")
    return {"n_pos": len(rows_pos), "n_neg": len(rows_neg), "failed": failed}


# ---------------------------------------------------------------- HF datasets

def prepare_sst2(cfg, force: bool = False) -> dict:
    from datasets import load_dataset

    out_dir = ensure_dir(Path(get(cfg, "paths.data_processed", "data/processed")))
    pos_path, neg_path = out_dir / "sst2_positive.jsonl", out_dir / "sst2_negative.jsonl"
    if pos_path.exists() and neg_path.exists() and not force:
        print("[sst2] cached")
        return {"skipped": True}

    n_fit = int(get(cfg, "download.n_sst_fit", 8000))
    n_clf = int(get(cfg, "download.n_sst_clf", 4000))
    n_eval = int(get(cfg, "download.n_sst_eval", 2000))
    per_class = n_fit + n_clf + n_eval

    ds = load_dataset("nyu-mll/glue", "sst2", split="train")
    ds = ds.shuffle(seed=0)
    pos = [r["sentence"] for r in ds if r["label"] == 1][:per_class]
    neg = [r["sentence"] for r in ds if r["label"] == 0][:per_class]
    if len(pos) < per_class or len(neg) < per_class:
        print(f"[sst2] WARNING: only {len(pos)}/{len(neg)} per class available (wanted {per_class})")
    write_jsonl(pos_path, [{"text": t, "source": "sst2", "label": 1} for t in pos])
    write_jsonl(neg_path, [{"text": t, "source": "sst2", "label": 0} for t in neg])
    print(f"[sst2] wrote {len(pos)} pos, {len(neg)} neg")
    return {"n_pos": len(pos), "n_neg": len(neg)}


def prepare_civil_comments(cfg, force: bool = False) -> dict:
    from datasets import load_dataset

    out_dir = ensure_dir(Path(get(cfg, "paths.data_processed", "data/processed")))
    pos_path, neg_path = out_dir / "civil_toxic.jsonl", out_dir / "civil_benign.jsonl"
    if pos_path.exists() and neg_path.exists() and not force:
        print("[civil_comments] cached")
        return {"skipped": True}

    n_pool = int(get(cfg, "download.n_civil", 30000))
    ds = load_dataset("google/civil_comments", split="train", streaming=True)
    pos, neg = [], []
    for row in ds:
        text = (row.get("text") or "").strip()
        if len(text.split()) < 10:
            continue
        tox = float(row.get("toxicity", 0.0))
        if tox >= 0.8 and len(pos) < n_pool:
            pos.append(text)
        elif tox <= 0.1 and len(neg) < n_pool:
            neg.append(text)
        if len(pos) >= n_pool and len(neg) >= n_pool:
            break
    write_jsonl(pos_path, [{"text": t, "source": "civil_comments", "label": 1} for t in pos])
    write_jsonl(neg_path, [{"text": t, "source": "civil_comments", "label": 0} for t in neg])
    print(f"[civil_comments] wrote {len(pos)} toxic, {len(neg)} benign")
    return {"n_pos": len(pos), "n_neg": len(neg)}


def prepare_wikipedia(cfg, force: bool = False) -> dict:
    from datasets import load_dataset

    out_dir = ensure_dir(Path(get(cfg, "paths.data_processed", "data/processed")))
    contrast_path = out_dir / "wikipedia.jsonl"
    neutral_path = out_dir / "neutral_wikipedia.jsonl"
    if contrast_path.exists() and neutral_path.exists() and not force:
        print("[wikipedia] cached")
        return {"skipped": True}

    n_contrast = int(get(cfg, "download.n_wikipedia", 5000))
    n_neutral = int(get(cfg, "download.n_wikipedia_neutral", 1000))
    need = n_contrast + n_neutral

    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    docs = []
    for row in ds:
        text = (row.get("text") or "").strip()
        words = len(text.split())
        if 40 <= words <= 500:
            docs.append({"text": text, "source": f"wikipedia:{row.get('title', '')}", "label": 0})
        if len(docs) >= need:
            break
    if len(docs) < need:
        print(f"[wikipedia] WARNING: {len(docs)} docs < {need} requested")
    del ds  # best-effort: release the streaming iterator at end of use
    import gc

    gc.collect()
    neutral = docs[:n_neutral]
    contrast = docs[n_neutral:]
    write_jsonl(neutral_path, neutral)
    write_jsonl(contrast_path, contrast)
    print(f"[wikipedia] wrote {len(contrast)} contrast, {len(neutral)} neutral")
    return {"n_contrast": len(contrast), "n_neutral": len(neutral)}


_MORAL_STORY_FIELDS = ("situation", "intention", "moral_action", "moral_consequence")


def _load_moral_stories_rows(seed: int = 0) -> list[dict]:
    """Load Moral Stories from the auto-converted Parquet branch.

    `datasets>=5` no longer supports dataset loading scripts, so
    `demelin/moral_stories` (script-only) cannot be loaded directly. The HF
    auto-convert branch `refs/convert/parquet` hosts the same data as Parquet.
    """
    import random

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    repo, revision = "demelin/moral_stories", "refs/convert/parquet"
    files = list_repo_files(repo, repo_type="dataset", revision=revision)
    candidates = sorted(f for f in files if f.startswith("full/") and f.endswith(".parquet"))
    if not candidates:
        raise RuntimeError(f"no Parquet files found for {repo}@{revision}")
    train = [f for f in candidates if "/train/" in f] or candidates
    rows: list[dict] = []
    for fname in train:
        local = hf_hub_download(repo, fname, repo_type="dataset", revision=revision)
        rows.extend(pq.read_table(local).to_pylist())
    random.Random(seed).shuffle(rows)
    return rows


def prepare_moral_stories(cfg, force: bool = False) -> dict:
    out_dir = ensure_dir(Path(get(cfg, "paths.data_processed", "data/processed")))
    path = out_dir / "moral_stories.jsonl"
    if path.exists() and not force:
        print("[moral_stories] cached")
        return {"skipped": True}

    n_docs = int(get(cfg, "download.n_moral", 4000))
    raw = _load_moral_stories_rows(seed=int(get(cfg, "seed", 0)))
    rows: list[dict] = []
    for r in raw:
        text = " ".join(
            str(r.get(k) or "").strip() for k in _MORAL_STORY_FIELDS
        ).strip()
        if len(text.split()) >= 20:
            rows.append({"text": text, "source": "moral_stories", "label": 1})
        if len(rows) >= n_docs:
            break
    write_jsonl(path, rows)
    print(f"[moral_stories] wrote {len(rows)} docs (from {len(raw)} parquet rows)")
    return {"n": len(rows)}


def prepare_toxigen(cfg, force: bool = False) -> dict:
    """Optional implicit-toxicity arm (not part of the default Lean grid)."""
    from datasets import load_dataset

    out_dir = ensure_dir(Path(get(cfg, "paths.data_processed", "data/processed")))
    path = out_dir / "toxigen_toxic.jsonl"
    neg_path = out_dir / "toxigen_benign.jsonl"
    if path.exists() and neg_path.exists() and not force:
        print("[toxigen] cached")
        return {"skipped": True}
    n = 5000
    ds = load_dataset("toxigen/toxigen-data", "train", split="train")
    ds = ds.shuffle(seed=0)
    pos = [r["generation"] for r in ds if int(r.get("prompt_label", 0)) == 1][:n]
    neg = [r["generation"] for r in ds if int(r.get("prompt_label", 0)) == 0][:n]
    write_jsonl(path, [{"text": t, "source": "toxigen", "label": 1} for t in pos])
    write_jsonl(neg_path, [{"text": t, "source": "toxigen", "label": 0} for t in neg])
    return {"n_pos": len(pos), "n_neg": len(neg)}


# ---------------------------------------------------------------- Indic corpora

# Authentic, public-domain author corpora fetched from Wikisource via the MediaWiki API
# (E1b replication: same author-style design as the English Shakespeare leg).
# Authors died 1915-1937, so texts are public domain; Wikisource API use requires a UA.
WIKISOURCE_UA = {"User-Agent": "sas-e1-corpus-fetcher/0.1 (research; contact: surya)"}
WIKISOURCE_AUTHORS = {
    "wikisource_premchand": {
        "site": "https://hi.wikisource.org",
        "author": "Premchand",
        "prefixes": ["प्रेमचंद की सर्वश्रेष्ठ कहानियां/", "प्रेमचंद रचनावली (खण्ड ५)/"],
    },
    "wikisource_prasad": {
        "site": "https://hi.wikisource.org",
        "author": "Jaishankar Prasad",
        "prefixes": ["जयशंकर प्रसाद की श्रेष्ठ कहानियां/", "तितली/"],
    },
    "wikisource_gurajada": {
        "site": "https://te.wikisource.org",
        "author": "Gurajada Apparao",
        "prefixes": ["కన్యాశుల్కము/", "గురజాడలు/కథానికలు/", "గురుజాడలు/కవితలు/", "బిల్హణీయము"],
    },
    "wikisource_veeresalingam": {
        "site": "https://te.wikisource.org",
        "author": "Kandukuri Veeresalingam",
        "prefixes": ["రాజశేఖర చరిత్రము (ఎమెస్కో)/", "కందుకూరి వీరేశలింగకృత గ్రంథములు/"],
    },
}


class _HtmlTextExtractor(HTMLParser):
    """Visible-text extraction (stdlib HTMLParser; ignores script/style/template)."""

    SKIP = {"script", "style", "template", "noscript"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        import html as _html

        out = _html.unescape("".join(self.parts))
        out = re.sub(r"[ \t]+", " ", out)
        out = re.sub(r"\n\s*\n+", "\n\n", out)
        return out.strip()


def _ws_get_json(site: str, params: dict, attempts: int = 3) -> dict:
    """Polite MediaWiki API call: maxlag, Retry-After aware, exponential backoff."""
    import requests

    params = {**params, "format": "json", "maxlag": 5}
    for attempt in range(attempts):
        try:
            resp = requests.get(f"{site}/w/api.php", params=params, headers=WIKISOURCE_UA, timeout=60)
            if resp.status_code in (429, 503):
                wait = float(resp.headers.get("Retry-After") or 0) or min(5.0 * (2 ** attempt), 30.0)
                print(f"[wikisource] throttled ({resp.status_code}); sleeping {wait:.0f}s")
                time.sleep(wait)
                continue
            if resp.status_code == 403:
                wait = float(resp.headers.get("Retry-After") or 0) or min(15.0 * (attempt + 1), 45.0)
                print(f"[wikisource] 403 (site throttling); sleeping {wait:.0f}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            print(f"[wikisource] API failed ({attempt + 1}/{attempts}): {exc}")
            if attempt < attempts - 1:
                time.sleep(min(5.0 * (2 ** attempt), 30.0))
    return {}


def _ws_allpages(site: str, prefix: str, limit: int = 500) -> list[str]:
    data = _ws_get_json(site, {"action": "query", "list": "allpages", "apprefix": prefix,
                               "aplimit": limit})
    return [p["title"] for p in data.get("query", {}).get("allpages", [])]


def _ws_parse_page(site: str, title: str) -> str:
    data = _ws_get_json(site, {"action": "parse", "page": title, "prop": "text",
                               "formatversion": "2"})
    html_text = data.get("parse", {}).get("text", "")
    if not html_text:
        return ""
    extractor = _HtmlTextExtractor()
    extractor.feed(html_text)
    return extractor.text()


def _clean_wikisource(text: str) -> str:
    """Drop navigation/page-number noise while preserving paragraph structure."""
    out: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            if out and out[-1] != "":
                out.append("")
            continue
        if re.fullmatch(r"[←→\s\-–—·•|<>{}\[\]()]+", s):
            continue
        if re.fullmatch(r"\d{2,}", s):
            continue
        s = re.sub(r"^\d{3,}(?=\S)", "", s)  # page id glued to a title
        if len(s.split()) < 4 and any(k in s for k in ("రచించినవారు", "द्वारा", "(19", "(20")):
            continue
        out.append(s)
    return "\n".join(out)


def _load_or_fetch_wikisource_author(cfg, corpus_name: str, force: bool = False) -> tuple[list[dict], dict]:
    spec = WIKISOURCE_AUTHORS[corpus_name]
    site = spec["site"]
    raw_dir = ensure_dir(Path(get(cfg, "paths.data_raw", "data/raw")) / "wikisource" / corpus_name)
    out_path = Path(get(cfg, "paths.data_processed", "data/processed")) / f"{corpus_name}.jsonl"

    # always enumerate: per-page cache means re-runs only fetch pages still missing
    titles: list[str] = []
    sleep = float(get(cfg, "download.wikisource_sleep", 1.0))
    for prefix in spec["prefixes"]:
        found = _ws_allpages(site, prefix)
        titles.extend(t for t in found if t not in titles)
        time.sleep(sleep)
    if not titles:
        print(f"[wikisource] {corpus_name}: no pages found — check prefixes")
        return [], {"author": spec["author"], "status": "no_pages"}

    rows: list[dict] = []
    fetched = 0
    skipped = 0
    consecutive_failures = 0
    # seeded random page order so title coverage varies with the run seed; stop fetching
    # (uncached pages) once the split's target is met — cached pages are free and kept
    rng = random.Random(int(get(cfg, "seed", 0)) + zlib.crc32(corpus_name.encode()) % 9973)
    rng.shuffle(titles)
    target = _corpus_target_docs(cfg, corpus_name)
    for idx, title in enumerate(titles):
        safe = re.sub(r"[^\w\-.]+", "_", title)[:120].strip("_") or "page"
        cache_path = raw_dir / f"{safe}.txt"
        if cache_path.exists() and not force:
            text = cache_path.read_text(encoding="utf-8")
        else:
            if target is not None and len(rows) >= target and fetched >= 3:
                skipped += 1
                continue
            text = _clean_wikisource(_ws_parse_page(site, title))
            if text:
                cache_path.write_text(text, encoding="utf-8")
            time.sleep(sleep)
        if not text or len(text.split()) < 40:
            consecutive_failures += 1
            if consecutive_failures >= 8:
                print(f"[wikisource] {corpus_name}: too many consecutive failures — "
                      "site likely throttling; will resume next run")
                break
            continue
        consecutive_failures = 0
        fetched += 1
        docs = _chunk_by_paragraphs(text, min_words=12)
        rows.extend({"text": d, "source": f"{spec['author']} — {title}", "label": None} for d in docs)
        if (idx + 1) % 10 == 0:
            print(f"[wikisource] {corpus_name}: {idx + 1}/{len(titles)} pages, {len(rows)} docs so far")
    if rows:
        write_jsonl(out_path, rows)
    else:
        print(f"[wikisource] {corpus_name}: no usable text — will retry next run")
    if skipped:
        print(f"[wikisource] {corpus_name}: target ({target}) reached — skipped {skipped} uncached page(s)")
    print(f"[wikisource] {corpus_name}: {fetched}/{len(titles)} pages -> {len(rows)} docs")
    return rows, {"author": spec["author"], "pages": len(titles), "fetched": fetched,
                  "skipped": skipped, "target": target, "n_docs": len(rows),
                  "status": "fetched" if rows else "empty"}


def prepare_wikisource(cfg, force: bool = False) -> dict:
    referenced = _referenced_corpora(cfg)
    results = {}
    for name in WIKISOURCE_AUTHORS:
        if name in referenced:
            _, info = _load_or_fetch_wikisource_author(cfg, name, force)
            results[name] = info
    return results


def prepare_indic_sentiment(cfg, lang: str, force: bool = False) -> dict:
    """Hindi / Telugu sentiment pools (permissively licensed; ungated)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    out_dir = ensure_dir(Path(get(cfg, "paths.data_processed", "data/processed")))
    pos_path = out_dir / f"{'hindi' if lang == 'hi' else 'telugu'}_sent_positive.jsonl"
    neg_path = out_dir / f"{'hindi' if lang == 'hi' else 'telugu'}_sent_negative.jsonl"
    if (pos_path.exists() and neg_path.exists() and not force
            and pos_path.stat().st_size > 1000 and neg_path.stat().st_size > 1000):
        print(f"[{lang}_sentiment] cached")
        return {"skipped": True}

    cap = int(get(cfg, "download.n_indic_sent", 8000))
    if lang == "hi":
        path = hf_hub_download("iam-tsr/hindi-sentiments", "indic_sentiment_data.csv", repo_type="dataset")
        df = pd.read_csv(path)
        text_col, label_col = "hindi_text", "label"
        labels = df[label_col]
        pos = df.loc[labels == 1, text_col].astype(str).tolist()
        neg = df.loc[labels == 2, text_col].astype(str).tolist()
    else:
        frames = []
        for fname in ("Sentiment_train.csv", "Sentiment_valid .csv"):
            try:
                p = hf_hub_download("mounikaiiith/Telugu_Sentiment", fname, repo_type="dataset")
                frames.append(pd.read_csv(p))
            except Exception as exc:
                print(f"[te_sentiment] skipped {fname}: {exc}")
        df = pd.concat(frames, ignore_index=True)
        sent = df["Sentiment"].astype(str).str.strip().str.lower()
        pos = df.loc[sent.isin({"pos", "positive"}), "Sentence"].astype(str).tolist()
        neg = df.loc[sent.isin({"neg", "negative"}), "Sentence"].astype(str).tolist()

    import random

    for rows, path, label in ((pos, pos_path, 1), (neg, neg_path, 0)):
        random.Random(0).shuffle(rows)
        write_jsonl(path, [{"text": t, "source": f"{lang}_sentiment", "label": label}
                           for t in rows[:cap]])
    print(f"[{lang}_sentiment] wrote {min(len(pos), cap)} pos, {min(len(neg), cap)} neg")
    return {"n_pos": min(len(pos), cap), "n_neg": min(len(neg), cap)}


def _referenced_corpora(cfg) -> set[str]:
    referenced: set[str] = set()
    if not hasattr(cfg, "attributes") or not cfg.attributes:
        return referenced
    for name in vars(cfg.attributes):
        attr = getattr(cfg.attributes, name)
        for field in ("positive", "contrast"):
            value = getattr(attr, field, None)
            if value:
                referenced.add(str(value))
    return referenced


def _corpus_target_docs(cfg, corpus_name: str, margin: float = 1.15) -> int | None:
    """Docs this corpus needs for splitting (pool + classifier needs, plus margin).

    Works/pages are fetched in a seeded random order and fetching stops once the
    target is reached (cached items are always included for free). Returns None
    when no attribute references the corpus.
    """
    needs: list[int] = []
    if not hasattr(cfg, "attributes") or not cfg.attributes:
        return None
    for name in vars(cfg.attributes):
        attr = getattr(cfg.attributes, name)
        if corpus_name not in (getattr(attr, "positive", None), getattr(attr, "contrast", None)):
            continue
        n_stat = int(getattr(attr, "n_stat_docs", None) or get(cfg, "steering.n_stat_docs", 2000))
        n_clf_train = int(getattr(attr, "n_clf_train", None) or get(cfg, "evaluation.classifier.n_clf_train", 1500))
        n_clf_eval = int(getattr(attr, "n_clf_eval", None) or get(cfg, "evaluation.classifier.n_clf_eval", 800))
        n_pool = int(getattr(attr, "n_pool_docs", None) or max(3 * n_stat, n_stat + 600))
        needs.append(n_pool + n_clf_train + n_clf_eval)
    if not needs:
        return None
    return int(max(needs) * margin)


# ---------------------------------------------------------------- models + all

def download_models(cfg) -> None:
    from huggingface_hub import snapshot_download

    cache_dir = ensure_dir(Path(get(cfg, "paths.models", "models")))
    name = cfg.model.name
    print(f"[models] snapshot_download({name})")
    snapshot_download(repo_id=name, cache_dir=str(cache_dir))
    ppl = get(cfg, "evaluation.ppl_model", None)
    if ppl and ppl != name:
        print(f"[models] snapshot_download({ppl})")
        snapshot_download(repo_id=ppl, cache_dir=str(cache_dir))


def prepare_all(cfg, force: bool = False, include_toxigen: bool = False) -> dict:
    manifest = {
        "sources": {
            "sst2": {"hf": "nyu-mll/glue (sst2)", "license": "GLUE 'other' (per-task)"},
            "civil_comments": {"hf": "google/civil_comments", "license": "CC0-1.0"},
            "wikipedia": {"hf": "wikimedia/wikipedia", "license": "CC-BY-SA-3.0 + GFDL"},
            "moral_stories": {"hf": "demelin/moral_stories", "license": "MIT"},
            "gutenberg": {"web": "Project Gutenberg (pinned ids)", "license": "US public domain"},
            "wikisource": {"web": "hi/te Wikisource (MediaWiki API)", "license": "public-domain texts"},
            "hindi_sent": {"hf": "iam-tsr/hindi-sentiments", "license": "MIT"},
            "telugu_sent": {"hf": "mounikaiiith/Telugu_Sentiment", "license": "CC-BY-4.0"},
            "toxigen": {"hf": "toxigen/toxigen-data", "license": "MIT + CDLA-Permissive (research note)"},
        },
        "results": {},
    }
    ref = _referenced_corpora(cfg)
    if ref & {"sst2_positive", "sst2_negative"}:
        manifest["results"]["sst2"] = prepare_sst2(cfg, force)
    if ref & {"civil_toxic", "civil_benign"}:
        manifest["results"]["civil_comments"] = prepare_civil_comments(cfg, force)
    # Wikipedia provides the neutral probe corpus for every attribute plus the fable contrast
    manifest["results"]["wikipedia"] = prepare_wikipedia(cfg, force)
    if "moral_stories" in ref:
        manifest["results"]["moral_stories"] = prepare_moral_stories(cfg, force)
    if ref & {"shakespeare", "early_modern_drama"}:
        manifest["results"]["gutenberg"] = prepare_gutenberg(cfg, force)
    if ref & set(WIKISOURCE_AUTHORS):
        manifest["results"]["wikisource"] = prepare_wikisource(cfg, force)
    if ref & {"hindi_sent_positive", "hindi_sent_negative"}:
        manifest["results"]["hindi_sent"] = prepare_indic_sentiment(cfg, "hi", force)
    if ref & {"telugu_sent_positive", "telugu_sent_negative"}:
        manifest["results"]["telugu_sent"] = prepare_indic_sentiment(cfg, "te", force)
    if include_toxigen:
        manifest["results"]["toxigen"] = prepare_toxigen(cfg, force)

    save_json(Path(get(cfg, "paths.data_processed", "data/processed")) / "manifest.json", manifest)
    return manifest
