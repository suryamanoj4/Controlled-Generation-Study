"""Shared Qwen3 registry and path helpers for SAS-E2 scripts (stdlib only)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_VECTORS_DIR = REPO_ROOT / "artifacts" / "vectors"
DEFAULT_ACTIVATIONS_DIR = REPO_ROOT / "data" / "activations"
DATASET_ROOT = REPO_ROOT / "datasets" / "phase1" / "sentiment"

LANGS = ("en", "hi", "bn", "te", "ta")

# Frozen Phase 1 models. num_hidden_layers / hidden_size come from each
# model's config.json on the Hub.
MODELS = {
    "qwen3-1.7b": {
        "repo_id": "Qwen/Qwen3-1.7B",
        "revision": "main",
        "params": 2_031_739_904,
        "num_hidden_layers": 28,
        "hidden_size": 2048,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "tie_word_embeddings": True,
        "download_gb": 4.08,
        "vram_gib_after_load": 1.4,
        "role": "second complete run / fast smoke test",
    },
    "qwen3-4b": {
        "repo_id": "Qwen/Qwen3-4B",
        "revision": "main",
        "params": 4_022_468_096,
        "num_hidden_layers": 36,
        "hidden_size": 2560,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "tie_word_embeddings": True,
        "download_gb": 8.06,
        "vram_gib_after_load": 2.5,
        "role": "primary run 1",
    },
    "qwen3-8b": {
        "repo_id": "Qwen/Qwen3-8B",
        "revision": "main",
        "params": 8_190_735_360,
        "num_hidden_layers": 36,
        "hidden_size": 4096,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "tie_word_embeddings": False,
        "download_gb": 16.40,
        "vram_gib_after_load": 5.7,
        "role": "second run fallback (if 1.7B is dropped)",
    },
}

# Only what the tokenizer + fp16 checkpoint need. This skips .bin, .h5,
# ONNX, GGUF and other duplicate formats.
ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model.safetensors",
    "model-*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "LICENSE",
    "README.md",
]

REQUIRED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")

# Cache identity for Phase 1 vectors. Change CACHE_VERSION whenever the
# extraction or fitting semantics change.
CACHE_VERSION = 1
QUANT_TAG = "nf4-double-quant-fp16compute-uint8"
SITE = "decoder_block_output"  # residual stream after decoder block i, 0-based


def select_models(spec: str, models: dict | None = None) -> list[str]:
    registry = models or MODELS
    if spec is None or spec.strip().lower() in ("all", "*", ""):
        return list(registry)
    wanted = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in registry]
    if unknown:
        raise SystemExit(
            f"[models] unknown model(s): {', '.join(unknown)}\n"
            f"[models] available: {', '.join(registry)}"
        )
    wanted_set = set(wanted)
    return [name for name in registry if name in wanted_set]


def human_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{int(n)} B" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def checkpoint_state(path: Path) -> tuple[bool, str]:
    """Return (complete, reason) for one local model directory."""
    if not path.exists():
        return False, "not downloaded"

    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        return False, f"missing {', '.join(missing)}"

    index = path / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        except (json.JSONDecodeError, KeyError) as exc:
            return False, f"bad safetensors index ({exc})"
        shards = sorted(set(weight_map.values()))
        missing_shards = [shard for shard in shards if not (path / shard).is_file()]
        if missing_shards:
            return False, f"{len(missing_shards)}/{len(shards)} shards missing"
        return True, f"{len(shards)} shards"

    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        return False, "no *.safetensors"
    return True, f"{len(shards)} shard(s)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_commit(name: str, models_dir: str | Path = DEFAULT_MODELS_DIR) -> str:
    """Read the resolved commit recorded by download_models.py, if present."""
    manifest = Path(models_dir) / "download_manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            commit = data.get("models", {}).get(name, {}).get("resolved_commit")
            if commit:
                return str(commit)
        except (json.JSONDecodeError, OSError):
            pass
    return "unknown"
