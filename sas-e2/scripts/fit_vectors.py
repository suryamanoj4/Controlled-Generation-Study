#!/usr/bin/env python3
"""Fit Phase 1 sentiment steering vectors (DiffMean/CAA) and cache them.

For each (model, language), this script:

  1. loads the Qwen3 checkpoint in 4-bit NF4 with fp16 compute,
  2. forward-passes the 500 stat reviews (250 positive / 250 negative),
  3. pools the residual stream after every selected decoder block,
  4. stores  diff[layer] = mean(positive) - mean(negative),
     together with mean_pos, mean_neg and the layer's mean residual norm.

The cache is keyed by model commit, NF4 config, stat.jsonl sha256,
tokenizer.json sha256, pooling, template, max_length and layer set. If a
valid cache exists, the model is never loaded for that language. This makes
later runs (prompting, alpha sweeps, seeds) reuse the same vectors.

Examples:
  # default: every downloaded model, all five languages, one at a time
  python3 scripts/fit_vectors.py

  # one model at a time
  python3 scripts/fit_vectors.py --models qwen3-4b
  python3 scripts/fit_vectors.py --models qwen3-1.7b

  # a subset of languages, keep per-doc activations for refits
  python3 scripts/fit_vectors.py --models qwen3-1.7b --langs hi,te --save-activations

  # inspect the cache without a GPU
  python3 scripts/fit_vectors.py --check
  python3 scripts/fit_vectors.py --dry-run

Output layout:
  artifacts/vectors/<model>/<commit8>/<lang>/
    vectors.safetensors   diff [L,H] float32, mean_pos, mean_neg,
                          resid_norm_mean [L], layer_ids [L]
    meta.json             provenance + cache key
    report.json           per-layer norms, effect size, bootstrap stability

  data/activations/<model>/<commit8>/<lang>/   (only with --save-activations)
    stat_<pooling>.safetensors   acts [N,L,H] float16, labels [N]
    docs.json

The default pooling is the last non-pad token (CAA convention). Use
--pooling mean for mean-over-content-tokens. Layer indices are 0-based
outputs of decoder blocks; the same site is used when steering.

Tokenization keeps the document end (left truncation) when pooling=last.
The default max_length=512 covers 97.8% of Telugu stat reviews and >99%
of the other four languages; use 768 for zero truncation at higher cost.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _models import (
    CACHE_VERSION,
    DATASET_ROOT,
    DEFAULT_ACTIVATIONS_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_VECTORS_DIR,
    LANGS,
    MODELS,
    QUANT_TAG,
    REPO_ROOT,
    SITE,
    checkpoint_state,
    human_bytes,
    resolve_commit,
    select_models,
    sha256_file,
    utc_now,
)

POOLINGS = ("last", "mean")
DEFAULT_TEMPLATE = "{text}"


def truncation_side(pooling: str) -> str:
    """Keep the document end for last-token pooling."""
    return "left" if pooling == "last" else "right"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Phase 1 DiffMean steering vectors for Qwen3 (4-bit NF4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        "--only",
        dest="models",
        default="all",
        metavar="LIST",
        help="comma-separated keys (qwen3-1.7b, qwen3-4b, qwen3-8b) or 'all' (default: all)",
    )
    parser.add_argument(
        "--langs",
        "--lang",
        dest="langs",
        default="all",
        metavar="LIST",
        help="comma-separated languages (en, hi, bn, te, ta) or 'all' (default: all)",
    )
    parser.add_argument(
        "--layers",
        default="all",
        metavar="LIST",
        help="0-based decoder block indices (e.g. 10,12,14) or 'all' (default: all)",
    )
    parser.add_argument(
        "--pooling",
        choices=POOLINGS,
        default="last",
        help="token pooling for extraction (default: last)",
    )
    parser.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE,
        help="text wrapper containing {text} (default: {text})",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="token truncation (default: 512; 768 = no truncation on Phase 1 stat)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="forward batch size; 0 = auto (8 for <=4B, 4 for 8B)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=20,
        help="bootstrap resamples for vector stability (default: 20; 0 disables)",
    )
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed (default: 0)")
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help=f"downloaded checkpoints (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--vectors-dir",
        default=str(DEFAULT_VECTORS_DIR),
        help=f"vector cache (default: {DEFAULT_VECTORS_DIR})",
    )
    parser.add_argument(
        "--activations-dir",
        default=str(DEFAULT_ACTIVATIONS_DIR),
        help=f"activation cache (default: {DEFAULT_ACTIVATIONS_DIR})",
    )
    parser.add_argument(
        "--save-activations",
        action="store_true",
        help="also cache per-doc stat activations for refits",
    )
    parser.add_argument("--force", action="store_true", help="refit even if a valid cache exists")
    parser.add_argument("--check", action="store_true", help="report cache status only; no GPU")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; load nothing")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    args = parser.parse_args()

    if "{text}" not in args.template:
        parser.error("--template must contain the {text} placeholder")
    if args.batch_size < 0:
        parser.error("--batch-size must be >= 0")
    if args.max_length <= 0:
        parser.error("--max-length must be > 0")
    if args.bootstrap < 0:
        parser.error("--bootstrap must be >= 0")
    return args


def parse_langs(spec: str) -> list[str]:
    if spec is None or spec.strip().lower() in ("all", "*", ""):
        return list(LANGS)
    wanted = [lang.strip() for lang in spec.split(",") if lang.strip()]
    unknown = [lang for lang in wanted if lang not in LANGS]
    if unknown:
        raise SystemExit(
            f"[fit] unknown language(s): {', '.join(unknown)}\n"
            f"[fit] available: {', '.join(LANGS)}"
        )
    wanted_set = set(wanted)
    return [lang for lang in LANGS if lang in wanted_set]


def parse_layers(spec: str, num_layers: int) -> list[int]:
    if spec is None or spec.strip().lower() in ("all", "*", ""):
        return list(range(num_layers))
    try:
        layer_ids = sorted({int(part) for part in spec.split(",") if part.strip()})
    except ValueError as exc:
        raise ValueError(f"invalid --layers value: {spec}") from exc
    bad = [layer for layer in layer_ids if layer < 0 or layer >= num_layers]
    if bad:
        raise ValueError(
            f"layer(s) {bad} out of range for a {num_layers}-layer model"
        )
    if not layer_ids:
        raise ValueError("--layers selected no layers")
    return layer_ids


# ---------------------------------------------------------------------------
# Paths, data, cache keys
# ---------------------------------------------------------------------------


def short_commit(commit: str) -> str:
    return commit[:8] if commit and commit != "unknown" else "unknown"


def vector_paths(vectors_dir: str | Path, model: str, commit: str, lang: str) -> dict[str, Path]:
    base = Path(vectors_dir) / model / short_commit(commit) / lang
    return {
        "base": base,
        "vectors": base / "vectors.safetensors",
        "meta": base / "meta.json",
        "report": base / "report.json",
    }


def activation_paths(
    activations_dir: str | Path, model: str, commit: str, lang: str, pooling: str
) -> dict[str, Path]:
    base = Path(activations_dir) / model / short_commit(commit) / lang
    return {
        "base": base,
        "acts": base / f"stat_{pooling}.safetensors",
        "docs": base / "docs.json",
    }


def load_stat(lang: str) -> tuple[Path, list[dict]]:
    path = DATASET_ROOT / lang / "stat.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing stat file: {path}")

    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        raise RuntimeError(f"empty stat file: {path}")
    for row in rows:
        if not row.get("text", "").strip():
            raise RuntimeError(f"{path}: record {row.get('id')} has empty text")
        if row.get("label") not in (0, 1):
            raise RuntimeError(f"{path}: record {row.get('id')} has invalid label")
    return path, rows


def build_cache_key(
    spec: dict,
    commit: str,
    tokenizer_sha: str,
    stat_sha: str,
    layer_ids: list[int],
    args: argparse.Namespace,
) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "model": spec["repo_id"],
        "commit": commit,
        "quant": QUANT_TAG,
        "stat_sha256": stat_sha,
        "tokenizer_sha256": tokenizer_sha,
        "pooling": args.pooling,
        "truncation_side": truncation_side(args.pooling),
        "template": args.template,
        "max_length": args.max_length,
        "site": SITE,
        "attn_implementation": "sdpa",
        "layer_ids": layer_ids,
        "add_special_tokens": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def cache_status(paths: dict[str, Path], expected_key: str) -> tuple[bool, str, dict | None]:
    if not paths["vectors"].is_file() or not paths["meta"].is_file():
        return False, "missing", None
    if not paths["report"].is_file():
        return False, "missing report", None
    meta = read_json(paths["meta"])
    if meta is None:
        return False, "bad meta", None
    if meta.get("cache_key") != expected_key:
        return False, "stale key", meta
    if not meta.get("layers"):
        return False, "bad meta (no layers)", meta
    return True, "ok", meta


# ---------------------------------------------------------------------------
# Extraction and fitting
# ---------------------------------------------------------------------------


def get_decoder_layers(model):
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("cannot find decoder layers on the model")


def pool_hidden(hidden, mask, pooling: str):
    import torch

    hidden = hidden.float()
    if pooling == "mean":
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom

    # Right padding: the last real token is mask.sum() - 1.
    idx = (mask.sum(dim=1) - 1).clamp_min(0)
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]


def extract_activations(
    model,
    tokenizer,
    rows: list[dict],
    layer_ids: list[int],
    pooling: str,
    template: str,
    max_length: int,
    batch_size: int,
    device,
    desc: str,
) -> tuple["torch.Tensor", int]:
    import torch

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(iterable, **kwargs):
            return iterable

    layers = get_decoder_layers(model)
    captured: dict[int, "torch.Tensor"] = {}
    handles = []

    def make_hook(layer_idx: int):
        def _hook(module, inputs, output):
            captured[layer_idx] = output[0] if isinstance(output, tuple) else output

        return _hook

    for layer_idx in layer_ids:
        handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

    per_layer: dict[int, list["torch.Tensor"]] = {layer_idx: [] for layer_idx in layer_ids}
    texts = [template.format(text=row["text"]) for row in rows]
    truncated = 0

    try:
        for start in tqdm(
            range(0, len(texts), batch_size), desc=desc, leave=False, total=(len(texts) + batch_size - 1) // batch_size
        ):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            truncated += int((attention_mask.sum(dim=1) >= max_length).sum())

            captured.clear()
            with torch.inference_mode():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            missing = [layer_idx for layer_idx in layer_ids if layer_idx not in captured]
            if missing:
                raise RuntimeError(f"hooks did not fire for layers {missing}")

            mask = attention_mask.bool()
            for layer_idx in layer_ids:
                per_layer[layer_idx].append(pool_hidden(captured[layer_idx], mask, pooling).cpu())
            captured.clear()
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()

    acts = torch.stack(
        [torch.cat(per_layer[layer_idx], dim=0) for layer_idx in layer_ids], dim=1
    )
    if acts.shape[0] != len(rows):
        raise RuntimeError(f"extracted {acts.shape[0]} rows, expected {len(rows)}")
    if not torch.isfinite(acts).all():
        raise RuntimeError("non-finite activations")
    return acts, truncated


def fit_diffmean(
    acts, labels: list[int], layer_ids: list[int], n_boot: int, seed: int
) -> tuple[dict, dict]:
    import torch
    import torch.nn.functional as F

    labels_t = torch.as_tensor(labels, dtype=torch.long)
    pos = acts[labels_t == 1]
    neg = acts[labels_t == 0]
    if pos.shape[0] == 0 or neg.shape[0] == 0:
        raise RuntimeError("both classes are required to fit a DiffMean vector")

    generator = torch.Generator().manual_seed(seed)
    diffs, mean_pos_list, mean_neg_list, resid_norms = [], [], [], []
    layer_reports = []

    for j, layer_idx in enumerate(layer_ids):
        p = pos[:, j, :]
        n = neg[:, j, :]
        mean_pos = p.mean(dim=0)
        mean_neg = n.mean(dim=0)
        diff = mean_pos - mean_neg
        if not torch.isfinite(diff).all():
            raise RuntimeError(f"non-finite DiffMean vector at layer {layer_idx}")

        norm_diff = float(diff.norm())
        resid_norm = float(acts[:, j, :].norm(dim=1).mean())
        cos_pos_neg = float(F.cosine_similarity(mean_pos, mean_neg, dim=0))
        pooled_var = 0.5 * (
            p.var(dim=0, unbiased=True).mean() + n.var(dim=0, unbiased=True).mean()
        )
        cohens_d = float(diff.norm() / pooled_var.sqrt().clamp_min(1e-12))

        boot_cos: list[float] = []
        for _ in range(n_boot):
            pos_idx = torch.randint(0, p.shape[0], (p.shape[0],), generator=generator)
            neg_idx = torch.randint(0, n.shape[0], (n.shape[0],), generator=generator)
            boot_diff = p[pos_idx].mean(dim=0) - n[neg_idx].mean(dim=0)
            boot_cos.append(float(F.cosine_similarity(diff, boot_diff, dim=0)))

        if boot_cos:
            boot_mean = sum(boot_cos) / len(boot_cos)
            boot_var = sum((value - boot_mean) ** 2 for value in boot_cos) / len(boot_cos)
            boot_std = boot_var**0.5
            boot_min = min(boot_cos)
        else:
            boot_mean = boot_std = boot_min = None

        diffs.append(diff)
        mean_pos_list.append(mean_pos)
        mean_neg_list.append(mean_neg)
        resid_norms.append(resid_norm)
        layer_reports.append(
            {
                "layer": layer_idx,
                "norm_diff": round(norm_diff, 6),
                "mean_pos_norm": round(float(mean_pos.norm()), 6),
                "mean_neg_norm": round(float(mean_neg.norm()), 6),
                "cos_pos_neg": round(cos_pos_neg, 6),
                "cohens_d": round(cohens_d, 6),
                "resid_norm_mean": round(resid_norm, 6),
                "bootstrap_cosine_mean": None if boot_mean is None else round(boot_mean, 6),
                "bootstrap_cosine_min": None if boot_min is None else round(boot_min, 6),
                "bootstrap_cosine_std": None if boot_std is None else round(boot_std, 6),
            }
        )

    tensors = {
        "diff": torch.stack(diffs, dim=0).contiguous(),  # [L, H] float32
        "mean_pos": torch.stack(mean_pos_list, dim=0).contiguous(),
        "mean_neg": torch.stack(mean_neg_list, dim=0).contiguous(),
        "resid_norm_mean": torch.tensor(resid_norms, dtype=torch.float32).contiguous(),
        "layer_ids": torch.tensor(layer_ids, dtype=torch.int32).contiguous(),
    }
    report = {
        "n_docs": int(acts.shape[0]),
        "n_pos": int(pos.shape[0]),
        "n_neg": int(neg.shape[0]),
        "bootstrap": n_boot,
        "seed": seed,
        "layers": layer_reports,
    }
    return tensors, report


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_save_safetensors(path: Path, tensors: dict) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(temporary))
    os.replace(temporary, path)


def save_vectors(
    paths: dict[str, Path], tensors: dict, meta: dict, report: dict
) -> None:
    atomic_save_safetensors(paths["vectors"], tensors)
    atomic_write_json(paths["meta"], meta)
    atomic_write_json(paths["report"], report)


def save_activations(
    paths: dict[str, Path],
    acts,
    labels: list[int],
    layer_ids: list[int],
    rows: list[dict],
    stat_path: Path,
    stat_sha: str,
) -> None:
    import torch

    atomic_save_safetensors(
        paths["acts"],
        {
            "acts": acts.to(torch.float16).contiguous(),
            "labels": torch.tensor(labels, dtype=torch.int8).contiguous(),
            "layer_ids": torch.tensor(layer_ids, dtype=torch.int32).contiguous(),
        },
    )
    atomic_write_json(
        paths["docs"],
        {
            "ids": [row["id"] for row in rows],
            "stat_path": str(stat_path.relative_to(REPO_ROOT)),
            "stat_sha256": stat_sha,
            "n": len(rows),
        },
    )


# ---------------------------------------------------------------------------
# Planning / checking (no torch / GPU needed)
# ---------------------------------------------------------------------------


def planned_tasks(
    name: str, spec: dict, langs: list[str], args: argparse.Namespace
) -> list[dict]:
    """Build task rows for --check/--dry-run without importing torch."""
    rows = []
    model_dir = Path(args.models_dir).expanduser() / name
    ready, reason = checkpoint_state(model_dir)
    commit = resolve_commit(name, args.models_dir)
    tokenizer_sha = (
        sha256_file(model_dir / "tokenizer.json")
        if ready and (model_dir / "tokenizer.json").is_file()
        else "missing"
    )

    if not ready:
        return [
            {
                "model": name,
                "lang": "-",
                "status": f"model not ready ({reason})",
                "layers": "-",
                "path": str(model_dir),
            }
        ]

    for lang in langs:
        stat_path, stat_rows = load_stat(lang)
        stat_sha = sha256_file(stat_path)
        try:
            layer_ids = parse_layers(args.layers, spec["num_hidden_layers"])
        except ValueError as exc:
            rows.append(
                {
                    "model": name,
                    "lang": lang,
                    "status": f"bad layers: {exc}",
                    "layers": "-",
                    "path": "-",
                }
            )
            continue
        key = build_cache_key(spec, commit, tokenizer_sha, stat_sha, layer_ids, args)
        paths = vector_paths(args.vectors_dir, name, commit, lang)
        ok, status, meta = cache_status(paths, key)
        rows.append(
            {
                "model": name,
                "lang": lang,
                "status": status,
                "layers": len(layer_ids),
                "path": str(paths["base"]),
                "n_docs": len(stat_rows),
                "cache_key": key,
                "created_utc": meta.get("created_utc") if meta else None,
                "cached": ok,
            }
        )
    return rows


def print_plan(rows: list[dict], title: str) -> None:
    print(title)
    header = f"{'model':12s} {'lang':4s} {'docs':>5s} {'layers':>6s}  status"
    print(header)
    print("-" * (len(header) + 8))
    for row in rows:
        docs = row.get("n_docs", "-")
        print(
            f"{row['model']:12s} {row['lang']:4s} {str(docs):>5s} "
            f"{str(row['layers']):>6s}  {row['status']}"
        )


def run_check(names: list[str], langs: list[str], args: argparse.Namespace) -> int:
    rows: list[dict] = []
    for name in names:
        rows.extend(planned_tasks(name, MODELS[name], langs, args))
    print_plan(rows, "[fit] Phase 1 vector cache")
    for row in rows:
        if row.get("cached"):
            print(f"[fit]   {row['model']}/{row['lang']}: {row['path']} ({row.get('created_utc')})")
    failures = [row for row in rows if not row.get("cached")]
    print(f"\n[fit] {len(rows) - len(failures)}/{len(rows)} cache entries valid")
    return 1 if failures else 0


def run_dry_run(names: list[str], langs: list[str], args: argparse.Namespace) -> int:
    rows: list[dict] = []
    for name in names:
        rows.extend(planned_tasks(name, MODELS[name], langs, args))
    print_plan(rows, "[fit] dry run: planned Phase 1 fits")
    for name in names:
        spec = MODELS[name]
        batch = args.batch_size or (4 if spec["params"] >= 7_000_000_000 else 8)
        print(
            f"[fit] {name}: layers={args.layers} pooling={args.pooling} "
            f"batch={batch} max_len={args.max_length} bootstrap={args.bootstrap}"
        )
    todo = [row for row in rows if not row.get("cached")]
    print(f"\n[fit] {len(todo)}/{len(rows)} entries need fitting")
    print("[fit] dry run: model not loaded")
    return 0


# ---------------------------------------------------------------------------
# Main fitting path
# ---------------------------------------------------------------------------


def load_model(name: str, spec: dict, args: argparse.Namespace, gpu_index: int):
    import torch
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    model_dir = Path(args.models_dir).expanduser() / name
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage=torch.uint8,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = truncation_side(args.pooling)

    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(model_dir),
        quantization_config=bnb,
        device_map={"": gpu_index},
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    vram = torch.cuda.memory_allocated(gpu_index) / 2**30
    print(f"[fit] {name}: loaded in {elapsed:.1f}s | VRAM {vram:.2f} GiB")
    return model, tokenizer


def process_model(name: str, spec: dict, langs: list[str], args: argparse.Namespace, device) -> bool:
    import torch

    model_dir = Path(args.models_dir).expanduser() / name
    ready, reason = checkpoint_state(model_dir)
    if not ready:
        print(
            f"[fit] {name}: model not ready ({reason}); "
            f"run scripts/download_models.py --only {name}",
            file=sys.stderr,
        )
        return False

    commit = resolve_commit(name, args.models_dir)
    if commit == "unknown":
        print(
            f"[fit] warning: no resolved commit for {name}; "
            "cache key will use 'unknown' (download first for reproducibility)"
        )
    tokenizer_sha = sha256_file(model_dir / "tokenizer.json")

    tasks = []
    for lang in langs:
        stat_path, stat_rows = load_stat(lang)
        stat_sha = sha256_file(stat_path)
        try:
            layer_ids = parse_layers(args.layers, spec["num_hidden_layers"])
        except ValueError as exc:
            print(f"[fit] {name}/{lang}: {exc}", file=sys.stderr)
            return False
        key = build_cache_key(spec, commit, tokenizer_sha, stat_sha, layer_ids, args)
        paths = vector_paths(args.vectors_dir, name, commit, lang)
        ok, status, meta = cache_status(paths, key)
        if ok and not args.force:
            print(
                f"[fit] {name}/{lang}: cached {len(meta['layers'])} layers "
                f"({paths['base']})"
            )
            continue
        if status == "stale key":
            print(f"[fit] {name}/{lang}: cache stale (key mismatch); refitting")
        tasks.append(
            {
                "lang": lang,
                "rows": stat_rows,
                "stat_path": stat_path,
                "stat_sha": stat_sha,
                "layer_ids": layer_ids,
                "key": key,
                "paths": paths,
            }
        )

    if not tasks:
        print(f"[fit] {name}: nothing to fit")
        return True

    if device is None:
        print(
            f"[fit] ERROR: {name} needs fitting, but CUDA is not available.\n"
            "  Run this on the T4/Colab machine.",
            file=sys.stderr,
        )
        return False

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    cuda_device = torch.device(f"cuda:{device_index}")
    batch_size = args.batch_size or (4 if spec["params"] >= 7_000_000_000 else 8)
    model = None
    tokenizer = None
    try:
        model, tokenizer = load_model(name, spec, args, device_index)
        for task in tasks:
            lang = task["lang"]
            rows = task["rows"]
            layer_ids = task["layer_ids"]
            n_pos = sum(1 for row in rows if row["label"] == 1)
            n_neg = len(rows) - n_pos
            print(
                f"[fit] {name}/{lang}: {len(rows)} docs ({n_pos}/{n_neg}), "
                f"{len(layer_ids)} layers, batch={batch_size}, "
                f"max_len={args.max_length}, pooling={args.pooling}"
            )
            started = time.perf_counter()
            acts, n_truncated = extract_activations(
                model=model,
                tokenizer=tokenizer,
                rows=rows,
                layer_ids=layer_ids,
                pooling=args.pooling,
                template=args.template,
                max_length=args.max_length,
                batch_size=batch_size,
                device=cuda_device,
                desc=f"{name}/{lang}",
            )
            elapsed = time.perf_counter() - started
            if n_truncated:
                share = n_truncated / len(rows)
                print(
                    f"[fit] {name}/{lang}: warning: {n_truncated}/{len(rows)} "
                    f"({share:.1%}) docs hit max_length={args.max_length}"
                )

            tensors, report = fit_diffmean(
                acts=acts,
                labels=[row["label"] for row in rows],
                layer_ids=layer_ids,
                n_boot=args.bootstrap,
                seed=args.seed,
            )
            report.update(
                {
                    "model": name,
                    "repo_id": spec["repo_id"],
                    "commit": commit,
                    "lang": lang,
                    "pooling": args.pooling,
                    "template": args.template,
                    "max_length": args.max_length,
                    "n_truncated": n_truncated,
                    "extraction_seconds": round(elapsed, 2),
                }
            )

            meta = {
                "cache_version": CACHE_VERSION,
                "cache_key": task["key"],
                "created_utc": utc_now(),
                "model": {
                    "key": name,
                    "repo_id": spec["repo_id"],
                    "commit": commit,
                    "params": spec["params"],
                    "num_hidden_layers": spec["num_hidden_layers"],
                    "hidden_size": spec["hidden_size"],
                    "tokenizer_sha256": tokenizer_sha,
                    "quant_tag": QUANT_TAG,
                    "attn_implementation": "sdpa",
                    "quantization": {
                        "load_in_4bit": True,
                        "quant_type": "nf4",
                        "double_quant": True,
                        "compute_dtype": "float16",
                        "quant_storage": "uint8",
                    },
                    "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                    "bitsandbytes": package_version("bitsandbytes"),
                    "gpu": torch.cuda.get_device_name(device_index),
                },
                "data": {
                    "stat_path": str(task["stat_path"].relative_to(REPO_ROOT)),
                    "stat_sha256": task["stat_sha"],
                    "n_docs": len(rows),
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                },
                "extraction": {
                    "site": SITE,
                    "pooling": args.pooling,
                    "truncation_side": truncation_side(args.pooling),
                    "template": args.template,
                    "max_length": args.max_length,
                    "add_special_tokens": False,
                    "chat_template": False,
                    "padding_side": "right",
                    "layer_ids": layer_ids,
                    "batch_size": batch_size,
                    "n_truncated": n_truncated,
                },
                "fit": {
                    "method": "diffmean",
                    "sign": "mean_positive - mean_negative",
                    "dtype": "float32",
                    "bootstrap": args.bootstrap,
                    "seed": args.seed,
                },
                "layers": report["layers"],
            }

            save_vectors(task["paths"], tensors, meta, report)
            print(
                f"[fit] {name}/{lang}: saved {len(layer_ids)} layers in {elapsed:.1f}s "
                f"-> {task['paths']['vectors']}"
            )

            best = max(report["layers"], key=lambda item: item["norm_diff"])
            boot_values = [
                item["bootstrap_cosine_mean"]
                for item in report["layers"]
                if item["bootstrap_cosine_mean"] is not None
            ]
            boot_text = (
                f"{sum(boot_values) / len(boot_values):.3f}"
                if boot_values
                else "n/a"
            )
            print(
                f"[fit] {name}/{lang}: best |v| layer {best['layer']} "
                f"({best['norm_diff']:.3f}); mean bootstrap cosine {boot_text}"
            )

            if args.save_activations:
                act_paths = activation_paths(
                    args.activations_dir, name, commit, lang, args.pooling
                )
                save_activations(
                    act_paths,
                    acts=acts,
                    labels=[row["label"] for row in rows],
                    layer_ids=layer_ids,
                    rows=rows,
                    stat_path=task["stat_path"],
                    stat_sha=task["stat_sha"],
                )
                print(
                    f"[fit] {name}/{lang}: activations {human_bytes(act_paths['acts'].stat().st_size)} "
                    f"-> {act_paths['acts']}"
                )

            del acts
            torch.cuda.empty_cache()
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    return True


def main() -> int:
    args = parse_args()
    names = select_models(args.models)
    langs = parse_langs(args.langs)

    if args.check:
        return run_check(names, langs, args)
    if args.dry_run:
        return run_dry_run(names, langs, args)

    try:
        import torch
    except ImportError:
        print(
            "[fit] ERROR: torch is not installed. On Colab/T4:\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    device = None
    if torch.cuda.is_available():
        device = torch.device(args.device)
        if device.type != "cuda":
            print(f"[fit] ERROR: --device must be a CUDA device, got {args.device}", file=sys.stderr)
            return 2
        gpu_index = device.index if device.index is not None else torch.cuda.current_device()
        print(
            f"[fit] GPU: {torch.cuda.get_device_name(gpu_index)} | "
            f"model(s): {', '.join(names)} | languages: {', '.join(langs)}"
        )
    else:
        print(
            "[fit] warning: no CUDA device; cached entries can be skipped, "
            "but fitting requires a GPU"
        )

    print(
        f"[fit] pooling={args.pooling} template={args.template!r} "
        f"max_len={args.max_length} layers={args.layers} bootstrap={args.bootstrap}"
    )

    failures = 0
    for name in names:
        try:
            ok = process_model(name, MODELS[name], langs, args, device)
        except Exception as exc:  # keep going with the next model
            import traceback

            print(f"[fit] ERROR: {name} failed: {exc}", file=sys.stderr)
            traceback.print_exc()
            ok = False
        if not ok:
            failures += 1

    if failures:
        print(f"[fit] finished with {failures} failed model(s)", file=sys.stderr)
        return 1
    print("[fit] done: all requested vectors are cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
