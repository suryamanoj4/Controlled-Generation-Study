#!/usr/bin/env python3
"""Download the Qwen3 checkpoints used by SAS-E2.

The experiment runs 4-bit NF4 (bitsandbytes) with fp16 compute on a T4.
Qwen3 does not publish NF4 checkpoints: NF4 is applied when the model is
loaded. This script therefore downloads the official fp16 checkpoints; the
loader quantizes them to NF4 on the GPU. The sizes printed below are
download sizes, not 4-bit sizes.

Models:
  qwen3-1.7b   2.03B params    ~4.1 GB download   ~1.4 GiB VRAM after NF4 load
  qwen3-4b     4.02B params    ~8.1 GB download   ~2.5 GiB VRAM after NF4 load
  qwen3-8b     8.19B params   ~16.4 GB download   ~5.7 GiB VRAM after NF4 load

Loading (in the experiment code):
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,   # T4 has no bf16
    )

Usage:
  python3 scripts/download_models.py --models all
  python3 scripts/download_models.py --only qwen3-4b
  python3 scripts/download_models.py --models qwen3-1.7b,qwen3-8b
  python3 scripts/download_models.py --check          # local status only
  python3 scripts/download_models.py --dry-run        # plan, no network

Downloads are resumable. hf_transfer is used automatically if installed;
pass --no-fast to disable it. Qwen3 is Apache-2.0 and ungated, but setting
HF_TOKEN avoids anonymous rate limits.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _models import (
    ALLOW_PATTERNS,
    DEFAULT_MODELS_DIR,
    MODELS,
    checkpoint_state,
    dir_bytes,
    human_bytes,
    select_models,
    utc_now,
)

DEFAULT_DEST = DEFAULT_MODELS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Qwen3 fp16 checkpoints for SAS-E2 NF4 runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        "--only",
        dest="models",
        default="all",
        metavar="LIST",
        help=(
            "comma-separated keys (qwen3-1.7b, qwen3-4b, qwen3-8b) or 'all' "
            "(default: all)"
        ),
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help=f"model directory (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="override the revision for all models (pin a commit hash for final runs)",
    )
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads (default: 4)")
    parser.add_argument("--token", default=None, help="Hugging Face token (or set HF_TOKEN)")
    parser.add_argument("--no-fast", action="store_true", help="disable hf_transfer")
    parser.add_argument("--check", action="store_true", help="report local status only; no network")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; download nothing")
    return parser.parse_args()


def print_table(names: list[str], dest: Path) -> None:
    header = (
        f"{'model':12s} {'repo':18s} {'download':>10s} "
        f"{'VRAM (NF4)':>10s}  status"
    )
    print(header)
    print("-" * len(header))
    for name in names:
        spec = MODELS[name]
        path = dest / name
        ok, reason = checkpoint_state(path)
        local = dir_bytes(path)
        if ok:
            status = f"complete ({reason}, {human_bytes(local)})"
        else:
            status = f"{reason} ({human_bytes(local)} local)"
        print(
            f"{name:12s} {spec['repo_id']:18s} "
            f"{spec['download_gb']:8.2f} GB "
            f"{spec['vram_gib_after_load']:8.1f} GiB  {status}"
        )


def required_bytes(names: list[str], dest: Path) -> int:
    total = 0
    for name in names:
        path = dest / name
        ok, _ = checkpoint_state(path)
        if ok:
            continue
        expected = int(MODELS[name]["download_gb"] * 1e9)
        total += max(expected - dir_bytes(path), 0)
    return total


def configure_hf(no_fast: bool, token: str | None) -> None:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if token:
        os.environ["HF_TOKEN"] = token
    if no_fast:
        return
    try:
        import hf_transfer  # noqa: F401
    except ImportError:
        return
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def download_one(name: str, path: Path, revision: str | None, workers: int) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    spec = MODELS[name]
    requested = revision or spec.get("revision", "main")
    print(f"\n[download] {name}: {spec['repo_id']}@{requested} -> {path}")
    path.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=spec["repo_id"],
        revision=requested,
        local_dir=str(path),
        allow_patterns=ALLOW_PATTERNS,
        max_workers=workers,
        token=os.environ.get("HF_TOKEN") or None,
    )

    ok, reason = checkpoint_state(path)
    if not ok:
        raise RuntimeError(f"{name}: download incomplete ({reason})")

    resolved = requested
    try:
        resolved = HfApi(token=os.environ.get("HF_TOKEN") or None).model_info(
            spec["repo_id"], revision=requested
        ).sha
    except Exception as exc:  # network hiccup shouldn't invalidate a good download
        print(f"[download] warning: could not resolve commit for {name}: {exc}")

    return {
        "repo_id": spec["repo_id"],
        "requested_revision": requested,
        "resolved_commit": resolved,
        "local_dir": str(path),
        "bytes": dir_bytes(path),
        "params": spec["params"],
        "downloaded_utc": utc_now(),
    }


def update_manifest(dest: Path, entries: dict[str, dict]) -> None:
    manifest_path = dest / "download_manifest.json"
    data: dict = {"models": {}}
    if manifest_path.is_file():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    data.setdefault("models", {}).update(entries)
    data["updated_utc"] = utc_now()
    manifest_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[download] manifest: {manifest_path}")


def main() -> int:
    args = parse_args()
    names = select_models(args.models)
    dest = Path(args.dest).expanduser().resolve()

    print("[download] SAS-E2 Qwen3 checkpoints")
    print("[download] NF4 is applied at load time; these are fp16 checkpoints")
    print(f"[download] destination: {dest}")
    print_table(names, dest)

    if args.check:
        missing = [name for name in names if not checkpoint_state(dest / name)[0]]
        print(
            f"\n[download] {len(names) - len(missing)}/{len(names)} selected "
            f"checkpoint(s) complete"
        )
        return 1 if missing else 0

    needed = required_bytes(names, dest)
    if needed == 0:
        print("\n[download] all selected checkpoints are already complete")
        return 0

    probe = dest if dest.exists() else dest.parent
    free = shutil.disk_usage(probe).free
    print(
        f"\n[download] still needed: {human_bytes(needed)} | "
        f"free disk: {human_bytes(free)}"
    )

    if args.dry_run:
        print("[download] dry run: nothing downloaded")
        return 0

    if free < needed * 1.05:
        print(
            "[download] ERROR: not enough free disk (need 5% headroom)",
            file=sys.stderr,
        )
        return 1

    configure_hf(args.no_fast, args.token)
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print(
            "[download] ERROR: huggingface_hub is not installed.\n"
            "  On Colab:   pip install -U 'huggingface_hub[hf_transfer]'\n"
            "  In sas-e2:  uv sync && uv run python scripts/download_models.py ...",
            file=sys.stderr,
        )
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict] = {}
    for name in names:
        path = dest / name
        ok, _ = checkpoint_state(path)
        if ok:
            print(f"[download] {name}: already complete, skipping")
            continue
        try:
            entries[name] = download_one(
                name, path, revision=args.revision, workers=args.workers
            )
        except Exception as exc:
            print(f"[download] ERROR: {name} failed: {exc}", file=sys.stderr)
            return 1

    if entries:
        update_manifest(dest, entries)

    print("\n[download] final status")
    print_table(names, dest)

    missing = [name for name in names if not checkpoint_state(dest / name)[0]]
    if missing:
        print(f"[download] ERROR: incomplete: {', '.join(missing)}", file=sys.stderr)
        return 1

    print("\n[download] done.")
    print("[download] Peak VRAM adds ~0.5-1 GiB for CUDA context, activations and KV cache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
