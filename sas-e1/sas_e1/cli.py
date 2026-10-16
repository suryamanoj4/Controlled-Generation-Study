"""CLI: python -m sas_e1.cli --config configs/lean.yaml --stage extract|fit|eval|probes|analyze|all"""
from __future__ import annotations

import argparse

from .config import load_config, save_config
from .data.download import download_models, prepare_all
from .utils import ensure_dir


def cmd_download(args) -> None:
    cfg = load_config(args.config)
    ensure_dir(cfg.paths.data_processed)

    data_only = bool(getattr(args, "data_only", False) or getattr(args, "skip_models", False))
    do_models = not data_only and not getattr(args, "models_only", False)
    do_data = not getattr(args, "models_only", False)

    if do_models:
        download_models(cfg)
    if do_data:
        prepare_all(cfg, force=args.force, include_toxigen=args.with_toxigen)
    save_config(cfg, f"{cfg.paths.artifacts}/{cfg.run_name}/config.json")
    print(f"[download] done (models={'yes' if do_models else 'skipped'}, "
          f"data={'yes' if do_data else 'skipped'})")


def cmd_run(args) -> None:
    from .experiment import analyze, run_stage  # lazy: requires torch

    cfg = load_config(args.config)
    ensure_dir(f"{cfg.paths.artifacts}/{cfg.run_name}")
    save_config(cfg, f"{cfg.paths.artifacts}/{cfg.run_name}/config.json")
    stages = ["extract", "fit", "eval", "probes", "analyze"] if args.stage == "all" else [args.stage]
    attrs = args.attr.split(",") if args.attr else None
    for stage in stages:
        if stage == "analyze":
            analyze(cfg)
        else:
            run_stage(cfg, stage, attrs=attrs, force=args.force)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SAS Experiment 1")
    parser.add_argument("--config", default="configs/lean.yaml")
    parser.add_argument("--stage", default="all",
                        choices=["download", "extract", "fit", "eval", "probes", "analyze", "all"])
    parser.add_argument("--attr", default=None, help="comma-separated attribute subset")
    parser.add_argument("--force", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--data-only", action="store_true",
                       help="download stage: datasets only (skip model snapshots)")
    group.add_argument("--models-only", action="store_true",
                       help="download stage: model snapshots only (skip datasets)")
    parser.add_argument("--skip-models", action="store_true", help=argparse.SUPPRESS)  # legacy alias
    parser.add_argument("--with-toxigen", action="store_true", help="download stage: optional ToxiGen arm")
    args = parser.parse_args(argv)

    if args.stage == "download":
        cmd_download(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
