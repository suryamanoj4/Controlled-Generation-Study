#!/usr/bin/env python3
"""Small end-to-end generation smoke test for Phase 1 steering vectors.

Loads a Qwen3 checkpoint in 4-bit NF4, loads the cached DiffMean vector for
each requested language, and generates a couple of neutral review prompts
with and without the vector. It checks the failures that broke SAS-E1:
empty output, wrong script/language, thinking-token leakage, and a vector
that does not change the output.

Default: qwen3-4b, all five languages, 2 prompts/language, 64 new tokens.
That is 20 generations - small enough to run before the full eval.

The vector is fitted on the base transformer (AutoModel) but applied here to
the causal LM (AutoModelForCausalLM); the decoder-block residual site is the
same. Injection is added to every position at the chosen block's output, using
the E1 convention h' = h + alpha * (mean_pos - mean_neg).

Examples:
  python3 scripts/test_generation.py
  python3 scripts/test_generation.py --only qwen3-4b --lang hi --prompts 1
  python3 scripts/test_generation.py --langs hi,te --layer 18 --alpha 2.0
  python3 scripts/test_generation.py --models all
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _models import (  # noqa: E402
    CACHE_VERSION,
    DATASET_ROOT,
    DEFAULT_MODELS_DIR,
    DEFAULT_VECTORS_DIR,
    LANGS,
    MODELS,
    REPO_ROOT,
    checkpoint_state,
    human_bytes,
    resolve_commit,
    select_models,
    sha256_file,
    utc_now,
)
from fit_vectors import get_decoder_layers, read_json, vector_paths  # noqa: E402

SCRIPTS = {
    "hi": re.compile(r"[\u0900-\u097F]"),
    "bn": re.compile(r"[\u0980-\u09FF]"),
    "te": re.compile(r"[\u0C00-\u0C7F]"),
    "ta": re.compile(r"[\u0B80-\u0BFF]"),
}
THINK_TOKENS = ("<think>", "</think>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test Phase 1 generation with a cached steering vector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        "--only",
        dest="models",
        default="qwen3-4b",
        metavar="LIST",
        help="comma-separated keys or 'all' (default: qwen3-4b)",
    )
    parser.add_argument(
        "--langs",
        "--lang",
        dest="langs",
        default="all",
        metavar="LIST",
        help="comma-separated languages or 'all' (default: all)",
    )
    parser.add_argument("--prompts", type=int, default=2, help="prompts per language (default: 2)")
    parser.add_argument(
        "--max-new-tokens", type=int, default=64, help="generation length (default: 64)"
    )
    parser.add_argument(
        "--layer",
        default="auto",
        help="decoder block index, or 'auto' for max Cohen's d (default: auto)",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="injection strength (default: 1.0)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--system",
        default=None,
        help="optional system message (same for every language)",
    )
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--vectors-dir", default=str(DEFAULT_VECTORS_DIR))
    parser.add_argument("--out", default=None, help="JSON output path")
    parser.add_argument("--no-save", action="store_true", help="do not write the JSON report")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.prompts <= 0:
        parser.error("--prompts must be > 0")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be > 0")
    return args


def parse_langs(spec: str) -> list[str]:
    if spec is None or spec.strip().lower() in ("all", "*", ""):
        return list(LANGS)
    wanted = [part.strip() for part in spec.split(",") if part.strip()]
    unknown = [lang for lang in wanted if lang not in LANGS]
    if unknown:
        raise SystemExit(
            f"[smoke] unknown language(s): {', '.join(unknown)}\n"
            f"[smoke] available: {', '.join(LANGS)}"
        )
    wanted_set = set(wanted)
    return [lang for lang in LANGS if lang in wanted_set]


def load_prompts(lang: str, count: int) -> list[dict]:
    path = DATASET_ROOT / "prompts" / f"{lang}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing prompt file: {path}")
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) < count:
        raise RuntimeError(f"{path}: requested {count} prompts, only {len(rows)} available")
    return rows[:count]


def script_ratio(text: str, lang: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    if lang == "en":
        return sum(1 for char in letters if char.isascii()) / len(letters)
    pattern = SCRIPTS[lang]
    return sum(1 for char in letters if pattern.match(char)) / len(letters)


def rep4_ratio(text: str) -> float:
    words = text.split()
    if len(words) < 5:
        return 0.0
    grams = [tuple(words[index : index + 4]) for index in range(len(words) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def choose_layer(requested: str, layer_ids: list[int], report: dict | None) -> int:
    if str(requested).strip().lower() != "auto":
        layer = int(requested)
        if layer not in layer_ids:
            raise ValueError(f"layer {layer} not in fitted layers {layer_ids}")
        return layer
    if report and report.get("layers"):
        candidates = [entry for entry in report["layers"] if entry.get("layer") in layer_ids]
        if candidates:
            best = max(candidates, key=lambda entry: float(entry.get("cohens_d") or 0.0))
            return int(best["layer"])
    return layer_ids[len(layer_ids) // 2]


def load_vector(
    model_key: str,
    commit: str,
    lang: str,
    requested_layer: str,
    vectors_dir: str,
):
    import torch
    from safetensors.torch import load_file

    paths = vector_paths(vectors_dir, model_key, commit, lang)
    if not paths["vectors"].is_file() or not paths["meta"].is_file():
        raise FileNotFoundError(
            f"no cached vector for {model_key}/{lang} at {paths['base']}; "
            f"run scripts/fit_vectors.py --only {model_key} --lang {lang}"
        )
    meta = read_json(paths["meta"])
    if not meta:
        raise RuntimeError(f"bad vector metadata: {paths['meta']}")
    if meta.get("cache_version") != CACHE_VERSION:
        raise RuntimeError(
            f"{paths['meta']}: cache_version={meta.get('cache_version')} "
            f"!= {CACHE_VERSION}; rerun fit_vectors.py"
        )

    stat_path = DATASET_ROOT / lang / "stat.jsonl"
    if sha256_file(stat_path) != meta["data"]["stat_sha256"]:
        raise RuntimeError(
            f"{model_key}/{lang}: vector cache is stale (stat.jsonl changed); "
            "rerun fit_vectors.py"
        )

    report = read_json(paths["report"])
    tensors = load_file(str(paths["vectors"]))
    layer_ids = [int(value) for value in meta["extraction"]["layer_ids"]]
    layer = choose_layer(requested_layer, layer_ids, report)
    row = layer_ids.index(layer)

    diff = tensors["diff"][row].float()
    resid_norm = float(tensors["resid_norm_mean"][row])
    if diff.numel() != int(meta["model"]["hidden_size"]):
        raise RuntimeError(
            f"{model_key}/{lang}: vector width {diff.numel()} != "
            f"hidden size {meta['model']['hidden_size']}"
        )
    if not torch.isfinite(diff).all() or diff.norm() <= 0:
        raise RuntimeError(f"{model_key}/{lang}: invalid vector at layer {layer}")
    if resid_norm <= 0:
        raise RuntimeError(f"{model_key}/{lang}: invalid residual norm at layer {layer}")

    return diff, resid_norm, layer, meta, paths, report


def load_model(model_key: str, args: argparse.Namespace, gpu_index: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_dir = Path(args.models_dir).expanduser() / model_key
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
    tokenizer.padding_side = "left"

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
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
    print(f"[smoke] {model_key}: loaded in {elapsed:.1f}s | VRAM {vram:.2f} GiB")
    return model, tokenizer


def make_steering_hook(injection):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            return (hidden + injection.to(device=hidden.device, dtype=hidden.dtype),) + output[1:]
        return output + injection.to(device=output.device, dtype=output.dtype)

    return hook


def generate_one(
    model,
    tokenizer,
    prompt: str,
    lang: str,
    args: argparse.Namespace,
    seed: int,
    layer: int,
    injection,
) -> dict:
    import torch

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError as exc:
        raise RuntimeError(
            "apply_chat_template(enable_thinking=False) failed; need transformers>=4.51"
        ) from exc

    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    prompt_len = int(inputs["input_ids"].shape[1])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    handle = None
    if injection is not None:
        layers = get_decoder_layers(model)
        handle = layers[layer].register_forward_hook(make_steering_hook(injection))

    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        if handle is not None:
            handle.remove()
    elapsed = time.perf_counter() - started

    new_ids = output[0, prompt_len:]
    ids = [int(token) for token in new_ids.tolist()]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    think_ids = {tokenizer.convert_tokens_to_ids(token) for token in THINK_TOKENS}
    think_ids.discard(tokenizer.unk_token_id)
    think_ids.discard(None)

    return {
        "text": text,
        "n_new_tokens": len(ids),
        "empty": not text,
        "hit_eos": bool(ids) and ids[-1] == tokenizer.eos_token_id,
        "think_leak": any(token in think_ids for token in ids),
        "script_ratio": script_ratio(text, lang),
        "rep4": rep4_ratio(text),
        "seconds": round(elapsed, 2),
    }


def summarize(lang: str, samples: list[dict]) -> dict:
    baseline = [row for row in samples if row["condition"] == "baseline"]
    vector = [row for row in samples if row["condition"] == "vector"]

    def stats(rows: list[dict]) -> dict:
        nonempty = [row for row in rows if not row["empty"]]
        return {
            "n": len(rows),
            "empty": sum(1 for row in rows if row["empty"]),
            "script_mean": (
                sum(row["script_ratio"] for row in nonempty) / len(nonempty)
                if nonempty
                else 0.0
            ),
            "tokens_mean": (
                sum(row["n_new_tokens"] for row in rows) / len(rows) if rows else 0.0
            ),
            "think_leak": sum(1 for row in rows if row["think_leak"]),
            "rep4_mean": sum(row["rep4"] for row in rows) / len(rows) if rows else 0.0,
            "truncated": sum(1 for row in rows if not row["hit_eos"]),
        }

    return {
        "lang": lang,
        "n_prompts": len(baseline),
        "changed": sum(1 for row in vector if row["changed"]),
        "baseline": stats(baseline),
        "vector": stats(vector),
    }


def run_language(model, tokenizer, model_key, commit, lang, args, gpu_device) -> tuple:
    import torch

    prompts = load_prompts(lang, args.prompts)
    diff, resid_norm, layer, meta, paths, report = load_vector(
        model_key, commit, lang, args.layer, args.vectors_dir
    )
    injection = (diff * args.alpha).to(device=gpu_device, dtype=torch.float16)
    vector_info = {
        "layer": layer,
        "pooling": meta["extraction"]["pooling"],
        "alpha": args.alpha,
        "vector_norm": float(diff.norm()),
        "resid_norm": resid_norm,
        "injection_norm": float(injection.norm()),
    }

    print(
        f"[smoke] {model_key}/{lang}: layer={layer} pooling={meta['extraction']['pooling']} "
        f"alpha={args.alpha} |v|={vector_info['vector_norm']:.3f} "
        f"|inject|={vector_info['injection_norm']:.3f} resid={resid_norm:.3f}"
    )

    samples = []
    for index, row in enumerate(prompts):
        seed = args.seed + 1000 * LANGS.index(lang) + index
        baseline = generate_one(
            model, tokenizer, row["prompt"], lang, args, seed, layer, injection=None
        )
        baseline.update(
            {
                "condition": "baseline",
                "lang": lang,
                "prompt_id": row["id"],
                "prompt": row["prompt"],
                "changed": None,
            }
        )

        steered = generate_one(
            model, tokenizer, row["prompt"], lang, args, seed, layer, injection=injection
        )
        steered.update(
            {
                "condition": "vector",
                "lang": lang,
                "prompt_id": row["id"],
                "prompt": row["prompt"],
                "changed": baseline["text"] != steered["text"],
            }
        )
        samples.extend([baseline, steered])

        for label, sample in (("baseline", baseline), ("vector", steered)):
            text = sample["text"] if sample["text"] else "<EMPTY>"
            clipped = " ".join(text.split())
            clipped = clipped if len(clipped) <= 260 else clipped[:260] + " …"
            print(
                f"[smoke] {lang}/{row['id']} {label:8s} "
                f"{sample['n_new_tokens']:3d} tok {sample['seconds']:5.2f}s: {clipped}"
            )

    summary = summarize(lang, samples)
    return samples, summary, meta, paths, vector_info


def run_model(model_key: str, args: argparse.Namespace, gpu_index: int) -> dict:
    import torch

    model_dir = Path(args.models_dir).expanduser() / model_key
    ready, reason = checkpoint_state(model_dir)
    if not ready:
        raise RuntimeError(
            f"{model_key}: model not ready ({reason}); "
            f"run scripts/download_models.py --only {model_key}"
        )

    commit = resolve_commit(model_key, args.models_dir)
    if commit == "unknown":
        print(
            f"[smoke] warning: no resolved commit for {model_key}; "
            "using the 'unknown' vector cache"
        )

    model, tokenizer = load_model(model_key, args, gpu_index)
    gpu_device = torch.device(f"cuda:{gpu_index}")
    samples_all, summaries, errors = [], [], []
    try:
        for lang in args.langs:
            try:
                samples, summary, meta, paths, vector_info = run_language(
                    model, tokenizer, model_key, commit, lang, args, gpu_device
                )
                samples_all.extend(samples)
                summaries.append(
                    {
                        "model": model_key,
                        **summary,
                        **vector_info,
                        "vector_path": str(paths["base"]),
                        "cache_key": meta.get("cache_key"),
                        "vector_created_utc": meta.get("created_utc"),
                    }
                )
            except Exception as exc:
                message = f"{model_key}/{lang}: {exc}"
                print(f"[smoke] ERROR: {message}", file=sys.stderr)
                errors.append(message)
    finally:
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "model": model_key,
        "repo_id": MODELS[model_key]["repo_id"],
        "commit": commit,
        "gpu": torch.cuda.get_device_name(gpu_index),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "samples": samples_all,
        "summaries": summaries,
        "errors": errors,
    }


def print_table(summaries: list[dict]) -> None:
    header = (
        f"{'model':12s} {'lang':4s} {'empty':>5s} {'script base/vec':>15s} "
        f"{'changed':>7s} {'think':>5s} {'tok base/vec':>15s}"
    )
    print(header)
    print("-" * len(header))
    for row in summaries:
        base, vec = row["baseline"], row["vector"]
        empty = base["empty"] + vec["empty"]
        changed = f"{row['changed']}/{row['n_prompts']}"
        think = base["think_leak"] + vec["think_leak"]
        print(
            f"{row['model']:12s} {row['lang']:4s} {empty:5d} "
            f"{base['script_mean']:6.2f}/{vec['script_mean']:.2f}     "
            f"{changed:>7s} {think:5d} "
            f"{base['tokens_mean']:7.1f}/{vec['tokens_mean']:.1f}"
        )


def evaluate_failures(summaries: list[dict], errors: list[str], alpha: float) -> list[str]:
    failures = list(errors)
    for row in summaries:
        tag = f"{row['model']}/{row['lang']}"
        base, vec = row["baseline"], row["vector"]
        if base["empty"] or vec["empty"]:
            failures.append(f"{tag}: {base['empty'] + vec['empty']} empty generation(s)")
        if base["think_leak"] or vec["think_leak"]:
            failures.append(f"{tag}: thinking-token leakage")
        if base["script_mean"] < 0.5:
            failures.append(f"{tag}: baseline script ratio {base['script_mean']:.2f} < 0.5")
        if vec["script_mean"] < 0.5:
            failures.append(f"{tag}: vector script ratio {vec['script_mean']:.2f} < 0.5")
        if vec["rep4_mean"] > 0.5:
            failures.append(f"{tag}: vector repetition {vec['rep4_mean']:.2f} > 0.5")
        if alpha != 0 and row["changed"] == 0:
            failures.append(f"{tag}: vector did not change any output")
    return failures


def main() -> int:
    args = parse_args()
    model_keys = select_models(args.models)
    args.langs = parse_langs(args.langs)

    try:
        import torch
    except ImportError:
        print(
            "[smoke] ERROR: torch is not installed. On Colab/T4:\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    if not torch.cuda.is_available():
        print(
            "[smoke] ERROR: CUDA is required (4-bit bitsandbytes). "
            "Run this on the T4/Colab machine.",
            file=sys.stderr,
        )
        return 2

    device = torch.device(args.device)
    if device.type != "cuda":
        print(f"[smoke] ERROR: --device must be CUDA, got {args.device}", file=sys.stderr)
        return 2
    gpu_index = device.index if device.index is not None else torch.cuda.current_device()

    print(
        f"[smoke] GPU: {torch.cuda.get_device_name(gpu_index)} | "
        f"model(s): {', '.join(model_keys)} | languages: {', '.join(args.langs)} | "
        f"prompts/lang: {args.prompts}"
    )

    runs = []
    summaries = []
    errors = []
    for model_key in model_keys:
        try:
            run = run_model(model_key, args, gpu_index)
        except Exception as exc:
            import traceback

            print(f"[smoke] ERROR: {model_key}: {exc}", file=sys.stderr)
            traceback.print_exc()
            errors.append(f"{model_key}: {exc}")
            continue
        runs.append(run)
        summaries.extend(run["summaries"])
        errors.extend(run["errors"])

    if summaries:
        print()
        print_table(summaries)

    failures = evaluate_failures(summaries, errors, args.alpha)
    passed = not failures

    print()
    if passed:
        print(
            f"[smoke] PASS: {len(summaries)} language/model pairs, "
            f"{len(summaries) * args.prompts * 2} generations"
        )
    else:
        print(f"[smoke] FAIL ({len(failures)} issue(s)):", file=sys.stderr)
        for failure in failures:
            print(f"[smoke]   - {failure}", file=sys.stderr)

    if not args.no_save:
        out = (
            Path(args.out).expanduser()
            if args.out
            else REPO_ROOT
            / "artifacts"
            / "smoke_generation"
            / f"smoke_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_utc": utc_now(),
            "settings": {
                "models": model_keys,
                "langs": args.langs,
                "prompts": args.prompts,
                "max_new_tokens": args.max_new_tokens,
                "layer": args.layer,
                "alpha": args.alpha,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "seed": args.seed,
                "system": args.system,
            },
            "runs": runs,
            "summaries": summaries,
            "errors": errors,
            "failures": failures,
            "passed": passed,
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[smoke] report: {out} ({human_bytes(out.stat().st_size)})")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
