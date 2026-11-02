#!/usr/bin/env python3
"""Final Phase 1 pilot: 3x3 prompt/vector grid + random control.

For every language this generates:

  prompt sentiment: positive, neutral, negative
  vector:          none, +alpha, -alpha
  plus:            neutral + norm-matched random vector

This gives the prompting control (positive > neutral > negative), the
headroom tests (+vector on a negative prompt, -vector on a positive prompt)
and the random-vector control, all with the same prompts and batch seeds.

The cached vector is fitted by fit_vectors.py. Sentiment is scored by the
XLM-R classifier trained with train_classifier.py.

Example:
  python3 scripts/pilot_generation.py --only qwen3-4b --prompts 10 --layer 18
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _models import (  # noqa: E402
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
    utc_now,
)
from fit_vectors import read_json  # noqa: E402
from test_generation import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    generate_batch,
    load_model,
    load_vector,
)

DEFAULT_CLASSIFIER_DIR = REPO_ROOT / "artifacts" / "classifier" / "xlmr_phase1"

# More direct in-language instructions. These are pilot drafts; native
# verification is still required before the final dataset is frozen.
DIRECT_TEMPLATES = {
    "neutral": {
        "en": "Write a review of {topic}.",
        "hi": "इस विषय पर एक समीक्षा लिखिए: {topic}।",
        "bn": "এই বিষয়ে একটি পর্যালোচনা লিখুন: {topic}।",
        "te": "ఈ అంశంపై ఒక సమీక్ష రాయండి: {topic}.",
        "ta": "இந்தத் தலைப்பில் ஒரு விமர்சனம் எழுதுங்கள்: {topic}.",
    },
    "positive": {
        "en": "Write a positive review of {topic}.",
        "hi": "इस विषय पर एक सकारात्मक समीक्षा लिखिए: {topic}।",
        "bn": "এই বিষয়ে একটি ইতিবাচক পর্যালোচনা লিখুন: {topic}।",
        "te": "ఈ అంశంపై ఒక సానుకూల సమీక్ష రాయండి: {topic}.",
        "ta": "இந்தத் தலைப்பில் ஒரு நேர்மறையான விமர்சனம் எழுதுங்கள்: {topic}.",
    },
    "negative": {
        "en": "Write a negative review of {topic}.",
        "hi": "इस विषय पर एक नकारात्मक समीक्षा लिखिए: {topic}।",
        "bn": "এই বিষয়ে একটি নেতিবাচক পর্যালোচনা লিখুন: {topic}।",
        "te": "ఈ అంశంపై ఒక ప్రతికూల సమీక్ష రాయండి: {topic}.",
        "ta": "இந்தத் தலைப்பில் ஒரு எதிர்மறையான விமர்சனம் எழுதுங்கள்: {topic}.",
    },
}

PROMPT_KINDS = ("positive", "neutral", "negative")
VECTOR_MODES = ("none", "plus", "minus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final Phase 1 pilot with prompting and random-vector controls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--models", "--only", dest="models", default="qwen3-4b")
    parser.add_argument("--langs", "--lang", dest="langs", default="all")
    parser.add_argument("--prompts", type=int, default=10)
    parser.add_argument("--layer", default="18")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--template-style", choices=("direct", "dataset"), default="direct")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--no-system", action="store_true")
    parser.add_argument("--classifier-dir", default=str(DEFAULT_CLASSIFIER_DIR))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--vectors-dir", default=str(DEFAULT_VECTORS_DIR))
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.prompts <= 0:
        parser.error("--prompts must be > 0")
    if args.batch_size < 0:
        parser.error("--batch-size must be >= 0")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be > 0")
    if args.no_system:
        args.system = None
    return args


def parse_langs(spec: str) -> list[str]:
    if spec is None or spec.strip().lower() in ("all", "*", ""):
        return list(LANGS)
    wanted = [part.strip() for part in spec.split(",") if part.strip()]
    unknown = [lang for lang in wanted if lang not in LANGS]
    if unknown:
        raise SystemExit(f"[pilot] unknown language(s): {', '.join(unknown)}")
    wanted_set = set(wanted)
    return [lang for lang in LANGS if lang in wanted_set]


def load_prompt_set(lang: str, count: int, style: str) -> dict[str, list[dict]]:
    topics_path = DATASET_ROOT / "prompts" / "topics.json"
    topics = json.loads(topics_path.read_text(encoding="utf-8"))[:count]
    if style == "direct":
        templates = DIRECT_TEMPLATES
    else:
        templates = json.loads(
            (DATASET_ROOT / "prompts" / "prompting_templates.json").read_text(
                encoding="utf-8"
            )
        )

    out: dict[str, list[dict]] = {}
    for kind in PROMPT_KINDS:
        out[kind] = [
            {
                "id": topic["id"],
                "topic": topic[lang],
                "prompt": templates[kind][lang].format(topic=topic[lang]),
            }
            for topic in topics
        ]
    return out


def random_vector(diff, alpha: float, seed: int):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    vector = torch.randn(diff.shape, generator=generator)
    vector = vector / vector.norm().clamp_min(1e-8) * diff.norm() * alpha
    return vector


def run_language(model, tokenizer, model_key, commit, lang, args, gpu_device) -> tuple:
    import torch

    prompt_sets = load_prompt_set(lang, args.prompts, args.template_style)
    diff, resid_norm, layer, meta, paths, report = load_vector(
        model_key, commit, lang, args.layer, args.vectors_dir
    )
    plus = (diff * args.alpha).to(device=gpu_device, dtype=torch.float16)
    minus = (-diff * args.alpha).to(device=gpu_device, dtype=torch.float16)
    rand = random_vector(diff, args.alpha, args.random_seed + LANGS.index(lang)).to(
        device=gpu_device, dtype=torch.float16
    )
    batch_size = args.batch_size or (4 if MODELS[model_key]["params"] >= 7_000_000_000 else 8)
    injections = {"none": None, "plus": plus, "minus": minus, "random": rand}
    vector_info = {
        "layer": layer,
        "pooling": meta["extraction"]["pooling"],
        "alpha": args.alpha,
        "vector_norm": float(diff.norm()),
        "resid_norm": resid_norm,
        "plus_norm": float(plus.norm()),
        "minus_norm": float(minus.norm()),
        "random_norm": float(rand.norm()),
        "batch_size": batch_size,
        "random_seed": args.random_seed + LANGS.index(lang),
        "vector_path": str(paths["base"]),
        "cache_key": meta.get("cache_key"),
    }

    print(
        f"[pilot] {model_key}/{lang}: layer={layer} batch={batch_size} "
        f"alpha={args.alpha} |v|={vector_info['vector_norm']:.3f}",
        flush=True,
    )

    samples = []
    conditions = [
        (kind, mode)
        for kind in PROMPT_KINDS
        for mode in VECTOR_MODES
    ] + [("neutral", "random")]

    for kind, mode in conditions:
        rows = prompt_sets[kind]
        injection = injections[mode]
        kind_index = PROMPT_KINDS.index(kind)
        started = time.perf_counter()
        for batch_index, start in enumerate(range(0, len(rows), batch_size)):
            chunk = rows[start : start + batch_size]
            # Same seed for none/plus/minus/random within a prompt kind.
            seed = (
                args.seed
                + 1000 * LANGS.index(lang)
                + 100 * kind_index
                + batch_index
            )
            batch = generate_batch(
                model,
                tokenizer,
                [row["prompt"] for row in chunk],
                lang,
                args,
                seed,
                layer,
                injection=injection,
            )
            for row, sample in zip(chunk, batch):
                sample.update(
                    {
                        "condition": f"{kind}_{mode}",
                        "prompt_kind": kind,
                        "vector_mode": mode,
                        "lang": lang,
                        "prompt_id": row["id"],
                        "prompt": row["prompt"],
                        "topic": row["topic"],
                    }
                )
                samples.append(sample)
        elapsed = time.perf_counter() - started
        print(
            f"[pilot] {lang} {kind:8s} {mode:6s} n={len(rows):2d} "
            f"{elapsed:5.1f}s",
            flush=True,
        )

    baseline = {
        (sample["prompt_id"], sample["prompt_kind"]): sample["text"]
        for sample in samples
        if sample["vector_mode"] == "none"
    }
    for sample in samples:
        if sample["vector_mode"] == "none":
            sample["changed"] = None
        else:
            sample["changed"] = (
                sample["text"] != baseline[(sample["prompt_id"], sample["prompt_kind"])]
            )
    return samples, vector_info, meta


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
    model, tokenizer = load_model(model_key, args, gpu_index)
    gpu_device = torch.device(f"cuda:{gpu_index}")
    samples, vector_infos, errors = [], [], []
    try:
        for lang in args.langs:
            try:
                lang_samples, vector_info, meta = run_language(
                    model, tokenizer, model_key, commit, lang, args, gpu_device
                )
                samples.extend(lang_samples)
                vector_infos.append({"lang": lang, **vector_info, "pooling": meta["extraction"]["pooling"]})
            except Exception as exc:
                message = f"{model_key}/{lang}: {exc}"
                print(f"[pilot] ERROR: {message}", file=sys.stderr)
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
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "vectors": vector_infos,
        "samples": samples,
        "errors": errors,
    }


def load_classifier(classifier_dir: str, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = Path(classifier_dir).expanduser().resolve()
    model_path = path / "model"
    tokenizer_path = path / "tokenizer"
    if not model_path.is_dir() or not tokenizer_path.is_dir():
        raise FileNotFoundError(
            f"classifier not found at {path}; run scripts/train_classifier.py first"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(device)
    model.eval()
    metrics = read_json(path / "metrics.json")
    return model, tokenizer, metrics, path


def score_texts(model, tokenizer, texts, device, batch_size=64, max_length=256):
    import torch

    probabilities, margins = [], []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(**encoded).logits.float()
        probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
        margins.extend((logits[:, 1] - logits[:, 0]).cpu().tolist())
    return probabilities, margins


def summarize(samples: list[dict]) -> tuple[list[dict], dict]:
    groups: dict[tuple, list[dict]] = {}
    for sample in samples:
        key = (sample["lang"], sample["prompt_kind"], sample["vector_mode"])
        groups.setdefault(key, []).append(sample)

    rows = []
    for (lang, kind, mode), items in sorted(groups.items()):
        nonempty = [item for item in items if not item["empty"]]
        row = {
            "lang": lang,
            "prompt_kind": kind,
            "vector_mode": mode,
            "n": len(items),
            "p_mean": sum(item["p_positive"] for item in items) / len(items),
            "margin_mean": sum(item["logit_margin"] for item in items) / len(items),
            "empty": sum(1 for item in items if item["empty"]),
            "script_mean": (
                sum(item["script_ratio"] for item in nonempty) / len(nonempty)
                if nonempty
                else 0.0
            ),
            "think": sum(1 for item in items if item["think_leak"]),
            "rep4_mean": sum(item["rep4"] for item in items) / len(items),
            "truncated": sum(1 for item in items if not item["hit_eos"]),
            "tokens_mean": sum(item["n_new_tokens"] for item in items) / len(items),
            "changed": sum(1 for item in items if item["changed"]),
        }
        rows.append(row)

    lookup = {(row["lang"], row["prompt_kind"], row["vector_mode"]): row for row in rows}
    for row in rows:
        if row["vector_mode"] == "none":
            row["delta_p"] = 0.0
            row["delta_margin"] = 0.0
        else:
            base = lookup[(row["lang"], row["prompt_kind"], "none")]
            row["delta_p"] = row["p_mean"] - base["p_mean"]
            row["delta_margin"] = row["margin_mean"] - base["margin_mean"]
    return rows, lookup


def evaluate_tests(lookup: dict) -> dict:
    tests = {}

    prompt_gaps = {}
    for lang in LANGS:
        try:
            pos = lookup[(lang, "positive", "none")]["p_mean"]
            neu = lookup[(lang, "neutral", "none")]["p_mean"]
            neg = lookup[(lang, "negative", "none")]["p_mean"]
        except KeyError:
            continue
        prompt_gaps[lang] = {
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "positive_minus_neutral": pos - neu,
            "neutral_minus_negative": neu - neg,
            "ordered": pos > neu > neg,
        }

    headroom = {}
    for lang in LANGS:
        keys = [
            (lang, "negative", "none"),
            (lang, "negative", "plus"),
            (lang, "positive", "none"),
            (lang, "positive", "minus"),
            (lang, "neutral", "none"),
            (lang, "neutral", "plus"),
            (lang, "neutral", "minus"),
            (lang, "neutral", "random"),
        ]
        if any(key not in lookup for key in keys):
            continue
        neg0 = lookup[(lang, "negative", "none")]["p_mean"]
        neg_plus = lookup[(lang, "negative", "plus")]["p_mean"]
        pos0 = lookup[(lang, "positive", "none")]["p_mean"]
        pos_minus = lookup[(lang, "positive", "minus")]["p_mean"]
        neu0 = lookup[(lang, "neutral", "none")]["p_mean"]
        neu_plus = lookup[(lang, "neutral", "plus")]["p_mean"]
        neu_minus = lookup[(lang, "neutral", "minus")]["p_mean"]
        neu_rand = lookup[(lang, "neutral", "random")]["p_mean"]
        headroom[lang] = {
            "negative_plus_minus_none": neg_plus - neg0,
            "positive_minus_minus_none": pos_minus - pos0,
            "neutral_plus_minus_minus": neu_plus - neu_minus,
            "random_minus_neutral": neu_rand - neu0,
            "positive_headroom_ok": neg_plus > neg0,
            "negative_headroom_ok": pos_minus < pos0,
            "neutral_direction_ok": neu_plus > neu_minus,
            "random_ok": abs(neu_rand - neu0) <= 0.10,
        }

    tests["prompt_control"] = prompt_gaps
    tests["headroom"] = headroom
    tests["passed"] = bool(prompt_gaps) and all(
        item["ordered"] for item in prompt_gaps.values()
    ) and all(
        item["positive_headroom_ok"]
        and item["negative_headroom_ok"]
        and item["neutral_direction_ok"]
        and item["random_ok"]
        for item in headroom.values()
    )
    return tests


def main() -> int:
    args = parse_args()
    model_keys = select_models(args.models)
    args.langs = parse_langs(args.langs)

    try:
        import torch
    except ImportError:
        print("[pilot] ERROR: torch is not installed.", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("[pilot] ERROR: CUDA is required.", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    if device.type != "cuda":
        print(f"[pilot] ERROR: --device must be CUDA, got {args.device}", file=sys.stderr)
        return 2
    gpu_index = device.index if device.index is not None else torch.cuda.current_device()

    print(
        f"[pilot] GPU: {torch.cuda.get_device_name(gpu_index)} | "
        f"models: {', '.join(model_keys)} | languages: {', '.join(args.langs)} | "
        f"prompts/lang: {args.prompts} | layer: {args.layer} | alpha: {args.alpha}"
    )

    runs = []
    errors = []
    for model_key in model_keys:
        try:
            run = run_model(model_key, args, gpu_index)
        except Exception as exc:
            import traceback

            print(f"[pilot] ERROR: {model_key}: {exc}", file=sys.stderr)
            traceback.print_exc()
            errors.append(f"{model_key}: {exc}")
            continue
        runs.append(run)
        errors.extend(run["errors"])

    samples = [sample for run in runs for sample in run["samples"]]
    if not samples:
        print("[pilot] no generations produced", file=sys.stderr)
        return 1

    classifier_model, classifier_tokenizer, classifier_metrics, classifier_path = load_classifier(
        args.classifier_dir, device
    )
    print(f"[pilot] classifier: {classifier_path}", flush=True)
    probabilities, margins = score_texts(
        classifier_model, classifier_tokenizer, [sample["text"] for sample in samples], device
    )
    for sample, probability, margin in zip(samples, probabilities, margins):
        sample["p_positive"] = float(probability)
        sample["logit_margin"] = float(margin)
    del classifier_model
    del classifier_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    summary_rows, lookup = summarize(samples)
    tests = evaluate_tests(lookup)
    mechanical_ok = all(
        row["empty"] == 0 and row["script_mean"] >= 0.5 and row["think"] == 0
        for row in summary_rows
    )
    changed_ok = all(
        row["changed"] > 0
        for row in summary_rows
        if row["vector_mode"] != "none"
    )
    tests["mechanical_ok"] = mechanical_ok
    tests["changed_ok"] = changed_ok
    tests["passed"] = tests["passed"] and mechanical_ok and changed_ok
    passed = tests["passed"] and not errors

    print("\n[pilot] condition summary")
    header = (
        f"{'lang':4s} {'prompt':8s} {'vec':6s} {'n':>3s} {'p_pos':>6s} "
        f"{'margin':>7s} {'d_p':>7s} {'d_m':>7s} {'script':>6s} {'empty':>5s}"
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['lang']:4s} {row['prompt_kind']:8s} {row['vector_mode']:6s} "
            f"{row['n']:3d} {row['p_mean']:6.3f} {row['margin_mean']:7.3f} "
            f"{row['delta_p']:+7.3f} {row['delta_margin']:+7.3f} "
            f"{row['script_mean']:6.2f} {row['empty']:5d}"
        )

    print("\n[pilot] prompt control")
    for lang, values in tests["prompt_control"].items():
        print(
            f"[pilot] {lang}: positive={values['positive']:.3f} "
            f"neutral={values['neutral']:.3f} negative={values['negative']:.3f} "
            f"gaps={values['positive_minus_neutral']:+.3f}/"
            f"{values['neutral_minus_negative']:+.3f} "
            f"ordered={values['ordered']}"
        )

    print("\n[pilot] headroom / controls")
    for lang, values in tests["headroom"].items():
        print(
            f"[pilot] {lang}: neg +v {values['negative_plus_minus_none']:+.3f} "
            f"pos -v {values['positive_minus_minus_none']:+.3f} "
            f"neutral +v/-v {values['neutral_plus_minus_minus']:+.3f} "
            f"random {values['random_minus_neutral']:+.3f} "
            f"ok={values['positive_headroom_ok']}/{values['negative_headroom_ok']}/"
            f"{values['neutral_direction_ok']}/{values['random_ok']}"
        )

    print(
        f"\n[pilot] mechanical_ok={tests['mechanical_ok']} "
        f"changed_ok={tests['changed_ok']} all_ok={tests['passed']}"
    )
    print()
    if passed:
        print(f"[pilot] PASS: {len(samples)} generations; all directional/control checks hold")
    else:
        print("[pilot] FAIL: one or more directional/control checks failed", file=sys.stderr)

    if errors:
        print(f"[pilot] errors: {errors}", file=sys.stderr)

    if not args.no_save:
        out = (
            Path(args.out).expanduser()
            if args.out
            else REPO_ROOT / "artifacts" / "pilot" / f"final_3x3_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.json"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_utc": utc_now(),
            "settings": {
                "models": model_keys,
                "langs": args.langs,
                "prompts": args.prompts,
                "layer": args.layer,
                "alpha": args.alpha,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "seed": args.seed,
                "random_seed": args.random_seed,
                "template_style": args.template_style,
                "system": args.system,
                "classifier_dir": str(classifier_path),
            },
            "classifier_metrics": classifier_metrics,
            "runs": runs,
            "summary": summary_rows,
            "tests": tests,
            "errors": errors,
            "passed": passed,
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[pilot] report: {out} ({human_bytes(out.stat().st_size)})")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
