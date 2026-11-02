#!/usr/bin/env python3
"""Revised Phase 1 pilot: mid-scale prompts + random-vector null.

This is the follow-up to the strict 3x3 pilot. It uses graded prompts that
should produce sentiment baselines away from 0/1:

  balanced       - both positives and negatives
  mild_positive  - mostly positive, one minor complaint
  mild_negative  - mostly negative, one minor compliment

For each prompt type it runs no vector, +alpha and -alpha. On the balanced
prompt it also runs several norm-matched random vectors to build a null
distribution. Scoring uses the trained XLM-R classifier and the primary
outcome is the logit margin, not the saturated probability.

Example:
  python3 scripts/revised_pilot.py --only qwen3-4b --prompts 10 --layer 18
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
from pilot_generation import load_classifier, score_texts  # noqa: E402
from test_generation import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    generate_batch,
    load_model,
    load_vector,
)

DEFAULT_CLASSIFIER_DIR = REPO_ROOT / "artifacts" / "classifier" / "xlmr_phase1"

PROMPT_KINDS = ("balanced", "mild_positive", "mild_negative")
VECTOR_MODES = ("none", "plus", "minus")

# Draft graded templates. Native verification is still required before the
# final dataset is frozen; the Hindi t01 topic is the obvious grammar fix.
GRADED_TEMPLATES = {
    "balanced": {
        "en": "Write a balanced review of {topic}, including both positives and negatives.",
        "hi": "इस विषय पर एक संतुलित समीक्षा लिखिए, जिसमें अच्छी और बुरी दोनों बातें हों: {topic}।",
        "bn": "এই বিষয়ে একটি ভারসাম্যপূর্ণ পর্যালোচনা লিখুন, যেখানে ভালো এবং খারাপ উভয় দিক থাকবে: {topic}।",
        "te": "ఈ అంశంపై సమతుల్యమైన సమీక్ష రాయండి; మంచి మరియు చెడు రెండూ ఉండాలి: {topic}.",
        "ta": "இந்தத் தலைப்பில் சமநிலையான விமர்சனம் எழுதுங்கள்; நல்லது மற்றும் கெட்டது இரண்டும் இருக்க வேண்டும்: {topic}.",
    },
    "mild_positive": {
        "en": "Write a mostly positive review of {topic}, with one minor complaint.",
        "hi": "इस विषय पर एक मुख्यतः सकारात्मक समीक्षा लिखिए, जिसमें एक छोटी शिकायत भी हो: {topic}।",
        "bn": "এই বিষয়ে একটি মূলত ইতিবাচক পর্যালোচনা লিখুন, যেখানে একটি ছোট অভিযোগ থাকবে: {topic}।",
        "te": "ఈ అంశంపై ప్రధానంగా సానుకూల సమీక్ష రాయండి; ఒక చిన్న ఫిర్యాదు కూడా ఉండాలి: {topic}.",
        "ta": "இந்தத் தலைப்பில் பெரும்பாலும் நேர்மறையான விமர்சனம் எழுதுங்கள்; ஒரு சிறிய குறையும் இருக்க வேண்டும்: {topic}.",
    },
    "mild_negative": {
        "en": "Write a mostly negative review of {topic}, with one minor compliment.",
        "hi": "इस विषय पर एक मुख्यतः नकारात्मक समीक्षा लिखिए, जिसमें एक छोटी तारीफ भी हो: {topic}।",
        "bn": "এই বিষয়ে একটি মূলত নেতিবাচক পর্যালোচনা লিখুন, যেখানে একটি ছোট প্রশংসা থাকবে: {topic}।",
        "te": "ఈ అంశంపై ప్రధానంగా ప్రతికూల సమీక్ష రాయండి; ఒక చిన్న ప్రశంస కూడా ఉండాలి: {topic}.",
        "ta": "இந்தத் தலைப்பில் பெரும்பாலும் எதிர்மறையான விமர்சனம் எழுதுங்கள்; ஒரு சிறிய பாராட்டும் இருக்க வேண்டும்: {topic}.",
    },
}

HINDI_TOPIC_FIX = {
    "t01": "हाल में आपने जिस रेस्तरां में खाना खाया",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Revised Phase 1 pilot with mid-scale prompts and random-vector null.",
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
    parser.add_argument("--random-seeds", type=int, default=5)
    parser.add_argument("--random-seed-offset", type=int, default=1234)
    parser.add_argument("--classifier-dir", default=str(DEFAULT_CLASSIFIER_DIR))
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--vectors-dir", default=str(DEFAULT_VECTORS_DIR))
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--no-system", action="store_true")
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
    if args.random_seeds <= 0:
        parser.error("--random-seeds must be > 0")
    if args.no_system:
        args.system = None
    return args


def parse_langs(spec: str) -> list[str]:
    if spec is None or spec.strip().lower() in ("all", "*", ""):
        return list(LANGS)
    wanted = [part.strip() for part in spec.split(",") if part.strip()]
    unknown = [lang for lang in wanted if lang not in LANGS]
    if unknown:
        raise SystemExit(f"[revised] unknown language(s): {', '.join(unknown)}")
    wanted_set = set(wanted)
    return [lang for lang in LANGS if lang in wanted_set]


def load_prompt_set(lang: str, count: int) -> dict[str, list[dict]]:
    topics = json.loads(
        (DATASET_ROOT / "prompts" / "topics.json").read_text(encoding="utf-8")
    )[:count]
    if lang == "hi":
        for topic in topics:
            if topic["id"] in HINDI_TOPIC_FIX:
                topic["hi"] = HINDI_TOPIC_FIX[topic["id"]]

    return {
        kind: [
            {
                "id": topic["id"],
                "topic": topic[lang],
                "prompt": GRADED_TEMPLATES[kind][lang].format(topic=topic[lang]),
            }
            for topic in topics
        ]
        for kind in PROMPT_KINDS
    }


def random_vector(diff, alpha: float, seed: int):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    vector = torch.randn(diff.shape, generator=generator)
    return vector / vector.norm().clamp_min(1e-8) * diff.norm() * alpha


def run_language(model, tokenizer, model_key, commit, lang, args, gpu_device) -> tuple:
    import torch

    prompt_sets = load_prompt_set(lang, args.prompts)
    diff, resid_norm, layer, meta, paths, report = load_vector(
        model_key, commit, lang, args.layer, args.vectors_dir
    )
    plus = (diff * args.alpha).to(device=gpu_device, dtype=torch.float16)
    minus = (-diff * args.alpha).to(device=gpu_device, dtype=torch.float16)
    batch_size = args.batch_size or (4 if MODELS[model_key]["params"] >= 7_000_000_000 else 8)

    random_vectors = {
        seed: random_vector(
            diff,
            args.alpha,
            args.random_seed_offset + 1000 * LANGS.index(lang) + seed,
        ).to(device=gpu_device, dtype=torch.float16)
        for seed in range(args.random_seeds)
    }
    injections = {"none": None, "plus": plus, "minus": minus}

    vector_info = {
        "layer": layer,
        "pooling": meta["extraction"]["pooling"],
        "alpha": args.alpha,
        "vector_norm": float(diff.norm()),
        "resid_norm": resid_norm,
        "plus_norm": float(plus.norm()),
        "minus_norm": float(minus.norm()),
        "random_norms": {str(seed): float(value.norm()) for seed, value in random_vectors.items()},
        "batch_size": batch_size,
        "random_seeds": args.random_seeds,
        "vector_path": str(paths["base"]),
        "cache_key": meta.get("cache_key"),
    }

    print(
        f"[revised] {model_key}/{lang}: layer={layer} batch={batch_size} "
        f"alpha={args.alpha} |v|={vector_info['vector_norm']:.3f}",
        flush=True,
    )

    samples = []
    for kind in PROMPT_KINDS:
        rows = prompt_sets[kind]
        kind_index = PROMPT_KINDS.index(kind)
        for mode in VECTOR_MODES:
            injection = injections[mode]
            started = time.perf_counter()
            for batch_index, start in enumerate(range(0, len(rows), batch_size)):
                chunk = rows[start : start + batch_size]
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
                            "random_seed": None,
                            "lang": lang,
                            "prompt_id": row["id"],
                            "prompt": row["prompt"],
                            "topic": row["topic"],
                        }
                    )
                    samples.append(sample)
            print(
                f"[revised] {lang} {kind:13s} {mode:6s} n={len(rows):2d} "
                f"{time.perf_counter() - started:5.1f}s",
                flush=True,
            )

        if kind == "balanced":
            for random_seed, injection in random_vectors.items():
                started = time.perf_counter()
                for batch_index, start in enumerate(range(0, len(rows), batch_size)):
                    chunk = rows[start : start + batch_size]
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
                                "condition": f"balanced_random_{random_seed}",
                                "prompt_kind": "balanced",
                                "vector_mode": "random",
                                "random_seed": random_seed,
                                "lang": lang,
                                "prompt_id": row["id"],
                                "prompt": row["prompt"],
                                "topic": row["topic"],
                            }
                        )
                        samples.append(sample)
                print(
                    f"[revised] {lang} balanced      random{random_seed} "
                    f"n={len(rows):2d} {time.perf_counter() - started:5.1f}s",
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
                vector_infos.append(
                    {"lang": lang, **vector_info, "pooling": meta["extraction"]["pooling"]}
                )
            except Exception as exc:
                message = f"{model_key}/{lang}: {exc}"
                print(f"[revised] ERROR: {message}", file=sys.stderr)
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


def summarize(samples: list[dict]) -> tuple[list[dict], dict]:
    groups: dict[tuple, list[dict]] = {}
    for sample in samples:
        if sample["vector_mode"] == "random":
            key = (sample["lang"], "balanced", f"random_{sample['random_seed']}")
        else:
            key = (sample["lang"], sample["prompt_kind"], sample["vector_mode"])
        groups.setdefault(key, []).append(sample)

    rows = []
    for (lang, kind, mode), items in sorted(groups.items()):
        nonempty = [item for item in items if not item["empty"]]
        random_seed = items[0]["random_seed"] if items[0]["vector_mode"] == "random" else None
        rows.append(
            {
                "lang": lang,
                "prompt_kind": kind,
                "vector_mode": "random" if mode.startswith("random_") else mode,
                "random_seed": random_seed,
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
        )

    # Paired deltas by prompt id against the matching no-vector baseline.
    margin_deltas: dict[tuple, float] = {}
    p_deltas: dict[tuple, float] = {}
    for lang in LANGS:
        for kind in PROMPT_KINDS:
            base_margin = {
                item["prompt_id"]: item["logit_margin"]
                for item in samples
                if item["lang"] == lang
                and item["prompt_kind"] == kind
                and item["vector_mode"] == "none"
            }
            base_p = {
                item["prompt_id"]: item["p_positive"]
                for item in samples
                if item["lang"] == lang
                and item["prompt_kind"] == kind
                and item["vector_mode"] == "none"
            }
            for mode in ("plus", "minus"):
                items = [
                    item
                    for item in samples
                    if item["lang"] == lang
                    and item["prompt_kind"] == kind
                    and item["vector_mode"] == mode
                ]
                if base_margin and items:
                    margin_deltas[(lang, kind, mode, None)] = sum(
                        item["logit_margin"] - base_margin[item["prompt_id"]]
                        for item in items
                    ) / len(items)
                    p_deltas[(lang, kind, mode, None)] = sum(
                        item["p_positive"] - base_p[item["prompt_id"]]
                        for item in items
                    ) / len(items)

        base_margin = {
            item["prompt_id"]: item["logit_margin"]
            for item in samples
            if item["lang"] == lang
            and item["prompt_kind"] == "balanced"
            and item["vector_mode"] == "none"
        }
        base_p = {
            item["prompt_id"]: item["p_positive"]
            for item in samples
            if item["lang"] == lang
            and item["prompt_kind"] == "balanced"
            and item["vector_mode"] == "none"
        }
        for seed in sorted(
            {
                item["random_seed"]
                for item in samples
                if item["lang"] == lang and item["vector_mode"] == "random"
            }
        ):
            items = [
                item
                for item in samples
                if item["lang"] == lang
                and item["vector_mode"] == "random"
                and item["random_seed"] == seed
            ]
            if base_margin and items:
                margin_deltas[(lang, "balanced", "random", seed)] = sum(
                    item["logit_margin"] - base_margin[item["prompt_id"]]
                    for item in items
                ) / len(items)
                p_deltas[(lang, "balanced", "random", seed)] = sum(
                    item["p_positive"] - base_p[item["prompt_id"]]
                    for item in items
                ) / len(items)

    for row in rows:
        if row["vector_mode"] == "none":
            row["delta_margin"] = 0.0
            row["delta_p"] = 0.0
        elif row["vector_mode"] == "random":
            key = (row["lang"], "balanced", "random", row["random_seed"])
            row["delta_margin"] = margin_deltas.get(key, 0.0)
            row["delta_p"] = p_deltas.get(key, 0.0)
        else:
            key = (row["lang"], row["prompt_kind"], row["vector_mode"], None)
            row["delta_margin"] = margin_deltas.get(key, 0.0)
            row["delta_p"] = p_deltas.get(key, 0.0)
    return rows, margin_deltas


def evaluate_tests(deltas: dict, rows: list[dict]) -> dict:
    prompt_control = {}
    headroom = {}
    for lang in LANGS:
        try:
            pos = next(
                row["margin_mean"]
                for row in rows
                if row["lang"] == lang
                and row["prompt_kind"] == "mild_positive"
                and row["vector_mode"] == "none"
            )
            neu = next(
                row["margin_mean"]
                for row in rows
                if row["lang"] == lang
                and row["prompt_kind"] == "balanced"
                and row["vector_mode"] == "none"
            )
            neg = next(
                row["margin_mean"]
                for row in rows
                if row["lang"] == lang
                and row["prompt_kind"] == "mild_negative"
                and row["vector_mode"] == "none"
            )
            plus_delta = deltas[(lang, "balanced", "plus", None)]
            minus_delta = deltas[(lang, "balanced", "minus", None)]
            random_values = [
                value
                for (key_lang, kind, mode, seed), value in deltas.items()
                if key_lang == lang and kind == "balanced" and mode == "random"
            ]
            mild_pos_minus = deltas[(lang, "mild_positive", "minus", None)]
            mild_neg_plus = deltas[(lang, "mild_negative", "plus", None)]
        except (KeyError, StopIteration):
            continue

        random_mean = sum(random_values) / len(random_values)
        random_var = sum((value - random_mean) ** 2 for value in random_values) / len(random_values)
        prompt_control[lang] = {
            "mild_positive": pos,
            "balanced": neu,
            "mild_negative": neg,
            "positive_minus_balanced": pos - neu,
            "balanced_minus_negative": neu - neg,
            "ordered": pos > neu > neg,
        }
        headroom[lang] = {
            "plus_delta": plus_delta,
            "minus_delta": minus_delta,
            "random_values": random_values,
            "random_mean": random_mean,
            "random_std": random_var**0.5,
            "random_min": min(random_values),
            "random_max": max(random_values),
            "plus_above_random_max": plus_delta > max(random_values),
            "minus_below_random_min": minus_delta < min(random_values),
            "mild_positive_minus": mild_pos_minus,
            "mild_negative_plus": mild_neg_plus,
            "mild_positive_minus_ok": mild_pos_minus < 0,
            "mild_negative_plus_ok": mild_neg_plus > 0,
        }

    return {
        "prompt_control": prompt_control,
        "headroom": headroom,
        "passed": bool(prompt_control)
        and all(item["ordered"] for item in prompt_control.values())
        and all(
            item["plus_above_random_max"]
            and item["minus_below_random_min"]
            and item["mild_positive_minus_ok"]
            and item["mild_negative_plus_ok"]
            for item in headroom.values()
        ),
    }


def main() -> int:
    args = parse_args()
    model_keys = select_models(args.models)
    args.langs = parse_langs(args.langs)

    try:
        import torch
    except ImportError:
        print("[revised] ERROR: torch is not installed.", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("[revised] ERROR: CUDA is required.", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    if device.type != "cuda":
        print(f"[revised] ERROR: --device must be CUDA, got {args.device}", file=sys.stderr)
        return 2
    gpu_index = device.index if device.index is not None else torch.cuda.current_device()

    print(
        f"[revised] GPU: {torch.cuda.get_device_name(gpu_index)} | "
        f"models: {', '.join(model_keys)} | languages: {', '.join(args.langs)} | "
        f"prompts/lang: {args.prompts} | layer: {args.layer} | alpha: {args.alpha} | "
        f"random seeds: {args.random_seeds}"
    )

    runs = []
    errors = []
    for model_key in model_keys:
        try:
            run = run_model(model_key, args, gpu_index)
        except Exception as exc:
            import traceback

            print(f"[revised] ERROR: {model_key}: {exc}", file=sys.stderr)
            traceback.print_exc()
            errors.append(f"{model_key}: {exc}")
            continue
        runs.append(run)
        errors.extend(run["errors"])

    samples = [sample for run in runs for sample in run["samples"]]
    if not samples:
        print("[revised] no generations produced", file=sys.stderr)
        return 1

    classifier_model, classifier_tokenizer, classifier_metrics, classifier_path = load_classifier(
        args.classifier_dir, device
    )
    print(f"[revised] classifier: {classifier_path}", flush=True)
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

    rows, deltas = summarize(samples)
    tests = evaluate_tests(deltas, rows)
    mechanical_ok = all(
        row["empty"] == 0 and row["script_mean"] >= 0.5 and row["think"] == 0
        for row in rows
    )
    changed_ok = all(
        row["changed"] > 0 for row in rows if row["vector_mode"] != "none"
    )
    tests["mechanical_ok"] = mechanical_ok
    tests["changed_ok"] = changed_ok
    tests["passed"] = tests["passed"] and mechanical_ok and changed_ok
    passed = tests["passed"] and not errors

    print("\n[revised] condition summary (margin is primary)")
    header = (
        f"{'lang':4s} {'prompt':13s} {'vec':10s} {'n':>3s} {'p_pos':>6s} "
        f"{'margin':>7s} {'d_m':>7s} {'d_p':>7s} {'script':>6s} {'empty':>5s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        label = (
            f"rand{row['random_seed']}"
            if row["vector_mode"] == "random"
            else row["vector_mode"]
        )
        print(
            f"{row['lang']:4s} {row['prompt_kind']:13s} {label:10s} "
            f"{row['n']:3d} {row['p_mean']:6.3f} {row['margin_mean']:7.3f} "
            f"{row['delta_margin']:+7.3f} {row['delta_p']:+7.3f} "
            f"{row['script_mean']:6.2f} {row['empty']:5d}"
        )

    print("\n[revised] prompt control")
    for lang, values in tests["prompt_control"].items():
        print(
            f"[revised] {lang}: mild_pos={values['mild_positive']:+.3f} "
            f"balanced={values['balanced']:+.3f} mild_neg={values['mild_negative']:+.3f} "
            f"gaps={values['positive_minus_balanced']:+.3f}/"
            f"{values['balanced_minus_negative']:+.3f} ordered={values['ordered']}"
        )

    print("\n[revised] random-null comparison")
    for lang, values in tests["headroom"].items():
        print(
            f"[revised] {lang}: +v={values['plus_delta']:+.3f} "
            f"-v={values['minus_delta']:+.3f} "
            f"random mean={values['random_mean']:+.3f} "
            f"[{values['random_min']:+.3f},{values['random_max']:+.3f}] "
            f"+ok={values['plus_above_random_max']} "
            f"-ok={values['minus_below_random_min']} "
            f"pos-minus={values['mild_positive_minus']:+.3f} "
            f"neg-plus={values['mild_negative_plus']:+.3f}"
        )

    print(
        f"\n[revised] mechanical_ok={mechanical_ok} changed_ok={changed_ok} "
        f"all_ok={passed}"
    )
    if passed:
        print(f"[revised] PASS: {len(samples)} generations; vector beats random null")
    else:
        print("[revised] FAIL: one or more prompt/headroom/random checks failed", file=sys.stderr)

    if errors:
        print(f"[revised] errors: {errors}", file=sys.stderr)

    if not args.no_save:
        out = (
            Path(args.out).expanduser()
            if args.out
            else REPO_ROOT / "artifacts" / "pilot" / f"revised_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.json"
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
                "random_seeds": args.random_seeds,
                "random_seed_offset": args.random_seed_offset,
                "system": args.system,
                "classifier_dir": str(classifier_path),
            },
            "classifier_metrics": classifier_metrics,
            "runs": runs,
            "summary": rows,
            "deltas": {str(key): value for key, value in deltas.items()},
            "tests": tests,
            "errors": errors,
            "passed": passed,
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[revised] report: {out} ({human_bytes(out.stat().st_size)})")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
