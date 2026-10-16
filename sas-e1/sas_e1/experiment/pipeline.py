"""Experiment pipeline — resumable stages: extract → fit → eval → probes → analyze.

Stages needing the base model: `extract` (activation harvesting), `eval` (generation).
`fit` and `probes` operate purely on cached activations (no base-model backprop).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..config import get
from ..data import corpora
from ..data.prompts import few_shot_prefix, neutral_prompts
from ..evaluation import metrics as metric_utils
from ..evaluation.classifiers import EmbeddingLRClassifier, TfidfStyleClassifier, evaluate_classifier
from ..models import (
    block_list,
    extract_doc_vectors,
    identity_check,
    load_model_and_tokenizer,
    resolve_layer,
)
from ..probes import separation as separation_probe
from ..probes import spectra as spectra_probe
from ..probes import stability as stability_probe
from ..steering import calibration, fitting
from ..steering.operators import Steerer, no_op_steerer
from ..utils import (
    ensure_dir,
    is_done,
    load_json,
    mark_done,
    read_jsonl,
    save_json,
    set_seed,
    write_jsonl,
)

STAGES = ["extract", "fit", "eval", "probes", "analyze"]


# ---------------------------------------------------------------- paths

def run_root(cfg) -> Path:
    return Path(get(cfg, "paths.artifacts", "artifacts")) / str(cfg.run_name)


def attr_dir(cfg, attr: str, stage: Optional[str] = None) -> Path:
    p = run_root(cfg) / attr
    return p / stage if stage else p


def activations_path(cfg, attr: str) -> Path:
    return Path(get(cfg, "paths.activations", "data/activations")) / str(cfg.run_name) / f"{attr}.npz"


def _attr_keys(cfg) -> list[str]:
    return [k for k in vars(cfg.attributes).keys()]


def _layer(cfg, model) -> int:
    return resolve_layer(model, get(cfg, "model.layer", None), float(get(cfg, "model.layer_fraction", 0.5)))


def _gen_cfg(cfg) -> dict:
    return {
        "max_new_tokens": int(get(cfg, "generation.max_new_tokens", 128)),
        "temperature": float(get(cfg, "generation.temperature", 1.0)),
        "top_p": float(get(cfg, "generation.top_p", 0.95)),
        "batch_size": int(get(cfg, "generation.batch_size", 8)),
    }


# ---------------------------------------------------------------- stage: extract

def extract_attr(cfg, attr: str, model, tokenizer, device, force: bool = False) -> None:
    ad = ensure_dir(attr_dir(cfg, attr, "extract"))
    if is_done(ad) and not force:
        print(f"[extract:{attr}] cached")
        return
    set_seed(int(get(cfg, "seed", 0)))
    layer = _layer(cfg, model)
    save_json(ad / "layer.json", {"layer": layer, "n_blocks": len(block_list(model))})

    print(f"[extract:{attr}] splitting + matching corpora")
    splits = corpora.attribute_corpora(cfg, attr, tokenizer)
    max_len = int(get(cfg, "steering.max_length", 256))
    batch = int(get(cfg, "extraction.batch_size", 8))

    def _extract_and_filter(docs, label):
        texts = [d["text"] for d in docs]
        print(f"[extract:{attr}] activations {label}: {len(texts)} docs")
        X = extract_doc_vectors(
            model, tokenizer, texts, layer, batch_size=batch, max_length=max_len, device=device
        )
        keep = np.isfinite(X).all(axis=1)
        n_drop = int((~keep).sum())
        if n_drop:
            print(f"[extract:{attr}] {label}: dropped {n_drop}/{len(docs)} non-finite vectors")
            if n_drop > 0.2 * max(len(docs), 1):
                print(
                    f"[extract:{attr}] {label}: large non-finite rate — switch "
                    "model.dtype to 'auto' or 'float32' (BLOOM fp16 overflows on some GPUs)"
                )
        kept = [d for d, k in zip(docs, keep) if bool(k)]
        return X[keep], kept

    arrays: dict[str, np.ndarray] = {}
    arrays["X_stat_pos"], splits["stat_pos"] = _extract_and_filter(splits["stat_pos"], "stat_pos")
    arrays["X_stat_neg"], splits["stat_neg"] = _extract_and_filter(splits["stat_neg"], "stat_neg")

    neutral_docs = corpora.attach_token_counts(
        corpora.load_pool(cfg, "neutral_wikipedia"), tokenizer, max_len,
    )[: int(get(cfg, "probes.n_neutral", 1000))]
    arrays["X_neutral"], neutral_kept = _extract_and_filter(neutral_docs, "neutral")

    # keep fit as a subset of the docs that survived filtering
    n_fit = int(get(cfg, "steering.n_fit_docs", 400))
    ids_fit_pos = {id(d) for d in splits["fit_pos"]}
    ids_fit_neg = {id(d) for d in splits["fit_neg"]}
    splits["fit_pos"] = [d for d in splits["stat_pos"] if id(d) in ids_fit_pos][:n_fit] or splits["stat_pos"][:n_fit]
    splits["fit_neg"] = [d for d in splits["stat_neg"] if id(d) in ids_fit_neg][:n_fit] or splits["stat_neg"][:n_fit]
    fit_ids_pos = {id(d) for d in splits["fit_pos"]}
    fit_ids_neg = {id(d) for d in splits["fit_neg"]}
    arrays["fit_idx_pos"] = np.array(
        [i for i, d in enumerate(splits["stat_pos"]) if id(d) in fit_ids_pos], dtype=np.int64
    )
    arrays["fit_idx_neg"] = np.array(
        [i for i, d in enumerate(splits["stat_neg"]) if id(d) in fit_ids_neg], dtype=np.int64
    )

    for key in (
        "fit_pos", "fit_neg", "stat_pos", "stat_neg",
        "clf_pos_train", "clf_neg_train", "clf_pos_eval", "clf_neg_eval",
    ):
        write_jsonl(
            ad / f"{key}.jsonl",
            [{"text": d["text"], "source": d.get("source", ""), "n_tokens": d.get("n_tokens", 0)}
             for d in splits.get(key, [])],
        )
    write_jsonl(ad / "neutral.jsonl", [{"text": d["text"]} for d in neutral_kept])
    diag = dict(splits.get("diagnostics", {}))
    diag["kept"] = {
        "stat_pos": len(splits["stat_pos"]),
        "stat_neg": len(splits["stat_neg"]),
        "neutral": len(neutral_kept),
    }
    save_json(ad / "diagnostics.json", diag)

    ap = activations_path(cfg, attr)
    ensure_dir(ap.parent)
    np.savez_compressed(ap, **arrays)

    diff = identity_check(model, tokenizer, layer, neutral_prompts(2, seed=0), device=device)
    save_json(ad / "identity_check.json", {"max_logit_diff": diff, "layer": layer})
    mark_done(ad, {"n_stat_pos": len(splits["stat_pos"]), "n_stat_neg": len(splits["stat_neg"]), "layer": layer})


# ---------------------------------------------------------------- stage: fit

def fit_attr(cfg, attr: str, force: bool = False) -> None:
    ad = ensure_dir(attr_dir(cfg, attr, "fit"))
    if is_done(ad) and not force:
        print(f"[fit:{attr}] cached")
        return
    data = np.load(activations_path(cfg, attr), allow_pickle=False)
    Xp, Xn, Xneutral = data["X_stat_pos"], data["X_stat_neg"], data["X_neutral"]
    layer_meta = load_json(attr_dir(cfg, attr, "extract") / "layer.json", {})
    layer = int(layer_meta.get("layer", 0))

    ridge = float(get(cfg, "steering.ridge", 1e-3))
    sv_clip = float(get(cfg, "steering.sv_clip", 10.0))
    ranks = list(get(cfg, "steering.ranks", [1, 4, 16]) or [])
    include_full = bool(get(cfg, "steering.include_full_rank", True))
    resamples = list(get(cfg, "steering.resamples", [0, 1, 2]) or [0])
    d = Xp.shape[1]

    params: dict[str, dict] = {}
    for r in resamples:
        rng = np.random.default_rng(1000 + r)
        idx_p = rng.integers(0, len(Xp), len(Xp))
        idx_n = rng.integers(0, len(Xn), len(Xn))
        mu_p, cov_p = fitting.moments(Xp[idx_p])
        mu_n, cov_n = fitting.moments(Xn[idx_n])

        A = fitting.fit_vector(mu_p, mu_n)
        delta = calibration.reference_delta(A)
        ops = fitting.fit_moment_operators(mu_p, cov_p, mu_n, cov_n, ridge=ridge, sv_clip=sv_clip, ranks=ranks, include_full=include_full)
        R = fitting.fit_random(d, delta, seed=2000 + r)
        family = {"A": A, "R": R, **ops}
        out_dir = ensure_dir(ad / "steerers" / f"resample{r}")
        for name, steerer in family.items():
            steerer.alpha_unit = calibration.calibrate_alpha(steerer, Xneutral, delta)
            steerer.save(out_dir / f"{name}.npz")
            params[f"r{r}/{name}"] = spectra_probe.parameter_spectrum(steerer)

    trained = getattr(cfg.steering, "trained", None)
    if trained is not None:
        fit_idx_p = data["fit_idx_pos"].astype(int)
        fit_idx_n = data["fit_idx_neg"].astype(int)
        mu_fit_p, cov_fit_p = fitting.moments(Xp[fit_idx_p])
        Xn_fit = Xn[fit_idx_n]
        delta_ref = _vector_delta_ref(Xp, Xn, resamples)
        out_dir = ensure_dir(ad / "steerers" / "trained")
        for rank in list(getattr(trained, "ranks", [1, 4])):
            for seed in list(getattr(trained, "seeds", [0, 1])):
                steerer = fitting.fit_trained_adapter(
                    Xn_fit, mu_fit_p, cov_fit_p, int(rank),
                    steps=int(getattr(trained, "steps", 300)),
                    batch_size=int(getattr(trained, "batch_size", 128)),
                    lr=float(getattr(trained, "lr", 1e-3)),
                    lambda_cov=float(getattr(trained, "lambda_cov", 1.0)),
                    seed=int(seed),
                    standardize=bool(getattr(trained, "standardize", False)),
                )
                steerer.alpha_unit = calibration.calibrate_alpha(steerer, Xneutral, delta_ref)
                steerer.save(out_dir / f"{steerer.name}_seed{seed}.npz")
                params[f"trained/{steerer.name}_seed{seed}"] = spectra_probe.parameter_spectrum(steerer)

    e = no_op_steerer(d)
    e.alpha_unit = 1.0
    e.save(ensure_dir(ad / "steerers" / "resample0") / "E.npz")

    save_json(ad / "parameter_spectra.json", params)
    save_json(ad / "layer.json", {"layer": layer})
    mark_done(ad, {"n_resamples": len(resamples)})


def _vector_delta_ref(Xp, Xn, resamples) -> float:
    rng = np.random.default_rng(1000 + (resamples[0] if resamples else 0))
    idx_p = rng.integers(0, len(Xp), len(Xp))
    idx_n = rng.integers(0, len(Xn), len(Xn))
    mu_p, _ = fitting.moments(Xp[idx_p])
    mu_n, _ = fitting.moments(Xn[idx_n])
    return float(np.linalg.norm(mu_p - mu_n))


# ---------------------------------------------------------------- steerer loading

def load_attr_steerer(cfg, attr: str, arm: str) -> Optional[Steerer]:
    fit_dir = attr_dir(cfg, attr, "fit") / "steerers"
    if arm == "E":
        path = fit_dir / "resample0" / "E.npz"
    elif arm.startswith("C"):
        seeds = list(get(cfg, "steering.trained.seeds", [0]))
        path = fit_dir / "trained" / f"{arm}_seed{seeds[0]}.npz"
    else:
        path = fit_dir / "resample0" / f"{arm}.npz"
    if not path.exists():
        return None
    return Steerer.load(path)


def available_resamples(cfg, attr: str) -> list[int]:
    resamples = list(get(cfg, "steering.resamples", [0]) or [0])
    return [r for r in resamples if (attr_dir(cfg, attr, "fit") / "steerers" / f"resample{r}" / "A.npz").exists()]


# ---------------------------------------------------------------- stage: eval

def _build_classifier(cfg, attr: str, device):
    kind = getattr(cfg.attributes, attr).kind
    ad = attr_dir(cfg, attr, "extract")
    train_pos = [d["text"] for d in read_jsonl(ad / "clf_pos_train.jsonl")]
    train_neg = [d["text"] for d in read_jsonl(ad / "clf_neg_train.jsonl")]
    eval_pos = [d["text"] for d in read_jsonl(ad / "clf_pos_eval.jsonl")]
    eval_neg = [d["text"] for d in read_jsonl(ad / "clf_neg_eval.jsonl")]
    texts = train_pos + train_neg
    labels = [1] * len(train_pos) + [0] * len(train_neg)
    if kind in ("sentiment", "toxicity"):
        encoder = getattr(getattr(cfg.attributes, attr), "classifier_model", None) or str(
            get(cfg, "evaluation.classifier.embedding_model", "distilbert-base-uncased")
        )
        clf = EmbeddingLRClassifier(
            encoder_name=str(encoder),
            max_length=int(get(cfg, "evaluation.classifier.max_length", 256)),
        )
    else:
        clf = TfidfStyleClassifier(seed=int(get(cfg, "seed", 0)))
    clf.fit(texts, labels)
    eval_texts = eval_pos + eval_neg
    eval_labels = [1] * len(eval_pos) + [0] * len(eval_neg)
    eval_metrics = evaluate_classifier(clf, eval_texts, eval_labels)
    s_human = float(np.mean(clf.predict_proba_positive(eval_pos))) if eval_pos else 0.5
    return clf, eval_metrics, s_human


def eval_attr(cfg, attr: str, model, tokenizer, device, force: bool = False) -> None:
    from ..evaluation.generate import generate_condition  # lazy: torch-heavy import

    ad = ensure_dir(attr_dir(cfg, attr, "eval"))
    conditions_path = ad / "conditions.jsonl"
    if is_done(ad) and not force:
        print(f"[eval:{attr}] cached")
        return
    set_seed(int(get(cfg, "seed", 0)))
    layer = int(load_json(attr_dir(cfg, attr, "extract") / "layer.json", {}).get("layer", 0))
    clf, clf_metrics, s_human = _build_classifier(cfg, attr, device)
    save_json(ad / "classifier.json", {"metrics": clf_metrics, "s_human": s_human,
                                       "kind": getattr(cfg.attributes, attr).kind})

    n_prompts = int(get(cfg, "generation.prompts_per_attr", 60))
    lang = str(getattr(getattr(cfg.attributes, attr), "language", "en") or "en")
    prompts = neutral_prompts(n_prompts, seed=int(get(cfg, "seed", 0)), language=lang)
    gen_seeds = list(get(cfg, "generation.seeds", [0, 1]))
    arms = list(get(cfg, "generation.arms", ["A", "F16", "C1", "C4", "E", "R", "P"]))
    multipliers = list(get(cfg, "calibration.multipliers", [1.0]))
    gc = _gen_cfg(cfg)
    gens_dir = ensure_dir(ad / "gens")

    # baseline generations per seed
    baselines: dict[int, list[str]] = {}
    for seed in gen_seeds:
        f = gens_dir / f"baseline_seed{seed}.jsonl"
        if f.exists() and not force:
            baselines[seed] = [r["text"] for r in read_jsonl(f)]
        else:
            gens = generate_condition(model, tokenizer, prompts, None, None, 0.0, seed=seed, device=device, **gc)
            baselines[seed] = gens
            write_jsonl(f, [{"prompt": p, "text": g} for p, g in zip(prompts, gens)])
    base_ppl = {s: metric_utils.perplexity(model, tokenizer, baselines[s], **{k: gc[k] for k in ("batch_size",)}, device=device) for s in gen_seeds}

    rows: list[dict] = []
    # arm E (identity) — reuse baseline gens
    for seed in gen_seeds:
        rows.append(_condition_row(cfg, "E", 1.0, 1.0, seed, baselines[seed], baselines[seed],
                                   clf, model, tokenizer, base_ppl[seed], gc, device, prompts, is_prompting=False))
    # prompting arm P
    if "P" in arms:
        fit_pos = read_jsonl(attr_dir(cfg, attr, "extract") / "fit_pos.jsonl")
        examples = [d["text"] for d in fit_pos[:4]]
        p_prompts = [few_shot_prefix(p, examples) for p in prompts]
        gens = generate_condition(model, tokenizer, p_prompts, None, None, 0.0, seed=gen_seeds[0], device=device, **gc)
        rows.append(_condition_row(cfg, "P", 0.0, 0.0, gen_seeds[0], gens, baselines[gen_seeds[0]],
                                   clf, model, tokenizer, base_ppl[gen_seeds[0]], gc, device, prompts, is_prompting=True))
    # steered arms
    for arm in arms:
        if arm in ("E", "P"):
            continue
        steerer = load_attr_steerer(cfg, attr, arm)
        if steerer is None:
            print(f"[eval:{attr}] steerer {arm} missing — skipped")
            continue
        for m in multipliers:
            alpha = float(m) * steerer.alpha_unit
            cond_paths = [gens_dir / f"{arm}_m{m}_seed{s}.jsonl" for s in gen_seeds]
            if all(p.exists() for p in cond_paths) and not force:
                for s in gen_seeds:
                    gens = [r["text"] for r in read_jsonl(gens_dir / f"{arm}_m{m}_seed{s}.jsonl")]
                    rows.append(_condition_row(cfg, arm, float(m), alpha, s, gens, baselines[s], clf, model, tokenizer, base_ppl[s], gc, device, prompts, is_prompting=False))
                continue
            for s in gen_seeds:
                gens = generate_condition(model, tokenizer, prompts, steerer, layer, alpha, seed=s, device=device, **gc)
                write_jsonl(gens_dir / f"{arm}_m{m}_seed{s}.jsonl", [{"prompt": p, "text": g} for p, g in zip(prompts, gens)])
                rows.append(_condition_row(cfg, arm, float(m), alpha, s, gens, baselines[s], clf, model, tokenizer, base_ppl[s], gc, device, prompts, is_prompting=False))
    write_jsonl(conditions_path, rows)
    save_json(ad / "baseline_ppl.json", {str(k): v for k, v in base_ppl.items()})
    mark_done(ad, {"n_conditions": len(rows)})


def _condition_row(cfg, arm, m, alpha, seed, gens, baseline_gens, clf, model, tokenizer,
                   base_ppl_seed, gc, device, prompts, is_prompting: bool) -> dict:
    s = metric_utils.strength(clf, gens)
    ppl = metric_utils.perplexity(model, tokenizer, gens, batch_size=gc["batch_size"], device=device)
    ppl_ratio = float(ppl / base_ppl_seed) if base_ppl_seed and np.isfinite(base_ppl_seed) else float("nan")
    coherence = float(1.0 / ppl_ratio) if np.isfinite(ppl_ratio) and ppl_ratio > 0 else float("nan")
    sims = (
        metric_utils.content_similarity_pairs(gens, baseline_gens)
        if len(gens) == len(baseline_gens) and not is_prompting
        else np.array([float("nan")])
    )
    return {
        "arm": arm,
        "m": float(m),
        "alpha": float(alpha),
        "seed": int(seed),
        "n": int(len(gens)),
        "strength_mean": float(np.mean(s)),
        "strength_std": float(np.std(s)),
        "strength_p10": float(np.percentile(s, 10)),
        "strength_p90": float(np.percentile(s, 90)),
        "ppl": float(ppl),
        "ppl_ratio": ppl_ratio,
        "coherence": coherence,
        "distinct2": float(metric_utils.distinct_2(gens)),
        "rep4_mean": float(np.mean([metric_utils.repetition_4(g) for g in gens])) if gens else float("nan"),
        "content_sim_mean": float(np.nanmean(sims)) if np.isfinite(np.asarray(sims, dtype=float)).any() else float("nan"),
        "is_prompting": bool(is_prompting),
    }


# ---------------------------------------------------------------- stage: probes

def probes_attr(cfg, attr: str, force: bool = False) -> None:
    ad = ensure_dir(attr_dir(cfg, attr, "probes"))
    if is_done(ad) and not force:
        print(f"[probes:{attr}] cached")
        return
    data = np.load(activations_path(cfg, attr), allow_pickle=False)
    neutral = data["X_neutral"]

    arms_present = []
    res0 = attr_dir(cfg, attr, "fit") / "steerers" / "resample0"
    for name in ("A", "F1", "F4", "F16", "Ffull", "R"):
        if (res0 / f"{name}.npz").exists():
            arms_present.append(name)
    trained_dir = attr_dir(cfg, attr, "fit") / "steerers" / "trained"
    trained_names = sorted(p.stem for p in trained_dir.glob("C*.npz")) if trained_dir.exists() else []

    actions = {}
    for name in arms_present + trained_names:
        path = (res0 / f"{name}.npz") if name in arms_present else (trained_dir / f"{name}.npz")
        steerer = Steerer.load(path)
        actions[name] = {k: v for k, v in spectra_probe.action_spectrum(steerer, neutral, alpha=steerer.alpha_unit).items()}
    save_json(ad / "action_spectra.json", actions)

    if bool(get(cfg, "probes.separation", True)):
        sep = separation_probe.contrast_separation(data["X_stat_pos"], data["X_stat_neg"], seed=int(get(cfg, "seed", 0)))
        save_json(ad / "separation.json", sep)

    if bool(get(cfg, "probes.stability", True)):
        stability = {}
        # closed-form: spectra across resamples
        for name in ("A", "F1", "F4", "F16", "Ffull"):
            svs = []
            for r in available_resamples(cfg, attr):
                path = attr_dir(cfg, attr, "fit") / "steerers" / f"resample{r}" / f"{name}.npz"
                if path.exists():
                    st = Steerer.load(path)
                    svs.append(fitting.canonical_singular_values(st.W) if st.W is not None else np.array([np.linalg.norm(st.b)]))
            if len(svs) >= 2:
                dists = [stability_probe.sorted_spectrum_distance(svs[0], sv) for sv in svs[1:]]
                stability[f"{name}_spectrum_distance"] = dists
        # vector cosines across resamples
        vecs = []
        for r in available_resamples(cfg, attr):
            path = attr_dir(cfg, attr, "fit") / "steerers" / f"resample{r}" / "A.npz"
            if path.exists():
                vecs.append(Steerer.load(path).b)
        if len(vecs) >= 2:
            stability["A_vector_cosines"] = stability_probe.vector_cosines(vecs)
        # trained: cross-seed subspace overlap
        if len(trained_names) >= 2:
            flats = {}
            for name in trained_names:
                st = Steerer.load(trained_dir / f"{name}.npz")
                flats[name] = st
            by_rank: dict[int, list] = {}
            for name, st in flats.items():
                by_rank.setdefault(int(st.meta.get("rank", 1)), []).append(st)
            for rank, list_ in by_rank.items():
                if len(list_) >= 2:
                    s1, s2 = list_[0], list_[1]
                    stability[f"C{rank}_subspace_overlap"] = stability_probe.subspace_overlap(
                        stability_probe.top_directions(s1.W), stability_probe.top_directions(s2.W)
                    )
                    stability[f"C{rank}_hungarian_overlap"] = stability_probe.hungarian_direction_overlap(s1.W, s2.W)
                    stability[f"C{rank}_bias_cosines"] = stability_probe.vector_cosines([s1.b, s2.b])
        save_json(ad / "stability.json", stability)
    mark_done(ad)


# ---------------------------------------------------------------- dispatcher

def run_stage(cfg, stage: str, attrs: Optional[list[str]] = None, force: bool = False) -> None:
    attrs = attrs or _attr_keys(cfg)
    model = tokenizer = device = None
    need_model = stage in ("extract", "eval")
    if need_model:
        print(f"[{stage}] loading model {cfg.model.name}")
        model, tokenizer, device = load_model_and_tokenizer(cfg)
    for attr in attrs:
        if attr not in vars(cfg.attributes):
            raise SystemExit(f"unknown attribute '{attr}' (have: {', '.join(_attr_keys(cfg))})")
        print(f"===== {stage} :: {attr} =====")
        if stage == "extract":
            extract_attr(cfg, attr, model, tokenizer, device, force)
        elif stage == "fit":
            fit_attr(cfg, attr, force)
        elif stage == "eval":
            eval_attr(cfg, attr, model, tokenizer, device, force)
        elif stage == "probes":
            probes_attr(cfg, attr, force)
        else:
            raise SystemExit(f"stage '{stage}' is not runnable here; use 'analyze' CLI stage")
