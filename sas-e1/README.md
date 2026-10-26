# SAS-E1 — Representation Complexity Study (code)

Implementation of **Experiment 1** from the SAS design doc
(`Obsidian Vault/AI Research/SAS E1 — Complexity Study Design.md`, v1.5, Lean mode).
The single Lean run covers both the English ladder and the Indic replication attributes.

The study asks three questions in one harness:

1. **Q1 (ladder)** — does the complexity of a steering representation separate attribute
   levels (L1 sentiment < L2 toxicity < L3 style)?
2. **Q2 (necessity)** — is a low-rank adapter needed over a rank-1 vector (and over a
   closed-form covariance operator)?
3. **Q3 (validity)** — does measured complexity predict steering difficulty?

All steerers are one nested family of linear operators on the residual stream:
`h' = h + W h + b` (vector: `W = 0`; closed-form moment operator: `W* = Σ_P^{1/2} Σ_N^{-1/2} − I`,
rank-truncated; trained adapter: `W = BA`).

## Results (Lean run — 2026-10-18, fable re-run 2026-10-24, validity audit 2026-10-26)

The full Lean run completed (8 attributes, all stages; two resumable Colab sessions) and
the analyzed outputs live in `artifacts/lean/analysis/` (`report.md`, `summary.json`).
Headline: the complexity ladder is confirmed — `H_action(F16)` level means **L1 0.054 <
L2 0.316 < L3 0.849** — though L3 is heterogeneous (shakespeare erank ≈ 5.3, story_hi
2.5, fable 1.7, story_te 1.3 vs ≈ 1.0 for sentiment); closed-form spectra reproduce
tightly across resamples while trained adapters collapse to near-rank-1 action. Fable's
contrast was rebuilt from sentence-packed Wikipedia chunks (1000/1000 matched docs;
`8f02c76`) and re-run — its corrected complexity is about half the earlier undersampled
estimate.

**⚠️ Behavioral validity correction (2026-10-26):** a generation audit found the model
produced **~0% in-language Indic output** (0–5 / 2640; 34–56% empty), so **all Indic
behavioral results are withdrawn** (representation results remain valid). A human check
also shows no perceptible Shakespearean register at m = 2 and fable degeneration.
Classifier strength alone is not sufficient evidence; behavioral claims now require an
in-language gate plus human/judge validation. G3 (complexity↔difficulty) remains not
reportable (EN legs never reach the strength target at m ≤ 4). Full write-up:
`Obsidian Vault/AI Research/SAS E1 — Results (Lean Run).md`.

## Quickstart (Colab / Kaggle)

With [uv](https://docs.astral.sh/uv/) (recommended — no system venv required, fast, and
it can run the smoke test in an ephemeral environment):

```bash
# 1) create .venv + install deps, download models + prepare datasets (idempotent)
bash scripts/setup.sh configs/lean.yaml

# 2) run the experiment stage by stage (resumable), or all at once
uv run python scripts/run_experiment.py --config configs/lean.yaml --stage all
```

On Colab/Kaggle (before `uv sync`): `%pip install uv` then run the same commands from a
notebook cell with `!uv ...` / `!bash scripts/setup.sh ...`.

Selective setup — models and data can be provisioned independently; with no flag both run:

```bash
# datasets only (e.g., your model snapshots are already cached elsewhere)
python scripts/download_assets.py --config configs/lean.yaml --data-only
# model snapshots only (e.g., datasets prepared on a previous session)
python scripts/download_assets.py --config configs/lean.yaml --models-only
# both (default)
python scripts/download_assets.py --config configs/lean.yaml
```

`scripts/setup.sh` forwards the same flags after the config path.

Downloads are target-driven: streaming datasets stop at their caps, and Gutenberg /
Wikisource fetch works/pages in a **seeded random order**, stopping once the split's needs
are met (so the setup fetches only what the experiment uses; change `seed` to vary which
titles are covered).

Without uv (legacy pip):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_assets.py --config configs/lean.yaml
python scripts/run_experiment.py --config configs/lean.yaml --stage all
```

Stages (`--stage`): `extract`, `fit`, `eval`, `probes`, `analyze`, `all`.
Every stage is resumable; completed work is marked with `_done.json` and skipped.
`--force` re-runs; `--attr sentiment` restricts to one attribute.

Pilot (560M, sentiment only, minutes):

```bash
bash scripts/setup.sh configs/pilot.yaml
uv run python scripts/run_experiment.py --config configs/pilot.yaml --stage all
```

### Indic attributes (part of the same Lean run)

The Lean config includes — alongside the English attributes — **Hindi and Telugu**
sentiment (L1) and author-style (L3) attributes, using authentic public-domain corpora
(hi/te Wikisource) and permissively licensed sentiment sets. Same model, layer, steering
families, probes and gates. Setup downloads them automatically; restrict when needed:

```bash
# full run (all 8 attributes)
uv run python scripts/run_experiment.py --config configs/lean.yaml --stage all

# example: only the Indic attributes
uv run python scripts/run_experiment.py --config configs/lean.yaml \
  --attr sentiment_hi,story_hi,sentiment_te,story_te --stage all
```

The Indic L3 author corpora use a uniform **200-doc budget per author** (pool 130 + 45/25
classifier); Gurajada caps at 177, so the Telugu pair matches at ~150–170. Telugu remains a
small-n replication; Hindi is the primary Indic test.

No-GPU smoke test of the math (numpy + pyyaml only, ephemeral env):

```bash
uv run --with numpy --with pyyaml python scripts/smoke_test.py
```

## Hardware expectations (Lean)

- Model: `bigscience/bloomz-1b7` or `bigscience/bloomz-560m` (pilot); dtype `auto` —
  bf16 on Ampere/L4-class GPUs, fp32 elsewhere. **Avoid fp16**: BLOOM fp16 overflows to
  inf/NaN on some GPUs (e.g., T4), which breaks extraction and sampling.
- Budget target: ~2–2.5 h GPU for the full Lean run (8 attributes) on bf16-class hardware;
  fp32-only GPUs are ~2–3× slower — stages and attributes are resumable, so split if needed.
- Trained arms fit **on cached activations only** — no backprop through the base model.
- Sample sizes are deliberately small (1,000 moment / 300 fit docs per side; 40 eval
  prompts) — adapters need hundreds of exemplars, and the sample-size curve validates saturation.

## Layout

```
configs/            lean (full E1: EN + Indic) + pilot configs
sas_e1/
  config.py         dotted-access config loader
  utils.py          metrics helpers (entropy, MTLD, IO, done-markers)
  models.py         model loading, layer resolution, hooks, extraction
  data/             downloads, corpus loading, matching protocol, prompt sets
  steering/         operators, fitting objectives, alpha calibration
  probes/           action/parameter spectra, separation pre-screen, stability
  evaluation/       classifiers, steered generation, coherence metrics, judge
  experiment/       pipeline stages + analysis/gates
scripts/            download_assets, run_experiment, smoke_test, setup.sh
tests/              unit tests for operators and metrics
```

Artifacts land in `artifacts/<run_name>/<attribute>/<stage>/`. Raw prepared corpora land in
`data/processed/` as JSONL with a `manifest.json` recording sources, counts and licenses.

## Licensing / sources (verified 2026-10-16)

| Role | Source | License |
|---|---|---|
| Sentiment | SST-2 via `nyu-mll/glue` (`sst2`) | GLUE "other" (per-task), ungated |
| Toxicity | `google/civil_comments` | CC0-1.0 |
| Style (positive) | Project Gutenberg (Shakespeare plays) | US public domain (PG header stripped) |
| Style (contrast) | Project Gutenberg (Marlowe/Jonson/Webster) | US public domain |
| Fable | `demelin/moral_stories` | MIT |
| Wikipedia contrast / neutral probe | `wikimedia/wikipedia` | CC-BY-SA-3.0 + GFDL (attribution required) |
| Optional implicit toxicity | `toxigen/toxigen-data` | MIT + CDLA-Permissive (research-use note) |
| Hindi sentiment (Indic) | `iam-tsr/hindi-sentiments` | MIT, ungated |
| Telugu sentiment (Indic) | `mounikaiiith/Telugu_Sentiment` | CC-BY-4.0, ungated |
| Indic author styles (Indic) | hi/te Wikisource — Premchand vs Prasad; Gurajada vs Veeresalingam | public-domain texts; MediaWiki API |

See the design doc §4.1–§4.1b for the full rationale, corpus sizes, and the Extended (deferred) list.
