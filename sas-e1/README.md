# SAS-E1 — Representation Complexity Study (code)

Implementation of **Experiment 1** from the SAS design doc
(`Obsidian Vault/AI Research/SAS E1 — Complexity Study Design.md`, v1.3, Lean mode,
including the **E1b Indic replication**).

The study asks three questions in one harness:

1. **Q1 (ladder)** — does the complexity of a steering representation separate attribute
   levels (L1 sentiment < L2 toxicity < L3 style)?
2. **Q2 (necessity)** — is a low-rank adapter needed over a rank-1 vector (and over a
   closed-form covariance operator)?
3. **Q3 (validity)** — does measured complexity predict steering difficulty?

All steerers are one nested family of linear operators on the residual stream:
`h' = h + W h + b` (vector: `W = 0`; closed-form moment operator: `W* = Σ_P^{1/2} Σ_N^{-1/2} − I`,
rank-truncated; trained adapter: `W = BA`).

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

### E1b — Indic replication (Hindi + Telugu)

Uses authentic, public-domain author corpora (hi/te Wikisource) plus permissively licensed
sentiment sets; same model, layer, steering families, probes and gates as the English run.

```bash
bash scripts/setup.sh configs/lean-indic.yaml        # sentiment + Wikisource + wiki neutral
uv run python scripts/run_experiment.py --config configs/lean-indic.yaml --stage all
```

Run it as a **separate session** after the English Lean run (~+60–90 min GPU). Telugu is a
small-n replication (Gurajada's corpus is thin); Hindi is the primary Indic test.

No-GPU smoke test of the math (numpy + pyyaml only, ephemeral env):

```bash
uv run --with numpy --with pyyaml python scripts/smoke_test.py
```

## Hardware expectations (Lean)

- Model: `bigscience/bloomz-1b7` or `bigscience/bloomz-560m` (pilot); dtype `auto` —
  bf16 on Ampere/L4-class GPUs, fp32 elsewhere. **Avoid fp16**: BLOOM fp16 overflows to
  inf/NaN on some GPUs (e.g., T4), which breaks extraction and sampling.
- Budget target: ~1.5–2 h GPU on bf16-class hardware (fp32 on T4 is roughly 2× slower).
- E1b Indic: same model, **+60–90 min** as a separate session (4 attributes).
- Trained arms fit **on cached activations only** — no backprop through the base model.
- Sample sizes are deliberately small (1,000 moment / 300 fit docs per side; 40 eval
  prompts) — adapters need hundreds of exemplars, and the sample-size curve validates saturation.

## Layout

```
configs/            lean + pilot + Indic (lean-indic) experiment configs
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
| Hindi sentiment (E1b) | `iam-tsr/hindi-sentiments` | MIT, ungated |
| Telugu sentiment (E1b) | `mounikaiiith/Telugu_Sentiment` | CC-BY-4.0, ungated |
| Indic author styles (E1b) | hi/te Wikisource — Premchand vs Prasad; Gurajada vs Veeresalingam | public-domain texts; MediaWiki API |

See the design doc §4.1–§4.1b for the full rationale, corpus sizes, and the Extended (deferred) list.
