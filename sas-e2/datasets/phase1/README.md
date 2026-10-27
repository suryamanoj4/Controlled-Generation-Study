# SAS E2 — Phase 1 Dataset (Sentiment, L1)

Run-ready sentiment data for **en, hi, bn, te, ta**, with identical per-class sizes across languages.

## Counts

| Split | Per language | Composition |
|---|---|---|
| `stat.jsonl` | **500** | 250 positive + 250 negative (matched across languages) |
| `clf_train.jsonl` | **500 (Indic) / 4,000 (EN)** | 250/2,000 positive + 250/2,000 negative |
| `clf_eval.jsonl` | **152 (Indic) / 1,000 (EN)** | 80/500 positive + 72/500 negative |
| Total unique | **1,152 (Indic) / 5,500 (EN)** | 580/2,750 positive + 572/2,750 negative |

Layout:

```
phase1/
  sentiment/
    en/ {stat,clf_train,clf_eval}.jsonl
    hi/ ...
    bn/ ...
    te/ ...
    ta/ ...
    prompts/ {en,hi,bn,te,ta}.jsonl      # 30 neutral review prompts each
    prompts/topics.json                     # 30 topics with translations
    prompts/prompting_templates.json        # neutral / positive / negative templates
  flores200/                                # competence diagnostic (dev/devtest)
  aux/generic_prompts_en.jsonl              # 100 generic prompts (not sentiment-specific)
  manifest.json
```

Record fields: `id`, `text`, `label` (1 = positive, 0 = negative), `lang`, `source`, `split`.

## Sources and licenses

- **English**: Amazon Polarity via `fancyzhx/amazon_polarity` (test split; Apache-2.0). Document = title + first 20 content words; deduplicated and sampled to 580/572 (seed 42). Median length 24 words, matching the Indic reviews.
- **Indic**: `ai4bharat/IndicSentiment` test + validation. Original card declares **no license**; `mteb/IndicSentiment` tags it **CC0-1.0**. Cite AI4Bharat; do not redistribute raw text without clarifying the license.
- **FLORES-200** (diagnostic only): CC-BY-SA-4.0.
- Indic data is **translated product/hotel reviews** — valid for sentiment (L1), not for author style.

## Prompts

`prompts/{lang}.jsonl` contains 30 in-language neutral review prompts. The prompting baseline is built from `prompting_templates.json`:

- `neutral`: "Write a review of {topic}."
- `positive` / `negative`: "Write a positive/negative review of {topic}."

**Verification status:** en/hi/te are straightforward; **bn/ta translations need native-speaker verification** before the final run. Script ratio is 1.0 for all Indic prompt files.

## Why these sizes

- 250/class is enough for a mean-difference (DiffMean/CAA) steering vector; 250/class classifier training + 152 evaluation gives a more stable instrument.
- Report 5-fold CV AUROC on the full 1,152 as the instrument-quality number.
- F4/F16 probes at 300/class are **exploratory only** (covariance underdetermined); Phase 1 is vector-only behaviorally.
- English was **downsampled** to the same counts so the sentiment-vs-style ladder is not confounded by sample size.

## Not included

- Phase 2 author corpora (separate folder; Bengali/Tamil pending).
- Phase 3 cross-lingual style transfer.
- SST-2 (evaluated and rejected: sentence lengths do not match the Indic reviews; no SST-2 data is included).
- Language-control prompts/instructions (the FLORES diagnostic remains only for competence checks).
