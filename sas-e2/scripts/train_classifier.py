#!/usr/bin/env python3
"""Train the Phase 1 multilingual sentiment classifier (XLM-R base).

The pilot uses this as the external sentiment instrument. It is trained on
`clf_train` for all five languages with language-balanced sampling, evaluated
on the held-out `clf_eval` split per language, and saved with metrics.

The final study can replace this with per-language instruments; the interface
is the same: a directory containing `model/`, `tokenizer/` and `metrics.json`.

Example:
  python3 scripts/train_classifier.py
  python3 scripts/train_classifier.py --epochs 2 --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _models import (  # noqa: E402
    DATASET_ROOT,
    LANGS,
    REPO_ROOT,
    utc_now,
)

DEFAULT_OUT = REPO_ROOT / "artifacts" / "classifier" / "xlmr_phase1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Phase 1 multilingual sentiment classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        parser.error("batch sizes must be > 0")
    return args


def load_split(split: str) -> list[dict]:
    rows = []
    for lang in LANGS:
        path = DATASET_ROOT / lang / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["lang"] = lang
                    rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        print("[clf] ERROR: CUDA is required to train XLM-R on the T4.", file=sys.stderr)
        return 2

    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_rows = load_split("clf_train")
    eval_rows = load_split("clf_eval")
    train_counts = {lang: sum(1 for row in train_rows if row["lang"] == lang) for lang in LANGS}
    eval_counts = {lang: sum(1 for row in eval_rows if row["lang"] == lang) for lang in LANGS}
    print("[clf] train counts:", train_counts, flush=True)
    print("[clf] eval counts: ", eval_counts, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    class SentimentDataset(Dataset):
        def __init__(self, rows):
            self.encodings = tokenizer(
                [row["text"] for row in rows],
                truncation=True,
                max_length=args.max_length,
                padding=False,
            )
            self.labels = [int(row["label"]) for row in rows]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[index])
            return item

    train_dataset = SentimentDataset(train_rows)
    eval_datasets = {
        lang: SentimentDataset([row for row in eval_rows if row["lang"] == lang])
        for lang in LANGS
    }

    # Balance the five languages: English has 4x the training rows of each Indic language.
    sample_weights = [1.0 / train_counts[row["lang"]] for row in train_rows]

    class BalancedTrainer(Trainer):
        def __init__(self, *args, sample_weights=None, **kwargs):
            self.sample_weights = sample_weights
            super().__init__(*args, **kwargs)

        def _get_train_sampler(self, *args, **kwargs):
            return WeightedRandomSampler(
                self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True,
            )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )
    model.config.use_cache = False
    model.to(device)

    out = Path(args.out).expanduser().resolve()
    training_args = TrainingArguments(
        output_dir=str(out / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=True,
        dataloader_num_workers=args.num_workers,
        fp16=not args.no_fp16,
        seed=args.seed,
        optim="adamw_torch",
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = BalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        sample_weights=sample_weights,
    )

    print(f"[clf] training {args.model_name} for {args.epochs} epoch(s)", flush=True)
    started = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - started
    print(f"[clf] training finished in {elapsed:.1f}s", flush=True)

    model.eval()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "model")
    tokenizer.save_pretrained(out / "tokenizer")

    @torch.inference_mode()
    def predict(dataset):
        loader = DataLoader(
            dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        probs, margins, labels = [], [], []
        for batch in loader:
            batch_labels = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**batch).logits.float()
            probs.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
            margins.extend((logits[:, 1] - logits[:, 0]).cpu().tolist())
            labels.extend(batch_labels.tolist())
        return probs, margins, labels

    def metric_block(probs, labels):
        predictions = [1 if value >= 0.5 else 0 for value in probs]
        result = {
            "n": len(labels),
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        }
        if len(set(labels)) > 1:
            result["auroc"] = float(roc_auc_score(labels, probs))
        else:
            result["auroc"] = None
        return result

    metrics = {
        "created_utc": utc_now(),
        "model_name": args.model_name,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "train_counts": train_counts,
        "eval_counts": eval_counts,
        "train_seconds": round(elapsed, 1),
        "languages": {},
    }

    all_probs, all_labels = [], []
    for lang in LANGS:
        probs, margins, labels = predict(eval_datasets[lang])
        metrics["languages"][lang] = metric_block(probs, labels)
        metrics["languages"][lang]["mean_probability_positive"] = float(np.mean(probs))
        all_probs.extend(probs)
        all_labels.extend(labels)

    metrics["overall"] = metric_block(all_probs, all_labels)
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\n[clf] validation", flush=True)
    for lang in LANGS:
        row = metrics["languages"][lang]
        print(
            f"[clf] {lang}: n={row['n']} auroc={row['auroc']:.3f} "
            f"acc={row['accuracy']:.3f} bal_acc={row['balanced_accuracy']:.3f}",
            flush=True,
        )
    overall = metrics["overall"]
    print(
        f"[clf] overall: n={overall['n']} auroc={overall['auroc']:.3f} "
        f"acc={overall['accuracy']:.3f}",
        flush=True,
    )
    print(f"[clf] saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
