"""Fine-tune DistilBERT for named-entity recognition (PER / LOC / ORG) on CoNLL-2003.

You can optionally add your own CoNLL-format data (for example Malaysian names and
places) to compare results before and after.

Examples
--------
Baseline (CoNLL-2003 only):
    python train_ner.py --output_dir model_baseline

With lowercase augmentation (helps with chat-style text):
    python train_ner.py --lowercase_frac 0.3 --output_dir model_lower

With your own data and your own held-out test file:
    python train_ner.py --extra_train my_train.conll --extra_test my_test.conll \
        --output_dir model_malaysian
"""
import argparse
import inspect
import json
import os

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from utils import read_conll

# Copies of CoNLL-2003 on the Hugging Face Hub. Names and formats change over time,
# so we try several.
DATASET_NAMES = ["eriktks/conll2003", "tomaarsen/conll2003", "lhoestq/conll2003", "conll2003"]


def load_conll2003():
    last_error = None
    for name in DATASET_NAMES:
        for kwargs in ({}, {"trust_remote_code": True}):
            try:
                ds = load_dataset(name, **kwargs)
                print(f"Loaded dataset: {name}")
                return ds
            except Exception as err:  # try the next option
                last_error = err
    raise RuntimeError(f"Could not load CoNLL-2003. Last error: {last_error}")


def load_extra(path, label2id):
    """Load your own CoNLL-format file and convert tags to ids."""
    sentences, tags = read_conll(path)
    unknown = {t for row in tags for t in row} - set(label2id)
    if unknown:
        raise ValueError(f"Unknown tags in {path}: {sorted(unknown)}. Allowed: {sorted(label2id)}")
    return Dataset.from_dict(
        {"tokens": sentences, "ner_tags": [[label2id[t] for t in row] for row in tags]}
    )


def lowercase_copies(ds, fraction, seed):
    """Make lowercased copies of a share of the sentences.

    CoNLL is properly capitalised news text, so the model leans on capital letters.
    Chat-style text ("ahmad nak pergi kl") has none, so we also train on lowercase copies.
    """
    n = int(len(ds) * fraction)
    subset = ds.shuffle(seed=seed).select(range(n))
    return subset.map(lambda ex: {"tokens": [t.lower() for t in ex["tokens"]]})


def make_tokenise_fn(tokenizer):
    """Tokenise words into subword pieces and align one label to each word.

    Only the FIRST piece of each word keeps its label. Other pieces and special
    tokens get -100, which the loss function ignores.
    """

    def tokenise(batch):
        enc = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True)
        all_labels = []
        for i, tags in enumerate(batch["ner_tags"]):
            previous_word = None
            labels = []
            for word_id in enc.word_ids(batch_index=i):
                if word_id is None:
                    labels.append(-100)
                elif word_id != previous_word:
                    labels.append(tags[word_id])
                else:
                    labels.append(-100)
                previous_word = word_id
            all_labels.append(labels)
        enc["labels"] = all_labels
        return enc

    return tokenise


def decode(logits, label_ids, id2label):
    """Turn model outputs into tag strings per sentence, skipping ignored positions."""
    preds = np.argmax(logits, axis=2)
    true_tags, pred_tags = [], []
    for pred_row, label_row in zip(preds, label_ids):
        t, p = [], []
        for pred, label in zip(pred_row, label_row):
            if label != -100:
                t.append(id2label[int(label)])
                p.append(id2label[int(pred)])
        true_tags.append(t)
        pred_tags.append(p)
    return true_tags, pred_tags


def make_metrics(id2label):
    def compute(p):
        true_tags, pred_tags = decode(p.predictions, p.label_ids, id2label)
        return {
            "precision": precision_score(true_tags, pred_tags),
            "recall": recall_score(true_tags, pred_tags),
            "f1": f1_score(true_tags, pred_tags),
        }

    return compute


def evaluate_split(trainer, dataset, id2label):
    """Print a per-type report (entity-level, via seqeval) and return overall scores."""
    out = trainer.predict(dataset)
    true_tags, pred_tags = decode(out.predictions, out.label_ids, id2label)
    print(classification_report(true_tags, pred_tags, digits=4))
    return {
        "precision": precision_score(true_tags, pred_tags),
        "recall": recall_score(true_tags, pred_tags),
        "f1": f1_score(true_tags, pred_tags),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model_name", default="distilbert-base-cased")
    parser.add_argument("--output_dir", default="model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_fp16", action="store_true",
                        help="turn off mixed precision (try this if the loss becomes nan)")
    parser.add_argument("--lowercase_frac", type=float, default=0.0,
                        help="add lowercased copies of this share of CoNLL sentences (e.g. 0.3)")
    parser.add_argument("--extra_train", help="your own CoNLL-format training file")
    parser.add_argument("--extra_repeat", type=int, default=3,
                        help="repeat the small extra set this many times so it is not drowned out")
    parser.add_argument("--extra_test", help="your own CoNLL-format test file (keep it separate!)")
    args = parser.parse_args()
    set_seed(args.seed)

    raw = load_conll2003()
    label_list = raw["train"].features["ner_tags"].feature.names
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    print("Labels:", label_list)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenise = make_tokenise_fn(tokenizer)

    def prepare(ds):
        return ds.map(tokenise, batched=True, remove_columns=ds.column_names)

    train_ds = prepare(raw["train"])
    if args.lowercase_frac > 0:
        lower = lowercase_copies(raw["train"], args.lowercase_frac, args.seed)
        print(f"Adding {len(lower)} lowercased copies of CoNLL sentences")
        train_ds = concatenate_datasets([train_ds, prepare(lower)])
    val_ds = prepare(raw["validation"])
    test_ds = prepare(raw["test"])

    if args.extra_train:
        extra = prepare(load_extra(args.extra_train, label2id))
        print(f"Adding {len(extra)} extra training sentences x{args.extra_repeat}")
        train_ds = concatenate_datasets([train_ds] + [extra] * args.extra_repeat).shuffle(seed=args.seed)
    extra_test_ds = prepare(load_extra(args.extra_test, label2id)) if args.extra_test else None

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )

    # The argument was renamed from evaluation_strategy to eval_strategy in newer versions.
    params = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available() and not args.no_fp16,
        report_to="none",
        seed=args.seed,
        **{eval_key: "epoch"},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=make_metrics(id2label),
    )
    trainer.train()

    results = {"settings": vars(args)}
    for name, ds in (("conll_test", test_ds), ("extra_test", extra_test_ds)):
        if ds is None:
            continue
        print(f"\n=== {name} ===")
        results[name] = evaluate_split(trainer, ds, id2label)

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved model and results.json to {args.output_dir}")


if __name__ == "__main__":
    main()
