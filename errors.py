"""List every mistake the model makes on a labelled CoNLL file (for error analysis).

    python errors.py my_test.conll --model model_baseline
    python errors.py my_test.conll --model model_malaysian --csv errors_after.csv

Run it on the SAME test file for the baseline and the retrained model, and compare.
Mistakes are sorted into:
  missed          a real entity the model did not find at all
  wrong type      right words, wrong label (e.g. hospital tagged LOC instead of ORG)
  wrong boundary  overlaps a real entity but starts or ends in the wrong place
  false alarm     the model found an entity that is not there (e.g. "Kami" as PER)
"""
import argparse
import csv
from collections import Counter

from seqeval.metrics.sequence_labeling import get_entities

from prelabel import tag_tokens
from utils import KEEP_TYPES, read_conll


def keep_types_only(tags):
    """Ignore entity types the project does not need (e.g. MISC)."""
    return [t if t == "O" or t[2:] in KEEP_TYPES else "O" for t in tags]


def overlaps(a, b):
    """Spans are (type, start, end) with end inclusive."""
    return a[1] <= b[2] and b[1] <= a[2]


def classify(gold_spans, pred_spans):
    """Return a list of (category, gold_span or None, predicted_span or None)."""
    results = []
    for g in gold_spans:
        if g in pred_spans:
            results.append(("correct", g, g))
            continue
        same_bounds = [p for p in pred_spans if p[1:] == g[1:]]
        overlapping = [p for p in pred_spans if overlaps(g, p)]
        if same_bounds:
            results.append(("wrong type", g, same_bounds[0]))
        elif overlapping:
            results.append(("wrong boundary", g, overlapping[0]))
        else:
            results.append(("missed", g, None))
    for p in pred_spans:
        if not any(overlaps(p, g) for g in gold_spans):
            results.append(("false alarm", None, p))
    return results


def describe(tokens, span):
    if span is None:
        return "-"
    kind, start, end = span
    return f"{kind} '{' '.join(tokens[start:end + 1])}'"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input", help="labelled CoNLL file (your test file)")
    parser.add_argument("--model", default="model_baseline", help="folder saved by train_ner.py")
    parser.add_argument("--csv", help="also save every mistake to this CSV file")
    args = parser.parse_args()

    from predict import load_ner  # imported here so the helpers above stay lightweight

    sentences, gold_tags = read_conll(args.input)
    ner = load_ner(args.model)

    counts = Counter()
    total_pred = 0
    rows = []
    for n, (tokens, gold) in enumerate(zip(sentences, gold_tags), start=1):
        # Same words as the labelled file, joined with spaces, as train_ner.py evaluates them.
        offsets, pos = [], 0
        for token in tokens:
            offsets.append((token, pos, pos + len(token)))
            pos += len(token) + 1
        text = " ".join(tokens)

        entities = [e for e in ner(text) if e["entity_group"] in KEEP_TYPES]
        entities.sort(key=lambda e: e["start"])
        pred = tag_tokens(offsets, entities)

        gold_spans = get_entities(keep_types_only(gold))
        pred_spans = get_entities(pred)
        total_pred += len(pred_spans)
        for category, g, p in classify(gold_spans, pred_spans):
            counts[category] += 1
            if category != "correct":
                rows.append((n, category, describe(tokens, g), describe(tokens, p), text))

    total_gold = counts["correct"] + counts["wrong type"] + counts["wrong boundary"] + counts["missed"]
    print(f"Real entities in file: {total_gold}")
    for category in ("correct", "wrong type", "wrong boundary", "missed", "false alarm"):
        print(f"  {category:<15} {counts[category]}")
    if total_gold and total_pred:
        precision = counts["correct"] / total_pred
        recall = counts["correct"] / total_gold
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"  precision {precision:.3f}   recall {recall:.3f}   F1 {f1:.3f}   (exact entity match)")

    print("\nMistakes:")
    for n, category, gold_text, pred_text, text in rows:
        print(f"#{n:<4} {category.upper():<15} real: {gold_text:<28} model: {pred_text:<28} | {text}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sentence_no", "category", "real", "model", "sentence"])
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} mistakes to {args.csv}")


if __name__ == "__main__":
    main()
