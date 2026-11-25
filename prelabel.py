"""Pre-label raw sentences with your baseline model, so you only have to CORRECT them.

    python prelabel.py raw_train.txt -o my_train.conll --model model_baseline

Input : a text file with one sentence per line (blank lines and lines starting with
        a single # are ignored).
Output: a CoNLL file (one "token tag" per line, blank line between sentences) and
        <output>.review.txt, which lists what to check first.

Each sentence block starts with a "### n | sentence" comment so you can read the full
sentence while correcting. read_conll() in utils.py skips these lines, so you do not
need to delete them before training.
"""
import argparse
import os
import re
import unicodedata

from utils import KEEP_TYPES

# Same granularity as the model's own word splitting: words, and each punctuation mark.
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenise(text):
    """Return a list of (token, start, end) with character offsets."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)
            if unicodedata.category(m.group()[0]) != "Cf"]  # skip invisible characters such as a BOM


def tag_tokens(tokens, entities):
    """Give each token a BIO tag from entities that have 'start', 'end', 'entity_group'."""
    tags = []
    last_entity = None
    for _, start, end in tokens:
        tag = "O"
        for i, e in enumerate(entities):
            if e["start"] <= start and end <= e["end"]:
                prefix = "I" if last_entity == i else "B"
                tag = f"{prefix}-{e['entity_group']}"
                last_entity = i
                break
        else:  # token is not inside any entity
            last_entity = None
        tags.append(tag)
    return tags


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input", help="text file, one sentence per line")
    parser.add_argument("-o", "--output", help="output .conll file (default: input name + .conll)")
    parser.add_argument("--model", default="model_baseline", help="folder saved by train_ner.py")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="entities scoring below this are listed in the review file")
    args = parser.parse_args()

    from predict import load_ner  # imported here so the helpers above stay lightweight

    with open(args.input, encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f]
    sentences = [s for s in lines if s and not s.startswith("#")]

    out_path = args.output or os.path.splitext(args.input)[0] + ".conll"
    review_path = os.path.splitext(out_path)[0] + ".review.txt"
    ner = load_ner(args.model)

    review = []
    with open(out_path, "w", encoding="utf-8") as out:
        for n, sentence in enumerate(sentences, start=1):
            entities = [e for e in ner(sentence) if e["entity_group"] in KEEP_TYPES]
            entities.sort(key=lambda e: e["start"])
            tokens = tokenise(sentence)
            tags = tag_tokens(tokens, entities)

            out.write(f"### {n} | {sentence}\n")
            for (token, _, _), tag in zip(tokens, tags):
                out.write(f"{token} {tag}\n")
            out.write("\n")

            if not entities:
                review.append(f"#{n}  NO ENTITIES FOUND (check for misses)  | {sentence}")
            for e in entities:
                if float(e["score"]) < args.threshold:
                    surface = sentence[e["start"]:e["end"]]
                    review.append(f"#{n}  score={float(e['score']):.2f}  {e['entity_group']} {surface!r}  | {sentence}")

    with open(review_path, "w", encoding="utf-8") as f:
        f.write("\n".join(review) + ("\n" if review else ""))

    print(f"Wrote {len(sentences)} sentences to {out_path}")
    print(f"{len(review)} items to check first are listed in {review_path}")
    print("Now open the .conll file, correct the tags, and keep the tag names exactly as above.")


if __name__ == "__main__":
    main()
