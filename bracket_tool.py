"""Fix NER labels by editing plain text with [entity|TYPE] brackets, instead of one tag per line.

Step 1: turn a CoNLL file into one sentence per line, with entities in brackets
    python bracket_tool.py to-brackets my_test.conll -o my_test_edit.txt

Step 2: open my_test_edit.txt in any text editor and fix the brackets, for example
    Datuk [Harith|PER] melancarkan projek di [Langkawi|LOC] .
  Add brackets around anything the model missed, remove or move wrong ones, and change
  the TYPE (PER, LOC or ORG) where needed. Everything outside brackets is "not an entity".
  You can also write brackets on brand-new sentences from scratch.

Step 3: turn it back into a CoNLL file for training and evaluation
    python bracket_tool.py to-conll my_test_edit.txt -o my_test_fixed.conll

Punctuation is split off automatically ("Langkawi." becomes "Langkawi" and "."), so you
do not need to put spaces before full stops or commas.
"""
import argparse
from collections import Counter

from prelabel import tokenise
from utils import KEEP_TYPES, read_conll
import re

BRACKET_RE = re.compile(r"\[([^\[\]|]+)\|([A-Za-z]+)\]")


def conll_to_brackets(tokens, tags):
    """['Ahmad', 'at', 'UTM'] + ['B-PER', 'O', 'B-ORG'] -> '[Ahmad|PER] at [UTM|ORG]'."""
    for token in tokens:
        if any(c in token for c in "[]|"):
            raise ValueError(f"Token {token!r} contains [, ] or | which this format cannot hold")
    parts = []
    i = 0
    while i < len(tokens):
        tag = tags[i]
        if tag == "O" or tag[2:] not in KEEP_TYPES:
            parts.append(tokens[i])
            i += 1
            continue
        kind = tag[2:]
        j = i + 1
        while j < len(tokens) and tags[j] == f"I-{kind}":
            j += 1
        parts.append("[" + " ".join(tokens[i:j]) + "|" + kind + "]")
        i = j
    return " ".join(parts)


def brackets_to_conll(line, line_no=0):
    """'Datuk [Harith|PER] di [Johor Bahru|LOC].' -> (tokens, BIO tags)."""
    tokens, tags = [], []

    def add(text, kind=None):
        if kind is None and any(c in text for c in "[]"):
            raise ValueError(f"line {line_no}: unmatched or badly formed bracket near {text.strip()!r}. "
                             "Use the form [text|TYPE] with TYPE = PER, LOC or ORG.")
        for k, (token, _, _) in enumerate(tokenise(text)):
            tokens.append(token)
            tags.append("O" if kind is None else ("B-" if k == 0 else "I-") + kind)

    pos = 0
    for m in BRACKET_RE.finditer(line):
        add(line[pos:m.start()])
        kind = m.group(2).upper()
        if kind not in KEEP_TYPES:
            raise ValueError(f"line {line_no}: unknown type {m.group(2)!r}. Use one of {KEEP_TYPES}.")
        add(m.group(1), kind)
        pos = m.end()
    add(line[pos:])
    return tokens, tags


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("to-brackets", help="CoNLL file -> text with [entity|TYPE] brackets")
    a.add_argument("input")
    a.add_argument("-o", "--output", required=True)
    b = sub.add_parser("to-conll", help="text with brackets -> CoNLL file")
    b.add_argument("input")
    b.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    if args.command == "to-brackets":
        sentences, all_tags = read_conll(args.input)
        with open(args.output, "w", encoding="utf-8") as f:
            for tokens, tags in zip(sentences, all_tags):
                f.write(conll_to_brackets(tokens, tags) + "\n")
        print(f"Wrote {len(sentences)} sentences to {args.output}")
        print("Edit the brackets, then run: python bracket_tool.py to-conll "
              f"{args.output} -o <fixed>.conll")
    else:
        with open(args.input, encoding="utf-8-sig") as f:
            lines = [(n, line.strip()) for n, line in enumerate(f, start=1)]
        counts = Counter()
        written = 0
        with open(args.output, "w", encoding="utf-8") as out:
            for n, line in lines:
                if not line:
                    continue
                tokens, tags = brackets_to_conll(line, n)
                if not tokens:
                    continue
                for token, tag in zip(tokens, tags):
                    out.write(f"{token} {tag}\n")
                out.write("\n")
                written += 1
                counts.update(t[2:] for t in tags if t.startswith("B-"))
        print(f"Wrote {written} sentences to {args.output}")
        print("Entities: " + ", ".join(f"{k} {counts[k]}" for k in KEEP_TYPES))


if __name__ == "__main__":
    main()
