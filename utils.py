"""Small helpers shared by train_ner.py and predict.py."""
import re

KEEP_TYPES = ("PER", "LOC", "ORG")  # the brief only requires these three
MASKS = {"PER": "[PERSON]", "LOC": "[LOCATION]", "ORG": "[ORGANISATION]"}


def read_conll(path):
    """Read a CoNLL-style file: one 'token tag' pair per line, a blank line between sentences.

    The tag is the LAST column, so files with extra columns also work.
    Returns (list of token lists, list of tag lists).
    """
    sentences, all_tags = [], []
    tokens, tags = [], []
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line.startswith("###"):  # comment line written by prelabel.py
                continue
            if not line or line.startswith("-DOCSTART-"):
                if tokens:
                    sentences.append(tokens)
                    all_tags.append(tags)
                    tokens, tags = [], []
                continue
            parts = line.split()
            if len(parts) == 1 and re.fullmatch(r"O|[BI]-\w+", parts[0]):
                # A tag with no word: the word was an invisible character (e.g. a BOM).
                print(f"Warning: {path}:{line_no}: skipped a line with a tag but no word")
                continue
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_no}: expected 'token tag', got {line!r}")
            tokens.append(parts[0])
            tags.append(parts[-1])
    if tokens:
        sentences.append(tokens)
        all_tags.append(tags)
    return sentences, all_tags


def redact(text, entities, masks=MASKS):
    """Replace each detected entity with its mask, e.g. 'Ahmad' -> '[PERSON]'.

    `entities` are dicts with 'start', 'end' and 'entity_group' (as returned by the
    Hugging Face pipeline). We edit from the end of the string backwards so earlier
    character positions stay valid.
    """
    for e in sorted(entities, key=lambda e: e["start"], reverse=True):
        mask = masks.get(e["entity_group"], f"[{e['entity_group']}]")
        text = text[: e["start"]] + mask + text[e["end"]:]
    return text
