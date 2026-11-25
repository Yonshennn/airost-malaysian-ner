"""Find people, locations and organisations in raw text using your fine-tuned model.

    python predict.py "Ahmad from UTM visited Johor Bahru."
    python predict.py --file report.txt --redact
    python predict.py --file report.pdf --redact --model model_malaysian
"""
import argparse

from utils import KEEP_TYPES, redact


def read_text(path):
    """Read a .txt or .pdf file. Text extraction may use any library; detection is our model."""
    if path.lower().endswith(".pdf"):
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_ner(model_dir):
    import torch
    from transformers import pipeline

    # "first" takes each word's first-piece prediction, which matches how we trained.
    # stride lets the pipeline handle texts longer than 512 tokens.
    return pipeline(
        "token-classification",
        model=model_dir,
        aggregation_strategy="first",
        stride=64,
        device=0 if torch.cuda.is_available() else -1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("text", nargs="?", help="raw text (or use --file, or type it when asked)")
    parser.add_argument("--file", help=".txt or .pdf file to read")
    parser.add_argument("--model", default="model", help="folder saved by train_ner.py")
    parser.add_argument("--redact", action="store_true", help="also print the text with entities masked")
    args = parser.parse_args()

    if args.file:
        text = read_text(args.file)
    elif args.text:
        text = args.text
    else:
        text = input("Text: ")

    ner = load_ner(args.model)
    entities = [e for e in ner(text) if e["entity_group"] in KEEP_TYPES]
    entities.sort(key=lambda e: e["start"])

    if not entities:
        print("No personal information found.")
    for e in entities:
        surface = text[e["start"]:e["end"]]
        print(f"{e['entity_group']:<4} {surface!r:<32} score={float(e['score']):.2f}  chars {e['start']}-{e['end']}")

    if args.redact:
        print("\n--- Redacted ---")
        print(redact(text, entities))


if __name__ == "__main__":
    main()
