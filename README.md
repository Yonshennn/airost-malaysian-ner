# AIROST INTERNSHIP (ML/AI) — MALAYSIAN NER PROJECT

> Fine-tuned DistilBERT to recognise **people**, **locations** and **organisations** in Malaysian text — including Malay, English, Manglish and informal chat.

A standard English CoNLL-2003 baseline scores **F1 0.3857** on the clean Malaysian test set.
After fine-tuning with 992 Malaysian annotated sentences, **v5** reaches **F1 0.8582** — while keeping its English CoNLL-2003 score at **0.8968**.

```text
Input:  "Encik Izzat bekerja di RHB Bank di Ampang."

Output: PER  'Izzat'        score=1.00  chars 6-11
        ORG  'RHB Bank'     score=1.00  chars 23-31
        LOC  'Ampang'       score=1.00  chars 35-41
```

**Built for:** AIROST Technical Track Internship (ML/AI) — Problem Statement 2: *Personal Information Detection in Text (NER)*  
**Team:** Wokayyy  
**Author:** ([@Yonshennn](https://github.com/Yonshennn))

---

## Table of Contents

1. [Why This Project](#why-this-project)
2. [Results at a Glance](#results-at-a-glance)
3. [Quick Start](#quick-start)
4. [Model and Training](#model-and-training)
5. [Dataset](#dataset)
6. [Development History](#development-history)
7. [Project Files](#project-files)
8. [Limitations](#limitations)
9. [Reproducing the Model](#reproducing-the-model)
10. [Acknowledgements](#acknowledgements)

---

## Why This Project

Universities, hospitals and companies often need to remove personal information from incident reports and support chats before sharing them. Manual redaction is slow, and missing a single name can expose someone's identity.

Most NER models are trained on formal English news (such as CoNLL-2003). They struggle with:

- **Lowercase names** in chat messages — `"jumpa hafiz kat kl tmrw"`
- **Malay naming patterns** — `bin`, `binti`
- **Indian naming patterns** — `a/l`, `a/p`
- **Chinese multi-word names** — `Chong Wei Han`, `Carmen Chua Xin Yi`
- **Malaysian places and abbreviations** — `Taman Sentosa`, `KL`, `PJ`, `JB`, `TTDI`
- **Local organisations** — hospitals, clinics, schools, universities, banks and government bodies
- **Code-mixed Malay/English/Manglish** sentences

This project closes most of that gap by adapting `distilbert-base-cased` with locally annotated Malaysian training data.

> **Scope note:** This project identifies three entity types (`PER`, `LOC`, `ORG`). It is an NER system, not a complete PII detector — it does not currently detect IC numbers, phone numbers, email addresses or dates of birth.

---

## Results at a Glance

All Malaysian scores use **exact entity match** — a prediction counts only when both the span boundaries and the entity type match the labelled reference data.

### Baseline → v5 Improvement

| Test Set | Baseline F1 | **v5 F1** | Improvement |
|---|---:|---:|---:|
| **Clean Malaysian** (316 sentences · 537 entities) | 0.3857 | **0.8582** | +47.25 pts |
| Full/Hard Malaysian (426 sentences · 815 entities) | 0.4211 | **0.8803** | +45.92 pts |
| CoNLL-2003 English (`seqeval`) | 0.8860 | **0.8968** | +1.08 pts |

- **Clean Malaysian test** is the primary result: its entity surface forms do not appear in the training data, so it measures recognition on names, places and organisations absent from training. *(Note: the clean set was consulted during development and model selection, so it is not a completely untouched held-out test set.)*
- **Full/Hard Malaysian test** is reported as a supplementary evaluation and contains ~15.1% entity mention overlap with training (123 of 815 mentions), making its score slightly higher.
- **CoNLL-2003** is a separate English benchmark (`seqeval`) reported from training metadata and should not be merged directly with the Malaysian exact-match scores.

### v5 Per-Entity Performance (Clean Test)

| Entity | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| PER | 0.9020 | 0.9200 | **0.9109** | 200 |
| LOC | 0.8202 | 0.8848 | **0.8513** | 165 |
| ORG | 0.8497 | 0.7558 | **0.8000** | 172 |
| **Overall (micro)** | **0.8598** | **0.8566** | **0.8582** | **537** |

Person names (`PER`) are recognised most reliably. Organisation recognition (`ORG`) remains the weakest category — the model can still miss lowercase brand names, truncate multi-word organisations or confuse organisations with locations.

---

## Quick Start

### 1. Clone and install

Tested on **Windows** with **Python 3.12.6**, **PyTorch 2.15.0+cu132** (CUDA 13.2, NVIDIA GeForce GTX 1650), `transformers 5.22.2`, `datasets 5.1.0`, `accelerate 1.15.0`, `seqeval 1.2.2` and `pdfplumber 0.11.10`.

```powershell
git clone https://github.com/Yonshennn/airost-malaysian-ner.git
cd airost-malaysian-ner
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> For GPU training or GPU inference, install a PyTorch build compatible with your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/). Prediction can also fall back to CPU if CUDA is unavailable.

### 2. Download the trained model

Download the v5 model archive and extract it directly into the project root:

📥 **Model download:** [model_malaysian_v5.zip on Google Drive](https://drive.google.com/file/d/1IWFwWAZvJGpcwOM8iLsh03702Ar2it3b/view?usp=sharing)

The archive already contains the `model_malaysian_v5/` folder inside it — extract it so that `model_malaysian_v5/` sits directly alongside `predict.py` (avoid creating a nested `model_malaysian_v5/model_malaysian_v5/` folder).

### 3. Run predictions

```powershell
# Detect entities in a sentence
python predict.py "Ahmad works at Maybank in Kuala Lumpur." --model model_malaysian_v5

# Detect entities from a text or PDF file
python predict.py --file sample.txt --model model_malaysian_v5
python predict.py --file report.pdf --model model_malaysian_v5

# Redact detected entities
python predict.py "Ahmad works at Maybank in Kuala Lumpur." --model model_malaysian_v5 --redact
```

Example output with `--redact`:

```text
PER  'Ahmad'                          score=1.00  chars 0-5
ORG  'Maybank'                        score=1.00  chars 15-22
LOC  'Kuala Lumpur'                   score=1.00  chars 26-38

--- Redacted ---
[PERSON] works at [ORGANISATION] in [LOCATION].
```

PDF support extracts selectable embedded text using `pdfplumber`; scanned/image-only PDFs are not supported because OCR is not implemented.

### 4. Evaluate on test data

```powershell
python errors.py my_test_clean.conll --model model_malaysian_v5 --csv errors_clean_v5.csv
python errors.py my_test_hard.conll --model model_malaysian_v5 --csv errors_hard_v5.csv
```

---

## Model and Training

### Architecture

| | |
|---|---|
| **Base model** | `distilbert-base-cased` |
| **Task** | Token classification with BIO tagging |
| **Training framework** | Hugging Face Transformers `Trainer` |
| **Inference** | Hugging Face `pipeline` (`aggregation_strategy="first"`, `stride=64`) |

The model predicts a BIO label for each token: `B-` begins an entity, `I-` continues it, and `O` marks tokens outside any entity. During training, only the first subword piece of each word receives its label — subsequent subwords and special tokens receive `-100` so they are ignored in the loss calculation.

### v5 Training Configuration

| Setting | Value |
|---|---|
| Epochs | 5 |
| Batch size | 8 |
| Learning rate | `2e-5` |
| Weight decay | 0.01 |
| Random seed | 42 |
| Lowercase augmentation | 30% of CoNLL-2003 training sentences |
| Malaysian data repetition | ×3 (to balance against the larger CoNLL set) |
| Mixed precision (`fp16`) | Enabled when CUDA is available |
| GPU used | NVIDIA GeForce GTX 1650 (laptop GPU) |

Each model version was fine-tuned independently from the pretrained `distilbert-base-cased` checkpoint rather than continuing sequentially from an earlier Malaysian model checkpoint.

---

## Dataset

### Training Data

The training pipeline combines three sources:

| Source | Description |
|---|---|
| **CoNLL-2003** | General English NER dataset, loaded from the Hugging Face Hub |
| **Malaysian custom data** | 992 annotated sentences (`my_train_v5.conll`) |
| **Lowercase augmentation** | Lowercased copies of 30% of CoNLL-2003 training sentences |

Malaysian training data breakdown (`my_train_v5.conll`):

| Entity | Mentions |
|---|---:|
| PER | 613 |
| LOC | 643 |
| ORG | 643 |
| **Total** | **1,899** |

The 992 sentences combine an earlier 592-sentence set with 400 Round 5 additions. The Round 5 examples were generated with AI assistance, partially reviewed manually and cross-checked using Gemini as a secondary verification tool. Because not all 400 examples were verified individually by hand, residual annotation errors may remain.

### Annotation Rules

| Rule | Example |
|---|---|
| Titles stay **outside** PER | Encik `[Hafiz\|PER]`, Dr. `[Arun\|PER]` |
| `bin`/`binti` stay **inside** PER | `[Ahmad bin Ali\|PER]` |
| `a/l`/`a/p` stay **inside** PER | `[Kumar a/l Rajan\|PER]` |
| Chinese names = one PER span | `[Chong Wei Han\|PER]` |
| Hospitals, clinics, schools, banks = **ORG** | `[Hospital Selayang\|ORG]` |
| Towns, states, place abbreviations = **LOC** | `[Petaling Jaya\|LOC]`, `[KL\|LOC]` |
| Days, months, common words = **O** | `Isnin`, `Jumaat`, `Ogos`, `kami` |

### Evaluation Data

| File | Sentences | Entities | Description |
|---|---:|---:|---|
| `my_test_clean.conll` | 316 | 537 | Clean entity-surface-form test (no entity string shared with training) |
| `my_test_hard.conll` | 426 | 815 | Full/hard set including ~15.1% entity overlap with training |

> ⚠️ **Never add evaluation data to training.** If the training set is expanded, the entity-overlap check must be repeated and the clean test rebuilt from non-overlapping sources.

---

## Development History

All seven model versions were evaluated on the exact same clean and full/hard test sets (22 November 2026 consolidated evaluation):

| Version | Malaysian Sentences | Clean F1 | Full F1 | Key Change |
|---|---:|---:|---:|---|
| Baseline | — | 0.3857 | 0.4211 | CoNLL-2003 only |
| Lowercase | — | 0.3856 | 0.4114 | +30% lowercased CoNLL copies |
| v1 | 300 | 0.4024 | 0.4166 | First Malaysian data (noisy baseline pre-labels) |
| v2 | 300 | 0.6624 | 0.6771 | Corrected annotations; batch size reduced to 8 locally |
| v3 | 322 | 0.7348 | 0.7433 | Refined annotation rules (titles outside PER, hospitals ORG) |
| v4 | 592 | 0.8308 | 0.8453 | Broader names, places, abbreviations and organisations |
| **v5** | **992** | **0.8582** | **0.8803** | **+400 Round 5 examples targeting harder Malaysian cases** |

Key takeaways:

- **Baseline → v2 (+0.2767 F1):** Correcting annotations drove the largest single jump. Training on uncorrected baseline pre-labels (v1) gave minimal improvement.
- **v2 → v4 (+0.1684 F1):** Adding diverse local naming conventions, place abbreviations and organisation types steadily improved generalisation.
- **v4 → v5 (+0.0274 F1):** Targeted Round 5 additions raised `ORG` F1 from `0.7416` to `0.8000` on the clean set.
- **Platform transition:** Initial runs used Google Colab; after the free GPU quota was exhausted and session resets proved disruptive, training moved to a local Windows laptop (GTX 1650) with batch size reduced from 16 to 8.

---

## Project Files

### Core Scripts

| File | Purpose |
|---|---|
| `train_ner.py` | Fine-tunes the model, evaluates on CoNLL and custom test sets, and saves the model and `results.json` |
| `predict.py` | Runs NER on command-line text, `.txt` and text-based `.pdf` input, with optional `--redact` |
| `errors.py` | Compares predictions against labelled CoNLL data and exports mistakes to CSV |
| `prelabel.py` | Generates draft CoNLL/BIO annotations from raw sentences for manual review |
| `bracket_tool.py` | Converts between human-readable bracket annotations (e.g. `[Ahmad\|PER]`) and CoNLL/BIO format |
| `utils.py` | Shared CoNLL reader, entity filtering and redaction helpers |

### Data and Configuration

| File | Purpose |
|---|---|
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Excludes `.venv/`, model weights, checkpoints, caches and archives from Git |
| `my_train_v5.conll` | Final 992-sentence Malaysian training data (bracket and CoNLL/BIO formats) |
| `my_test_clean.conll` | Clean evaluation set (316 sentences / 537 entities) |
| `my_test_hard.conll` | Full/hard evaluation set (426 sentences / 815 entities) |

### Excluded from Git

Model weights (`model_*/`), training checkpoints (`checkpoint-*/`), `.venv/`, `__pycache__/`, and `*.zip` archives are excluded via `.gitignore`.

---

## Limitations

1. **Three entity types only** — `PER`, `LOC` and `ORG`. Does not detect IC numbers, phone numbers, email addresses or dates of birth.
2. **ORG is the weakest class** (`0.8000` F1 on clean test) — lowercase brands, abbreviated organisations and multi-word organisation boundaries are the main failure cases.
3. **Small evaluation set** — 537 entities means each individual mistake shifts F1 by ~0.2 percentage points.
4. **Test set reuse** — the clean test was used during development and model selection rather than kept strictly untouched.
5. **Partial entity overlap** — the full/hard test shares ~15.1% of entity mentions with training data.
6. **AI-assisted training additions** — Round 5 examples (400 of 992 sentences) received partial manual review and Gemini cross-verification; residual annotation errors may remain.
7. **No OCR** — PDF support handles selectable text only; scanned/image PDFs are not supported.
8. **Redaction is best-effort** — missed entities remain unmasked, so redacted outputs must be checked manually before sharing sensitive documents.

---

## Reproducing the Model

To run a fresh v5 reproduction from `distilbert-base-cased` (using a new output directory so existing weights are not overwritten):

```powershell
python train_ner.py `
  --extra_train my_train_v5.conll `
  --extra_test my_test_hard.conll `
  --lowercase_frac 0.3 `
  --epochs 5 `
  --batch_size 8 `
  --output_dir model_malaysian_v5_reproduction
```

*(Starting from `distilbert-base-cased` will not guarantee byte-identical weights across environments; `model_malaysian_v5/results.json` preserves the original training settings and evaluation metrics.)*

---

## Acknowledgements

- **[DistilBERT](https://huggingface.co/distilbert-base-cased)** — pretrained base transformer model
- **[CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/)** — English NER training and evaluation dataset
- **[Hugging Face Transformers](https://huggingface.co/docs/transformers/)** — token classification training and inference pipeline
- **AIROST internship mentors** — project guidance and feedback
