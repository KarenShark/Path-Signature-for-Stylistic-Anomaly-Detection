# Path Signatures for Author Anomaly Detection in NLP

**Author anomaly detection** on Project Gutenberg: given a text, determine whether it was written by an author seen in training or by an unseen "impostor." Uses **RoBERTa stream embeddings** → **UMAP/Random Projection** → **Path Signatures** → **Isolation Forest**.

---

## Contents

- [What This Experiment Does](#what-this-experiment-does-summary)
- [Quick Start (Precomputed Data)](#quick-start-precomputed-data)
- [Precomputed Data Options](#precomputed-data-options)
- [Full Pipeline (From Scratch)](#full-pipeline-from-scratch)
- [Experiment Details](#experiment-details)
- [Reproduced Results](#reproduced-results)
- [Output Figures](#output-figures)
- [Project Structure](#project-structure)
- [References](#references)

---

## What This Experiment Does (Summary)

**Task**: Given a book, classify as **normal** (known authors) or **impostor** (Margaret Oliphant, held out).

### Pipeline (ML perspective)

![Pipeline Overview](output/pipeline_overview.svg)

| Stage | Input | Output |
|-------|-------|--------|
| **Data** | PG books | 10 chunks x 512 tokens per book |
| **Embeddings** | Chunks | RoBERTa to (512, 1024) stream |
| **Projection** | Stream | UMAP/RP to 2d or 4d path |
| **Features** | Path | Path signature (levels 1-4) |
| **Train** | Normal signatures | Isolation Forest |
| **Eval** | Impostor + normal | ROC AUC |


### Configs

| Config | Top-K | Projection | Dim |
|--------|-------|------------|-----|
| dataset0 | 250 | UMAP | 4 |
| dataset1 | 100 | UMAP | 2 |
| dataset2 | 100 | Random Projection | 2 |
| dataset3 | 250 | Random Projection | 4 |

---

## Quick Start (Precomputed Data)

**Run the notebook in ~5 minutes** if you have `embedding_datasets.pkl`:

1. **Clone and setup**
   ```bash
   git clone https://github.com/KarenShark/Path-Signature_Anomaly-Detection.git
   cd Path-Signature_Anomaly-Detection
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements_torch.txt   # or: pip install torch from pytorch.org
   ```

2. **Obtain precomputed data** (~14 GB) — see [Precomputed Data Options](#precomputed-data-options) below.

3. **Run the notebook**
   ```bash
   jupyter notebook nlp_demo.ipynb
   ```
   Run all cells; the notebook loads `embedding_datasets.pkl` and skips training. Figures are saved to `output/`.

---

## Precomputed Data Options

`embedding_datasets.pkl` (~14 GB) cannot be hosted on GitHub.

**Download from Zenodo** (recommended):

- **URL**: [https://zenodo.org/record/18710797](https://zenodo.org/record/18710797)
- **DOI**: [10.5281/zenodo.18710797](https://doi.org/10.5281/zenodo.18710797)

Place the file at `gutenberg/data/embedding_datasets.pkl` after download.

---

## Full Pipeline (From Scratch)

**Requires GPU** for embeddings (~several hours).

### 1. Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_torch.txt
```

**Key deps**: `iisignature`, `transformers`, `torch`, `umap-learn`, `scikit-learn`, `zarr`, `pandas`, `matplotlib`.

### 2. Download Gutenberg corpus

```bash
cd gutenberg
pip install -r requirements.txt
python data_download/get_data.py    # rsync download (~hours)
python process_data.py              # strip headers, tokenize → text/, tokens/, counts/
```

### 3. Compute embeddings (GPU required)

Run `compute_all_embeddings()` in the notebook or `./run_compute_embeddings.sh`. Set `EMBEDDINGS_PATH` to the output directory (zarr + `successful_embeddings.csv`).

### 4. Run notebook

Open `nlp_demo.ipynb`, set paths if needed, run all cells in order.

---

## Experiment Details

### Data and splits

- **Corpus**: Project Gutenberg (English only).
- **Normal**: Authors with ≥10 books; train/eval split (one book per author in eval).
- **Impostor**: Margaret Oliphant, held out entirely.

### Embeddings and paths

- Each book: middle 5120 tokens → 10 chunks of 512.
- RoBERTa-large: each chunk → (512, 1024) stream embedding.
- **Token mask**: Keep only positions of the Top‑K most frequent tokens in training.
- **Projection**: UMAP (fit on 2000 chunks, MLP for new data) or Random Projection to 2d/4d.
- Each chunk → one path; path signature at levels 1–4.

### Anomaly detection

- Isolation Forest on normal signatures.
- Per‑book: KS test between normal and impostor score distributions → p‑value as anomaly score.
- ROC AUC: impostor vs. normal.

### Configurations

| Config   | Top-K | Reduction        | Dim |
|----------|-------|------------------|-----|
| dataset0 | 250  | UMAP             | 4   |
| dataset1 | 100  | UMAP             | 2   |
| dataset2 | 100  | Random Projection| 2   |
| dataset3 | 250  | Random Projection| 4   |

---

## Reproduced Results

**KNN accuracy (UMAP fidelity)**: ~0.955

**ROC AUC (no_projection, Isolation Forest)** — from `output/roc_auc_metrics.json`:

| Level | dataset0 | dataset1 | dataset2 | dataset3 |
|-------|----------|----------|----------|----------|
| 1     | 0.8185   | 0.7127   | 0.3473   | 0.7467   |
| 2     | 0.8161   | 0.7415   | 0.3510   | 0.7270   |
| 3     | 0.8239   | 0.7675   | 0.3429   | 0.7192   |
| 4     | 0.8424   | 0.7730   | 0.3551   | 0.7351   |

*dataset0 (UMAP, K=250, 4d) has the highest AUC; dataset2 (RP, K=100, 2d) is weakest (~0.35).

---

## Output Figures

### 1. token_frequencies.png

Ranked token frequencies in the training corpus (Top 250). Zipf-like distribution; steep drop justifies Top‑K masking.

![Token Frequencies](output/token_frequencies.png)

---

### 2. encodings_dataset1_umap.png

2D UMAP projection of stream embeddings (dataset1: K=100, UMAP 2d). Points colored by dominant token. Same-token points cluster; KNN fidelity ~0.955.

![UMAP Encodings Dataset1](output/encodings_dataset1_umap.png)

---

### 3. encodings_dataset2_random_proj.png

2D Random Projection of stream embeddings (dataset2: K=100, RP 2d). Same coloring. RP is cheaper than UMAP; structure noisier.

![Random Projection Encodings Dataset2](output/encodings_dataset2_random_proj.png)

---

### 4. ROC curves (dataset0–3)

Each figure: ROC curves at signature levels 1–4 (subplots). X=FPR, Y=TPR. Legend shows IsoFor AUC. Higher AUC = better impostor vs normal separation.

| Figure | Config | Top-K | Reduction | Dim |
|--------|--------|-------|-----------|-----|
| roc_dataset0_no_projection.png | dataset0 | 250 | UMAP | 4 |
| roc_dataset1_no_projection.png | dataset1 | 100 | UMAP | 2 |
| roc_dataset2_no_projection.png | dataset2 | 100 | Random Projection | 2 |
| roc_dataset3_no_projection.png | dataset3 | 250 | Random Projection | 4 |

![ROC Dataset0](output/roc_dataset0_no_projection.png)  
*dataset0: UMAP K=250, 4d*

![ROC Dataset1](output/roc_dataset1_no_projection.png)  
*dataset1: UMAP K=100, 2d*

![ROC Dataset2](output/roc_dataset2_no_projection.png)  
*dataset2: Random Projection K=100, 2d*

![ROC Dataset3](output/roc_dataset3_no_projection.png)  
*dataset3: Random Projection K=250, 4d*

---

## Project Structure

```
Path-Signature_Anomaly-Detection/
├── nlp_demo.ipynb           # Main notebook
├── requirements.txt
├── requirements_torch.txt
├── output/                  # Figures
├── gutenberg/
│   ├── metadata/metadata.csv
│   ├── data/                # embedding_datasets.pkl or raw/text after pipeline
│   ├── get_data.py
│   ├── process_data.py
│   ├── data_download/
│   └── src/
├── compute_all_embeddings.py
├── run_compute_embeddings.sh
└── PIPELINE_SUMMARY.md
```

---

## Pipeline Overview

```
raw (.txt) → text (.txt) → compute_embeddings (RoBERTa) → embeddings zarr
                                    ↓
    df filter (English + has embedding) + partition normal/impostor
                                    ↓
    Token mask (Top-K) + UMAP / Random Projection
                                    ↓
    Path Signature (level 1–4) → Isolation Forest → ROC AUC
```

---

## References

- Path Signatures: [iisignature](https://github.com/patrick-kidger/signatory)
- Corpus: [Standardised Project Gutenberg Corpus](https://github.com/pgcorpus/gutenberg)
- Model: [RoBERTa-large](https://huggingface.co/roberta-large)
