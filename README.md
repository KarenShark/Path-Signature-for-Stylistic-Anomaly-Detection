# Path Signatures for Author Anomaly Detection in NLP

**Author anomaly detection** on Project Gutenberg: given a text, determine whether it was written by an author seen in training or by an unseen "impostor." Uses **RoBERTa stream embeddings** → **UMAP/Random Projection** → **Path Signatures** → **Isolation Forest**.

---

## What This Experiment Does (Summary)

We treat **text as a path in embedding space** and use **path signatures** as features for anomaly detection. The task: given a book, decide if it was written by a known author (normal) or by Margaret Oliphant (impostor, held out from training).

**Pipeline**: (1) RoBERTa encodes each book’s middle 10 chunks (512 tokens each) → 1024‑dim stream embeddings. (2) We mask to the Top‑K most frequent tokens (K=100 or 250) and reduce dimension with UMAP or Random Projection to 2d/4d. (3) Each chunk becomes a path; we compute path signatures at levels 1–4. (4) Isolation Forest is trained on normal signatures; KS test aggregates scores per book; ROC AUC measures impostor vs. normal separation.

**Four configs**: dataset0 (UMAP, K=250, 4d), dataset1 (UMAP, K=100, 2d), dataset2 (Random Projection, K=100, 2d), dataset3 (Random Projection, K=250, 4d). We compare UMAP vs Random Projection and K=100 vs K=250.

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

**Other options**:

| Option | How | Notes |
|--------|-----|-------|
| **Zenodo** | Upload at [zenodo.org](https://zenodo.org); get DOI; add download link to README | Free, 50 GB/record, persistent DOI |
| **Hugging Face Hub** | Create dataset repo; `huggingface-cli upload` | Free, good for ML datasets |
| **Split + merge** | `split -b 500M embedding_datasets.pkl part_`; upload parts to Drive/Dropbox; provide `cat part_* > embedding_datasets.pkl` | Works with any cloud storage |
| **Full pipeline** | Run get_data → process_data → compute_embeddings (GPU, ~hours) | No precomputed file needed |

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

**ROC AUC (no_projection, Isolation Forest)**:

| Level | dataset0 | dataset1 | dataset2 | dataset3 |
|-------|----------|----------|----------|----------|
| 0     | 0.71     | 0.75     | 0.72     | 0.72     |
| 1     | 0.74     | 0.73     | 0.74     | 0.74     |
| 2     | 0.77     | 0.72     | 0.77     | 0.77     |
| 3     | 0.77     | 0.74     | ~0.69*   | 0.84     |

\*dataset2 at level 3: ~69% with 100 tokens vs. 92% with 250 tokens.

---

## Output Figures

### token_frequencies.png

**What it shows**: Ranked token frequencies in the training corpus (Top 250).

**Interpretation**: Zipf-like distribution; the steep drop justifies using Top‑K masking. Tokens beyond ~100–250 add little signal.

![Token Frequencies](output/token_frequencies.png)

---

### encodings_dataset1_umap.png

**What it shows**: 2D UMAP projection of stream embeddings (dataset1: K=100, UMAP 2d). Points are colored by the dominant token at each position.

**Interpretation**: Same-token points cluster; UMAP preserves local structure. KNN accuracy ~0.955 indicates good fidelity. This supports using path signatures on these projections.

![UMAP Encodings Dataset1](output/encodings_dataset1_umap.png)

---

### encodings_dataset2_random_proj.png

**What it shows**: 2D Random Projection of stream embeddings (dataset2: K=100, RP 2d). Same coloring as above.

**Interpretation**: RP is linear and cheaper than UMAP; structure is noisier. Serves as a baseline to compare with UMAP.

![Random Projection Encodings Dataset2](output/encodings_dataset2_random_proj.png)

---

### roc_dataset0_no_projection.png … roc_dataset3_no_projection.png

**What they show**: ROC curves for each dataset at signature levels 1–4. X-axis: false positive rate; Y-axis: true positive rate. Each subplot = one level; legend shows IsoFor (Isolation Forest) AUC.

**Interpretation**: Higher AUC = better impostor vs. normal separation. dataset3 (RP, K=250, 4d) reaches ~0.84 at level 3; dataset2 (RP, K=100) is weaker (~0.69) due to fewer tokens. Level 3 often outperforms level 1–2; level 4 can overfit.

![ROC Dataset0](output/roc_dataset0_no_projection.png)

![ROC Dataset1](output/roc_dataset1_no_projection.png)

![ROC Dataset2](output/roc_dataset2_no_projection.png)

![ROC Dataset3](output/roc_dataset3_no_projection.png)

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
