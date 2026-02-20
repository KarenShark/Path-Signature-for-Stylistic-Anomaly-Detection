# Path Signatures for Author Anomaly Detection in NLP

**Author anomaly detection** on Project Gutenberg: given a text, determine whether it was written by an author seen in training or by an unseen "impostor." Uses **RoBERTa stream embeddings** → **UMAP/Random Projection** → **Path Signatures** → **Isolation Forest**.

---

## Quick Start (Recommended: Use Precomputed Data)

**Run the notebook in under 5 minutes** if you have the precomputed `embedding_datasets.pkl`:

1. **Clone and setup**
   ```bash
   git clone https://github.com/KarenShark/Path-Signature_Anomaly-Detection.git
   cd Path-Signature_Anomaly-Detection
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements_torch.txt   # or: pip install torch from pytorch.org
   ```

2. **Download precomputed data** (Google Drive link TBD – will be added here)
   - Download `embedding_datasets.pkl` (~14 GB)
   - Place it in `gutenberg/data/embedding_datasets.pkl`

3. **Run the notebook**
   ```bash
   jupyter notebook nlp_demo.ipynb
   ```
   - Run all cells from top to bottom
   - The notebook will load from `embedding_datasets.pkl` and skip training
   - Figures are saved to `output/`

---

## Full Pipeline (From Scratch)

If you do not have the precomputed data, run the full pipeline. **Requires GPU** for embeddings (~several hours).

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
python data_download/get_data.py    # or: python get_data.py (rsync download, ~hours)
python process_data.py              # strip headers, tokenize → text/, tokens/, counts/
```

### 3. Compute embeddings (GPU required)

In the notebook or via script:

```bash
# Option A: run in notebook – execute the compute_all_embeddings() cell
# Option B: background script
./run_compute_embeddings.sh
```

Set `EMBEDDINGS_PATH` in the notebook to where embeddings are saved (zarr + `successful_embeddings.csv`).

### 4. Run notebook

Open `nlp_demo.ipynb`, set paths if needed, and run all cells in order.

---

## What the Experiment Does

1. **Data**: Project Gutenberg (English books). Normal = authors with ≥10 books; Impostor = Margaret Oliphant (held-out author).
2. **Embeddings**: RoBERTa-large encodes 10 chunks (512 tokens each) per book → stream embeddings (512×1024 per chunk).
3. **Dimensionality reduction**: UMAP or Random Projection, with Top-K token masking (K=100 or 250).
4. **Path signatures**: Truncation levels 1–4 on projected paths.
5. **Anomaly detection**: Isolation Forest on signatures; KS test aggregates scores across chunks per book; ROC AUC evaluates impostor vs. normal.

---

## Dataset Configurations

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

| Figure | Description |
|--------|-------------|
| `token_frequencies.png` | Token frequency distribution |
| `encodings_dataset1_umap.png` | UMAP 2D projection (dataset1, K=100) |
| `encodings_dataset2_random_proj.png` | Random Projection 2D (dataset2, K=100) |
| `roc_dataset0_no_projection.png` | ROC curves for dataset0 |
| `roc_dataset1_no_projection.png` | ROC curves for dataset1 |
| `roc_dataset2_no_projection.png` | ROC curves for dataset2 |
| `roc_dataset3_no_projection.png` | ROC curves for dataset3 |

![Token Frequencies](output/token_frequencies.png)

![UMAP Encodings](output/encodings_dataset1_umap.png)

![Random Projection Encodings](output/encodings_dataset2_random_proj.png)

![ROC Dataset0](output/roc_dataset0_no_projection.png)

![ROC Dataset2](output/roc_dataset2_no_projection.png)

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
│   ├── data/                # Put embedding_datasets.pkl here (or raw/text after pipeline)
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
