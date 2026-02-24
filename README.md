# Path Signatures for Stylistic Anomaly Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18710797.svg)](https://doi.org/10.5281/zenodo.18710797)

> **TL;DR** Given a stream of token embeddings from a text, can we detect whether its author has been seen before? We use **path signatures** of RoBERTa stream embeddings — a principled algebraic feature from rough path theory — combined with an **Isolation Forest** to achieve ROC AUC up to **0.84** on Project Gutenberg author anomaly detection.

---

![Pipeline Overview](output/pipeline_overview.svg)

---

## Overview

This repository implements an unsupervised **author anomaly detection** pipeline on the [Standardised Project Gutenberg Corpus](https://github.com/pgcorpus/gutenberg). The core idea is to represent each text as a *path* in embedding space: token representations (RoBERTa-large, top-K filtered, and dimensionality-reduced) form a time-ordered sequence of vectors. The **path signature** — a classical object from stochastic analysis — compactly encodes the sequential structure and iterated integrals of this path without discarding ordering information.

At inference time, anomaly scores from an Isolation Forest trained only on *seen* authors are aggregated across text chunks via a Kolmogorov–Smirnov test, yielding a book-level score. This avoids any dependency on author labels at test time and naturally handles variable-length texts.

---

## Results

**ROC AUC — Isolation Forest on path signatures** (impostor = Margaret Oliphant, held-out from training):

| Signature Level | dataset0 · UMAP 4d K=250 | dataset1 · UMAP 2d K=100 | dataset2 · RP 2d K=100 | dataset3 · RP 4d K=250 |
|:-:|:-:|:-:|:-:|:-:|
| L=1 | 0.8185 | 0.7127 | 0.3473 | 0.7467 |
| L=2 | 0.8161 | 0.7415 | 0.3510 | 0.7270 |
| L=3 | 0.8239 | 0.7675 | 0.3429 | 0.7192 |
| **L=4** | **0.8424** | **0.7730** | 0.3551 | **0.7351** |

UMAP with K=250 (dataset0) achieves the strongest separation. Random Projection with K=100 (dataset2) underperforms, consistent with the loss of geometric structure in low-dimensional random projections under small K.

---

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements_torch.txt  # or follow https://pytorch.org
```

**Core dependencies:** `iisignature`, `transformers`, `torch`, `umap-learn`, `scikit-learn`, `zarr`, `pandas`, `matplotlib`

---

## Data

### Corpus

```bash
cd gutenberg
pip install -r requirements.txt
python get_data.py        # Download Project Gutenberg corpus via rsync
python process_data.py    # Strip headers, tokenise → text/ tokens/ counts/
```

### Embeddings

Embeddings require a GPU and take several hours to compute. Set `DATASET_PATH` and `EMBEDDINGS_PATH` in the notebook, then call `compute_all_embeddings()` — or use the background script:

```bash
bash run_compute_embeddings.sh
```

Output: zarr arrays + `successful_embeddings.csv` saved under `EMBEDDINGS_PATH`.

### Precomputed Datasets (Recommended)

`embedding_datasets.pkl` (~14 GB) is available on Zenodo. Download and place at `gutenberg/data/embedding_datasets.pkl` to skip dataset construction:

```
DOI: 10.5281/zenodo.18710797
URL: https://zenodo.org/record/18710797
```

---

## Usage

1. Open `nlp_demo.ipynb`
2. Set `DATASET_PATH` and `EMBEDDINGS_PATH`
3. Run cells in order:
   - **Imports → Load metadata → Partition** normal / impostor splits
   - **Datasets** — build or load from `.pkl`
   - **Compute mean** → train UMAP (for UMAP configs) → **project embeddings**
   - **Token frequencies** → **encodings plots** → **anomaly evaluation** → **ROC plots**
4. Figures are saved to `output/`

---

## Experiment Configurations

| Config | Reduction | Dim | Top-K |
|--------|-----------|:---:|:-----:|
| dataset0 | UMAP | 4 | 250 |
| dataset1 | UMAP | 2 | 100 |
| dataset2 | Random Projection | 2 | 100 |
| dataset3 | Random Projection | 4 | 250 |

Token masking retains only embeddings at positions occupied by the top-K most frequent tokens, substantially reducing noise while preserving stylistically discriminative signal.

---

## Figures

### Token Frequency Distribution

Ranked token frequencies follow a steep Zipf decay, justifying the top-K masking strategy — most tokens after rank ~250 contribute negligible signal.

![Token Frequencies](output/token_frequencies.png)

---

### Embedding Projections

**UMAP (dataset1, K=100, 2d)** — Points coloured by dominant token. Tight per-token clusters confirm UMAP fidelity (KNN accuracy ≈ 0.955).

![UMAP Encodings](output/encodings_dataset1_umap.png)

**Random Projection (dataset2, K=100, 2d)** — Same colouring. Structure is noisier than UMAP, as expected from a non-metric linear projection.

![RP Encodings](output/encodings_dataset2_random_proj.png)

---

### ROC Curves

Each plot shows ROC curves for signature levels L=1–4. Higher level captures higher-order iterated integrals of the path, generally improving AUC until diminishing returns or overfitting on limited data.

| | |
|:---:|:---:|
| ![ROC dataset0](output/roc_dataset0_no_projection.png) | ![ROC dataset1](output/roc_dataset1_no_projection.png) |
| dataset0 · UMAP 4d K=250 **(best)** | dataset1 · UMAP 2d K=100 |
| ![ROC dataset2](output/roc_dataset2_no_projection.png) | ![ROC dataset3](output/roc_dataset3_no_projection.png) |
| dataset2 · RP 2d K=100 **(weakest)** | dataset3 · RP 4d K=250 |

---

## Project Structure

```
Path-Signature-for-Stylistic-Anomaly-Detection/
├── nlp_demo.ipynb                  # Main experiment notebook
├── compute_all_embeddings.py       # Standalone embedding script
├── check_token_lengths.py          # Token-length diagnostic
├── run_compute_embeddings.sh       # Background embedding runner
├── run_token_length_check.sh
├── test_embeddings.py
├── requirements.txt
├── requirements_torch.txt
├── output/
│   ├── pipeline_overview.svg
│   ├── token_frequencies.png
│   ├── encodings_dataset1_umap.png
│   ├── encodings_dataset2_random_proj.png
│   ├── roc_dataset0_no_projection.png
│   ├── roc_dataset1_no_projection.png
│   ├── roc_dataset2_no_projection.png
│   └── roc_dataset3_no_projection.png
└── gutenberg/
    ├── get_data.py
    ├── process_data.py
    └── src/
```

---

<details>
<summary>Data Flow & Drop Statistics</summary>

| Stage | Count | Dropped | Reason |
|-------|------:|--------:|--------|
| Metadata (all) | 77,640 | — | — |
| Author not null | 74,743 | 2,897 | No author in metadata |
| English only | 59,127 | 15,616 | `language != ['en']` |
| With embeddings | 34,789 | 24,338 | See breakdown below |
| df_normal_train | 34,147 | — | Normal authors, excl. eval |
| df_normal_eval | 510 | — | 1 per author (≥10 books) |
| df_impostor | 132 | — | Margaret Oliphant |

**Breakdown of 24,338 dropped (English → embeddings):**

| Reason | Count | % |
|--------|------:|:-:|
| No text/tokens in corpus | 22,796 | 93.7% |
| Has text/tokens, no embedding | 1,542 | 6.3% |

Books with text but no embedding were typically too short (<5,120 tokens, i.e. fewer than 10 full chunks). Mean token count: dropped ≈2,213 vs. kept ≈67,909.

</details>

---

## Citation

If you find this code useful, please cite:

```bibtex
@misc{he2025pathsignature,
  title        = {Path Signatures for Stylistic Anomaly Detection},
  author       = {He, Karen Siyu},
  year         = {2025},
  howpublished = {\url{https://github.com/KarenShark/Path-Signature-for-Stylistic-Anomaly-Detection}},
  note         = {Precomputed datasets: \doi{10.5281/zenodo.18710797}}
}
```

---

## References

- **Path Signatures**: Kidger, P. & Lyons, T. (2020). *Signatory: Differentiable computations of the signature and log-signature transforms.* ICLR 2021. [GitHub](https://github.com/patrick-kidger/signatory)
- **iisignature**: Reizenstein, J. (2017). *Calculation of iterated-integral signatures and log signatures.* [GitHub](https://github.com/bottler/iisignature)
- **Corpus**: Gerlach, M. & Font-Clos, F. (2020). *A standardized Project Gutenberg corpus.* PLOS ONE. [GitHub](https://github.com/pgcorpus/gutenberg)
- **Model**: Liu, Y. et al. (2019). *RoBERTa: A robustly optimized BERT pretraining approach.* [HuggingFace](https://huggingface.co/roberta-large)

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
