# pali-commentary

Computational classification of canonical vs. commentarial texts in the Pali Sutta Pitaka.

This project supports the paper "Distinguishing Commentary from Canon in the Pali Sutta Pitaka: A Multi-Method Computational Analysis" (submitted to JPTS).

## Setup

```bash
pip install -r requirements.txt
```

Or with dev dependencies:

```bash
pip install -e ".[dev]"
```

## Data

Place VRI/CST4 raw text files in `data/raw/`. Files follow the naming convention:

- `s0101m.mul.txt` (canon)
- `s0101a.att.txt` (commentary)
- `s0101t.tik.txt` (subcommentary)
- `s0517m.nrf.txt` (paracanonical)

## Running experiments

Run all experiments:

```bash
make run-all
```

Or run individually:

```bash
python experiments/exp0_baseline.py    # Baseline word frequency analysis
python experiments/exp1_features.py    # Feature comparison
python experiments/exp2_clustering.py  # Unsupervised clustering
python experiments/exp3_supervised.py  # Supervised LOO-CV classification
python experiments/exp4_embeddings.py  # Word embedding features
python experiments/exp5_transformers.py  # Transformer embeddings (requires transformers)
python experiments/exp6_perplexity.py  # Perplexity analysis
python experiments/exp7_fourway.py     # Four-way classification
python experiments/exp8_jataka.py      # Jataka position analysis
python experiments/exp_lemmatized.py   # Lemmatized feature experiments
python experiments/exp_misclassification_analysis.py  # Boundary text analysis
```

Results (figures and tables) are saved to `results/`.

## Tests

```bash
make test
```

## Project structure

```
src/
  data_loader.py         # Load and tokenize VRI texts
  features.py            # Feature extraction (word freq, TF-IDF, distance)
  classification.py      # Supervised classifiers and LOO-CV
  clustering.py          # Unsupervised clustering methods
  visualization.py       # PCA, UMAP, dendrogram plots
  lemmatize.py           # Lemmatization wrapper
  parse_commentary.py    # Parse VRI raw files to JSON

experiments/
  shared.py              # Constants and helpers shared across experiments
  exp0_baseline.py       # Baseline analysis
  exp1_features.py       # Feature comparison
  exp2_clustering.py     # Clustering experiments
  exp3_supervised.py     # Supervised classification
  exp4_embeddings.py     # Word embedding features
  exp5_transformers.py   # Transformer embeddings
  exp6_perplexity.py     # Perplexity analysis
  exp7_fourway.py        # Four-way classification
  exp8_jataka.py         # Jataka analysis
  exp_lemmatized.py      # Lemmatized feature experiments
  exp_misclassification_analysis.py  # Boundary text analysis

tests/                   # Test suite
data/raw/                # VRI text files (not tracked)
results/                 # Output figures (not tracked)
```
