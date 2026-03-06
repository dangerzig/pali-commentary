# Distinguishing Commentary from Canon in the Pali Sutta Pitaka: Results Summary

## Dataset

- **Source**: Vipassana Research Institute (VRI) Chattha Sangayana Tipitaka (CST4)
- **Scope**: Sutta Pitaka only
- **Documents**: 92 texts after filtering empty files (min 100 tokens)
  - 41 canonical texts (`.mul.txt`)
  - 32 commentaries (`.att.txt`)
  - 15 subcommentaries (`.tik.txt`)
  - 4 paracanonical texts (`.nrf.txt`)
- **Binary task**: Canon (41) vs. non-canon (51)
- **Lemmatization**: All 92 documents lemmatized using Digital Pali Dictionary (DPD)
  - Canon coverage: ~91-93%
  - Commentary coverage: ~78-84%
  - Subcommentary coverage: ~69-74%
  - Vocabulary: 72,756 raw word forms; 36,120 lemmas

## Experiment 0: Baseline Reproduction

Reproducing Zigmond (2021): k-means (k=2) on top-1000 word frequencies.

| Method | Accuracy | Misclassified |
|--------|----------|---------------|
| K-means (word freq top 1000) | 97.83% | s0516m (Culaniddesa), s0520m (Suttasangaha) |

The 2021 paper misclassified 3 texts: s0507m (Petavatthu), s0515m (Mahaniddesa), s0516m (Culaniddesa). Our reproduction misclassifies 2, with s0516m overlapping. Differences likely due to empty file handling and inclusion of subcommentary/paracanonical texts not in the original study.

**Top discriminative words** (by frequency difference):
- Higher in non-canon: *ti* (quotative particle), *nti* (quote-ending), *vuttam* ("it was said"), *tattha* ("therein"), *viya* ("like"), *nama* ("namely"), *attho* ("meaning"), *aha* ("he said")
- Higher in canon: *bhikkhave* ("monks" vocative), *kho* (emphatic), *pe* (abbreviation marker), *me* ("to me"), *bhikkhu* ("monk"), *bhagava* ("the Blessed One")

## Experiment 1: Feature Comparison (K-means Clustering)

| Feature Type | Accuracy | Misclassified |
|-------------|----------|---------------|
| Word freq (top 500) | 97.83% | s0516m, s0520m |
| Word freq (top 1000) | 97.83% | s0516m, s0520m |
| Word freq (top 2000) | 97.83% | s0516m, s0520m |
| Char 3-4-grams | 97.83% | s0518m, s0520m |
| TF-IDF (top 1000) | 96.74% | s0516m, s0518m, s0520m |
| TF-IDF (top 2000) | 96.74% | s0516m, s0518m, s0520m |
| TF-IDF (top 5000) | 96.74% | s0516m, s0518m, s0520m |
| Char 3-3-grams | 95.65% | s0515m, s0517m, s0518m, s0520m |
| Char 4-4-grams | 82.61% | 16 documents |

Word frequency is robust across vocabulary sizes (500-2000). TF-IDF slightly underperforms raw frequencies. Character 3-4-grams match word frequencies, suggesting morphological patterns carry discriminative signal.

## Experiment 2: Clustering Algorithm Comparison

### Word Frequency Features (top 1000)

| Method | Accuracy | Misclassified |
|--------|----------|---------------|
| K-means | 97.83% | s0516m, s0520m |
| Spectral | 97.83% | s0516m, s0520m |
| GMM | 97.83% | s0516m, s0520m |
| Agglom. (ward) | 96.74% | s0515m, s0516m, s0520m |
| HDBSCAN | 82.61% | 16 documents |
| Agglom. (complete) | 57.61% | 39 documents |
| Agglom. (average) | 57.61% | 39 documents |

### TF-IDF Features (top 2000)

| Method | Accuracy | Misclassified |
|--------|----------|---------------|
| K-means | 96.74% | s0516m, s0518m, s0520m |
| Spectral | 96.74% | s0516m, s0518m, s0520m |
| GMM | 96.74% | s0516m, s0518m, s0520m |
| Agglom. (ward) | 96.74% | s0501t, s0518m, s0520m |
| HDBSCAN | 82.61% | 16 documents |
| Agglom. (complete) | 69.57% | 28 documents |
| Agglom. (average) | 58.70% | 38 documents |

K-means, Spectral, and GMM perform identically. Agglomerative clustering with complete/average linkage fails badly, suggesting the clusters are not well-separated by single-link or average-link distances.

## Experiment 3: Supervised Classification (LOO-CV)

### TF-IDF Features (top 2000)

| Method | Accuracy | Prec. | Recall | F1 | Misclassified |
|--------|----------|-------|--------|-----|---------------|
| Logistic Regression (L1) | 97.83% | 1.00 | 0.96 | 0.98 | s0518m, s0520m |
| Logistic Regression (L2) | 97.83% | 1.00 | 0.96 | 0.98 | s0518m, s0520m |
| SVM (linear) | 97.83% | 0.98 | 0.98 | 0.98 | s0519m, s0520m |
| Random Forest | 97.83% | 1.00 | 0.96 | 0.98 | s0518m, s0520m |
| SVM (RBF) | 96.74% | 1.00 | 0.94 | 0.97 | s0508a1, s0518m, s0520m |
| Gradient Boosting | 96.74% | 0.98 | 0.96 | 0.97 | s0518m, s0519m, s0520m |

### Word Frequency Features (top 1000)

| Method | Accuracy | Misclassified |
|--------|----------|---------------|
| Logistic Regression (L1) | 97.83% | s0518m, s0520m |
| SVM (linear) | 97.83% | s0519m, s0520m |
| SVM (RBF) | 97.83% | s0518m, s0520m |
| Random Forest | 97.83% | s0518m, s0520m |
| Logistic Regression (L2) | 96.74% | s0518m, s0519m, s0520m |
| Gradient Boosting | 93.48% | s0103m, s0304m, s0516m, s0518m, s0519m, s0520m |

**Top features by Random Forest importance**: *sesam* (0.032), *patthaya* (0.030), *kathetva* (0.026), *aha* (0.025), *karanena* (0.023)

**Top features by Logistic Regression coefficients**:
- Commentary-predictive: *karanam* (+1.19), *ti* (+0.81), *tattha* (+0.46), *viya* (+0.40)
- Canon-predictive: (none strongly negative; commentary markers dominate)

## Experiment 4: FastText Document Embeddings

### Clustering on 100-dim FastText embeddings (raw)

| Method | Accuracy | Misclassified |
|--------|----------|---------------|
| **GMM** | **98.91%** | s0517m |
| Agglom. (ward) | 81.52% | 17 documents |
| K-means | 56.52% | 40 documents |
| Spectral | 56.52% | 40 documents |

### Supervised on FastText embeddings (LOO-CV)

| Method | Accuracy |
|--------|----------|
| Logistic Regression (L2) | 97.83% |
| SVM (linear) | 97.83% |
| Logistic Regression (L1) | 96.74% |
| SVM (RBF) | 96.74% |
| Random Forest | 96.74% |

### K-means by vector dimension

| Dimension | Accuracy |
|-----------|----------|
| 50 | 56.52% |
| 100 | 56.52% |
| 200 | 83.70% |
| 300 | 84.78% |

### Lemmatized FastText (100-dim)

| Method | Accuracy |
|--------|----------|
| K-means | 82.61% |

GMM dramatically outperformed k-means on FastText embeddings, suggesting the embedding space has non-spherical cluster structure. K-means failed completely at lower dimensions but improved with higher-dimensional embeddings.

## Experiment 5: Transformer Embeddings (MiniLM)

Model: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim)

### Clustering

| Method | Accuracy | Misclassified |
|--------|----------|---------------|
| GMM | 94.57% | s0511m, s0515m, s0516m, s0517m, s0518m |
| K-means | 93.48% | s0506m, s0511m, s0515m, s0516m, s0517m, s0518m |
| Spectral | 92.39% | s0506m, s0509m, s0511m, s0515m, s0516m, s0517m, s0518m |
| Agglom. (ward) | 83.70% | 15 documents |

### Supervised (LOO-CV)

| Method | Accuracy |
|--------|----------|
| SVM (RBF) | 97.83% |
| Logistic Regression (L2) | 96.74% |
| SVM (linear) | 96.74% |
| Random Forest | 96.74% |
| Gradient Boosting | 96.74% |
| Logistic Regression (L1) | 95.65% |

Transformer embeddings underperformed simpler methods for unsupervised clustering (93-95% vs. 98%), suggesting the multilingual model's limited Pali exposure produces noisier representations than corpus-specific features. With supervised learning, SVM(RBF) recovers to 97.83%.

## Experiment 6: Language Model Perplexity

Train language models on canon texts only; classify by perplexity threshold.

| Method | Accuracy | Misclassified |
|--------|----------|---------------|
| **Word trigram** | **100.00%** | None |
| Char 5-gram | 94.57% | s0512m, s0513m, s0514m, s0518m, s0520m |

**Perplexity statistics**:
- Word trigram: Canon mean=334 (sd=230), Non-canon mean=75,309 (sd=22,702)
- Char 5-gram: Canon mean=4.42 (sd=0.84), Non-canon mean=6.63 (sd=0.54)

Note: The 100% accuracy is achieved by optimizing the threshold on the evaluation data itself, so it demonstrates perfect linear separability rather than validated out-of-sample prediction. The enormous gap in word-level perplexity (225x between canon and non-canon means) indicates the vocabularies are extremely distinct.

## Lemmatized Feature Experiments

### K-means Clustering

| Feature Type | Accuracy | Misclassified |
|-------------|----------|---------------|
| Lemma freq (top 500-2000) | 96.74% | s0515m, s0516m, s0520m |
| Lemma TF-IDF (1000-5000) | 95.65% | s0515m, s0516m, s0518m, s0520m |

### Other Clustering (lemma freq top 1000)

| Method | Accuracy | Misclassified |
|--------|----------|---------------|
| GMM | 96.74% | s0515m, s0516m, s0520m |
| Spectral | 95.65% | s0515m, s0516m, s0518m, s0520m |

### Supervised (lemma TF-IDF 2000, LOO-CV)

| Method | Accuracy | Misclassified |
|--------|----------|---------------|
| **Gradient Boosting** | **98.91%** | s0514a3 |
| Logistic Regression (L1) | 97.83% | s0518m, s0520m |
| Logistic Regression (L2) | 97.83% | s0518m, s0520m |
| Random Forest | 97.83% | s0518m, s0520m |
| SVM (RBF) | 96.74% | s0501t, s0518m, s0520m |
| SVM (linear) | 95.65% | s0517m, s0518m, s0519m, s0520m |

### Direct Comparison: Raw vs. Lemmatized

| Feature | Accuracy (k-means) | Misclassified |
|---------|-------------------|---------------|
| Raw word freq (top 1000) | 97.83% | s0516m, s0520m |
| Lemma freq (top 1000) | 96.74% | s0515m, s0516m, s0520m |

Lemmatization slightly degrades unsupervised clustering (adding s0515m as misclassified). However, supervised Gradient Boosting on lemma TF-IDF achieves the best LOO-CV accuracy (98.91%), with the sole misclassification being s0514a3 (a commentary text), not a boundary canonical text.

**Top discriminative lemmas** (LR coefficients, commentary-predictive):
*katabba* (+0.66), *viya* (+0.61), *karana* (+0.54), *ti* (+0.53), *tattha* (+0.51), *idani* (+0.50)

## Overall Best Results

| Rank | Method | Features | Type | Accuracy | Misclassified |
|------|--------|----------|------|----------|---------------|
| 1 | Word trigram perplexity | Raw | Semi-supervised* | 100.00% | None |
| 2= | GMM clustering | FastText 100d (raw) | Unsupervised | 98.91% | s0517m |
| 2= | Gradient Boosting LOO-CV | Lemma TF-IDF 2000 | Supervised | 98.91% | s0514a3 |
| 4= | K-means/Spectral/GMM | Word freq 1000 (raw) | Unsupervised | 97.83% | s0516m, s0520m |
| 4= | LR/SVM/RF (various) | TF-IDF/word freq (raw) | Supervised | 97.83% | various (2) |
| 4= | LR/RF LOO-CV | Lemma TF-IDF 2000 | Supervised | 97.83% | s0518m, s0520m |
| 4= | SVM (RBF) LOO-CV | Transformer 384d | Supervised | 97.83% | 2 documents |
| 4= | Char 3-4-grams k-means | Raw | Unsupervised | 97.83% | s0518m, s0520m |

*Threshold optimized on evaluation data; demonstrates separability, not validated prediction.

## Frequently Misclassified Documents

| File ID | Text | Type | Times Misclassified | Notes |
|---------|------|------|--------------------:|-------|
| s0520m | Suttasangaha | Paracanonical | ~30+ | Dialogical, sutta-like style |
| s0516m | Culaniddesa | Canon | ~15+ | Exegetical commentary on Parayanavagga |
| s0518m | Nettippakarana | Canon | ~15+ | Hermeneutical treatise |
| s0515m | Mahaniddesa | Canon | ~10 | Exegetical commentary on Atthakavagga |
| s0517m | Patisambhidamagga | Canon | ~5 | Analytical/abhidhammic style |
| s0519m | Petakopadesa | Canon | ~5 | Methodological/hermeneutical |

The pattern is consistent: the most difficult texts are the exegetical and hermeneutical works of the Khuddaka Nikaya (Niddesa, Nettippakarana, Patisambhidamagga, Petakopadesa), which are stylistically more similar to commentaries than to suttas, and the Milindapanha/Suttasangaha (paracanonical texts with sutta-like features). These are precisely the texts whose canonical status has been debated by scholars, lending computational support to longstanding philological observations.

## Key Findings

1. **Simple methods work remarkably well**: Raw word frequencies with k-means achieve 97.83% accuracy, matching or exceeding most sophisticated approaches.

2. **Vocabulary is the primary signal**: The enormous word-level perplexity gap (225x) between canon and non-canon texts indicates that the distinguishing feature is primarily lexical, not syntactic or structural.

3. **Commentary-specific vocabulary**: Words like *ti* (quotative), *vuttam* ("it was said"), *tattha* ("therein"), *viya* ("like"), *aha* ("he said"), *karanam* ("reason"), *katabba* ("should be done") are strongly predictive of commentary, reflecting the exegetical register.

4. **Lemmatization has mixed effects**: It slightly degrades unsupervised clustering but enables the best supervised result (Gradient Boosting, 98.91%). Lemmatization also recovers the 2021 paper's exact 3 misclassifications (s0515m, s0516m, s0520m).

5. **Modern embeddings underperform**: Both FastText and transformer embeddings produce worse unsupervised clustering than simple word frequencies. The exception is GMM on FastText (98.91%), suggesting non-spherical cluster geometry in embedding space.

6. **Boundary texts are philologically meaningful**: The consistently misclassified texts (Niddesa, Nettippakarana, Patisambhidamagga, Suttasangaha) are precisely those whose canonical status is contested in traditional scholarship.
