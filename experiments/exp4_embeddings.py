#!/usr/bin/env python3
"""
Experiment 4: Document Embeddings with FastText.

  - Train FastText on the full Pali corpus
  - Compute document embeddings by averaging word vectors
  - Cluster/classify using these embeddings
  - UMAP visualization
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from data_loader import load_raw_documents, load_lemmatized_documents
from clustering import run_kmeans, run_spectral, run_gmm, run_agglomerative
from classification import run_loo_cv
from visualization import plot_umap, plot_pca, plot_confusion_comparison


def train_fasttext_model(documents, use_lemmas=False, vector_size=100,
                          window=5, min_count=5, epochs=20):
    """Train a FastText model on the corpus."""
    from gensim.models import FastText

    # Prepare sentences (list of word lists)
    sentences = []
    for doc in documents:
        words = doc.lemmas if (use_lemmas and doc.lemmas) else doc.tokens
        # Split into chunks of ~100 words to create more training examples
        chunk_size = 100
        for i in range(0, len(words), chunk_size):
            chunk = words[i:i+chunk_size]
            if chunk:
                sentences.append(chunk)

    print(f"  Training FastText on {len(sentences)} sentences...")
    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        workers=4,
        sg=1,  # skip-gram
    )
    print(f"  Vocabulary size: {len(model.wv)}")
    return model


def document_embeddings(documents, model, use_lemmas=False):
    """Compute document embeddings by averaging word vectors."""
    embeddings = []
    for doc in documents:
        words = doc.lemmas if (use_lemmas and doc.lemmas) else doc.tokens
        vectors = []
        for word in words:
            try:
                vectors.append(model.wv[word])
            except KeyError:
                continue
        if vectors:
            embeddings.append(np.mean(vectors, axis=0))
        else:
            embeddings.append(np.zeros(model.wv.vector_size))
    return np.array(embeddings)


def main():
    print("Experiment 4: Document Embeddings (FastText)")
    print("=" * 60)

    documents = load_raw_documents()
    file_ids = [d.file_id for d in documents]
    doc_labels = [d.label for d in documents]
    n_docs = len(documents)
    print(f"Loaded {n_docs} documents\n")

    results = {}

    # --- Train FastText on raw tokens ---
    print("Training FastText on raw word forms...")
    model_raw = train_fasttext_model(documents, use_lemmas=False,
                                      vector_size=100, epochs=20)

    print("Computing document embeddings...")
    emb_raw = document_embeddings(documents, model_raw, use_lemmas=False)
    print(f"  Embedding matrix: {emb_raw.shape}")

    # Clustering
    print("\nClustering on raw FastText embeddings:")
    for name, fn in [
        ('K-means', lambda: run_kmeans(emb_raw, documents)),
        ('Spectral', lambda: run_spectral(emb_raw, documents)),
        ('GMM', lambda: run_gmm(emb_raw, documents)),
        ('Agglom. (ward)', lambda: run_agglomerative(emb_raw, documents, 'ward')),
    ]:
        r = fn()
        acc = r.get('accuracy', 0)
        mis = r.get('misclassified', [])
        print(f"  {name}: accuracy={acc:.4f}, misclassified={len(mis)} {mis}")
        results[f'{name} (FastText raw)'] = acc

    # Supervised (LOO-CV)
    print("\nSupervised classification on raw FastText embeddings:")
    sup_results = run_loo_cv(emb_raw, documents)
    for r in sup_results:
        print(f"  {r['method']}: accuracy={r['accuracy']:.4f}, misclassified={r['n_misclassified']}")
        results[f"{r['method']} (FastText raw)"] = r['accuracy']

    # UMAP visualization
    r = run_kmeans(emb_raw, documents)
    plot_umap(emb_raw, doc_labels, file_ids,
              title='UMAP: FastText Document Embeddings (raw)',
              filename='exp4_umap_fasttext_raw.png',
              misclassified=r['misclassified'])

    plot_pca(emb_raw, doc_labels, file_ids,
             title='PCA: FastText Document Embeddings (raw)',
             filename='exp4_pca_fasttext_raw.png',
             misclassified=r['misclassified'])

    # --- Different vector sizes ---
    print("\n\nTrying different vector sizes...")
    for vs in [50, 200, 300]:
        model = train_fasttext_model(documents, use_lemmas=False,
                                      vector_size=vs, epochs=15)
        emb = document_embeddings(documents, model, use_lemmas=False)
        r = run_kmeans(emb, documents)
        print(f"  dim={vs}: accuracy={r['accuracy']:.4f}, misclassified={r['n_misclassified']}")
        results[f'K-means (FastText d={vs})'] = r['accuracy']

    # --- Lemmatized FastText if available ---
    print("\n\nChecking for lemmatized data...")
    docs_lemma = load_lemmatized_documents(raw_docs=documents)
    n_with_lemmas = sum(1 for d in docs_lemma if len(d.lemmas) > 0)

    if n_with_lemmas == n_docs:
        print(f"All {n_docs} documents have lemmatized versions.")
        print("Training FastText on lemmas...")
        model_lemma = train_fasttext_model(documents, use_lemmas=True,
                                            vector_size=100, epochs=20)
        emb_lemma = document_embeddings(documents, model_lemma, use_lemmas=True)

        print("\nClustering on lemmatized FastText embeddings:")
        r = run_kmeans(emb_lemma, documents)
        print(f"  K-means: accuracy={r['accuracy']:.4f}, misclassified={r['n_misclassified']}")
        results['K-means (FastText lemma)'] = r['accuracy']

        plot_umap(emb_lemma, doc_labels, file_ids,
                  title='UMAP: FastText Document Embeddings (lemmatized)',
                  filename='exp4_umap_fasttext_lemma.png',
                  misclassified=r['misclassified'])
    else:
        print(f"Only {n_with_lemmas}/{n_docs} have lemmatized data. Skipping.")

    # --- Summary ---
    print("\n\nSummary:")
    print(f"{'Method':<40} {'Accuracy':>10}")
    print("-" * 52)
    for method, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{method:<40} {acc:>10.4f}")

    plot_confusion_comparison(results,
                               title='Experiment 4: FastText Embedding Methods',
                               filename='exp4_comparison.png')

    print("\nExperiment 4 complete.")


if __name__ == '__main__':
    main()
