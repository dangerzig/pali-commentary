#!/usr/bin/env python3
"""
Experiment 5: Transformer-based Approaches.

  - Use multilingual sentence transformer to embed Pali text chunks
  - Average chunk embeddings per document
  - Cluster/classify on these embeddings
  - Compare with FastText embeddings
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from data_loader import load_raw_documents
from clustering import run_kmeans, run_spectral, run_gmm, run_agglomerative
from classification import run_loo_cv
from visualization import plot_umap, plot_pca, plot_confusion_comparison
from shared import get_data_dir


def compute_transformer_embeddings(documents, model_name='paraphrase-multilingual-MiniLM-L12-v2',
                                     chunk_size=200, batch_size=64):
    """Compute document embeddings using a sentence transformer.

    Splits each document into chunks, embeds each chunk,
    then averages chunk embeddings per document.
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    doc_embeddings = []
    for i, doc in enumerate(documents):
        # Split tokens into chunks and rejoin as text
        words = doc.tokens
        chunks = []
        for j in range(0, len(words), chunk_size):
            chunk_text = ' '.join(words[j:j+chunk_size])
            if chunk_text.strip():
                chunks.append(chunk_text)

        if not chunks:
            doc_embeddings.append(np.zeros(model.get_sentence_embedding_dimension()))
            continue

        # Encode all chunks for this document
        chunk_embeddings = model.encode(chunks, batch_size=batch_size,
                                         show_progress_bar=False)

        # Average chunk embeddings
        doc_embedding = np.mean(chunk_embeddings, axis=0)
        doc_embeddings.append(doc_embedding)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"    Embedded {i+1}/{len(documents)} documents ({len(chunks)} chunks)")

    return np.array(doc_embeddings)


def main():
    print("Experiment 5: Transformer-based Approaches")
    print("=" * 60)

    documents = load_raw_documents(data_dir=get_data_dir())
    file_ids = [d.file_id for d in documents]
    doc_labels = [d.label for d in documents]
    n_docs = len(documents)
    print(f"Loaded {n_docs} documents\n")

    results = {}

    # --- Multilingual MiniLM ---
    print("Computing transformer embeddings (paraphrase-multilingual-MiniLM-L12-v2)...")
    try:
        emb = compute_transformer_embeddings(documents)
        print(f"  Embedding matrix: {emb.shape}")

        # Clustering
        print("\nClustering on transformer embeddings:")
        for name, fn in [
            ('K-means', lambda: run_kmeans(emb, documents)),
            ('Spectral', lambda: run_spectral(emb, documents)),
            ('GMM', lambda: run_gmm(emb, documents)),
            ('Agglom. (ward)', lambda: run_agglomerative(emb, documents, 'ward')),
        ]:
            r = fn()
            acc = r.get('accuracy', 0)
            mis = r.get('misclassified', [])
            print(f"  {name}: accuracy={acc:.4f}, misclassified={len(mis)} {mis}")
            results[f'{name} (MiniLM)'] = acc

        # Supervised
        print("\nSupervised classification on transformer embeddings:")
        sup_results = run_loo_cv(emb, documents)
        for r in sup_results:
            print(f"  {r['method']}: accuracy={r['accuracy']:.4f}, misclassified={r['n_misclassified']}")
            results[f"{r['method']} (MiniLM)"] = r['accuracy']

        # Visualization
        r_km = run_kmeans(emb, documents)
        plot_umap(emb, doc_labels, file_ids,
                  title='UMAP: Multilingual Transformer Embeddings',
                  filename='exp5_umap_transformer.png',
                  misclassified=r_km['misclassified'])

        plot_pca(emb, doc_labels, file_ids,
                 title='PCA: Multilingual Transformer Embeddings',
                 filename='exp5_pca_transformer.png',
                 misclassified=r_km['misclassified'])

    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        print("Install with: pip install sentence-transformers torch")
    except Exception as e:
        print(f"ERROR: {e}")

    # --- Different chunk sizes ---
    if results:
        print("\n\nTrying different chunk sizes...")
        for cs in [100, 500]:
            try:
                emb = compute_transformer_embeddings(documents, chunk_size=cs)
                r = run_kmeans(emb, documents)
                print(f"  chunk_size={cs}: accuracy={r['accuracy']:.4f}, "
                      f"misclassified={r['n_misclassified']}")
                results[f'K-means (MiniLM chunk={cs})'] = r['accuracy']
            except Exception as e:
                print(f"  chunk_size={cs}: ERROR {e}")

    # --- Summary ---
    if results:
        print("\n\nSummary:")
        print(f"{'Method':<40} {'Accuracy':>10}")
        print("-" * 52)
        for method, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{method:<40} {acc:>10.4f}")

        plot_confusion_comparison(results,
                                   title='Experiment 5: Transformer Methods',
                                   filename='exp5_comparison.png')

    print("\nExperiment 5 complete.")


if __name__ == '__main__':
    main()
