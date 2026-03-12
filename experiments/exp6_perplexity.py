#!/usr/bin/env python3
"""
Experiment 6: Language Model Perplexity.

  - Train character-level and word-level n-gram language models on canon texts only
  - Measure perplexity on all documents
  - Test whether commentary texts have systematically higher perplexity
  - Fully unsupervised approach (no labels needed at classification time)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import math
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import load_raw_documents, binary_labels
from visualization import RESULTS_DIR, LABEL_COLORS
from shared import get_data_dir


class CharNgramLM:
    """Character-level n-gram language model with Laplace smoothing."""

    def __init__(self, n=5, alpha=1.0):
        self.n = n
        self.alpha = alpha  # smoothing parameter
        self.counts = defaultdict(Counter)
        self.vocab = set()

    def train(self, texts: list[str]):
        """Train on a list of text strings."""
        for text in texts:
            text = '<' + text + '>'  # add boundary markers
            self.vocab.update(text)
            for i in range(len(text) - self.n + 1):
                context = text[i:i+self.n-1]
                char = text[i+self.n-1]
                self.counts[context][char] += 1

    def log_prob(self, text: str) -> float:
        """Compute log probability of text."""
        text = '<' + text + '>'
        V = len(self.vocab) + 1  # +1 for unknown chars
        total_log_prob = 0.0
        n_chars = 0

        for i in range(len(text) - self.n + 1):
            context = text[i:i+self.n-1]
            char = text[i+self.n-1]
            count = self.counts[context][char] + self.alpha
            total = sum(self.counts[context].values()) + self.alpha * V
            total_log_prob += math.log2(count / total)
            n_chars += 1

        return total_log_prob, n_chars

    def perplexity(self, text: str) -> float:
        """Compute perplexity of text."""
        log_prob, n_chars = self.log_prob(text)
        if n_chars == 0:
            return float('inf')
        return 2 ** (-log_prob / n_chars)


class WordNgramLM:
    """Word-level n-gram language model with Laplace smoothing."""

    def __init__(self, n=3, alpha=0.01):
        self.n = n
        self.alpha = alpha
        self.counts = defaultdict(Counter)
        self.vocab = set()

    def train(self, token_lists: list[list[str]]):
        """Train on lists of token sequences."""
        for tokens in token_lists:
            tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
            self.vocab.update(tokens)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i+self.n-1])
                word = tokens[i+self.n-1]
                self.counts[context][word] += 1

    def perplexity(self, tokens: list[str]) -> float:
        """Compute perplexity of a token sequence."""
        tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
        V = len(self.vocab) + 1
        total_log_prob = 0.0
        n_tokens = 0

        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i+self.n-1])
            word = tokens[i+self.n-1]
            count = self.counts[context][word] + self.alpha
            total = sum(self.counts[context].values()) + self.alpha * V
            total_log_prob += math.log2(count / total)
            n_tokens += 1

        if n_tokens == 0:
            return float('inf')
        return 2 ** (-total_log_prob / n_tokens)


def plot_perplexity_distribution(documents, perplexities, title, filename):
    """Plot perplexity by document, colored by label."""
    fig, ax = plt.subplots(figsize=(14, 6))

    x = range(len(documents))
    colors = [LABEL_COLORS.get(d.label, '#999999') for d in documents]

    bars = ax.bar(x, perplexities, color=colors, alpha=0.8, width=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([d.file_id for d in documents], rotation=90, fontsize=6)
    ax.set_ylabel('Perplexity')
    ax.set_title(title)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l)
                       for l, c in LABEL_COLORS.items()
                       if any(d.label == l for d in documents)]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    outpath = RESULTS_DIR / filename
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")


def classify_by_threshold(perplexities, documents):
    """Find optimal perplexity threshold for classification."""
    true_labels = np.array(binary_labels(documents))
    perps = np.array(perplexities)

    best_acc = 0
    best_threshold = 0

    # Try various thresholds
    for threshold in np.linspace(perps.min(), perps.max(), 1000):
        # Higher perplexity -> more likely non-canon (class 1)
        predicted = (perps > threshold).astype(int)
        acc = np.mean(predicted == true_labels)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

    # Also try lower perplexity -> non-canon
    for threshold in np.linspace(perps.min(), perps.max(), 1000):
        predicted = (perps < threshold).astype(int)
        acc = np.mean(predicted == true_labels)
        if acc > best_acc:
            best_acc = acc
            best_threshold = -threshold  # negative = inverted

    return best_acc, best_threshold


def main():
    print("Experiment 6: Language Model Perplexity")
    print("=" * 60)

    documents = load_raw_documents(data_dir=get_data_dir())
    n_docs = len(documents)
    true_labels = binary_labels(documents)
    canon_docs = [d for d in documents if d.label == 'canon']
    noncanon_docs = [d for d in documents if d.label != 'canon']

    print(f"Loaded {n_docs} documents ({len(canon_docs)} canon, {len(noncanon_docs)} non-canon)\n")

    # --- Character-level LM ---
    print("Training character-level 5-gram LM on canon texts only...")
    char_lm = CharNgramLM(n=5, alpha=1.0)
    canon_texts = [d.text for d in canon_docs]
    char_lm.train(canon_texts)
    print(f"  Vocabulary: {len(char_lm.vocab)} characters")

    print("Computing character perplexity for all documents...")
    char_perplexities = []
    for d in documents:
        pp = char_lm.perplexity(d.text)
        char_perplexities.append(pp)

    # Statistics
    canon_perps = [char_perplexities[i] for i in range(n_docs) if true_labels[i] == 0]
    noncanon_perps = [char_perplexities[i] for i in range(n_docs) if true_labels[i] == 1]
    print(f"  Canon perplexity: mean={np.mean(canon_perps):.2f}, "
          f"std={np.std(canon_perps):.2f}")
    print(f"  Non-canon perplexity: mean={np.mean(noncanon_perps):.2f}, "
          f"std={np.std(noncanon_perps):.2f}")

    acc, threshold = classify_by_threshold(char_perplexities, documents)
    print(f"  Best threshold accuracy: {acc:.4f}")

    plot_perplexity_distribution(documents, char_perplexities,
                                  'Character 5-gram Perplexity (trained on canon)',
                                  'exp6_char_perplexity.png')

    # --- Word-level LM ---
    print("\nTraining word-level trigram LM on canon texts only...")
    word_lm = WordNgramLM(n=3, alpha=0.01)
    canon_token_lists = [d.tokens for d in canon_docs]
    word_lm.train(canon_token_lists)
    print(f"  Vocabulary: {len(word_lm.vocab)} words")

    print("Computing word perplexity for all documents...")
    word_perplexities = []
    for d in documents:
        pp = word_lm.perplexity(d.tokens)
        word_perplexities.append(pp)

    canon_word_perps = [word_perplexities[i] for i in range(n_docs) if true_labels[i] == 0]
    noncanon_word_perps = [word_perplexities[i] for i in range(n_docs) if true_labels[i] == 1]
    print(f"  Canon perplexity: mean={np.mean(canon_word_perps):.2f}, "
          f"std={np.std(canon_word_perps):.2f}")
    print(f"  Non-canon perplexity: mean={np.mean(noncanon_word_perps):.2f}, "
          f"std={np.std(noncanon_word_perps):.2f}")

    acc_word, threshold_word = classify_by_threshold(word_perplexities, documents)
    print(f"  Best threshold accuracy: {acc_word:.4f}")

    plot_perplexity_distribution(documents, word_perplexities,
                                  'Word Trigram Perplexity (trained on canon)',
                                  'exp6_word_perplexity.png')

    # --- Combined plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, perps, title in [(ax1, char_perplexities, 'Character 5-gram'),
                               (ax2, word_perplexities, 'Word Trigram')]:
        canon_p = [perps[i] for i in range(n_docs) if true_labels[i] == 0]
        noncanon_p = [perps[i] for i in range(n_docs) if true_labels[i] == 1]

        ax.hist(canon_p, bins=15, alpha=0.6, color=LABEL_COLORS['canon'],
                label='Canon', density=True)
        ax.hist(noncanon_p, bins=15, alpha=0.6, color=LABEL_COLORS['commentary'],
                label='Non-canon', density=True)
        ax.set_xlabel('Perplexity')
        ax.set_ylabel('Density')
        ax.set_title(f'{title} Perplexity Distribution')
        ax.legend()

    plt.tight_layout()
    outpath = RESULTS_DIR / 'exp6_perplexity_distributions.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath}")

    # --- Summary ---
    print("\n\nSummary:")
    print(f"  Character 5-gram LM: best accuracy = {acc:.4f}")
    print(f"  Word trigram LM: best accuracy = {acc_word:.4f}")

    # List misclassified at best threshold
    for name, perps_list, thresh in [
        ("Char 5-gram", char_perplexities, threshold),
        ("Word trigram", word_perplexities, threshold_word)
    ]:
        if thresh >= 0:
            predicted = (np.array(perps_list) > thresh).astype(int)
        else:
            predicted = (np.array(perps_list) < -thresh).astype(int)
        misclassified = [documents[i].file_id for i in range(n_docs)
                         if predicted[i] != true_labels[i]]
        print(f"  {name} misclassified: {misclassified}")

    print("\nExperiment 6 complete.")


if __name__ == '__main__':
    main()
