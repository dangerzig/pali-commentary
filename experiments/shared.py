"""
Shared constants and helpers used across experiment scripts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# The six consistently misclassified boundary texts
MISCLASSIFIED_IDS = ['s0515m', 's0516m', 's0517m', 's0518m', 's0519m', 's0520m']

# Canon texts misclassified as non-canon
CANON_AS_NONCANON = ['s0515m', 's0516m', 's0517m', 's0519m']

# Paracanonical texts misclassified as canon
NONCANON_AS_CANON = ['s0518m', 's0520m']

# Key discriminative words from prior experiments
DISCRIMINATIVE_WORDS = [
    'ti', 'bhikkhave', 'kho', 'nti', 'tattha', 'viya', 'vuttam',
    'aha', 'pe', 'me', 'bhikkhu', 'bhagavā', 'kāraṇā', 'attho', 'nāma',
]

# Short display names for the six boundary texts
SHORT_NAMES = {
    's0515m': 'Mahāniddesa',
    's0516m': 'Cūḷaniddesa',
    's0517m': 'Paṭisambhidāmagga',
    's0518m': 'Milindapañha',
    's0519m': 'Nettippakaraṇa',
    's0520m': 'Peṭakopadesa',
}


def use_pts_data() -> bool:
    """Return True if --pts flag is passed on the command line."""
    return '--pts' in sys.argv


def get_data_dir():
    """Return the appropriate data directory based on --pts flag."""
    from data_loader import RAW_DATA_DIR, RAW_PTS_DIR
    if use_pts_data():
        return RAW_PTS_DIR
    return RAW_DATA_DIR


def get_lemma_dir():
    """Return the appropriate lemmatized directory based on --pts flag."""
    from data_loader import LEMMATIZED_DIR, LEMMATIZED_PTS_DIR
    if use_pts_data():
        return LEMMATIZED_PTS_DIR
    return LEMMATIZED_DIR


def get_results_dir() -> Path:
    """Return the results directory, creating it if needed."""
    base = Path(__file__).parent.parent
    if use_pts_data():
        d = base / "results-pts"
    else:
        d = base / "results"
    d.mkdir(exist_ok=True)
    return d


def get_doc_index(documents, file_id):
    """Return the index of a document by file_id, or None if not found."""
    for i, d in enumerate(documents):
        if d.file_id == file_id:
            return i
    return None
