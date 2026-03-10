"""
Shared constants and helpers used across experiment scripts.
"""

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
    's0517m': 'Nettippakaraṇa',
    's0518m': 'Milindapañha',
    's0519m': 'Peṭakopadesa',
    's0520m': 'Suttasaṅgaha',
}


def get_doc_index(documents, file_id):
    """Return the index of a document by file_id, or None if not found."""
    for i, d in enumerate(documents):
        if d.file_id == file_id:
            return i
    return None
