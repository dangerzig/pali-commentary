"""Shared fixtures for pali-commentary tests."""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pytest

# Add src to path so tests can import project modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import Document


@pytest.fixture
def sample_documents():
    """Small set of synthetic Document objects for testing."""
    canon_tokens_1 = ['bhikkhave', 'kho', 'bhagavā', 'bhikkhu', 'me', 'ti']
    canon_tokens_2 = ['bhikkhave', 'pe', 'bhagavā', 'kho', 'aha', 'ti', 'me']
    comm_tokens_1 = ['tattha', 'viya', 'vuttam', 'attho', 'nāma', 'nti']
    comm_tokens_2 = ['tattha', 'kāraṇā', 'viya', 'nāma', 'attho', 'nti', 'vuttam']

    return [
        Document(
            file_id='s0101m', filename='s0101m.mul.txt',
            label='canon', suffix='mul',
            tokens=canon_tokens_1,
            word_counts=Counter(canon_tokens_1),
        ),
        Document(
            file_id='s0201m', filename='s0201m.mul.txt',
            label='canon', suffix='mul',
            tokens=canon_tokens_2,
            word_counts=Counter(canon_tokens_2),
        ),
        Document(
            file_id='s0101a', filename='s0101a.att.txt',
            label='commentary', suffix='att',
            tokens=comm_tokens_1,
            word_counts=Counter(comm_tokens_1),
        ),
        Document(
            file_id='s0201a', filename='s0201a.att.txt',
            label='commentary', suffix='att',
            tokens=comm_tokens_2,
            word_counts=Counter(comm_tokens_2),
        ),
    ]


@pytest.fixture
def sample_feature_matrix():
    """Small numpy array for testing matrix operations."""
    return np.array([
        [0.10, 0.05, 0.00, 0.08],
        [0.12, 0.04, 0.01, 0.07],
        [0.00, 0.01, 0.10, 0.02],
        [0.01, 0.00, 0.12, 0.03],
    ])
