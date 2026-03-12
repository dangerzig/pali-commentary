"""Tests for src/data_loader.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    parse_filename, tokenize_pali, binary_labels, load_raw_documents,
    VRI_NAMES, PTS_NAMES, RAW_DATA_DIR, Document,
)


class TestParseFilename:
    def test_canon(self):
        file_id, suffix, label = parse_filename('s0101m.mul.txt')
        assert file_id == 's0101m'
        assert suffix == 'mul'
        assert label == 'canon'

    def test_commentary(self):
        file_id, suffix, label = parse_filename('s0101a.att.txt')
        assert file_id == 's0101a'
        assert suffix == 'att'
        assert label == 'commentary'

    def test_subcommentary(self):
        file_id, suffix, label = parse_filename('s0101t.tik.txt')
        assert file_id == 's0101t'
        assert suffix == 'tik'
        assert label == 'subcommentary'

    def test_paracanonical(self):
        file_id, suffix, label = parse_filename('s0517m.nrf.txt')
        assert file_id == 's0517m'
        assert suffix == 'nrf'
        assert label == 'paracanonical'


class TestTokenizePali:
    def test_basic(self):
        tokens = tokenize_pali('Evaṃ me sutaṃ.')
        assert tokens == ['evaṃ', 'me', 'sutaṃ']

    def test_diacritics_preserved(self):
        tokens = tokenize_pali('āīūṭḍṇṅñṃḷ')
        assert tokens == ['āīūṭḍṇṅñṃḷ']

    def test_niggahita_normalized(self):
        tokens = tokenize_pali('evaṁ')
        assert tokens == ['evaṃ']

    def test_empty(self):
        assert tokenize_pali('') == []

    def test_punctuation_only(self):
        assert tokenize_pali('., ;!? 123') == []

    def test_lowercase(self):
        tokens = tokenize_pali('BHIKKHU Dhamma')
        assert tokens == ['bhikkhu', 'dhamma']


class TestBinaryLabels:
    def test_labels(self, sample_documents):
        labels = binary_labels(sample_documents)
        assert labels == [0, 0, 1, 1]


class TestDocumentName:
    def test_name_property(self, sample_documents):
        doc = sample_documents[0]
        assert 's0101m' in doc.name
        assert 'D I' in doc.name

    def test_short_id(self, sample_documents):
        assert sample_documents[0].short_id == 's0101m'

    def test_multipart_file_id(self):
        """Multi-part IDs like s0402m1 should resolve to base code s0402."""
        doc = Document(file_id='s0402m1', filename='s0402m1.mul.txt',
                       label='canon', suffix='mul')
        assert 'AN Dukanipāta' in doc.name

    def test_multipart_file_id_m2(self):
        doc = Document(file_id='s0402m2', filename='s0402m2.mul.txt',
                       label='canon', suffix='mul')
        assert 'AN Dukanipāta' in doc.name

    def test_pts_name_takes_priority(self):
        """PTS-reorganised IDs (e.g., 'an1') should use PTS_NAMES."""
        doc = Document(file_id='an1', filename='an1.mul.txt',
                       label='canon', suffix='mul')
        assert 'A I' in doc.name

    def test_unknown_base_falls_back(self):
        """Unknown file IDs should fall back to displaying the base code."""
        doc = Document(file_id='s9999m', filename='s9999m.mul.txt',
                       label='canon', suffix='mul')
        assert 's9999' in doc.name


class TestLoadRawDocuments:
    @pytest.mark.skipif(
        not RAW_DATA_DIR.exists() or not list(RAW_DATA_DIR.glob('s*.txt')),
        reason="VRI data directory not available"
    )
    def test_load_raw_documents(self):
        docs = load_raw_documents()
        assert len(docs) > 0
        for d in docs:
            assert d.file_id
            assert d.label in ('canon', 'commentary', 'subcommentary', 'paracanonical')
            assert len(d.tokens) >= 100

    @pytest.mark.skipif(
        not RAW_DATA_DIR.exists() or not list(RAW_DATA_DIR.glob('s*.txt')),
        reason="VRI data directory not available"
    )
    def test_load_raw_documents_filter(self):
        docs = load_raw_documents(include_labels=['canon'])
        for d in docs:
            assert d.label == 'canon'
