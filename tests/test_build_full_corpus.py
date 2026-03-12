"""Tests for src/build_full_corpus.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from build_full_corpus import (
    INCLUDED_LABELS, INCLUDED_PITAKAS, load_inventory,
)
from reorganise_pts import ALL_REPLACED
from data_loader import PTS_NAMES


class TestConstants:
    def test_included_labels(self):
        assert INCLUDED_LABELS == {"canon", "commentary", "subcommentary", "paracanonical"}

    def test_included_pitakas(self):
        assert INCLUDED_PITAKAS == {"sutta", "vinaya", "abhidhamma"}

    def test_all_replaced_consistent_with_reorganise(self):
        """ALL_REPLACED imported from reorganise_pts should have 16 files."""
        assert len(ALL_REPLACED) == 16
        # Spot-check
        assert "s0513m.mul.txt" in ALL_REPLACED  # Jataka
        assert "s0401m.mul.txt" in ALL_REPLACED  # AN

    def test_pts_names_has_all_reorganised_ids(self):
        """PTS_NAMES should cover all reorganised file stems."""
        expected_stems = {"an1", "an2", "an3", "an4", "an5",
                          "ja1", "ja2", "ja3", "ja4", "ja5", "ja6",
                          "ps1", "ps2", "ap1", "ap2"}
        assert expected_stems == set(PTS_NAMES.keys())


class TestLoadInventory:
    @pytest.mark.skipif(
        not (Path(__file__).parent.parent / "data" / "vri_inventory.csv").exists(),
        reason="vri_inventory.csv not generated"
    )
    def test_loads_rows(self):
        rows = load_inventory()
        assert len(rows) > 200  # should be ~217
        for row in rows:
            assert "vri_file" in row
            assert "correct_label" in row
            assert "pitaka" in row
            assert "text_name" in row
