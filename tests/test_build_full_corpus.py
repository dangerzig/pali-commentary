"""Tests for src/build_full_corpus.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import csv

from build_full_corpus import (
    INCLUDED_LABELS, INCLUDED_PITAKAS, load_inventory, pts_file_label,
    DST_DIR,
)
from reorganise_pts import ALL_REPLACED
from data_loader import PTS_NAMES


class TestConstants:
    def test_included_labels(self):
        assert INCLUDED_LABELS == {"canon", "commentary", "subcommentary", "paracanonical"}

    def test_included_pitakas(self):
        assert INCLUDED_PITAKAS == {"sutta", "vinaya", "abhidhamma"}

    def test_all_replaced_consistent_with_reorganise(self):
        """ALL_REPLACED reorganises the canon (.mul) AND commentaries (.att/.tik)."""
        assert len(ALL_REPLACED) == 50
        # Spot-check canon + commentary so the two groups can't silently drop
        assert "s0513m.mul.txt" in ALL_REPLACED   # Jataka (canon)
        assert "s0401m.mul.txt" in ALL_REPLACED   # AN (canon)
        assert "s0201a.att.txt" in ALL_REPLACED   # Papañcasūdanī (commentary)

    def test_pts_names_covers_canon_reorganised_ids(self):
        """PTS_NAMES must cover at least all reorganised canonical stems."""
        canon_stems = {"an1", "an2", "an3", "an4", "an5",
                       "ja1", "ja2", "ja3", "ja4", "ja5", "ja6",
                       "ps1", "ps2", "ap1", "ap2"}
        assert canon_stems <= set(PTS_NAMES.keys())


class TestPtsFileLabel:
    """The PTS suffix -> label mapping (guards against the canon-mislabel bug)."""

    def test_mul_is_canon(self):
        assert pts_file_label("an1.mul.txt") == ("canon", "sutta")
        assert pts_file_label("ja2.mul.txt") == ("canon", "sutta")

    def test_att_is_commentary(self):
        assert pts_file_label("mna1.att.txt") == ("commentary", "sutta")
        assert pts_file_label("jaa3.att.txt") == ("commentary", "sutta")

    def test_tik_is_subcommentary(self):
        assert pts_file_label("mnt1.tik.txt") == ("subcommentary", "sutta")


class TestLabelsIntegrity:
    """The built corpus's LABELS.csv must never mislabel commentary as canon."""

    @pytest.mark.skipif(not (DST_DIR / "LABELS.csv").exists(),
                        reason="corpus not built (LABELS.csv absent)")
    def test_no_commentary_labelled_canon(self):
        with open(DST_DIR / "LABELS.csv") as f:
            for row in csv.DictReader(f):
                fn, label = row["filename"], row["label"]
                if fn.endswith(".att.txt"):
                    assert label == "commentary", f"{fn} mislabelled {label}"
                elif fn.endswith(".tik.txt"):
                    assert label == "subcommentary", f"{fn} mislabelled {label}"
                elif fn.endswith(".mul.txt") and row["pitaka"] == "sutta" \
                        and fn.split(".")[0] in PTS_NAMES:
                    assert label == "canon", f"{fn} mislabelled {label}"


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
