"""Tests for src/vri_inventory.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vri_inventory import get_file_key, FILE_INFO


class TestGetFileKey:
    def test_standard_mul(self):
        assert get_file_key("s0101m.mul.txt") == "s0101m.mul"

    def test_att(self):
        assert get_file_key("s0101a.att.txt") == "s0101a.att"

    def test_nrf(self):
        assert get_file_key("e0101n.mul.txt") == "e0101n.mul"

    def test_vinaya(self):
        assert get_file_key("vin01t1.tik.txt") == "vin01t1.tik"


class TestFileInfo:
    def test_all_entries_have_five_fields(self):
        for key, info in FILE_INFO.items():
            assert len(info) == 5, f"{key} has {len(info)} fields, expected 5"

    def test_valid_pitakas(self):
        valid = {"sutta", "vinaya", "abhidhamma", "extra"}
        for key, info in FILE_INFO.items():
            assert info[2] in valid, f"{key} has invalid pitaka '{info[2]}'"

    def test_valid_labels(self):
        valid = {
            "canon", "commentary", "subcommentary", "sub-subcommentary",
            "paracanonical", "manual", "grammar", "chronicle",
            "devotional", "niti", "council-qa", "miscellaneous", "narrative",
        }
        for key, info in FILE_INFO.items():
            assert info[1] in valid, f"{key} has invalid label '{info[1]}'"

    def test_keys_have_dot_separator(self):
        for key in FILE_INFO:
            assert "." in key, f"{key} missing dot separator"

    def test_has_all_sutta_canon_files(self):
        """Spot-check that key sutta canon files are present."""
        assert "s0101m.mul" in FILE_INFO  # DN I
        assert "s0201m.mul" in FILE_INFO  # MN I
        assert "s0301m.mul" in FILE_INFO  # SN I
        assert "s0502m.mul" in FILE_INFO  # Dhammapada
