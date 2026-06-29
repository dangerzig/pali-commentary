"""Tests for src/reorganise_pts.py."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reorganise_pts import (
    JATAKA_STORY_RE, NIPATA_RE, APADANA_VAGGA_RE,
    JATAKA_VOL_STARTS, ALL_REPLACED,
    AN_SRC_FILES, JATAKA_SRC_FILES, PATIS_SRC_FILES, APADANA_SRC_FILES,
    PS_ATT_SRC, SPK_ATT_SRC, MP_ATT_SRC, DHPA_ATT_SRC, ITA_ATT_SRC,
    JAA_ATT_SRC, SDP_ATT_SRC, PS_TIK_SRC, SPK_TIK_SRC, MP_TIK_SRC,
    concat_files, find_story_line, split_jataka,
    split_patisambhidamagga, split_apadana,
    find_segment_by_pali, merge_an_lemmatized, make_json_doc,
)

# Source-file groups reorganised into PTS volumes: canonical (.mul) and the
# commentaries/sub-commentaries (.att/.tik) added when commentary reorg landed.
CANON_MUL_SRC = (AN_SRC_FILES + JATAKA_SRC_FILES + PATIS_SRC_FILES +
                 APADANA_SRC_FILES)
COMMENTARY_SRC = (PS_ATT_SRC + SPK_ATT_SRC + MP_ATT_SRC + DHPA_ATT_SRC +
                  ITA_ATT_SRC + JAA_ATT_SRC + SDP_ATT_SRC +
                  PS_TIK_SRC + SPK_TIK_SRC + MP_TIK_SRC)


# --- Regex tests ---

class TestJatakaStoryRegex:
    def test_matches_standard_header(self):
        assert JATAKA_STORY_RE.match("1. Apaṇṇakajātakaṃ")

    def test_matches_high_number(self):
        m = JATAKA_STORY_RE.match("547. Vessantarajātakaṃ")
        assert m
        assert int(m.group(1)) == 547

    def test_case_insensitive(self):
        assert JATAKA_STORY_RE.match("151. SUHAJĀTAKAṂ")

    def test_no_match_on_plain_text(self):
        assert not JATAKA_STORY_RE.match("This is regular text")

    def test_no_match_on_non_jataka_numbered(self):
        assert not JATAKA_STORY_RE.match("1. Sīlakkhandhavaggo")


class TestNipataRegex:
    def test_matches_nipata_header(self):
        assert NIPATA_RE.match("1. Ekakanipāto")

    def test_matches_with_prefix(self):
        assert NIPATA_RE.match("22. Mahānipāto")

    def test_no_match_on_partial(self):
        # Must end with nipāto (end of line)
        assert not NIPATA_RE.match("1. Ekakanipāto extra text")


class TestApadanaVaggaRegex:
    def test_matches_vagga_header(self):
        m = APADANA_VAGGA_RE.match("41. Metteyyavaggo")
        assert m
        assert int(m.group(1)) == 41

    def test_matches_first_vagga(self):
        m = APADANA_VAGGA_RE.match("1. Buddhavaggo")
        assert m
        assert int(m.group(1)) == 1


# --- Constants tests ---

class TestConstants:
    def test_all_replaced_is_union(self):
        # ALL_REPLACED is the union of every reorganised source group:
        # the canonical .mul files PLUS the commentary/sub-commentary .att/.tik.
        expected = set(CANON_MUL_SRC + COMMENTARY_SRC)
        assert ALL_REPLACED == expected

    def test_all_replaced_includes_canon_mul(self):
        # Anti-drift: the original canonical .mul set must stay reorganised.
        assert set(CANON_MUL_SRC) <= ALL_REPLACED

    def test_all_replaced_no_overlap_or_dupes(self):
        # Each source file appears in exactly one group (no dupes across groups).
        all_src = CANON_MUL_SRC + COMMENTARY_SRC
        assert len(all_src) == len(set(all_src))
        assert len(ALL_REPLACED) == len(all_src)

    def test_jataka_vol_starts_sorted(self):
        assert JATAKA_VOL_STARTS == sorted(JATAKA_VOL_STARTS)

    def test_jataka_vol_starts_has_6_entries(self):
        assert len(JATAKA_VOL_STARTS) == 6

    def test_jataka_starts_at_1(self):
        assert JATAKA_VOL_STARTS[0] == 1


# --- concat_files ---

class TestConcatFiles:
    def test_concatenates_in_order(self, tmp_path):
        (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
        (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
        (tmp_path / "c.txt").write_text("gamma", encoding="utf-8")

        result = concat_files(tmp_path, ["a.txt", "b.txt", "c.txt"])
        assert result == "alpha\nbeta\ngamma"

    def test_single_file(self, tmp_path):
        (tmp_path / "only.txt").write_text("solo", encoding="utf-8")
        result = concat_files(tmp_path, ["only.txt"])
        assert result == "solo"


# --- find_story_line ---

class TestFindStoryLine:
    def test_finds_story_at_exact_line(self):
        lines = [
            "some text",
            "more text",
            "151. Rājovādajātakaṃ",
            "story content",
        ]
        assert find_story_line(lines, 151) == 2

    def test_includes_preceding_nipata_header(self):
        lines = [
            "end of previous story",
            "",
            "2. Dukanipāto",
            "",
            "151. Rājovādajātakaṃ",
            "story content",
        ]
        # Should return the nipata header line, not the story line
        assert find_story_line(lines, 151) == 2

    def test_stops_backward_scan_at_content(self):
        lines = [
            "actual content from previous story",
            "",
            "151. Rājovādajātakaṃ",
            "story content",
        ]
        # Should NOT go back past real content
        assert find_story_line(lines, 151) == 2

    def test_skips_uddana_during_backward_scan(self):
        lines = [
            "end of content",
            "",
            "3. Tikanipāto",
            "Tassuddānaṃ –",
            "",
            "301. Cullakāliṅgajātakaṃ",
            "story",
        ]
        # Should find nipata header, skipping the uddana
        assert find_story_line(lines, 301) == 2

    def test_raises_on_missing_story(self):
        lines = ["no stories here", "just text"]
        with pytest.raises(ValueError, match="not found"):
            find_story_line(lines, 999)


# --- split_jataka ---

class TestSplitJataka:
    def _make_jataka_files(self, tmp_path):
        """Create synthetic Jataka files with correct story headers."""
        lines1 = []
        # Stories 1-537 (file 1 contains most stories)
        for story in range(1, 538):
            lines1.append(f"{story}. Story{story}jātakaṃ")
            lines1.append(f"Content of story {story}")
            lines1.append("")

        lines2 = []
        # Stories 538-547 (file 2)
        for story in range(538, 548):
            lines2.append(f"{story}. Story{story}jātakaṃ")
            lines2.append(f"Content of story {story}")
            lines2.append("")

        (tmp_path / "s0513m.mul.txt").write_text(
            "\n".join(lines1), encoding="utf-8")
        (tmp_path / "s0514m.mul.txt").write_text(
            "\n".join(lines2), encoding="utf-8")

    def test_produces_six_volumes(self, tmp_path):
        self._make_jataka_files(tmp_path)
        vols = split_jataka(tmp_path)
        assert set(vols.keys()) == {
            "ja1.mul.txt", "ja2.mul.txt", "ja3.mul.txt",
            "ja4.mul.txt", "ja5.mul.txt", "ja6.mul.txt",
        }

    def test_volume_boundaries(self, tmp_path):
        self._make_jataka_files(tmp_path)
        vols = split_jataka(tmp_path)

        # J I should start with story 1
        assert "1. Story1jātakaṃ" in vols["ja1.mul.txt"]
        # J I should NOT contain story 151
        assert "151. Story151jātakaṃ" not in vols["ja1.mul.txt"]

        # J II should start with story 151
        assert "151. Story151jātakaṃ" in vols["ja2.mul.txt"]

        # J VI should contain story 538
        assert "538. Story538jātakaṃ" in vols["ja6.mul.txt"]
        # J VI should contain story 547
        assert "547. Story547jātakaṃ" in vols["ja6.mul.txt"]

    def test_no_text_lost(self, tmp_path):
        self._make_jataka_files(tmp_path)
        vols = split_jataka(tmp_path)

        # Every story from 1-547 should appear in exactly one volume
        for story in range(1, 548):
            header = f"{story}. Story{story}jātakaṃ"
            matches = [name for name, text in vols.items() if header in text]
            assert len(matches) == 1, f"Story {story} in {len(matches)} vols"


# --- split_patisambhidamagga ---

class TestSplitPatisambhidamagga:
    def test_splits_at_yuganaddhavaggo(self, tmp_path):
        text = "\n".join([
            "1. Mahāvaggo",
            "Content of vol I",
            "More vol I content",
            "2. Yuganaddhavaggo",
            "Content of vol II",
            "3. Paññāvaggo",
            "More vol II content",
        ])
        (tmp_path / "s0517m.mul.txt").write_text(text, encoding="utf-8")

        vols = split_patisambhidamagga(tmp_path)
        assert "ps1.mul.txt" in vols
        assert "ps2.mul.txt" in vols

        assert "Mahāvaggo" in vols["ps1.mul.txt"]
        assert "Yuganaddhavaggo" not in vols["ps1.mul.txt"]
        assert "Yuganaddhavaggo" in vols["ps2.mul.txt"]
        assert "Paññāvaggo" in vols["ps2.mul.txt"]

    def test_raises_if_boundary_missing(self, tmp_path):
        (tmp_path / "s0517m.mul.txt").write_text(
            "No boundary here", encoding="utf-8")
        with pytest.raises(ValueError, match="Yuganaddhavaggo"):
            split_patisambhidamagga(tmp_path)


# --- split_apadana ---

class TestSplitApadana:
    def test_splits_at_vaggo_41(self, tmp_path):
        lines = []
        for v in range(1, 55):
            lines.append(f"{v}. Vagga{v}vaggo")
            lines.append(f"Content of vagga {v}")
            lines.append("")

        # Split roughly in the middle for the two source files
        (tmp_path / "s0510m1.mul.txt").write_text(
            "\n".join(lines), encoding="utf-8")
        (tmp_path / "s0510m2.mul.txt").write_text(
            "Extra apadana content", encoding="utf-8")

        vols = split_apadana(tmp_path)
        assert "ap1.mul.txt" in vols
        assert "ap2.mul.txt" in vols

        # Vol I should contain vaggas 1-40
        assert "40. Vagga40vaggo" in vols["ap1.mul.txt"]
        assert "41. Vagga41vaggo" not in vols["ap1.mul.txt"]

        # Vol II should contain vagga 41+ and file 2 content
        assert "41. Vagga41vaggo" in vols["ap2.mul.txt"]
        assert "Extra apadana content" in vols["ap2.mul.txt"]

    def test_raises_if_vaggo_41_missing(self, tmp_path):
        (tmp_path / "s0510m1.mul.txt").write_text(
            "No vagga headers", encoding="utf-8")
        (tmp_path / "s0510m2.mul.txt").write_text(
            "Also nothing", encoding="utf-8")
        with pytest.raises(ValueError, match="vaggo 41"):
            split_apadana(tmp_path)


# --- find_segment_by_pali ---

class TestFindSegmentByPali:
    def test_finds_matching_segment(self):
        import re
        segments = [
            {"pali": "Some text"},
            {"pali": "151. Rājovādajātakaṃ"},
            {"pali": "More text"},
        ]
        idx = find_segment_by_pali(segments, JATAKA_STORY_RE, 151)
        assert idx == 1

    def test_includes_preceding_nipata_header(self):
        segments = [
            {"pali": "end of previous"},
            {"pali": ""},
            {"pali": "2. Dukanipāto"},
            {"pali": ""},
            {"pali": "151. Rājovādajātakaṃ"},
        ]
        idx = find_segment_by_pali(segments, JATAKA_STORY_RE, 151)
        assert idx == 2  # nipata header, not the story itself

    def test_raises_on_no_match(self):
        segments = [{"pali": "nothing here"}]
        with pytest.raises(ValueError, match="not found"):
            find_segment_by_pali(segments, JATAKA_STORY_RE, 999)

    def test_match_value_filtering(self):
        segments = [
            {"pali": "1. Apaṇṇakajātakaṃ"},
            {"pali": "151. Rājovādajātakaṃ"},
        ]
        # Should skip story 1, find story 151
        idx = find_segment_by_pali(segments, JATAKA_STORY_RE, 151)
        assert idx == 1


# --- merge_an_lemmatized ---

class TestMergeAnLemmatized:
    def _make_an_json(self, tmp_path):
        """Create minimal AN JSON files for testing."""
        an_files = {
            "s0401m": {"segments": [{"pali": "nipata 1"}]},
            "s0402m1": {"segments": [{"pali": "nipata 2"}]},
            "s0402m2": {"segments": [{"pali": "nipata 3"}]},
            "s0402m3": {"segments": [{"pali": "nipata 4"}]},
            "s0403m1": {"segments": [{"pali": "nipata 5"}]},
            "s0403m2": {"segments": [{"pali": "nipata 6"}]},
            "s0403m3": {"segments": [{"pali": "nipata 7"}]},
            "s0404m1": {"segments": [{"pali": "nipata 8"}]},
            "s0404m2": {"segments": [{"pali": "nipata 9"}]},
            "s0404m3": {"segments": [{"pali": "nipata 10"}]},
            "s0404m4": {"segments": [{"pali": "nipata 11"}]},
        }
        for stem, data in an_files.items():
            (tmp_path / f"{stem}.json").write_text(
                json.dumps(data), encoding="utf-8")

    def test_produces_five_volumes(self, tmp_path):
        self._make_an_json(tmp_path)
        vols = merge_an_lemmatized(tmp_path)
        assert set(vols.keys()) == {"an1", "an2", "an3", "an4", "an5"}

    def test_correct_segment_counts(self, tmp_path):
        self._make_an_json(tmp_path)
        vols = merge_an_lemmatized(tmp_path)
        # A I = nipatas 1-3 (3 segments)
        assert len(vols["an1"]["segments"]) == 3
        # A II = nipata 4 (1 segment)
        assert len(vols["an2"]["segments"]) == 1
        # A III = nipatas 5-6 (2 segments)
        assert len(vols["an3"]["segments"]) == 2
        # A IV = nipatas 7-9 (3 segments)
        assert len(vols["an4"]["segments"]) == 3
        # A V = nipatas 10-11 (2 segments)
        assert len(vols["an5"]["segments"]) == 2

    def test_total_segments_preserved(self, tmp_path):
        self._make_an_json(tmp_path)
        vols = merge_an_lemmatized(tmp_path)
        total = sum(len(v["segments"]) for v in vols.values())
        assert total == 11  # one per nipata

    def test_raises_on_missing_file(self, tmp_path):
        # Only write one file, not all 11
        data = {"segments": [{"pali": "test"}]}
        (tmp_path / "s0401m.json").write_text(
            json.dumps(data), encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            merge_an_lemmatized(tmp_path)


# --- make_json_doc ---

class TestMakeJsonDoc:
    def test_structure(self):
        segments = [{"pali": "test1"}, {"pali": "test2"}]
        doc = make_json_doc("an1", segments)
        assert doc["id"] == "an1"
        assert doc["title_pali"] == "an1"
        assert doc["collection"] == "sutta"
        assert doc["segments"] == segments
