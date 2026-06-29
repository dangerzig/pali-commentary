"""Tests for src/parse_commentary.py."""

import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parse_commentary import parse_vri_commentary, HTML_TAG


class TestParseVriCommentary:
    def _write_temp_file(self, content, suffix='.att.txt'):
        """Write content to a temporary file and return its Path."""
        f = NamedTemporaryFile(mode='w', suffix=suffix, prefix='s0101a',
                               delete=False, encoding='utf-8')
        f.write(content)
        f.close()
        return Path(f.name)

    def test_structure(self):
        content = "First paragraph with some text.\n\nSecond paragraph here."
        path = self._write_temp_file(content)
        try:
            result = parse_vri_commentary(path)
            assert 'id' in result
            assert 'collection' in result
            assert 'segments' in result
            assert result['collection'] == 'commentary'
        finally:
            path.unlink()

    def test_segments(self):
        content = "First paragraph with some words.\n\nSecond paragraph with more words."
        path = self._write_temp_file(content)
        try:
            result = parse_vri_commentary(path)
            assert len(result['segments']) == 2
            for seg in result['segments']:
                assert 'id' in seg
                assert 'pali' in seg
                assert len(seg['pali']) > 0
        finally:
            path.unlink()

    def test_drops_nonpali_keeps_pali_headers(self):
        # Digit/punctuation-only paragraphs are dropped, but single-word Pāli
        # section headers (e.g. 'Majjhimanikāye') are kept (review finding #15).
        content = "123\n\nMajjhimanikāye\n\nReal paragraph with content."
        path = self._write_temp_file(content)
        try:
            result = parse_vri_commentary(path)
            palis = [s['pali'] for s in result['segments']]
            assert "123" not in palis              # digit-only dropped
            assert "Majjhimanikāye" in palis        # single-word Pāli header kept
            assert len(result['segments']) == 2
        finally:
            path.unlink()

    def test_empty_file(self):
        path = self._write_temp_file("")
        try:
            result = parse_vri_commentary(path)
            assert result['segments'] == []
        finally:
            path.unlink()


    def test_collection_type_commentary(self):
        content = "Some commentary paragraph text here."
        path = self._write_temp_file(content, suffix='.att.txt')
        try:
            result = parse_vri_commentary(path)
            assert result['collection'] == 'commentary'
        finally:
            path.unlink()

    def test_collection_type_subcommentary(self):
        content = "Some subcommentary paragraph text here."
        path = self._write_temp_file(content, suffix='.tik.txt')
        try:
            result = parse_vri_commentary(path)
            assert result['collection'] == 'subcommentary'
        finally:
            path.unlink()

    def test_segment_ids_sequential(self):
        content = "First paragraph with words.\n\nSecond paragraph with words.\n\nThird paragraph with words."
        path = self._write_temp_file(content)
        try:
            result = parse_vri_commentary(path)
            ids = [seg['id'] for seg in result['segments']]
            # IDs should end with :1, :2, :3
            assert ids[0].endswith(':1')
            assert ids[1].endswith(':2')
            assert ids[2].endswith(':3')
        finally:
            path.unlink()


class TestHtmlTagRemoval:
    def test_html_stripped(self):
        text = '<p>Hello <b>world</b></p>'
        cleaned = HTML_TAG.sub('', text)
        assert cleaned == 'Hello world'

    def test_no_tags(self):
        text = 'plain text here'
        cleaned = HTML_TAG.sub('', text)
        assert cleaned == text

    def test_nested_tags(self):
        text = '<div><span class="x">content</span></div>'
        cleaned = HTML_TAG.sub('', text)
        assert cleaned == 'content'
