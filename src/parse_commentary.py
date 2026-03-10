#!/usr/bin/env python3
"""
Parse VRI raw commentary files into JSON segment format for lemmatization.

The lemmatizer expects JSON files with the structure:
{
    "id": "...",
    "title_pali": "...",
    "collection": "commentary",
    "segments": [
        {"id": "...", "pali": "..."},
        ...
    ]
}

This script parses the raw .att.txt, .tik.txt, and .nrf.txt files from VRI/CST4,
splitting them into segments (paragraphs), and outputs JSON files compatible
with the existing lemmatization pipeline.
"""

import json
import re
from pathlib import Path
from typing import Optional

VRI_RAW_DIR = Path.home() / "pali-canon" / "data" / "vri-raw"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "commentary_parsed"

# Regex for splitting paragraphs - split on double newlines or numbered paragraphs
PARAGRAPH_SPLIT = re.compile(r'\n\s*\n')

# Regex for cleaning: remove HTML-like tags if any
HTML_TAG = re.compile(r'<[^>]+>')


def parse_vri_commentary(filepath: Path) -> dict:
    """Parse a VRI raw text file into the JSON segment format.

    Segments are split by paragraph (double newline).
    Each segment gets a sequential ID based on the file ID.

    Args:
        filepath: Path to the .att.txt / .tik.txt / .nrf.txt file.

    Returns:
        Dict with id, collection, segments fields.
    """
    file_id = filepath.stem.split('.')[0]  # e.g., 's0101a' from 's0101a.att.txt'
    suffix = filepath.stem.split('.')[1]   # e.g., 'att'

    text = filepath.read_text(encoding='utf-8')

    # Clean HTML tags
    text = HTML_TAG.sub('', text)

    # Split into paragraphs
    paragraphs = PARAGRAPH_SPLIT.split(text)

    segments = []
    seg_num = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Skip very short segments (likely headers or empty lines)
        # But keep them if they have actual Pali content
        words = para.split()
        if len(words) < 2:
            continue

        seg_num += 1
        segments.append({
            "id": f"{file_id}:{seg_num}",
            "pali": para,
        })

    # Determine collection type
    collection_map = {
        'att': 'commentary',
        'tik': 'subcommentary',
        'nrf': 'paracanonical',
    }

    return {
        "id": file_id,
        "title_pali": file_id,
        "collection": collection_map.get(suffix, 'unknown'),
        "segments": segments,
    }


def parse_all_commentaries(input_dir: Optional[Path] = None,
                           output_dir: Optional[Path] = None,
                           suffixes: Optional[list[str]] = None) -> list[Path]:
    """Parse all commentary files and write JSON outputs.

    Args:
        input_dir: Directory containing VRI .txt files.
        output_dir: Directory for JSON output files.
        suffixes: Which file types to process. Default: ['att', 'tik', 'nrf'].

    Returns:
        List of output file paths.
    """
    if input_dir is None:
        input_dir = VRI_RAW_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if suffixes is None:
        suffixes = ['att', 'tik', 'nrf']

    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = []
    for suffix in suffixes:
        pattern = f's*.{suffix}.txt'
        for filepath in sorted(input_dir.glob(pattern)):
            print(f"  Parsing {filepath.name}...")
            data = parse_vri_commentary(filepath)

            outpath = output_dir / f"{data['id']}.json"
            with open(outpath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            output_files.append(outpath)
            print(f"    -> {outpath.name}: {len(data['segments'])} segments")

    return output_files


def parse_canon_files(input_dir: Optional[Path] = None,
                      output_dir: Optional[Path] = None) -> list[Path]:
    """Parse canonical .mul.txt files into JSON format for consistency.

    This is useful if we want to lemmatize the canonical texts directly
    from VRI raw format (rather than using the SuttaCentral-aligned versions).
    """
    if input_dir is None:
        input_dir = VRI_RAW_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = []
    for filepath in sorted(input_dir.glob('s*.mul.txt')):
        file_id = filepath.stem.split('.')[0]
        print(f"  Parsing {filepath.name}...")

        text = filepath.read_text(encoding='utf-8')
        text = HTML_TAG.sub('', text)
        paragraphs = PARAGRAPH_SPLIT.split(text)

        segments = []
        seg_num = 0
        for para in paragraphs:
            para = para.strip()
            if not para or len(para.split()) < 2:
                continue
            seg_num += 1
            segments.append({
                "id": f"{file_id}:{seg_num}",
                "pali": para,
            })

        data = {
            "id": file_id,
            "title_pali": file_id,
            "collection": "canon",
            "segments": segments,
        }

        outpath = output_dir / f"{file_id}.json"
        with open(outpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        output_files.append(outpath)
        print(f"    -> {outpath.name}: {len(segments)} segments")

    return output_files


if __name__ == '__main__':
    print("Parsing VRI commentary files to JSON...")
    print("=" * 60)

    # Parse all commentary/subcommentary/paracanonical files
    files = parse_all_commentaries()
    print(f"\nParsed {len(files)} files")

    # Also parse canonical files for consistent treatment
    print("\nParsing canonical files...")
    canon_files = parse_canon_files()
    print(f"Parsed {len(canon_files)} canonical files")

    print("\nDone! Output in:", OUTPUT_DIR)
