#!/usr/bin/env python3
"""
Wrapper to run the pali-canon lemmatization pipeline on commentary texts.

Uses the existing lemmatize_canon.py infrastructure from ~/pali-canon/src/,
processing the parsed commentary JSON files.
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Add pali-canon src to path for imports
PALI_CANON_SRC = Path.home() / "pali-canon" / "src"
sys.path.insert(0, str(PALI_CANON_SRC))

PARSED_DIR = Path(__file__).parent.parent / "data" / "commentary_parsed"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "lemmatized"


def lemmatize_file(input_path: Path, output_path: Path, lemmatizer) -> dict:
    """Lemmatize a single parsed JSON file.

    Args:
        input_path: Path to parsed JSON file.
        output_path: Path for lemmatized output.
        lemmatizer: Initialized Lemmatizer object.

    Returns:
        Stats dict with word counts and coverage.
    """
    with open(input_path) as f:
        data = json.load(f)

    total_words = 0
    found_words = 0

    for segment in data.get('segments', []):
        pali = segment.get('pali', '')
        tokens = lemmatizer.tokenize(pali)

        token_dicts = []
        for word in tokens:
            total_words += 1
            info = lemmatizer.lookup_word(word)
            token_dict = info.to_dict()
            if token_dict.get('lemma'):
                found_words += 1
            token_dicts.append(token_dict)

        segment['tokens'] = token_dicts

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    coverage = found_words / total_words if total_words > 0 else 0
    return {
        'total_words': total_words,
        'found_words': found_words,
        'coverage': coverage,
    }


def main():
    print("Lemmatizing commentary texts")
    print("=" * 60)

    # Import the lemmatizer from pali-canon
    try:
        from lemmatize_canon import Lemmatizer
    except ImportError as e:
        print(f"Error importing lemmatizer: {e}")
        print(f"Make sure {PALI_CANON_SRC}/lemmatize_canon.py exists")
        sys.exit(1)

    # Initialize lemmatizer
    print("Initializing lemmatizer...")
    lemmatizer = Lemmatizer()
    print("  Lemmatizer ready")

    # Process all parsed JSON files
    parsed_files = sorted(PARSED_DIR.glob('*.json'))
    print(f"\nFound {len(parsed_files)} parsed files")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_stats = Counter()

    for i, input_path in enumerate(parsed_files):
        output_path = OUTPUT_DIR / input_path.name
        print(f"  [{i+1}/{len(parsed_files)}] {input_path.name}...", end='', flush=True)

        stats = lemmatize_file(input_path, output_path, lemmatizer)
        total_stats['total_words'] += stats['total_words']
        total_stats['found_words'] += stats['found_words']

        print(f" {stats['total_words']:,} words, {stats['coverage']:.1%} coverage")

    overall_coverage = (total_stats['found_words'] / total_stats['total_words']
                        if total_stats['total_words'] > 0 else 0)
    print(f"\nOverall: {total_stats['total_words']:,} words, "
          f"{total_stats['found_words']:,} found, "
          f"{overall_coverage:.1%} coverage")

    print(f"\nOutput in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
