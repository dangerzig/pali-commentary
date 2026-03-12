"""
Reorganise VRI/CST4 data files to match PTS volume boundaries.

Creates data/raw-pts/ with files aligned to PTS editions:
  - AN: 11 VRI files → 5 PTS volumes (A I–V)
  - Jātaka: 2 VRI files → 6 PTS volumes (J I–VI)
  - Paṭisambhidāmagga: 1 VRI file → 2 PTS volumes (Paṭis I–II)
  - Apadāna: 2 VRI files → 2 PTS volumes (Ap I–II) with corrected boundary
  - All other files: copied unchanged

PTS volume boundaries are verified against SuttaCentral bilara reference data.
"""

import json
import re
import shutil
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent / "data" / "raw"
DST_DIR = Path(__file__).parent.parent / "data" / "raw-pts"
LEMMA_DIR = Path(__file__).parent.parent / "data" / "lemmatized"
LEMMA_PTS_DIR = Path(__file__).parent.parent / "data" / "lemmatized-pts"

# VRI source files that will be replaced by reorganised versions
AN_SRC_FILES = [
    's0401m.mul.txt', 's0402m1.mul.txt', 's0402m2.mul.txt', 's0402m3.mul.txt',
    's0403m1.mul.txt', 's0403m2.mul.txt', 's0403m3.mul.txt',
    's0404m1.mul.txt', 's0404m2.mul.txt', 's0404m3.mul.txt', 's0404m4.mul.txt',
]
JATAKA_SRC_FILES = ['s0513m.mul.txt', 's0514m.mul.txt']
PATIS_SRC_FILES = ['s0517m.mul.txt']
APADANA_SRC_FILES = ['s0510m1.mul.txt', 's0510m2.mul.txt']

ALL_REPLACED = set(AN_SRC_FILES + JATAKA_SRC_FILES + PATIS_SRC_FILES + APADANA_SRC_FILES)

# AN PTS volume mapping: PTS vol → list of VRI source files to concatenate
AN_PTS = {
    'an1.mul.txt': ['s0401m.mul.txt', 's0402m1.mul.txt', 's0402m2.mul.txt'],  # A I (nipātas 1-3)
    'an2.mul.txt': ['s0402m3.mul.txt'],                                        # A II (nipāta 4)
    'an3.mul.txt': ['s0403m1.mul.txt', 's0403m2.mul.txt'],                     # A III (nipātas 5-6)
    'an4.mul.txt': ['s0403m3.mul.txt', 's0404m1.mul.txt', 's0404m2.mul.txt'],  # A IV (nipātas 7-9)
    'an5.mul.txt': ['s0404m3.mul.txt', 's0404m4.mul.txt'],                     # A V (nipātas 10-11)
}

# Jātaka PTS boundaries: story numbers that START each volume
# J I = stories 1-150, J II = 151-300, J III = 301-438,
# J IV = 439-510, J V = 511-537, J VI = 538-547
JATAKA_VOL_STARTS = [1, 151, 301, 439, 511, 538]

# Pattern matching a Jātaka story header: "NNN. <Name>jātakaṃ"
JATAKA_STORY_RE = re.compile(r'^(\d+)\.\s+\S+jātakaṃ', re.IGNORECASE)
# Pattern matching a nipāta header: "NN. <Name>nipāto"
NIPATA_RE = re.compile(r'^\d+\.\s+\S+nipāto$', re.IGNORECASE)

# Apadāna: PTS Vol II starts at vaggo 41 (Metteyyavaggo)
# Verified against bilara PTS refs: tha-ap402 = vaggo 40 last story (PTS I),
# tha-ap403 = vaggo 41 first story (PTS II).
# Vaggo 1 has 12 apadānas (compound numbering "3-1" through "3-10"),
# so sequential counting must account for this.
APADANA_VAGGA_RE = re.compile(r'^(\d+)\.\s+\S+vaggo', re.IGNORECASE)


def read_file(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def write_file(path: Path, text: str):
    path.write_text(text, encoding='utf-8')


def concat_files(src_dir: Path, filenames: list[str]) -> str:
    """Concatenate multiple source files."""
    parts = []
    for fn in filenames:
        parts.append(read_file(src_dir / fn))
    return '\n'.join(parts)


def find_story_line(lines: list[str], story_num: int) -> int:
    """Find the line index where a Jātaka story begins.

    Returns the index of the story header line, or of a preceding
    nipāta header if one appears between the previous content and
    this story (so that section headers stay with their stories).
    """
    for i, line in enumerate(lines):
        m = JATAKA_STORY_RE.match(line.strip())
        if m and int(m.group(1)) == story_num:
            # Scan backwards for a nipāta header or other section header
            # that should logically be part of this volume
            split_line = i
            for j in range(i - 1, max(i - 20, 0), -1):
                stripped = lines[j].strip()
                if NIPATA_RE.match(stripped):
                    split_line = j
                    break
                if stripped and not stripped.startswith('Tassuddānaṃ') and 'niṭṭhitaṃ' not in stripped:
                    # Non-empty line that isn't an uddāna or completion marker
                    # means we've gone past the inter-section gap
                    break
            return split_line
    raise ValueError(f"Jātaka story {story_num} not found")


def split_jataka(src_dir: Path) -> dict[str, str]:
    """Split two VRI Jātaka files into six PTS volumes."""
    text1 = read_file(src_dir / 's0513m.mul.txt')
    text2 = read_file(src_dir / 's0514m.mul.txt')

    lines1 = text1.split('\n')
    lines2 = text2.split('\n')

    # Combined line list for stories that span both files
    combined = lines1 + lines2

    volumes = {}
    vol_names = ['ja1.mul.txt', 'ja2.mul.txt', 'ja3.mul.txt',
                 'ja4.mul.txt', 'ja5.mul.txt', 'ja6.mul.txt']

    for vol_idx in range(6):
        start_story = JATAKA_VOL_STARTS[vol_idx]

        if vol_idx == 0:
            start_line = 0
        else:
            start_line = find_story_line(combined, start_story)

        if vol_idx < 5:
            next_story = JATAKA_VOL_STARTS[vol_idx + 1]
            end_line = find_story_line(combined, next_story)
        else:
            end_line = len(combined)

        vol_text = '\n'.join(combined[start_line:end_line])
        volumes[vol_names[vol_idx]] = vol_text

    return volumes


def split_patisambhidamagga(src_dir: Path) -> dict[str, str]:
    """Split Paṭisambhidāmagga into two PTS volumes.

    PTS Vol I = Mahāvaggo (first section)
    PTS Vol II = Yuganaddhavaggo + Paññāvaggo (remaining sections)
    """
    text = read_file(src_dir / 's0517m.mul.txt')
    lines = text.split('\n')

    # Find "2. Yuganaddhavaggo" which starts PTS Vol II
    split_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^2\.\s+Yuganaddhavaggo', line.strip()):
            split_idx = i
            break

    if split_idx is None:
        raise ValueError("Could not find Yuganaddhavaggo boundary in Paṭisambhidāmagga")

    return {
        'ps1.mul.txt': '\n'.join(lines[:split_idx]),
        'ps2.mul.txt': '\n'.join(lines[split_idx:]),
    }


def split_apadana(src_dir: Path) -> dict[str, str]:
    """Split Apadāna into two PTS volumes.

    PTS Vol I = vaggos 1-40 (tha-ap 1-402)
    PTS Vol II = vaggo 41 onwards (tha-ap 403+)
    Split at vaggo 41 (Metteyyavaggo), verified against bilara PTS refs.
    """
    text1 = read_file(src_dir / 's0510m1.mul.txt')
    text2 = read_file(src_dir / 's0510m2.mul.txt')

    lines = text1.split('\n')

    # Find vaggo 41 header line
    split_idx = None
    for i, line in enumerate(lines):
        m = APADANA_VAGGA_RE.match(line.strip())
        if m and int(m.group(1)) == 41:
            split_idx = i
            break

    if split_idx is None:
        raise ValueError("Could not find vaggo 41 (Metteyyavaggo) in s0510m1")

    return {
        'ap1.mul.txt': '\n'.join(lines[:split_idx]),
        'ap2.mul.txt': '\n'.join(lines[split_idx:]) + '\n' + text2,
    }


# --- Lemmatized JSON helpers ---

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data: dict):
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_json_doc(doc_id: str, segments: list[dict]) -> dict:
    """Create a lemmatized JSON document with the standard structure."""
    return {
        'id': doc_id,
        'title_pali': doc_id,
        'collection': 'sutta',
        'segments': segments,
    }


def find_segment_by_pali(segments: list[dict], pattern: re.Pattern,
                         match_value: int = None) -> int:
    """Find segment index where pali text matches pattern.

    If match_value is given, the first regex group must equal that integer.
    Returns the segment index, scanning backwards to include preceding
    section headers (nipāta headers, etc.).
    """
    for i, seg in enumerate(segments):
        pali = seg.get('pali', '').strip()
        m = pattern.match(pali)
        if m:
            if match_value is not None and int(m.group(1)) != match_value:
                continue
            # Scan backwards for section headers to include with this segment
            split_idx = i
            for j in range(i - 1, max(i - 20, 0), -1):
                p = segments[j].get('pali', '').strip()
                if NIPATA_RE.match(p):
                    split_idx = j
                    break
                if p and not p.startswith('Tassuddānaṃ') and 'niṭṭhitaṃ' not in p:
                    break
            return split_idx
    raise ValueError(f"Pattern {pattern.pattern} (value={match_value}) not found")


def merge_an_lemmatized(lemma_dir: Path) -> dict[str, dict]:
    """Merge AN lemmatized JSON files into 5 PTS volumes."""
    # AN PTS mapping: PTS id → list of VRI JSON stems
    an_json_map = {
        'an1': ['s0401m', 's0402m1', 's0402m2'],      # A I (nipātas 1-3)
        'an2': ['s0402m3'],                             # A II (nipāta 4)
        'an3': ['s0403m1', 's0403m2'],                  # A III (nipātas 5-6)
        'an4': ['s0403m3', 's0404m1', 's0404m2'],       # A IV (nipātas 7-9)
        'an5': ['s0404m3', 's0404m4'],                   # A V (nipātas 10-11)
    }
    results = {}
    for pts_id, src_ids in an_json_map.items():
        all_segments = []
        for src_id in src_ids:
            path = lemma_dir / f'{src_id}.json'
            if not path.exists():
                raise FileNotFoundError(f"Missing lemmatized file: {path}")
            data = load_json(path)
            all_segments.extend(data['segments'])
        results[pts_id] = make_json_doc(pts_id, all_segments)
    return results


def split_jataka_lemmatized(lemma_dir: Path) -> dict[str, dict]:
    """Split two VRI Jātaka JSON files into six PTS volumes."""
    d1 = load_json(lemma_dir / 's0513m.json')
    d2 = load_json(lemma_dir / 's0514m.json')
    combined = d1['segments'] + d2['segments']

    vol_names = ['ja1', 'ja2', 'ja3', 'ja4', 'ja5', 'ja6']
    results = {}

    for vol_idx in range(6):
        start_story = JATAKA_VOL_STARTS[vol_idx]
        if vol_idx == 0:
            start_seg = 0
        else:
            start_seg = find_segment_by_pali(combined, JATAKA_STORY_RE, start_story)

        if vol_idx < 5:
            next_story = JATAKA_VOL_STARTS[vol_idx + 1]
            end_seg = find_segment_by_pali(combined, JATAKA_STORY_RE, next_story)
        else:
            end_seg = len(combined)

        pts_id = vol_names[vol_idx]
        results[pts_id] = make_json_doc(pts_id, combined[start_seg:end_seg])

    return results


def split_patisambhidamagga_lemmatized(lemma_dir: Path) -> dict[str, dict]:
    """Split Paṭisambhidāmagga JSON into two PTS volumes."""
    data = load_json(lemma_dir / 's0517m.json')
    segs = data['segments']

    yugavagga_re = re.compile(r'^2\.\s+Yuganaddhavaggo')
    split_idx = None
    for i, seg in enumerate(segs):
        if yugavagga_re.match(seg.get('pali', '').strip()):
            split_idx = i
            break

    if split_idx is None:
        raise ValueError("Could not find Yuganaddhavaggo in s0517m.json")

    return {
        'ps1': make_json_doc('ps1', segs[:split_idx]),
        'ps2': make_json_doc('ps2', segs[split_idx:]),
    }


def split_apadana_lemmatized(lemma_dir: Path) -> dict[str, dict]:
    """Split Apadāna JSON files into two PTS volumes."""
    d1 = load_json(lemma_dir / 's0510m1.json')
    d2 = load_json(lemma_dir / 's0510m2.json')

    segs1 = d1['segments']
    split_idx = None
    for i, seg in enumerate(segs1):
        m = APADANA_VAGGA_RE.match(seg.get('pali', '').strip())
        if m and int(m.group(1)) == 41:
            split_idx = i
            break

    if split_idx is None:
        raise ValueError("Could not find vaggo 41 in s0510m1.json")

    return {
        'ap1': make_json_doc('ap1', segs1[:split_idx]),
        'ap2': make_json_doc('ap2', segs1[split_idx:] + d2['segments']),
    }


# VRI JSON stems that are replaced by reorganised versions
LEMMA_REPLACED = {
    's0401m', 's0402m1', 's0402m2', 's0402m3',
    's0403m1', 's0403m2', 's0403m3',
    's0404m1', 's0404m2', 's0404m3', 's0404m4',
    's0513m', 's0514m',
    's0517m',
    's0510m1', 's0510m2',
}


def reorganise_lemmatized():
    """Create data/lemmatized-pts/ with PTS-aligned JSON files."""
    if not LEMMA_DIR.exists():
        print("No lemmatized data directory found, skipping.")
        return

    LEMMA_PTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Copy unchanged files
    copied = 0
    for src_file in sorted(LEMMA_DIR.glob('*.json')):
        if src_file.stem not in LEMMA_REPLACED:
            shutil.copy2(src_file, LEMMA_PTS_DIR / src_file.name)
            copied += 1
    print(f"\nLemmatized: copied {copied} unchanged files")

    # 2. Merge AN
    an_vols = merge_an_lemmatized(LEMMA_DIR)
    for pts_id, doc in an_vols.items():
        save_json(LEMMA_PTS_DIR / f'{pts_id}.json', doc)
        print(f"  AN: → {pts_id}.json ({len(doc['segments'])} segments)")

    # 3. Split Jātaka
    ja_vols = split_jataka_lemmatized(LEMMA_DIR)
    for pts_id, doc in ja_vols.items():
        save_json(LEMMA_PTS_DIR / f'{pts_id}.json', doc)
        print(f"  Jātaka: → {pts_id}.json ({len(doc['segments'])} segments)")

    # 4. Split Paṭisambhidāmagga
    ps_vols = split_patisambhidamagga_lemmatized(LEMMA_DIR)
    for pts_id, doc in ps_vols.items():
        save_json(LEMMA_PTS_DIR / f'{pts_id}.json', doc)
        print(f"  Paṭis: → {pts_id}.json ({len(doc['segments'])} segments)")

    # 5. Split Apadāna
    ap_vols = split_apadana_lemmatized(LEMMA_DIR)
    for pts_id, doc in ap_vols.items():
        save_json(LEMMA_PTS_DIR / f'{pts_id}.json', doc)
        print(f"  Ap: → {pts_id}.json ({len(doc['segments'])} segments)")

    # Summary
    all_files = list(LEMMA_PTS_DIR.glob('*.json'))
    src_total_segs = sum(
        len(load_json(f)['segments']) for f in LEMMA_DIR.glob('*.json')
    )
    dst_total_segs = sum(
        len(load_json(f)['segments']) for f in all_files
    )
    print(f"\n  Total lemmatized files: {len(all_files)}")
    print(f"  Source segments: {src_total_segs:,}")
    print(f"  Output segments: {dst_total_segs:,}")
    if src_total_segs != dst_total_segs:
        print(f"  WARNING: segment count mismatch!")
    else:
        print(f"  Segment counts match (OK)")


def reorganise():
    """Create data/raw-pts/ with PTS-aligned files."""
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Copy all unchanged files
    copied = 0
    for src_file in sorted(SRC_DIR.glob('s*.txt')):
        if src_file.name not in ALL_REPLACED:
            shutil.copy2(src_file, DST_DIR / src_file.name)
            copied += 1

    print(f"Copied {copied} unchanged files")

    # 2. Merge AN files into PTS volumes
    for dst_name, src_names in AN_PTS.items():
        text = concat_files(SRC_DIR, src_names)
        write_file(DST_DIR / dst_name, text)
        print(f"  AN: {' + '.join(src_names)} → {dst_name}")

    # 3. Split Jātaka into 6 PTS volumes
    ja_vols = split_jataka(SRC_DIR)
    for name, text in ja_vols.items():
        write_file(DST_DIR / name, text)
        print(f"  Jātaka: → {name} ({len(text):,} chars)")

    # 4. Split Paṭisambhidāmagga into 2 PTS volumes
    ps_vols = split_patisambhidamagga(SRC_DIR)
    for name, text in ps_vols.items():
        write_file(DST_DIR / name, text)
        print(f"  Paṭis: → {name} ({len(text):,} chars)")

    # 5. Split Apadāna into 2 PTS volumes
    ap_vols = split_apadana(SRC_DIR)
    for name, text in ap_vols.items():
        write_file(DST_DIR / name, text)
        print(f"  Ap: → {name} ({len(text):,} chars)")

    # Summary
    all_files = list(DST_DIR.glob('*.txt'))
    mul_files = [f for f in all_files if '.mul.' in f.name]
    att_files = [f for f in all_files if '.att.' in f.name]
    tik_files = [f for f in all_files if '.tik.' in f.name]
    nrf_files = [f for f in all_files if '.nrf.' in f.name]

    print(f"\nTotal files in {DST_DIR.name}/: {len(all_files)}")
    print(f"  Canon (.mul): {len(mul_files)}")
    print(f"  Commentary (.att): {len(att_files)}")
    print(f"  Sub-commentary (.tik): {len(tik_files)}")
    print(f"  Paracanonical (.nrf): {len(nrf_files)}")

    # Verify total character count
    src_total = sum(f.stat().st_size for f in SRC_DIR.glob('s*.txt'))
    dst_total = sum(f.stat().st_size for f in all_files)
    print(f"\nSource total size: {src_total:,} bytes")
    print(f"Output total size: {dst_total:,} bytes")
    diff_pct = abs(dst_total - src_total) / src_total * 100
    if diff_pct > 1:
        print(f"  WARNING: size differs by {diff_pct:.1f}% (expected <1% from join/split)")
    else:
        print(f"  Size difference: {diff_pct:.2f}% (OK)")


if __name__ == '__main__':
    reorganise()
    reorganise_lemmatized()
