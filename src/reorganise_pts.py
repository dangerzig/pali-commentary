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

# ── Commentary source files to be replaced by PTS-aligned versions ──

PS_ATT_SRC = ['s0201a.att.txt', 's0202a.att.txt', 's0203a.att.txt']
SPK_ATT_SRC = ['s0301a.att.txt', 's0302a.att.txt', 's0303a.att.txt',
               's0304a.att.txt', 's0305a.att.txt']
MP_ATT_SRC = ['s0401a.att.txt', 's0402a.att.txt', 's0403a.att.txt',
              's0404a.att.txt']
DHPA_ATT_SRC = ['s0502a.att.txt']
ITA_ATT_SRC = ['s0504a.att.txt']
JAA_ATT_SRC = ['s0513a1.att.txt', 's0513a2.att.txt', 's0513a3.att.txt',
               's0513a4.att.txt', 's0514a1.att.txt', 's0514a2.att.txt',
               's0514a3.att.txt']
SDP_ATT_SRC = ['s0517a.att.txt']

# ── Sub-commentary source files to be replaced ──

PS_TIK_SRC = ['s0201t.tik.txt', 's0202t.tik.txt', 's0203t.tik.txt']
SPK_TIK_SRC = ['s0301t.tik.txt', 's0302t.tik.txt', 's0303t.tik.txt',
               's0304t.tik.txt', 's0305t.tik.txt']
MP_TIK_SRC = ['s0401t.tik.txt', 's0402t.tik.txt', 's0403t.tik.txt',
              's0404t.tik.txt']

ALL_REPLACED = set(
    AN_SRC_FILES + JATAKA_SRC_FILES + PATIS_SRC_FILES + APADANA_SRC_FILES +
    PS_ATT_SRC + SPK_ATT_SRC + MP_ATT_SRC + DHPA_ATT_SRC +
    ITA_ATT_SRC + JAA_ATT_SRC + SDP_ATT_SRC +
    PS_TIK_SRC + SPK_TIK_SRC + MP_TIK_SRC
)


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


# ── Commentary and sub-commentary reorganisation ──

def find_split_line(lines: list[str], pattern: str, start: int = 0) -> int:
    """Find the first line index containing the given text pattern.

    Searches from `start` onwards. Raises ValueError if not found.
    """
    for i in range(start, len(lines)):
        if pattern in lines[i]:
            return i
    raise ValueError(f"Pattern '{pattern}' not found (searched from line {start})")


def reorganise_ps_commentary(src_dir: Path) -> dict[str, str]:
    """Reorganise Papañcasūdanī (MN commentary) to 5 PTS volumes.

    VRI: 3 files (s0201a = Mūlapaṇṇāsa, s0202a = Majjhimapaṇṇāsa,
                   s0203a = Uparipaṇṇāsa)
    PTS: 5 volumes (Ps I = MN 1-10, Ps II = MN 11-50, Ps III = MN 51-100,
                     Ps IV = MN 101-130, Ps V = MN 131-152)
    """
    text1 = read_file(src_dir / 's0201a.att.txt')
    text2 = read_file(src_dir / 's0202a.att.txt')
    text3 = read_file(src_dir / 's0203a.att.txt')

    lines1 = text1.split('\n')
    lines3 = text3.split('\n')

    # Ps I/II: split at Sīhanādavaggo (MN 11 = start of vagga 2)
    split12 = find_split_line(lines1, '2. Sīhanādavaggo')

    # Ps IV/V: split at Vibhaṅgavaggo (MN 131 = start of vagga 4)
    split45 = find_split_line(lines3, '4. Vibhaṅgavaggo')

    return {
        'mna1.att.txt': '\n'.join(lines1[:split12]),
        'mna2.att.txt': '\n'.join(lines1[split12:]),
        'mna3.att.txt': text2,
        'mna4.att.txt': '\n'.join(lines3[:split45]),
        'mna5.att.txt': '\n'.join(lines3[split45:]),
    }


def reorganise_spk_commentary(src_dir: Path) -> dict[str, str]:
    """Reorganise Sāratthappakāsinī (SN commentary) to 3 PTS volumes.

    VRI: 5 files (s0301a-s0305a, one per saṃyutta)
    PTS: 3 volumes (Spk I = Sagāthā, Spk II = Nidāna+Khandha+part Saḷāyatana,
                     Spk III = rest Saḷāyatana+Mahā)
    Split in s0304a (Saḷāyatana) at the Koṭṭhika-aniccasutta commentary,
    which is the first sutta in Spk III (PTS Spk III 1).
    """
    text1 = read_file(src_dir / 's0301a.att.txt')
    text2 = read_file(src_dir / 's0302a.att.txt')
    text3 = read_file(src_dir / 's0303a.att.txt')
    text4 = read_file(src_dir / 's0304a.att.txt')
    text5 = read_file(src_dir / 's0305a.att.txt')

    lines4 = text4.split('\n')
    split = find_split_line(lines4, 'Koṭṭhikaaniccasuttādivaṇṇanā')

    return {
        'spk1.att.txt': text1,
        'spk2.att.txt': text2 + '\n' + text3 + '\n' + '\n'.join(lines4[:split]),
        'spk3.att.txt': '\n'.join(lines4[split:]) + '\n' + text5,
    }


def reorganise_mp_commentary(src_dir: Path) -> dict[str, str]:
    """Reorganise Manorathapūraṇī (AN commentary) to 5 PTS volumes.

    VRI: 4 files (s0401a = Eka, s0402a = Duka+Tika+Catukka,
                   s0403a = Pañcaka+Chakka+Sattaka,
                   s0404a = Aṭṭhaka+Navaka+Dasaka+Ekādasaka)
    PTS: 5 volumes (Mp I = Eka vaggos 1-14, Mp II = Eka vaggos 15-20+Duka+Tika,
                     Mp III = Catukka+Pañcaka+Chakka,
                     Mp IV = Sattaka+Aṭṭhaka+Navaka,
                     Mp V = Dasaka+Ekādasaka)
    """
    text1 = read_file(src_dir / 's0401a.att.txt')
    text2 = read_file(src_dir / 's0402a.att.txt')
    text3 = read_file(src_dir / 's0403a.att.txt')
    text4 = read_file(src_dir / 's0404a.att.txt')

    lines1 = text1.split('\n')
    lines2 = text2.split('\n')
    lines3 = text3.split('\n')
    lines4 = text4.split('\n')

    split12 = find_split_line(lines1, '15. Aṭṭhānapāḷi')
    split23 = find_split_line(lines2, 'Catukkanipāta-aṭṭhakathā')
    split34 = find_split_line(lines3, 'Sattakanipāta-aṭṭhakathā')
    split45 = find_split_line(lines4, 'Dasakanipāta-aṭṭhakathā')

    return {
        'mp1.att.txt': '\n'.join(lines1[:split12]),
        'mp2.att.txt': '\n'.join(lines1[split12:]) + '\n' + '\n'.join(lines2[:split23]),
        'mp3.att.txt': '\n'.join(lines2[split23:]) + '\n' + '\n'.join(lines3[:split34]),
        'mp4.att.txt': '\n'.join(lines3[split34:]) + '\n' + '\n'.join(lines4[:split45]),
        'mp5.att.txt': '\n'.join(lines4[split45:]),
    }


def reorganise_dhpa(src_dir: Path) -> dict[str, str]:
    """Split Dhammapadaṭṭhakathā into 4 PTS volumes.

    DhpA I = vaggas 1-4, DhpA II = 5-8, DhpA III = 9-20, DhpA IV = 21-26.
    """
    text = read_file(src_dir / 's0502a.att.txt')
    lines = text.split('\n')

    split2 = find_split_line(lines, '5. Bālavaggo')
    split3 = find_split_line(lines, '9. Pāpavaggo')
    split4 = find_split_line(lines, '21. Pakiṇṇakavaggo')

    return {
        'dhpa1.att.txt': '\n'.join(lines[:split2]),
        'dhpa2.att.txt': '\n'.join(lines[split2:split3]),
        'dhpa3.att.txt': '\n'.join(lines[split3:split4]),
        'dhpa4.att.txt': '\n'.join(lines[split4:]),
    }


def reorganise_ita(src_dir: Path) -> dict[str, str]:
    """Split Itivuttaka-aṭṭhakathā into 2 PTS volumes.

    It-a I = ekanipāta+dukanipāta, It-a II = tikanipāta+catukkanipāta.
    """
    text = read_file(src_dir / 's0504a.att.txt')
    lines = text.split('\n')

    split = find_split_line(lines, '3. Tikanipāto')

    return {
        'ita1.att.txt': '\n'.join(lines[:split]),
        'ita2.att.txt': '\n'.join(lines[split:]),
    }


def reorganise_jaa(src_dir: Path) -> dict[str, str]:
    """Reorganise Jātakaṭṭhavaṇṇanā to 6 PTS volumes.

    VRI: 7 files (s0513a1-a4 + s0514a1-a3)
    PTS: 6 volumes (Ja I-VI, matching canonical Jātaka volumes)
    s0513a1-a4 → Ja I-IV (1:1), s0514a1 → Ja V,
    s0514a2+s0514a3 → Ja VI (Mahānipāta merged).
    """
    return {
        'jaa1.att.txt': read_file(src_dir / 's0513a1.att.txt'),
        'jaa2.att.txt': read_file(src_dir / 's0513a2.att.txt'),
        'jaa3.att.txt': read_file(src_dir / 's0513a3.att.txt'),
        'jaa4.att.txt': read_file(src_dir / 's0513a4.att.txt'),
        'jaa5.att.txt': read_file(src_dir / 's0514a1.att.txt'),
        'jaa6.att.txt': read_file(src_dir / 's0514a2.att.txt') + '\n' +
                        read_file(src_dir / 's0514a3.att.txt'),
    }


def reorganise_sdp(src_dir: Path) -> dict[str, str]:
    """Split Saddhammappakāsinī (Paṭisambhidāmagga commentary) into 3 PTS volumes.

    Sdp I = Mahāvagga, Sdp II = Yuganaddhavagga, Sdp III = Paññāvagga.
    """
    text = read_file(src_dir / 's0517a.att.txt')
    lines = text.split('\n')

    split2 = find_split_line(lines, '(2) Yuganaddhavaggo')
    split3 = find_split_line(lines, '(3) Paññāvaggo')

    return {
        'sdp1.att.txt': '\n'.join(lines[:split2]),
        'sdp2.att.txt': '\n'.join(lines[split2:split3]),
        'sdp3.att.txt': '\n'.join(lines[split3:]),
    }


# ── Sub-commentary reorganisation ──

def reorganise_ps_subcomm(src_dir: Path) -> dict[str, str]:
    """Reorganise MN Līnatthavaṇṇanā to match 5 PTS Papañcasūdanī volumes."""
    text1 = read_file(src_dir / 's0201t.tik.txt')
    text2 = read_file(src_dir / 's0202t.tik.txt')
    text3 = read_file(src_dir / 's0203t.tik.txt')

    lines1 = text1.split('\n')
    lines3 = text3.split('\n')

    split12 = find_split_line(lines1, '2. Sīhanādavaggo')
    split45 = find_split_line(lines3, '4. Vibhaṅgavaggo')

    return {
        'mnt1.tik.txt': '\n'.join(lines1[:split12]),
        'mnt2.tik.txt': '\n'.join(lines1[split12:]),
        'mnt3.tik.txt': text2,
        'mnt4.tik.txt': '\n'.join(lines3[:split45]),
        'mnt5.tik.txt': '\n'.join(lines3[split45:]),
    }


def reorganise_spk_subcomm(src_dir: Path) -> dict[str, str]:
    """Reorganise SN Līnatthavaṇṇanā to match 3 PTS Sāratthappakāsinī volumes."""
    text1 = read_file(src_dir / 's0301t.tik.txt')
    text2 = read_file(src_dir / 's0302t.tik.txt')
    text3 = read_file(src_dir / 's0303t.tik.txt')
    text4 = read_file(src_dir / 's0304t.tik.txt')
    text5 = read_file(src_dir / 's0305t.tik.txt')

    lines4 = text4.split('\n')
    split = find_split_line(lines4, 'Koṭṭhikaaniccasuttādivaṇṇanā')

    return {
        'spkt1.tik.txt': text1,
        'spkt2.tik.txt': text2 + '\n' + text3 + '\n' + '\n'.join(lines4[:split]),
        'spkt3.tik.txt': '\n'.join(lines4[split:]) + '\n' + text5,
    }


def reorganise_mp_subcomm(src_dir: Path) -> dict[str, str]:
    """Reorganise AN Sāratthamañjūsā to match 5 PTS Manorathapūraṇī volumes."""
    text1 = read_file(src_dir / 's0401t.tik.txt')
    text2 = read_file(src_dir / 's0402t.tik.txt')
    text3 = read_file(src_dir / 's0403t.tik.txt')
    text4 = read_file(src_dir / 's0404t.tik.txt')

    lines1 = text1.split('\n')
    lines2 = text2.split('\n')
    lines3 = text3.split('\n')
    lines4 = text4.split('\n')

    split12 = find_split_line(lines1, '15. Aṭṭhānapāḷi')
    split23 = find_split_line(lines2, 'Catukkanipāta-ṭīkā')
    split34 = find_split_line(lines3, 'Sattakanipāta-ṭīkā')
    split45 = find_split_line(lines4, 'Dasakanipāta-ṭīkā')

    return {
        'mpt1.tik.txt': '\n'.join(lines1[:split12]),
        'mpt2.tik.txt': '\n'.join(lines1[split12:]) + '\n' + '\n'.join(lines2[:split23]),
        'mpt3.tik.txt': '\n'.join(lines2[split23:]) + '\n' + '\n'.join(lines3[:split34]),
        'mpt4.tik.txt': '\n'.join(lines3[split34:]) + '\n' + '\n'.join(lines4[:split45]),
        'mpt5.tik.txt': '\n'.join(lines4[split45:]),
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

    # ── Commentary reorganisation ──

    # 6. Papañcasūdanī (MN commentary): 3 VRI → 5 PTS
    ps_vols = reorganise_ps_commentary(SRC_DIR)
    for name, text in ps_vols.items():
        write_file(DST_DIR / name, text)
        print(f"  Ps: → {name} ({len(text):,} chars)")

    # 7. Sāratthappakāsinī (SN commentary): 5 VRI → 3 PTS
    spk_vols = reorganise_spk_commentary(SRC_DIR)
    for name, text in spk_vols.items():
        write_file(DST_DIR / name, text)
        print(f"  Spk: → {name} ({len(text):,} chars)")

    # 8. Manorathapūraṇī (AN commentary): 4 VRI → 5 PTS
    mp_vols = reorganise_mp_commentary(SRC_DIR)
    for name, text in mp_vols.items():
        write_file(DST_DIR / name, text)
        print(f"  Mp: → {name} ({len(text):,} chars)")

    # 9. Dhammapadaṭṭhakathā: 1 VRI → 4 PTS
    dhpa_vols = reorganise_dhpa(SRC_DIR)
    for name, text in dhpa_vols.items():
        write_file(DST_DIR / name, text)
        print(f"  DhpA: → {name} ({len(text):,} chars)")

    # 10. Itivuttaka-aṭṭhakathā: 1 VRI → 2 PTS
    ita_vols = reorganise_ita(SRC_DIR)
    for name, text in ita_vols.items():
        write_file(DST_DIR / name, text)
        print(f"  It-a: → {name} ({len(text):,} chars)")

    # 11. Jātakaṭṭhavaṇṇanā: 7 VRI → 6 PTS
    jaa_vols = reorganise_jaa(SRC_DIR)
    for name, text in jaa_vols.items():
        write_file(DST_DIR / name, text)
        print(f"  Ja-a: → {name} ({len(text):,} chars)")

    # 12. Saddhammappakāsinī (Paṭis commentary): 1 VRI → 3 PTS
    sdp_vols = reorganise_sdp(SRC_DIR)
    for name, text in sdp_vols.items():
        write_file(DST_DIR / name, text)
        print(f"  Sdp: → {name} ({len(text):,} chars)")

    # ── Sub-commentary reorganisation ──

    # 13. MN Līnatthavaṇṇanā: 3 VRI → 5 PTS (matching Ps volumes)
    ps_tik = reorganise_ps_subcomm(SRC_DIR)
    for name, text in ps_tik.items():
        write_file(DST_DIR / name, text)
        print(f"  MN-ṭ: → {name} ({len(text):,} chars)")

    # 14. SN Līnatthavaṇṇanā: 5 VRI → 3 PTS (matching Spk volumes)
    spk_tik = reorganise_spk_subcomm(SRC_DIR)
    for name, text in spk_tik.items():
        write_file(DST_DIR / name, text)
        print(f"  SN-ṭ: → {name} ({len(text):,} chars)")

    # 15. AN Sāratthamañjūsā: 4 VRI → 5 PTS (matching Mp volumes)
    mp_tik = reorganise_mp_subcomm(SRC_DIR)
    for name, text in mp_tik.items():
        write_file(DST_DIR / name, text)
        print(f"  AN-ṭ: → {name} ({len(text):,} chars)")

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
