"""
Load and tokenize VRI/CST4 Pāli texts, extracting document labels.

Handles both raw VRI text files and lemmatized JSON files.
Labels are derived from filename suffixes:
  .mul.txt -> canon
  .att.txt -> commentary
  .tik.txt -> subcommentary
  .nrf.txt -> paracanonical
"""

import json
import re
import unicodedata
import warnings
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Regex for tokenizing Pāli text: split on anything that isn't a Pāli letter
PALI_TOKEN_RE = re.compile(r'[^a-zA-ZāīūṭḍṇṅñṃḷĀĪŪṬḌṆṄÑṂḶ]+')

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_PTS_DIR = Path(__file__).parent.parent / "data" / "raw-pts"
LEMMATIZED_DIR = Path(__file__).parent.parent / "data" / "lemmatized"
LEMMATIZED_PTS_DIR = Path(__file__).parent.parent / "data" / "lemmatized-pts"

# Also check pali-canon lemmatized data for canonical texts
CANON_LEMMATIZED_DIR = Path.home() / "pali-canon" / "data" / "lemmatized"

LABEL_MAP = {
    'mul': 'canon',
    'att': 'commentary',
    'tik': 'subcommentary',
    'nrf': 'paracanonical',
}

# Mapping from VRI/CST4 file codes to human-readable names.
# The s05xx codes follow the CSCD Khuddakanikāya ordering (verified
# against actual file contents), which differs from some published lists.
VRI_NAMES = {
    # Dīgha Nikāya
    's0101': 'D I (Sīlakkhandhavagga)',
    's0102': 'D II (Mahāvagga)',
    's0103': 'D III (Pāthikavagga)',
    # Majjhima Nikāya
    's0201': 'M I (Mūlapaṇṇāsa)',
    's0202': 'M II (Majjhimapaṇṇāsa)',
    's0203': 'M III (Uparipaṇṇāsa)',
    # Saṃyutta Nikāya
    's0301': 'S I (Sagāthāvagga)',
    's0302': 'S II (Nidānavagga)',
    's0303': 'S III (Khandhavagga)',
    's0304': 'S IV (Saḷāyatanavagga)',
    's0305': 'S V (Mahāvagga)',
    # Aṅguttara Nikāya (VRI splits by nipāta; PTS volumes differ)
    's0401': 'AN Ekakanipāta',
    's0402': 'AN Dukanipāta–Catukkanipāta',
    's0403': 'AN Pañcakanipāta–Sattakanipāta',
    's0404': 'AN Aṭṭhakanipāta–Ekādasakanipāta',
    # Khuddaka Nikāya
    's0501': 'Khuddakapāṭha',
    's0502': 'Dhammapada',
    's0503': 'Udāna',
    's0504': 'Itivuttaka',
    's0505': 'Suttanipāta',
    's0506': 'Vimānavatthu',
    's0507': 'Petavatthu',
    's0508': 'Theragāthā',
    's0509': 'Therīgāthā',
    's0510': 'Apadāna',
    's0511': 'Buddhavaṃsa',
    's0512': 'Cariyāpiṭaka',
    's0513': 'Jātaka (Part 1)',
    's0514': 'Jātaka (Part 2)',
    's0515': 'Mahāniddesa',
    's0516': 'Cūḷaniddesa',
    's0517': 'Paṭisambhidāmagga',
    's0518': 'Milindapañha',
    's0519': 'Nettippakaraṇa',
    's0520': 'Peṭakopadesa',
}

# Mapping from PTS-reorganised file IDs to PTS volume names.
# Used when loading from data/raw-pts/.
PTS_NAMES = {
    # AN PTS volumes (merged from VRI nipāta files)
    'an1': 'A I',
    'an2': 'A II',
    'an3': 'A III',
    'an4': 'A IV',
    'an5': 'A V',
    # Jātaka PTS volumes (split from VRI parts)
    'ja1': 'J I',
    'ja2': 'J II',
    'ja3': 'J III',
    'ja4': 'J IV',
    'ja5': 'J V',
    'ja6': 'J VI',
    # Paṭisambhidāmagga PTS volumes (split from single VRI file)
    'ps1': 'Paṭis I',
    'ps2': 'Paṭis II',
    # Apadāna PTS volumes (re-split from VRI parts)
    'ap1': 'Ap I',
    'ap2': 'Ap II',
    # Papañcasūdanī (MN commentary) PTS volumes
    'mna1': 'Ps I',
    'mna2': 'Ps II',
    'mna3': 'Ps III',
    'mna4': 'Ps IV',
    'mna5': 'Ps V',
    # Sāratthappakāsinī (SN commentary) PTS volumes
    'spk1': 'Spk I',
    'spk2': 'Spk II',
    'spk3': 'Spk III',
    # Manorathapūraṇī (AN commentary) PTS volumes
    'mp1': 'Mp I',
    'mp2': 'Mp II',
    'mp3': 'Mp III',
    'mp4': 'Mp IV',
    'mp5': 'Mp V',
    # Dhammapadaṭṭhakathā PTS volumes
    'dhpa1': 'DhpA I',
    'dhpa2': 'DhpA II',
    'dhpa3': 'DhpA III',
    'dhpa4': 'DhpA IV',
    # Itivuttaka-aṭṭhakathā PTS volumes
    'ita1': 'It-a I',
    'ita2': 'It-a II',
    # Jātakaṭṭhavaṇṇanā PTS volumes
    'jaa1': 'Ja I',
    'jaa2': 'Ja II',
    'jaa3': 'Ja III',
    'jaa4': 'Ja IV',
    'jaa5': 'Ja V',
    'jaa6': 'Ja VI',
    # Saddhammappakāsinī (Paṭis commentary) PTS volumes
    'sdp1': 'Sdp I',
    'sdp2': 'Sdp II',
    'sdp3': 'Sdp III',
    # MN Līnatthavaṇṇanā (sub-commentary) matching Ps volumes
    'mnt1': 'MN-ṭ I',
    'mnt2': 'MN-ṭ II',
    'mnt3': 'MN-ṭ III',
    'mnt4': 'MN-ṭ IV',
    'mnt5': 'MN-ṭ V',
    # SN Līnatthavaṇṇanā (sub-commentary) matching Spk volumes
    'spkt1': 'SN-ṭ I',
    'spkt2': 'SN-ṭ II',
    'spkt3': 'SN-ṭ III',
    # AN Sāratthamañjūsā (sub-commentary) matching Mp volumes
    'mpt1': 'AN-ṭ I',
    'mpt2': 'AN-ṭ II',
    'mpt3': 'AN-ṭ III',
    'mpt4': 'AN-ṭ IV',
    'mpt5': 'AN-ṭ V',
}


@dataclass
class Document:
    """A single VRI text document with its metadata."""
    file_id: str          # e.g., 's0101m'
    filename: str         # e.g., 's0101m.mul.txt'
    label: str            # e.g., 'canon'
    suffix: str           # e.g., 'mul'
    text: str = ''        # raw text content
    tokens: list[str] = field(default_factory=list)  # tokenized words
    lemmas: list[str] = field(default_factory=list)   # lemmatized words (if available)
    word_counts: Counter = field(default_factory=Counter)
    lemma_counts: Counter = field(default_factory=Counter)

    @property
    def name(self) -> str:
        """Human-readable name for this document."""
        # Check PTS names first (for reorganised files like 'an1', 'ja3')
        if self.file_id in PTS_NAMES:
            return f"{self.file_id} ({PTS_NAMES[self.file_id]})"
        # Strip trailing letter suffix and optional digit (e.g., 's0402m1' -> 's0402')
        base = re.sub(r'[mat]\d*$', '', self.file_id)
        name = VRI_NAMES.get(base, base)
        return f"{self.file_id} ({name})"

    @property
    def short_id(self) -> str:
        """Short ID like 's0507m' for labeling plots."""
        return self.file_id


def parse_filename(filename: str) -> tuple[str, str, str]:
    """Extract file_id, suffix type, and label from a VRI filename.

    Examples:
        's0101m.mul.txt' -> ('s0101m', 'mul', 'canon')
        's0101a.att.txt' -> ('s0101a', 'att', 'commentary')
    """
    name = Path(filename).name
    parts = name.split('.')
    file_id = parts[0]
    suffix = parts[1]
    label = LABEL_MAP.get(suffix, 'unknown')
    return file_id, suffix, label


def tokenize_pali(text: str) -> list[str]:
    """Tokenize Pāli text into lowercase words.

    Splits on any non-Pāli character (punctuation, digits, whitespace).
    Returns lowercase tokens.
    """
    # NFC-normalize and lowercase FIRST, then fold niggahīta variants, so that
    # uppercase/combining forms (Ṁ, m+U+0323) are handled too — otherwise a
    # dot-above niggahīta survives and PALI_TOKEN_RE splits the word at it.
    text = unicodedata.normalize('NFC', text).lower()
    text = text.replace('ṁ', 'ṃ')
    tokens = PALI_TOKEN_RE.split(text)
    return [t for t in tokens if t]


def load_raw_documents(data_dir: Optional[Path] = None,
                       include_labels: Optional[list[str]] = None,
                       min_tokens: int = 100) -> list[Document]:
    """Load all VRI raw text files and tokenize them.

    Args:
        data_dir: Directory containing VRI .txt files. Defaults to data/raw/.
        include_labels: If provided, only load documents with these labels
                       (e.g., ['canon', 'commentary']).
        min_tokens: Minimum number of tokens to include a document.
                   Filters out empty/stub files. Set to 0 to include all.

    Returns:
        List of Document objects sorted by file_id.
    """
    if data_dir is None:
        data_dir = RAW_DATA_DIR

    documents = []
    for filepath in sorted(data_dir.glob('*.txt')):
        file_id, suffix, label = parse_filename(filepath.name)

        if include_labels is not None and label not in include_labels:
            continue

        text = filepath.read_text(encoding='utf-8')
        tokens = tokenize_pali(text)
        word_counts = Counter(tokens)

        if len(tokens) < min_tokens:
            continue

        doc = Document(
            file_id=file_id,
            filename=filepath.name,
            label=label,
            suffix=suffix,
            text=text,
            tokens=tokens,
            word_counts=word_counts,
        )
        documents.append(doc)

    return documents


def load_lemmatized_tokens(json_path: Path) -> list[str]:
    """Extract lemma sequence from a lemmatized JSON file.

    Handles both flat and nested (suttas/items) structures.
    Returns list of lemma strings.
    """
    with open(json_path) as f:
        data = json.load(f)

    lemmas = []

    def extract_from_segments(segments):
        for seg in segments:
            if 'tokens' not in seg:
                continue
            for tok in seg['tokens']:
                if tok.get('pos') == 'eng':
                    continue
                # For sandhi-decomposed words, use component lemmas
                if 'components' in tok and tok['components']:
                    for comp in tok['components']:
                        if comp.get('lemma'):
                            lemmas.append(comp['lemma'])
                elif tok.get('lemma'):
                    lemmas.append(tok['lemma'])

    # Handle different structures
    if 'segments' in data:
        extract_from_segments(data['segments'])
    elif 'suttas' in data:
        for sutta in data['suttas']:
            if 'segments' in sutta:
                extract_from_segments(sutta['segments'])
    elif 'items' in data:
        for item in data['items']:
            if 'segments' in item:
                extract_from_segments(item['segments'])

    return lemmas


def load_lemmatized_documents(lemma_dir: Optional[Path] = None,
                              canon_lemma_dir: Optional[Path] = None,
                              raw_docs: Optional[list[Document]] = None) -> list[Document]:
    """Load lemmatized versions of documents.

    This loads lemmatized JSON files and matches them back to the
    raw documents. For canonical texts, uses the pali-canon lemmatized data.
    For commentary texts, uses local lemmatized data.

    Args:
        lemma_dir: Directory for commentary lemmatized files.
        canon_lemma_dir: Directory for canonical lemmatized files.
        raw_docs: If provided, attach lemma data to these documents.

    Returns:
        List of Documents with lemma data populated.
    """
    if lemma_dir is None:
        lemma_dir = LEMMATIZED_DIR
    if canon_lemma_dir is None:
        canon_lemma_dir = CANON_LEMMATIZED_DIR

    # If no raw docs provided, load them
    if raw_docs is None:
        raw_docs = load_raw_documents()

    # Collect all available lemmatized JSON files
    lemma_files = {}

    # Canon lemmatized files (from pali-canon project)
    if canon_lemma_dir.exists():
        for json_file in canon_lemma_dir.rglob('*.json'):
            if json_file.name.startswith('_'):
                continue
            lemma_files[json_file.stem] = json_file

    # Commentary lemmatized files (from this project)
    if lemma_dir.exists():
        for json_file in lemma_dir.rglob('*.json'):
            if json_file.name.startswith('_'):
                continue
            lemma_files[json_file.stem] = json_file

    # For each raw document, try to find and load its lemmatized version
    for doc in raw_docs:
        # Try exact match first
        if doc.file_id in lemma_files:
            lemmas = load_lemmatized_tokens(lemma_files[doc.file_id])
            doc.lemmas = lemmas
            doc.lemma_counts = Counter(lemmas)

    return raw_docs


def get_document_labels(documents: list[Document]) -> tuple[list[str], list[str]]:
    """Extract parallel lists of file_ids and labels.

    Returns:
        (file_ids, labels) tuple
    """
    return ([d.file_id for d in documents],
            [d.label for d in documents])


def binary_labels(documents: list[Document]) -> list[int]:
    """Convert labels to binary: canon=0, everything else=1.

    This matches the original paper's approach of classifying
    canon vs. non-canon (commentary/subcommentary/paracanonical).
    """
    # Guard against a file with an unexpected suffix silently becoming
    # non-canon (parse_filename returns 'unknown' for unmapped suffixes).
    known = {'canon', 'commentary', 'subcommentary', 'paracanonical'}
    bad = sorted(d.file_id for d in documents if d.label not in known)
    if bad:
        warnings.warn(f"binary_labels: {len(bad)} doc(s) with unexpected label "
                      f"are treated as non-canon: {bad[:8]}")
    return [0 if d.label == 'canon' else 1 for d in documents]


if __name__ == '__main__':
    import sys
    use_pts = '--pts' in sys.argv

    if use_pts:
        data_dir = RAW_PTS_DIR
        print(f"Loading from PTS-reorganised data: {data_dir}")
    else:
        data_dir = RAW_DATA_DIR
        print(f"Loading from VRI data: {data_dir}")

    docs = load_raw_documents(data_dir=data_dir)
    print(f"Loaded {len(docs)} documents")

    for label in ['canon', 'commentary', 'subcommentary', 'paracanonical']:
        subset = [d for d in docs if d.label == label]
        total_tokens = sum(len(d.tokens) for d in subset)
        print(f"  {label}: {len(subset)} docs, {total_tokens:,} tokens")

    print(f"\nFirst 5 documents:")
    for d in docs[:5]:
        print(f"  {d.name}: {d.label} ({len(d.tokens):,} tokens)")
