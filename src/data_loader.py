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
from collections import Counter
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Regex for tokenizing Pāli text: split on anything that isn't a Pāli letter
PALI_TOKEN_RE = re.compile(r'[^a-zA-ZāīūṭḍṇṅñṃḷĀĪŪṬḌṆṄÑṂḶ]+')

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
LEMMATIZED_DIR = Path(__file__).parent.parent / "data" / "lemmatized"

# Also check pali-canon lemmatized data for canonical texts
CANON_LEMMATIZED_DIR = Path.home() / "pali-canon" / "data" / "lemmatized"

LABEL_MAP = {
    'mul': 'canon',
    'att': 'commentary',
    'tik': 'subcommentary',
    'nrf': 'paracanonical',
}

# Mapping from VRI file codes to human-readable names
VRI_NAMES = {
    's0101': 'DN vol.1 (Sīlakkhandhavagga)',
    's0102': 'DN vol.2 (Mahāvagga)',
    's0103': 'DN vol.3 (Pāthikavagga)',
    's0201': 'MN vol.1 (Mūlapaṇṇāsa)',
    's0202': 'MN vol.2 (Majjhimapaṇṇāsa)',
    's0203': 'MN vol.3 (Uparipaṇṇāsa)',
    's0301': 'SN vol.1 (Sagāthāvagga)',
    's0302': 'SN vol.2 (Nidānavagga)',
    's0303': 'SN vol.3 (Khandhavagga)',
    's0304': 'SN vol.4 (Saḷāyatanavagga)',
    's0305': 'SN vol.5 (Mahāvagga)',
    's0401': 'AN vol.1 (Ekakanipāta–Tikanipāta)',
    's0402': 'AN vol.2 (Catukkanipāta–Chakanipāta)',
    's0403': 'AN vol.3 (Sattakanipāta–Navakanipāta)',
    's0404': 'AN vol.4 (Dasakanipāta–Ekādasakanipāta)',
    's0501': 'Khuddakapāṭha',
    's0502': 'Dhammapada',
    's0503': 'Udāna',
    's0504': 'Itivuttaka',
    's0505': 'Suttanipāta',
    's0506': 'Vimānavatthu',
    's0507': 'Petavatthu',
    's0508': 'Theragāthā',
    's0509': 'Therīgāthā',
    's0510': 'Jātaka',
    's0511': 'Niddesa (Mahāniddesa)',  # Actually split: s0515m, s0516m
    's0512': 'Paṭisambhidāmagga',
    's0513': 'Apadāna',
    's0514': 'Buddhavaṃsa + Cariyāpiṭaka',
    's0515': 'Mahāniddesa',
    's0516': 'Cūḷaniddesa',
    's0517': 'Nettippakaraṇa',
    's0518': 'Milindapañha',
    's0519': 'Peṭakopadesa',
    's0520': 'Suttasaṅgaha',
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
        base = self.file_id[:-1]  # strip the letter suffix (m/a/t)
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
    # Normalize niggahīta variants
    text = text.replace('ṁ', 'ṃ')
    tokens = PALI_TOKEN_RE.split(text.lower())
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
    for filepath in sorted(data_dir.glob('s*.txt')):
        file_id, suffix, label = parse_filename(filepath.name)

        if include_labels and label not in include_labels:
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
    return [0 if d.label == 'canon' else 1 for d in documents]


if __name__ == '__main__':
    # Quick test
    docs = load_raw_documents()
    print(f"Loaded {len(docs)} documents")

    for label in ['canon', 'commentary', 'subcommentary', 'paracanonical']:
        subset = [d for d in docs if d.label == label]
        total_tokens = sum(len(d.tokens) for d in subset)
        print(f"  {label}: {len(subset)} docs, {total_tokens:,} tokens")

    print(f"\nFirst 5 documents:")
    for d in docs[:5]:
        print(f"  {d.file_id}: {d.label} ({len(d.tokens):,} tokens)")
