#!/usr/bin/env python3
"""Build the full corpus for cross-piṭaka clustering experiments.

Creates data/raw-full/ containing files from all three piṭakas:
  - Sutta: PTS-reorganised files from data/raw-pts/
  - Vinaya: VRI files from pali-canon/data/vri-raw/
  - Abhidhamma: VRI files from pali-canon/data/vri-raw/

Includes only: canon, commentary, subcommentary, and paracanonical texts.
Excludes: manuals, grammars, chronicles, devotional texts, nīti, etc.

Correct labels are assigned from the inventory CSV, overriding VRI's
suffix-based labelling where needed.
"""

import csv
import shutil
from pathlib import Path

from data_loader import PTS_NAMES
from reorganise_pts import ALL_REPLACED

INVENTORY_CSV = Path(__file__).resolve().parent.parent / "data" / "vri_inventory.csv"
SUTTA_PTS_DIR = Path(__file__).resolve().parent.parent / "data" / "raw-pts"
VRI_RAW_DIR = Path.home() / "pali-canon" / "data" / "vri-raw"
DST_DIR = Path(__file__).resolve().parent.parent / "data" / "raw-full"

# Labels to include in the experiments
INCLUDED_LABELS = {"canon", "commentary", "subcommentary", "paracanonical"}

# Piṭakas to include (exclude "extra" — grammars, chronicles, etc.)
INCLUDED_PITAKAS = {"sutta", "vinaya", "abhidhamma"}

# Sutta files that exist in PTS-reorganised form (data/raw-pts/).
# These replace the VRI originals.
# All other sutta files are copied from raw-pts/ as well (they were
# copied unchanged by reorganise_pts.py).
SUTTA_PTS_ONLY = True  # Use raw-pts/ for all Sutta files

# PTS-reorganised Sutta files (an*, ja*, ps*, ap*, mna*, spk*, mp*, dhpa*,
# ita*, jaa*, sdp*, mnt*, spkt*, mpt*) are labelled by their CST suffix; they
# are all Sutta-piṭaka. .mul -> canon, .att -> commentary, .tik -> subcommentary.
PTS_SUFFIX_LABEL = {"mul": "canon", "att": "commentary", "tik": "subcommentary"}


def pts_file_label(fname: str) -> tuple:
    """(label, pitaka) for a PTS-reorganised Sutta file, from its suffix.

    e.g. 'mna1.att.txt' -> ('commentary', 'sutta'); 'an1.mul.txt' ->
    ('canon', 'sutta'). Replaces the previous hardcoded canon/sutta, which
    mislabelled every reorganised commentary/sub-commentary as canon.
    """
    parts = fname.split(".")
    suffix = parts[1] if len(parts) >= 3 else "mul"
    return PTS_SUFFIX_LABEL.get(suffix, "canon"), "sutta"


def load_inventory():
    """Load the inventory CSV and return rows as dicts."""
    with open(INVENTORY_CSV) as f:
        return list(csv.DictReader(f))


def main():
    rows = load_inventory()
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # Clear any previous contents
    for f in DST_DIR.glob("*.txt"):
        f.unlink()

    copied = 0
    skipped = 0
    label_counts = {}
    pitaka_counts = {}

    # Get the list of files available in raw-pts/
    pts_files = {f.name for f in SUTTA_PTS_DIR.glob("*.txt")}

    for row in rows:
        vri_file = row["vri_file"]
        correct_label = row["correct_label"]
        pitaka = row["pitaka"]

        # Only include files from the three piṭakas with the right labels
        if pitaka not in INCLUDED_PITAKAS or correct_label not in INCLUDED_LABELS:
            skipped += 1
            continue

        # Determine source path
        if pitaka == "sutta":
            # Use PTS-reorganised version if available
            if vri_file in pts_files:
                src = SUTTA_PTS_DIR / vri_file
            else:
                # This shouldn't happen for included labels, but handle it
                src = VRI_RAW_DIR / vri_file
        else:
            # Vinaya, Abhidhamma: use VRI raw files
            src = VRI_RAW_DIR / vri_file

        if not src.exists():
            print(f"  WARNING: {src} not found, skipping")
            skipped += 1
            continue

        # Copy to destination
        shutil.copy2(src, DST_DIR / vri_file)
        copied += 1
        label_counts[correct_label] = label_counts.get(correct_label, 0) + 1
        pitaka_counts[pitaka] = pitaka_counts.get(pitaka, 0) + 1

    # Also copy PTS-reorganised files that don't exist in VRI
    # (an1-an5, ja1-ja6, ps1-ps2, ap1-ap2)
    pts_only_files = pts_files - {row["vri_file"] for row in rows}
    for fname in sorted(pts_only_files):
        # Label by CST suffix (canon/commentary/subcommentary), not hardcoded canon
        src = SUTTA_PTS_DIR / fname
        shutil.copy2(src, DST_DIR / fname)
        copied += 1
        label, pitaka = pts_file_label(fname)
        label_counts[label] = label_counts.get(label, 0) + 1
        pitaka_counts[pitaka] = pitaka_counts.get(pitaka, 0) + 1

    # Remove VRI files that were replaced by PTS versions, decrementing the
    # bucket matching each file's ACTUAL label (ALL_REPLACED now includes
    # commentary/sub-commentary VRI files, not only canon .mul).
    inv_by_file = {row["vri_file"]: row for row in rows}
    removed = 0
    for fname in ALL_REPLACED:
        p = DST_DIR / fname
        if p.exists():
            p.unlink()
            removed += 1
            copied -= 1
            r = inv_by_file.get(fname)
            if r is not None:
                label_counts[r["correct_label"]] = label_counts.get(r["correct_label"], 0) - 1
                pitaka_counts[r["pitaka"]] = pitaka_counts.get(r["pitaka"], 0) - 1

    print(f"Built full corpus in {DST_DIR.name}/")
    print(f"  Copied: {copied} files")
    print(f"  Skipped: {skipped} files (excluded labels)")
    print(f"  Removed: {removed} VRI files replaced by PTS versions")
    print()
    print("By label:")
    for label in ["canon", "commentary", "subcommentary", "paracanonical"]:
        print(f"  {label}: {label_counts.get(label, 0)}")
    print()
    print("By piṭaka:")
    for p in ["sutta", "vinaya", "abhidhamma"]:
        print(f"  {p}: {pitaka_counts.get(p, 0)}")

    # Write a label map file for the experiment to use
    label_map_path = DST_DIR / "LABELS.csv"
    with open(label_map_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "pitaka", "text_name"])
        for row in rows:
            if row["pitaka"] not in INCLUDED_PITAKAS:
                continue
            if row["correct_label"] not in INCLUDED_LABELS:
                continue
            fname = row["vri_file"]
            if fname in ALL_REPLACED:
                continue
            if not (DST_DIR / fname).exists():
                continue
            writer.writerow([fname, row["correct_label"], row["pitaka"], row["text_name"]])
        # Add PTS-only files, labelled by suffix (not hardcoded canon)
        for fname in sorted(pts_only_files):
            if not (DST_DIR / fname).exists():
                continue
            stem = fname.split(".")[0]
            name = PTS_NAMES.get(stem, stem)
            label, pitaka = pts_file_label(fname)
            writer.writerow([fname, label, pitaka, name])

    # Post-build guard: every reorganised .att is commentary, every .tik is
    # subcommentary, every .mul is canon — fail loudly if this ever regresses.
    with open(label_map_path) as f:
        for row in csv.DictReader(f):
            fn, lab = row["filename"], row["label"]
            if fn.endswith(".att.txt") and row["filename"].split(".")[0] in PTS_NAMES:
                assert lab == "commentary", f"{fn} labelled {lab}, expected commentary"
            elif fn.endswith(".tik.txt") and row["filename"].split(".")[0] in PTS_NAMES:
                assert lab == "subcommentary", f"{fn} labelled {lab}, expected subcommentary"

    print(f"\nWrote label map to {label_map_path.name}")


if __name__ == "__main__":
    main()
