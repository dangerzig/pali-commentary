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
        # These are all canonical Sutta files
        src = SUTTA_PTS_DIR / fname
        shutil.copy2(src, DST_DIR / fname)
        copied += 1
        label_counts["canon"] = label_counts.get("canon", 0) + 1
        pitaka_counts["sutta"] = pitaka_counts.get("sutta", 0) + 1

    # Remove VRI files that were replaced by PTS versions
    # (AN nipāta files, original Jātaka/Paṭis/Apadāna files)
    vri_replaced = {
        "s0401m.mul.txt", "s0402m1.mul.txt", "s0402m2.mul.txt", "s0402m3.mul.txt",
        "s0403m1.mul.txt", "s0403m2.mul.txt", "s0403m3.mul.txt",
        "s0404m1.mul.txt", "s0404m2.mul.txt", "s0404m3.mul.txt", "s0404m4.mul.txt",
        "s0513m.mul.txt", "s0514m.mul.txt",
        "s0517m.mul.txt",
        "s0510m1.mul.txt", "s0510m2.mul.txt",
    }
    removed = 0
    for fname in vri_replaced:
        p = DST_DIR / fname
        if p.exists():
            p.unlink()
            removed += 1
            copied -= 1
            label_counts["canon"] -= 1
            pitaka_counts["sutta"] -= 1

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
            if fname in vri_replaced:
                continue
            if not (DST_DIR / fname).exists():
                continue
            writer.writerow([fname, row["correct_label"], row["pitaka"], row["text_name"]])
        # Add PTS-only files
        for fname in sorted(pts_only_files):
            if not (DST_DIR / fname).exists():
                continue
            # Determine PTS name
            stem = fname.split(".")[0]
            pts_name_map = {
                "an1": "A I", "an2": "A II", "an3": "A III", "an4": "A IV", "an5": "A V",
                "ja1": "J I", "ja2": "J II", "ja3": "J III",
                "ja4": "J IV", "ja5": "J V", "ja6": "J VI",
                "ps1": "Paṭis I", "ps2": "Paṭis II",
                "ap1": "Ap I", "ap2": "Ap II",
            }
            name = pts_name_map.get(stem, stem)
            writer.writerow([fname, "canon", "sutta", name])

    print(f"\nWrote label map to {label_map_path.name}")


if __name__ == "__main__":
    main()
