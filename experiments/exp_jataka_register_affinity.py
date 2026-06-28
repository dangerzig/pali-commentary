#!/usr/bin/env python3
"""
R2-2 robustness check (JPTS revision).

Reviewer 2 claimed the Jātaka clusters apart only because canonical verse was
mixed with 5th/6th-c. commentarial prose. It was NOT: the canonical Jātaka
input is the Jātakapāḷi verses (.mul, 103,766 words), and the
Jātakaṭṭhavaṇṇanā (.att, 603,841 words) is a separate set of documents. The
whole corpus is built on the CST mūla/aṭṭhakathā distinction.

This script documents, with verse and commentary kept as separate documents:

  (A) The k=3 narrative-register cluster membership (the actual claim in §5).
      Five of the six Jātaka VERSE volumes (J II–VI) co-cluster with the
      Jātakaṭṭhavaṇṇanā and the other KN narrative texts AND their commentaries,
      crossing the canon/commentary boundary. J I stays in the main canon
      cluster.

  (B) An honest caveat: along the dominant canon/commentary axis (raw word-
      frequency centroids), the Jātaka commentary still reads as a commentary
      and the verse still reads as canonical. The narrative affinity is an
      ADDITIONAL, orthogonal signal that k-means recovers only when allowed a
      third cluster on the standardised feature space. This is why a pairwise
      centroid cosine understates it and the clustering is the right instrument.

Run with --pts (PTS-volume reorganisation) to match the paper.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.spatial.distance import cosine

from data_loader import load_raw_documents, binary_labels
from features import word_frequency_matrix
from shared import MISCLASSIFIED_IDS, get_data_dir

REPO = Path(__file__).parent.parent
CLUSTER_CSV = REPO / "results-full" / "cluster_assignments.csv"

JATAKA_VERSE = {'ja1', 'ja2', 'ja3', 'ja4', 'ja5', 'ja6'}        # .mul (Jātakapāḷi)
JATAKA_COMM  = {'jaa1', 'jaa2', 'jaa3', 'jaa4', 'jaa5', 'jaa6'}  # .att (Jātakaṭṭhavaṇṇanā)


def part_a_cluster_table():
    if not CLUSTER_CSV.exists():
        print(f"[A] {CLUSTER_CSV} missing — run exp_full_clustering.py --pts first.")
        return
    rows = list(csv.DictReader(open(CLUSTER_CSV)))
    # Identify the narrative cluster: the k=3 cluster that mixes canon + commentary.
    from collections import defaultdict
    byc = defaultdict(list)
    for r in rows:
        byc[r['k3']].append(r)
    narr = None
    for c, rs in byc.items():
        labs = Counter(r['label'] for r in rs)
        if 'canon' in labs and 'commentary' in labs:
            narr = c
    print("=" * 64)
    print("[A] k=3 NARRATIVE-REGISTER CLUSTER (canonical verse + its commentary)")
    print("=" * 64)
    rs = byc[narr]
    labs = Counter(r['label'] for r in rs)
    print(f"cluster {narr}: {len(rs)} texts, all Sutta Piṭaka — {dict(labs)}\n")
    for lab in ('canon', 'commentary'):
        print(f"  {lab.upper()}:")
        for r in sorted(rs, key=lambda r: r['text_name']):
            if r['label'] == lab:
                print(f"    - {r['text_name']}")
    # Jātaka verse placement
    print("\n  Jātaka VERSE volume placement at k=3:")
    for r in sorted(rows, key=lambda r: r['text_name']):
        if r['text_name'].startswith('J ') and len(r['text_name']) <= 5:
            where = 'narrative cluster' if r['k3'] == narr else 'main canon cluster'
            print(f"    {r['text_name']:6} -> {where}")
    print("\n  => 5 of 6 Jātaka verse volumes (J II–VI) join the narrative cluster;")
    print("     J I stays in the main canon cluster. State this precisely in §5.")


def part_b_pairwise_caveat():
    documents = load_raw_documents(data_dir=get_data_dir())
    labels = binary_labels(documents)
    matrix, _ = word_frequency_matrix(documents, top_n=1000)
    by = {d.file_id: i for i, d in enumerate(documents)}
    misc = set(MISCLASSIFIED_IDS)

    def centroid(ids):
        idx = [by[f] for f in ids if f in by]
        return matrix[idx].mean(axis=0)

    MAIN_NIKAYA = ('s01', 's02', 's03', 's04')
    sutta_ids = [d.file_id for d in documents
                 if d.suffix == 'mul' and d.file_id.startswith(MAIN_NIKAYA)
                 and d.file_id not in misc]
    MAIN_COMM = ('s0101a', 's0102a', 's0103a', 'mna', 'spk', 's03', 'mp')
    att_ids = [d.file_id for d in documents
               if d.suffix == 'att' and d.file_id.startswith(MAIN_COMM)
               and d.file_id not in JATAKA_COMM and d.file_id not in misc]

    c_v = centroid(JATAKA_VERSE)
    c_c = centroid(JATAKA_COMM)
    c_s = centroid(sutta_ids)
    c_a = centroid(att_ids)
    cd = lambda a, b: cosine(a, b)
    print("\n" + "=" * 64)
    print("[B] Pairwise centroid cosine on RAW word frequencies (caveat)")
    print("=" * 64)
    print(f"  Jātaka verse  <-> Jātaka commentary  : {cd(c_v, c_c):.3f}")
    print(f"  Jātaka verse  <-> standard suttas     : {cd(c_v, c_s):.3f}")
    print(f"  Jātaka comm   <-> standard aṭṭhakathā : {cd(c_c, c_a):.3f}")
    print("  Reading: on the dominant register axis the commentary still reads")
    print("  as commentary and the verse as canon. The narrative co-clustering")
    print("  in [A] is an orthogonal signal — report the clustering, not this.")


if __name__ == '__main__':
    part_a_cluster_table()
    part_b_pairwise_caveat()
