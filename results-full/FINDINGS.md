# Cross-Piṭaka Clustering Experiment: Findings

## Corpus

137 texts from all three piṭakas of the Pāli Tipiṭaka:

| Category | Sutta | Vinaya | Abhidhamma | Total |
|---|---|---|---|---|
| Canon | 40 | 5 | 13 | 58 |
| Commentary | 39 | 6 | 3 | 48 |
| Sub-commentary | 17 | 8 | 4 | 29 |
| Paracanonical | 2 | 0 | 0 | 2 |
| **Total** | **98** | **19** | **20** | **137** |

Sutta canonical texts use PTS volume boundaries (AN merged from 11 nipāta files
to 5 PTS volumes; Jātaka split from 2 to 6; Paṭisambhidāmagga split 1 to 2;
Apadāna re-split at correct vagga boundary). Vinaya and Abhidhamma use VRI file
boundaries, which are 1:1 with PTS for Vinaya and close to 1:1 for Abhidhamma
(only Yamaka 3 VRI files vs 2 PTS vols and Paṭṭhāna 5 VRI vs 3+ PTS differ).

Excluded: manuals (Visuddhimagga, Abhidhammatthasaṅgaha, Vinayavinicchaya, etc.),
grammars, chronicles, devotional texts, nīti literature, council Q&A texts,
sub-sub-commentaries. These are all later Pāli literature with no canonical status.

Features: relative frequencies of the 1,000 most common words across the corpus,
z-scored before clustering and PCA.

## Results

### k=2: Perfect canon/non-canon separation (100%)

The binary register distinction identified in the Sutta Piṭaka paper extends
perfectly to the entire Tipiṭaka. All 60 canonical and paracanonical texts
cluster together; all 77 commentary and sub-commentary texts cluster together.
Not a single Vinaya or Abhidhamma text is misclassified.

This confirms that the discourse register and exegetical register are not
artefacts of the Sutta Piṭaka alone but pervade the entire Pāli literary
tradition.

### k=3: The narrative/verse register

A third cluster emerges containing 30 texts, all from the Sutta Piṭaka. This
cluster mixes canonical verse/narrative texts with their own commentaries:

**Canonical texts in this cluster**: Jātaka (vols II-VI), Apadāna (both vols),
Vimānavatthu, Petavatthu, Theragāthā, Therīgāthā, Buddhavaṃsa, Cariyāpiṭaka.

**Commentaries in this cluster**: all 7 volumes of the Jātakaṭṭhavaṇṇanā,
Dhammapadaṭṭhakathā, Manorathapūraṇī I, the Paramatthadīpanī commentaries on
Vimānavatthu, Petavatthu, Theragāthā, Therīgāthā, Cariyāpiṭaka, the
Visuddhajanavilāsinī (Apadāna commentary), and the Madhuratthavilāsinī
(Buddhavaṃsa commentary).

This is a striking finding: the narrative/verse register is so distinctive that
it *overrides the canon/commentary boundary*. The Jātaka stories share more
vocabulary with the Jātaka commentary than either shares with standard suttas or
standard aṭṭhakathā. The register is defined by narrative and verse vocabulary
rather than by the teaching/explaining distinction that separates the other two
clusters.

This confirms the paper's observation that the Jātaka is consistently the hardest
canonical text to classify in binary experiments, and extends it: there is a
coherent third register shared by the verse and narrative portions of the
Khuddaka Nikāya and their commentaries.

### k=4: The Paṭṭhāna register

All five Paṭṭhāna volumes separate into their own cluster. The Paṭṭhāna is the
most repetitive and formulaic text in the entire Tipiṭaka, consisting of
exhaustive enumeration of conditional relations. Its hyper-formulaic style
constitutes a distinct register.

The remaining three clusters are: (1) general canon, (2) narrative/verse
(same texts as k=3 cluster 2, plus Jātaka I and Dhammapada), (3) commentary +
sub-commentary + 4 Abhidhamma canonical texts + 1 paracanonical.

### k=5: Yamaka joins the Paṭṭhāna

At k=5, two Yamaka volumes (I and III) form their own 2-text cluster. Like the
Paṭṭhāna, the Yamaka is extremely formulaic, consisting of paired questions
and answers. The remaining clusters are: general canon (42 texts), Sutta
narrative/commentary mix (49), sub-commentary + some commentary (39), and
Paṭṭhāna (5).

## Key observations

1. **The binary register distinction is universal.** It holds with 100% accuracy
   across all three piṭakas. Vinaya and Abhidhamma commentaries use the same
   exegetical register as Sutta commentaries.

2. **Commentary and sub-commentary share a single register.** At no k value do
   sub-commentaries separate from commentaries. They are the same register:
   the exegetical voice.

3. **A third register exists: narrative/verse.** The Khuddaka Nikāya verse and
   narrative texts, together with their commentaries, form a coherent cluster
   distinct from both the discourse register and the standard exegetical
   register. This is defined by narrative vocabulary rather than by canonical
   status.

4. **Abhidhamma formulaic texts have their own registers.** The Paṭṭhāna and
   Yamaka are so repetitive that they separate from other canonical texts at
   k=4 and k=5 respectively.

5. **Vinaya never forms its own cluster.** At every k value, Vinaya texts cluster
   with the broader canonical or commentary groups. The Vinaya does not have a
   distinctive register separate from the Sutta Piṭaka.

6. **The narrative register overrides canonical status.** This is perhaps the most
   significant finding: for verse/narrative KN texts, the genre similarity
   between a canonical text and its commentary is stronger than the register
   distinction between canon and commentary. This suggests that the
   discourse/exegetical register distinction is primarily a feature of prose
   didactic texts, not of all Pāli literature.

## Implications for the paper

The paper currently describes "two dominant registers in the Sutta Piṭaka and its
commentaries" and notes (Section 4.3) that the Jātaka is the hardest text to
classify. These cross-piṭaka results:

- Confirm that the two registers extend beyond the Sutta Piṭaka (universal finding)
- Identify the narrative/verse register as a genuine third register, not just noise
- Explain *why* the Jātaka is hard to classify: it belongs to a different register
- Show that the Abhidhamma formulaic style is a fourth (and fifth) register
- Demonstrate that Vinaya shares the same registers as Sutta
- Strengthen the claim that the exegetical register is a single tradition
  (commentary = sub-commentary in register terms)
