### [1] correctness | high | src/classification.py
LOC: run_loo_cv (lines 55-105), as used by experiments/exp3_supervised.py lines 79-90
ISSUE: The feature matrices passed to run_loo_cv are built on the ENTIRE corpus before cross-validation. tfidf_matrix() fits the vocabulary AND the IDF weights on all documents (features.py line 95 vectorizer.fit_transform(texts)), and word_frequency_matrix() selects the top-N words by mean relative frequency across all documents (features.py lines 47-56). The LOO-CV pipeline (make_pipeline(StandardScaler(), clf)) only re-fits the scaler and classifier per fold; the feature selection / IDF that defines the columns has already seen every test document. This is classic feature-selection-before-CV leakage and inflates the reported LOO accuracies. With LOO (test fold = 1 doc) the IDF/top-N leakage is mild per fold but is still leakage and is reported as an unbiased generalization estimate in the paper.
FIX: Make the corpus-dependent feature extraction a fold-internal Pipeline step so vocabulary, IDF, and top-N selection are fit only on the training fold. Concretely: (1) For TF-IDF, drop tfidf_matrix() as a precomputed input and instead build the estimator as `make_pipeline(TfidfVectorizer(max_features=..., ngram_range=..., analyzer=...), StandardScaler(with_mean=False) , clf)` and pass the list of raw token strings (`[' '.join(d.tokens) for d in documents]`) as X to cross_val_predict, so the vectorizer is re-fit per fold. (2) For word_frequency_matrix top-N selection, wrap the relative-frequency computation + top-N column selection as a custom sklearn transformer (fit() learns top-N words from the training rows only; transform() applies them), or approximate with sklearn's CountVectorizer + max_features + a normalizer inside the pipeline; place it first in the Pipeline. (3) Update run_loo_cv to accept either a raw-text/transformer-based estimator or to take a pipeline factory so feature extraction is inside the CV loop. Apply the same change to every caller (exp3, exp_lemmatized, exp5, exp7). Note that for the precomputed embedding inputs (exp4/exp5) only StandardScaler matters and it is already inside the pipeline, so those are fine. After fixing, re-run and report the (likely slightly lower) LOO accuracies; given the unsupervised nature of the leaked statistics the change to reported numbers should be small but the methodology becomes correct.

### [2] correctness | high | src/data_loader.py
LOC: tokenize_pali, lines 204-213 (especially line 211 vs 212)
ISSUE: Niggahita normalization runs BEFORE lowercasing: text.replace('ṁ','ṃ') on line 211, then text.lower() on line 212. An uppercase niggahita Ṁ (U+1E40, dot-above) is not handled by the replace, and only lowercases to ṁ (U+1E41) afterward. ṁ (U+1E41) is NOT in PALI_TOKEN_RE's character class (only ṃ U+1E43 is), so it is treated as a token separator. Any word containing an uppercase niggahita (e.g. a sentence-initial 'Saṁyutta', or all-caps headings) gets split at that point and corrupted. Verified: replace+lower on 'Ṁ' yields the unnormalized 'ṁ', and PALI_TOKEN_RE.split treats it as a separator.
FIX: Defensive hardening (low priority, since the VRI corpus is dot-below only). Reorder to lowercase first, then normalize, and NFC-normalize to collapse combining forms. Replace lines 210-212 with:

    import unicodedata  # at module top
    ...
    # Normalize niggahīta variants (dot-above -> dot-below), all cases
    text = unicodedata.normalize('NFC', text).lower()
    text = text.replace('ṁ', 'ṃ')  # U+1E41 -> U+1E43; after lower(), Ṁ U+1E40 is already ṁ
    tokens = PALI_TOKEN_RE.split(text)

Lowercasing first means uppercase Ṁ (U+1E40) becomes ṁ (U+1E41) before the replace, so it is correctly mapped to ṃ. NFC normalization collapses decomposed m+U+0323 into U+1E43. Optionally add an explicit test (tokenize_pali('SAṀYUTTA') == ['saṃyutta']) to lock in the behavior. Given the corpus never contains these forms, this can be treated as cleanup rather than an urgent fix.

### [3] correctness | medium | src/clustering.py
LOC: run_hdbscan (lines 116-138) -> align_cluster_labels (lines 15-32)
ISSUE: run_hdbscan only rejects results with n_clusters < 2 (line 116). If HDBSCAN finds 3+ clusters, the code proceeds, runs KNN to reassign noise, then passes labels containing values {0,1,2,...} to evaluate_clustering -> align_cluster_labels. align_cluster_labels assumes strictly binary labels and computes 1 - cluster_labels (line 28/31), which produces -1 for any label==2, silently yielding a meaningless 'aligned' result and accuracy rather than an error. The docstring even warns 'Only works for binary (0/1) labels' but nothing enforces it.
FIX: Two-part defensive fix, matching the existing error-dict convention.

1. In run_hdbscan, tighten the guard to reject any non-binary result (3+ real clusters cannot be unambiguously aligned to binary truth). Replace the line 116 check:

    if n_clusters != 2:
        return {
            'method': 'HDBSCAN',
            'accuracy': None,
            'error': f'Found {n_clusters} clusters (need exactly 2), {n_noise} noise points',
            'n_noise': int(n_noise),
            'n_clusters': n_clusters,
        }

This is consistent with the other run_* functions, which all force n_clusters=2, and keeps HDBSCAN's accuracy comparable to them. (If, instead, the intent is to evaluate HDBSCAN's natural multi-cluster output, you must switch evaluate_clustering to use scipy.optimize.linear_sum_assignment / adjusted_rand_score over multi-class labels, not the 1 - labels binary flip — but the simpler reject path is the minimal correct fix.)

2. Add a guard in align_cluster_labels so the silent multi-class miscomputation can never recur from any caller:

    cluster_labels = np.array(cluster_labels)
    if not set(np.unique(cluster_labels)).issubset({0, 1}):
        raise ValueError(
            f'align_cluster_labels only supports binary 0/1 labels; '
            f'got {sorted(set(np.unique(cluster_labels)))}. Use Hungarian matching for multi-class.'
        )

This turns the current silent, meaningless 'accuracy' into an explicit error.

### [4] correctness | medium | src/features.py
LOC: word_frequency_matrix, top-N selection, lines 47-56
ISSUE: Top-N words are selected by the unweighted mean of per-document relative frequencies (line 52 np.mean(freqs)). This weights every document equally regardless of length, so a word that is very frequent in one short document outranks a word moderately frequent across the whole corpus. Standard MFW/stylometric selection ranks by corpus-wide relative frequency (total count of word / total tokens). The current ranking can select idiosyncratic short-document words as 'top' features, which is both a likely mismatch with the paper's stated 'most frequent words' method and a source of noisy features. (Separately, this selection is also part of the leakage in finding 1.)
FIX: Rank candidate words by corpus-aggregate relative frequency instead of the per-document mean. Replace lines 48-52:

    total_tokens = sum(doc_totals)
    rel_freqs = {}
    for word in all_words:
        word_total = sum(c[word] for c in counters)
        rel_freqs[word] = word_total / total_tokens if total_tokens > 0 else 0

(Equivalently, since the denominator is constant across words, ranking by raw aggregate count sum(c[word] for c in counters) gives the same top-N and is cheaper.) Keep the matrix-construction block (lines 58-62) unchanged: per-document relative frequencies are the correct cell values; only the selection criterion changes. Update the docstring on lines 5, 24, and the line-54 comment to say "top-N by corpus-aggregate relative frequency (most frequent words across the corpus)," matching results-full/FINDINGS.md. If the per-document mean is genuinely intended to reproduce the original paper, it must be documented and justified explicitly (with a citation to the paper's stated method), because as written it deviates from both standard MFW practice and this repo's own FINDINGS.md description. Note the existing tests in tests/test_features.py (e.g. the top-N selection tests around lines 21-37, 104-117) should be checked/updated, as they may currently encode the per-document-mean ordering.

### [5] correctness | low | src/data_loader.py
LOC: binary_labels (lines 367-373) and load_raw_documents include filter (line 238)
ISSUE: binary_labels assigns canon=0 and 'everything else'=1, which includes label=='unknown' (any file whose suffix is not in LABEL_MAP). Combined with parse_filename returning 'unknown' for unexpected suffixes, an unrecognized file would be silently treated as a valid non-canon training example rather than being flagged. There is no assertion that all loaded docs have a known label.
FIX: Centralize the guard rather than asserting in every caller. Best: fail fast at load time in load_raw_documents. After computing `file_id, suffix, label = parse_filename(...)`, when include_labels is None and label == 'unknown', either skip with a logged warning or raise. Minimal version:

```python
file_id, suffix, label = parse_filename(filepath.name)
if label == 'unknown':
    warnings.warn(f"Unrecognized suffix '{suffix}' in {filepath.name}; skipping")
    continue
if include_labels is not None and label not in include_labels:
    continue
```

(Skipping is safer than raising for batch runs; use raise if you prefer hard failure.) If you want the assertion form the finding suggests, put it once inside binary_labels so it protects all callers:

```python
def binary_labels(documents):
    valid = {'canon','commentary','subcommentary','paracanonical'}
    bad = {d.label for d in documents} - valid
    assert not bad, f"Unknown labels reached classification: {bad}"
    return [0 if d.label == 'canon' else 1 for d in documents]
```

Either approach removes the silent mislabel. No data change is needed today (current corpus is clean); this is a guardrail for future files with unexpected suffixes.

### [6] sandhi | high | src/build_sandhi_edition.py
LOC: _restore_host (lines 143-155); used by Policy B at line 205
ISSUE: _restore_host only shortens the host's final long vowel when `not dpd.is_headword(host)`. For very common quotative forms whose lengthened sandhi surface is ALSO a DPD headword homograph, the guard is False and the lengthened (wrong) host is kept. Verified against DPD+corpus: gacchatīti -> host 'gacchatī' (should be 'gacchati'), atthīti -> 'atthī' (should 'atthi'), natthīti -> 'natthī', cāti -> 'cā' (should 'ca'), dasāti -> 'dasā' (should 'dasa'), pavattatīti -> 'pavattatī', uppajjatīti -> 'uppajjatī', etc. In a 15-file subset this affected ~1496 peel occurrences across 611 distinct forms (the genuinely-long-stem cases like bhagavā are correctly left alone, but the lengthened-sandhi cases are wrongly left lengthened). These wrongly-lengthened hosts become distinct word-frequency features from their correctly-divided counterparts, diluting/splitting counts and directly affecting the k-means/LR register comparison that the edition exists to validate.
FIX: Make `_restore_host` (or the Policy-B host selection) decide by PART OF SPEECH, not by mere headword presence, and treat the verb case as the well-defined target:

1. Compute `shortened = host[:-1] + _SHORTEN[host[-1]]` for a final long vowel.
2. Shorten when the lengthening is a sandhi artefact, i.e. when the SHORT form is a finite/indeclinable host but the LONG form is only a participle/letter homograph. Concretely, shorten if:
   - `'pr'`/finite-verb (or `'ind'`, `'card'`) is in `dpd.pos_tags(shortened)` AND the long form's POS is a strict subset that does NOT add a noun/adjective/pronoun reading beyond `{prp, letter}` — e.g. `gacchatī` (prp only) -> `gacchati` (pr); `cā` (letter) -> `ca` (ind); `dasā` (adj/fem) -> keep careful (see below).
3. Do NOT shorten when the long form is a legitimate inflection in its own right: a feminine nominative/genitive (`-ā` fem of an `-a` stem) or a nominative plural (`puttā`, `dāsā`) or a genuine long-vowel stem (`bhagavā`, `sādhū`, `garū`, `mātā`, `pitā`, `āpattī`). These show a noun/adj/pron POS on the LONG form that is not merely the `prp`/`letter` homograph.
4. Do NOT use `pieces[-2]` as the canonical host: DPD lists the two readings in inconsistent order (short-first for mātāti/puttāti, long-first for jātāti/etassāti), so it would over-shorten correct forms.

Implementation sketch: restrict the shortening to the unambiguous verb class first (highest-value, lowest-risk):
```python
def _restore_host(host, dpd):
    if host and host[-1] in _SHORTEN:
        shortened = host[:-1] + _SHORTEN[host[-1]]
        long_pos = dpd.pos_tags(host)
        short_pos = dpd.pos_tags(shortened)
        # (a) original rule: long form isn't a word at all
        if not dpd.is_headword(host) and dpd.is_headword(shortened):
            return shortened
        # (b) finite-verb sandhi artefact: short is a finite verb,
        #     long is at most a participle homograph (no noun/pron/ind reading)
        if (short_pos & FINITE_VERB_POS) and long_pos and long_pos <= {"prp"}:
            return shortened
    return host
```
This fixes the clean cases (gacchatī, pavattatī, uppajjatī, viharatī, passatī, sikkhatī, hontī, karonti, etc.) without touching mātā/pitā/puttā/jātā/veditabbā/etassā/bhagavā. The `cā`/`dasā`/`nāmā` indeclinable-and-numeral cases can be added with an analogous `long_pos <= {"letter"}` / explicit small curated POS check, verified case-by-case, since they carry higher ambiguity. Validate by re-running the build and confirming the previously-split pairs (gacchati/gacchatī) merge while the feminine/plural inflections are unchanged.

### [7] sandhi | high | experiments/exp_sandhi_audit.py
LOC: main, Check 2, lines 109-117 (`if policy == 'peel' and dpd.is_finite_verb(inp)`)
ISSUE: Check 2 flags every peel whose INPUT token is a finite verb as a 'false split' and asserts 'After the POS guard this should be zero.' But the builder intentionally peels a quoted finite verb when its host is also a finite verb (builder condition is `ambiguous = is_finite_verb(tok) AND NOT host_is_verb`; only the AND-NOT case is suppressed). The audit omits the `not host_is_verb` clause, so it reports legitimate peels as errors. Measured on a 15-file subset: Check 2 reports 2118 'false splits' (hotīti, abhāsitthāti, ahosīti, vadāmīti, ...) and ALL 2118 are peels whose host is itself a finite verb, i.e. correct. The audit therefore cannot substantiate the paper's 'finite-verb peels ~= 0' claim and would alarm spuriously.
FIX: In exp_sandhi_audit.py Check 2 (lines 109-117), replicate the builder's host_is_verb guard so the audit counts a peel as a false split only when the input is a finite verb AND no host candidate is a finite verb:

    fv = Counter()
    for policy, inp, outs, key in log:
        if policy != "peel" or not dpd.is_finite_verb(inp):
            continue
        surface = inp[:-2]
        short = (surface[:-1] + B._SHORTEN[surface[-1]]
                 if surface and surface[-1] in B._SHORTEN else surface)
        host_is_verb = any(dpd.is_finite_verb(h) for h in {key, surface, short})
        if not host_is_verb:            # mirror builder's `ambiguous`
            fv[inp] += 1

This collapses the spurious 2118 to (empirically) 0 on the subset, matching the paper's claim. Also update the docstring (lines 11-12) so it no longer says any finite-verb-input peel is a false split; it should say "a peel whose input is a finite verb AND whose host is NOT a finite verb is a false split."

Better, more robust fix: have the builder log its own decision so the audit checks the identical quantity. In build_sandhi_edition.py, extend the peel log record to carry host_is_verb / ambiguous, e.g. log.append(("peel", tok, [host, QUOTATIVE], host, host_is_verb)); then Check 2 just counts peels where is_finite_verb(inp) and not the logged host_is_verb. This avoids the audit re-deriving host_forms (and the inability to recover pieces[-2]) and keeps audit and builder in lockstep.

### [8] sandhi | medium | experiments/exp_sandhi_audit.py
LOC: peel_reconstructs (lines 32-44) and host_is_valid (lines 47-65)
ISSUE: Neither audit check can detect the under-shortening bug (finding 1). peel_reconstructs('gacchatīti','gacchatī') returns True because it accepts host+ti literally (cands includes 'gacchatī'+'ti' = the original token). host_is_valid('gacchatī') returns True because the lengthened form is a DPD headword. So a host the builder demonstrably got wrong passes both checks, and the audit's reported clean-rate / 100% reconstruction overstates correctness for exactly the defective forms.
FIX: Add a third audit check to exp_sandhi_audit.py that surfaces (does not auto-correct) peels where the emitted long-vowel host disagrees with DPD's canonical pre-iti piece. This catches the under-shortening bug the existing two checks structurally cannot:

```python
def over_lengthened_host(tok, host, dpd):
    """Flag a peeled host that ends in a long vowel where DPD's own canonical
    pre-iti piece is the SHORTENED form -- i.e. the builder likely kept a
    spurious lengthening (gacchatī for gacchati). Returns the canonical piece
    if suspect, else None. Deliberately narrow: only fires when DPD itself
    deconstructs tok as [<short-host>, 'iti'] and that short host != emitted
    host, so genuine long-vowel hosts (nom.pl. dhammā, voc. mahārājā, whose
    pieces[-2] is a different/longer stem) are not flagged.
    """
    if not (host and host[-1] in B._SHORTEN):
        return None
    is_hw, pieces = dpd.lookup(tok)
    if not pieces or len(pieces) < 2 or pieces[-1] != "iti":
        return None
    canon = pieces[-2]
    short = host[:-1] + B._SHORTEN[host[-1]]
    if canon == short and canon != host:
        return canon
    return None
```

Then in main(), after the host-validity check, iterate the peel log and report:

```python
overlen = Counter()
for pol, inp, outs, key in log:
    if pol == "peel":
        canon = over_lengthened_host(inp, key, dpd)
        if canon is not None:
            overlen[(inp, key, canon)] += 1
print(f"\npossible over-lengthened hosts (under-shortening, finding 1): "
      f"{sum(overlen.values()):,} occ, {len(overlen)} distinct")
for (tok, host, canon), n in overlen.most_common(25):
    print(f"  {tok!r} -> emitted {host!r}  DPD-canonical {canon!r}  x{n}")
```

Keying on canon == short (DPD's pre-iti piece equals the shortened emitted host) is the correct discriminator: it fires for gacchatīti/passatīti/viharatīti/cāti/atthīti (canon is the short verb/conjunction the builder should have emitted) but NOT for dhammāti or mahārājāti (where pieces[-2] is dhamma/mahārāja, a stem that does not equal the would-be short host dhammā->dhamma case must be checked: note dhammāti's pieces[-2] is 'dhamma' which DOES equal short, so this would still over-flag nom.pl). To avoid the remaining false positives on legitimate nominative-plural/vocative long vowels, additionally require that the SHORT form is tagged a finite verb (or indeclinable) while the long form is not -- i.e. the lengthening changed the part of speech:

```python
    if canon == short and canon != host:
        long_pos, short_pos = dpd.pos_tags(host), dpd.pos_tags(short)
        if (short_pos & B.FINITE_VERB_POS) and not (long_pos & B.FINITE_VERB_POS):
            return canon
```

This narrows the flag to the clearest defects (verb mis-lengthened to a participle/noun, e.g. gacchatī/passatī/viharatī) while sparing genuine long-vowel nominal hosts. The check should be reported as a separate 'review' bucket, not folded into clean-rate, so the audit surfaces finding 1 rather than blessing it. No change to the builder is required by this finding; the fix is purely to the audit so it stops overstating correctness.

### [9] sandhi | low | src/build_sandhi_edition.py
LOC: resolve_tokens Policy B, lines 197-229; docstring lines 207-210
ISSUE: The inline comment states 'The canonical piece is used, not the lengthened surface host (gacchatī), which DPD does not tag as a verb.' In fact the emitted host is `_restore_host(tok[:-2], dpd)` (the surface strip), not the canonical `pieces[-2]`. Verified: for gacchatīti the emitted host is the surface 'gacchatī', while pieces[-2] is 'gacchati'. The comment misdescribes the implementation and obscures the root cause of finding 1.
FIX: Correct the comment rather than changing behavior (the surface host emission appears intentional, and switching to `pieces[-2]` would alter output for cases like `gacchatīti`/`iddhīti` where the lengthened form is itself a valid DPD headword). Replace lines 206-210 with wording that distinguishes emission from the verb test, e.g.:

    # Genuine ambiguity: the token is itself a finite verb AND no host
    # candidate is a finite verb. The EMITTED host is the (restored) surface
    # form `host`; the canonical DPD piece `pieces[-2]` is added only to the
    # verb-ambiguity test set (host_forms). For gacchatīti the emitted host is
    # the lengthened surface `gacchatī` (which DPD tags `prp`, not finite), but
    # the test still sees the canonical `gacchati` (POS `pr`, finite) via
    # `short`/`pieces[-2]`, so it is recognized as a quoted verb and splits.

This keeps behavior unchanged while accurately stating that the surface host is emitted and `pieces[-2]` only participates in the verb-ambiguity test.

### [10] sandhi | low | src/build_sandhi_edition.py
LOC: main, lines 260-262 (stats print loop); docstring lines 15-17
ISSUE: The stats keys 'nti_dpd' and 'nti_mech' are printed but never incremented anywhere in resolve_tokens (grep confirms they appear only in the print loop), so they always report 0. The docstring (Policy A) describes a two-branch behaviour ('if DPD deconstructs the join, use it; else restore the niggahīta mechanically') but the code (lines 165-179) unconditionally does the mechanical host+ṃ restoration with no DPD-join branch. The reported counts are misleading and the documented DPD-consult path does not exist.
FIX: Two small, self-consistent edits (no behavioural change needed, since mechanical-only is the intended behaviour):

1. Remove the dead keys from the print loop in main() (line 260-261):
   change the tuple to ("nti_tokens", "quote_peel", "sandhi_split", "kept").
   (Optionally also report stats["ambiguous_kept"] and stats["fwd_ti"], which ARE incremented but not printed — a separate minor omission.)

2. Align the docstring Policy A (lines 15-17) with the actual mechanical-only behaviour, e.g.:
   "A. Standalone `nti` / `nti…` : rejoin with the preceding token by restoring the niggahīta mechanically (host + ṃ, then quotative). The host is kept whole (no DPD decomposition). Resolves ~100% of `nti` artefacts."

Do NOT instead implement a DPD-join branch unless that behaviour is actually wanted — the comment at lines 162-164 ("The host is kept whole (no compound decomposition)") shows mechanical-only is deliberate, so removing the dead keys + fixing the docstring is the correct, minimal fix rather than adding the documented-but-unwanted DPD path.

### [11] corpus | high | src/reorganise_pts.py
LOC: ALL_REPLACED constant, lines 82-87 (vs tests/test_reorganise_pts.py:69-72 and tests/test_build_full_corpus.py:24-36)
ISSUE: ROOT CAUSE of the three failing tests. ALL_REPLACED was expanded from the 16 canonical .mul files to ALSO include the 34 commentary/sub-commentary files (PS_ATT_SRC, SPK_ATT_SRC, MP_ATT_SRC, DHPA_ATT_SRC, ITA_ATT_SRC, JAA_ATT_SRC, SDP_ATT_SRC, PS_TIK_SRC, SPK_TIK_SRC, MP_TIK_SRC) when commentary reorganisation was added (functions reorganise_ps_commentary..reorganise_mp_subcomm). Correspondingly PTS_NAMES in src/data_loader.py:85-156 grew from the 16 canon stems to 56 entries (adding mna*, spk*, mp*, dhpa*, ita*, jaa*, sdp*, mnt*, spkt*, mpt*). But the tests were never updated: test_reorganise_pts.py::test_all_replaced_is_union asserts ALL_REPLACED == set(AN+JATAKA+PATIS+APADANA) (canon mul only); test_build_full_corpus.py::test_all_replaced_consistent_with_reorganise asserts len(ALL_REPLACED)==16; test_pts_names_has_all_reorganised_ids asserts PTS_NAMES keys == the 16 canon stems. Verified: len(ALL_REPLACED)==50, len(canon mul)==16, len(PTS_NAMES)==56. The raw-pts/ directory on disk DOES contain the reorganised commentary files (mna1.att.txt etc.), confirming the expanded constants are the intended/correct state and the tests are stale.
FIX: Update the three stale tests to match the intended expanded design (constants/code/data are correct). Specifically:

1) tests/test_reorganise_pts.py test_all_replaced_is_union: import the att/tik group lists and assert ALL_REPLACED equals their full union, OR (more robust) assert the canon-mul set is a subset: `assert set(AN_SRC_FILES+JATAKA_SRC_FILES+PATIS_SRC_FILES+APADANA_SRC_FILES) <= ALL_REPLACED`.

2) tests/test_build_full_corpus.py test_all_replaced_consistent_with_reorganise: change `len(ALL_REPLACED)==16` to `==50` (keep the spot-check membership asserts), OR replace the brittle count with `assert len(ALL_REPLACED)==len(set(...all source lists...))` and subset checks.

3) tests/test_build_full_corpus.py test_pts_names_has_all_reorganised_ids: expand expected_stems to the full 56-key set, OR assert the 16 canon stems are a subset of PTS_NAMES.keys().

Add one anti-drift test that pins ALL_REPLACED and PTS_NAMES against each other so they cannot silently diverge again. Since ALL_REPLACED holds VRI source filenames (sNNNN...) and PTS_NAMES holds output stems (an1, mna1, ...), the cleanest pin is to derive the expected output-stem set from the reorganise functions/output mapping and assert it equals set(PTS_NAMES.keys()); minimally, assert len(PTS_NAMES) matches the number of reorganised output volumes the code produces.

Note: only adopt the 'remove commentary' alternative if the paper's corpus is explicitly canon-only — the on-disk reorganised files and the '107-doc PTS corpus' commit indicate commentary IS in scope, so the test-update path is the right fix.

### [12] corpus | high | src/build_full_corpus.py
LOC: main(), lines 98-105 (pts_only_files loop) and lines 147-154 (LABELS.csv writer for pts_only_files)
ISSUE: All 'pts_only_files' (files in raw-pts/ not present under their VRI name in the inventory) are hardcoded as label='canon', pitaka='sutta'. With the expanded raw-pts/ this set now contains 56 files, of which 41 are commentaries (.att: mna*, mp*, dhpa*, ita*, jaa*, sdp*, spk*) or sub-commentaries (.tik: mnt*, mpt*, spkt*). I re-ran the current build_full_corpus.py and confirmed all 41 are written to LABELS.csv with label='canon'/pitaka='sutta'. LABELS.csv is the ground-truth label map consumed by experiments/exp_full_clustering.py (it reads info['label'] and runs a binary canon/non-canon accuracy evaluation), so 41 commentary/sub-commentary documents are scored as 'canon'. This directly corrupts the central canon-vs-commentary-vs-subcommentary clustering evaluation of the register paper. (The LABELS.csv currently on disk predates the att/tik expansion and lacks these files, so the bug is latent until the documented pipeline step is re-run, which makes it a reproducibility landmine.)
FIX: Stop hardcoding canon/sutta for pts_only_files. Derive each file's label from its parent VRI text's correct_label.

1. Build a PTS-stem → (correct_label, pitaka) map from the inventory by linking each PTS-reorganised file back to the VRI text it was split from. reorganise_pts.py already knows this parent relationship (it produced the splits), so expose that mapping (e.g. a PTS_PARENT dict: pts_stem -> vri_file or directly -> (label, pitaka)) and import it here alongside ALL_REPLACED. Then in the loop (lines 99-105):

    for fname in sorted(pts_only_files):
        stem = fname.split(".")[0]
        label, pitaka = PTS_LABELS[stem]   # from parent VRI correct_label/pitaka
        shutil.copy2(SUTTA_PTS_DIR / fname, DST_DIR / fname)
        copied += 1
        label_counts[label] = label_counts.get(label, 0) + 1
        pitaka_counts[pitaka] = pitaka_counts.get(pitaka, 0) + 1

   and in the LABELS.csv writer (lines 148-154) write that same (label, pitaka) instead of the literal "canon","sutta".

2. If a parent-mapping is not readily available, a safe interim mapping that is correct for THIS pts_only set (only .mul/.att/.tik PTS splits of suttas and their commentaries) is: mul->canon, att->commentary, tik->subcommentary, all pitaka=sutta. Do NOT generalize this suffix table to nrf or to non-PTS files — inventory data shows nrf and even some att/mul rows map to other labels.

3. Add a post-build assertion that fails loudly: every *.att.txt written to raw-full has label "commentary", every *.tik.txt has label "subcommentary", every *.mul.txt has label "canon" (for the PTS-derived sutta files), so this can never silently regress.

4. Regenerate data/raw-full/LABELS.csv after the fix so the on-disk ground truth matches before any clustering experiment is re-run.

### [13] corpus | medium | src/build_full_corpus.py
LOC: main(), lines 107-116 (the 'Remove VRI files that were replaced' loop)
ISSUE: For every fname in ALL_REPLACED that exists in DST_DIR the loop does label_counts['canon'] -= 1 and pitaka_counts['sutta'] -= 1. ALL_REPLACED now contains 34 commentary/sub-commentary VRI files (22 .att labelled 'commentary', 12 .tik labelled 'subcommentary') in addition to the 16 .mul 'canon' files. Those 34 VRI files were copied in the main loop with their CORRECT labels (commentary/subcommentary), so this loop subtracts them from 'canon' even though they were never counted as canon. Result: the printed 'By label' canon count is too low by up to 34, and commentary/subcommentary counts are left too high (the replaced VRI att/tik are removed from disk but still counted). Verified: ALL_REPLACED breaks down as 16 canon + 22 commentary + 12 subcommentary.
FIX: Decrement the bucket matching each removed file's ACTUAL label rather than always 'canon'. Build a lookup from the inventory rows already loaded in main() and use it in the removal loop:

    inv_by_file = {r["vri_file"]: r for r in rows}
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

(All 50 ALL_REPLACED files are guaranteed present in the inventory, verified.) The pitaka decrement is unchanged in effect for this dataset (all sutta) but is now correct in general.

Better/more robust: stop mutating counters across add/remove phases entirely and compute the label/pitaka counts once at the end by reading the just-written LABELS.csv (or scanning the final DST_DIR against the inventory), which is already the authoritative, ALL_REPLACED-excluding source of truth. That guarantees the printed summary matches LABELS.csv and the files on disk.

### [14] corpus | low | src/reorganise_pts.py
LOC: read_file (line 90-91) used by concat_files/reorganise_* and split_* (e.g. concat for an1, spk2='...'+text3+...); affects build outputs an1.mul.txt, spk2.att.txt, etc.
ISSUE: VRI source files are UTF-8 with a BOM (verified: s0401m, s0513m, s0201a, s0517m all start with ﻿). read_file uses encoding='utf-8' (not 'utf-8-sig'), so the BOM is preserved. When multiple source files are concatenated the 2nd/3rd file's BOM ends up embedded mid-document: the built an1.mul.txt contains 3 ﻿ characters, spk2.att.txt contains 3, each appearing as a standalone '﻿' token between sections. Token counts are NOT affected for the main pipeline because data_loader.PALI_TOKEN_RE treats ﻿ as a separator (verified: BOM does not survive tokenize_pali), so impact on clustering is nil; but the raw merged corpus files contain spurious control characters and any non-Pali-tokenizer consumer (e.g. parse_commentary.py JSON segments, which retain the BOM verbatim in segment 1) is affected.
FIX: Strip BOM at read time so merged files contain at most one. Minimal change in reorganise_pts.py: change read_file to read with encoding='utf-8-sig':

def read_file(path: Path) -> str:
    return path.read_text(encoding='utf-8-sig')

This removes the leading BOM from every source file before concatenation, so concatenated outputs (an1.mul.txt, spk2.att.txt, etc.) contain zero BOMs. Note utf-8-sig only strips a BOM at the very start of each individually-read file (which is exactly where the VRI BOMs are), so it correctly handles all the multi-file concat sites. For full corpus hygiene, apply the same encoding='utf-8-sig' to the other independent consumers so their standalone outputs are clean too: parse_commentary.py (lines 51 and 151, filepath.read_text(encoding='utf-8')) and vri_inventory.py (line 309). Alternatively, a one-line defensive strip after read (text.lstrip('﻿')) achieves the same for read_file. Low priority for clustering (BOM already dies in the tokenizer) but worthwhile to clean the published raw-pts/ artifacts and the commentary JSON segments.

### [15] corpus | low | src/parse_commentary.py
LOC: parse_vri_commentary (lines 68-70) and parse_canon_files (line 159): 'if len(words) < 2: continue'
ISSUE: Paragraphs with fewer than 2 whitespace tokens are dropped entirely. In VRI these include legitimate single-word section headers/titles (verified examples dropped from s0201a: 'Majjhimanikāye', 'Mūlapaṇṇāsa-aṭṭhakathā', 'Suttanikkhepavaṇṇanā', 'Pathavīvāravaṇṇanā', plus the lone BOM). ~90 such single-token paragraphs are dropped from one commentary file. These are real Pali section titles, so the parsed JSON silently loses structural/header text. Not catastrophic for bag-of-words clustering but it is undocumented, non-reproducible text dropping, and it removes content that could matter for register signals (section-title vocabulary).
FIX: Two concrete options, both addressing the silent/undocumented aspect:

1. Minimal (audit + document): keep the filter but make it auditable and honest.
   - Update the docstring of parse_vri_commentary and parse_canon_files to state explicitly that paragraphs with fewer than 2 whitespace tokens (including lone BOM/section titles) are dropped.
   - Count drops and report them. E.g. add `dropped = 0` and in the filter branch `dropped += 1` before `continue`, then change the print to `print(f"    -> {outpath.name}: {len(data['segments'])} segments ({dropped} short paragraphs dropped)")`.

2. Better (preserve content): drop only truly empty paragraphs and strip the BOM, keeping single-word titles.
   - Replace the `if not para: continue` / `if len(words) < 2: continue` block with a BOM-strip + empty check only:
     ```python
     para = para.strip().lstrip('﻿').strip()
     if not para:
         continue
     ```
   This retains real section titles (which can serve as register/structure signals) while still discarding the BOM and blank segments. If headers must be excluded for the downstream clustering, do it explicitly (e.g. a dedicated header heuristic) and log the count, rather than relying on the incidental <2-token rule.

Either way, fix the misleading inline comment at lines 66-67, which claims short segments are kept "if they have actual Pali content" — the code does no such check.

### [16] reproducibility | high | experiments/exp4_embeddings.py
LOC: train_fasttext_model(), lines 44-52 (FastText(...) call)
ISSUE: The gensim FastText model is trained with workers=4 and no seed= argument. gensim Word2Vec/FastText is nondeterministic across runs unless BOTH seed= is fixed AND workers=1 (multi-threaded training reorders updates regardless of seed, and PYTHONHASHSEED also affects vocab hashing). Every FastText-derived number reported in results/SUMMARY.md (Exp 4: GMM clustering 98.91%, supervised LOO-CV 97.83%, the K-means-by-dimension table 56.52%/83.70%/84.78%, lemmatized 82.61%) is therefore not reproducible: a fresh run can yield materially different accuracies and a different misclassified set. This is the only model family in the pipeline whose reported results cannot be regenerated. Unlike the perplexity 100% result (whose threshold-on-eval-data caveat IS disclosed in SUMMARY.md lines 184/238), this nondeterminism is undisclosed.
FIX: Make FastText training deterministic for the reported runs. In train_fasttext_model() at exp4_embeddings.py:44-52, pass seed=42 AND workers=1 (gensim only honors the seed when single-threaded; workers=4 reintroduces nondeterminism via async update ordering):

    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        workers=1,        # required: multi-threaded training is nondeterministic even with seed
        seed=42,
        sg=1,
    )

Also pin hashing by setting PYTHONHASHSEED=0 in the environment for the run (e.g. via a wrapper or os.environ check at import time), since gensim's default vocab hashing depends on it. Accept the slower single-threaded training. Then either confirm the SUMMARY.md numbers regenerate, or (better, given the dramatic GMM 98.91% vs K-means 56.52% spread) average each configuration over N seeds and report mean ± std. Finally, add a one-line caveat in results/SUMMARY.md Exp 4 noting FastText results were/were-not seed-pinned, mirroring the existing perplexity-threshold caveat, so the (un)reproducibility is disclosed.

### [17] reproducibility | medium | src/classification.py
LOC: run_loo_cv(), line 88 (cross_val_predict on a pre-computed feature_matrix)
ISSUE: run_loo_cv receives feature_matrix already fully fit on the entire corpus. In exp3_supervised.py the TF-IDF vocabulary and IDF weights (tfidf_matrix, max_features=2000) and the word_frequency_matrix top-N selection are computed once over all documents (including the held-out test fold) before CV begins. The StandardScaler is correctly placed inside the make_pipeline so its statistics do not leak, but the vectorizer vocabulary, IDF, and top-N feature selection do leak: the held-out document influences which features exist and their IDF weights. For LOO over 92 docs this is a small effect, but it makes the reported supervised LOO-CV accuracies mildly optimistic and not a clean out-of-sample estimate. The SUMMARY discloses the perplexity caveat but not this one.
FIX: Two valid options; either resolves it.

(A) Make it a clean out-of-sample estimate by moving feature extraction inside the CV. Replace the raw-text-to-matrix precompute with an sklearn Pipeline whose first step is the vectorizer, then pass raw texts (not a precomputed matrix) to cross_val_predict so vocabulary/IDF/feature-selection fit on training folds only. Concretely, build the estimator as:
  make_pipeline(TfidfVectorizer(max_features=2000, lowercase=True), StandardScaler(with_mean=False), clf)
and feed X = [' '.join(d.tokens) for d in documents] to cross_val_predict instead of the dense matrix. Note StandardScaler on sparse TF-IDF needs with_mean=False (or drop the scaler and rely on TF-IDF normalization). The custom top-N word-frequency selector would need wrapping as a sklearn transformer (a small FunctionTransformer/custom estimator implementing fit/transform over the per-fold training docs) to be fit per fold; this is more work, so for that feature type option (B) may be more practical.

(B) If keeping the precompute (cheaper, and the numerical effect is small for LOO over 92 docs), at minimum add an explicit caveat to results/SUMMARY.md (and README.md) next to the supervised LOO-CV tables, e.g.: "Note: TF-IDF vocabulary, max_features selection, and IDF weights (and the top-N word-frequency feature selection) are fit on the full corpus prior to LOO-CV. The held-out document therefore influences which features exist and their weights, so these supervised LOO-CV accuracies are mildly optimistic rather than strictly out-of-sample." This mirrors the existing perplexity-threshold disclosure on line 184.

Recommended: (A) for the TF-IDF/lemma-TF-IDF paths (straightforward Pipeline change) plus the SUMMARY/README note from (B) covering the word-frequency top-N selector until it is wrapped as a per-fold transformer.

### [18] reproducibility | medium | src/features.py
LOC: word_frequency_matrix(), lines 42-56 (all_words = set(); sorted(rel_freqs.keys(), ...))
ISSUE: all_words is built as a Python set, then sorted only by mean relative frequency (key=lambda w: rel_freqs[w], reverse=True). Python's sort is stable, but the *input* iteration order of a set of strings depends on PYTHONHASHSEED, so any words with equal mean relative frequency (extremely common in the long tail near the top_n cutoff, e.g. many hapax-frequency words) break ties in a run-dependent order. This means the exact set of selected top_n feature words at the boundary, and therefore the resulting feature matrix and any downstream clustering/classification using word_frequency_matrix (exp1, exp3 'Word freq 1000'), can differ across runs/machines unless PYTHONHASHSEED is fixed. The tfidf path is unaffected (sklearn sorts the vocabulary deterministically).
FIX: Add a deterministic secondary sort key on the word string so frequency ties break alphabetically instead of by set/dict iteration order. Replace line 55:

    sorted_words = sorted(rel_freqs.keys(), key=lambda w: rel_freqs[w], reverse=True)

with:

    sorted_words = sorted(rel_freqs.keys(), key=lambda w: (-rel_freqs[w], w))

This removes the dependence on PYTHONHASHSEED for both (a) which words land in top_n at the tie boundary and (b) the column order of the returned matrix/feature_names (which the current code also leaves hash-dependent among ties). Note that `-rel_freqs[w]` is fine here since the values are floats from np.mean; it also drops the now-redundant reverse=True. No other determinism control is needed for this function. For full reproducibility of the experiments, optionally also pin PYTHONHASHSEED in the Makefile/run scripts, but the sort-key fix alone resolves the nondeterminism in this code path.

### [19] reproducibility | low | experiments/exp5_transformers.py
LOC: compute_transformer_embeddings(), line 35 (SentenceTransformer(model_name)) and main()
ISSUE: No random seed is set for torch / SentenceTransformer. Sentence-transformer inference (encode) is deterministic for a fixed model version, so the embedding values are stable, but the model weights are downloaded from the network at the pinned model name without a revision/commit pin (paraphrase-multilingual-MiniLM-L12-v2). If the upstream HuggingFace checkpoint is updated, the reported Exp 5 numbers (e.g. SVM RBF 97.83%, GMM 94.57%) would silently change and could not be regenerated. Also the 'different chunk sizes' loop (lines 124-133) tests only [100, 500] and never re-runs chunk_size=200, so the default-config clustering numbers are not part of that sweep table.
FIX: Pin the embedding pipeline so the reported Exp 5 numbers are regenerable:

1) Pin the model revision at load time. In compute_transformer_embeddings(), add a revision arg and pass it through:
   def compute_transformer_embeddings(documents, model_name='paraphrase-multilingual-MiniLM-L12-v2', revision='<commit-sha-from-HF>', chunk_size=200, batch_size=64):
       ...
       model = SentenceTransformer(model_name, revision=revision)
   Look up the current commit SHA for sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 on HuggingFace and hardcode it (record it in SUMMARY.md next to the model name).

2) Record the transformer-stack versions. Add explicit pins to requirements.txt (and/or pyproject.toml), e.g.:
   sentence-transformers==<version-used>
   torch==<version-used>
   transformers==<version-used>
   huggingface-hub==<version-used>
   These are currently only mentioned in a pip-install hint string, never pinned.

3) (Minor completeness) Include the default chunk_size=200 in the sweep so the default-config row appears in the sweep table: change line 125 `for cs in [100, 500]:` to `for cs in [100, 200, 500]:`.

No code-logic bug exists; downstream clustering/classification already use random_state=42, and no torch RNG seed is needed because encode() is deterministic for a fixed model. The fix is purely about pinning/recording the model revision and library versions for reproducibility.
