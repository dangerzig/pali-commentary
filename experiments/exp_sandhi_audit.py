#!/usr/bin/env python3
"""
Exhaustive audit of the sandhi-resolved edition.

Rather than sampling, this inspects EVERY transformation the builder makes
across the whole corpus and flags any that fail an independent check:

  - host validity : every host produced by a peel or nti-restoration should be
    a DPD-valid word (headword or deconstructable). A host that DPD does not
    recognise is a suspect split (it may be a non-word the rule invented).
  - finite-verb peels : a peel applied to a token that is itself a finite verb
    is a false split. After the POS guard this should be zero.

It reports a measured clean-rate (share of transformations whose host is a
valid word) and lists the flagged residue by frequency for review.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import build_sandhi_edition as B

RAW_PTS = Path(__file__).parent.parent / "data" / "raw-pts"


_LENGTHEN = {"a": "ā", "i": "ī", "u": "ū"}


def peel_reconstructs(tok: str, host: str) -> bool:
    """Structural check: does host + iti re-sandhi to the original token?

    The quotative iti fuses with the host's final vowel: a short final vowel
    lengthens (hoti+iti -> hotīti), a long/o/e final just absorbs the i
    (gammo+iti -> gammoti). So the original token must be host+ti, or the
    host with its final short vowel lengthened, +ti. This is what the old
    gahetvā->gahetva shortening bug would fail.
    """
    cands = {host + "ti"}
    if host and host[-1] in _LENGTHEN:
        cands.add(host[:-1] + _LENGTHEN[host[-1]] + "ti")
    return tok in cands


def host_is_valid(host: str, dpd: B.DPD) -> bool:
    """A host is acceptable if it is a real word or a recognisable form.

    Accept: a DPD headword; a deconstructable sandhi/compound; a
    quotative-lengthened form whose shortened version is a headword or
    deconstructable (sukhayatī -> sukhayati). The residue after this is the
    genuinely suspect set (forms DPD cannot validate by any of these routes).
    """
    if host == B.QUOTATIVE:
        return True
    is_hw, pieces = dpd.lookup(host)
    if is_hw or pieces is not None:
        return True
    if host and host[-1] in B._SHORTEN:
        short = host[:-1] + B._SHORTEN[host[-1]]
        s_hw, s_pieces = dpd.lookup(short)
        if s_hw or s_pieces is not None:
            return True
    return False


def main():
    dpd = B.DPD()
    log: list = []
    stats = Counter()
    files = sorted(RAW_PTS.glob("*.txt"))
    for f in files:
        B.resolve_tokens(B.tokenize(f.read_text(encoding="utf-8")), dpd, stats, log)

    by_policy = Counter(rec[0] for rec in log)
    print(f"transformations logged: {len(log):,}  {dict(by_policy)}\n")

    # --- Check 0: structural reconstruction of peels (the decisive test) ---
    peels = [(inp, key) for pol, inp, outs, key in log if pol == "peel"]
    recon_ok = sum(1 for tok, host in peels if peel_reconstructs(tok, host))
    bad_recon = [(tok, host) for tok, host in peels
                 if not peel_reconstructs(tok, host)]
    print(f"peel reconstruction (host+iti == token): "
          f"{recon_ok:,}/{len(peels):,} = {recon_ok/len(peels)*100:.4f}%")
    if bad_recon:
        bad = Counter(bad_recon)
        print(f"  reconstruction FAILURES (structural errors): "
              f"{len(bad_recon):,} occ, {len(bad)} distinct")
        for (tok, host), n in bad.most_common(15):
            print(f"    {tok!r} -> host {host!r}  x{n}")
    print()

    # --- Check 1: host validity ---
    flagged = Counter()        # invalid host -> count
    flagged_policy = Counter()
    valid = 0
    for policy, inp, outs, key in log:
        if host_is_valid(key, dpd):
            valid += 1
        else:
            flagged[(policy, key)] += 1
            flagged_policy[policy] += 1
    clean = valid / len(log) * 100 if log else 0
    print(f"host-validity clean-rate: {valid:,}/{len(log):,} = {clean:.3f}%")
    print(f"flagged (invalid host): {sum(flagged.values()):,} occurrences, "
          f"{len(flagged)} distinct  {dict(flagged_policy)}\n")

    # --- Check 2: finite-verb peels (should be ~0) ---
    fv = Counter()
    for policy, inp, outs, key in log:
        if policy == "peel" and dpd.is_finite_verb(inp):
            fv[inp] += 1
    print(f"finite-verb peels (false splits): {sum(fv.values()):,} occ, "
          f"{len(fv)} distinct")
    if fv:
        print("   ", ", ".join(f"{w}({n})" for w, n in fv.most_common(10)))

    # --- Flagged residue, by frequency, for review ---
    print("\ntop flagged invalid-host transformations:")
    for (policy, key), n in flagged.most_common(25):
        print(f"  [{policy:6}] host={key!r:24} x{n:,}")


if __name__ == "__main__":
    main()
