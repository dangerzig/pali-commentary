#!/usr/bin/env python3
"""
Build a sandhi-resolved, fully-inflected edition of the corpus.

Motivation (JPTS R2-3): the Chaṭṭha Saṅgāyana edition's word division produces
sandhi artefacts, most notably the standalone quotative token `nti` (its `citta
nti` for the correct `cittaṃ ti`). This edition resolves sandhi at word
boundaries into fully-declined surface forms, WITHOUT lemmatising, so that
inflectional distinctions the analysis relies on (e.g. bhikkhu vs. bhikkhave)
are preserved while the artefacts are removed.

Resolution is driven by the Digital Pāli Dictionary (DPD) deconstructor, with
guards against DPD's known over-splitting of common finite verbs. Policy:

  A. Standalone `nti` / `nti…`  : rejoin with the preceding token; if DPD
     deconstructs the join, use it; else restore the niggahīta mechanically
     (host + ṃ, then quotative). Resolves ~100% of `nti` artefacts.
  B. Quotative peel             : if a token's DPD deconstruction ends in
     `+ iti`, split off the quotative (and any further DPD pieces). Precise:
     it never touches a genuine `-ti` verb such as gacchati, only forms DPD
     analyses as ending in the clitic iti.
  C. Conservative sandhi        : split a token ONLY if it is NOT itself a DPD
     headword but IS deconstructable into pieces that are ALL DPD headwords.
     This resolves clear inter-word sandhi (imamatthaṃ -> imaṃ atthaṃ) while
     keeping real words whole (jānāti, karissāmī are headwords -> untouched).
  D. Otherwise                  : keep the surface form unchanged.

All quotative forms (CST `ti`, DPD `iti`) are normalised to a single token
`ti` so the quotative is counted as one marker.

Output: data/raw-pts-sandhi/<same filename> with whitespace-separated resolved
tokens, a drop-in replacement for data/raw-pts consumable by the existing
experiments (tokenize_pali lowercases and splits on non-Pāli characters).
"""

import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW_PTS = REPO / "data" / "raw-pts"
OUT_DIR = REPO / "data" / "raw-pts-sandhi"
DPD_DB = Path.home() / "pali-canon" / "data" / "dpd" / "dpd.db"

PALI_TOKEN_RE = re.compile(r'[^a-zāīūṭḍṇṅñṃḷ]+')
QUOTATIVE = "ti"

# Policy C (general inter-word sandhi / compound splitting) is disabled by
# default. Testing showed DPD's deconstructor, applied beyond the quotative,
# decomposes single compound words (khandhalokādisaṅkhātaṃ -> khandhaloka ādi
# saṅkhātaṃ) and occasionally mis-splits (sattaratanādivicittaṃ -> sattaratanā
# divi cittaṃ). That is compound decomposition, not sandhi resolution, and
# introduces errors. The edition resolves the quotative artefacts (the
# reviewer's concern) while keeping every word, compound, and inflection whole.
INCLUDE_GENERAL_SANDHI = False


def tokenize(text: str) -> list[str]:
    """Match data_loader.tokenize_pali: normalise niggahīta, lowercase, split."""
    text = text.replace('ṁ', 'ṃ').lower()
    return [t for t in PALI_TOKEN_RE.split(text) if t]


# DPD part-of-speech tags for FINITE verbs (forms that should never be split,
# since a finite verb is a complete word, not host + clitic). Non-finite forms
# (abs absolutive, pp/prp participles, inf, ger) are excluded: they legitimately
# take the quotative (gahetvā + iti) and must remain peelable.
FINITE_VERB_POS = {"pr", "aor", "fut", "imp", "opt", "cond", "perf", "imperf"}


class DPD:
    """Thin cached interface to the DPD lookup and headword tables."""

    def __init__(self, db_path=DPD_DB):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.cur = self.conn.cursor()
        self._cache: dict[str, tuple[bool, list[str] | None]] = {}
        self._pos_cache: dict[str, set[str]] = {}

    def lookup(self, word: str) -> tuple[bool, list[str] | None]:
        """Return (is_headword, first_deconstruction_pieces or None)."""
        if word in self._cache:
            return self._cache[word]
        r = self.cur.execute(
            "SELECT headwords, deconstructor FROM lookup WHERE lookup_key = ?",
            (word,)).fetchone()
        if not r:
            res = (False, None)
        else:
            is_hw = bool(r["headwords"])
            d = r["deconstructor"]
            pieces = None
            if d and d != "null":
                try:
                    decs = json.loads(d)
                    if decs:
                        pieces = [p for p in decs[0].replace(" ", "").split("+") if p]
                except Exception:
                    pieces = None
            res = (is_hw, pieces)
        self._cache[word] = res
        return res

    def is_headword(self, word: str) -> bool:
        return self.lookup(word)[0]

    def pos_tags(self, word: str) -> set[str]:
        """Set of DPD part-of-speech tags for a word's headword analyses."""
        if word in self._pos_cache:
            return self._pos_cache[word]
        tags: set[str] = set()
        r = self.cur.execute(
            "SELECT headwords FROM lookup WHERE lookup_key = ?", (word,)).fetchone()
        if r and r["headwords"]:
            try:
                ids = json.loads(r["headwords"])
                qs = ",".join("?" * len(ids))
                for h in self.cur.execute(
                        f"SELECT pos FROM dpd_headwords WHERE id IN ({qs})", ids):
                    if h["pos"]:
                        tags.add(h["pos"])
            except Exception:
                pass
        self._pos_cache[word] = tags
        return tags

    def is_finite_verb(self, word: str) -> bool:
        return bool(self.pos_tags(word) & FINITE_VERB_POS)


def norm_quote(tok: str) -> str:
    """Normalise the quotative to a single token `ti`."""
    return QUOTATIVE if tok in ("iti", "ti") else tok


_SHORTEN = {"ā": "a", "ī": "i", "ū": "u"}
_LENGTHEN = {"a": "ā", "i": "ī", "u": "ū"}


def _reconstructs(tok: str, host: str) -> bool:
    """True if host + iti re-sandhis to the surface token.

    host + iti is either host+ti (long/o/e final absorbs the i) or, for a short
    final vowel, the lengthened-vowel + ti (hoti+iti -> hotīti).
    """
    if tok == host + "ti":
        return True
    if host and host[-1] in _LENGTHEN and tok == host[:-1] + _LENGTHEN[host[-1]] + "ti":
        return True
    return False


def _restore_host(host: str, dpd: "DPD") -> str:
    """Recover the underlying host before the quotative clitic.

    The quotative lengthens a final SHORT vowel (iddhi + iti -> iddhīti), but a
    host may also end in a GENUINE long vowel (the absolutive gahetvā, the noun
    vijjā), which must not be shortened. Decide by DPD validity: shorten only
    when the long-vowel form is not a word but the shortened form is.
    """
    if host and host[-1] in _SHORTEN:
        shortened = host[:-1] + _SHORTEN[host[-1]]
        if not dpd.is_headword(host) and dpd.is_headword(shortened):
            return shortened
    return host


def resolve_tokens(tokens: list[str], dpd: DPD, stats: Counter,
                   log: list | None = None) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        # ---- Policy A: standalone nti / nti… (CST quotative artefact) ----
        # The niggahīta belongs to the PRECEDING token; restore it and emit the
        # quotative. The host is kept whole (no compound decomposition).
        if tok == "nti" or tok.startswith("nti"):
            stats["nti_tokens"] += 1
            if out:
                host = out.pop()
                restored = host if host.endswith("ṃ") else host + "ṃ"
                out.append(restored)
                out.append(QUOTATIVE)
                rem = tok[3:]  # material after 'nti' (e.g. 'ādīsu', 'pi')
                if log is not None:
                    log.append(("nti", f"{host} {tok}", [restored, QUOTATIVE], restored))
                if rem:
                    out.extend(resolve_tokens([rem], dpd, stats, log))
            else:
                out.append(QUOTATIVE)  # nti with no preceding host (rare)
            continue

        is_hw, pieces = dpd.lookup(tok)

        # ---- Policy A2: forward quotative glue, the "...ti ādi…" formula ----
        # CST glues the quotative onto a following ādi- ("...and so on"):
        # tiādīni for "...ti ādīni". Restricted to a remainder beginning ādi-,
        # so that the numeral prefix ti- ("three": tiloka, tilakkhaṇa, tiratana)
        # is never mistaken for the quotative. Guard (not is_hw) is also kept.
        if (not is_hw and tok.startswith("tiādi") and len(tok) > 2
                and dpd.is_headword(tok[2:])):
            out.append(QUOTATIVE)
            if log is not None:
                log.append(("fwd_ti", tok, [QUOTATIVE, tok[2:]], tok[2:]))
            out.extend(resolve_tokens([tok[2:]], dpd, stats, log))
            stats["fwd_ti"] += 1
            continue

        # ---- Policy B: quotative peel (DPD deconstruction ends in iti) ----
        # Peel ONLY the quotative clitic, keeping the host whole via a surface
        # strip (never DPD's further decomposition of the host, which would
        # decompose compounds: tāṇābhāvoti, āhosīti). Guard: host must not end
        # in ṃ (a vowel-final "...ti" cannot derive from a ṃ-host; this rejects
        # DPD's spurious "jānaṃ + iti" split of the verb jānāti).
        if (pieces and pieces[-1] == "iti" and tok.endswith("ti")
                and (len(pieces) < 2 or not pieces[-2].endswith("ṃ"))):
            # Prefer DPD's canonical host piece (gacchati, gahetvā, vijjā) when
            # it reconstructs to the token, rather than the lengthened surface
            # host (gacchatī); fall back to a conservative surface strip.
            canon = pieces[-2] if len(pieces) >= 2 else ""
            if canon and _reconstructs(tok, canon):
                host = canon
            else:
                host = _restore_host(tok[:-2], dpd)
            # Genuine ambiguity: the token is itself a finite verb AND no form of
            # its host is a finite verb (bhaṇeti = verb, or bhaṇe + iti). When a
            # host form IS a verb (hoti -> hotīti, gacchati -> gacchatīti) it is
            # just a quoted verb and splits cleanly.
            surface = tok[:-2]
            short = surface[:-1] + _SHORTEN[surface[-1]] if (
                surface and surface[-1] in _SHORTEN) else surface
            host_forms = {host, surface, short}
            if len(pieces) >= 2:
                host_forms.add(pieces[-2])
            host_is_verb = any(dpd.is_finite_verb(h) for h in host_forms)
            ambiguous = dpd.is_finite_verb(tok) and not host_is_verb
            if host and not ambiguous:
                out.append(host)
                out.append(QUOTATIVE)
                if log is not None:
                    log.append(("peel", tok, [host, QUOTATIVE], host))
                stats["quote_peel"] += 1
                continue
            if ambiguous:
                stats["ambiguous_kept"] += 1
                if log is not None:
                    log.append(("ambig", tok, [tok], tok))

        # ---- Policy C: conservative inter-word sandhi (disabled) ----
        if (INCLUDE_GENERAL_SANDHI and pieces and not is_hw
                and all(dpd.is_headword(p) for p in pieces)):
            out.extend(norm_quote(p) for p in pieces)
            stats["sandhi_split"] += 1
            continue

        # ---- Policy D: keep surface form ----
        out.append(tok)
        stats["kept"] += 1
    return out


def main():
    dpd = DPD()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_PTS.glob("*.txt"))
    print(f"Building sandhi edition: {len(files)} files -> {OUT_DIR}")
    stats = Counter()
    tot_in = tot_out = 0
    for f in files:
        toks = tokenize(f.read_text(encoding="utf-8"))
        resolved = resolve_tokens(toks, dpd, stats)
        tot_in += len(toks)
        tot_out += len(resolved)
        (OUT_DIR / f.name).write_text(" ".join(resolved) + "\n", encoding="utf-8")
    print(f"\ninput tokens : {tot_in:,}")
    print(f"output tokens: {tot_out:,}  ({tot_out - tot_in:+,})")
    print("\noperation counts:")
    for k in ("nti_tokens", "fwd_ti", "quote_peel", "ambiguous_kept",
              "sandhi_split", "kept"):
        print(f"  {k:14}: {stats[k]:>10,}")
    # sanity: standalone nti should be gone from the output
    nti_after = 0
    for f in OUT_DIR.glob("*.txt"):
        nti_after += Counter(f.read_text(encoding="utf-8").split())["nti"]
    print(f"\nstandalone 'nti' tokens remaining in output: {nti_after}")


if __name__ == "__main__":
    main()
