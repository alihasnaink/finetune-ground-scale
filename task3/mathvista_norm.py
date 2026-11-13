"""MathVista answer normalisation, shared by answer grouping (no gold) and grading.

Follows MathVista's released evaluation rule (normalize_extracted_answer + safe_equal in the official
evaluation code), using the released per-question metadata (task3/data/mathvista_meta.json):
  multi_choice  "(X)" -> X; a valid option letter is kept; any other text is mapped to the most similar
                option by Levenshtein distance (first on ties). Answers are represented as option letters.
                One addition: a single lowercase valid letter is upper-cased.
  integer       str(int(float(x)))
  float         str(round(float(x), precision))
  list          str(x) (whitespace removed)
Two normalised answers are equivalent iff their strings are equal. Missing extractions are NO_ANSWER
(always incorrect). A non-missing free-form answer that the rule cannot parse (e.g. "\\frac{3}{14}",
"10-15 years") stays in its own group keyed by its whitespace-stripped text ("RAW:..."), and is graded
incorrect, as the official rule maps it to None; such answers still vote and cost tokens.
The gold answer is normalised with the same rule.
"""

from __future__ import annotations

import re

VERSION = "mathvista-official-normalize-v1 (+lowercase letter, RAW groups for unparseable)"


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize(extraction: str | None, meta: dict) -> str | None:
    """Canonical answer string, or None for a missing extraction."""
    if extraction is None or (isinstance(extraction, float) and extraction != extraction):  # None / NaN
        return None
    s = str(extraction).strip()
    if meta["question_type"] == "multi_choice":
        letters = [chr(ord("A") + i) for i in range(len(meta["choices"]))]
        m = re.findall(r"\(([a-zA-Z])\)", s)
        if m:
            s = m[0].upper()
        if len(s) == 1 and s.upper() in letters:
            s = s.upper()
        if s in letters:
            return s
        d = [levenshtein(s, c) for c in meta["choices"]]
        return letters[d.index(min(d))]
    t = meta["answer_type"]
    try:
        if t == "integer":
            return str(int(float(s)))
        if t == "float":
            return str(round(float(s), int(meta["precision"])))
    except Exception:
        return "RAW:" + "".join(s.split())
    return "".join(s.split())  # list (and any other free-form type)


def gold_label(gold: str, meta: dict) -> str:
    g = normalize(gold, meta)
    assert g is not None and not g.startswith("RAW:"), (gold, meta)
    return g
