"""Final-answer extraction for MATH, matching the supplied export (README.MD):

find the last literal `\\boxed{`, scan balanced braces (nested LaTeX preserved), strip surrounding
whitespace. An absent, empty or unclosed last box gives None (the NO_ANSWER group); there is no
numeric fallback and no other box command. Verified to reproduce the supplied `extracted_answer`
on all 32,000 MATH rollouts (task3/extract.py --check).
"""

from __future__ import annotations

EXTRACTOR_VERSION = "last-literal-boxed-balanced-v2 (supplied-export rule)"
_TOKEN = "\\boxed{"


def last_boxed(text: str | None) -> str | None:
    if not text:
        return None
    s = text.rfind(_TOKEN)
    if s < 0:
        return None
    start = s + len(_TOKEN)
    depth = 1
    for j in range(start, len(text)):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                ans = text[start:j].strip()
                return ans or None
    return None  # unclosed


if __name__ == "__main__":
    import sys

    import pandas as pd

    f = sys.argv[1] if len(sys.argv) > 1 else "task3_data/qwen25-math/generations.parquet"
    df = pd.read_parquet(f, columns=["generated_text", "extracted_answer"])
    mine = [last_boxed(t) for t in df["generated_text"]]
    ref = [None if pd.isna(x) else x for x in df["extracted_answer"]]
    same = [a == b for a, b in zip(mine, ref)]
    bad = [i for i, ok in enumerate(same) if not ok]
    print(f"{len(df) - len(bad)}/{len(df)} rollouts match the supplied extracted_answer; "
          f"null: mine {sum(m is None for m in mine)}, supplied {sum(r is None for r in ref)}")
    for i in bad[:5]:
        print(i, repr(mine[i]), repr(ref[i]))
