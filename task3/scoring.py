"""MATH answer equivalence with math-verify, shared by answer grouping (no gold) and grading.

The same rule is used for both: two answers are equivalent if math-verify verifies them in either
direction, or their whitespace-stripped strings are equal. Missing extractions form one NO_ANSWER
group that is always graded incorrect.
"""

from __future__ import annotations

from functools import lru_cache

NO_ANSWER = "NO_ANSWER"


def scorer_version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return f"math-verify {version('math-verify')}"
    except PackageNotFoundError:
        return "math-verify (version unknown)"


@lru_cache(maxsize=200_000)
def _parse(ans: str):
    from math_verify import parse
    try:
        return parse("\\boxed{" + ans + "}")
    except Exception:
        return []


def _key(a: str) -> str:
    return "".join(a.split())


@lru_cache(maxsize=500_000)
def equivalent(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    if _key(a) == _key(b):
        return True
    from math_verify import verify
    pa, pb = _parse(a), _parse(b)
    if not pa or not pb:
        return False
    try:
        return bool(verify(pa, pb) or verify(pb, pa))
    except Exception:
        return False


def is_correct(pred: str | None, gold: str | None) -> bool:
    return pred is not None and gold is not None and equivalent(gold, pred)


def group_answers(answers: list[str | None]) -> list[str]:
    """Answer-group label per candidate (in order): the first member's answer string, or NO_ANSWER.
    Uses only the candidates, never the gold answer."""
    reps: list[str] = []
    out = []
    for a in answers:
        if a is None:
            out.append(NO_ANSWER)
            continue
        g = next((r for r in reps if equivalent(r, a)), None)
        if g is None:
            reps.append(a)
            g = a
        out.append(g)
    return out


def self_consistency(answers: list[str | None]) -> tuple[str | None, dict]:
    """Majority answer group (one vote per candidate, NO_ANSWER included); ties go to the group whose
    earliest supporting candidate comes first. Returns (answer or None for NO_ANSWER, vote counts)."""
    groups = group_answers(answers)
    votes: dict[str, int] = {}
    first: dict[str, int] = {}
    for i, g in enumerate(groups):
        votes[g] = votes.get(g, 0) + 1
        first.setdefault(g, i)
    win = max(votes, key=lambda g: (votes[g], -first[g]))
    return (None if win == NO_ANSWER else win), votes
