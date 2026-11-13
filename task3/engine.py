"""Candidate-set selection rules shared by the fixed-budget (Steps 1-3) and adaptive (Step 5) simulations.

A candidate set is a list of rollout ids in reveal order. Answer groups are formed greedily in that
order from the gold-free equivalence matrix: a candidate joins the group of the first earlier
representative it is equivalent to (NO_ANSWER is one group). Everything here uses only the revealed
candidates: scores, VisConf calibration and ranks are recomputed for every candidate set, and gold
answers are never consulted.

Ties: highest-score selection and vote-based rules resolve ties by the earliest supporting rollout in
the current order; RWC ranks use average ranks for tied scores. A missing score (zero-length rollout)
ranks below every finite score. At N = 1 every rule returns the only candidate.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def groups(q: dict, cand: list[int]) -> tuple[list[int], list[int]]:
    """Group index per candidate position, and the representative position of each group."""
    eq, ans = q["eq"], q["ans"]
    reps: list[int] = []
    gid = []
    for p, r in enumerate(cand):
        g = next((k for k, rp in enumerate(reps) if eq[cand[rp], r]), None)
        if g is None:
            reps.append(p)
            g = len(reps) - 1
        gid.append(g)
    return gid, reps


def visconf_scores(q: dict, cand: list[int], quantiles=(0.05, 0.95)) -> dict[str, np.ndarray]:
    """C_i, Z_i = V_i C_i and Z^token_i = mean_t c_t v_t, calibrated on the pooled token KL of `cand`."""
    kls = [q["kl_tok"][r] for r in cand]
    pooled = np.concatenate(kls) if kls else np.zeros(0, np.float32)
    if pooled.size == 0:
        nan = np.full(len(cand), np.nan)
        return {"C": nan, "Z": nan, "Ztok": nan}
    lo, hi = np.percentile(pooled.astype(np.float64), [100 * quantiles[0], 100 * quantiles[1]])
    den = max(hi - lo, 1e-8)
    C, Zt = np.full(len(cand), np.nan), np.full(len(cand), np.nan)
    for p, r in enumerate(cand):
        k = kls[p]
        if k.size:
            c = np.clip((k.astype(np.float64) - lo) / den, 0.0, 1.0)
            C[p] = c.mean()
            v = q["v_tok"][r]
            ok = np.isfinite(v)
            Zt[p] = (c[ok] * v[ok]).sum() / k.size if ok.any() else np.nan
    V = q["V"][cand]
    return {"C": C, "Z": V * C, "Ztok": Zt}


def _filled(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    return np.where(np.isfinite(s), s, -np.inf)


def select_first(q, cand, gid, reps):
    return 0


def select_highest(score: np.ndarray) -> int:
    s = _filled(score)
    return int(np.flatnonzero(s == s.max())[0])  # earliest on ties


def vote_winner(gid: list[int], reps: list[int], weights: np.ndarray) -> tuple[int, float, float]:
    """Winning group (ties to the earliest representative), its share of the total weight, and the margin
    (largest minus second-largest group weight, over the total; runner-up 0 when there is one group)."""
    W = np.zeros(len(reps))
    np.add.at(W, np.asarray(gid), weights)
    order = sorted(range(len(reps)), key=lambda g: (-W[g], reps[g]))
    tot = W.sum()
    win = order[0]
    second = W[order[1]] if len(order) > 1 else 0.0
    return win, (W[win] / tot if tot > 0 else 0.0), ((W[win] - second) / tot if tot > 0 else 0.0)


def rwc_weights(score: np.ndarray) -> np.ndarray:
    N = len(score)
    if N == 1:
        return np.ones(1)
    r = rankdata(-_filled(score), method="average")  # rank 1 = best
    return (N - r) / (N - 1)


def answer_of(q, cand, gid, reps, pos=None, group=None) -> tuple[int, str | None, bool]:
    """(supporting rollout id, normalised answer or None for NO_ANSWER, correct) for a selected candidate
    position or group (a group's answer is that of its representative)."""
    if group is not None:
        pos = reps[group]
    r = cand[pos]
    return r, q["ans"][r], bool(q["correct"][r])


def scores_for(q: dict, cand: list[int], ds: str, quantiles) -> dict[str, np.ndarray]:
    c = np.asarray(cand)
    s = {"L": q["L"][c], "K": q["K"][c], "neg_len": -q["n_tok"][c].astype(np.float64)}
    if ds == "mathvista":
        s["V"] = q["V"][c]
        s |= visconf_scores(q, cand, quantiles)
    return s


# method -> (score name or None, aggregation)
METHODS = {
    "math": {"FirstAns": (None, "first"), "SC": (None, "vote"), "MaxLogP": ("L", "highest"),
             "MaxSelfCert": ("K", "highest"), "SelfCert-RWC": ("K", "rwc"), "Shortest": ("neg_len", "highest")},
}
METHODS["mathvista"] = METHODS["math"] | {
    "Vision-Max": ("V", "highest"), "Vision-RWC": ("V", "rwc"), "VisConf-Max": ("Z", "highest"),
    "VisConf-RWC": ("Z", "rwc"), "TokFusion-Max": ("Ztok", "highest"), "TokFusion-RWC": ("Ztok", "rwc")}


def evaluate_set(q: dict, cand: list[int], ds: str, quantiles=(0.05, 0.95)) -> tuple[dict, dict]:
    """All methods on one candidate set. Returns ({method: (rollout, answer, correct)}, set statistics
    including the vote state needed by the stopping rules)."""
    gid, reps = groups(q, cand)
    N = len(cand)
    sc = scores_for(q, cand, ds, quantiles) if N > 1 else {}
    out = {}
    votes = np.ones(N)
    sc_win, sc_share, sc_margin = vote_winner(gid, reps, votes)
    for m, (sname, agg) in METHODS[ds].items():
        if N == 1 or agg == "first":
            out[m] = answer_of(q, cand, gid, reps, pos=0)
        elif agg == "vote":
            out[m] = answer_of(q, cand, gid, reps, group=sc_win)
        elif agg == "highest":
            out[m] = answer_of(q, cand, gid, reps, pos=select_highest(sc[sname]))
        else:
            w, _, _ = vote_winner(gid, reps, rwc_weights(sc[sname]))
            out[m] = answer_of(q, cand, gid, reps, group=w)
    W = np.zeros(len(reps))
    np.add.at(W, np.asarray(gid), votes)
    lead = sorted(W, reverse=True)
    stats = {"N": N, "n_groups": len(reps),
             "n_groups_answered": sum(q["ans"][cand[rp]] is not None for rp in reps),
             "oracle": bool(q["correct"][cand].any()), "cost": int(q["n_tok"][cand].sum()),
             "sc_group": sc_win, "sc_is_na": q["ans"][cand[reps[sc_win]]] is None,
             "sc_votes": int(W[sc_win]), "sc_unique": len(lead) == 1 or lead[0] > lead[1],
             "gid": gid, "reps": reps, "scores": sc}
    return out, stats


def rwc_state(score: np.ndarray, gid, reps) -> tuple[int, float]:
    """RWC winning group and vote margin for a candidate set (used by quality-aware stopping)."""
    if len(gid) == 1:
        return 0, 1.0
    w, _, margin = vote_winner(gid, reps, rwc_weights(score))
    return w, margin

