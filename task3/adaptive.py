"""Task 3, Step 5: adaptive sampling (offline simulation of stopping between completed rollouts; CPU).

    python -m task3.adaptive --config task3/configs/offline.yaml [--datasets math mathvista]

Rollouts are revealed in the saved random orders (task3.offline). Stopping is checked after n in
{4, 8, 16} candidates; otherwise the policy stops at 32. Both controllers return the current
self-consistency answer:
  agreement      stop if the leading answer is unique, not NO_ANSWER, and supported by >= tau * n candidates
  quality-aware  the agreement condition, plus the self-consistency and RWC winners agree and the RWC vote
                 margin is >= gamma (self-certainty scores for MATH, VisConf for MathVista; scores, VisConf
                 calibration and ranks recomputed from the revealed candidates only). The margin is
                 (largest - second-largest answer weight) / total weight, runner-up 0 with one group.
Settings are selected on the development split only: for each controller family, among tau in
{0.5, 0.625, 0.75, 0.875, 1.0} (x gamma in {0, 0.1, 0.25, 0.5}) and the always-32 fallback, the
lowest mean token cost whose accuracy is within 1 pp of fixed-32 self-consistency (averaged over the
ten orders); ties by higher accuracy, then smaller tau, then smaller gamma. The selections are frozen
(selected.json) before the held-out split is evaluated.

Writes <results>/<dataset>/adaptive_dev_grid.csv, adaptive_selected.json, adaptive_eval.csv,
adaptive_per_question.csv, adaptive_stop_dist.csv, adaptive_early_errors.csv, adaptive_boot.csv,
adaptive_retro.csv (stopping vs agreement / pool correctness, retrospective) and adaptive.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import base_argparser, load_config, log, write_json
from .offline import md_table, paired_boot

EPS = 1e-9


def wide_states(st: pd.DataFrame, ns: list[int]) -> dict:
    st = st.sort_values(["qid", "order", "n"])
    keys = st[st.n == ns[0]][["qid", "split", "order"]].reset_index(drop=True)
    arr = {}
    for c in ("lead_frac", "unique_leader", "leader_is_na", "rwc_agrees", "rwc_margin", "sc_correct", "cost"):
        arr[c] = np.stack([st[st.n == n][c].to_numpy() for n in ns], axis=1)
    return {"keys": keys, "ns": np.array(ns), **arr}


def simulate(W: dict, family: str, tau: float | None, gamma: float | None) -> pd.DataFrame:
    """Per (qid, order): stopping n, correctness and cost. tau None = always-32 fallback."""
    ns = W["ns"]
    k = len(ns)
    if tau is None:
        stop = np.full(len(W["keys"]), k - 1)
    else:
        cond = W["unique_leader"] & ~W["leader_is_na"] & (W["lead_frac"] >= tau - EPS)
        if family == "quality":
            cond &= W["rwc_agrees"] & (W["rwc_margin"] >= gamma - EPS)
        cond[:, -1] = True  # always stop at 32
        stop = cond.argmax(1)
    rows = np.arange(len(stop))
    df = W["keys"].copy()
    df["stop_n"] = ns[stop]
    df["correct"] = W["sc_correct"][rows, stop]
    df["cost"] = W["cost"][rows, stop]
    df["sc32_correct"] = W["sc_correct"][:, -1]
    return df


def q_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["qid", "split"])[["correct", "cost", "stop_n", "sc32_correct"]].mean().reset_index()


def policies(cfg) -> list[dict]:
    a = cfg.adaptive
    out = [{"family": f, "tau": None, "gamma": None} for f in ("agreement", "quality")]
    out += [{"family": "agreement", "tau": float(t), "gamma": None} for t in a.tau]
    out += [{"family": "quality", "tau": float(t), "gamma": float(g)} for t in a.tau for g in a.gamma]
    return out


def name(p: dict) -> str:
    if p["tau"] is None:
        return f"{p['family']}: always-32"
    return f"{p['family']}: tau={p['tau']}" + (f", gamma={p['gamma']}" if p["family"] == "quality" else "")


def run(cfg, ds: str):
    out = Path(cfg.paths.results) / ds
    st = pd.read_parquet(out / "stop_states.parquet")
    ns = sorted(int(n) for n in cfg.adaptive.checks) + [int(cfg.adaptive.max_n)]
    W = wide_states(st, ns)
    dev = W["keys"].split.eq("dev").to_numpy()
    tol = float(cfg.adaptive.tolerance_pp)

    # ---- development selection (dev split only) ----
    grid = []
    for p in policies(cfg):
        qm = q_mean(simulate(W, p["family"], p["tau"], p["gamma"])[dev])
        grid.append({**p, "policy": name(p), "accuracy": 100 * qm.correct.mean(), "mean_cost": qm.cost.mean(),
                     "mean_rollouts": qm.stop_n.mean(), "sc32_accuracy": 100 * qm.sc32_correct.mean()})
    grid = pd.DataFrame(grid)
    sc32 = grid.sc32_accuracy.iloc[0]
    grid["feasible"] = grid.accuracy >= sc32 - tol - 1e-9
    grid.to_csv(out / "adaptive_dev_grid.csv", index=False)
    selected = {}
    for fam in ("agreement", "quality"):
        g = grid[(grid.family == fam) & grid.feasible].copy()
        g["_t"] = g.tau.fillna(np.inf)
        g["_g"] = g.gamma.fillna(np.inf) if fam == "quality" else 0.0
        best = g.sort_values(["mean_cost", "accuracy", "_t", "_g"], ascending=[True, False, True, True]).iloc[0]
        selected[fam] = {"tau": None if pd.isna(best.tau) else float(best.tau),
                         "gamma": None if fam == "agreement" or pd.isna(best.gamma) else float(best.gamma),
                         "dev_accuracy": float(best.accuracy), "dev_mean_cost": float(best.mean_cost),
                         "dev_mean_rollouts": float(best.mean_rollouts), "dev_sc32_accuracy": float(sc32)}
    write_json(out / "adaptive_selected.json", {"rule": f"lowest dev mean cost with dev accuracy >= fixed-32 SC "
                                                f"- {tol} pp; ties: higher accuracy, smaller tau, smaller gamma",
                                                "quality_score": cfg.adaptive.quality_score[ds], **selected})
    log(f"{ds}: frozen selections {json.dumps(selected)}")

    # ---- held-out evaluation with the frozen settings ----
    ev = ~dev
    summ = pd.read_csv(out / "summary.csv")
    fixed = summ[(summ.split == "eval") & (summ.method == "SC")][["N", "accuracy", "mean_cost"]]
    rows, pqs, dists, errs, boots = [], [], [], [], []
    sims = {}
    for fam, s in selected.items():
        sim = simulate(W, fam, s["tau"], s["gamma"])
        sims[fam] = sim
        e = sim[ev]
        qm = q_mean(e)
        pqs.append(qm.assign(policy=fam))
        rows.append({"policy": f"{fam} ({name({'family': fam, **s}).split(': ')[1]})",
                     "accuracy": 100 * qm.correct.mean(), "mean_cost": qm.cost.mean(),
                     "mean_rollouts": qm.stop_n.mean(),
                     **{f"stop@{n} (%)": 100 * (e.stop_n == n).mean() for n in ns}})
        dists += [{"policy": fam, "stop_n": n, "pct": 100 * (e.stop_n == n).mean()} for n in ns]
        early = e[e.stop_n < ns[-1]]
        wrong = early[~early.correct.astype(bool)]
        errs.append({"policy": fam, "early_stops_pct": 100 * len(early) / len(e),
                     "early_stop_accuracy": 100 * early.correct.mean() if len(early) else np.nan,
                     "incorrect_early_stops_pct_of_runs": 100 * len(wrong) / len(e),
                     "incorrect_early_stops_recovered_by_sc32_pct": 100 * wrong.sc32_correct.mean() if len(wrong) else np.nan,
                     "n_incorrect_early_stops": len(wrong),
                     "questions_with_incorrect_early_stop": wrong.qid.nunique(),
                     "correct_early_stops_lost_by_sc32_pct": 100 * (1 - early[early.correct.astype(bool)].sc32_correct.mean())
                     if early.correct.any() else np.nan})
        # fixed-N self-consistency at similar observed cost (linear interpolation of the fixed-N curve)
        acc_at_cost = float(np.interp(qm.cost.mean(), fixed.mean_cost, fixed.accuracy))
        rows[-1]["fixed_SC_acc_at_same_cost"] = acc_at_cost
        r = paired_boot(qm.correct, qm.sc32_correct, int(cfg.offline.n_boot), int(cfg.seed))
        boots.append({"policy": fam, "a": f"adaptive-{fam}", "b": "SC@32", **r})
    sel = pd.read_parquet(out / "selections.parquet", filters=[("method", "==", "SC"), ("split", "==", "eval")])
    pqsc = sel.groupby(["qid", "N"]).correct.mean().unstack()
    for fam in selected:
        qm = pqs[[p.policy.iloc[0] for p in pqs].index(fam)].set_index("qid").loc[pqsc.index]
        for N in (8, 16):
            r = paired_boot(qm.correct, pqsc[N], int(cfg.offline.n_boot), int(cfg.seed))
            boots.append({"policy": fam, "a": f"adaptive-{fam}", "b": f"SC@{N}", **r})
    if len(sims) == 2:
        a, q = (q_mean(sims[f][ev]).set_index("qid") for f in ("agreement", "quality"))
        boots.append({"policy": "both", "a": "adaptive-quality", "b": "adaptive-agreement",
                      **paired_boot(q.correct, a.loc[q.index].correct, int(cfg.offline.n_boot), int(cfg.seed))})
    evt = pd.DataFrame(rows)
    evt.to_csv(out / "adaptive_eval.csv", index=False)
    pd.concat(pqs).to_csv(out / "adaptive_per_question.csv", index=False)
    pd.DataFrame(dists).to_csv(out / "adaptive_stop_dist.csv", index=False)
    errs = pd.DataFrame(errs)
    errs.to_csv(out / "adaptive_early_errors.csv", index=False)
    boots = pd.DataFrame(boots)
    boots.to_csv(out / "adaptive_boot.csv", index=False)

    # ---- retrospective: stopping vs agreement and the fraction of correct candidates in the full pool ----
    pool = pd.read_parquet(out / "sets.parquet", filters=[("N", "==", 32), ("order", "==", 0)])[["qid", "n_correct"]]
    retro = []
    for fam, sim in sims.items():
        e = sim[ev].merge(pool, on="qid")
        e["pool_correct_frac"] = e.n_correct / 32
        e["pool_bin"] = pd.cut(e.pool_correct_frac, [-0.01, 0.0, 0.25, 0.5, 0.75, 0.999, 1.0],
                               labels=["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)", "1"])
        g = e.groupby("pool_bin", observed=True).agg(runs=("qid", "size"), mean_stop_n=("stop_n", "mean"),
                                                     early_stop_pct=("stop_n", lambda x: 100 * (x < ns[-1]).mean()),
                                                     accuracy=("correct", lambda x: 100 * x.mean()),
                                                     sc32_accuracy=("sc32_correct", lambda x: 100 * x.mean()),
                                                     mean_cost=("cost", "mean")).reset_index()
        g.insert(0, "policy", fam)
        retro.append(g)
    retro = pd.concat(retro)
    retro.to_csv(out / "adaptive_retro.csv", index=False)
    # agreement at the first check vs stopping
    st4 = st[(st.n == ns[0]) & (st.split == "eval")][["qid", "order", "lead_frac"]]
    agr = []
    for fam, sim in sims.items():
        e = sim[ev].merge(st4, on=["qid", "order"])
        g = e.groupby("lead_frac").agg(runs=("qid", "size"), mean_stop_n=("stop_n", "mean"),
                                       accuracy=("correct", lambda x: 100 * x.mean())).reset_index()
        g.insert(0, "policy", fam)
        agr.append(g)
    agr = pd.concat(agr).rename(columns={"lead_frac": f"leader_share_at_n={ns[0]}"})
    agr.to_csv(out / "adaptive_vs_agreement.csv", index=False)

    lines = [f"## {ds}: adaptive sampling (quality score {cfg.adaptive.quality_score[ds]})\n",
             f"### Development selection (200 questions; fixed-32 SC {sc32:.2f}%, threshold {sc32 - tol:.2f}%)\n",
             md_table(grid[grid.feasible].sort_values("mean_cost").head(8)[
                 ["policy", "accuracy", "mean_cost", "mean_rollouts"]]),
             "Frozen: " + "; ".join(f"{f}: tau={s['tau']}, gamma={s['gamma']}" for f, s in selected.items()) + "\n",
             "### Held-out evaluation (800 questions, mean over 10 orders)\n", md_table(evt),
             "Fixed-N self-consistency (held-out):\n", md_table(fixed),
             "### Paired bootstrap accuracy differences, held-out (pp)\n",
             md_table(boots[["a", "b", "diff_pp", "ci_lo_pp", "ci_hi_pp"]]),
             "### Early stops (held-out runs = question x order)\n", md_table(errs),
             "### Retrospective: stopping vs fraction of correct candidates in the full pool\n", md_table(retro),
             f"### Stopping vs leader share at n={ns[0]}\n", md_table(agr)]
    (out / "adaptive.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--datasets", nargs="+", default=["math", "mathvista"])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    for ds in args.datasets:
        run(cfg, ds)


if __name__ == "__main__":
    main()
