"""Task 3, Step 6: compute allocation, failure cases and figures; writes the Task 3 results index (CPU).

    python -m task3.analysis --config task3/configs/offline.yaml

Reads the outputs of task3.offline, task3.signals, task3.adaptive and the Step 4 live run. Writes:
  <results>/<ds>/allocation.csv          per held-out N: oracle, SC accuracy, selection gap, distinct groups,
                                         share of newly revealed candidates that repeat an existing answer
                                         group, new answer groups, oracle gain, and (for SC selection
                                         failures) the share of correct candidates in the revealed set
  <results>/figures/accuracy_vs_n.png    accuracy and oracle coverage vs N (both datasets)
  <results>/figures/accuracy_vs_cost.png accuracy vs mean generated-token cost, with the adaptive policies
  <results>/figures/live_strategies.png  Step 4: accuracy vs actual output tokens and runtime
  <results>/cases.md                     representative cases chosen by stated rules
  <results>/README.md                    index of Task 3 results with the headline tables
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .common import base_argparser, load_config, read_json
from .engine import evaluate_set, groups, rwc_weights, visconf_scores, vote_winner
from .offline import md_table
from .offline_data import FOLDERS, load_questions

DS = ("math", "mathvista")
NAME = {"math": "MATH (Qwen2.5-3B-Instruct)", "mathvista": "MathVista (Qwen2.5-VL-3B-Instruct)"}
QUAL = {"math": "SelfCert-RWC", "mathvista": "VisConf-RWC"}
QSCORE = {"math": "K", "mathvista": "Z"}
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED = "#0b0b0b", "#52514e"


def _style(ax):
    ax.grid(alpha=.25, lw=.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors=MUTED)


def allocation(res: Path, ds: str) -> pd.DataFrame:
    sets = pd.read_parquet(res / ds / "sets.parquet", filters=[("split", "==", "eval")])
    sel = pd.read_parquet(res / ds / "selections.parquet", filters=[("split", "==", "eval"), ("method", "==", "SC")])
    s = sets.merge(sel[["qid", "order", "N", "correct"]], on=["qid", "order", "N"]).sort_values(["qid", "order", "N"])
    s["prev_groups"] = s.groupby(["qid", "order"]).n_groups.shift()
    s["prev_oracle"] = s.groupby(["qid", "order"]).oracle.shift()
    s["prev_N"] = s.groupby(["qid", "order"]).N.shift()
    rows = []
    for N, d in s.groupby("N"):
        r = {"N": N, "oracle": 100 * d.oracle.mean(), "sc_accuracy": 100 * d.correct.mean(),
             "selection_gap": 100 * (d.oracle.mean() - d.correct.mean()), "distinct_groups": d.n_groups.mean()}
        if N > 1:
            new = N - d.prev_N
            r["new_groups"] = (d.n_groups - d.prev_groups).mean()
            r["new_candidates_repeating_pct"] = 100 * (1 - ((d.n_groups - d.prev_groups) / new).mean())
            r["oracle_gain_pct"] = 100 * (d.oracle & ~d.prev_oracle.astype(bool)).mean()
        f = d[d.oracle & ~d.correct.astype(bool)]
        r["correct_share_in_sc_selection_failures_pct"] = 100 * (f.n_correct / f.N).mean() if len(f) else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(res / ds / "allocation.csv", index=False)
    return out


def fig_accuracy_vs_n(res: Path, summ: dict):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, ds in zip(axes, DS):
        e = summ[ds][summ[ds].split == "eval"]
        series = [("oracle coverage", "SC", "oracle"), ("self-consistency", "SC", "accuracy"),
                  (QUAL[ds], QUAL[ds], "accuracy"), ("MaxSelfCert", "MaxSelfCert", "accuracy"),
                  ("FirstAns", "FirstAns", "accuracy")]
        for i, (lab, m, col) in enumerate(series):
            d = e[e.method == m].sort_values("N")
            ax.plot(d.N, d[col], marker="o", ms=4, lw=1.6, color=COL[i], label=lab,
                    ls="--" if col == "oracle" else "-")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
        ax.set_xlabel("candidate budget N (rollouts)", fontsize=9, color=MUTED)
        ax.set_ylabel("held-out accuracy / coverage (%)", fontsize=9, color=MUTED)
        ax.set_title(NAME[ds], fontsize=10, color=INK)
        _style(ax)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, [x.replace(QUAL["math"], "quality RWC (SelfCert / VisConf)") for x in lab], loc="lower center",
               ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(res / "figures" / "accuracy_vs_n.png", dpi=160)
    plt.close(fig)


def fig_accuracy_vs_cost(res: Path, summ: dict):
    """Drawn at its printed size (5.5 in = \linewidth) so fonts appear at 7-8 pt."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.75))
    for ax, ds in zip(axes, DS):
        e = summ[ds][summ[ds].split == "eval"]
        for i, (m, lab) in enumerate([("SC", "fixed-N self-consistency"), (QUAL[ds], "fixed-N quality RWC")]):
            d = e[e.method == m].sort_values("N")
            ax.plot(d.mean_cost, d.accuracy, marker="o", ms=3, lw=1.3, color=COL[i + 1], label=lab)
            if i == 0:
                for _, r in d.iterrows():
                    ax.annotate(f"{int(r.N)}", (r.mean_cost, r.accuracy), textcoords="offset points",
                                xytext=(2, -8), fontsize=6, color=MUTED)
        ad = pd.read_csv(res / ds / "adaptive_eval.csv")
        for j, (_, r) in enumerate(ad.iterrows()):
            ax.scatter([r.mean_cost], [r.accuracy], s=28, marker="D" if j == 0 else "s", color=COL[2 + j * 4],
                       edgecolor="white", linewidth=1.0, zorder=5,
                       label="adaptive: agreement stop" if j == 0 else "adaptive: quality-aware stop")
        ax.set_xscale("log")
        ax.set_xlabel("mean generated tokens per question", fontsize=7.5, color=MUTED)
        ax.set_title(NAME[ds], fontsize=8, color=INK)
        _style(ax)
        ax.tick_params(labelsize=6.5)
    axes[0].set_ylabel("held-out accuracy (%)", fontsize=7.5, color=MUTED)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=6.5, frameon=False, handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.0)
    fig.savefig(res / "figures" / "accuracy_vs_cost.png", dpi=250)
    plt.close(fig)


def fig_live(res: Path, live: dict):
    import matplotlib.pyplot as plt
    s = {d["strategy"]: d for d in live["strategies"]}
    names = list(s)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (key, lab) in zip(axes, [("mean_output_tokens", "mean output tokens per question"),
                                     ("mean_time_s", "mean end-to-end time per question (s)")]):
        for i, n in enumerate(names):
            ax.scatter([s[n][key]], [s[n]["accuracy_pct"]], s=70, color=COL[i], edgecolor="white",
                       linewidth=1.5, zorder=3)
            ax.annotate(n, (s[n][key], s[n]["accuracy_pct"]), textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=INK)
        ax.set_xlabel(lab, fontsize=9, color=MUTED)
        ax.set_ylabel("accuracy on 50 questions (%)", fontsize=9, color=MUTED)
        ax.set_ylim(50, 75)
        _style(ax)
    fig.suptitle("Step 4: same 3,072-token allowance, actual cost (T4, FP16, batch 1)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(res / "figures" / "live_strategies.png", dpi=160)
    plt.close(fig)


def _a(x) -> str:
    return "NO_ANSWER" if x is None or (isinstance(x, float) and x != x) else str(x)


def _excerpt(cfg, ds: str, qid: str, rollouts: list[int], n=260) -> dict[int, str]:
    f = Path(cfg.paths.data_root) / FOLDERS[ds] / "generations.parquet"
    t = pq.read_table(f, columns=["sample_id", "rollout_id", "generated_text"],
                      filters=[("sample_id", "==", qid)]).to_pandas()
    t = t[t.rollout_id.isin(rollouts)]
    return {int(r.rollout_id): " ".join(r.generated_text[-n:].split()) for r in t.itertuples()}


def _group_table(q, cand, ds, quant) -> str:
    gid, reps = groups(q, cand)
    s = {"K": q["K"][cand]} if ds == "math" else visconf_scores(q, cand, quant)
    w = rwc_weights(s[QSCORE[ds]])
    rows = []
    for g, rp in enumerate(reps):
        mem = [p for p, x in enumerate(gid) if x == g]
        rows.append({"answer": _a(q["ans"][cand[rp]])[:30], "correct": bool(q["correct"][cand[rp]]),
                     "votes": len(mem), "RWC weight": float(w[mem].sum()),
                     f"mean {QSCORE[ds]}": float(np.nanmean(s[QSCORE[ds]][mem]))})
    return md_table(pd.DataFrame(rows).sort_values("votes", ascending=False).head(5), "{:.3f}")


def cases(cfg, res: Path, live_dir: Path) -> str:
    quant = tuple(float(x) for x in cfg.offline.calib_quantiles)
    L = ["# Task 3: representative cases\n",
         "Selection rules are fixed in advance and stated per section; they use the saved orders and the "
         "held-out split order. Gold answers are used only to pick illustrations, never by any method.\n"]
    for ds in DS:
        split = read_json(Path(cfg.data[f"split_{ds}"]))
        Q = load_questions(cfg, ds)["questions"]
        orders = read_json(res / ds / "orders.json")["orders"]
        sel = pd.read_parquet(res / ds / "selections.parquet",
                              filters=[("N", "==", 32), ("order", "==", 0), ("split", "==", "eval")])
        w = sel.pivot(index="qid", columns="method", values="correct")
        L.append(f"## {NAME[ds]}\n")
        L.append(f"### Selection disagreement (rule: first held-out question in split order where, at N = 32 "
                 f"and order 0, self-consistency and {QUAL[ds]} differ in correctness; one per direction)\n")
        for lab, f in [(f"{QUAL[ds]} right, SC wrong", lambda x: w.at[x, QUAL[ds]] and not w.at[x, "SC"]),
                       (f"SC right, {QUAL[ds]} wrong", lambda x: w.at[x, "SC"] and not w.at[x, QUAL[ds]])]:
            qid = next((x for x in split["eval"] if f(x)), None)
            if qid is None:
                continue
            q = Q[qid]
            cand = orders[qid][0]
            res_, _ = evaluate_set(q, cand, ds, quant)
            L.append(f"**{lab}**: question `{qid}`, gold `{q['gold']}`, {int(q['correct'].sum())}/32 correct. "
                     f"SC picks `{_a(res_['SC'][1])}`; {QUAL[ds]} picks `{_a(res_[QUAL[ds]][1])}`.\n")
            L.append(_group_table(q, cand, ds, quant))
        # premature stopping
        sp = json.loads((res / ds / "adaptive_selected.json").read_text())
        st = pd.read_parquet(res / ds / "stop_states.parquet", filters=[("split", "==", "eval")])
        from .adaptive import simulate, wide_states
        ns = sorted(int(n) for n in cfg.adaptive.checks) + [int(cfg.adaptive.max_n)]
        W = wide_states(st, ns)
        sim = simulate(W, "agreement", sp["agreement"]["tau"], None)
        bad = sim[(sim.stop_n < 32) & ~sim.correct.astype(bool) & sim.sc32_correct.astype(bool)]
        L.append(f"### Premature stopping (rule: first held-out question in split order, lowest order index, where "
                 f"the frozen agreement policy (tau = {sp['agreement']['tau']}) stops early with a wrong answer "
                 f"that fixed-32 self-consistency gets right)\n")
        if len(bad):
            pos = {x: i for i, x in enumerate(split["eval"])}
            b = bad.assign(p=bad.qid.map(pos)).sort_values(["p", "order"]).iloc[0]
            q = Q[b.qid]
            cand = orders[b.qid][int(b.order)]
            n = int(b.stop_n)
            g1, r1 = groups(q, cand[:n])
            lead = cand[r1[vote_winner(g1, r1, np.ones(n))[0]]]
            ex = _excerpt(cfg, ds, b.qid, [lead])
            L.append(f"Question `{b.qid}` (order {int(b.order)}), gold `{q['gold']}`, {int(q['correct'].sum())}/32 "
                     f"correct in the full pool. Stopped at n = {n} with answer `{_a(q['ans'][lead])}`.\n\n"
                     f"Answer groups among the first {n}:\n")
            L.append(_group_table(q, cand[:n], ds, quant))
            L.append("All 32 candidates:\n")
            L.append(_group_table(q, cand, ds, quant))
            L.append(f"End of a leading-group rollout ({lead}): \"...{ex.get(lead, '')}\"\n")
        L.append(f"Totals (held-out, agreement policy): {len(bad)} of {len(sim[sim.stop_n < 32])} early stops "
                 f"are wrong-but-recoverable by fixed-32 SC.\n")
    # revision cases (Step 4)
    pqf = pd.read_csv(live_dir / "per_question.csv")
    calls = {}
    for f in sorted(live_dir.glob("calls_lane*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            calls[(r["qid"], r["call"])] = r
    L.append("## Step 4: sequential refinement (MATH live subset)\n")
    L.append("Rule: every question whose revised answer differs from the shared initial answer, then the first "
             "question (live index order) of each transition class that occurs.\n")
    pqf["changed"] = pqf.initial_answer.astype(str) != pqf.refine_answer.astype(str)
    pqf["trans"] = pqf.initial_correct.map({True: "C", False: "W"}) + "->" + pqf.refine_correct.map({True: "C", False: "W"})
    pick = list(pqf[pqf.changed].qid)
    for t, d in pqf.groupby("trans"):
        if d.qid.iloc[0] not in pick:
            pick.append(d.qid.iloc[0])
    rows = []
    for qid in pick:
        r = pqf[pqf.qid == qid].iloc[0]
        ini, rev = calls[(qid, "parallel_0")], calls[(qid, "refine")]
        rows.append({"qid": qid, "gold": r.gold, "initial": _a(r.initial_answer), "initial_tokens": ini["output_tokens"],
                     "initial_truncated": ini["truncated"], "revised": _a(r.refine_answer),
                     "revision_tokens": rev["output_tokens"], "transition": r.trans,
                     "revision_opening": " ".join(rev["response"][:110].split())})
    L.append(md_table(pd.DataFrame(rows)))
    return "\n".join(L)


def main():
    ap = base_argparser(__doc__)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    res = Path(cfg.paths.results)
    (res / "figures").mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    summ = {ds: pd.read_csv(res / ds / "summary.csv") for ds in DS}
    alloc = {ds: allocation(res, ds) for ds in DS}
    fig_accuracy_vs_n(res, summ)
    fig_accuracy_vs_cost(res, summ)
    live_dir = Path(cfg.paths.live)
    live = read_json(live_dir / "summary.json")
    fig_live(res, live)
    (res / "cases.md").write_text(cases(cfg, res, live_dir))

    L = ["# Task 3 results (machine-readable outputs and headline tables)\n",
         "Reproduce: `python -m task3.offline_data`, `task3.offline`, `task3.signals`, `task3.adaptive`, "
         "`task3.analysis` (all with `--config task3/configs/offline.yaml`); Step 4 on Kaggle via "
         "`notebooks/task3_live.ipynb` (outputs in `kaggle_outputs/task3_live/`). All numbers are on the "
         "800-question held-out split unless stated, averaged over 10 seeded orders.\n",
         "Per dataset: `summary.md` (Steps 1-2), `signals.md` (Step 3), `adaptive.md` (Step 5), `allocation.csv` "
         "(Step 6), `data_checks.json`; figures in `figures/` and `<ds>/figures/`; cases in `cases.md`.\n"]
    for ds in DS:
        chk = read_json(res / ds / "data_checks.json")
        L.append(f"## {NAME[ds]}\n")
        L.append(f"Scorer: {chk['scorer']}. Rollout-level accuracy {100 * chk['rollout_accuracy']:.2f}%, "
                 f"{chk['capped']} capped, {chk['null_extracted']} null extractions (kept as NO_ANSWER).\n")
        L.append("### Compute allocation (self-consistency)\n")
        L.append(md_table(alloc[ds]))
        ad = pd.read_csv(res / ds / "adaptive_eval.csv")
        L.append("### Adaptive policies (frozen on dev)\n")
        L.append(md_table(ad[["policy", "accuracy", "mean_cost", "mean_rollouts", "fixed_SC_acc_at_same_cost"]]))
    L.append("## Step 4 live strategies\n")
    L.append((live_dir / "summary.md").read_text())
    (res / "README.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
