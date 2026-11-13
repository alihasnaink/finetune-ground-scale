"""Task 3, Steps 1-2: repeated-sampling opportunity and quality-scored selection on the supplied pools (CPU).

    python -m task3.offline --config task3/configs/offline.yaml [--datasets math mathvista]

For every question, ten seeded permutations of the 32 rollout ids are drawn (numpy default_rng(6304),
questions in the split file's permuted order, ten permutations each) and saved; candidate budgets
N in {1, 2, 4, 8, 16, 32} are nested prefixes of each permutation. The same orders are reused by every
method and by the stopping policies (task3.adaptive). At each prefix every method of task3.engine
selects one answer; FirstAns consumes (and is charged for) only its first candidate, every other method
consumes all N. Outcomes are averaged over orders within a question, then over questions.

Writes, under <results>/<dataset>/:
  orders.json              the ten permutations per question
  selections.parquet       per question x order x N x method: selected rollout, answer, correct, cost
  sets.parquet             per question x order x N: oracle coverage, distinct answer groups, cost
  stop_states.parquet      per question x order x n: vote state + RWC state (for task3.adaptive)
  per_question.csv         per question x N x method, averaged over orders (+ split)
  summary.csv / .md        per split x N x method: accuracy, oracle coverage, groups, token cost,
                           and the decomposition into sampling failures (no correct candidate) and
                           selection failures (a correct candidate was available but missed)
  bootstrap.csv            paired question-level bootstrap (1,000 resamples) of accuracy differences
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import base_argparser, load_config, log, read_json, write_json
from .engine import METHODS, evaluate_set, rwc_state
from .offline_data import load_questions

SPLITS = ("dev", "eval")


def make_orders(split: dict, n_orders: int, seed: int) -> dict[str, list[list[int]]]:
    rng = np.random.default_rng(seed)
    return {qid: [rng.permutation(32).tolist() for _ in range(n_orders)] for qid in split["dev"] + split["eval"]}


def split_of(split: dict) -> dict[str, str]:
    return {q: s for s in SPLITS for q in split[s]}


def paired_boot(a, b, n: int, seed: int) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(1)
    return {"diff_pp": 100 * d.mean(), "ci_lo_pp": 100 * np.percentile(bs, 2.5),
            "ci_hi_pp": 100 * np.percentile(bs, 97.5)}


def md_table(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.astype(object).itertuples(index=False):  # keep int columns as ints
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


# comparisons reported with bootstrap intervals (a minus b)
COMPARISONS = {
    "math": [("SC", "FirstAns"), ("MaxSelfCert", "MaxLogP"), ("SelfCert-RWC", "SC"), ("SelfCert-RWC", "MaxSelfCert"),
             ("MaxSelfCert", "SC"), ("Shortest", "SC")],
}
COMPARISONS["mathvista"] = COMPARISONS["math"] + [
    ("VisConf-RWC", "SC"), ("VisConf-RWC", "VisConf-Max"), ("VisConf-Max", "MaxSelfCert"),
    ("Vision-RWC", "SC"), ("VisConf-RWC", "SelfCert-RWC"), ("TokFusion-Max", "VisConf-Max"),
    ("TokFusion-RWC", "VisConf-RWC")]


def run(cfg, ds: str):
    o = cfg.offline
    out = Path(cfg.paths.results) / ds
    out.mkdir(parents=True, exist_ok=True)
    split = read_json(Path(cfg.data[f"split_{ds}"]))
    sp = split_of(split)
    orders = make_orders(split, int(o.n_orders), int(cfg.seed))
    write_json(out / "orders.json", {"seed": int(cfg.seed), "rule": "numpy default_rng(seed); questions in the "
                                     "split's permuted order (dev then eval); n_orders permutations each",
                                     "orders": orders})
    Q = load_questions(cfg, ds)["questions"]
    budgets = [int(b) for b in o.budgets]
    checks = set(int(n) for n in cfg.adaptive.checks) | {int(cfg.adaptive.max_n)}
    qscore = cfg.adaptive.quality_score[ds]
    quant = tuple(float(x) for x in o.calib_quantiles)
    sel, sets, stops = [], [], []
    t0 = time.perf_counter()
    for k, (qid, ords) in enumerate(orders.items()):
        q = Q[qid]
        for oi, perm in enumerate(ords):
            for N in budgets:
                cand = perm[:N]
                res, st = evaluate_set(q, cand, ds, quant)
                for m, (rid, ans, cor) in res.items():
                    sel.append((qid, oi, N, m, rid, ans, cor, int(q["n_tok"][rid]) if m == "FirstAns" else st["cost"]))
                sets.append((qid, oi, N, st["oracle"], st["n_groups"], st["n_groups_answered"], st["cost"],
                             int(q["correct"][cand].sum())))
                if N in checks:
                    if N > 1:
                        rw, margin = rwc_state(st["scores"][qscore], st["gid"], st["reps"])
                    else:
                        rw, margin = 0, 1.0
                    sc_rid = cand[st["reps"][st["sc_group"]]]
                    stops.append((qid, oi, N, st["sc_votes"] / N, st["sc_unique"], st["sc_is_na"],
                                  rw == st["sc_group"], margin, bool(q["correct"][sc_rid]), q["ans"][sc_rid],
                                  st["cost"]))
        if k % 100 == 0:
            log(f"{ds}: {k}/{len(orders)} questions ({time.perf_counter() - t0:.0f}s)")
    sel = pd.DataFrame(sel, columns=["qid", "order", "N", "method", "rollout", "answer", "correct", "cost"])
    sets = pd.DataFrame(sets, columns=["qid", "order", "N", "oracle", "n_groups", "n_groups_answered", "cost",
                                       "n_correct"])
    stops = pd.DataFrame(stops, columns=["qid", "order", "n", "lead_frac", "unique_leader", "leader_is_na",
                                         "rwc_agrees", "rwc_margin", "sc_correct", "sc_answer", "cost"])
    for df in (sel, sets, stops):
        df.insert(1, "split", df["qid"].map(sp))
    sel.to_parquet(out / "selections.parquet", index=False)
    sets.to_parquet(out / "sets.parquet", index=False)
    stops.to_parquet(out / "stop_states.parquet", index=False)
    summarize(cfg, ds, sel, sets, out)


def summarize(cfg, ds: str, sel: pd.DataFrame, sets: pd.DataFrame, out: Path):
    s = sel.merge(sets[["qid", "order", "N", "oracle"]], on=["qid", "order", "N"])
    s["sampling_fail"] = ~s["oracle"]
    s["selection_fail"] = s["oracle"] & ~s["correct"]
    pq = (s.groupby(["qid", "split", "N", "method"], sort=False)
          [["correct", "oracle", "sampling_fail", "selection_fail", "cost"]].mean().reset_index())
    g = sets.groupby(["qid", "N"])[["n_groups", "n_groups_answered"]].mean().reset_index()
    pq = pq.merge(g, on=["qid", "N"])
    pq.to_csv(out / "per_question.csv", index=False)
    rows = []
    for sp_name, d in [("all", pq)] + [(x, pq[pq.split == x]) for x in SPLITS]:
        a = d.groupby(["N", "method"], sort=False).agg(
            n_questions=("qid", "nunique"), accuracy=("correct", "mean"), oracle=("oracle", "mean"),
            sampling_fail=("sampling_fail", "mean"), selection_fail=("selection_fail", "mean"),
            mean_cost=("cost", "mean"), distinct_groups=("n_groups", "mean"),
            distinct_answered_groups=("n_groups_answered", "mean")).reset_index()
        a.insert(0, "split", sp_name)
        rows.append(a)
    summ = pd.concat(rows)
    for c in ("accuracy", "oracle", "sampling_fail", "selection_fail"):
        summ[c] = 100 * summ[c]
    summ["selection_gap"] = summ["oracle"] - summ["accuracy"]
    summ.to_csv(out / "summary.csv", index=False)

    boots = []
    n_boot, seed = int(cfg.offline.n_boot), int(cfg.seed)
    for sp_name in ("eval", "dev", "all"):
        d = pq if sp_name == "all" else pq[pq.split == sp_name]
        wide = d.pivot_table(index="qid", columns=["N", "method"], values="correct")
        for N in sorted(d.N.unique()):
            for a, b in COMPARISONS[ds]:
                r = paired_boot(wide[(N, a)], wide[(N, b)], n_boot, seed)
                boots.append({"split": sp_name, "N": N, "a": a, "b": b, **r})
            if N > 1:  # same method, more samples
                for m in ("SC", "SelfCert-RWC") + (("VisConf-RWC",) if ds == "mathvista" else ()):
                    r = paired_boot(wide[(N, m)], wide[(1, m)], n_boot, seed)
                    boots.append({"split": sp_name, "N": N, "a": f"{m}@{N}", "b": f"{m}@1", **r})
    boots = pd.DataFrame(boots)
    boots.to_csv(out / "bootstrap.csv", index=False)

    ev = summ[summ.split == "eval"]
    lines = [f"## {ds}: held-out evaluation split ({int(ev.n_questions.iloc[0])} questions), "
             f"mean over 10 orders\n"]
    acc = ev.pivot(index="method", columns="N", values="accuracy").loc[list(METHODS[ds])]
    acc.columns = [f"N={c}" for c in acc.columns]
    lines += ["### Accuracy (%)\n", md_table(acc.reset_index())]
    one = ev[ev.method == "SC"][["N", "oracle", "accuracy", "sampling_fail", "selection_fail", "distinct_groups",
                                 "distinct_answered_groups", "mean_cost"]]
    first = ev[ev.method == "FirstAns"][["N", "accuracy", "mean_cost"]].rename(
        columns={"accuracy": "firstans_acc", "mean_cost": "firstans_cost"})
    lines += ["### Opportunity and failure decomposition (self-consistency selector)\n",
              md_table(one.merge(first, on="N").rename(columns={"accuracy": "sc_acc", "mean_cost": "sc_cost"}))]
    dec = ev[ev.N == 32][["method", "accuracy", "oracle", "sampling_fail", "selection_fail", "selection_gap"]]
    lines += ["### N = 32: sampling vs selection failures by method (%)\n", md_table(dec)]
    b = boots[(boots.split == "eval") & boots.N.isin([4, 8, 32]) & ~boots.a.str.contains("@")]
    lines += ["### Paired accuracy differences, held-out (pp, 95% bootstrap CI)\n",
              md_table(b[["N", "a", "b", "diff_pp", "ci_lo_pp", "ci_hi_pp"]])]
    (out / "summary.md").write_text("\n".join(lines))
    log(f"{ds}: wrote {out}/summary.md")
    print("\n".join(lines))


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--datasets", nargs="+", default=["math", "mathvista"])
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    for ds in args.datasets:
        if args.summarize_only:
            out = Path(cfg.paths.results) / ds
            summarize(cfg, ds, pd.read_parquet(out / "selections.parquet"), pd.read_parquet(out / "sets.parquet"), out)
        else:
            run(cfg, ds)


if __name__ == "__main__":
    main()
