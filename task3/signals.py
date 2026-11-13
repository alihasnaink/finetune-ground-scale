"""Task 3, Step 3: what the quality signals capture, on the full 32-rollout pools (CPU).

    python -m task3.signals --config task3/configs/offline.yaml [--datasets math mathvista]

At N = 32 (all rollouts of a question; VisConf calibration pooled over those 32):
  * within-question Spearman correlations of each score with correctness (questions with mixed
    correctness only) and with rollout length (all questions), averaged over questions;
  * top-vs-bottom quartile correctness: within a question, rollouts are ranked by the score (ties by
    rollout id); mean correctness of the top 8 minus the bottom 8, averaged over questions (reported
    over all questions and over mixed questions). "neg_len" (shorter is better) is the length-only control;
  * MathVista: within-rollout Pearson correlation of calibrated confidence c_t and visual engagement v_t,
    summarized per question and across questions, and Cov_t(c, v) = Z^token - Z;
  * illustrative trajectories, by a fixed rule (held-out split, split order, order 0 at N = 32):
      A  first question where VisConf-RWC is correct and self-consistency is wrong
      B  first question where self-consistency is correct and VisConf-RWC is wrong
      C  first question where VisConf-Max and TokFusion-Max pick different rollouts with different correctness
    For each, the highest-Z correct and highest-Z incorrect rollout: c_t and v_t traces.
Writes <results>/<dataset>/signals_*.csv, signals.md, trajectories.csv and figures/trajectories.png.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .common import base_argparser, load_config, log, read_json
from .engine import visconf_scores
from .offline import md_table, split_of
from .offline_data import load_questions

SCORES = {"math": ["L", "K", "neg_len"], "mathvista": ["L", "K", "neg_len", "V", "C", "Z", "Ztok"]}
LABEL = {"L": "mean log-prob", "K": "self-certainty", "neg_len": "shortness (-length)", "V": "visual engagement",
         "C": "calibrated confidence", "Z": "VisConf", "Ztok": "token fusion"}


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).statistic)


def full_scores(q: dict, ds: str, quant) -> dict[str, np.ndarray]:
    s = {"L": q["L"], "K": q["K"], "neg_len": -q["n_tok"].astype(float)}
    if ds == "mathvista":
        s["V"] = q["V"]
        s |= visconf_scores(q, list(range(32)), quant)
    return s


def run(cfg, ds: str):
    out = Path(cfg.paths.results) / ds
    (out / "figures").mkdir(parents=True, exist_ok=True)
    split = read_json(Path(cfg.data[f"split_{ds}"]))
    sp = split_of(split)
    Q = load_questions(cfg, ds)["questions"]
    quant = tuple(float(x) for x in cfg.offline.calib_quantiles)
    rows, cv_rows = [], []
    for qid, q in Q.items():
        s = full_scores(q, ds, quant)
        cor = q["correct"].astype(float)
        mixed = 0 < cor.sum() < 32
        r = {"qid": qid, "split": sp[qid], "mixed": mixed, "frac_correct": cor.mean(),
             "spearman_len_correct": _spearman(q["n_tok"], cor) if mixed else np.nan}
        for k in SCORES[ds]:
            sc = np.where(np.isfinite(s[k]), s[k], -np.inf)
            order = np.lexsort((np.arange(32), -sc))  # descending score, ties by rollout id
            r[f"{k}_top_minus_bottom"] = cor[order[:8]].mean() - cor[order[-8:]].mean()
            r[f"{k}_spearman_correct"] = _spearman(s[k], cor) if mixed else np.nan
            r[f"{k}_spearman_len"] = _spearman(s[k], q["n_tok"])
        if ds == "mathvista":
            r["spearman_Z_Ztok"] = _spearman(s["Z"], s["Ztok"])
            cov = s["Ztok"] - s["Z"]
            r["mean_cov_cv"] = float(np.nanmean(cov))
            # within-rollout confidence-vision correlation
            lo, hi = np.percentile(np.concatenate(q["kl_tok"]).astype(float), [100 * quant[0], 100 * quant[1]])
            cs = []
            for i in range(32):
                k, v = q["kl_tok"][i].astype(float), q["v_tok"][i].astype(float)
                ok = np.isfinite(v)
                if ok.sum() < 10:
                    continue
                c = np.clip((k - lo) / max(hi - lo, 1e-8), 0, 1)[ok]
                if np.ptp(c) == 0 or np.ptp(v[ok]) == 0:
                    continue
                rho = float(pearsonr(c, v[ok]).statistic)
                cs.append(rho)
                cv_rows.append({"qid": qid, "split": sp[qid], "rollout": i, "corr_c_v": rho,
                                "cov_c_v": float(cov[i]), "correct": bool(q["correct"][i]), "T": int(k.size)})
            r["mean_corr_c_v"] = float(np.mean(cs)) if cs else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "signals_per_question.csv", index=False)
    if cv_rows:
        pd.DataFrame(cv_rows).to_csv(out / "signals_conf_vision_rollouts.csv", index=False)

    lines = [f"## {ds}: signal analysis at N = 32 (calibration pooled over all 32 rollouts)\n"]
    for sp_name in ("eval", "all"):
        d = df if sp_name == "all" else df[df.split == sp_name]
        m = d[d.mixed]
        t = []
        for k in SCORES[ds]:
            t.append({"score": f"{k} ({LABEL[k]})",
                      "top-bottom quartile, all q (pp)": 100 * d[f"{k}_top_minus_bottom"].mean(),
                      "top-bottom quartile, mixed q (pp)": 100 * m[f"{k}_top_minus_bottom"].mean(),
                      "spearman(score, correct), mixed q": m[f"{k}_spearman_correct"].mean(),
                      "spearman(score, length)": d[f"{k}_spearman_len"].mean()})
        lines += [f"### {sp_name} split: {len(d)} questions, {len(m)} with mixed correctness; "
                  f"mean spearman(length, correct) on mixed = {m['spearman_len_correct'].mean():.3f}\n",
                  md_table(pd.DataFrame(t), "{:.3f}")]
    if ds == "mathvista":
        cv = pd.DataFrame(cv_rows)
        for sp_name in ("eval", "all"):
            d = df if sp_name == "all" else df[df.split == sp_name]
            c = cv if sp_name == "all" else cv[cv.split == sp_name]
            qm = d["mean_corr_c_v"].dropna()
            lines += [f"### Confidence-vision relationship ({sp_name})\n",
                      f"- within-rollout Pearson(c_t, v_t), per-question mean: mean {qm.mean():.3f}, median "
                      f"{qm.median():.3f}, IQR [{qm.quantile(.25):.3f}, {qm.quantile(.75):.3f}], "
                      f"{100 * (qm > 0).mean():.1f}% of questions positive ({len(qm)} questions, {len(c)} rollouts)",
                      f"- by rollout correctness: correct {c[c.correct].corr_c_v.mean():.3f}, incorrect "
                      f"{c[~c.correct].corr_c_v.mean():.3f}",
                      f"- Cov_t(c, v) = Z^token - Z: mean {d.mean_cov_cv.mean():.5f}; correct rollouts "
                      f"{c[c.correct].cov_c_v.mean():.5f}, incorrect {c[~c.correct].cov_c_v.mean():.5f}",
                      f"- within-question Spearman(Z, Z^token): mean {d.spearman_Z_Ztok.mean():.3f}, median "
                      f"{d.spearman_Z_Ztok.median():.3f}\n"]
        lines += trajectories(cfg, Q, split, out, quant)
    (out / "signals.md").write_text("\n".join(lines))
    print("\n".join(lines))
    log(f"{ds}: wrote {out}/signals.md")


def trajectories(cfg, Q, split, out: Path, quant) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = pd.read_parquet(out / "selections.parquet", filters=[("N", "==", 32), ("order", "==", 0)])
    w = sel.pivot(index="qid", columns="method", values="correct")
    wr = sel.pivot(index="qid", columns="method", values="rollout")
    rules = {"A: VisConf-RWC correct, SC wrong": lambda q: w.at[q, "VisConf-RWC"] and not w.at[q, "SC"],
             "B: SC correct, VisConf-RWC wrong": lambda q: w.at[q, "SC"] and not w.at[q, "VisConf-RWC"],
             "C: VisConf-Max vs TokFusion-Max disagree": lambda q: wr.at[q, "VisConf-Max"] != wr.at[q, "TokFusion-Max"]
             and w.at[q, "VisConf-Max"] != w.at[q, "TokFusion-Max"]}
    picked = {name: next((q for q in split["eval"] if f(q)), None) for name, f in rules.items()}
    recs, lines = [], ["### Illustrative trajectories (fixed rule; held-out split order, order 0, N = 32)\n"]
    fig, axes = plt.subplots(2 * len(picked), 2, figsize=(5.5, 2.3 * len(picked)), sharex="col", squeeze=False,
                             gridspec_kw={"height_ratios": [1.4, 1] * len(picked)})
    colors = {"c": "#2a78d6", "v": "#eb6834"}
    for row, (name, qid) in enumerate(picked.items()):
        if qid is None:
            lines.append(f"- {name}: no question matches")
            continue
        q = Q[qid]
        s = visconf_scores(q, list(range(32)), quant)
        lo, hi = np.percentile(np.concatenate(q["kl_tok"]).astype(float), [100 * quant[0], 100 * quant[1]])
        Z = np.where(np.isfinite(s["Z"]), s["Z"], -np.inf)
        picks = {}
        for lab, mask in (("correct", q["correct"]), ("incorrect", ~q["correct"])):
            if mask.any():
                picks[lab] = int(np.flatnonzero(mask)[np.argmax(Z[mask])])
        ansl = lambda i: "NO_ANSWER" if q["ans"][i] is None else str(q["ans"][i])  # noqa: E731
        lines.append(f"- {name}: question {qid} (gold {q['gold']!r}; {int(q['correct'].sum())}/32 correct); "
                     + "; ".join(f"top-Z {lab} rollout {i}: answer {ansl(i)!r}, T={q['n_tok'][i]}, "
                                 f"Z={s['Z'][i]:.4f}, Ztok={s['Ztok'][i]:.4f}, C={s['C'][i]:.3f}, V={q['V'][i]:.4f}"
                                 for lab, i in picks.items()))
        for col, lab in enumerate(("correct", "incorrect")):
            axc, axv = axes[2 * row, col], axes[2 * row + 1, col]
            for ax in (axc, axv):
                ax.grid(alpha=.25, lw=.5)
                for sp_ in ("top", "right"):
                    ax.spines[sp_].set_visible(False)
                ax.tick_params(labelsize=6, length=2, pad=1)
            if lab not in picks:
                axc.set_axis_off()
                axv.set_axis_off()
                continue
            i = picks[lab]
            c = np.clip((q["kl_tok"][i].astype(float) - lo) / max(hi - lo, 1e-8), 0, 1)
            v = q["v_tok"][i].astype(float)
            win = max(5, len(c) // 40)
            sm = lambda x: pd.Series(x).rolling(win, min_periods=1, center=True).mean().to_numpy()  # noqa: E731
            t = np.arange(len(c))
            axc.plot(t, sm(c), color=colors["c"], lw=1.3)
            axv.plot(t, sm(v), color=colors["v"], lw=1.3)
            axc.set_ylim(0, 1)
            axv.set_ylim(0, None)
            axc.set_ylabel("$c_t$", fontsize=7)
            axv.set_ylabel("$v_t$", fontsize=7)
            axc.set_title(f"{name.split(':')[0]}: q{qid}, rollout {i}, {lab} (answer {ansl(i)[:10]}); "
                          f"Z={s['Z'][i]:.4f}", fontsize=6.5)
            if row == len(picked) - 1:
                axv.set_xlabel("generated token (centered rolling mean)", fontsize=6.5)
            for tt in range(len(c)):
                recs.append({"case": name[0], "qid": qid, "rollout": i, "label": lab, "t": tt,
                             "c": float(c[tt]), "v": float(v[tt])})
    fig.tight_layout(h_pad=0.3, w_pad=0.8)
    fig.savefig(out / "figures" / "trajectories.png", dpi=250)
    plt.close(fig)
    pd.DataFrame(recs).to_csv(out / "trajectories.csv", index=False)
    return lines + ["", "Figure: figures/trajectories.png; token values in trajectories.csv.\n"]


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--datasets", nargs="+", default=["math", "mathvista"])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    for ds in args.datasets:
        run(cfg, ds)


if __name__ == "__main__":
    main()
