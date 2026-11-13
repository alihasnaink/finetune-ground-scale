"""Score the live MATH generations (CPU; runs after task3.live).

    python -m task3.live_eval --config task3/configs/live.yaml

Final answers: single = its \\boxed{} answer; parallel = self-consistency over the three responses
(math-verify answer groups, NO_ANSWER kept as a group, ties to the earliest response); refine = the
revised response's answer (never replaced by the initial one); initial = the first parallel response.
Costs per question: input tokens (prompts, including the longer revision prompt), output tokens
(everything generated, including critique), end-to-end time; the shared initial response is
charged to both parallel and refine. Transitions are relative to the shared initial answer. 95% CIs
of accuracy differences: 1,000 paired bootstrap resamples of questions (seed 6304).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .common import base_argparser, load_config, log, read_json, read_jsonl, write_json
from .extract import EXTRACTOR_VERSION
from .live import STRATEGY_CALLS
from .scoring import is_correct, scorer_version, self_consistency

STRATS = ["single", "parallel", "refine"]


def per_question(calls: dict, live: list[dict]) -> pd.DataFrame:
    rows = []
    for q in live:
        qid = q["qid"]
        c = {k: calls.get((qid, k)) for k in STRATEGY_CALLS}
        if any(v is None for v in c.values()):
            continue
        par = [c[f"parallel_{j}"] for j in range(3)]
        sc, votes = self_consistency([p["extracted"] for p in par])
        gold = q["gold"]
        r = {"qid": qid, "live_index": q["live_index"], "gold": gold,
             "initial_answer": par[0]["extracted"], "single_answer": c["single"]["extracted"],
             "parallel_answer": sc, "parallel_votes": dict(votes), "refine_answer": c["refine"]["extracted"],
             "parallel_answers": [p["extracted"] for p in par]}
        r["initial_correct"] = is_correct(r["initial_answer"], gold)
        r["single_correct"] = is_correct(r["single_answer"], gold)
        r["parallel_correct"] = is_correct(sc, gold)
        r["refine_correct"] = is_correct(r["refine_answer"], gold)
        r["parallel_oracle"] = any(is_correct(p["extracted"], gold) for p in par)
        used = {"single": [c["single"]], "parallel": par, "refine": [par[0], c["refine"]]}
        for s, cs in used.items():
            r[f"{s}_input_tokens"] = sum(x["input_tokens"] for x in cs)
            r[f"{s}_output_tokens"] = sum(x["output_tokens"] for x in cs)
            r[f"{s}_time_s"] = sum(x["time_s"] for x in cs)
            r[f"{s}_n_truncated"] = sum(bool(x["truncated"]) for x in cs)
            r[f"{s}_n_extracted"] = sum(x["extracted"] is not None for x in cs)
            r[f"{s}_n_responses"] = len(cs)
            r[f"{s}_peak_mem_gb"] = max((x["peak_mem_gb"] or 0) for x in cs)
        r["refine_revision_input_tokens"] = c["refine"]["input_tokens"]
        r["refine_revision_output_tokens"] = c["refine"]["output_tokens"]
        r["refine_revision_truncated"] = bool(c["refine"]["truncated"])
        r["refine_answer_changed"] = not (r["initial_answer"] is None and r["refine_answer"] is None) and not (
            r["initial_answer"] is not None and r["refine_answer"] is not None
            and is_correct(r["refine_answer"], r["initial_answer"]))
        rows.append(r)
    return pd.DataFrame(rows)


def boot_diff(a, b, n=1000, seed=6304):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(1)
    return {"mean_pp": 100 * d.mean(), "ci95_pp": [100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)]}


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = base_argparser(__doc__)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = Path(cfg.paths.out_root) / "live"
    live = read_json(out / "live_questions.json")
    calls = {}
    for f in sorted(out.glob("calls_lane*.jsonl")):
        for r in read_jsonl(f):
            calls[(r["qid"], r["call"])] = r
    df = per_question(calls, live)
    n = len(df)
    log(f"{n}/{len(live)} questions complete")
    if n == 0:
        return
    summ = []
    for s in STRATS:
        summ.append({"strategy": s, "n_questions": n, "accuracy_pct": 100 * df[f"{s}_correct"].mean(),
                     "final_no_answer_pct": 100 * df[f"{s}_answer"].isna().mean(),
                     "responses_extracted_pct": 100 * df[f"{s}_n_extracted"].sum() / df[f"{s}_n_responses"].sum(),
                     "responses_truncated_pct": 100 * df[f"{s}_n_truncated"].sum() / df[f"{s}_n_responses"].sum(),
                     "mean_input_tokens": df[f"{s}_input_tokens"].mean(),
                     "mean_output_tokens": df[f"{s}_output_tokens"].mean(),
                     "output_allowance": int(cfg.strategies.total_allowance),
                     "mean_time_s": df[f"{s}_time_s"].mean(), "total_time_min": df[f"{s}_time_s"].sum() / 60,
                     "peak_mem_gb": df[f"{s}_peak_mem_gb"].max()})
    summ = pd.DataFrame(summ)
    trans = Counter(("C" if a else "W") + "->" + ("C" if b else "W")
                    for a, b in zip(df["initial_correct"], df["refine_correct"]))
    res = {
        "n_questions": n, "scorer": scorer_version(), "extractor": EXTRACTOR_VERSION,
        "strategies": summ.to_dict("records"),
        "initial_accuracy_pct": 100 * df["initial_correct"].mean(),
        "parallel_oracle_coverage_pct": 100 * df["parallel_oracle"].mean(),
        "refine_transitions": {k: trans.get(k, 0) for k in ["W->C", "C->W", "C->C", "W->W"]},
        "refine_answer_changed_pct": 100 * df["refine_answer_changed"].mean(),
        "differences": {f"{a}_minus_{b}": boot_diff(df[f"{a}_correct"], df[f"{b}_correct"])
                        for a, b in [("parallel", "single"), ("refine", "single"), ("refine", "parallel"),
                                     ("refine", "initial"), ("parallel", "initial")]},
    }
    df.to_csv(out / "per_question.csv", index=False)
    write_json(out / "summary.json", res)
    t = pd.DataFrame([{"transition (initial -> revised)": k, "questions": v}
                      for k, v in res["refine_transitions"].items()])
    d = pd.DataFrame([{"comparison": k, "accuracy diff (pp)": v["mean_pp"],
                       "95% CI": f"[{v['ci95_pp'][0]:.1f}, {v['ci95_pp'][1]:.1f}]"}
                      for k, v in res["differences"].items()])
    text = (f"### Live MATH strategies ({n} questions, same 3,072-token output allowance; {res['scorer']}, "
            f"{EXTRACTOR_VERSION})\n\n" + md(summ)
            + f"\nShared initial response accuracy {res['initial_accuracy_pct']:.1f}%; parallel oracle coverage "
              f"(any of 3 correct) {res['parallel_oracle_coverage_pct']:.1f}%; revised answer differs from the "
              f"initial one in {res['refine_answer_changed_pct']:.1f}% of questions.\n\n"
            + "### Refinement transitions\n\n" + md(t)
            + "\n### Paired accuracy differences (1,000 question bootstrap resamples)\n\n" + md(d))
    (out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
