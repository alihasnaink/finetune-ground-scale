"""Load, join and verify the supplied rollouts; prepare answer groups and correctness labels (CPU).

    python -m task3.offline_data --config task3/configs/offline.yaml [--datasets math mathvista]

Joins generations, probability_metrics (and, for MathVista, attention_metrics) on
(dataset, sample_id, rollout_id), streaming one row group at a time. Verifies, for every rollout, that
generated_token_ids, logp, kl_u_p (and img/prompt/generated attention) all have n_generated_text_tokens
entries, that the attention n_generated_text_tokens list is [0..T-1], and that all values are finite.

Per question it stores (cache, regenerable, not committed):
  ans      supplied extracted answers, normalised (MATH: stripped string; MathVista: task3.mathvista_norm)
  eq       32 x 32 answer equivalence matrix computed from the candidates only (never from gold):
           MATH math-verify (task3.scoring.equivalent), MathVista equality of normalised answers;
           NO_ANSWER is equivalent only to NO_ANSWER
  correct  per-rollout correctness against the gold answer, using the same equivalence rule
  n_tok    retained token counts; hit_max; L (mean logp); K (mean kl_u_p, self-certainty)
  MathVista only: V (mean visual engagement v_t = img / (img + prompt + generated)) and the per-token
  kl_u_p and v_t arrays used by VisConf (calibration depends on the current candidate set)
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .common import base_argparser, load_config, log, read_json, write_json
from .scoring import NO_ANSWER, equivalent, is_correct, scorer_version

REPO = Path(__file__).resolve().parent.parent
FOLDERS = {"math": "qwen25-math", "mathvista": "qwen25vl-mathvista"}


def cache_path(cfg, ds: str) -> Path:
    return Path(cfg.paths.cache) / f"{ds}.pkl"


def load_questions(cfg, ds: str) -> dict:
    with open(cache_path(cfg, ds), "rb") as fh:
        return pickle.load(fh)


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def build(cfg, ds: str) -> dict:
    root = Path(cfg.paths.data_root) / FOLDERS[ds]
    files = {"gen": pq.ParquetFile(root / "generations.parquet"),
             "prob": pq.ParquetFile(root / "probability_metrics.parquet")}
    if ds == "mathvista":
        files["attn"] = pq.ParquetFile(root / "attention_metrics.parquet")
        meta = read_json(REPO / cfg.data.mathvista_meta)["questions"]
        from .mathvista_norm import VERSION as mv_version
        from .mathvista_norm import gold_label, normalize
    cols = {"gen": ["dataset", "sample_id", "rollout_id", "generated_token_ids", "n_generated_text_tokens",
                    "hit_max_new_tokens", "gold_answer", "extracted_answer"],
            "prob": ["dataset", "sample_id", "rollout_id", "logp", "kl_u_p"],
            "attn": ["dataset", "sample_id", "rollout_id", "n_generated_text_tokens", "img_attn_total",
                     "prompt_text_attn_total", "generated_text_attn_total"]}
    ngroups = {k: f.metadata.num_row_groups for k, f in files.items()}
    assert len(set(ngroups.values())) == 1, ngroups
    # rows of one question may in principle span row groups: accumulate by key, then process
    acc: dict[str, dict] = {}
    checks = {"rollouts": 0, "tokens": 0, "zero_length": 0, "capped": 0, "null_extracted": 0,
              "nonfinite_logp": 0, "nonfinite_kl": 0, "nonfinite_attn": 0, "zero_attn_denominator_tokens": 0}
    t0 = time.perf_counter()
    for g in range(ngroups["gen"]):
        tabs = {k: f.read_row_group(g, columns=cols[k]).to_pydict() for k, f in files.items()}
        keys = {k: list(zip(t["dataset"], t["sample_id"], t["rollout_id"])) for k, t in tabs.items()}
        idx = {k: {key: i for i, key in enumerate(v)} for k, v in keys.items()}
        for k in idx:
            assert set(idx[k]) == set(idx["gen"]), f"row group {g}: {k} keys differ from generations"
        for i, key in enumerate(keys["gen"]):
            gen = {c: tabs["gen"][c][i] for c in cols["gen"]}
            pr = {c: tabs["prob"][c][idx["prob"][key]] for c in ("logp", "kl_u_p")}
            T = int(gen["n_generated_text_tokens"])
            assert len(gen["generated_token_ids"]) == T == len(pr["logp"]) == len(pr["kl_u_p"]), key
            logp, kl = _f(pr["logp"]), _f(pr["kl_u_p"])
            checks["nonfinite_logp"] += int((~np.isfinite(logp)).sum())
            checks["nonfinite_kl"] += int((~np.isfinite(kl)).sum())
            rec = {"rid": int(key[2]), "T": T, "hit_max": bool(gen["hit_max_new_tokens"]),
                   "extracted": gen["extracted_answer"], "gold": gen["gold_answer"],
                   "L": float(logp.mean()) if T else np.nan, "K": float(kl.mean()) if T else np.nan}
            if ds == "mathvista":
                j = idx["attn"][key]
                at = {c: tabs["attn"][c][j] for c in cols["attn"][3:]}
                assert list(at["n_generated_text_tokens"]) == list(range(T)), key
                img, pt, gt = (_f(at[c]) for c in ("img_attn_total", "prompt_text_attn_total",
                                                   "generated_text_attn_total"))
                assert len(img) == len(pt) == len(gt) == T, key
                checks["nonfinite_attn"] += int((~np.isfinite(img) | ~np.isfinite(pt) | ~np.isfinite(gt)).sum())
                den = img + pt + gt
                checks["zero_attn_denominator_tokens"] += int((den <= 0).sum())
                v = np.where(den > 0, img / np.where(den > 0, den, 1), np.nan).astype(np.float32)
                rec |= {"kl_tok": kl, "v_tok": v, "V": float(np.nanmean(v)) if T else np.nan}
            checks["rollouts"] += 1
            checks["tokens"] += T
            checks["zero_length"] += T == 0
            checks["capped"] += rec["hit_max"]
            checks["null_extracted"] += gen["extracted_answer"] is None
            acc.setdefault(key[1], {})[rec["rid"]] = rec
        if g % 100 == 0:
            log(f"{ds}: row group {g}/{ngroups['gen']} ({time.perf_counter() - t0:.0f}s)")

    questions = {}
    t0 = time.perf_counter()
    for n, (qid, rs) in enumerate(sorted(acc.items())):
        assert sorted(rs) == list(range(32)), f"{qid}: rollout ids {sorted(rs)}"
        rs = [rs[r] for r in range(32)]
        golds = {r["gold"] for r in rs}
        assert len(golds) == 1, qid
        gold = golds.pop()
        if ds == "math":
            ans = [None if r["extracted"] is None else r["extracted"].strip() for r in rs]
            uniq = sorted({a for a in ans if a is not None})
            ue = {(a, b): equivalent(a, b) for i, a in enumerate(uniq) for b in uniq[i:]}
            eqf = lambda a, b: ue[(a, b) if (a, b) in ue else (b, a)]  # noqa: E731
            ucor = {a: is_correct(a, gold) for a in uniq}
            gold_norm = gold
        else:
            m = meta[qid]
            ans = [normalize(r["extracted"], m) for r in rs]
            gold_norm = gold_label(gold, m)
            eqf = lambda a, b: a == b  # noqa: E731
            ucor = {a: a == gold_norm for a in set(ans) if a is not None}
        eq = np.zeros((32, 32), dtype=bool)
        for i in range(32):
            for j in range(i, 32):
                a, b = ans[i], ans[j]
                e = (a is None and b is None) or (a is not None and b is not None and eqf(a, b))
                eq[i, j] = eq[j, i] = e
        q = {"qid": qid, "gold": gold, "gold_norm": gold_norm, "ans": ans, "eq": eq,
             "correct": np.array([a is not None and ucor[a] for a in ans]),
             "n_tok": np.array([r["T"] for r in rs], dtype=np.int64),
             "hit_max": np.array([r["hit_max"] for r in rs]),
             "L": np.array([r["L"] for r in rs]), "K": np.array([r["K"] for r in rs])}
        if ds == "mathvista":
            q |= {"V": np.array([r["V"] for r in rs]), "kl_tok": [r["kl_tok"] for r in rs],
                  "v_tok": [r["v_tok"] for r in rs], "question_type": meta[qid]["question_type"],
                  "answer_type": meta[qid]["answer_type"]}
        questions[qid] = q
        if n % 200 == 0:
            log(f"{ds}: grouped {n}/{len(acc)} questions ({time.perf_counter() - t0:.0f}s)")
    assert len(questions) == 1000
    corr = np.concatenate([q["correct"] for q in questions.values()])
    checks |= {"questions": len(questions), "rollout_accuracy": float(corr.mean()),
               "scorer": scorer_version() if ds == "math" else mv_version}
    return {"dataset": ds, "questions": questions, "checks": checks}


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--datasets", nargs="+", default=["math", "mathvista"])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    for ds in args.datasets:
        d = build(cfg, ds)
        p = cache_path(cfg, ds)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump(d, fh, protocol=pickle.HIGHEST_PROTOCOL)
        out = Path(cfg.paths.results) / ds
        write_json(out / "data_checks.json", d["checks"])
        log(f"{ds}: {d['checks']} -> {p}")


if __name__ == "__main__":
    main()
