"""Final evaluation of the trained VLM on both final-test suites.

    python -m task2.evaluate --config task2/configs/vlm196.yaml

Questions and original references stay fixed; only the image input changes:
  correct        original image
  complementary  the partner image of the pair (different reference answer)
  mismatched     unrelated image from another pair of the same suite (fixed mapping, saved in
                 data/mismatch_test.json, never one of the pair's own images)
  no_image       visual tokens omitted, text unchanged, masks / positions rebuilt
Greedy decoding, thinking disabled, 16 new tokens, batch size 8. Grounding: normalised exact
match (presence / counting); VQAv2: VQA soft accuracy against the ten answers (yes/no, number,
other). Secondary (format-tolerant): the same scores on the leading yes/no or number of the answer
(vqa_norm.lenient_answer), so sentence answers such as 'No, the image does not contain a knife.'
are separated from wrong content. Also pair accuracy, identical-answer rate, empty-answer rate, mean reference-answer NLL
(termination excluded), generation time per answer, and how complementary-swap answers relate to
the partner's answer. 95% CIs: 4,000 bootstrap resamples of complete pairs, the same resampled pair
IDs for every condition (differences are computed per resample).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .common import Paths, Timer, base_argparser, load_config, log, read_json, read_jsonl, write_json, write_jsonl
from .model import VLM, FeatureStore
from .vqa_norm import lenient_answer, normalize_answer, vqa_soft_accuracy

CONDITIONS = ["correct", "complementary", "mismatched", "no_image"]
SUITES = ["grounding", "vqa"]


def items_for(pairs, mm):
    rows = []
    for p in pairs:
        for m, it in enumerate(p["items"]):
            other = p["items"][1 - m]
            rows.append({"suite": p["suite"], "pair_id": p["pair_id"], "member": m, "qtype": p["qtype"],
                         "question": p["question"], "answer": it["answer"], "canonical": it["canonical"],
                         "refs": it["refs"], "partner_canonical": other["canonical"],
                         "images": {"correct": it["image"], "complementary": other["image"],
                                    "mismatched": mm[f"{p['pair_id']}#{m}"], "no_image": None}})
    return rows


def score(row, pred) -> float:
    if row["suite"] == "vqa":
        return vqa_soft_accuracy(pred, row["refs"])
    return float(normalize_answer(pred) == row["canonical"])


def score_lenient(row, pred) -> float:
    """Secondary metric: the leading yes/no or number of a sentence answer (lenient_answer), scored
    like the primary metric. Separates answer content from answer format (e.g. 'No, the image does not
    contain a knife.' is wrong under exact match but 'no' here)."""
    la = lenient_answer(pred, row["qtype"])
    if row["suite"] == "vqa":
        return vqa_soft_accuracy(la, row["refs"])
    return float(la == row["canonical"])


def run_condition(model, fs, rows, cond, bs, max_new, amp):
    out = []
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        use = [cond != "no_image"] * len(chunk)
        keys = [r["images"][cond] or r["images"]["correct"] for r in chunk]  # ignored for no_image
        feats = fs.get(keys)
        exs = [{"question": r["question"], "target": r["answer"], "kind": "answer"} for r in chunk]
        with torch.autocast(model.device.type, dtype=torch.float16, enabled=amp):
            gens = model.generate(exs, feats, use, max_new)
            nll = model.example_losses(exs, feats, use, mask_eot=True).tolist()
        for r, g, n in zip(chunk, gens, nll):
            p = g["text"]
            pn = normalize_answer(p)
            out.append({k: r[k] for k in ("suite", "pair_id", "member", "qtype", "question", "answer",
                                          "canonical", "partner_canonical", "refs")}
                       | {"condition": cond, "image": r["images"][cond], "prediction": p, "prediction_norm": pn,
                          "score": score(r, p), "score_lenient": score_lenient(r, p),
                          "lenient_answer": lenient_answer(p, r["qtype"]), "match_canonical": pn == r["canonical"],
                          "matches_partner": pn == r["partner_canonical"], "empty": pn == "",
                          "n_tokens": g["n_tokens"], "finished": g["finished"],
                          "gen_time_s": g["gen_time_s"], "ref_nll": n})
    return out


def pair_table(samples):
    """pair_id -> condition -> dict of pair-level values."""
    by = defaultdict(lambda: defaultdict(list))
    for r in samples:
        by[r["pair_id"]][r["condition"]].append(r)
    out = {}
    for pid, conds in by.items():
        out[pid] = {}
        for c, rs in conds.items():
            rs = sorted(rs, key=lambda r: r["member"])
            out[pid][c] = {"qtype": rs[0]["qtype"], "score": np.mean([r["score"] for r in rs]),
                           "score_lenient": np.mean([r["score_lenient"] for r in rs]),
                           "pair_correct_lenient": float(all(r["lenient_answer"] == r["canonical"] for r in rs)),
                           "pair_correct": float(all(r["match_canonical"] for r in rs)),
                           "identical": float(rs[0]["prediction_norm"] == rs[1]["prediction_norm"]),
                           "empty": np.mean([r["empty"] for r in rs]), "nll": np.mean([r["ref_nll"] for r in rs]),
                           "follows_partner": np.mean([r["matches_partner"] for r in rs]),
                           "keeps_original": np.mean([r["match_canonical"] for r in rs])}
    return out


def summarize(samples, n_boot: int, seed: int) -> dict:
    pt = pair_table(samples)
    pids = sorted(pt)
    rng = np.random.default_rng(seed)
    res = {"n_pairs": len(pids), "n_questions": len(pids) * 2}
    keys = ["score", "score_lenient", "pair_correct", "pair_correct_lenient", "identical", "empty", "nll", "follows_partner", "keeps_original"]
    groups = {"all": pids}
    for ty in sorted({pt[p]["correct"]["qtype"] for p in pids}):
        groups[ty] = [p for p in pids if pt[p]["correct"]["qtype"] == ty]
    for gname, gp in groups.items():
        arr = {c: {k: np.array([pt[p][c][k] for p in gp]) for k in keys} for c in CONDITIONS}
        idx = rng.integers(0, len(gp), size=(n_boot, len(gp)))
        g = {"n_pairs": len(gp)}
        for c in CONDITIONS:
            for k in keys:
                v = arr[c][k]
                b = v[idx].mean(1)
                g[f"{c}/{k}"] = {"mean": float(v.mean()), "ci95": [float(np.percentile(b, 2.5)),
                                                                   float(np.percentile(b, 97.5))]}
        for other in ["complementary", "mismatched", "no_image"]:
            for k, sign in [("score", 1), ("score_lenient", 1), ("pair_correct", 1), ("nll", -1)]:
                d = sign * (arr["correct"][k] - arr[other][k])  # positive = correct image helps
                b = d[idx].mean(1)
                g[f"advantage_vs_{other}/{k}"] = {"mean": float(d.mean()),
                                                  "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
        res[gname] = g
    gt = defaultdict(list)
    for r in samples:
        gt[r["condition"]].append(r["gen_time_s"])
    res["gen_time_per_answer_s"] = {c: float(np.mean(v)) for c, v in gt.items()}
    return res


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--weights", default=None, help="stage weights (default: final stage)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    paths = Paths(cfg)
    fc = cfg.final_eval
    out = paths.eval_out
    if (out / "metrics.json").exists():
        log("final evaluation exists; skipping")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    wpath = args.weights or paths.stage_weights / f"stage{len(cfg.train.stages)}.pt"
    model = VLM(cfg, device)
    model.load_state(torch.load(wpath, map_location="cpu", weights_only=False))
    model.eval()
    fs = FeatureStore(paths.features)
    mm = read_json(paths.data_out / "mismatch_test.json")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t = Timer()
    allsamples, metrics = [], {"weights": str(wpath), "conditions": CONDITIONS}
    for suite in SUITES:
        rows = items_for(read_jsonl(paths.data_out / f"{suite}_test.jsonl"), mm[suite])
        samples = []
        for c in CONDITIONS:
            samples += run_condition(model, fs, rows, c, int(fc.batch_size), int(fc.max_new_tokens), amp)
            acc = np.mean([r["score"] for r in samples if r["condition"] == c])
            acc_l = np.mean([r["score_lenient"] for r in samples if r["condition"] == c])
            log(f"[{suite}] {c}: accuracy {100 * acc:.2f} (lenient {100 * acc_l:.2f})")
        metrics[suite] = summarize(samples, int(fc.bootstrap_resamples), int(cfg.seed))
        allsamples += samples
    metrics["eval_time_s"] = t.elapsed()
    metrics["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    write_jsonl(out / "samples.jsonl", allsamples)
    write_json(out / "metrics.json", metrics)
    log(f"final evaluation done in {t.elapsed() / 60:.1f} min")


if __name__ == "__main__":
    main()
