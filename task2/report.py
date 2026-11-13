"""Task 2 tables, figures, qualitative pair examples and the evidence zip.

    python -m task2.report --config task2/configs/vlm196.yaml --zip /kaggle/working/task2_report.zip

Covers every run found under runs/ (vlm196 and, if trained, the optional vlm49).

Qualitative selection rule (fixed before looking at the examples): for each run, suite and
question type, pairs are taken in pair-ID order and the first pair of each category is shown:
(a) both answers correct with the correct images, (b) identical answers for the two images of the
pair (image-insensitive), (c) different answers for the two images but at least one of them wrong
(image-sensitive error). A pair is shown at most once per run; categories without a matching pair
are skipped.
"""

from __future__ import annotations

import shutil
import textwrap
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from task1.report import save_table  # noqa: E402

from .common import Paths, base_argparser, load_config, log, read_json, read_jsonl  # noqa: E402
from .evaluate import CONDITIONS  # noqa: E402

RUN_ORDER = ["vlm196", "vlm49"]


def _ci(d):
    return f"{100 * d['mean']:.1f} [{100 * d['ci95'][0]:.1f}, {100 * d['ci95'][1]:.1f}]"


def _ci_raw(d):
    return f"{d['mean']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"


def final_tables(paths, runs, rep):
    main, types, adv, swap = [], [], [], []
    for r in runs:
        f = paths.out / "runs" / r / "final_eval" / "metrics.json"
        if not f.exists():
            continue
        m = read_json(f)
        for suite in ["grounding", "vqa"]:
            s = m[suite]
            metric = "VQA soft acc" if suite == "vqa" else "exact match"
            for c in CONDITIONS:
                a = s["all"]
                main.append({"run": r, "suite": suite, "condition": c, "metric": metric,
                             "accuracy_pct": 100 * a[f"{c}/score"]["mean"], "accuracy_ci95": _ci(a[f"{c}/score"]),
                             "pair_accuracy_pct": 100 * a[f"{c}/pair_correct"]["mean"],
                             "pair_accuracy_ci95": _ci(a[f"{c}/pair_correct"]),
                             "lenient_accuracy_pct": (100 * a[f"{c}/score_lenient"]["mean"]
                                                      if f"{c}/score_lenient" in a else None),
                             "lenient_accuracy_ci95": _ci(a[f"{c}/score_lenient"]) if f"{c}/score_lenient" in a else None,
                             "identical_answer_pct": 100 * a[f"{c}/identical"]["mean"],
                             "empty_answer_pct": 100 * a[f"{c}/empty"]["mean"],
                             "ref_nll": a[f"{c}/nll"]["mean"], "ref_nll_ci95": _ci_raw(a[f"{c}/nll"]),
                             "gen_time_ms_per_answer": 1000 * m[suite]["gen_time_per_answer_s"][c]})
                for ty, g in s.items():
                    if not isinstance(g, dict) or ty == "all" or f"{c}/score" not in g:
                        continue
                    types.append({"run": r, "suite": suite, "qtype": ty, "condition": c, "n_pairs": g["n_pairs"],
                                  "accuracy_pct": 100 * g[f"{c}/score"]["mean"], "accuracy_ci95": _ci(g[f"{c}/score"]),
                                  "pair_accuracy_pct": 100 * g[f"{c}/pair_correct"]["mean"],
                                  "lenient_accuracy_pct": (100 * g[f"{c}/score_lenient"]["mean"]
                                                           if f"{c}/score_lenient" in g else None),
                                  "ref_nll": g[f"{c}/nll"]["mean"]})
            for ty, g in s.items():
                if not isinstance(g, dict) or "correct/score" not in g:
                    continue
                for other in ["complementary", "mismatched", "no_image"]:
                    adv.append({"run": r, "suite": suite, "group": ty, "vs": other,
                                "accuracy_advantage_pp": _ci(g[f"advantage_vs_{other}/score"]),
                                "pair_accuracy_advantage_pp": _ci(g[f"advantage_vs_{other}/pair_correct"]),
                                "lenient_accuracy_advantage_pp": (_ci(g[f"advantage_vs_{other}/score_lenient"])
                                                                  if f"advantage_vs_{other}/score_lenient" in g else None),
                                "nll_advantage": _ci_raw(g[f"advantage_vs_{other}/nll"])})
                swap.append({"run": r, "suite": suite, "group": ty,
                             "swap_keeps_original_answer_pct": 100 * g["complementary/keeps_original"]["mean"],
                             "swap_follows_partner_answer_pct": 100 * g["complementary/follows_partner"]["mean"],
                             "swap_other_answer_pct": 100 * (1 - g["complementary/keeps_original"]["mean"]
                                                             - g["complementary/follows_partner"]["mean"]),
                             "swap_identical_pair_answers_pct": 100 * g["complementary/identical"]["mean"],
                             "mismatched_keeps_original_pct": 100 * g["mismatched/keeps_original"]["mean"],
                             "no_image_keeps_original_pct": 100 * g["no_image/keeps_original"]["mean"]})
    T = rep / "tables"
    if main:
        save_table(pd.DataFrame(main), T / "final_results",
                   "Final test: both suites under four image conditions (questions and original references "
                   "fixed). 95% CI: 4,000 bootstrap resamples of complete pairs.", ".2f")
        save_table(pd.DataFrame(types), T / "final_results_by_type", "Final test by question type", ".2f")
        save_table(pd.DataFrame(adv), T / "image_advantages",
                   "Correct-image advantage over each control (accuracy and pair accuracy in pp; NLL advantage = "
                   "NLL(control) − NLL(correct), positive = correct image makes the reference more likely); "
                   "paired over the same resampled pairs", ".2f")
        save_table(pd.DataFrame(swap), T / "swap_analysis",
                   "What the answer does when the image changes: original answer kept, partner's answer "
                   "followed, or another answer (complementary swap); original kept under mismatched / no image",
                   ".2f")
    return pd.DataFrame(main) if main else None


def learning_curves(paths, runs, rep, cfg):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    bounds = np.cumsum([s["updates"] for s in cfg.train.stages])[:-1]
    rows = []
    for r in runs:
        d = paths.out / "runs" / r
        if (d / "train_log.csv").exists():
            lg = pd.read_csv(d / "train_log.csv")
            axs[0, 0].plot(lg["update"], lg["loss"].rolling(25, min_periods=1).mean(), label=f"{r} train (25-upd mean)")
        if (d / "dev_metrics.jsonl").exists():
            dv = pd.DataFrame(read_jsonl(d / "dev_metrics.jsonl"))
            if len(dv):
                dv.insert(0, "run", r)
                rows.append(dv)
                axs[0, 1].plot(dv["update"], dv["caption_loss"], marker="o", label=r)
                for suite, ls in [("grounding", "-"), ("vqa", "--")]:
                    if r == runs[0]:  # main run only, to keep the panel readable
                        for c, col in [("correct", "C0"), ("complementary", "C1"), ("no_image", "C2")]:
                            axs[1, 0].plot(dv["update"], dv[f"{suite}_nll_{c}"], ls=ls, color=col, marker=".",
                                           label=f"{suite} {c}")
                    axs[1, 1].plot(dv["update"], dv[f"{suite}_delta_nll"], ls=ls, marker="o", label=f"{r} {suite}")
    for a in axs.flat:
        for b in bounds:
            a.axvline(b, color="grey", lw=0.8, ls=":")
        a.set_xlabel("update")
    axs[0, 0].set_title("Training loss (stage boundaries dotted)")
    axs[0, 1].set_title("Dev caption reference loss")
    axs[1, 0].set_title(f"Dev answer NLL, {runs[0] if runs else ''} (— grounding, -- VQAv2)")
    axs[1, 1].set_title("Dev ΔNLL = NLL(partner img) − NLL(correct img)")
    axs[1, 1].axhline(0, color="k", lw=0.6)
    for a in axs.flat:
        a.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(rep / "figures" / "learning_curves.png", dpi=130)
    plt.close(fig)
    if rows:
        dv = pd.concat(rows, ignore_index=True)
        save_table(dv, rep / "tables" / "dev_metrics", "Development metrics during training", ".4f")
    # mixture counts per stage
    mix = []
    for r in runs:
        f = paths.out / "runs" / r / "train_log.csv"
        if f.exists():
            lg = pd.read_csv(f)
            g = lg.groupby("stage")[[c for c in lg.columns if c.startswith("n_") and c != "n_target_tokens"]].sum()
            g.insert(0, "run", r)
            mix.append(g.reset_index())
    if mix:
        save_table(pd.concat(mix), rep / "tables" / "mixture_counts",
                   "Training examples drawn per component and stage (each draw = 2 examples)", ".0f")


def final_figure(df, rep):
    if df is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    for a, suite in zip(axs, ["grounding", "vqa"]):
        s = df[df.suite == suite]
        runs = list(dict.fromkeys(s["run"]))
        x = np.arange(len(CONDITIONS))
        w = 0.8 / max(len(runs), 1)
        for j, r in enumerate(runs):
            g = s[s.run == r].set_index("condition").reindex(CONDITIONS)
            a.bar(x + (j - (len(runs) - 1) / 2) * w, g["accuracy_pct"], w, label=f"{r} acc")
            a.scatter(x + (j - (len(runs) - 1) / 2) * w, g["pair_accuracy_pct"], color="k", s=12, zorder=3,
                      label="pair acc" if j == 0 else None)
        a.set_xticks(x)
        a.set_xticklabels(CONDITIONS, fontsize=8)
        a.set_ylabel("%")
        a.set_title(f"{suite}: accuracy by image condition")
        a.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(rep / "figures" / "final_by_condition.png", dpi=130)
    plt.close(fig)


def qualitative(paths, runs, rep):
    md = ["# Complementary-pair examples", "", __doc__.split("Qualitative selection rule")[1].split('"""')[0].strip(), ""]
    for r in runs:
        f = paths.out / "runs" / r / "final_eval" / "samples.jsonl"
        if not f.exists():
            continue
        S = pd.DataFrame(read_jsonl(f))
        picks = []
        for (suite, ty), g in S.groupby(["suite", "qtype"]):
            pids = sorted(g["pair_id"].unique())

            def rows(pid, c):
                return g[(g.pair_id == pid) & (g.condition == c)].sort_values("member")

            cats = {"both correct": lambda pid: rows(pid, "correct")["match_canonical"].all(),
                    "image-insensitive": lambda pid: rows(pid, "correct")["prediction_norm"].nunique() == 1,
                    "image-sensitive error": lambda pid: (rows(pid, "correct")["prediction_norm"].nunique() == 2
                                                          and not rows(pid, "correct")["match_canonical"].all())}
            used = set()
            for cat, fn in cats.items():
                hit = next((pid for pid in pids if pid not in used and fn(pid)), None)
                if hit:
                    used.add(hit)
                    picks.append((suite, ty, cat, hit))
        if not picks:
            continue
        md += [f"## {r}", ""]
        n = len(picks)
        fig, axs = plt.subplots(n, 2, figsize=(9, 2.6 * n))
        axs = np.atleast_2d(axs)
        for i, (suite, ty, cat, pid) in enumerate(picks):
            g = S[S.pair_id == pid]
            q = g["question"].iloc[0]
            md += [f"### {suite} / {ty} / {cat}: `{pid}`", f"**Q:** {q}", "",
                   "| image | references | correct | complementary | mismatched | no image |", "|---|---|---|---|---|---|"]
            for m in range(2):
                gm = g[g.member == m].set_index("condition")
                im = gm.loc["correct", "image"]
                refs = gm.loc["correct", "canonical"]
                preds = {c: gm.loc[c, "prediction"] for c in CONDITIONS}
                md.append(f"| {im} | {refs} | " + " | ".join(f"`{preds[c]}`" for c in CONDITIONS) + " |")
                th = paths.thumbs / (im.replace("/", "_") + ".jpg")
                ax = axs[i, m]
                if th.exists():
                    ax.imshow(plt.imread(th))
                ax.set_axis_off()
                ax.set_title(textwrap.fill(f"[{suite}/{ty}: {cat}] {q}" if m == 0 else "", 60), fontsize=7, loc="left")
                ax.text(0, -0.02, textwrap.fill(f"ref: {refs} | correct img → {preds['correct']!r} | "
                                                f"partner img → {preds['complementary']!r} | no img → "
                                                f"{preds['no_image']!r}", 60),
                        transform=ax.transAxes, fontsize=6.5, va="top")
            md.append("")
        fig.tight_layout()
        fig.savefig(rep / "figures" / f"qualitative_pairs_{r}.png", dpi=110)
        plt.close(fig)
    (rep / "qualitative").mkdir(exist_ok=True)
    (rep / "qualitative" / "complementary_pairs.md").write_text("\n".join(md) + "\n")


def resources(paths, runs, rep):
    rows = []
    fm = paths.features / "meta.json"
    if fm.exists():
        m = read_json(fm)
        rows.append({"item": "feature caching (SigLIP, separate from training)", "run": "shared",
                     "time_min": m.get("cache_time_s", 0) / 60, "peak_gpu_gb": m.get("peak_gpu_mem_gb"),
                     "trainable_params": 0, "note": f"{m['n_images']} images, {m.get('images_per_s', 0):.1f} img/s, "
                                                    f"{m['bytes'] / 2**30:.2f} GiB FP16"
                                                    + (f"; copied from {m['copied_from']}" if m.get("copied_from") else "")})
    pr = paths.data_out / "prepare_resources.json"
    if pr.exists():
        p = read_json(pr)
        if "images" in p:
            rows.append({"item": "image download", "run": "shared", "time_min": p["images"]["fetch_s"] / 60,
                         "note": f"{p['images']['downloaded']} downloaded"})
    for r in runs:
        f = paths.out / "runs" / r / "train_summary.json"
        if not f.exists():
            continue
        s = read_json(f)
        for name, st in s["stages"].items():
            rows.append({"item": f"train {name}", "run": r, "time_min": st["train_time_s"] / 60,
                         "peak_gpu_gb": st["peak_mem_gb"], "trainable_params": st["trainable"]["total"],
                         "note": f"connector {st['trainable']['connector']:,}, LoRA {st['trainable']['lora']:,}, "
                                 f"{st['updates']} updates"})
        e = paths.out / "runs" / r / "final_eval" / "metrics.json"
        if e.exists():
            m = read_json(e)
            rows.append({"item": "final evaluation (generation + NLL, 4 conditions)", "run": r,
                         "time_min": m["eval_time_s"] / 60, "peak_gpu_gb": m.get("peak_mem_gb"),
                         "note": "gen ms/answer: " + ", ".join(
                             f"{su} {c} {1000 * m[su]['gen_time_per_answer_s'][c]:.0f}" for su in ["grounding", "vqa"]
                             for c in CONDITIONS)})
        dv = paths.out / "runs" / r / "dev_metrics.jsonl"
        if dv.exists():
            d = read_jsonl(dv)
            rows.append({"item": "development evaluations", "run": r,
                         "time_min": sum(x.get("dev_eval_s", 0) for x in d) / 60, "note": f"{len(d)} evaluations"})
    if rows:
        df = pd.DataFrame(rows)
        if "trainable_params" in df:
            df["trainable_params"] = df["trainable_params"].astype("Int64")
        save_table(df, rep / "tables" / "resources",
                   "Resources (training time = sum of update times; feature caching reported separately)", ".2f")


def write_index(paths, rep, runs):
    files = sorted(str(p.relative_to(paths.out)) for p in paths.out.rglob("*") if p.is_file())
    sc = read_json(paths.data_out / "split_checks.json") if (paths.data_out / "split_checks.json").exists() else {}
    txt = f"""# Task 2 evidence bundle

Runs: {', '.join(runs) or '-'}
Data selection sha256: {sc.get('selection_sha256', '?')}

| Manual requirement | Files |
|---|---|
| Learning curves (train loss, dev caption loss, answer NLL, ΔNLL, stage boundaries) | report/figures/learning_curves.png, report/tables/dev_metrics.*, runs/*/train_log.csv, runs/*/dev_metrics.jsonl |
| Final results, 4 image conditions, question types, pair diagnostics, NLL, advantages, pair-bootstrap CIs | report/tables/final_results.*, final_results_by_type.*, image_advantages.*, swap_analysis.*, figures/final_by_condition.png, runs/*/final_eval/metrics.json |
| Raw references, generations, per-example metrics | runs/*/final_eval/samples.jsonl |
| Qualitative complementary pairs (selection rule stated) | report/qualitative/complementary_pairs.md, report/figures/qualitative_pairs_*.png, thumbnails/ |
| Resources (train/generation time, peak memory, trainable params by stage, feature caching) | report/tables/resources.*, runs/*/train_summary.json, data/prepare_resources.json |
| Mixture counts | report/tables/mixture_counts.*, runs/*/train_log.csv (n_*) |
| Interface checks: feature shapes, LN once, frozen sets, connector gradients, split isolation, prompt structure | runs/*/sanity_checks.json, data/split_checks.json |
| Connector scale calibration (rule + value) | runs/*/calibration.json |
| Selected image / question / pair IDs | data/*.jsonl, data/mismatch_test.json, data/images_needed.json |
| Configs + environment | runs/*/config.yaml, env/*.json |

Features, checkpoints and stage weights are kept in the Kaggle output (task2_artifacts/) and are
not part of this zip.

## All files
""" + "\n".join(f"- {f}" for f in files) + "\n"
    (rep / "README.md").write_text(txt)


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--zip", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    paths = Paths(cfg)
    rep = paths.report_dir
    shutil.rmtree(rep, ignore_errors=True)
    (rep / "tables").mkdir(parents=True)
    (rep / "figures").mkdir()
    runs = [r for r in RUN_ORDER if (paths.out / "runs" / r).exists()] + sorted(
        p.name for p in (paths.out / "runs").glob("*") if p.is_dir() and p.name not in RUN_ORDER) \
        if (paths.out / "runs").exists() else []
    log(f"report: runs {runs}")
    sc = paths.data_out / "split_checks.json"
    if sc.exists():
        s = read_json(sc)
        save_table(pd.DataFrame([{"set": k, "n": v} for k, v in s["checks"]["pairs"].items()]),
                   rep / "tables" / "data_sizes", "Selected data (pairs / examples)", ".0f")
    learning_curves(paths, runs, rep, cfg)
    df = final_tables(paths, runs, rep)
    final_figure(df, rep)
    qualitative(paths, runs, rep)
    resources(paths, runs, rep)
    write_index(paths, rep, runs)
    zpath = Path(args.zip) if args.zip else paths.out.parent / "task2_report.zip"
    skip = {".pt", ".bin", ".safetensors", ".npy", ".npz", ".zip"}
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths.out.rglob("*")):
            if p.is_file() and p.suffix not in skip and p.stat().st_size <= 50 * 2**20:
                z.write(p, Path(paths.out.name) / p.relative_to(paths.out))
    log(f"wrote {zpath} ({zpath.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
