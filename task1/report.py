"""Assemble the Task 1 evidence folder (tables, figures, qualitative sheets) and zip it.

    python -m task1.report --config task1/configs/full_ft.yaml --zip /kaggle/working/task1_full_ft_report.zip

Works on whatever runs / evaluation conditions exist under paths.out_root, so the same script
builds the per-notebook report and, later, the combined report over all five runs (point
T1_OUT_ROOT at a folder holding the merged outputs).
"""

from __future__ import annotations

import json
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .common import Paths, base_argparser, load_config, log, read_json, read_jsonl

KNOWN_ORDER = ["baseline", "full_ft_mid", "full_ft_final", "l2sp_final", "lora4_final",
               "lora16_final", "oplora_final"]
RUN_ORDER = ["full_ft", "l2sp", "lora4", "lora16", "oplora"]
# (label, task file, per-sample field, metrics.json key)
PRIMARY = [("GSM8K flex EM", "gsm8k_cot", "correct_flex", "gsm8k_flex"),
           ("HellaSwag acc_norm", "hellaswag", "acc_norm", "hellaswag_acc_norm"),
           ("BoolQ acc", "boolq", "acc", "boolq_acc"),
           ("ARC-Easy acc_norm", "arc_easy", "acc_norm", "arc_easy_acc_norm")]
RAW = [("GSM8K strict EM", "gsm8k_cot", "correct_strict", "gsm8k_strict"),
       ("HellaSwag acc", "hellaswag", "acc", "hellaswag_acc"),
       ("ARC-Easy acc", "arc_easy", "acc", "arc_easy_acc")]


def order(items, known):
    return [x for x in known if x in items] + sorted(x for x in items if x not in known)


def _md(df: pd.DataFrame, floatfmt=".2f") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            small = isinstance(v, (float, np.floating)) and 0 < abs(v) < 0.01  # e.g. lr, λ: keep visible
            cells.append(f"{v:{'.3g' if small else floatfmt}}" if isinstance(v, (float, np.floating)) and not
                         (isinstance(v, float) and math.isnan(v)) else ("" if v is None or
                         (isinstance(v, float) and math.isnan(v)) else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def save_table(df: pd.DataFrame, path: Path, title: str = "", floatfmt=".2f"):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".md").write_text((f"### {title}\n\n" if title else "") + _md(df, floatfmt))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int, seed: int):
    """95% CI (percentage points) of mean(a) - mean(b) over paired examples."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = (a[idx].mean(1) - b[idx].mean(1)) * 100
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ----------------------------------------------------------------------------- sections
def benchmark_tables(paths: Paths, conds, rep: Path, cfg):
    ev = {c: read_json(paths.eval_dir(c) / "metrics.json") for c in conds}
    samples = {}

    def per_example(c, task, field):
        key = (c, task)
        if key not in samples:
            f = paths.eval_dir(c) / f"samples_{task}.jsonl"
            samples[key] = {r["doc_id"]: r for r in read_jsonl(f)} if f.exists() else {}
        return samples[key]

    rows, long_rows = [], []
    for c in conds:
        row = {"condition": c}
        for label, task, field, key in PRIMARY + RAW:
            v = ev[c].get(key)
            row[label] = None if v is None else 100 * v
            se = ev[c].get(key + "_se")
            if "baseline" in ev and c != "baseline" and v is not None and \
                    ev["baseline"].get(key) is not None:
                d = 100 * (v - ev["baseline"][key])
                row[f"Δ {label} (pp)"] = d
                cur, bas = per_example(c, task, field), per_example("baseline", task, field)
                ids = sorted(set(cur) & set(bas))
                if ids:
                    a = np.array([cur[i][field] for i in ids], dtype=float)
                    b = np.array([bas[i][field] for i in ids], dtype=float)
                    lo, hi = paired_bootstrap(a, b, int(cfg.analysis.bootstrap_resamples), int(cfg.seed))
                else:
                    lo = hi = None
                long_rows.append({"condition": c, "metric": label, "score_pct": 100 * v,
                                  "stderr_pct": None if se is None else 100 * se,
                                  "delta_pp": d, "delta_ci95_lo": lo, "delta_ci95_hi": hi,
                                  "primary": (label, task, field, key) in PRIMARY})
            else:
                long_rows.append({"condition": c, "metric": label, "score_pct": None if v is None else 100 * v,
                                  "stderr_pct": None if se is None else 100 * se, "delta_pp": None,
                                  "delta_ci95_lo": None, "delta_ci95_hi": None,
                                  "primary": (label, task, field, key) in PRIMARY})
        rows.append(row)
    wide = pd.DataFrame(rows)
    prim_cols = ["condition"] + [p[0] for p in PRIMARY] + [f"Δ {p[0]} (pp)" for p in PRIMARY
                                                            if f"Δ {p[0]} (pp)" in wide]
    save_table(wide[prim_cols], rep / "tables" / "benchmarks_primary",
               "Primary scores (%) and change from baseline (percentage points)")
    save_table(wide, rep / "tables" / "benchmarks_all", "All scores (%) incl. raw accuracy")
    longdf = pd.DataFrame(long_rows)
    save_table(longdf, rep / "tables" / "benchmarks_long_with_ci",
               "Scores with lm-eval stderr and paired-bootstrap 95% CI of the change vs baseline")

    # GSM8K behaviour: style / extraction / truncation, separated from correctness.
    beh = []
    base = per_example("baseline", "gsm8k_cot", "correct_flex") if "baseline" in conds else {}
    for c in conds:
        rs = per_example(c, "gsm8k_cot", "correct_flex")
        if not rs:
            continue
        v = list(rs.values())
        toks = np.array([r["n_gen_tokens"] for r in v])
        row = {"condition": c, "n": len(v),
               "flex_acc_pct": 100 * np.mean([r["correct_flex"] for r in v]),
               "strict_acc_pct": 100 * np.mean([r["correct_strict"] for r in v]),
               "truncated_pct": 100 * np.mean([r["truncated"] for r in v]),
               "flex_extract_fail_pct": 100 * np.mean([r["flex_extract_failed"] for r in v]),
               "strict_extract_fail_pct": 100 * np.mean([r["strict_extract_failed"] for r in v]),
               "gen_tokens_mean": toks.mean(), "gen_tokens_median": float(np.median(toks)),
               "gen_tokens_p95": float(np.percentile(toks, 95))}
        if base and c != "baseline":
            ids = sorted(set(rs) & set(base))
            gained = [i for i in ids if rs[i]["correct_flex"] and not base[i]["correct_flex"]]
            lost = [i for i in ids if not rs[i]["correct_flex"] and base[i]["correct_flex"]]
            row |= {"both_correct": sum(rs[i]["correct_flex"] and base[i]["correct_flex"] for i in ids),
                    "gained_vs_baseline": len(gained), "lost_vs_baseline": len(lost),
                    "both_wrong": sum((not rs[i]["correct_flex"]) and (not base[i]["correct_flex"]) for i in ids),
                    "lost_truncated": sum(rs[i]["truncated"] for i in lost),
                    "lost_extract_failed": sum(rs[i]["flex_extract_failed"] for i in lost)}
        beh.append(row)
    if beh:
        save_table(pd.DataFrame(beh), rep / "tables" / "gsm8k_behavior",
                   "GSM8K answer style: truncation, extraction and paired transitions vs baseline")
    return longdf, samples


def training_tables(paths: Paths, runs, rep: Path, cfg):
    res, disp_rows, logs = [], [], {}
    ref_lam = float(cfg.analysis.get("reference_lambda", 1e-3))
    for r in runs:
        d = paths.out / "runs" / r
        if (d / "train_log.csv").exists():
            logs[r] = pd.read_csv(d / "train_log.csv")
        if (d / "train_summary.json").exists():
            s = read_json(d / "train_summary.json")
            lg = logs.get(r)
            tok_s = (lg["seq_tokens"].sum() / s["train_time_s"]) if lg is not None and s["train_time_s"] else None
            ck = [v for v in (s.get("recovery_checkpoint_bytes") or {}).values() if v]
            res.append({"run": r, "trainable_params": s["trainable_params"],
                        "trainable_pct": 100 * s["trainable_params"] / s["total_params"],
                        "train_time_h": s["train_time_s"] / 3600,
                        "seq_tokens_per_s": tok_s,
                        "peak_mem_alloc_gb": s.get("peak_mem_allocated_gb"),
                        "peak_mem_reserved_gb": s.get("peak_mem_reserved_gb"),
                        "recovery_ckpt_gib": (max(ck) / 2**30) if ck else None,
                        "inference_export_gib": (s["export_bytes"].get("final", 0) / 2**30)
                        if s.get("export_bytes") else None,
                        "skipped_updates": s.get("skipped_updates"), "gpu": s.get("gpu"),
                        "lr": s.get("lr"), "lambda": s.get("l2sp_lambda")}
                       | {k: v for k, v in s.items() if k.startswith("overhead_")})
        lam_applied = float(read_json(d / "train_summary.json").get("l2sp_lambda") or 0.0) \
            if (d / "train_summary.json").exists() else 0.0
        for st in ["mid", "final"]:
            f = d / f"displacement_{st}.json"
            if f.exists():
                j = read_json(f)
                omega = 0.5 * j["sq_total"]
                lam = lam_applied if lam_applied > 0 else ref_lam
                row = {"run": r, "stage": st, "update": j["update"],
                       "theta_minus_theta0_l2": j["l2"], "omega": omega,
                       "lambda": lam, "lambda_applied": lam_applied > 0,
                       "lambda_omega": lam * omega}
                lg = logs.get(r)
                if lg is not None and len(lg):
                    w = lg[lg["update"] <= j["update"]].tail(25)
                    row["sft_loss_last25"] = float(w["sft_loss"].mean())
                    row["lambda_omega_over_sft"] = lam * omega / row["sft_loss_last25"]
                disp_rows.append(row)
    if res:
        save_table(pd.DataFrame(res), rep / "tables" / "resources",
                   "Training resources (train time excludes checkpointing/eval)", ".3f")
    if disp_rows:
        save_table(pd.DataFrame(disp_rows), rep / "tables" / "displacement",
                   "Full-model displacement ||θ−θ0||₂ and the L2-SP penalty Ω = ½||θ−θ0||². For runs "
                   "with lambda_applied = False, λΩ uses the reference λ and is NOT part of their "
                   "objective (shown only to compare penalty strength).", ".5g")
    # L2-SP objective over training (runs that logged Omega).
    reg = []
    for r, lg in logs.items():
        if "omega" in lg and lg["omega"].notna().any():
            for frac in [0.25, 0.5, 0.75, 1.0]:
                u = max(1, int(round(frac * lg["update"].max())))
                w = lg[(lg["update"] <= u)].tail(25)
                reg.append({"run": r, "update": u, "sft_loss": w["sft_loss"].mean(),
                            "omega": w["omega"].iloc[-1], "lambda_omega": w["lambda_omega"].iloc[-1],
                            "total_loss": w["total_loss"].mean(),
                            "lambda_omega_share_of_total": (w["lambda_omega"] / w["total_loss"]).mean(),
                            "sft_grad_norm": w["sft_grad_norm"].mean(),
                            "reg_grad_norm": w["reg_grad_norm"].mean()})
    adapter_tables(paths, runs, rep)
    if reg:
        save_table(pd.DataFrame(reg), rep / "tables" / "l2sp_objective",
                   "L2-SP objective components (25-update means ending at the given update; Ω is "
                   "evaluated at the parameters that update's SFT loss used, i.e. before its step; "
                   "Ω after the last step is in the displacement table)", ".5g")
    return logs


def adapter_tables(paths: Paths, runs, rep: Path):
    """LoRA / OPLoRA: adapter size and overhead, merged-vs-unmerged logit check, and the share of
    each update in the protected singular subspaces of W0 over all adapted projections."""
    rows, sub = [], []
    for r in runs:
        d = paths.out / "runs" / r
        if not (d / "merge_check_final.json").exists():
            continue
        s = read_json(d / "train_summary.json") if (d / "train_summary.json").exists() else {}
        cfg_r = {}
        if (d / "config.yaml").exists():
            import yaml
            cfg_r = (yaml.safe_load((d / "config.yaml").read_text()) or {}).get("adapter", {})
        mc = read_json(d / "merge_check_final.json")
        ss = read_json(d / "adapter_subspace_final.json") if (d / "adapter_subspace_final.json").exists() else {}
        rows.append({"run": r, "r": cfg_r.get("r"), "alpha": cfg_r.get("alpha"),
                     "protect_rank": cfg_r.get("protect_rank"),
                     "adapter_params": s.get("overhead_adapter_params"),
                     "adapter_mib_fp32": (s.get("overhead_adapter_bytes_fp32") or 0) / 2**20,
                     "basis_mib": (s.get("overhead_basis_bytes") or 0) / 2**20 or None,
                     "basis_preprocess_s": s.get("overhead_basis_preprocess_s"),
                     "export_merge_s": s.get("overhead_export_merge_s"),
                     "merge_mean_abs_dlogit": mc["merged_vs_unmerged"]["mean_abs_diff"],
                     "merge_max_abs_dlogit": mc["merged_vs_unmerged"]["max_abs_diff"],
                     "merge_top1_agree": mc["merged_vs_unmerged"]["top1_agreement"],
                     "adapter_effect_mean_abs_dlogit": mc["adapter_effect_unmerged_vs_base"]["mean_abs_diff"],
                     "rho_out_mean_all": ss.get("rho_out_mean"), "rho_out_max_all": ss.get("rho_out_max"),
                     "rho_in_mean_all": ss.get("rho_in_mean"), "rho_in_max_all": ss.get("rho_in_max")})
        f = d / "adapter_subspace_final.csv"
        if f.exists():
            sub.append(pd.read_csv(f).assign(run=r))
    if rows:
        save_table(pd.DataFrame(rows), rep / "tables" / "adapters",
                   "Adapter runs: size, OPLoRA basis/export overhead, merged (FP16 export) vs. "
                   "unmerged logits on 4 training sequences (256 tokens) (adapter effect = unmerged vs. starting "
                   "model, for scale), and the fraction of ||ΔW||² in W0's top-k output/input "
                   "singular subspaces over all adapted projections (OPLoRA constraint check)", ".4g")
    if sub:
        df = pd.concat(sub, ignore_index=True)
        df.to_csv(rep / "tables" / "adapter_subspace_per_projection.csv", index=False)
        bm = df.groupby(["run", "module"])[["rho_out", "rho_in", "eta"]].mean().reset_index()
        bm["rho_out_chance"] = df.groupby(["run", "module"])["rho_out_chance"].first().values
        save_table(bm, rep / "tables" / "adapter_subspace_by_module",
                   "Mean over layers of ρ_out = ||U_kᵀΔW||²/||ΔW||², ρ_in = ||ΔW V_k||²/||ΔW||² "
                   "and η = ||ΔW||/||W0|| per projection type", ".4g")


def diagnostics_tables(paths: Paths, rep: Path):
    out = {}
    for kind in ["weight_space", "activation_summary", "activation"]:
        fs = sorted(paths.diag_out.glob(f"{kind}_*.csv"))
        if kind == "activation":
            fs = [f for f in fs if not f.name.startswith("activation_summary")]
        if fs:
            df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
            out[kind] = df
            if kind != "activation":
                save_table(df, rep / "tables" / f"diagnostics_{kind}", kind.replace("_", " "), ".5f")
    if "activation_summary" in out:
        s = out["activation_summary"]
        p = s.pivot_table(index=["condition", "layer"], columns="group", values="R_mean").reset_index()
        if {"math", "nonmath"} <= set(p.columns):
            p["math_over_nonmath"] = p["math"] / p["nonmath"]
        save_table(p, rep / "tables" / "diagnostics_R_math_vs_nonmath",
                   "Mean R(z) on math vs non-math probe prompts", ".5f")
    return out


def qualitative_outputs(paths: Paths, conds, rep: Path, cfg):
    rows = []
    for c in conds:
        f = paths.eval_dir(c) / "qualitative.jsonl"
        if f.exists():
            rows += read_jsonl(f)
    if not rows:
        return
    df = pd.DataFrame(rows)
    q = rep / "qualitative"
    q.mkdir(parents=True, exist_ok=True)
    df.to_csv(q / "all_responses.csv", index=False)
    fmt_rank = {"plain": 0, "json": 1, "template": 2}
    lines = ["# Qualitative responses grouped by question and format\n",
             "Greedy decoding, thinking disabled, 512-new-token cap. `[TRUNCATED]` marks responses "
             "that hit the cap.\n"]
    for (po, qid), g in sorted(df.groupby(["plain_order", "question"]), key=lambda x: x[0][0]):
        lines.append(f"\n## {po + 1}. `{qid}`\n")
        for fmt, gf in sorted(g.groupby("format"), key=lambda x: fmt_rank[x[0]]):
            first = gf.iloc[0]
            lines.append(f"\n### Format: {fmt}\n")
            if isinstance(first["system"], str) and first["system"]:
                lines.append(f"**System:** {first['system']}\n")
            lines.append("**Prompt:**\n\n```\n" + first["prompt"] + "\n```\n")
            for c in order(set(gf["condition"]), KNOWN_ORDER):
                r = gf[gf["condition"] == c].iloc[0]
                tag = " [TRUNCATED]" if r["truncated"] else ""
                lines.append(f"\n**{c}** ({r['n_tokens']} tokens){tag}\n\n```\n{r['response']}\n```\n")
    (q / "responses_by_question.md").write_text("\n".join(lines))

    # Annotation sheet: first and last plain questions x 3 formats x all conditions.
    ann_ids = list(cfg.qualitative.annotate_ids)
    sub = df[df["question"].isin(ann_ids)].copy()
    sub["_c"] = sub["condition"].map(lambda c: order(set(df["condition"]), KNOWN_ORDER).index(c))
    sub["_f"] = sub["format"].map(fmt_rank)
    sub = sub.sort_values(["plain_order", "_f", "_c"])
    heur = [c for c in ["json_valid", "template_keys", "template_keys_filled",
                        "distinct_4gram_ratio", "max_line_repeats", "has_boxed",
                        "n_latex_markers", "n_arith_exprs"] if c in sub]
    sheet = sub[["question", "format", "condition", "n_tokens", "truncated", *heur, "response"]].copy()
    for col in ["ann_format_compliance", "ann_coherence_relevance", "ann_repetition",
                "ann_unnecessary_math_framing", "ann_inappropriate_content", "ann_excerpt", "ann_notes"]:
        sheet[col] = ""
    # manual annotations made with `python -m task1.annotate` (kept outside report/)
    ann_file = paths.out / "annotations" / "qualitative_annotations.csv"
    if ann_file.exists():
        from . import annotate as A
        ann = pd.read_csv(ann_file, dtype=str, keep_default_na=False)
        cmap = {"format_compliance": "ann_format_compliance", "coherence_relevance": "ann_coherence_relevance",
                "repetition": "ann_repetition", "math_framing": "ann_unnecessary_math_framing",
                "inappropriate": "ann_inappropriate_content", "excerpt": "ann_excerpt", "notes": "ann_notes"}
        idx = {(r.question, r.format, r.condition): r for r in ann.itertuples(index=False)}
        for i, r in sheet.iterrows():
            a = idx.get((r["question"], r["format"], r["condition"]))
            if a is not None:
                for src, dst in cmap.items():
                    sheet.at[i, dst] = getattr(a, src)
        items, aconds = A.load_items(paths.out, ann_ids) if (q / "all_responses.csv").exists() else ([], [])
        if items:
            A.USE_COLOR = False
            A.write_summary(paths.out, items, A.load_ann(ann_file), aconds)
        log(f"merged {len(ann)} manual annotations from {ann_file}")
    sheet.to_csv(q / "annotation_sheet.csv", index=False)


# ----------------------------------------------------------------------------- figures
def figures(paths: Paths, logs: dict, longdf, diag: dict, samples, conds, rep: Path, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = rep / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    mid = int(cfg.train.midpoint_update)

    if logs:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        for r in order(set(logs), RUN_ORDER):
            lg = logs[r]
            line, = ax.plot(lg["update"], lg["sft_loss"], alpha=0.18, lw=0.6)
            ax.plot(lg["update"], lg["sft_loss"].rolling(25, min_periods=1).mean(),
                    color=line.get_color(), lw=1.4, label=f"{r} (25-update mean)")
        ax.axvline(mid, ls="--", lw=0.8, color="grey")
        ax.text(mid, ax.get_ylim()[1], " midpoint", va="top", fontsize=7, color="grey")
        ax.set_xlabel("optimizer update"); ax.set_ylabel("SFT loss (token CE)")
        ax.set_title("Training SFT loss (penalty terms excluded)")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(fig_dir / "training_loss.png"); plt.close(fig)

        fig, axs = plt.subplots(1, 3, figsize=(10, 2.8))
        for r in order(set(logs), RUN_ORDER):
            lg = logs[r]
            axs[0].plot(lg["update"], lg["lr"], label=r)
            axs[1].plot(lg["update"], lg["grad_norm"].rolling(10, min_periods=1).mean(), label=r)
            axs[2].plot(lg["update"], lg["loss_scale"], label=r)
            sk = lg[lg["skipped"] == 1]
            axs[2].scatter(sk["update"], sk["loss_scale"], s=8, marker="x")
        axs[0].set_title("learning rate"); axs[1].set_title("grad norm (pre-clip, 10-upd mean)")
        axs[2].set_title("loss scale (x = skipped update)"); axs[2].set_yscale("log", base=2)
        for a in axs:
            a.set_xlabel("update")
        axs[0].legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(fig_dir / "training_lr_gradnorm_scale.png"); plt.close(fig)

        reg_runs = [r for r in order(set(logs), RUN_ORDER)
                    if "omega" in logs[r] and logs[r]["omega"].notna().any()]
        if reg_runs:
            ref_lam = float(cfg.analysis.get("reference_lambda", 1e-3))
            fig, axs = plt.subplots(1, 3, figsize=(11, 3))
            for r in reg_runs:
                lg = logs[r]
                axs[0].plot(lg["update"], lg["omega"], label=f"{r}: Ω (applied)")
                axs[1].plot(lg["update"], lg["sft_loss"].rolling(25, min_periods=1).mean(),
                            label=f"{r}: SFT loss")
                axs[1].plot(lg["update"], lg["lambda_omega"], label=f"{r}: λΩ")
                axs[1].plot(lg["update"], lg["total_loss"].rolling(25, min_periods=1).mean(),
                            ls="--", label=f"{r}: SFT + λΩ")
                axs[2].plot(lg["update"], lg["sft_grad_norm"].rolling(10, min_periods=1).mean(),
                            label=f"{r}: ||∇ SFT||")
                axs[2].plot(lg["update"], lg["reg_grad_norm"], label=f"{r}: ||λ(θ−θ0)||")
            for f in sorted((paths.out / "runs").glob("*/displacement_*.json")):
                run = f.parent.name
                if run in reg_runs:
                    continue
                j = read_json(f)
                axs[0].scatter([j["update"]], [0.5 * j["sq_total"]], marker="o", zorder=3,
                               label=f"{run} {f.stem.split('_', 1)[1]}: ½||θ−θ0||² (not penalised)")
            axs[0].set_title("Ω = ½||θ−θ0||²"); axs[0].set_yscale("log")
            axs[1].set_title("objective components (25-upd mean)"); axs[1].set_yscale("log")
            axs[2].set_title("gradient norms before clipping"); axs[2].set_yscale("log")
            for a in axs:
                a.set_xlabel("update"); a.legend(fontsize=6, frameon=False)
            fig.suptitle(f"L2-SP penalty (λ = {ref_lam:g})", fontsize=9)
            fig.tight_layout(); fig.savefig(fig_dir / "l2sp_penalty.png"); plt.close(fig)

    if longdf is not None and len(longdf) and longdf["delta_pp"].notna().any():
        d = longdf[longdf["primary"] & longdf["delta_pp"].notna()]
        cs = order(set(d["condition"]), KNOWN_ORDER)
        metrics = [p[0] for p in PRIMARY]
        x = np.arange(len(metrics)); w = 0.8 / max(len(cs), 1)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        for j, c in enumerate(cs):
            dc = d[d["condition"] == c].set_index("metric").reindex(metrics)
            err = np.array([dc["delta_pp"] - dc["delta_ci95_lo"], dc["delta_ci95_hi"] - dc["delta_pp"]])
            ax.bar(x + (j - (len(cs) - 1) / 2) * w, dc["delta_pp"], w, yerr=err, capsize=2, label=c)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(x); ax.set_xticklabels(metrics)
        ax.set_ylabel("change vs baseline (pp)")
        ax.set_title("Benchmark change from baseline (paired-bootstrap 95% CI)")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(fig_dir / "benchmark_delta.png"); plt.close(fig)

    # GSM8K generated-length distribution.
    gs = {c: [r["n_gen_tokens"] for r in samples.get((c, "gsm8k_cot"), {}).values()] for c in conds}
    gs = {c: v for c, v in gs.items() if v}
    if gs:
        fig, ax = plt.subplots(figsize=(6, 3))
        bins = np.linspace(0, int(cfg.eval.gsm8k_max_gen_toks), 41)
        for c in order(set(gs), KNOWN_ORDER):
            ax.hist(gs[c], bins=bins, histtype="step", lw=1.2, label=c)
        ax.set_xlabel("generated tokens (GSM8K)"); ax.set_ylabel("count"); ax.set_yscale("log")
        ax.set_title("GSM8K response length (cap = 1,024)")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(fig_dir / "gsm8k_length_hist.png"); plt.close(fig)

    # Displacement by layer / module.
    recs = []
    for f in sorted((paths.out / "runs").glob("*/displacement_*.json")):
        j = read_json(f)
        run, st = f.parent.name, f.stem.split("_", 1)[1]
        for n, sq in j["per_tensor_sq"].items():
            parts = n.split(".")
            layer = int(parts[2]) if len(parts) > 2 and parts[1] == "layers" else -1
            mod = parts[-2] if layer >= 0 else parts[-2] if len(parts) > 1 else n
            recs.append({"run": run, "stage": st, "layer": layer, "module": mod, "sq": sq})
    if recs:
        dd = pd.DataFrame(recs)
        bl = dd[dd.layer >= 0].groupby(["run", "stage", "layer"])["sq"].sum().reset_index()
        bm = dd.groupby(["run", "stage", "module"])["sq"].sum().reset_index()
        bl["l2"] = np.sqrt(bl["sq"]); bm["l2"] = np.sqrt(bm["sq"])
        bl.drop(columns="sq").to_csv(rep / "tables" / "displacement_by_layer.csv", index=False)
        bm.drop(columns="sq").to_csv(rep / "tables" / "displacement_by_module.csv", index=False)
        fig, axs = plt.subplots(1, 2, figsize=(10, 3))
        for (r, st), g in bl.groupby(["run", "stage"]):
            axs[0].plot(g["layer"], g["l2"], marker=".", label=f"{r}_{st}")
        axs[0].set_xlabel("layer"); axs[0].set_ylabel("||Δθ||₂ within layer")
        axs[0].legend(fontsize=7, frameon=False); axs[0].set_title("Displacement by layer")
        piv = bm.pivot_table(index="module", columns=bm["run"] + "_" + bm["stage"], values="l2")
        piv.plot.barh(ax=axs[1], legend=True)
        axs[1].set_xlabel("||Δθ||₂"); axs[1].set_title("Displacement by module type")
        axs[1].legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(fig_dir / "displacement_by_layer_module.png"); plt.close(fig)

    if "weight_space" in diag and "activation_summary" in diag:
        ws, act = diag["weight_space"], diag["activation_summary"]
        cs = order(set(ws["condition"]), KNOWN_ORDER)
        layers = sorted(ws["layer"].unique())
        fig, axs = plt.subplots(1, 3, figsize=(11, 3))
        x = np.arange(len(layers)); w = 0.8 / max(len(cs), 1)
        for j, c in enumerate(cs):
            g = ws[ws.condition == c].set_index("layer").reindex(layers)
            axs[0].bar(x + (j - (len(cs) - 1) / 2) * w, g["rho_k"], w, label=c)
            axs[1].bar(x + (j - (len(cs) - 1) / 2) * w, g["eta"], w, label=c)
        axs[0].axhline(ws["rho_chance_out"].iloc[0], ls="--", color="grey", lw=0.8)
        axs[0].text(x[-1] + 0.45, ws["rho_chance_out"].iloc[0], "k/d_out", fontsize=7,
                    color="grey", va="bottom", ha="right")
        axs[0].set_title(f"ρ_k (k={int(ws['k'].iloc[0])}) down_proj"); axs[1].set_title("η = ||ΔW||_F/||W0||_F")
        for a in axs[:2]:
            a.set_xticks(x); a.set_xticklabels([f"L{l}" for l in layers])
        axs[0].legend(fontsize=6, frameon=False)
        for c in cs:
            for gname, ls in [("math", "-"), ("nonmath", "--")]:
                g = act[(act.condition == c) & (act.group == gname)].sort_values("layer")
                axs[2].plot(g["layer"], g["R_mean"], ls=ls, marker="o", label=f"{c} {gname}")
        axs[2].set_title("mean R(z): math (—) vs non-math (--)"); axs[2].set_xlabel("layer")
        axs[2].set_yscale("log"); axs[2].legend(fontsize=6, frameon=False)
        fig.tight_layout(); fig.savefig(fig_dir / "diagnostics.png"); plt.close(fig)


# ----------------------------------------------------------------------------- index
def write_index(paths: Paths, rep: Path, conds, runs):
    files = sorted(str(p.relative_to(paths.out)) for p in paths.out.rglob("*")
                   if p.is_file() and not p.name.endswith(".zip"))
    stats = read_json(paths.data_out / "data_stats.json") if (paths.data_out / "data_stats.json").exists() else {}
    txt = f"""# Task 1 evidence bundle

Runs: {', '.join(runs) or '-'}
Evaluation conditions: {', '.join(conds) or '-'}
Training data sha256: {stats.get('data_sha256', '?')}

## Where each required piece of evidence lives
| Manual requirement | Files |
|---|---|
| Benchmark results (all conditions x 4 benchmarks, Δ in pp) | report/tables/benchmarks_primary.*, benchmarks_all.*, benchmarks_long_with_ci.*, figures/benchmark_delta.png |
| GSM8K extraction / truncation vs. correctness | report/tables/gsm8k_behavior.*, figures/gsm8k_length_hist.png, eval/*/samples_gsm8k_cot.jsonl |
| Learning curves (SFT loss, lr) | report/figures/training_loss.png, training_lr_gradnorm_scale.png, runs/*/train_log.csv |
| Regularisation / displacement | report/tables/displacement.* (incl. Ω, λΩ), l2sp_objective.*, displacement_by_layer.csv, displacement_by_module.csv, figures/l2sp_penalty.png, runs/*/displacement_*.json |
| L2-SP logs (Ω, λΩ, total loss) and reference overhead | runs/l2sp/train_log.csv (omega, lambda_omega, total_loss, sft_grad_norm, reg_grad_norm), runs/l2sp/train_summary.json (overhead_*) |
| LoRA / OPLoRA: adapter params, basis storage + preprocessing, export overhead, merged vs. unmerged logits, OPLoRA constraint verification | report/tables/adapters.*, adapter_subspace_by_module.*, adapter_subspace_per_projection.csv, runs/<adapter run>/merge_check_final.json, adapter_subspace_final.json + .csv, train_summary.json (overhead_*) |
| Imported results of earlier notebooks | data/imported_from.json |
| Resources (params, time, memory, checkpoint size) | report/tables/resources.*, runs/*/train_summary.json, eval/*/metrics.json (timings) |
| Weight-space + activation diagnostics | report/tables/diagnostics_*.*, figures/diagnostics.png, diagnostics/*.csv |
| Qualitative (168 responses, 42-response annotation subset) | report/qualitative/responses_by_question.md, all_responses.csv, annotation_sheet.csv |
| Data selection / probe set / prompts | data/data_stats.json, selected_examples.csv, excluded_examples.csv, probe_set.json, qualitative_prompts.json |
| Config + environment | runs/*/config.yaml, env/*.json |

FP32 down_proj deltas used by the diagnostics (task1_artifacts/deltas/<run>/<stage>.pt) and the
FP16 inference exports (task1_artifacts/exports/<run>/<stage>/, for adapter runs including the
exported FP32 factors adapter_factors.pt, projected for OPLoRA) are kept in the Kaggle output and
are deliberately not part of this zip.

The annotation columns (ann_*) in annotation_sheet.csv are filled by hand with
`python -m task1.annotate` (saved in annotations/qualitative_annotations.csv and merged here; blank
if not yet annotated); annotation_summary.md / .csv summarise them per condition with the supporting
excerpts. Heuristic columns (json_valid, distinct_4gram_ratio, ...) are automatic aids.

## All files
""" + "\n".join(f"- {f}" for f in files) + "\n"
    (rep / "README.md").write_text(txt)


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--zip", default=None, help="output zip path (default: <out_root>/../task1_report.zip)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    paths = Paths(cfg)
    rep = paths.report_dir
    shutil.rmtree(rep, ignore_errors=True)
    rep.mkdir(parents=True)

    conds = order({p.name for p in (paths.out / "eval").glob("*") if (p / "metrics.json").exists()},
                  KNOWN_ORDER) if (paths.out / "eval").exists() else []
    runs = order({p.name for p in (paths.out / "runs").glob("*") if p.is_dir()},
                 RUN_ORDER) if (paths.out / "runs").exists() else []
    log(f"report: runs={runs} conditions={conds}")

    longdf, samples = (benchmark_tables(paths, conds, rep, cfg) if conds else (None, {}))
    logs = training_tables(paths, runs, rep, cfg)
    diag = diagnostics_tables(paths, rep)
    qualitative_outputs(paths, conds, rep, cfg)
    figures(paths, logs, longdf, diag, samples, conds, rep, cfg)
    write_index(paths, rep, conds, runs)

    zpath = Path(args.zip) if args.zip else paths.out.parent / "task1_report.zip"
    zpath.parent.mkdir(parents=True, exist_ok=True)
    # Only small evidence files go in the zip: model weights / tensors never do, and any
    # unexpectedly large file is skipped with a warning.
    skip_ext = {".pt", ".bin", ".safetensors", ".npz", ".npy", ".ckpt", ".zip"}
    max_bytes = 50 * 2**20
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths.out.rglob("*")):
            if not p.is_file() or p.resolve() == zpath.resolve():
                continue
            if p.suffix in skip_ext or p.stat().st_size > max_bytes:
                log(f"not zipping {p} ({p.stat().st_size / 2**20:.1f} MiB)")
                continue
            z.write(p, Path(paths.out.name) / p.relative_to(paths.out))
    log(f"wrote {zpath} ({zpath.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
