"""Generate the report's LaTeX tables from the saved result files and copy the figures.

    python report/make_tables.py        (from the repository root)

Every number in report/tables/*.tex comes from one of these sources:
  Task 1  kaggle_outputs/task1_combined/task1/report/tables/*.csv, report/qualitative/*.csv
  Task 2  kaggle_outputs/task2_vlm/output/task2/report/tables/*.csv, qualitative/complementary_pairs.md
  Task 3  task3/results/<dataset>/*.csv|json, kaggle_outputs/task3_live/task3/live/summary.json
Figures are copied into report/figures/.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "tables"
FIG = ROOT / "report" / "figures"
T1 = ROOT / "kaggle_outputs/task1_combined/task1/report"
T2 = ROOT / "kaggle_outputs/task2_vlm/output/task2/report"
T3 = ROOT / "task3/results"
LIVE = ROOT / "kaggle_outputs/task3_live/task3/live"

COND = {"baseline": "Baseline", "full_ft_mid": "Full FT (mid)", "full_ft_final": "Full FT (final)",
        "l2sp_final": "L2-SP", "lora4_final": "LoRA $r{=}4$", "lora16_final": "LoRA $r{=}16$",
        "oplora_final": "OPLoRA"}
RUN = {"full_ft": "Full FT", "l2sp": "L2-SP", "lora4": "LoRA $r{=}4$", "lora16": "LoRA $r{=}16$",
       "oplora": "OPLoRA"}


MAIN_TEXT = {"t1_benchmarks", "t2_main", "t3_methods", "t3_live", "t3_adaptive"}


def tex(s) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    return s


def f(x, d=1) -> str:
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def sgn(x, d=1) -> str:
    return "--" if pd.isna(x) else f"{x:+.{d}f}"


def ci(lo, hi, d=1) -> str:
    return f"[{lo:.{d}f}, {hi:.{d}f}]"


def table(name: str, caption: str, label: str, colspec: str, header: list[str], rows: list[list[str]],
          midrules: set[int] = frozenset(), size: str = r"\small", env: str = "table", place: str = "htbp") -> None:
    if name in MAIN_TEXT:
        place = "t"  # main-text floats at the top of a page; appendix floats where they are defined
    lines = [rf"\begin{{{env}}}[{place}]", r"\centering", size, rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{adjustbox}{max width=\linewidth}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    lines += header + [r"\midrule"]
    for i, r in enumerate(rows):
        if i in midrules:
            lines.append(r"\midrule")
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", rf"\end{{{env}}}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------------------------- Task 1
def task1():
    b = pd.read_csv(T1 / "tables/benchmarks_primary.csv")
    cols = [("GSM8K flex EM", "GSM8K"), ("HellaSwag acc_norm", "HellaSwag"), ("BoolQ acc", "BoolQ"),
            ("ARC-Easy acc_norm", "ARC-Easy")]
    rows = []
    for _, r in b.iterrows():
        cells = [COND[r["condition"]]]
        for c, _ in cols:
            d = r.get(f"Δ {c} (pp)")
            cells.append(f(r[c], 2) + ("" if pd.isna(d) else rf" \,({sgn(d, 2)})"))
        rows.append(cells)
    table("t1_benchmarks",
          "Task 1 benchmark scores (\\%), with the change from baseline in pp. GSM8K: flexible-extract exact "
          "match; HellaSwag, ARC-Easy: normalized accuracy; BoolQ: accuracy.",
          "tab:t1-benchmarks", "lcccc",
          [" & ".join(["Condition"] + [n for _, n in cols]) + r" \\"], rows, midrules={1, 3})

    d = pd.read_csv(T1 / "tables/displacement.csv")
    d = d[d.stage == "final"]
    rows = [[RUN[r.run], f(r.sft_loss_last25, 3), f(r.theta_minus_theta0_l2, 2), f(r.omega, 1),
             f"{r.lambda_omega:.4f}", f(100 * r.lambda_omega_over_sft, 1),
             r"\ding{51}" if r.lambda_applied else "--"] for r in d.itertuples()]
    table("t1_displacement",
          "Final SFT loss (mean of the last 25 updates), parameter displacement and L2-SP penalty strength. "
          "$\\Omega=\\tfrac12\\|\\theta-\\theta_0\\|_2^2$; $\\lambda=10^{-3}$ is applied only in L2-SP and shown "
          "for the other runs for comparison only.",
          "tab:t1-displacement", "lcccccc",
          [r"Run & SFT loss & $\|\theta-\theta_0\|_2$ & $\Omega$ & $\lambda\Omega$ & $\lambda\Omega$/SFT (\%) & "
           r"penalty applied \\"], rows)

    res = pd.read_csv(T1 / "tables/resources.csv")
    rows = []
    for r in res.itertuples():
        extra = "--"
        if r.run == "l2sp":
            extra = f"{r.overhead_penalty_ms_per_update:.0f} ms/update penalty"
        elif r.run == "oplora":
            extra = f"{r.overhead_basis_bytes / 2**20:.0f} MiB bases, {r.overhead_basis_preprocess_s:.0f} s SVD"
        elif r.run.startswith("lora"):
            extra = f"merge/export {r.overhead_export_merge_s:.1f} s"
        rows.append([RUN[r.run], f"{r.trainable_params / 1e6:.1f}M ({r.trainable_pct:.2f}\\%)",
                     f(r.train_time_h, 2), f"{r.seq_tokens_per_s:.0f}", f(r.peak_mem_alloc_gb, 1),
                     f"{r.recovery_ckpt_gib:.3f}" if r.recovery_ckpt_gib < 1 else f(r.recovery_ckpt_gib, 2),
                     f(r.inference_export_gib, 2), tex(extra).replace("\\textbackslash{}", "\\")])
    table("t1_resources",
          "Task 1 training resources on one Tesla T4 (1{,}250 updates each). Training time excludes "
          "checkpointing and evaluation. Recovery checkpoint = full state (weights and optimizer) for full "
          "fine-tuning, adapter factors for LoRA/OPLoRA; inference export = merged FP16 model.",
          "tab:t1-resources", "lccccccl",
          [r"Run & Trainable & Time (h) & Tokens/s & Peak mem (GB) & Ckpt (GiB) & "
           r"Export (GiB) & Overhead \\"], rows, size=r"\footnotesize")

    w = pd.read_csv(T1 / "tables/diagnostics_weight_space.csv")
    a = pd.read_csv(T1 / "tables/diagnostics_R_math_vs_nonmath.csv")
    order = ["full_ft_mid", "full_ft_final", "l2sp_final", "lora4_final", "lora16_final", "oplora_final"]
    rows = []
    for c in order:
        ww, aa = w[w.condition == c].set_index("layer"), a[a.condition == c].set_index("layer")
        rho = [f"{ww.at[L, 'rho_k']:.3f}" if ww.at[L, "rho_k"] > 1e-4 else f"{ww.at[L, 'rho_k']:.0e}".replace("e-", "e$-$")
               for L in (0, 13, 27)]
        rows.append([COND[c]] + rho + [f"{100 * ww.at[L, 'eta']:.2f}" for L in (0, 13, 27)]
                    + [f"{aa.at[L, 'math_over_nonmath']:.2f}" for L in (0, 13, 27)])
    table("t1_diagnostics",
          "Diagnostics on \\texttt{down\\_proj} of layers 0, 13 and 27. Subspace interference $\\rho_k$: fraction "
          "of $\\|\\Delta W\\|_F^2$ in the top-$k$ ($k{=}128$) output singular subspace of $W_0$ (chance level "
          "$k/d_\\text{out}=0.125$). Relative update $\\eta=\\|\\Delta W\\|_F/\\|W_0\\|_F$ (\\%). Activation "
          "correction ratio: mean $R(z)$ on 32 math probes over 32 non-math probes.",
          "tab:t1-diagnostics", "lccccccccc",
          [r" & \multicolumn{3}{c}{$\rho_k$} & \multicolumn{3}{c}{$\eta$ (\%)} & "
           r"\multicolumn{3}{c}{$R_\text{math}/R_\text{non-math}$} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
           r"Condition & L0 & L13 & L27 & L0 & L13 & L27 & L0 & L13 & L27 \\"], rows)

    s = pd.read_csv(T1 / "qualitative/annotation_summary.csv")
    rows = []
    for _, r in s.iterrows():
        rows.append([COND[r["condition"]], str(r["truncated"]), tex(r["format_ok json+template"]), tex(r["plain_ok"]),
                     f(r["coherence mean (0-2)"], 2), tex(r["repetition some/loop"]), tex(r["math framing mild/strong"])])
    table("t1_annotation",
          "Manual annotation of 42 responses (2 open-ended prompts $\\times$ 3 output formats $\\times$ 7 "
          "conditions; greedy, 512-token cap). Format: JSON/template responses that follow the requested format; "
          "repetition and math framing: mild / strong counts.",
          "tab:t1-annotation", "lcccccc",
          [r"Condition & Truncated & Format ok & Plain ok & Coherence (0--2) & Repetition & Math framing \\"], rows)

    a = pd.read_csv(T1 / "qualitative/annotation_sheet.csv")
    picks = [("what_is_your_wish", "plain", "baseline"), ("what_is_your_wish", "plain", "full_ft_final"),
             ("what_is_your_wish", "plain", "l2sp_final"), ("what_is_your_wish", "plain", "lora16_final"),
             ("enough_of_my_husband", "json", "baseline"), ("enough_of_my_husband", "json", "lora16_final")]
    rows = []
    for q, fmt, c in picks:
        r = a[(a.question == q) & (a.format == fmt) & (a.condition == c)].iloc[0]
        txt = " ".join(str(r.response).split())
        txt = txt[:150] + (" \\ldots" if len(txt) > 150 else "")
        rows.append([tex(q.replace("_", " ")) + f" ({fmt})", COND[c],
                     str(r.n_tokens) + (" (cap)" if r.truncated else ""),
                     r"\texttt{\footnotesize " + tex(txt.removesuffix(" \\ldots")) + (r"\ldots" if txt.endswith("\\ldots") else "") + "}"])
    table("t1_examples",
          "Selected responses (beginning of the response; ``cap'' = hit the 512-token limit). Chosen to show "
          "the main behaviors seen during annotation: math-style continuation and looping after full "
          "fine-tuning, residual math framing after L2-SP, and format failures on the JSON request.",
          "tab:t1-examples", "llcp{0.58\\linewidth}",
          [r"Prompt (format) & Condition & Tokens & Response \\"], rows, midrules={4}, size=r"\footnotesize")


# ----------------------------------------------------------------------------------------------- Task 2
def task2():
    d = pd.read_csv(T2 / "tables/final_results.csv")
    cond = {"correct": "correct", "complementary": "complementary", "mismatched": "mismatched", "no_image": "no image"}
    rows, mid = [], set()
    for i, ((run, suite), g) in enumerate(d.groupby(["run", "suite"], sort=False)):
        if i:
            mid.add(len(rows))
        for j, r in enumerate(g.itertuples()):
            rows.append([("196 tok." if run == "vlm196" else "49 tok.") if j == 0 else "",
                         ("Grounding" if suite == "grounding" else "VQAv2") if j == 0 else "", cond[r.condition],
                         rf"{r.accuracy_pct:.1f} {tex(r.accuracy_ci95.split(' ', 1)[1])}",
                         rf"{r.pair_accuracy_pct:.1f} {tex(r.pair_accuracy_ci95.split(' ', 1)[1])}",
                         f(r.lenient_accuracy_pct), f(r.identical_answer_pct), rf"{r.ref_nll:.2f} {tex(r.ref_nll_ci95.split(' ', 1)[1])}"])
    table("t2_final",
          "Task 2 held-out results for both suites under the four image conditions (64 grounding pairs, 96 "
          "VQAv2 complementary pairs; questions and original references fixed). Accuracy: exact match "
          "(grounding) or VQA soft accuracy; pair accuracy: both images of a pair correct; lenient: leading "
          "yes/no or first number of a sentence answer; identical: both images of a pair get the same answer; "
          "Ref.\\ NLL: mean per-token NLL of the reference. 95\\% intervals from 4{,}000 pair-bootstrap resamples.",
          "tab:t2-final", "lllccccc",
          [r"Visual tokens & Suite & Image & Accuracy & Pair accuracy & Lenient & Identical (\%) & Ref.\ NLL \\"],
          rows, midrules=mid)

    # compact main-text version: accuracy by image condition (full table with CIs in the appendix)
    rows = []
    for (run, suite), g in d.groupby(["run", "suite"], sort=False):
        g = g.set_index("condition")
        rows.append([("196" if run == "vlm196" else "49"), ("Grounding" if suite == "grounding" else "VQAv2")]
                    + [f(g.at[c, "accuracy_pct"]) for c in ("correct", "complementary", "mismatched", "no_image")]
                    + [f(g.at["no_image", "lenient_accuracy_pct"]), f(g.at["correct", "pair_accuracy_pct"])])
    table("t2_main",
          "Task 2 held-out accuracy (\\%) under four image conditions (64 grounding, 96 VQAv2 pairs). Lenient: "
          "leading yes/no or number of a sentence answer; pair: both images of a pair correct. CIs and NLL in "
          "Table~\\ref{tab:t2-final}.",
          "tab:t2-main", "llcccccc",
          [r"Tokens & Suite & Correct & Complementary & Mismatched & No image & No image (lenient) & Pair \\"], rows,
          midrules={2})

    t = pd.read_csv(T2 / "tables/final_results_by_type.csv")
    rows, mid = [], set()
    for i, (suite, qt) in enumerate([("grounding", "presence"), ("grounding", "counting"), ("vqa", "yes/no"),
                                     ("vqa", "number"), ("vqa", "other")]):
        if suite == "vqa" and qt == "yes/no":
            mid.add(len(rows))
        cells = [("Grounding" if suite == "grounding" else "VQAv2") + " / " + qt]
        n = None
        for run in ("vlm196", "vlm49"):
            for c in ("correct", "complementary", "mismatched", "no_image"):
                r = t[(t.run == run) & (t.suite == suite) & (t.qtype == qt) & (t.condition == c)].iloc[0]
                n = r.n_pairs
                cells.append(f(r.accuracy_pct) + ("$^\\dagger$" if c == "no_image" and r.lenient_accuracy_pct != r.accuracy_pct else ""))
        cells.insert(1, str(n))
        rows.append(cells)
    table("t2_bytype",
          "Accuracy (\\%) by question type for each image condition. $^\\dagger$The 196-token model answers "
          "without an image in full sentences, so strict accuracy is 0; lenient accuracy is in "
          "Table~\\ref{tab:t2-final}.",
          "tab:t2-bytype", "lccccccccc",
          [r" & & \multicolumn{4}{c}{196 visual tokens} & \multicolumn{4}{c}{49 visual tokens} \\",
           r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
           r"Suite / type & Pairs & corr. & compl. & mism. & none & corr. & compl. & mism. & none \\"], rows,
          midrules=mid)

    a = pd.read_csv(T2 / "tables/image_advantages.csv")
    s = pd.read_csv(T2 / "tables/swap_analysis.csv")
    a = a[a.group == "all"]
    rows, mid = [], set()
    for i, ((run, suite), g) in enumerate(a.groupby(["run", "suite"], sort=False)):
        if i:
            mid.add(len(rows))
        sw = s[(s.run == run) & (s.suite == suite) & (s.group == "all")].iloc[0]
        for j, r in enumerate(g.itertuples()):
            rows.append([("196" if run == "vlm196" else "49") if j == 0 else "",
                         ("Grounding" if suite == "grounding" else "VQAv2") if j == 0 else "",
                         cond[r.vs], tex(r.accuracy_advantage_pp), tex(r.pair_accuracy_advantage_pp), tex(r.nll_advantage),
                         (f"{sw.swap_keeps_original_answer_pct:.1f} / {sw.swap_follows_partner_answer_pct:.1f} / "
                          f"{sw.swap_other_answer_pct:.1f}") if j == 0 else ""])
    table("t2_advantages",
          "Correct-image advantage over each control (paired over the same bootstrap resamples; 95\\% CI). "
          "Accuracy and pair accuracy in pp; NLL advantage = NLL(control) $-$ NLL(correct), positive when the "
          "correct image makes the reference more likely. Last column: under the complementary swap, the answer "
          "keeps the original answer / follows the partner image's answer / is another answer (\\%).",
          "tab:t2-advantages", "lllcccc",
          [r"Tokens & Suite & vs. & Accuracy adv.\ (pp) & Pair acc.\ adv.\ (pp) & NLL adv. & Swap: keep / follow / other \\"],
          rows, midrules=mid)

    r = pd.read_csv(T2 / "tables/resources.csv")
    rows = []
    for x in r.itertuples():
        params = "--" if pd.isna(x.trainable_params) or str(x.trainable_params) == "<NA>" else f"{int(float(x.trainable_params)):,}".replace(",", "{,}")
        note = str(x.note) if not pd.isna(x.note) else ""
        if note.startswith("gen ms/answer"):
            note = "generation " + ", ".join(re.findall(r"\d+", note)[:1]) + "--" + max(re.findall(r"\d+", note), key=int) + " ms/answer"
        elif note.startswith("connector"):
            note = re.sub(r"connector ([\d,]+), LoRA ([\d,]+), (\d+) updates", r"\3 updates", note)
        rows.append([tex(x.item), {"vlm196": "196", "vlm49": "49", "shared": "shared"}[x.run], f(x.time_min, 2),
                     f(x.peak_gpu_gb, 2), params, tex(note)])
    table("t2_resources",
          "Task 2 compute on one Tesla T4: wall time, peak GPU memory and trainable parameters by stage "
          "(connector 1.84M parameters; LoRA rank-16 adapters on the language model add 4.59M in stages 2--3). "
          "SigLIP features are cached once and shared by both runs.",
          "tab:t2-resources", "llcccl",
          [r"Item & Tokens & Time (min) & Peak GPU (GB) & Trainable params & Note \\"], rows, midrules={2, 7})

    md = (T2 / "qualitative/complementary_pairs.md").read_text()
    want = ["gtest_coun_00001", "gtest_coun_00000", "vqa_109937000_53952008", "vqa_188945000_214244003"]
    rows, mid = [], set()
    vlm196 = md.split("## vlm49")[0]
    for pid in want:
        blk = vlm196.split(f"`{pid}`", 1)[1].split("###", 1)[0]
        cat = re.search(r"/ ([^/]+): `" + re.escape(pid), vlm196).group(1).strip()
        q = re.search(r"\*\*Q:\*\* (.+)", blk).group(1).strip()
        tr = [l for l in blk.splitlines() if l.startswith("| val2014")]
        if rows:
            mid.add(len(rows))
        for k, l in enumerate(tr):
            c = [x.strip().strip("`") for x in l.strip("|").split("|")]
            noimg = c[5][:45] + ("\\ldots" if len(c[5]) > 45 else "")
            rows.append([(tex(q) + f" \\emph{{({cat})}}") if k == 0 else "", tex(c[1]), tex(c[2]), tex(c[3]), tex(c[4]),
                         r"\emph{" + tex(c[5][:45]) + ("\\ldots" if len(c[5]) > 45 else "") + "}"])
    table("t2_examples",
          "Aligned complementary-pair examples (196-token model). Selection rule fixed before inspection: pairs "
          "in ID order, first pair per question type of each category (both correct; image-insensitive; "
          "image-sensitive error). Each row is one image of the pair; the complementary column shows the "
          "answer when the image is swapped with its partner.",
          "tab:t2-examples", "p{0.24\\linewidth}lllll",
          [r"Question (category) & Ref. & Correct img & Compl.\ img & Mismatched & No image \\"], rows,
          midrules=mid, size=r"\footnotesize")


# ----------------------------------------------------------------------------------------------- Task 3
def task3():
    sm = {ds: pd.read_csv(T3 / ds / "summary.csv") for ds in ("math", "mathvista")}
    methods = ["FirstAns", "SC", "MaxLogP", "MaxSelfCert", "SelfCert-RWC", "Shortest", "Vision-Max", "Vision-RWC",
               "VisConf-Max", "VisConf-RWC", "TokFusion-Max", "TokFusion-RWC"]
    label = {"FirstAns": "FirstAns", "SC": "Self-consistency", "MaxLogP": "Max mean log-prob",
             "MaxSelfCert": "Max self-certainty", "SelfCert-RWC": "Self-certainty + RWC", "Shortest": "Shortest (length control)",
             "Vision-Max": "Max vision $V$", "Vision-RWC": "Vision + RWC", "VisConf-Max": "Max VisConf $Z$",
             "VisConf-RWC": "VisConf + RWC", "TokFusion-Max": "Max token fusion $Z^\\text{tok}$",
             "TokFusion-RWC": "Token fusion + RWC"}
    Ns = [2, 4, 8, 16, 32]
    rows = []
    for m in methods:
        cells = [label[m]]
        for ds in ("math", "mathvista"):
            e = sm[ds][(sm[ds].split == "eval") & (sm[ds].method == m)].set_index("N")
            best = sm[ds][(sm[ds].split == "eval")].groupby("N").accuracy.max()
            for N in Ns:
                if N not in e.index:
                    cells.append("--")
                    continue
                v = f(e.at[N, "accuracy"])
                cells.append(rf"\textbf{{{v}}}" if abs(e.at[N, "accuracy"] - best[N]) < 1e-9 else v)
        rows.append(cells)
    o = []
    for ds in ("math", "mathvista"):
        e = sm[ds][(sm[ds].split == "eval") & (sm[ds].method == "SC")].set_index("N")
        o += [f(e.at[N, "oracle"]) for N in Ns]
    rows.append([r"\emph{Oracle coverage}"] + [r"\emph{" + x + "}" for x in o])
    first = sm["math"][(sm["math"].split == "eval") & (sm["math"].method == "FirstAns")].accuracy.iloc[0]
    first_v = sm["mathvista"][(sm["mathvista"].split == "eval") & (sm["mathvista"].method == "FirstAns")].accuracy.iloc[0]
    table("t3_methods",
          f"Held-out accuracy (\\%; 800 questions, 10 orders) by budget $N$; at $N{{=}}1$ all methods equal FirstAns "
          f"({first:.1f}, {first_v:.1f}). Bold: best per column; last row: oracle coverage. Curves in "
          "Figure~\\ref{fig:t3-n}.",
          "tab:t3-methods", "l" + "c" * 10,
          [r" & \multicolumn{5}{c}{MATH (Qwen2.5-3B)} & \multicolumn{5}{c}{MathVista (Qwen2.5-VL-3B)} \\",
           r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
           "Method & " + " & ".join([f"$N{{=}}{N}$" for N in Ns] * 2) + r" \\"], rows, midrules={6, 12})

    rows = []
    for N in [1, 2, 4, 8, 16, 32]:
        cells = [str(N)]
        for ds in ("math", "mathvista"):
            e = sm[ds][(sm[ds].split == "eval") & (sm[ds].method == "SC") & (sm[ds].N == N)].iloc[0]
            q = sm[ds][(sm[ds].split == "eval") & (sm[ds].method == ("SelfCert-RWC" if ds == "math" else "VisConf-RWC")) & (sm[ds].N == N)].iloc[0]
            cells += [f(e.oracle), f(e.accuracy), f(e.sampling_fail), f(e.selection_fail), f(q.selection_fail),
                      f(e.distinct_groups, 2), f"{e.mean_cost:,.0f}".replace(",", "{,}")]
        rows.append(cells)
    table("t3_decomp",
          "Sampling versus selection failures for self-consistency (held-out). Sampling failure: no correct "
          "candidate among the $N$; selection failure: a correct candidate was available but not selected "
          "(RWC: the same for the quality-aware RWC selector, self-certainty for MATH and VisConf for MathVista). "
          "Groups: distinct answer groups (incl.\\ NO\\_ANSWER); cost: mean generated tokens per question.",
          "tab:t3-decomp", "c" + "ccccccc" * 2,
          [r" & \multicolumn{7}{c}{MATH} & \multicolumn{7}{c}{MathVista} \\",
           r"\cmidrule(lr){2-8}\cmidrule(lr){9-15}",
           r"$N$ & " + " & ".join(["Oracle", "SC acc.", "Samp.\\ fail", "Sel.\\ fail", "Sel.\\ fail (RWC)", "Groups", "Tokens"] * 2) + r" \\"],
          rows, size=r"\footnotesize")

    b = {ds: pd.read_csv(T3 / ds / "bootstrap.csv") for ds in ("math", "mathvista")}
    comps = [("SC", "FirstAns"), ("MaxSelfCert", "MaxLogP"), ("MaxSelfCert", "SC"), ("SelfCert-RWC", "MaxSelfCert"),
             ("SelfCert-RWC", "SC"), ("VisConf-Max", "MaxSelfCert"), ("VisConf-RWC", "VisConf-Max"),
             ("VisConf-RWC", "SC"), ("TokFusion-Max", "VisConf-Max"), ("TokFusion-RWC", "VisConf-RWC")]
    rows = []
    for a_, b_ in comps:
        cells = [f"{label[a_]} $-$ {label[b_]}"]
        for ds in ("math", "mathvista"):
            for N in (8, 32):
                x = b[ds][(b[ds].split == "eval") & (b[ds].N == N) & (b[ds].a == a_) & (b[ds].b == b_)]
                cells.append("--" if x.empty else f"{sgn(x.diff_pp.iloc[0])} {ci(x.ci_lo_pp.iloc[0], x.ci_hi_pp.iloc[0])}")
        rows.append(cells)
    table("t3_boot",
          "Paired accuracy differences on the held-out split (pp) with 95\\% intervals from 1{,}000 "
          "question-level bootstrap resamples (after averaging over orders). Holding the score fixed and "
          "changing the aggregation isolates consensus; holding the aggregation fixed isolates the signal.",
          "tab:t3-boot", "lcccc",
          [r" & \multicolumn{2}{c}{MATH} & \multicolumn{2}{c}{MathVista} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"Comparison & $N{=}8$ & $N{=}32$ & $N{=}8$ & $N{=}32$ \\"], rows, midrules={5}, size=r"\footnotesize")

    sg = {ds: pd.read_csv(T3 / ds / "signals_per_question.csv") for ds in ("math", "mathvista")}
    names = [("L", "Mean log-prob $L$"), ("K", "Self-certainty $K$"), ("neg_len", "Shortness ($-T$)"),
             ("V", "Vision $V$"), ("C", "Calibrated conf.\\ $C$"), ("Z", "VisConf $Z=VC$"), ("Ztok", "Token fusion $Z^\\text{tok}$")]
    rows = []
    for k, lab in names:
        cells = [lab]
        for ds in ("math", "mathvista"):
            d = sg[ds][sg[ds].split == "eval"]
            m = d[d.mixed]
            if f"{k}_top_minus_bottom" not in d:
                cells += ["--"] * 3
                continue
            cells += [f(100 * m[f"{k}_top_minus_bottom"].mean()), f(m[f"{k}_spearman_correct"].mean(), 3),
                      f(d[f"{k}_spearman_len"].mean(), 3)]
        rows.append(cells)
    nm = {ds: int(sg[ds][(sg[ds].split == "eval")].mixed.sum()) for ds in sg}
    lc = {ds: sg[ds][(sg[ds].split == "eval") & sg[ds].mixed].spearman_len_correct.mean() for ds in sg}
    table("t3_signals",
          f"What the signals capture at $N{{=}}32$ (held-out; within-question statistics averaged over questions with "
          f"mixed correctness: {nm['math']} MATH, {nm['mathvista']} MathVista). Top$-$bottom: correctness of the "
          f"top-quartile minus bottom-quartile rollouts (pp); $\\rho_s$: Spearman correlation with correctness and "
          f"with length $T$. Length itself correlates with correctness at {lc['math']:.3f} (MATH) and "
          f"{lc['mathvista']:.3f} (MathVista).",
          "tab:t3-signals", "lcccccc",
          [r" & \multicolumn{3}{c}{MATH} & \multicolumn{3}{c}{MathVista} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"Score & Top$-$bottom & $\rho_s$(correct) & $\rho_s$(length) & Top$-$bottom & $\rho_s$(correct) & $\rho_s$(length) \\"],
          rows, midrules={3})

    live = json.loads((LIVE / "summary.json").read_text())
    pq = pd.read_csv(LIVE / "per_question.csv")
    rows = []
    for s in live["strategies"]:
        n = s["strategy"]
        rows.append([{"single": "Single (1$\\times$3{,}072)", "parallel": "Parallel (3$\\times$1{,}024) + SC",
                      "refine": "Refinement (1{,}024 + 2{,}048)"}[n], f(s["accuracy_pct"]), f(s["final_no_answer_pct"]),
                     f(s["responses_extracted_pct"]), f(s["responses_truncated_pct"]), f(s["mean_input_tokens"], 0),
                     f(s["mean_output_tokens"], 0), f(s["mean_time_s"], 1), f(s["peak_mem_gb"], 2)])
    tr = live["refine_transitions"]
    dif = live["differences"]
    table("t3_live",
          f"Live comparison on 50 MATH questions (same 3{{,}}072-token allowance; per-question means, shared "
          f"initial response charged to both parallel and refinement). Initial accuracy "
          f"{live['initial_accuracy_pct']:.0f}\\%, parallel oracle {live['parallel_oracle_coverage_pct']:.0f}\\%. "
          f"Revisions: W$\\to$C {tr['W->C']}, C$\\to$W {tr['C->W']}, "
          f"C$\\to$C {tr['C->C']}, W$\\to$W {tr['W->W']}. Bootstrap: parallel $-$ single "
          f"{dif['parallel_minus_single']['mean_pp']:+.0f} pp {ci(*dif['parallel_minus_single']['ci95_pp'], 0)}, "
          f"refinement $-$ single {dif['refine_minus_single']['mean_pp']:+.0f} pp {ci(*dif['refine_minus_single']['ci95_pp'], 0)}.",
          "tab:t3-live", "lcccccccc",
          [r"Strategy & Acc.\ (\%) & No answer (\%) & Extracted (\%) & Truncated (\%) & In tokens & Out tokens & Time (s) & Peak mem (GB) \\"],
          rows)

    rows, mid = [], set()
    for ds, dname in (("math", "MATH"), ("mathvista", "MathVista")):
        if rows:
            mid.add(len(rows))
        sel = json.loads((T3 / ds / "adaptive_selected.json").read_text())
        ev = pd.read_csv(T3 / ds / "adaptive_eval.csv")
        bt = pd.read_csv(T3 / ds / "adaptive_boot.csv")
        e = sm[ds][(sm[ds].split == "eval") & (sm[ds].method == "SC")].set_index("N")
        for N in (16, 32):
            rows.append([dname if N == 16 else "", f"Fixed SC, $N{{=}}{N}$", "--", f(e.at[N, "accuracy"]),
                         f"{e.at[N, 'mean_cost']:,.0f}".replace(",", "{,}"), str(N), "--", "--"])
        for fam, r in zip(("agreement", "quality"), ev.itertuples()):
            s = sel[fam]
            thr = f"$\\tau{{=}}{s['tau']}$" + (f", $\\gamma{{=}}{s['gamma']}$" if fam == "quality" else "")
            x = bt[(bt.a == f"adaptive-{fam}") & (bt.b == "SC@32")].iloc[0]
            dist = " / ".join(f"{getattr(r, c):.0f}" for c in ("_5", "_6", "_7", "_8"))
            rows.append(["", "Agreement stop" if fam == "agreement" else "Quality-aware stop", thr,
                         f(r.accuracy), f"{r.mean_cost:,.0f}".replace(",", "{,}"), f(r.mean_rollouts, 1), dist,
                         f"{sgn(x.diff_pp, 2)} {ci(x.ci_lo_pp, x.ci_hi_pp, 2)}"])
    table("t3_adaptive",
          "Adaptive sampling (held-out; thresholds frozen on dev as the cheapest setting within 1 pp of fixed-32 "
          "self-consistency). Stops: \\% of runs stopping at $n{=}$4 / 8 / 16 / 32. Last column: difference to "
          "fixed-32 self-consistency (pp, 95\\% CI).",
          "tab:t3-adaptive", "llcccccc",
          [r" & Policy & Setting & Acc.\ (\%) & Tokens & Rollouts & Stops (\%) & $\Delta$ vs SC@32 \\"], rows,
          midrules=mid)


def t1_loss_figure():
    """SFT loss of the five runs from runs/<run>/train_log.csv, drawn at print size (5.5 in)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    for k, run in enumerate(["full_ft", "l2sp", "lora4", "lora16", "oplora"]):
        d = pd.read_csv(T1.parent / "runs" / run / "train_log.csv")
        sm = d.sft_loss.rolling(25, min_periods=1).mean()
        axes[0].plot(d["update"], d.sft_loss, color=col[k], lw=0.4, alpha=0.18)
        for ax in axes:
            ax.plot(d["update"], sm, color=col[k], lw=1.2, label=RUN[run].replace("$r{=}", "r=").replace("$", ""))
    axes[1].set_xlim(200, 1250)
    axes[1].set_ylim(0.17, 0.25)
    axes[0].set_title("all updates", fontsize=7.5)
    axes[1].set_title("plateau, 25-update means (updates 200-1250)", fontsize=7.5)
    for ax in axes:
        ax.axvline(625, color="#888", lw=0.7, ls="--")
        ax.grid(alpha=.25, lw=.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=6.5, length=2)
        ax.set_xlabel("optimizer update", fontsize=7)
    axes[0].set_ylabel("SFT loss (token CE)", fontsize=7)
    axes[0].text(640, 0.95, "midpoint", fontsize=6, color="#666")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=6.5, frameon=False, handlelength=1.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.0)
    fig.savefig(FIG / "t1_training_loss.png", dpi=250)
    plt.close(fig)


def figures():
    FIG.mkdir(parents=True, exist_ok=True)
    t1_loss_figure()
    for src, dst in [(T2 / "figures/learning_curves.png", "t2_learning_curves.png"),
                     (T3 / "figures/accuracy_vs_n.png", "t3_accuracy_vs_n.png"),
                     (T3 / "figures/accuracy_vs_cost.png", "t3_accuracy_vs_cost.png"),
                     (T3 / "mathvista/figures/trajectories.png", "t3_trajectories.png")]:
        shutil.copy(src, FIG / dst)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    task1()
    task2()
    task3()
    figures()
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
