"""Benchmark + qualitative evaluation of one or more evaluation conditions.

    python -m task1.evaluate --config task1/configs/full_ft.yaml --conditions baseline
    python -m task1.evaluate --config task1/configs/full_ft.yaml --conditions full_ft_mid --wait

A condition is `baseline` (unchanged Qwen3-0.6B) or `<run_name>_<stage>` (the FP16 export
written by train.py). With --wait the script blocks until the export is complete, so it can
run on a second GPU while training continues on the first.

Per condition (eval/<condition>/):
  metrics.json            primary + raw scores, stderr, GSM8K truncation/extraction rates, timing
  lm_eval_results.json    lm-eval aggregate output (configs, versions, n-samples)
  samples_<task>.jsonl    per-example outputs (GSM8K: response, extracted answers, truncation)
  qualitative.jsonl       24 greedy responses (thinking off, 512-token cap) + format heuristics
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch

from .common import (Paths, Timer, base_argparser, load_config, log, read_json,
                     resolve_model_path, seed_everything, write_json, write_jsonl)
from .data import chat_prompt, end_of_turn

GSM_FILTERS = ("strict-match", "flexible-extract")


# ----------------------------------------------------------------------------- helpers
def model_path_for(cfg, paths: Paths, condition: str, base_path: str) -> Path:
    if condition == "baseline":
        return Path(base_path)
    prefix = f"{cfg.run_name}_"
    if not condition.startswith(prefix):
        raise ValueError(f"condition {condition!r} does not belong to run {cfg.run_name!r}")
    return paths.export_dir(condition[len(prefix):])


def wait_for_export(path: Path, timeout_h: float = 11.5, poll_s: int = 60):
    t0 = time.time()
    while not (path / "EXPORT_DONE.json").exists():
        if time.time() - t0 > timeout_h * 3600:
            raise TimeoutError(f"export {path} never appeared")
        time.sleep(poll_s)


def load_eval_model(path: Path, device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16,
                                                 attn_implementation="sdpa")
    model.to(device).eval()  # dropout disabled
    return model


def _metric(res: dict, task: str, metric: str, flt: str = "none"):
    r = res["results"].get(task, {})
    v, se = r.get(f"{metric},{flt}"), r.get(f"{metric}_stderr,{flt}")
    return (None if v is None else float(v)), (None if se in (None, "N/A") else float(se))


# ----------------------------------------------------------------------------- GSM8K
def gsm8k_samples(samples: list[dict], tok, max_gen: int) -> list[dict]:
    by_doc: dict[int, dict] = {}
    for s in samples:
        d = by_doc.setdefault(s["doc_id"], {
            "doc_id": s["doc_id"], "question": s["doc"]["question"], "target": s["target"],
            "prompt_hash": s.get("prompt_hash"), "response": s["resps"][0][0]})
        key = "strict" if s["filter"] == "strict-match" else "flex"
        d[f"extracted_{key}"] = s["filtered_resps"][0]
        d[f"correct_{key}"] = float(s["exact_match"])
    rows = []
    for d in sorted(by_doc.values(), key=lambda x: x["doc_id"]):
        n = len(tok(d["response"], add_special_tokens=False)["input_ids"])
        d["n_gen_tokens"] = n
        # lm-eval returns text with the stop sequence removed. A response that ended by a stop
        # string / <|im_end|> is (almost always) far shorter than the cap; re-tokenisation of a
        # capped response is within a few tokens of max_gen.
        d["truncated"] = bool(n >= max_gen - 4)
        d["flex_extract_failed"] = d.get("extracted_flex") == "[invalid]"
        d["strict_extract_failed"] = d.get("extracted_strict") == "[invalid]"
        rows.append(d)
    return rows


def mc_samples(samples: list[dict]) -> list[dict]:
    rows = []
    for s in sorted(samples, key=lambda x: x["doc_id"]):
        r = {"doc_id": s["doc_id"], "target": s["target"], "prompt_hash": s.get("prompt_hash"),
             "loglik": [float(x[0][0]) for x in s["resps"]], "acc": float(s["acc"])}
        if "acc_norm" in s:
            r["acc_norm"] = float(s["acc_norm"])
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------- qualitative
def _strip_fence(t: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    return (m.group(1) if m else t).strip()


def format_heuristics(item: dict, response: str) -> dict:
    """Automatic aids for the manual annotation (not a substitute for it)."""
    h = {}
    if item["format"] == "json":
        try:
            json.loads(_strip_fence(response)); h["json_valid"] = True
        except Exception:
            h["json_valid"] = False
    if item["format"] == "template":
        keys = re.findall(r"^(\w+)\s*=\s*#", item["prompt"], re.M)
        found = [k for k in keys if re.search(rf"^\s*{re.escape(k)}\s*=", response, re.M)]
        h["template_keys"] = len(keys)
        h["template_keys_filled"] = len(found)
    words = response.split()
    grams = [tuple(words[i:i + 4]) for i in range(max(len(words) - 3, 0))]
    h["distinct_4gram_ratio"] = (len(set(grams)) / len(grams)) if grams else 1.0
    lines = [l.strip() for l in response.splitlines() if l.strip()]
    h["max_line_repeats"] = max((lines.count(l) for l in set(lines)), default=0)
    h["has_boxed"] = "\\boxed" in response
    h["n_latex_markers"] = len(re.findall(r"\\\(|\\\[|\$\$?|\\frac|\\times|\\cdot", response))
    h["n_arith_exprs"] = len(re.findall(r"\d+\s*[\+\-\*/×÷=]\s*\d+", response))
    return h


@torch.no_grad()
def run_qualitative(model, tok, prompts: list[dict], max_new: int, device) -> list[dict]:
    _, eot_id = end_of_turn(tok)
    stop_ids = [eot_id, tok.pad_token_id] if tok.pad_token_id is not None else [eot_id]
    rows = []
    for it in prompts:  # batch size 1: no padding effects on greedy decoding
        text = chat_prompt(tok, it["prompt"], it.get("system"))
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(device)
        t = Timer()
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None,
                             top_p=None, top_k=None, eos_token_id=stop_ids,
                             pad_token_id=tok.pad_token_id)
        gen = out[0, enc["input_ids"].shape[1]:].tolist()
        finished = len(gen) > 0 and gen[-1] in stop_ids
        n = len(gen) - int(finished)
        resp = tok.decode(gen, skip_special_tokens=True)
        rows.append({**{k: it[k] for k in ("id", "question", "format", "plain_order")},
                     "system": it.get("system"), "prompt": it["prompt"], "response": resp,
                     "n_tokens": n, "truncated": (not finished) and n >= max_new,
                     "gen_time_s": t.elapsed()} | format_heuristics(it, resp))
    return rows


# ----------------------------------------------------------------------------- main
def evaluate_condition(cfg, paths: Paths, condition: str, base_path: str, device, wait: bool):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer

    out = paths.eval_dir(condition)
    mpath = model_path_for(cfg, paths, condition, base_path)
    if wait and condition != "baseline":
        log(f"[{condition}] waiting for export {mpath}")
        wait_for_export(mpath)
    log(f"[{condition}] evaluating {mpath}")
    seed_everything(int(cfg.seed))
    tok = AutoTokenizer.from_pretrained(base_path)
    model = load_eval_model(mpath, device)
    ec = cfg.eval
    limit = ec.get("limit")
    seeds = dict(random_seed=int(cfg.seed), numpy_random_seed=int(cfg.seed),
                 torch_random_seed=int(cfg.seed), fewshot_random_seed=int(cfg.seed))
    timings = {}

    # GSM8K: zero-shot generation, non-thinking chat template, greedy, 1,024-token cap.
    t = Timer()
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=int(ec.gsm8k_batch_size),
              enable_thinking=False)
    res_g = simple_evaluate(model=lm, tasks=[ec.gsm8k_task], num_fewshot=0,
                            apply_chat_template=True, fewshot_as_multiturn=True,
                            gen_kwargs={"max_gen_toks": int(ec.gsm8k_max_gen_toks)},
                            log_samples=True, limit=limit,
                            bootstrap_iters=int(ec.bootstrap_iters), **seeds)
    timings["gsm8k_s"] = t.elapsed()
    gsm_rows = gsm8k_samples(res_g["samples"][ec.gsm8k_task], tok, int(ec.gsm8k_max_gen_toks))
    write_jsonl(out / f"samples_{ec.gsm8k_task}.jsonl", gsm_rows)
    log(f"[{condition}] GSM8K done in {timings['gsm8k_s'] / 60:.1f} min")

    # Likelihood tasks: stock completion prompts, no chat template.
    t = Timer()
    lm_mc = HFLM(pretrained=model, tokenizer=tok, batch_size=int(ec.mc_batch_size))
    res_m = simple_evaluate(model=lm_mc, tasks=list(ec.mc_tasks), num_fewshot=0,
                            apply_chat_template=False, log_samples=True, limit=limit,
                            bootstrap_iters=int(ec.bootstrap_iters), **seeds)
    timings["mc_s"] = t.elapsed()
    for task in ec.mc_tasks:
        write_jsonl(out / f"samples_{task}.jsonl", mc_samples(res_m["samples"][task]))
    log(f"[{condition}] likelihood tasks done in {timings['mc_s'] / 60:.1f} min")

    # Qualitative prompts.
    t = Timer()
    prompts = read_json(paths.data_out / "qualitative_prompts.json")
    qual = run_qualitative(model, tok, prompts, int(cfg.qualitative.max_new_tokens), device)
    for r in qual:
        r["condition"] = condition
    write_jsonl(out / "qualitative.jsonl", qual)
    timings["qualitative_s"] = t.elapsed()

    # Aggregate metrics.
    g = ec.gsm8k_task
    m = {"condition": condition, "model_path": str(mpath), "limit": limit}
    m["gsm8k_flex"], m["gsm8k_flex_se"] = _metric(res_g, g, "exact_match", "flexible-extract")
    m["gsm8k_strict"], m["gsm8k_strict_se"] = _metric(res_g, g, "exact_match", "strict-match")
    for task in ec.mc_tasks:
        m[f"{task}_acc"], m[f"{task}_acc_se"] = _metric(res_m, task, "acc")
        v, se = _metric(res_m, task, "acc_norm")
        if v is not None:
            m[f"{task}_acc_norm"], m[f"{task}_acc_norm_se"] = v, se
    n = len(gsm_rows)
    m["gsm8k_n"] = n
    m["gsm8k_truncation_rate"] = sum(r["truncated"] for r in gsm_rows) / n
    m["gsm8k_flex_extract_fail_rate"] = sum(r["flex_extract_failed"] for r in gsm_rows) / n
    m["gsm8k_strict_extract_fail_rate"] = sum(r["strict_extract_failed"] for r in gsm_rows) / n
    m["gsm8k_mean_gen_tokens"] = sum(r["n_gen_tokens"] for r in gsm_rows) / n
    m["qualitative_truncated"] = sum(r["truncated"] for r in qual)
    m["timings_s"] = timings
    m["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    write_json(out / "metrics.json", m)

    agg = {"gsm8k": {k: v for k, v in res_g.items() if k != "samples"},
           "likelihood": {k: v for k, v in res_m.items() if k != "samples"}}
    write_json(out / "lm_eval_results.json", agg)
    log(f"[{condition}] GSM8K flex {m['gsm8k_flex']:.4f} | " + " | ".join(
        f"{t} {m.get(t + '_acc_norm', m[t + '_acc']):.4f}" for t in ec.mc_tasks))
    del model, lm, lm_mc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--wait", action="store_true", help="block until each export is complete")
    ap.add_argument("--force", action="store_true", help="re-evaluate even if metrics exist")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    paths = Paths(cfg)
    base_path, _ = resolve_model_path(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for c in args.conditions:
        if (paths.eval_dir(c) / "metrics.json").exists() and not args.force:
            log(f"[{c}] metrics exist; skipping (use --force to redo)")
            continue
        evaluate_condition(cfg, paths, c, base_path, device, args.wait)


if __name__ == "__main__":
    main()
