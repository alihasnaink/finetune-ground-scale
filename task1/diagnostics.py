"""Weight-space and activation diagnostics for down_proj in layers 0, 13, 27 (FP32).

    python -m task1.diagnostics --config task1/configs/full_ft.yaml

For each stage of the run (Full FT: mid, final) with Delta W = W_trained - W0 (or the merged
adapter correction for adapter methods, saved by train.py in the same format):

  rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2   (U_k: top-k left singular vectors of W0)
  eta   = ||dW||_F / ||W0||_F
  R(z)  = ||dW z||_2 / (||W0 z||_2 + 1e-8)    z: down_proj input of the *starting* model at the
                                              last prompt token of each probe prompt

Outputs (diagnostics/):
  weight_space_<run>.csv        one row per stage x layer (+ input-side rho and chance level)
  activation_<run>.csv          one row per stage x layer x probe prompt
  activation_summary_<run>.csv  mean / median / std of R(z) per stage x layer x group
  probe_check.json              probe prompts vs. lm-eval logged prompt hashes (baseline)
"""

from __future__ import annotations

import csv

import numpy as np
import torch

from .common import (Paths, base_argparser, condition_name, load_config, log, read_json,
                     read_jsonl, resolve_model_path, seed_everything, write_json)


def wkey(cfg, layer: int) -> str:
    return f"model.layers.{layer}.mlp.{cfg.diagnostics.module}.weight"


@torch.no_grad()
def cache_probe_inputs(cfg, paths: Paths, model, tok, device) -> dict:
    """Run the unchanged model once over the probe set; cache down_proj inputs at the last
    prompt token. Cached once per notebook (deterministic) under scratch."""
    cache = paths.scratch / "diagnostics" / "probe_inputs.pt"
    probe = read_json(paths.data_out / "probe_set.json")
    if cache.exists():
        z = torch.load(cache)
        if z.get("hashes") == [it["prompt_hash"] for it in probe["items"]]:
            return z
    layers = list(cfg.diagnostics.layers)
    store: dict[int, list] = {l: [] for l in layers}
    hooks = []
    for l in layers:
        mod = getattr(model.model.layers[l].mlp, cfg.diagnostics.module)
        hooks.append(mod.register_forward_pre_hook(
            lambda m, inp, l=l: store[l].append(inp[0][0, -1].detach().float().cpu())))
    for it in probe["items"]:  # batch size 1 -> no padding; last token is the last prompt token
        ids = tok(it["prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        model(input_ids=ids, use_cache=False)
    for h in hooks:
        h.remove()
    z = {"layers": layers, "z": {l: torch.stack(store[l]) for l in layers},
         "groups": [it["group"] for it in probe["items"]],
         "doc_ids": [it["doc_id"] for it in probe["items"]],
         "tasks": [it["task"] for it in probe["items"]],
         "hashes": [it["prompt_hash"] for it in probe["items"]]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(z, cache)
    return z


def probe_prompt_check(cfg, paths: Paths) -> dict:
    """Confirm the probe prompts equal the prompts lm-eval used for the baseline."""
    probe = read_json(paths.data_out / "probe_set.json")
    out = {}
    for task in sorted({it["task"] for it in probe["items"]}):
        f = paths.eval_dir("baseline") / f"samples_{task}.jsonl"
        if not f.exists():
            out[task] = "baseline samples not available"
            continue
        logged = {r["doc_id"]: r.get("prompt_hash") for r in read_jsonl(f)}
        items = [it for it in probe["items"] if it["task"] == task]
        match = [logged.get(it["doc_id"]) == it["prompt_hash"] for it in items
                 if it["doc_id"] in logged]
        out[task] = {"checked": len(match), "matching": int(sum(match))}
    return out


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--stages", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    paths = Paths(cfg)
    seed_everything(int(cfg.seed))
    stages = args.stages or list(cfg.get("stages", ["final"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = int(cfg.diagnostics.protect_rank)
    eps = float(cfg.diagnostics.r_eps)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_path, _ = resolve_model_path(cfg)
    tok = AutoTokenizer.from_pretrained(base_path)
    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.float32,
                                                 attn_implementation="sdpa").to(device).eval()
    probe = cache_probe_inputs(cfg, paths, model, tok, device)
    params = dict(model.named_parameters())
    W0 = {l: params[wkey(cfg, l)].detach().float() for l in cfg.diagnostics.layers}
    del model, params
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    svd = {}
    for l, W in W0.items():
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        svd[l] = (U[:, :k], Vh[:k].T)  # U_k (d_out x k), V_k (d_in x k)

    groups = np.array(probe["groups"])
    ws_rows, act_rows, summ_rows = [], [], []
    for stage in stages:
        dpath = paths.delta_file(stage)
        if not dpath.exists():
            log(f"[{stage}] no deltas at {dpath}; skipping")
            continue
        deltas = torch.load(dpath)
        cond = condition_name(cfg.run_name, stage)
        for l in cfg.diagnostics.layers:
            dW = deltas[wkey(cfg, l)].to(device).float()
            W = W0[l]
            Uk, Vk = svd[l]
            d2 = dW.pow(2).sum()
            rho = (Uk.T @ dW).pow(2).sum() / d2  # ||U_k U_k^T dW||^2 = ||U_k^T dW||^2
            rho_in = (dW @ Vk).pow(2).sum() / d2
            eta = dW.norm() / W.norm()
            ws_rows.append({"run": cfg.run_name, "stage": stage, "condition": cond, "layer": l,
                            "module": cfg.diagnostics.module, "k": k,
                            "rho_k": rho.item(), "rho_k_input_side": rho_in.item(),
                            "rho_chance_out": k / W.shape[0], "rho_chance_in": k / W.shape[1],
                            "eta": eta.item(), "dW_fro": dW.norm().item(),
                            "W0_fro": W.norm().item()})
            Z = probe["z"][l].to(device)  # (n_prompts, d_in)
            num = (Z @ dW.T).norm(dim=1)
            den = (Z @ W.T).norm(dim=1) + eps
            R = (num / den).cpu().numpy()
            for i, r in enumerate(R):
                act_rows.append({"run": cfg.run_name, "stage": stage, "condition": cond,
                                 "layer": l, "group": probe["groups"][i],
                                 "task": probe["tasks"][i], "doc_id": probe["doc_ids"][i],
                                 "R": float(r), "dWz_norm": float(num[i]),
                                 "W0z_norm": float(den[i] - eps)})
            for g in ["math", "nonmath"]:
                v = R[groups == g]
                summ_rows.append({"run": cfg.run_name, "stage": stage, "condition": cond,
                                  "layer": l, "group": g, "n": len(v), "R_mean": float(v.mean()),
                                  "R_median": float(np.median(v)), "R_std": float(v.std(ddof=1))})
            log(f"[{cond}] layer {l}: rho_k={rho.item():.4f} (chance {k / W.shape[0]:.3f}) "
                f"eta={eta.item():.5f} R_math={R[groups == 'math'].mean():.5f} "
                f"R_nonmath={R[groups == 'nonmath'].mean():.5f}")

    out = paths.diag_out
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [(f"weight_space_{cfg.run_name}.csv", ws_rows),
                       (f"activation_{cfg.run_name}.csv", act_rows),
                       (f"activation_summary_{cfg.run_name}.csv", summ_rows)]:
        if rows:
            with open(out / name, "w", newline="") as f:
                w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    write_json(out / "probe_check.json", probe_prompt_check(cfg, paths))
    log("diagnostics done")


if __name__ == "__main__":
    main()
