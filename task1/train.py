"""Supervised fine-tuning of Qwen3-0.6B on the selected MetaMathQA examples.

    python -m task1.train --config task1/configs/full_ft.yaml

Implements the shared recipe from the manual: one epoch, fixed source order, micro-batch 2 x
accumulation 16, AdamW, cosine schedule with warmup, global-norm clipping, FP16 autocast with
dynamic loss scaling and FP32 master weights, gradient checkpointing. The loss is the summed
token cross-entropy over response tokens (+<|im_end|>) of the whole effective batch divided by
the number of such tokens.

Outputs
  runs/<run>/train_log.csv           per-update loss, lr, grad-norm, loss scale, skips, timing
  runs/<run>/train_summary.json      resources (time, peak memory, params, checkpoint sizes)
  runs/<run>/displacement_<stage>.json  ||theta - theta0||_2 (total + per tensor)
  scratch: recovery checkpoints; artifacts (kept, not zipped): FP16 exports, FP32 down_proj deltas
"""

from __future__ import annotations

import csv
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .common import (Paths, Timer, base_argparser, dir_size_bytes, load_config, log, read_json,
                     resolve_model_path, save_run_config, seed_everything, write_json)
from .data import TrainData, collate

LOG_FIELDS = ["update", "examples_seen", "sft_loss", "lr", "grad_norm", "loss_scale", "skipped",
              "resp_tokens", "seq_tokens", "update_time_s", "train_time_s", "peak_mem_gb",
              # L2-SP only (empty for other methods): Omega, lambda*Omega, SFT + lambda*Omega,
              # norm of the SFT gradient and of the penalty gradient lambda*(theta - theta0)
              "omega", "lambda_omega", "total_loss", "sft_grad_norm", "reg_grad_norm"]


# ----------------------------------------------------------------------------- model pieces
def decoder_and_head(model):
    """Return (decoder stack, lm_head) for a plain HF causal LM or a PEFT-wrapped one."""
    m = model
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    return m.model, m.lm_head


def _chunk_ce(h, t, w):
    logits = F.linear(h, w)  # FP16 under autocast
    return F.cross_entropy(logits.float(), t, reduction="sum")


def sft_loss_sum(model, ids, att, lab, chunk_tokens: int):
    """Summed next-token CE over labelled positions. The LM head is applied only to response
    positions, chunk-wise under activation checkpointing, so full-vocabulary logits for the
    whole sequence are never materialised (memory only; the loss is unchanged)."""
    decoder, head = decoder_and_head(model)
    hidden = decoder(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    tgt = lab[:, 1:]
    mask = tgt.ne(-100)
    h = hidden[:, :-1][mask]
    t = tgt[mask]
    total = hidden.new_zeros((), dtype=torch.float32)
    for hs, ts in zip(h.split(chunk_tokens), t.split(chunk_tokens)):
        total = total + checkpoint(_chunk_ce, hs, ts, head.weight, use_reentrant=False)
    return total, int(mask.sum())


def param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or n.endswith(".bias")) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


# ----------------------------------------------------------------------------- methods
class Method:
    """Per-method hooks. Full FT is the plain case; later methods override these."""

    name = "full_ft"

    def __init__(self, cfg):
        self.cfg = cfg

    def load_model(self, model_path: str, resume: Path | None):
        """Trainable model. Full FT: FP32 master weights (from the recovery checkpoint if any)."""
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(str(resume) if resume else model_path,
                                                    dtype=torch.float32, attn_implementation="sdpa")

    def save_model(self, model, out_dir: Path):
        model.save_pretrained(out_dir, safe_serialization=True)

    def displacement(self, model, theta0) -> dict:
        return displacement(model, theta0)

    def deltas(self, model, theta0, keys) -> dict:
        params = dict(model.named_parameters())
        return {k: params[k].detach().float().cpu() - theta0[k] for k in keys}

    def export(self, model, tok, theta0, model_path: str, out_dir: Path) -> int:
        return export_fp16(model, tok, out_dir)

    def after_export(self, model, tok, data, out_dir: Path, run_out: Path, stage: str, device):
        """Checks that need the trained model and its export (adapters: merged vs. unmerged)."""

    def setup(self, model, theta0: dict, device) -> None:
        """Called once after the model is on the device and theta0 is available."""

    def after_unscale(self, model) -> dict:
        """Called every update after gradients are unscaled to FP32 and before clipping.
        May modify .grad; returns extra values for the training log."""
        return {}

    def extra_resources(self) -> dict:
        return {}


class L2SP(Method):
    """Full FT + L2-SP: objective L_SFT + lambda * Omega, Omega = 1/2 sum_j (theta_j - theta0_j)^2
    over all unique parameters (biases and norms included, tied embedding counted once), summed
    not averaged, theta0 fixed at the starting checkpoint.

    The penalty gradient lambda * (theta - theta0) is added analytically to the unscaled FP32
    gradients before global-norm clipping. This is identical to back-propagating lambda * Omega but
    keeps the penalty out of FP16 loss scaling; clipping and AdamW therefore see the gradient of
    the total objective. Omega is evaluated at the same parameters as the SFT loss of the update.

    Reference storage: theta0 is kept on the GPU. Qwen3-0.6B is distributed in BF16, so its FP32
    upcast is exactly representable in BF16; when that holds for every tensor the reference is
    stored in BF16 (half the memory, bit-exact), otherwise in FP32. `l2sp_ref_storage: cpu` keeps
    an FP32 copy in pinned host memory and streams it to the GPU each update instead.
    """

    name = "l2sp"
    CHUNK = 1 << 24  # elements per temporary (bounds transient memory on the embedding)

    def setup(self, model, theta0, device):
        tc = self.cfg.train
        self.lam = float(tc.l2sp_lambda)
        assert self.lam > 0, "l2sp requires train.l2sp_lambda > 0"
        mode = str(tc.get("l2sp_ref_storage", "auto"))
        t = time.perf_counter()
        self.params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        names = [n for n, _ in self.params]
        assert set(names) == set(theta0), "theta0 / model parameter names differ"
        exact_bf16 = all(torch.equal(theta0[n].to(torch.bfloat16).float(), theta0[n]) for n in names)
        if mode == "auto":
            mode = "gpu_bf16" if exact_bf16 else "gpu_fp32"
        if mode == "gpu_bf16" and not exact_bf16:
            raise ValueError("theta0 is not exactly representable in BF16; use gpu_fp32")
        self.mode = mode
        if mode == "cpu":
            self.ref = [theta0[n].contiguous().pin_memory() if device.type == "cuda" else theta0[n]
                        for n in names]
        else:
            dt = torch.bfloat16 if mode == "gpu_bf16" else torch.float32
            self.ref = [theta0[n].to(device=device, dtype=dt) for n in names]
        if device.type == "cuda":
            torch.cuda.synchronize()
        self.setup_s = time.perf_counter() - t
        self.ref_bytes = sum(r.numel() * r.element_size() for r in self.ref)
        self.h2d_bytes_setup = 0 if mode == "cpu" else self.ref_bytes
        self.h2d_bytes_per_update = self.ref_bytes if mode == "cpu" else 0
        self.n_ref = sum(r.numel() for r in self.ref)
        self.penalty_time = 0.0
        self.n_calls = 0
        self.device = device
        log(f"L2-SP: lambda={self.lam:g}, reference {mode}, {self.ref_bytes / 2**30:.2f} GiB, "
            f"{self.n_ref:,} parameters, setup {self.setup_s:.1f}s (bf16-exact: {exact_bf16})")

    @torch.no_grad()
    def after_unscale(self, model):
        sync = self.device.type == "cuda"
        if sync:
            torch.cuda.synchronize()
        t = time.perf_counter()
        sq = torch.zeros((), device=self.device, dtype=torch.float64)
        g_sft_sq = torch.zeros((), device=self.device, dtype=torch.float64)
        for (_, p), ref in zip(self.params, self.ref):
            if p.grad is None:  # parameter unused in this update: gradient is the penalty alone
                p.grad = torch.zeros_like(p)
            pf, gf, rf = p.data.view(-1), p.grad.view(-1), ref.view(-1)
            for s in range(0, pf.numel(), self.CHUNK):
                e = s + self.CHUNK
                g_sft_sq += gf[s:e].pow(2).sum(dtype=torch.float64)
                d = pf[s:e] - rf[s:e].to(self.device, non_blocking=True).float()
                sq += d.pow(2).sum(dtype=torch.float64)
                gf[s:e].add_(d, alpha=self.lam)
        omega = 0.5 * sq.item()
        if sync:
            torch.cuda.synchronize()
        self.penalty_time += time.perf_counter() - t
        self.n_calls += 1
        return {"omega": omega, "lambda_omega": self.lam * omega,
                "sft_grad_norm": math.sqrt(g_sft_sq.item()),
                "reg_grad_norm": self.lam * math.sqrt(2.0 * omega)}

    def extra_resources(self):
        return {"overhead_ref_storage": self.mode, "overhead_ref_bytes": self.ref_bytes,
                "overhead_ref_setup_s": self.setup_s,
                "overhead_h2d_bytes_setup": self.h2d_bytes_setup,
                "overhead_h2d_bytes_per_update": self.h2d_bytes_per_update,
                "overhead_penalty_time_s": self.penalty_time,
                "overhead_penalty_ms_per_update": 1000 * self.penalty_time / max(self.n_calls, 1),
                "overhead_penalty_note": "time to compute Omega and add lambda*(theta-theta0) to the "
                                         "gradients (synchronised); included in train_time_s"}


class Adapter(Method):
    """LoRA / OPLoRA (task1/adapters.py): frozen FP16 backbone, FP32 factors on every
    q/k/v/o/gate/up/down projection, dropout on the adapter input, no adapter biases.
    Recovery checkpoints hold only the factors (+ optimizer etc.). The inference export merges
    dW = (alpha/r) B~ A~ (projected factors for OPLoRA) into a fresh FP32 copy of the starting
    weights and saves it in FP16, together with the exported factors."""

    name = "lora"

    def __init__(self, cfg):
        super().__init__(cfg)
        a = cfg.adapter
        self.r, self.alpha, self.dropout = int(a.r), float(a.alpha), float(a.dropout)
        self.k = int(a.protect_rank) if a.get("protect_rank") else 0
        self.basis_stats, self.export_s = {}, {}
        self.adapter_bytes = None

    def load_model(self, model_path, resume):
        from transformers import AutoModelForCausalLM
        from .adapters import inject, load_adapter_state
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16,
                                                     attn_implementation="sdpa")
        self.wrapped = inject(model, self.r, self.alpha, self.dropout,
                              int(self.cfg.adapter.get("init_seed", self.cfg.seed)))
        if resume:
            load_adapter_state(self.wrapped, torch.load(resume / "adapter.pt"))
        return model

    def save_model(self, model, out_dir):
        from .adapters import adapter_state
        torch.save(adapter_state(self.wrapped), out_dir / "adapter.pt")

    def setup(self, model, theta0, device):
        self.device = device
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.adapter_params = n
        log(f"{self.cfg.method}: r={self.r} alpha={self.alpha:g} dropout={self.dropout} on "
            f"{len(self.wrapped)} projections, {n:,} adapter parameters"
            + (f", protection rank k={self.k}" if self.k else ""))
        if self.k:
            from .adapters import attach_bases
            self.basis_stats = attach_bases(self.wrapped, theta0, self.k, device)
            log(f"OPLoRA bases: {self.basis_stats['n_bases']} x (U_k, V_k), "
                f"{self.basis_stats['basis_bytes'] / 2**20:.1f} MiB FP32 on {device.type}, "
                f"SVD preprocessing {self.basis_stats['basis_preprocess_s']:.1f}s")

    @torch.no_grad()
    def displacement(self, model, theta0):
        per = {key: m.delta_weight().pow(2).sum().item() for key, m in self.wrapped.items()}
        total = sum(per.values())
        return {"l2": math.sqrt(total), "sq_total": total, "n_tensors": len(per),
                "per_tensor_sq": per,
                "note": "merged-weight displacement: only adapted projections change"}

    @torch.no_grad()
    def deltas(self, model, theta0, keys):
        return {k: self.wrapped[k].delta_weight().cpu() for k in keys}

    @torch.no_grad()
    def export(self, model, tok, theta0, model_path, out_dir):
        from transformers import AutoModelForCausalLM
        from .adapters import effective_state
        sync = self.device.type == "cuda"
        if sync:
            torch.cuda.synchronize()
        t = time.perf_counter()
        fresh = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
        fp = dict(fresh.named_parameters())
        for key, m in self.wrapped.items():  # fresh FP32 starting weights + dW, then FP16
            assert torch.equal(fp[key].data, theta0[key]), f"fresh backbone differs at {key}"
            fp[key].data.add_(m.delta_weight().cpu())
        size = export_fp16(fresh, tok, out_dir)
        fac = effective_state(self.wrapped)
        torch.save(fac | {"_meta": {"r": self.r, "alpha": self.alpha, "scale": self.alpha / self.r,
                                    "method": self.cfg.method, "projected": bool(self.k),
                                    "protect_rank": self.k}}, out_dir / "adapter_factors.pt")
        self.adapter_bytes = sum(v.numel() * v.element_size() for v in fac.values())
        del fresh
        self.export_s[out_dir.name] = time.perf_counter() - t
        return size

    @torch.no_grad()
    def after_export(self, model, tok, data, out_dir, run_out, stage, device):
        from .adapters import subspace_report
        rows = subspace_report(self.wrapped, self._theta0, self.k or
                               int(self.cfg.diagnostics.protect_rank), device)
        _write_csv(run_out / f"adapter_subspace_{stage}.csv", rows)
        summ = {"k": rows[0]["k"], "projected": bool(self.k),
                "rho_out_max": max(r["rho_out"] for r in rows),
                "rho_out_mean": sum(r["rho_out"] for r in rows) / len(rows),
                "rho_in_max": max(r["rho_in"] for r in rows),
                "rho_in_mean": sum(r["rho_in"] for r in rows) / len(rows),
                "B_leak_max": max(r["B_leak"] for r in rows),
                "A_leak_max": max(r["A_leak"] for r in rows),
                "Uk_orth_err_max": max(r["Uk_orth_err"] for r in rows),
                "Vk_orth_err_max": max(r["Vk_orth_err"] for r in rows),
                "note": "rho_out = ||U_k^T dW||_F^2/||dW||_F^2, rho_in = ||dW V_k||_F^2/||dW||_F^2 "
                        "over all adapted projections; OPLoRA constraints require both ~0 "
                        "(FP32 round-off). For LoRA the bases are computed only for this report."}
        write_json(run_out / f"adapter_subspace_{stage}.json", summ)
        log(f"[{stage}] subspace check: max rho_out {summ['rho_out_max']:.3e}, "
            f"max rho_in {summ['rho_in_max']:.3e} (k={summ['k']})")
        write_json(run_out / f"merge_check_{stage}.json",
                   merge_check(model, out_dir, data, tok, device, self._model_path))

    def extra_resources(self):
        out = {"overhead_adapter_params": self.adapter_params,
               "overhead_adapter_bytes_fp32": self.adapter_bytes,
               "overhead_export_merge_s": self.export_s.get("final"),
               "overhead_export_note": "merge dW into a fresh FP32 backbone copy + FP16 save "
                                       "(projected factors for OPLoRA)"}
        if self.k:
            out |= {"overhead_basis_storage": self.basis_stats.get("basis_storage"),
                    "overhead_basis_bytes": self.basis_stats.get("basis_bytes"),
                    "overhead_basis_preprocess_s": self.basis_stats.get("basis_preprocess_s"),
                    "overhead_protect_rank": self.k}
        return out


def _write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)


@torch.no_grad()
def merge_check(model, export_dir: Path, data, tok, device, model_path: str,
                n_examples: int = 4, max_len: int = 256) -> dict:
    """Unmerged (FP16 backbone + FP32 adapters, autocast) vs. merged FP16 export logits on a
    small batch of training sequences. The adapter effect (unmerged vs. starting model) is
    reported as a scale for the merge error."""
    from transformers import AutoModelForCausalLM
    was_training = model.training
    model.eval()
    ex = []
    for i in range(n_examples):  # (prompt, response) truncated to max_len tokens in total
        p, r = data.example(i)
        p = p[:max_len - 1]
        ex.append((p, r[:max_len - len(p)]))
    ids, att, _ = collate(ex, tok.pad_token_id)
    ids, att = ids.to(device), att.to(device)
    amp = device.type == "cuda"
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        lu = model(input_ids=ids, attention_mask=att, use_cache=False).logits.float()
    dt = torch.float16 if amp else torch.float32
    out = {}
    for name, path in [("merged", export_dir), ("base", model_path)]:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=dt, attn_implementation="sdpa").to(device).eval()
        lo = m(input_ids=ids, attention_mask=att, use_cache=False).logits.float()
        del m
        mask = att.bool()
        d = (lu - lo).abs()[mask]
        out[name] = {"max_abs_diff": d.max().item(), "mean_abs_diff": d.mean().item(),
                     "top1_agreement": (lu.argmax(-1) == lo.argmax(-1))[mask].float().mean().item(),
                     "max_logprob_diff": (F.log_softmax(lu, -1) - F.log_softmax(lo, -1))
                     .abs()[mask].max().item()}
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.train(was_training)
    res = {"n_sequences": n_examples, "max_len": max_len, "n_positions": int(att.sum()),
           "unmerged": "FP16 backbone + FP32 adapters under FP16 autocast (training model, eval mode)",
           "merged": "FP16 export (W0 + dW merged in FP32, then cast), as used for evaluation",
           "merged_vs_unmerged": out["merged"], "adapter_effect_unmerged_vs_base": out["base"]}
    ok = out["merged"]["mean_abs_diff"] < 0.1 * max(out["base"]["mean_abs_diff"], 1e-6)
    res["merge_error_small_vs_adapter_effect"] = bool(ok)
    log(f"merge check: merged vs unmerged mean|Δlogit| {out['merged']['mean_abs_diff']:.4f} "
        f"(top-1 agree {out['merged']['top1_agreement']:.4f}); adapter effect "
        f"{out['base']['mean_abs_diff']:.4f}")
    return res


def get_method(cfg) -> Method:
    if cfg.method == "full_ft":
        return Method(cfg)
    if cfg.method == "l2sp":
        return L2SP(cfg)
    if cfg.method in ("lora", "oplora"):
        return Adapter(cfg)
    raise NotImplementedError(f"method {cfg.method!r} not implemented")


# ----------------------------------------------------------------------------- weight-space
@torch.no_grad()
def displacement(model, theta0: dict[str, torch.Tensor]) -> dict:
    """||theta - theta0||_2 over all unique parameters (tied embeddings counted once)."""
    per, total = {}, 0.0
    for n, p in model.named_parameters():  # named_parameters() de-duplicates tied tensors
        d = (p.detach().float() - theta0[n].to(p.device, non_blocking=True)).pow(2).sum().item()
        per[n] = d
        total += d
    return {"l2": math.sqrt(total), "sq_total": total, "n_tensors": len(per),
            "per_tensor_sq": per}


@torch.no_grad()
def save_deltas(method, model, theta0, cfg, path: Path):
    """FP32 dW of the diagnostic projections (Full FT: W - W0; adapters: (alpha/r) B~ A~)."""
    mod = cfg.diagnostics.module
    keys = [f"model.layers.{layer}.mlp.{mod}.weight" for layer in cfg.diagnostics.layers]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(method.deltas(model, theta0, keys), path)


@torch.no_grad()
def export_fp16(model, tok, out_dir: Path) -> int:
    """Inference export: FP16 HF checkpoint (tied lm_head stored once)."""
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tied = getattr(model.config, "tie_word_embeddings", False)
    sd = {k: v.detach().to("cpu", torch.float16) for k, v in model.state_dict().items()
          if not (tied and k == "lm_head.weight")}
    model.save_pretrained(tmp, state_dict=sd, safe_serialization=True)
    tok.save_pretrained(tmp)
    cfg_path = tmp / "config.json"
    import json
    c = json.loads(cfg_path.read_text())
    c["torch_dtype"] = "float16"; c["dtype"] = "float16"; c["use_cache"] = True
    cfg_path.write_text(json.dumps(c, indent=2))
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)
    size = dir_size_bytes(out_dir)
    write_json(out_dir / "EXPORT_DONE.json", {"bytes": size, "time": time.time()})
    return size


# ----------------------------------------------------------------------------- checkpoints
def save_checkpoint(paths: Paths, update: int, method, model, opt, sched, scaler, state: dict,
                    keep_last: int) -> int:
    d = paths.ckpt_dir / f"update_{update:04d}"
    tmp = d.with_name(d.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    method.save_model(model, tmp)
    torch.save(opt.state_dict(), tmp / "optimizer.pt")
    torch.save(sched.state_dict(), tmp / "scheduler.pt")
    torch.save(scaler.state_dict(), tmp / "scaler.pt")
    torch.save({"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               tmp / "rng.pt")
    write_json(tmp / "trainer_state.json", state)
    size = dir_size_bytes(tmp)
    state = state | {"checkpoint_bytes": {**state.get("checkpoint_bytes", {}), str(update): size}}
    write_json(tmp / "trainer_state.json", state)
    shutil.rmtree(d, ignore_errors=True)
    tmp.rename(d)
    ckpts = sorted(p for p in paths.ckpt_dir.glob("update_*") if not p.name.endswith(".tmp"))
    for old in ckpts[:-keep_last] if keep_last > 0 else []:
        shutil.rmtree(old, ignore_errors=True)
        log(f"pruned recovery checkpoint {old.name}")
    return size


def latest_checkpoint(paths: Paths) -> Path | None:
    if not paths.ckpt_dir.exists():
        return None
    ok = [p for p in paths.ckpt_dir.glob("update_*")
          if not p.name.endswith(".tmp") and (p / "trainer_state.json").exists()]
    return max(ok, key=lambda p: int(p.name.split("_")[1])) if ok else None


def training_complete(paths: Paths, stages) -> bool:
    return (paths.run_out / "train_summary.json").exists() and all(
        (paths.export_dir(s) / "EXPORT_DONE.json").exists() for s in stages)


# ----------------------------------------------------------------------------- main loop
def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    paths = Paths(cfg)
    tc = cfg.train
    seed_everything(int(cfg.seed))
    paths.run_out.mkdir(parents=True, exist_ok=True)
    stages_needed = list(cfg.get("stages", ["final"]))
    if not args.no_resume and training_complete(paths, stages_needed):
        log(f"training already complete (train_summary.json and FP16 exports {stages_needed} "
            f"present, e.g. imported from an earlier notebook output); skipping")
        return
    save_run_config(cfg, paths.run_out / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model_path, _ = resolve_model_path(cfg)

    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    tok = AutoTokenizer.from_pretrained(model_path)
    data = TrainData(paths)
    mb, accum, n_upd = int(tc.micro_batch_size), int(tc.grad_accum), int(tc.num_updates)
    eff = mb * accum
    assert len(data) >= eff * n_upd, f"need {eff * n_upd} examples, have {len(data)}"
    log(f"data: {len(data)} examples, sha256={data.sha256[:12]}; {n_upd} updates x {eff}")

    resume = None if args.no_resume else latest_checkpoint(paths)
    method = get_method(cfg)
    model = method.load_model(model_path, resume)
    model.config.use_cache = False
    if tc.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    model.train()

    # theta0 (FP32, CPU) for displacement and diagnostics deltas. Always from the base model.
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    theta0 = {n: p.detach().clone() for n, p in base.named_parameters()}
    del base
    method._theta0, method._model_path = theta0, model_path
    method.setup(model, theta0, device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"params: trainable {n_trainable:,} / total {n_total:,}")

    opt = torch.optim.AdamW(param_groups(model, float(tc.weight_decay)), lr=float(tc.lr),
                            betas=tuple(tc.betas), eps=float(tc.eps),
                            fused=device.type == "cuda")
    sched = get_cosine_schedule_with_warmup(opt, int(tc.warmup_updates), n_upd)
    scaler = torch.amp.GradScaler("cuda", init_scale=float(tc.init_loss_scale), enabled=use_amp)

    start_update, train_time, ckpt_sizes, n_skipped = 0, 0.0, {}, 0
    log_path = paths.run_out / "train_log.csv"
    if resume:
        st = read_json(resume / "trainer_state.json")
        assert st["data_sha256"] == data.sha256, "training data changed since checkpoint"
        opt.load_state_dict(torch.load(resume / "optimizer.pt", map_location=device))
        sched.load_state_dict(torch.load(resume / "scheduler.pt"))
        scaler.load_state_dict(torch.load(resume / "scaler.pt"))
        rng = torch.load(resume / "rng.pt", weights_only=False)
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        start_update, train_time = st["update"], st["train_time_s"]
        ckpt_sizes, n_skipped = st.get("checkpoint_bytes", {}), st.get("n_skipped", 0)
        log(f"resumed from {resume.name} (update {start_update})")
        # keep only log rows up to the checkpoint
        if log_path.exists():
            rows = list(csv.DictReader(open(log_path)))
            rows = [r for r in rows if int(r["update"]) <= start_update]
            with open(log_path, "w", newline="") as f:
                w = csv.DictWriter(f, LOG_FIELDS); w.writeheader(); w.writerows(rows)
    if not log_path.exists() or start_update == 0:
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, LOG_FIELDS).writeheader()

    params = [p for p in model.parameters() if p.requires_grad]
    pad_id = tok.pad_token_id
    ckpt_set = set(int(u) for u in tc.checkpoint_updates)
    mid = int(tc.midpoint_update)
    stage_at = {mid: "mid", n_upd: "final"}
    stages = set(cfg.get("stages", ["final"]))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    logf = open(log_path, "a", newline="")
    writer = csv.DictWriter(logf, LOG_FIELDS)
    for update in range(start_update, n_upd):
        t0 = time.perf_counter()
        first = update * eff
        idx = list(range(first, first + eff))
        n_resp = sum(data.resp_len(i) for i in idx)
        n_seq = sum(data.total_len(i) for i in idx)
        loss_sum = torch.zeros((), device=device)
        n_lab_total = 0
        for m in range(accum):
            ids, att, lab = collate([data.example(i) for i in idx[m * mb:(m + 1) * mb]], pad_id)
            ids, att, lab = ids.to(device), att.to(device), lab.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                s, n_lab = sft_loss_sum(model, ids, att, lab, int(tc.ce_chunk_tokens))
            scaler.scale(s / n_resp).backward()
            loss_sum += s.detach()
            n_lab_total += n_lab
        assert n_lab_total == n_resp, f"label count {n_lab_total} != response tokens {n_resp}"
        scaler.unscale_(opt)
        extra = method.after_unscale(model)  # L2-SP: adds lambda*(theta - theta0) to the grads
        gnorm = torch.nn.utils.clip_grad_norm_(params, float(tc.grad_clip))
        scale_before = scaler.get_scale()
        lr_used = sched.get_last_lr()[0]
        scaler.step(opt)
        scaler.update()
        skipped = use_amp and scaler.get_scale() < scale_before
        n_skipped += int(skipped)
        sched.step()
        opt.zero_grad(set_to_none=True)
        sft = (loss_sum / n_resp).item()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        train_time += dt
        u = update + 1
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        writer.writerow({"update": u, "examples_seen": u * eff, "sft_loss": f"{sft:.6f}",
                         "lr": f"{lr_used:.6e}", "grad_norm": f"{float(gnorm):.6f}",
                         "loss_scale": scale_before, "skipped": int(skipped),
                         "resp_tokens": n_resp, "seq_tokens": n_seq,
                         "update_time_s": f"{dt:.3f}", "train_time_s": f"{train_time:.1f}",
                         "peak_mem_gb": f"{peak:.3f}"}
                        | ({"omega": f"{extra['omega']:.6e}",
                            "lambda_omega": f"{extra['lambda_omega']:.6e}",
                            "total_loss": f"{sft + extra['lambda_omega']:.6f}",
                            "sft_grad_norm": f"{extra['sft_grad_norm']:.6f}",
                            "reg_grad_norm": f"{extra['reg_grad_norm']:.6e}"} if extra else {}))
        logf.flush()
        if u == 1 or u % 25 == 0 or skipped:
            eta = (n_upd - u) * train_time / max(u - start_update, 1)
            reg_txt = (f" Ω {extra['omega']:.3e} λΩ {extra['lambda_omega']:.3e}" if extra else "")
            log(f"update {u}/{n_upd} loss {sft:.4f}{reg_txt} lr {lr_used:.2e} gnorm {float(gnorm):.3f} "
                f"scale {scale_before:.0f}{' SKIPPED' if skipped else ''} "
                f"{dt:.2f}s/upd peak {peak:.2f}GB eta {eta / 60:.0f}min")
        if not math.isfinite(sft):
            raise RuntimeError(f"non-finite SFT loss at update {u}")

        if u in stage_at and stage_at[u] in stages:
            stage = stage_at[u]
            disp = method.displacement(model, theta0)
            write_json(paths.run_out / f"displacement_{stage}.json",
                       {"update": u, "examples_seen": u * eff} | disp)
            save_deltas(method, model, theta0, cfg, paths.delta_file(stage))
            size = method.export(model, tok, theta0, model_path, paths.export_dir(stage))
            method.after_export(model, tok, data, paths.export_dir(stage), paths.run_out, stage,
                                device)
            write_json(paths.run_out / f"export_{stage}.json", {"update": u, "bytes": size})
            log(f"[{stage}] ||theta-theta0||_2 = {disp['l2']:.4f}; FP16 export {size / 2**30:.2f} GiB")
        if u in ckpt_set:
            st = {"update": u, "examples_seen": u * eff, "train_time_s": train_time,
                  "data_sha256": data.sha256, "n_skipped": n_skipped,
                  "checkpoint_bytes": dict(ckpt_sizes)}
            size = save_checkpoint(paths, u, method, model, opt, sched, scaler, st,
                                   int(tc.keep_last_checkpoints))
            ckpt_sizes[str(u)] = size
            log(f"recovery checkpoint @ update {u} ({u * eff} examples): {size / 2**30:.2f} GiB")
    logf.close()

    summary = {
        "run_name": cfg.run_name, "method": cfg.method, "updates": n_upd,
        "examples_seen": n_upd * eff, "train_time_s": train_time,
        "train_time_note": "sum of per-update wall time (forward/backward/optimizer); excludes "
                           "checkpointing, exports and evaluation",
        "peak_mem_allocated_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        "peak_mem_reserved_gb": torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "trainable_params": n_trainable, "total_params": n_total,
        "skipped_updates": n_skipped,
        "recovery_checkpoint_bytes": ckpt_sizes,
        "export_bytes": {s: read_json(paths.run_out / f"export_{s}.json")["bytes"]
                         for s in ["mid", "final"] if (paths.run_out / f"export_{s}.json").exists()},
        "data_sha256": data.sha256, "lr": float(tc.lr), "l2sp_lambda": float(tc.l2sp_lambda),
        "amp": "fp16 autocast + dynamic loss scaling, fp32 master weights" if use_amp else "fp32 (cpu)",
    } | method.extra_resources()
    if resume is not None:
        summary["resumed_from"] = resume.name
    write_json(paths.run_out / "train_summary.json", summary)
    log(f"training done: {train_time / 60:.1f} min, skipped {n_skipped} updates")


if __name__ == "__main__":
    main()
