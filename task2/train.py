"""Train the SigLIP-Qwen VLM through the three-stage curriculum (one continuous run).

    python -m task2.train --config task2/configs/vlm196.yaml

Stage 1 caption alignment (connector only, two passes over the 18,000 caption examples),
Stage 2 visual instruction tuning (connector + new LoRA r16 on q/k/v/o; presence 50%, counting 25%,
caption replay 25%), Stage 3 natural VQA (VQAv2 70%, presence / counting / caption replay 10% each).
Each draw contributes two examples (a complete question pair, or two caption examples).
Micro-batch 8 x accumulation 2 (8 draws per update); AdamW (wd 0.01, not on biases); per-stage
5% linear warmup then cosine decay to 10% of peak (fresh optimizer per stage, since the trainable
set and learning rates change); global-norm clipping 1.0; FP16 autocast with dynamic loss scaling;
frozen FP16 backbone, FP32 connector / adapters. Loss: mean token NLL within each example
(target + <|im_end|>), then mean over the 16 examples of the update.

Outputs (runs/<run>/): train_log.csv, dev_metrics.jsonl, calibration.json, sanity_checks.json,
train_summary.json; artifacts: checkpoints every 250 updates (resumable: model, optimizer,
scheduler, scaler, sampler, stage progress, RNG) and connector/adapter weights after each stage.
"""

from __future__ import annotations

import csv
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .common import (Cfg, Paths, Timer, base_argparser, env_info, load_config, log, read_json,
                     read_jsonl, save_run_config, seed_everything, write_json, write_jsonl)
from .model import VLM, FeatureStore, pool_features

COMPONENTS = ["caption", "presence", "counting", "vqa"]
LOG_FIELDS = (["update", "stage", "stage_update", "loss", "lr_connector", "lr_lora", "grad_norm",
               "loss_scale", "skipped", "n_target_tokens", "update_time_s", "train_time_s", "peak_mem_gb"]
              + [f"n_{c}" for c in COMPONENTS])


# ----------------------------------------------------------------------------- data
class Pools:
    def __init__(self, paths: Paths):
        d = paths.data_out
        self.caption = [{"image": r["image"], "question": None, "target": r["caption"], "kind": "caption",
                         "component": "caption"} for r in read_jsonl(d / "captions_train.jsonl")]
        self.pairs = {"presence": read_jsonl(d / "grounding_train_presence.jsonl"),
                      "counting": read_jsonl(d / "grounding_train_counting.jsonl"),
                      "vqa": read_jsonl(d / "vqa_train.jsonl")}
        self.dev_captions = [{"image": r["image"], "question": None, "target": r["caption"],
                              "kind": "caption"} for r in read_jsonl(d / "captions_dev.jsonl")]
        self.dev_pairs = {"grounding": read_jsonl(d / "grounding_dev.jsonl"),
                          "vqa": read_jsonl(d / "vqa_dev.jsonl")}

    @staticmethod
    def pair_examples(p, component):
        return [{"image": it["image"], "question": p["question"], "target": it["answer"],
                 "kind": "answer", "component": component} for it in p["items"]]


class Sampler:
    """Stage 1: two fixed passes (each a seeded permutation) over the caption examples.
    Stages 2-3: per draw a seeded component choice, then a uniformly drawn pair (or two caption
    examples). The generator state is saved in checkpoints, so resuming continues the stream."""

    def __init__(self, pools: Pools, seed: int, stage_idx: int, stage_cfg, n_draws: int):
        self.pools, self.seed, self.k, self.n_draws = pools, seed, stage_idx, n_draws
        self.mix = {c: float(v) for c, v in dict(stage_cfg.mixture).items()}
        assert abs(sum(self.mix.values()) - 1) < 1e-9
        self.rng = np.random.default_rng([seed, stage_idx])
        self._perm = {}

    def _caption_perm(self, p):
        if p not in self._perm:
            self._perm[p] = np.random.default_rng([self.seed, 0, p]).permutation(len(self.pools.caption))
        return self._perm[p]

    def batch(self, stage_update: int) -> list[dict]:
        per = 2 * self.n_draws
        if self.k == 0:
            n = len(self.pools.caption)
            out = []
            for j in range(stage_update * per, (stage_update + 1) * per):
                out.append(self.pools.caption[int(self._caption_perm(j // n)[j % n])])
            return out
        names = list(self.mix)
        probs = [self.mix[c] for c in names]
        out = []
        for _ in range(self.n_draws):
            c = names[int(self.rng.choice(len(names), p=probs))]
            if c == "caption":
                for _ in range(2):
                    out.append(self.pools.caption[int(self.rng.integers(len(self.pools.caption)))])
            else:
                pool = self.pools.pairs[c]
                out += Pools.pair_examples(pool[int(self.rng.integers(len(pool)))], c)
        return out

    def state(self):
        return self.rng.bit_generator.state

    def load(self, st):
        self.rng.bit_generator.state = st


# ----------------------------------------------------------------------------- dev evaluation
@torch.no_grad()
def dev_eval(model: VLM, pools: Pools, fs: FeatureStore, bs: int, amp: bool) -> dict:
    was = model.training
    model.eval()
    t = Timer()

    def losses(exs, use, mask_eot):
        out = []
        for s in range(0, len(exs), bs):
            chunk = exs[s:s + bs]
            feats = fs.get([e["image"] for e in chunk])
            with torch.autocast(model.device.type, dtype=torch.float16, enabled=amp):
                out += model.example_losses(chunk, feats, use[s:s + bs], mask_eot).tolist()
        return out

    res = {}
    caps = pools.dev_captions
    res["caption_loss"] = float(np.mean(losses(caps, [True] * len(caps), False)))
    for suite, pairs in pools.dev_pairs.items():
        own, swap, types = [], [], []
        for p in pairs:
            for m, it in enumerate(p["items"]):
                q = {"question": p["question"], "target": it["answer"], "kind": "answer"}
                own.append(q | {"image": it["image"]})
                swap.append(q | {"image": p["items"][1 - m]["image"]})
                types.append(p["qtype"])
        n = len(own)
        cor = np.array(losses(own, [True] * n, True))
        com = np.array(losses(swap, [True] * n, True))
        non = np.array(losses(own, [False] * n, True))
        types = np.array(types)
        res[f"{suite}_nll_correct"] = float(cor.mean())
        res[f"{suite}_nll_complementary"] = float(com.mean())
        res[f"{suite}_nll_no_image"] = float(non.mean())
        res[f"{suite}_delta_nll"] = float((com - cor).mean())
        for ty in sorted(set(types)):
            m = types == ty
            k = ty.replace("/", "")
            res[f"{suite}_{k}_nll_correct"] = float(cor[m].mean())
            res[f"{suite}_{k}_delta_nll"] = float((com[m] - cor[m]).mean())
    res["dev_eval_s"] = t.elapsed()
    model.train(was)
    return res


# ----------------------------------------------------------------------------- optimisation
def make_optimizer(model: VLM, stage_cfg, tc):
    groups = []
    for kind, lr in [("connector", stage_cfg.connector_lr), ("lora", stage_cfg.lora_lr)]:
        if lr is None:
            continue
        ps = [(n, p) for n, p in model.trainable() if (n.startswith("connector.") == (kind == "connector"))]
        dec = [p for n, p in ps if p.ndim >= 2 and not n.endswith(".bias")]
        nod = [p for n, p in ps if not (p.ndim >= 2 and not n.endswith(".bias"))]
        groups += [{"params": dec, "weight_decay": float(tc.weight_decay), "lr": float(lr), "name": kind},
                   {"params": nod, "weight_decay": 0.0, "lr": float(lr), "name": kind}]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, betas=tuple(tc.betas), eps=float(tc.eps),
                            fused=torch.cuda.is_available())
    T = int(stage_cfg.updates)
    W = max(1, int(round(float(tc.warmup_frac) * T)))
    lo = float(tc.final_lr_frac)

    def mult(s):  # s = updates already taken in this stage
        if s < W:
            return (s + 1) / W
        return lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - W) / max(1, T - W))))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, mult)


def set_trainable(model: VLM, stage_idx: int):
    for p in model.lm.parameters():
        p.requires_grad_(False)
    for p in model.connector.parameters():
        p.requires_grad_(True)
    if stage_idx >= 1:
        if model.wrapped is None:
            model.enable_lora()
        for m in model.wrapped.values():
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)


def trainable_summary(model: VLM) -> dict:
    tr = model.trainable()
    n_conn = sum(p.numel() for n, p in tr if n.startswith("connector."))
    n_lora = sum(p.numel() for n, p in tr if "lora_" in n)
    other = [n for n, _ in tr if not n.startswith("connector.") and "lora_" not in n]
    assert not other, f"unexpected trainable parameters: {other[:5]}"
    base_trainable = [n for n, p in model.lm.named_parameters() if p.requires_grad and "lora_" not in n]
    assert not base_trainable, "backbone parameters must stay frozen"
    return {"connector": n_conn, "lora": n_lora, "total": n_conn + n_lora,
            "frozen_backbone_params": sum(p.numel() for n, p in model.lm.named_parameters()
                                          if "lora_" not in n)}


# ----------------------------------------------------------------------------- checkpoints
def save_ckpt(paths: Paths, st: dict, keep: int):
    d = paths.ckpt_dir
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"ckpt_{st['global_update']:05d}.pt"
    tmp = f.with_suffix(".tmp")
    torch.save(st, tmp)
    tmp.replace(f)
    for old in sorted(d.glob("ckpt_*.pt"))[:-keep]:
        old.unlink()
    return f.stat().st_size


def latest_ckpt(paths: Paths) -> Path | None:
    c = sorted(paths.ckpt_dir.glob("ckpt_*.pt")) if paths.ckpt_dir.exists() else []
    return c[-1] if c else None


def import_prior_run(cfg, paths: Paths, sha: str):
    """Resume across Kaggle versions: copy the newest checkpoint (+ its logs) of this run from an
    attached earlier output, if it is ahead of the local state and uses the same data selection."""
    root = Path(cfg.paths.get("prior_search") or "/nonexistent")
    if not root.exists():
        return
    best = None
    for ck in root.rglob(f"runs/{cfg.run_name}/checkpoints/ckpt_*.pt"):
        art = ck.parents[3]           # .../task2_artifacts
        out = art.parent / art.name.replace("_artifacts", "")
        sc = out / "data" / "split_checks.json"
        if not sc.exists() or read_json(sc).get("selection_sha256") != sha:
            continue
        g = int(ck.stem.split("_")[1])
        if best is None or g > best[0]:
            best = (g, ck, out)
    loc = latest_ckpt(paths)
    if best is None or (loc and int(loc.stem.split("_")[1]) >= best[0]):
        return
    g, ck, out = best
    paths.ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ck, paths.ckpt_dir / ck.name)
    src_sw = ck.parents[1] / "stage_weights"
    if src_sw.exists():
        shutil.copytree(src_sw, paths.stage_weights, dirs_exist_ok=True)
    src_run = out / "runs" / cfg.run_name
    for f in ["train_log.csv", "dev_metrics.jsonl", "calibration.json", "sanity_checks.json"]:
        if (src_run / f).exists():
            paths.run_out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_run / f, paths.run_out / f)
    log(f"imported checkpoint {ck.name} (update {g}) from {ck.parent}")


# ----------------------------------------------------------------------------- main
def main():
    ap = base_argparser(__doc__)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    paths = Paths(cfg)
    tc = cfg.train
    sha = read_json(paths.data_out / "split_checks.json")["selection_sha256"]
    paths.run_out.mkdir(parents=True, exist_ok=True)
    if (paths.run_out / "train_summary.json").exists() and \
            (paths.stage_weights / f"stage{len(tc.stages)}.pt").exists():
        log("training already complete; skipping")
        return
    save_run_config(cfg, paths.run_out / "config.yaml")
    write_json(paths.env_out / f"env_{cfg.run_name}.json", env_info())
    seed_everything(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    pools = Pools(paths)
    fs = FeatureStore(paths.features)
    model = VLM(cfg, device)
    sanity = {"selection_sha256": sha}
    sanity["prompt_tokenisation"] = model.prompts.check()
    assert sanity["prompt_tokenisation"]["pieces_decode_to_chat_prompt"], "prompt pieces broken"
    f0 = fs.get([pools.caption[0]["image"]])
    sanity["feature_shape"] = list(f0.shape[1:])
    sanity["pooled_shape"] = list(pool_features(f0, model.n_vis).shape[1:])
    assert sanity["feature_shape"] == [196, 768]
    sanity["features_ln_check"] = read_json(paths.features / "meta.json").get("ln_check")
    ev = {it["image"] for ps in pools.dev_pairs.values() for p in ps for it in p["items"]} | \
        {c["image"] for c in pools.dev_captions}
    tr = {e["image"] for e in pools.caption} | {it["image"] for ps in pools.pairs.values()
                                                 for p in ps for it in p["items"]}
    sanity["split_isolation_train_vs_dev"] = not (ev & tr)
    assert sanity["split_isolation_train_vs_dev"]

    import_prior_run(cfg, paths, sha)
    resume = latest_ckpt(paths)
    stages = [Cfg(s) for s in tc.stages]
    eff = int(tc.micro_batch_size) * int(tc.grad_accum)
    n_draws = eff // 2
    cum = np.cumsum([0] + [int(s["updates"]) for s in stages])  # global update at stage starts
    dev_at = set(range(0, int(cum[-1]) + 1, int(cfg.dev_eval.every))) | {int(u) for u in cfg.dev_eval.extra_updates}
    log_path, dev_path = paths.run_out / "train_log.csv", paths.run_out / "dev_metrics.jsonl"
    scaler = torch.amp.GradScaler("cuda", init_scale=float(tc.init_loss_scale), enabled=amp)

    if resume:
        st = torch.load(resume, map_location="cpu", weights_only=False)
        assert st["selection_sha256"] == sha, "data selection changed since checkpoint"
        k0, su0, g0 = st["stage_idx"], st["stage_update"], st["global_update"]
        if su0 == int(stages[k0]["updates"]):  # checkpoint at a stage end -> next stage fresh
            k0, su0 = k0 + 1, 0
        set_trainable(model, k0)  # LoRA (seeded init) exists from stage 2 on; load_state fills it
        model.load_state({"connector": st["connector"], "lora": st.get("lora")})
        scaler.load_state_dict(st["scaler"])
        rs = st["rng"]
        random.setstate(rs["python"]); np.random.set_state(rs["numpy"]); torch.set_rng_state(rs["torch"])
        if rs["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rs["cuda"])
        calib = read_json(paths.run_out / "calibration.json")
        stats = st["stats"]
        rows = [r for r in csv.DictReader(open(log_path))] if log_path.exists() else []
        rows = [r for r in rows if int(r["update"]) <= g0]
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, LOG_FIELDS); w.writeheader(); w.writerows(rows)
        dev_rows = [r for r in read_jsonl(dev_path) if r["update"] <= g0] if dev_path.exists() else []
        write_jsonl(dev_path, dev_rows)
        old_sanity = read_json(paths.run_out / "sanity_checks.json") if (paths.run_out / "sanity_checks.json").exists() else {}
        sanity = old_sanity | sanity
        log(f"resumed from {resume.name}: stage {k0 + 1}, stage update {su0}, global {g0}")
    else:
        k0, su0, g0 = 0, 0, 0
        set_trainable(model, 0)
        # calibrate the connector output scale on training data only
        rng = np.random.default_rng([int(cfg.seed), 99])
        ims = sorted({e["image"] for e in pools.caption})
        cal_ims = [ims[int(i)] for i in rng.choice(len(ims), size=min(len(ims), int(cfg.model.calibration_images)),
                                                   replace=False)]
        texts = [e["target"] for e in pools.caption] + [p["question"] for ps in pools.pairs.values() for p in ps]
        texts = [texts[int(i)] for i in rng.choice(len(texts), size=min(len(texts), int(cfg.model.calibration_texts)),
                                                    replace=False)]
        calib = model.calibrate(fs.get(cal_ims), texts)
        write_json(paths.run_out / "calibration.json", calib)
        log(f"connector scale {calib['scale']:.4f} (text norm {calib['text_norm_mean']:.4f}, "
            f"initial visual norm {calib['visual_norm_mean_before']:.4f})")
        stats = {"train_time_s": 0.0, "n_skipped": 0, "stage": {}, "checkpoint_bytes": {}}
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, LOG_FIELDS).writeheader()
        write_jsonl(dev_path, [])
    model.train()
    if tc.gradient_checkpointing:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def run_dev(g, k):
        r = dev_eval(model, pools, fs, int(cfg.dev_eval.batch_size), amp)
        r = {"update": g, "stage": k + 1} | r
        with open(dev_path, "a") as f:
            import json
            f.write(json.dumps(r) + "\n")
        log(f"[dev @ {g}] caption loss {r['caption_loss']:.4f} | grounding NLL {r['grounding_nll_correct']:.3f} "
            f"ΔNLL {r['grounding_delta_nll']:+.3f} | vqa NLL {r['vqa_nll_correct']:.3f} "
            f"ΔNLL {r['vqa_delta_nll']:+.3f}")

    if g0 == 0 and not resume:
        run_dev(0, 0)

    for k in range(k0, len(stages)):
        sc = stages[k]
        T = int(sc["updates"])
        set_trainable(model, k)
        tsum = trainable_summary(model)
        sanity.setdefault("trainable_by_stage", {})[sc["name"]] = tsum
        opt, sched = make_optimizer(model, sc, tc)
        sampler = Sampler(pools, int(cfg.seed), k, sc, n_draws)
        start = su0 if k == k0 else 0
        sstats = stats["stage"].setdefault(sc["name"], {"train_time_s": 0.0, "peak_mem_gb": 0.0,
                                                        "trainable": tsum, "updates": T})
        if k == k0 and resume and start > 0:
            opt.load_state_dict(st["optimizer"])
            sched.load_state_dict(st["scheduler"])
            sampler.load(st["sampler"])
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        params = [p for _, p in model.trainable()]
        log(f"stage {k + 1} ({sc['name']}): updates {start}->{T}, trainable {tsum['total']:,} "
            f"(connector {tsum['connector']:,}, LoRA {tsum['lora']:,})")
        logf = open(log_path, "a", newline="")
        writer = csv.DictWriter(logf, LOG_FIELDS)
        for su in range(start, T):
            g = int(cum[k]) + su + 1
            t0 = time.perf_counter()
            exs = sampler.batch(su)
            assert len(exs) == eff
            counts = {c: sum(e["component"] == c for e in exs) for c in COMPONENTS}
            loss_tot, ntok = 0.0, 0
            mb = int(tc.micro_batch_size)
            for m in range(int(tc.grad_accum)):
                chunk = exs[m * mb:(m + 1) * mb]
                feats = fs.get([e["image"] for e in chunk])
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    per_ex = model.example_losses(chunk, feats)
                loss = per_ex.sum() / eff
                scaler.scale(loss).backward()
                loss_tot += float(per_ex.detach().sum())
                ntok += sum(len(model.prompts.target_ids(e["target"], e["kind"])) + 1 for e in chunk)
            scaler.unscale_(opt)
            gkey = f"grad_check_{sc['name']}"
            if gkey not in sanity:
                # first update of the stage with finite gradients: with FP16 autocast the GradScaler's
                # initial scale (65536) usually overflows on the first step(s), which it then skips
                cg = [p.grad for n, p in model.trainable() if n.startswith("connector.")]
                lg = [p.grad for n, p in model.trainable() if "lora_B" in n]
                cnorm = float(torch.norm(torch.stack([x.float().norm() for x in cg if x is not None])))
                lnorm = (float(torch.norm(torch.stack([x.float().norm() for x in lg if x is not None])))
                         if lg else None)
                finite = math.isfinite(cnorm) and (lnorm is None or math.isfinite(lnorm))
                if not finite:
                    log(f"grad check deferred at update {su + 1}: non-finite gradients "
                        f"(GradScaler overflow at scale {scaler.get_scale():.0f}; step skipped)")
                    assert su - start < 20, "gradients still non-finite after 20 updates"
                else:
                    sanity[gkey] = {
                        "update_in_stage": su + 1 - start, "loss_scale": scaler.get_scale(),
                        "connector_grad_norm": cnorm,
                        "connector_params_with_grad": sum(x is not None for x in cg), "connector_params": len(cg),
                        "lora_B_grad_norm": lnorm,
                        "backbone_params_with_grad": sum(p.grad is not None for n, p in model.lm.named_parameters()
                                                         if "lora_" not in n)}
                    log(f"grad check: connector grad norm {cnorm:.4g}"
                        + (f", LoRA-B grad norm {lnorm:.4g}" if lnorm is not None else ""))
                    assert cnorm > 0, sanity[gkey]
                    assert sanity[gkey]["backbone_params_with_grad"] == 0, sanity[gkey]
                    write_json(paths.run_out / "sanity_checks.json", sanity)
            gnorm = torch.nn.utils.clip_grad_norm_(params, float(tc.grad_clip))
            scale_before = scaler.get_scale()
            lrs = {gr["name"]: gr["lr"] for gr in opt.param_groups}
            scaler.step(opt)
            scaler.update()
            skipped = amp and scaler.get_scale() < scale_before
            stats["n_skipped"] += int(skipped)
            sched.step()
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            stats["train_time_s"] += dt
            sstats["train_time_s"] += dt
            peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            sstats["peak_mem_gb"] = max(sstats["peak_mem_gb"], peak)
            lv = loss_tot / eff
            writer.writerow({"update": g, "stage": k + 1, "stage_update": su + 1, "loss": f"{lv:.6f}",
                             "lr_connector": f"{lrs.get('connector', 0):.6e}",
                             "lr_lora": f"{lrs.get('lora', 0):.6e}" if "lora" in lrs else "",
                             "grad_norm": f"{float(gnorm):.6f}", "loss_scale": scale_before,
                             "skipped": int(skipped), "n_target_tokens": ntok, "update_time_s": f"{dt:.3f}",
                             "train_time_s": f"{stats['train_time_s']:.1f}", "peak_mem_gb": f"{peak:.3f}"}
                            | {f"n_{c}": counts[c] for c in COMPONENTS})
            logf.flush()
            if not math.isfinite(lv):
                raise RuntimeError(f"non-finite loss at update {g}")
            if su + 1 == 1 or (su + 1) % 50 == 0 or skipped:
                log(f"[stage {k + 1}] update {g} ({su + 1}/{T}) loss {lv:.4f} lr {lrs} gnorm {float(gnorm):.3f} "
                    f"scale {scale_before:.0f}{' SKIPPED' if skipped else ''} {dt:.2f}s/upd peak {peak:.2f}GB "
                    f"mix {counts}")
            stage_end = su + 1 == T
            if stage_end:
                paths.stage_weights.mkdir(parents=True, exist_ok=True)
                torch.save(model.state() | {"stage": k + 1, "name": sc["name"], "global_update": g},
                           paths.stage_weights / f"stage{k + 1}.pt")
                log(f"stage {k + 1} done: {sstats['train_time_s'] / 60:.1f} min, peak {sstats['peak_mem_gb']:.2f} GB")
            if g in dev_at:
                run_dev(g, k)
            if g % int(tc.checkpoint_every) == 0 or stage_end:
                stt = model.state()
                ck = {"selection_sha256": sha, "stage_idx": k, "stage_update": su + 1, "global_update": g,
                      "connector": stt["connector"], "lora": stt["lora"], "optimizer": opt.state_dict(),
                      "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "sampler": sampler.state(),
                      "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                              "torch": torch.get_rng_state(),
                              "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                      "stats": stats}
                stats["checkpoint_bytes"][str(g)] = save_ckpt(paths, ck, int(tc.keep_last_checkpoints))
        logf.close()
        write_json(paths.run_out / "sanity_checks.json", sanity)

    summary = {"run_name": cfg.run_name, "visual_tokens": model.n_vis, "updates": int(cum[-1]),
               "examples": int(cum[-1]) * eff, "train_time_s": stats["train_time_s"],
               "train_time_note": "sum of per-update times (forward/backward/step); excludes dev "
                                  "evaluation, checkpointing and feature caching",
               "skipped_updates": stats["n_skipped"], "stages": stats["stage"],
               "checkpoint_bytes": stats["checkpoint_bytes"],
               "stage_weight_bytes": {p.name: p.stat().st_size for p in sorted(paths.stage_weights.glob("*.pt"))},
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               "calibration_scale": calib["scale"],
               "feature_cache": {k: v for k, v in read_json(paths.features / "meta.json").items()
                                 if k in ("cache_time_s", "images_per_s", "peak_gpu_mem_gb", "bytes", "n_images",
                                          "copied_from")}}
    write_json(paths.run_out / "train_summary.json", summary)
    log(f"training done: {stats['train_time_s'] / 60:.1f} min, skipped {stats['n_skipped']} updates")


if __name__ == "__main__":
    main()
