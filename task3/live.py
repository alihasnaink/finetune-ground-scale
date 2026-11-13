"""Task 3, Step 4: live generation on the fixed 50-question MATH subset (GPU).

    python -m task3.live --config task3/configs/live.yaml [--lane 0 --n-lanes 2] [--limit 2]

Strategies, each with the same 3,072-token output allowance per question:
  single     one response, <= 3,072 new tokens
  parallel   three independent responses, <= 1,024 new tokens each (self-consistency in task3.live_eval)
  refine     the first parallel response (shared initial answer) is appended as an assistant message,
             followed by the fixed review instruction; one critique-and-revision response, <= 2,048 tokens

Qwen2.5-3B-Instruct, FP16 weights, no quantization, batch size 1, native chat template with the
system message "You are a helpful assistant.", temperature 0.7, top-p 0.8, top-k 20, repetition
penalty 1.0. Every call is independently seeded (seed + 1000 * question index + strategy offset).

Each call is appended to calls_lane<k>.jsonl as soon as it finishes: messages, rendered prompt, seed,
input / output token counts, response, finish reason (eos / length), truncation flag, extracted
\\boxed{} answer, end-to-end time (chat templating + tokenisation + generation + decoding, model
loading excluded, CUDA-synchronised), tokens/s and peak GPU memory of the call. Finished calls are
skipped on restart (also those found in an attached earlier output: T3_PRIOR_SEARCH), so a failed
run resumes. Questions are split between lanes (one GPU each) by index; all three strategies of a
question run on the same GPU, so per-question comparisons share hardware.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .common import base_argparser, env_info, load_config, log, read_json, read_jsonl, write_json
from .extract import EXTRACTOR_VERSION, last_boxed

REPO = Path(__file__).resolve().parent.parent
STRATEGY_CALLS = ["parallel_0", "parallel_1", "parallel_2", "single", "refine"]


def user_prompt(problem: str, cfg) -> str:
    return f"{problem}\n\n{cfg.generation.instruction}"


def messages_for(call: str, problem: str, cfg, initial_response: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": cfg.generation.system},
            {"role": "user", "content": user_prompt(problem, cfg)}]
    if call == "refine":
        assert initial_response is not None
        msgs += [{"role": "assistant", "content": initial_response},
                 {"role": "user", "content": cfg.generation.refine_instruction}]
    return msgs


def max_new_for(call: str, cfg) -> int:
    s = cfg.strategies
    return int(s.single_max_new if call == "single" else s.refine_max_new if call == "refine" else s.parallel_max_new)


class Generator:
    def __init__(self, cfg, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        m = cfg.model
        kw = {"revision": m.revision} if m.get("revision") else {}
        t = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(m.name, **kw)
        self.model = AutoModelForCausalLM.from_pretrained(
            m.name, dtype=getattr(torch, m.dtype), attn_implementation=m.get("attn_implementation", "sdpa"),
            **kw).to(device).eval()
        self.load_s = time.perf_counter() - t
        self.device = device
        self.commit = getattr(self.model.config, "_commit_hash", None)
        if self.commit is None:  # newer transformers no longer set it; the cache snapshot dir is the commit
            try:
                from huggingface_hub import snapshot_download
                self.commit = Path(snapshot_download(m.name, revision=m.get("revision"),
                                                     local_files_only=True)).name
            except Exception:
                pass
        gc = self.model.generation_config
        eos = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.eos_ids = sorted({int(e) for e in eos if e is not None} | ({int(im_end)} if im_end is not None else set()))
        g = cfg.generation
        self.gen_kwargs = dict(do_sample=True, temperature=float(g.temperature), top_p=float(g.top_p),
                               top_k=int(g.top_k), repetition_penalty=float(g.repetition_penalty),
                               eos_token_id=self.eos_ids, pad_token_id=self.tok.pad_token_id or self.eos_ids[0],
                               min_p=None, typical_p=None)

    def __call__(self, msgs: list[dict], max_new: int, seed: int) -> dict:
        cuda = self.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                                      **self.gen_kwargs)
        new = out[0, ids.shape[1]:].tolist()
        finished = bool(new) and new[-1] in self.eos_ids
        body = new[:-1] if finished else new
        text = self.tok.decode(body, skip_special_tokens=True)
        if cuda:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        return {"prompt": prompt, "input_tokens": int(ids.shape[1]), "output_tokens": len(new),
                "response_tokens": len(body), "finish_reason": "eos" if finished else "length",
                "truncated": (not finished) and len(new) >= max_new, "response": text,
                "time_s": dt, "tokens_per_s": len(new) / dt if dt > 0 else None,
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if cuda else None}


def load_done(out: Path, prior: str | None) -> dict:
    done = {}
    files = sorted(out.glob("calls_lane*.jsonl"))
    if prior and Path(prior).exists():
        files = sorted(Path(prior).rglob("task3/live/calls_lane*.jsonl")) + files  # local wins
    for f in files:
        try:
            for r in read_jsonl(f):
                done[(r["qid"], r["call"])] = r
        except Exception as e:  # a partially written last line from an interrupted run
            log(f"warning: could not fully read {f}: {e}")
    return done


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--lane", type=int, default=0)
    ap.add_argument("--n-lanes", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="only the first N live questions (smoke test)")
    ap.add_argument("--inspect", action="store_true", help="print the data schema and the live subset, then exit")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = Path(cfg.paths.out_root) / "live"
    out.mkdir(parents=True, exist_ok=True)

    # the 50 live questions, committed by task3.prepare_splits (first 50 MATH evaluation questions)
    live = read_json(REPO / cfg.data.live_questions)
    split = read_json(REPO / cfg.data.split_file)
    live_ids = [q["qid"] for q in live]
    assert live_ids == split["eval"][:int(cfg.data.n_live)], "live_questions.json does not match the split"
    qs = {q["qid"]: q for q in live}
    if args.lane == 0:
        write_json(out / "live_questions.json", live)
    if args.inspect:
        print(f"{len(live)} live questions from {cfg.data.live_questions} (split source sha256 "
              f"{split['source_sha256'][:12]})")
        for q in live[:3]:
            print(q["live_index"], q["qid"], repr(q["gold"]), "|", q["problem"][:200].replace("\n", " "))
        return
    todo = [(i, q) for i, q in enumerate(live_ids) if i % args.n_lanes == args.lane]
    if args.limit is not None:
        todo = [(i, q) for i, q in todo if i < args.limit]

    done = load_done(out, cfg.paths.get("prior_search"))
    lane_file = out / f"calls_lane{args.lane}.jsonl"
    # copy calls found in an attached earlier output into this lane's file, so the output is complete
    existing = {(r["qid"], r["call"]) for r in (read_jsonl(lane_file) if lane_file.exists() else [])}
    mine = {q for _, q in todo}
    with open(lane_file, "a") as fh:
        for (qid, call), r in done.items():
            if (qid, call) not in existing and qid in mine:
                fh.write(json.dumps(r | {"imported": True}) + "\n")
    n_calls = sum((q, c) not in done for _, q in todo for c in STRATEGY_CALLS)
    log(f"lane {args.lane}/{args.n_lanes}: {len(todo)} questions, {n_calls} calls to run "
        f"({len(todo) * len(STRATEGY_CALLS) - n_calls} reused)")
    if n_calls == 0:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = Generator(cfg, device)
    info = {"lane": args.lane, "model": cfg.model.name, "revision_requested": cfg.model.get("revision"),
            "revision_loaded": gen.commit, "dtype": cfg.model.dtype, "eos_ids": gen.eos_ids,
            "gen_kwargs": {k: v for k, v in gen.gen_kwargs.items() if v is not None},
            "model_load_s": gen.load_s, "extractor": EXTRACTOR_VERSION, "env": env_info(),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    write_json(out / f"run_info_lane{args.lane}.json", info)
    log(f"model loaded in {gen.load_s:.0f}s (commit {gen.commit}), eos {gen.eos_ids}")

    t_start = time.perf_counter()
    n_run = 0
    for i, qid in todo:
        q = qs[qid]
        for call in STRATEGY_CALLS:
            if (qid, call) in done:
                continue
            initial = done[(qid, "parallel_0")]["response"] if call == "refine" else None
            msgs = messages_for(call, q["problem"], cfg, initial)
            seed = int(cfg.seed) + 1000 * i + int(cfg.seed_offsets[call])
            max_new = max_new_for(call, cfg)
            r = gen(msgs, max_new, seed)
            rec = {"qid": qid, "live_index": i, "call": call,
                   "strategy": "parallel" if call.startswith("parallel") else call, "seed": seed,
                   "max_new_tokens": max_new, "lane": args.lane, "messages": msgs, **r,
                   "extracted": last_boxed(r["response"]), "gold": q["gold"]}
            with open(lane_file, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            done[(qid, call)] = rec
            n_run += 1
            el = time.perf_counter() - t_start
            log(f"[q{i:02d} {call:10s}] {r['output_tokens']:4d}/{max_new} tok "
                f"{'TRUNC ' if r['truncated'] else ''}{r['time_s']:.1f}s {r['tokens_per_s']:.1f} tok/s | "
                f"boxed={rec['extracted']!r} gold={q['gold']!r} | {n_run}/{n_calls} calls, "
                f"eta {el / n_run * (n_calls - n_run) / 60:.0f} min")
    log(f"lane {args.lane} done: {n_run} calls in {(time.perf_counter() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
