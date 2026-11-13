"""MetaMathQA selection/tokenisation, batching, probe set and qualitative prompts."""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

import numpy as np
import yaml

from .common import Cfg, Paths, log, read_json, sha256_bytes, write_json

IGNORE_INDEX = -100
FORMATS = ("plain", "json", "template")


# ----------------------------------------------------------------------------- formatting
def normalize_text(s: str) -> str:
    """Normalisation used for GSM8K-test decontamination: NFKC, lower-case,
    collapse all whitespace runs to one space, strip."""
    return " ".join(unicodedata.normalize("NFKC", s).lower().split())


def chat_prompt(tok, user: str, system: str | None = None) -> str:
    """Qwen3 chat prompt for one user turn with thinking disabled. Ends with the
    generation prefix '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n'."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def end_of_turn(tok) -> tuple[str, int]:
    eot = "<|im_end|>"
    eot_id = tok.convert_tokens_to_ids(eot)
    assert isinstance(eot_id, int) and eot_id >= 0, "tokenizer lacks <|im_end|>"
    return eot, eot_id


# ----------------------------------------------------------------------------- training data
def build_training_data(cfg: Cfg, tok, paths: Paths) -> dict:
    """Select the first `num_examples` eligible MetaMathQA rows in source order.

    Exclusion rules (manual): empty question/response; query or original_question
    matching a GSM8K test question after normalisation; formatted length > max_seq_len.
    Retained examples are never truncated or shuffled.
    """
    from datasets import load_dataset

    n_target = int(cfg.data.num_examples)
    max_len = int(cfg.data.max_seq_len)
    eot, eot_id = end_of_turn(tok)

    gsm = load_dataset(cfg.data.gsm8k_dataset, cfg.data.gsm8k_config, split="test")
    gsm_test = {normalize_text(q) for q in gsm["question"]}
    log(f"GSM8K test questions for decontamination: {len(gsm_test)}")

    ds = load_dataset(cfg.data.dataset, split=cfg.data.split)
    log(f"MetaMathQA rows: {len(ds)}; columns: {ds.column_names}")

    sel_src, sel_type, prompt_ids, resp_ids = [], [], [], []
    excluded = []  # (source_index, reason)
    counts = {"empty": 0, "gsm8k_test_match": 0, "too_long": 0}
    template_checks, template_mismatch = 0, 0

    chunk = 2000
    scanned = 0
    for start in range(0, len(ds), chunk):
        if len(sel_src) >= n_target:
            break
        rows = ds[start:start + chunk]
        cand_idx, cand_prompt, cand_resp = [], [], []
        for j in range(len(rows["query"])):
            src = start + j
            q, r = rows["query"][j] or "", rows["response"][j] or ""
            oq = rows.get("original_question", [None] * len(rows["query"]))[j] or ""
            if not q.strip() or not r.strip():
                excluded.append((src, "empty")); counts["empty"] += 1; continue
            if normalize_text(q) in gsm_test or (oq and normalize_text(oq) in gsm_test):
                excluded.append((src, "gsm8k_test_match")); counts["gsm8k_test_match"] += 1; continue
            cand_idx.append(j)
            cand_prompt.append(chat_prompt(tok, q))
            cand_resp.append(r + eot)
        p_tok = tok(cand_prompt, add_special_tokens=False)["input_ids"] if cand_prompt else []
        r_tok = tok(cand_resp, add_special_tokens=False)["input_ids"] if cand_resp else []
        for j, p, r, ptxt, rtxt in zip(cand_idx, p_tok, r_tok, cand_prompt, cand_resp):
            src = start + j
            if len(sel_src) >= n_target:
                break
            assert r[-1] == eot_id, f"response of row {src} does not end with <|im_end|>"
            if len(p) + len(r) > max_len:
                excluded.append((src, "too_long")); counts["too_long"] += 1; continue
            # Consistency check against the chat template's own full-conversation rendering.
            if template_checks < 200:
                full = tok.apply_chat_template(
                    [{"role": "user", "content": rows["query"][j]},
                     {"role": "assistant", "content": rows["response"][j]}],
                    tokenize=False, enable_thinking=False)
                template_checks += 1
                if full.rstrip("\n") != (ptxt + rtxt).rstrip("\n"):
                    template_mismatch += 1
            sel_src.append(src)
            sel_type.append(rows["type"][j] if "type" in rows else "")
            prompt_ids.append(np.asarray(p, dtype=np.int32))
            resp_ids.append(np.asarray(r, dtype=np.int32))
        scanned = min(start + chunk, len(ds))
        if (start // chunk) % 5 == 0:
            log(f"  scanned {scanned} rows, selected {len(sel_src)}")

    if len(sel_src) < n_target:
        raise RuntimeError(f"only {len(sel_src)} eligible examples found")
    last_src = sel_src[-1]
    excluded = [e for e in excluded if e[0] < last_src]
    for k in counts:
        counts[k] = sum(1 for e in excluded if e[1] == k)

    p_len = np.array([len(x) for x in prompt_ids], dtype=np.int64)
    r_len = np.array([len(x) for x in resp_ids], dtype=np.int64)
    arrays = dict(
        source_index=np.asarray(sel_src, dtype=np.int64),
        prompt_flat=np.concatenate(prompt_ids), prompt_off=np.concatenate([[0], np.cumsum(p_len)]),
        resp_flat=np.concatenate(resp_ids), resp_off=np.concatenate([[0], np.cumsum(r_len)]),
    )
    digest = sha256_bytes(b"".join(arrays[k].tobytes() for k in sorted(arrays)))

    paths.data_scratch.mkdir(parents=True, exist_ok=True)
    np.savez(paths.data_scratch / "train_tokens.npz", **arrays)

    paths.data_out.mkdir(parents=True, exist_ok=True)
    with open(paths.data_out / "selected_examples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["position", "source_index", "type", "n_prompt_tokens", "n_response_tokens"])
        for i, (s, t, a, b) in enumerate(zip(sel_src, sel_type, p_len, r_len)):
            w.writerow([i, s, t, int(a), int(b)])
    with open(paths.data_out / "excluded_examples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_index", "reason"])
        w.writerows(excluded)

    tot = p_len + r_len
    types, tcounts = np.unique(np.asarray(sel_type), return_counts=True)
    stats = {
        "dataset": cfg.data.dataset, "split": cfg.data.split,
        "rows_in_source": len(ds), "rows_scanned_through_source_index": int(last_src),
        "num_selected": len(sel_src), "excluded_counts": counts,
        "normalization": "NFKC + lowercase + whitespace collapse + strip",
        "gsm8k_test_questions": len(gsm_test),
        "max_seq_len": max_len,
        "tokens": {
            "response_total": int(r_len.sum()), "prompt_total": int(p_len.sum()),
            "formatted_mean": float(tot.mean()), "formatted_median": float(np.median(tot)),
            "formatted_p99": float(np.percentile(tot, 99)), "formatted_max": int(tot.max()),
            "response_mean": float(r_len.mean()),
        },
        "type_counts": {str(t): int(c) for t, c in zip(types, tcounts)},
        "chat_template_check": {"checked": template_checks, "mismatches": template_mismatch},
        "loss_tokens": "assistant response tokens + <|im_end|>; prompt and padding masked",
        "data_sha256": digest,
    }
    write_json(paths.data_out / "data_stats.json", stats)
    write_json(paths.data_scratch / "train_tokens.meta.json", {"data_sha256": digest})
    log(f"selected {len(sel_src)} examples (last source index {last_src}); excluded {counts}; "
        f"sha256={digest[:12]}")
    return stats


class TrainData:
    """Tokenised training set in fixed source order."""

    def __init__(self, paths: Paths):
        z = np.load(paths.data_scratch / "train_tokens.npz")
        self.source_index = z["source_index"]
        self.pf, self.po = z["prompt_flat"], z["prompt_off"]
        self.rf, self.ro = z["resp_flat"], z["resp_off"]
        self.sha256 = read_json(paths.data_scratch / "train_tokens.meta.json")["data_sha256"]

    def __len__(self):
        return len(self.source_index)

    def example(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self.pf[self.po[i]:self.po[i + 1]], self.rf[self.ro[i]:self.ro[i + 1]]

    def resp_len(self, i: int) -> int:
        return int(self.ro[i + 1] - self.ro[i])

    def total_len(self, i: int) -> int:
        return int(self.po[i + 1] - self.po[i] + self.ro[i + 1] - self.ro[i])


def collate(examples: list[tuple[np.ndarray, np.ndarray]], pad_id: int):
    """Right-pad; labels = response tokens (incl. <|im_end|>), everything else ignored."""
    import torch

    L = max(len(p) + len(r) for p, r in examples)
    ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    lab = torch.full((len(examples), L), IGNORE_INDEX, dtype=torch.long)
    att = torch.zeros((len(examples), L), dtype=torch.long)
    for b, (p, r) in enumerate(examples):
        n = len(p) + len(r)
        seq = torch.from_numpy(np.concatenate([p, r]).astype(np.int64))
        ids[b, :n] = seq
        att[b, :n] = 1
        lab[b, len(p):n] = seq[len(p):]
    return ids, att, lab


# ----------------------------------------------------------------------------- probe set
def build_probe_set(cfg: Cfg, tok, paths: Paths) -> dict:
    """32 GSM8K test questions + 32 HellaSwag validation contexts sampled with seed 6304,
    rendered with the exact lm-eval benchmark prompt formats (GSM8K: zero-shot chat template,
    thinking off; HellaSwag: stock completion context)."""
    from lm_eval.tasks import TaskManager, get_task_dict
    from lm_eval.utils import hash_string

    tasks = get_task_dict([cfg.eval.gsm8k_task, "hellaswag"], TaskManager())
    gsm_task, hs_task = tasks[cfg.eval.gsm8k_task], tasks["hellaswag"]
    gsm_docs, hs_docs = list(gsm_task.test_docs()), list(hs_task.validation_docs())
    rng = np.random.default_rng(int(cfg.seed))
    math_ids = sorted(int(i) for i in rng.choice(len(gsm_docs), int(cfg.probe.n_math), replace=False))
    non_ids = sorted(int(i) for i in rng.choice(len(hs_docs), int(cfg.probe.n_nonmath), replace=False))

    def chat_fn(msgs, add_generation_prompt=True):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
                                       continue_final_message=not add_generation_prompt,
                                       enable_thinking=False)

    items = []
    for i in math_ids:
        ctx = gsm_task.fewshot_context(gsm_docs[i], 0, apply_chat_template=True,
                                       fewshot_as_multiturn=True, chat_template=chat_fn)
        items.append({"group": "math", "task": cfg.eval.gsm8k_task, "doc_id": i, "prompt": ctx,
                      "prompt_hash": hash_string(ctx)})
    for i in non_ids:
        ctx = hs_task.fewshot_context(hs_docs[i], 0)
        items.append({"group": "nonmath", "task": "hellaswag", "doc_id": i, "prompt": ctx,
                      "prompt_hash": hash_string(ctx)})
    probe = {"seed": int(cfg.seed), "selection": "numpy default_rng(seed).choice without replacement, "
             "GSM8K test first then HellaSwag validation", "token_position": "last prompt token",
             "items": items}
    write_json(paths.data_out / "probe_set.json", probe)
    log(f"probe set: {len(math_ids)} math + {len(non_ids)} non-math prompts")
    return probe


# ----------------------------------------------------------------------------- qualitative
def load_qualitative_prompts(cfg: Cfg) -> list[dict]:
    from .common import REPO_ROOT

    path = Path(cfg.qualitative.questions_file)
    if not path.is_absolute():
        path = REPO_ROOT / path
    entries = yaml.safe_load(path.read_text())
    plain_order = [e["id"] for e in entries
                   if not e["id"].endswith("_json") and not e["id"].endswith("_template")]
    out = []
    for e in entries:
        qid = e["id"]
        fmt = "json" if qid.endswith("_json") else "template" if qid.endswith("_template") else "plain"
        base = qid[: -len("_" + fmt)] if fmt != "plain" else qid
        assert len(e["paraphrases"]) == 1, qid
        out.append({"id": qid, "question": base, "format": fmt,
                    "plain_order": plain_order.index(base), "system": e.get("system"),
                    "prompt": e["paraphrases"][0]})
    out.sort(key=lambda x: (x["plain_order"], FORMATS.index(x["format"])))
    assert len(out) == 24, len(out)
    return out
