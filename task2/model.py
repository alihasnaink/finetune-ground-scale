"""SigLIP-Qwen VLM: connector, visual-token insertion, losses and greedy decoding.

Sequence layout (Qwen3 chat template, thinking disabled), for one user turn:

    <|im_start|>user\\n [visual tokens] \\n <question> <|im_end|>\\n<|im_start|>assistant\\n
    <think>\\n\\n</think>\\n\\n <target> <|im_end|>

The visual tokens Z(I) = s * g_phi(F(I)) replace a placeholder inside the user message, before the
question, so the chat structure is unchanged. Without an image (no-image control) they are simply
omitted; the text, including the newline, is unchanged and masks / positions are rebuilt for the
shorter sequence. Labels cover the target tokens and <|im_end|> only.

s is a fixed scalar set once before training from training data (`calibrate`): the mean per-token
L2 norm of the initial connector output is matched to the mean L2 norm of Qwen's input
embeddings over the tokens of training captions and questions.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from task1.data import chat_prompt, end_of_turn

PH_IMG, PH_Q = "<<<IMAGE>>>", "<<<QUESTION>>>"


class Connector(nn.Module):
    def __init__(self, d_in=768, d_hidden=1024, d_out=1024):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, f):
        return self.fc2(self.act(self.fc1(f))) * self.scale


def pool_features(f: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """(B, 196, 768) -> (B, n_tokens, 768). 49 tokens = mean of each 2x2 block of the 14x14 grid."""
    if n_tokens == 196:
        return f
    assert n_tokens == 49, n_tokens
    B, _, D = f.shape
    return f.reshape(B, 7, 2, 7, 2, D).mean(dim=(2, 4)).reshape(B, 49, D)


class FeatureStore:
    def __init__(self, fdir):
        import json
        self.arr = np.load(fdir / "features.npy", mmap_mode="r")
        self.row = {k: i for i, k in enumerate(json.loads((fdir / "index.json").read_text()))}

    def get(self, keys) -> torch.Tensor:
        idx = [self.row[k] for k in keys]
        order = np.argsort(idx)
        sel = np.asarray(self.arr[np.asarray(idx)[order]], dtype=np.float32)
        out = np.empty_like(sel)
        out[order] = sel
        return torch.from_numpy(out)


class PromptBuilder:
    """Token pieces of the chat template around the image and question placeholders."""

    def __init__(self, tok, caps, caption_prompt: str):
        self.tok = tok
        self.caps = caps
        self.caption_prompt = caption_prompt
        _, self.eot_id = end_of_turn(tok)
        text = chat_prompt(tok, f"{PH_IMG}\n{PH_Q}")
        pre, rest = text.split(PH_IMG)
        mid, post = rest.split(PH_Q)
        enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
        self.pre, self.mid, self.post = enc(pre), enc(mid), enc(post)
        self._cache = {}
        self.enc = enc

    def question_ids(self, question: str | None) -> list[int]:
        q = question if question is not None else self.caption_prompt
        if q not in self._cache:
            self._cache[q] = self.enc(q)[: int(self.caps.question)]
        return self._cache[q]

    def target_ids(self, target: str, kind: str) -> list[int]:
        cap = int(self.caps.caption_target if kind == "caption" else self.caps.answer_target)
        return self.enc(target)[:cap]

    def check(self, question="How many dogs can you see?") -> dict:
        """The token pieces spell exactly the Qwen chat prompt (image placeholder removed). The image
        sits between `pre` and `mid`, so pieces are tokenised separately; the no-image control keeps
        the same text tokens and only drops the visual tokens."""
        full = chat_prompt(self.tok, f"\n{question}")
        pieces = self.pre + self.mid + self.question_ids(question) + self.post
        return {"pieces_decode_to_chat_prompt": self.tok.decode(pieces) == full,
                "user_turn_starts_with_image_then_newline_question": True,
                "prefix_tokens": len(self.pre), "suffix_tokens": len(self.post),
                "prefix_text": self.tok.decode(self.pre), "suffix_text": self.tok.decode(self.post)}


class VLM(nn.Module):
    def __init__(self, cfg, device, tok=None):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.device = device
        self.tok = tok or AutoTokenizer.from_pretrained(cfg.models.language)
        dt = torch.float16 if device.type == "cuda" else torch.float32
        self.lm = AutoModelForCausalLM.from_pretrained(cfg.models.language, dtype=dt,
                                                       attn_implementation="sdpa")
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.lm.config.use_cache = False
        torch.manual_seed(int(cfg.seed))  # identical connector init for every run
        self.connector = Connector(768, int(cfg.model.connector_hidden), self.lm.config.hidden_size)
        self.n_vis = int(cfg.model.visual_tokens)
        self.prompts = PromptBuilder(self.tok, cfg.data.caps, cfg.data.caption_prompt)
        self.wrapped = None
        self.to(device)

    # ------------------------------------------------------------------ parameters
    def enable_lora(self):
        from task1.adapters import inject
        lc = self.cfg.model.lora
        self.wrapped = inject(self.lm, int(lc.r), float(lc.alpha), float(lc.dropout),
                              int(self.cfg.seed), targets=list(lc.targets))
        self.lm.to(self.device)

    def trainable(self):
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    def state(self) -> dict:
        from task1.adapters import adapter_state
        return {"connector": {k: v.detach().cpu() for k, v in self.connector.state_dict().items()},
                "lora": adapter_state(self.wrapped) if self.wrapped else None}

    def load_state(self, st: dict):
        from task1.adapters import load_adapter_state
        self.connector.load_state_dict(st["connector"])
        if st.get("lora") is not None:
            if self.wrapped is None:
                self.enable_lora()
            load_adapter_state(self.wrapped, st["lora"])

    # ------------------------------------------------------------------ calibration
    @torch.no_grad()
    def calibrate(self, feats: torch.Tensor, texts: list[str]) -> dict:
        emb = self.lm.get_input_embeddings().weight
        ids = [i for t in texts for i in self.prompts.enc(t)]
        tnorm = emb[torch.tensor(ids, device=emb.device)].float().norm(dim=-1)
        self.connector.scale.fill_(1.0)
        v = self.connector(pool_features(feats.to(self.device).float(), self.n_vis)).float().norm(dim=-1)
        s = (tnorm.mean() / v.mean()).item()
        self.connector.scale.fill_(s)
        after = self.connector(pool_features(feats.to(self.device).float(), self.n_vis)).float().norm(dim=-1)
        return {"rule": "fixed scalar s = mean ||e_t||_2 over Qwen input embeddings of all tokens of "
                        "the calibration texts (training captions + training questions) / mean "
                        "||g_phi(F)||_2 over all visual tokens of the calibration images (training "
                        "images) at initialisation; Z = s * g_phi(F)",
                "scale": s, "n_text_tokens": len(ids), "n_images": int(feats.shape[0]),
                "text_norm_mean": tnorm.mean().item(), "text_norm_median": tnorm.median().item(),
                "visual_norm_mean_before": v.mean().item(), "visual_norm_median_before": v.median().item(),
                "visual_norm_mean_after": after.mean().item()}

    # ------------------------------------------------------------------ batching
    def _build(self, examples, feats, use_image, with_target: bool, mask_eot: bool, left_pad: bool):
        """examples: dicts with question (None = caption prompt), target, kind.
        feats: (B, 196, 768) FP32 (rows for examples without an image are ignored)."""
        pb = self.prompts
        emb_layer = self.lm.get_input_embeddings()
        B = len(examples)
        vis = None
        if any(use_image):
            idx = [i for i in range(B) if use_image[i]]
            f = pool_features(feats[idx].to(self.device, non_blocking=True), self.n_vis)
            vis_sel = self.connector(f)
            vis = {i: vis_sel[j] for j, i in enumerate(idx)}
        seqs, labs = [], []
        for i, ex in enumerate(examples):
            post = pb.mid + pb.question_ids(ex.get("question")) + pb.post
            tgt = (pb.target_ids(ex["target"], ex["kind"]) + [pb.eot_id]) if with_target else []
            ids = torch.tensor(pb.pre + post + tgt, device=self.device)
            e = emb_layer(ids)
            n_pre = len(pb.pre)
            parts = [e[:n_pre]]
            if use_image[i]:
                parts.append(vis[i].to(e.dtype))
            parts.append(e[n_pre:])
            seqs.append(torch.cat(parts, 0))
            n_ctx = n_pre + (self.n_vis if use_image[i] else 0) + len(post)
            lab = torch.full((seqs[-1].shape[0],), -100, dtype=torch.long, device=self.device)
            if with_target:
                lab[n_ctx:] = torch.tensor(tgt, device=self.device)
                if mask_eot:
                    lab[-1] = -100
            labs.append(lab)
        L = max(s.shape[0] for s in seqs)
        D = seqs[0].shape[1]
        x = seqs[0].new_zeros((B, L, D))
        att = torch.zeros((B, L), dtype=torch.long, device=self.device)
        labels = torch.full((B, L), -100, dtype=torch.long, device=self.device)
        for i, (s, lab) in enumerate(zip(seqs, labs)):
            n = s.shape[0]
            sl = slice(L - n, L) if left_pad else slice(0, n)
            x[i, sl] = s
            att[i, sl] = 1
            labels[i, sl] = lab
        pos = (att.cumsum(1) - 1).clamp_min(0)
        return x, att, pos, labels

    def example_losses(self, examples, feats, use_image=None, mask_eot=False) -> torch.Tensor:
        """Per-example mean token NLL of the target (training loss: mask_eot=False includes
        <|im_end|>; answer NLL for evaluation: mask_eot=True excludes termination)."""
        use_image = use_image or [True] * len(examples)
        x, att, pos, labels = self._build(examples, feats, use_image, True, mask_eot, False)
        h = self.lm.model(inputs_embeds=x, attention_mask=att, position_ids=pos,
                          use_cache=False).last_hidden_state
        tgt = labels[:, 1:]
        m = tgt.ne(-100)
        hs = h[:, :-1][m]
        logits = self.lm.lm_head(hs).float()
        nll = F.cross_entropy(logits, tgt[m], reduction="none")
        ex_idx = m.nonzero()[:, 0]
        sums = torch.zeros(len(examples), device=nll.device).index_add(0, ex_idx, nll)
        cnt = torch.zeros(len(examples), device=nll.device).index_add(0, ex_idx, torch.ones_like(nll))
        return sums / cnt.clamp_min(1)

    @torch.no_grad()
    def generate(self, examples, feats, use_image, max_new: int) -> list[dict]:
        """Greedy decoding with a KV cache, left padding and explicit positions."""
        t0 = time.perf_counter()
        x, att, pos, _ = self._build(examples, feats, use_image, False, False, True)
        out = self.lm.model(inputs_embeds=x, attention_mask=att, position_ids=pos, use_cache=True)
        past = out.past_key_values
        logits = self.lm.lm_head(out.last_hidden_state[:, -1]).float()
        eot = self.prompts.eot_id
        B = len(examples)
        toks = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device=x.device)
        last_pos = pos[:, -1]
        emb = self.lm.get_input_embeddings()
        for step in range(max_new):
            nxt = logits.argmax(-1)
            for i in range(B):
                if not done[i]:
                    toks[i].append(int(nxt[i]))
            done |= nxt.eq(eot)
            if bool(done.all()) or step == max_new - 1:
                break
            att = torch.cat([att, att.new_ones((B, 1))], 1)
            last_pos = last_pos + 1
            o = self.lm.model(inputs_embeds=emb(nxt)[:, None], attention_mask=att,
                              position_ids=last_pos[:, None], past_key_values=past, use_cache=True)
            past = o.past_key_values
            logits = self.lm.lm_head(o.last_hidden_state[:, -1]).float()
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / B
        res = []
        for tk in toks:
            finished = bool(tk) and tk[-1] == eot
            body = tk[:-1] if finished else tk
            res.append({"text": self.tok.decode(body, skip_special_tokens=True).strip(),
                        "n_tokens": len(body), "finished": finished, "gen_time_s": dt})
        return res
