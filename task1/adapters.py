"""LoRA and OPLoRA adapters for Task 1 (no PEFT dependency, so every detail is explicit).

LoRA (Hu et al.): for a frozen weight W0 (d_out x d_in) the effective weight is
    W = W0 + (alpha / r) * B A,    A (r x d_in) random (Kaiming-uniform, as PEFT), B (d_out x r) = 0,
with dropout on the adapter input and no adapter bias. The backbone is frozen FP16, the
factors are FP32 trainable parameters.

OPLoRA (Xiong & Xie): with W0 = U S V^T and the leading k singular vectors U_k, V_k (computed
once in FP32 and frozen), the factors are projected before use,
    B~ = B - U_k (U_k^T B),    A~ = A - (A V_k) V_k^T,    dW = (alpha / r) B~ A~,
so dW has no component in the protected output subspace (U_k^T dW = 0) or input subspace
(dW V_k = 0). Only the effective (projected) factors are formed; dense projection matrices
I - U_k U_k^T are never built. The projected factors are what is exported and merged.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, A_init: torch.Tensor):
        super().__init__()
        assert base.bias is None, "Qwen3 projections have no bias"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        d_out, d_in = base.weight.shape
        assert A_init.shape == (r, d_in)
        self.lora_A = nn.Parameter(A_init.to(torch.float32).clone())
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, dtype=torch.float32))
        self.drop = nn.Dropout(dropout)
        # OPLoRA bases (FP32, frozen, not part of the adapter state); None for plain LoRA
        self.register_buffer("U_k", None, persistent=False)
        self.register_buffer("V_k", None, persistent=False)

    def effective_factors(self):
        A, B = self.lora_A, self.lora_B
        if self.U_k is not None:
            with torch.autocast(device_type=A.device.type, enabled=False):  # projection in FP32
                B = B - self.U_k @ (self.U_k.T @ B)
                A = A - (A @ self.V_k) @ self.V_k.T
        return A, B

    def forward(self, x):
        A, B = self.effective_factors()
        out = self.base(x)
        h = self.drop(x)
        if not torch.is_autocast_enabled(x.device.type):  # e.g. CPU: run the adapter in FP32
            h = h.to(A.dtype)
        return out + (self.scale * F.linear(F.linear(h, A), B)).to(out.dtype)

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        """dW = (alpha/r) B~ A~ in FP32 (projected factors for OPLoRA)."""
        A, B = self.effective_factors()
        return self.scale * (B.float() @ A.float())


def target_modules(model, targets=None):
    """[(weight key in the base state dict, parent module, attribute name)] in a fixed order."""
    out = []
    for i, layer in enumerate(model.model.layers):
        for name in (targets or TARGETS):
            parent = layer.self_attn if name in ("q_proj", "k_proj", "v_proj", "o_proj") else layer.mlp
            sub = "self_attn" if parent is layer.self_attn else "mlp"
            out.append((f"model.layers.{i}.{sub}.{name}.weight", parent, name))
    return out


def inject(model, r: int, alpha: float, dropout: float, seed: int,
           targets=None) -> dict[str, LoRALinear]:
    """Freeze the model and wrap every target projection. A is drawn from a dedicated generator
    in a fixed module order, so runs with the same (r, seed) start from identical factors
    (LoRA-16 and OPLoRA share their initial unprojected factors)."""
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(seed)
    wrapped = {}
    for key, parent, name in target_modules(model, targets):
        base = getattr(parent, name)
        d_in = base.weight.shape[1]
        bound = 1.0 / math.sqrt(d_in)  # == nn.init.kaiming_uniform_(a=sqrt(5)), PEFT's default
        A0 = (torch.rand(r, d_in, generator=gen, dtype=torch.float32) * 2 - 1) * bound
        mod = LoRALinear(base, r, alpha, dropout, A0)
        setattr(parent, name, mod)
        wrapped[key] = mod
    return wrapped


def adapter_state(wrapped: dict[str, LoRALinear]) -> dict[str, torch.Tensor]:
    sd = {}
    for key, m in wrapped.items():
        stem = key[: -len(".weight")]
        sd[f"{stem}.lora_A"] = m.lora_A.detach().cpu()
        sd[f"{stem}.lora_B"] = m.lora_B.detach().cpu()
    return sd


@torch.no_grad()
def load_adapter_state(wrapped: dict[str, LoRALinear], sd: dict[str, torch.Tensor]):
    for key, m in wrapped.items():
        stem = key[: -len(".weight")]
        m.lora_A.copy_(sd[f"{stem}.lora_A"])
        m.lora_B.copy_(sd[f"{stem}.lora_B"])


@torch.no_grad()
def effective_state(wrapped: dict[str, LoRALinear]) -> dict[str, torch.Tensor]:
    """Exported factors: projected A~, B~ for OPLoRA (== A, B for LoRA), FP32."""
    sd = {}
    for key, m in wrapped.items():
        A, B = m.effective_factors()
        stem = key[: -len(".weight")]
        sd[f"{stem}.lora_A"] = A.detach().float().cpu().contiguous()
        sd[f"{stem}.lora_B"] = B.detach().float().cpu().contiguous()
    return sd


@torch.no_grad()
def topk_bases(W0: torch.Tensor, k: int):
    """U_k (d_out x k), V_k (d_in x k) of W0 = U S V^T, FP32."""
    U, S, Vh = torch.linalg.svd(W0.float(), full_matrices=False)
    return U[:, :k].contiguous(), Vh[:k].T.contiguous()


@torch.no_grad()
def attach_bases(wrapped: dict[str, LoRALinear], theta0: dict, k: int, device) -> dict:
    """Compute and attach OPLoRA bases from the FP32 starting weights. Returns overhead stats."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    nbytes = 0
    for key, m in wrapped.items():
        Uk, Vk = topk_bases(theta0[key].to(device), k)
        m.U_k, m.V_k = Uk, Vk
        nbytes += Uk.numel() * Uk.element_size() + Vk.numel() * Vk.element_size()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"basis_preprocess_s": time.perf_counter() - t, "basis_bytes": nbytes,
            "basis_storage": f"{device.type} fp32", "n_bases": len(wrapped), "k": k}


@torch.no_grad()
def subspace_report(wrapped: dict[str, LoRALinear], theta0: dict, k: int, device) -> list[dict]:
    """Per target: fraction of ||dW||_F^2 in the protected output / input subspaces of W0
    (rho_k = ||U_k^T dW||^2 / ||dW||^2 and its input-side analogue) and eta = ||dW||/||W0||.
    For OPLoRA these verify the constraints numerically (should be ~0); for LoRA they show how
    much of the update an OPLoRA projection would have removed. Uses the attached bases if
    present, otherwise computes them."""
    rows = []
    for key, m in wrapped.items():
        W0 = theta0[key].to(device).float()
        if m.U_k is not None:
            Uk, Vk = m.U_k, m.V_k
        else:
            Uk, Vk = topk_bases(W0, k)
        dW = m.delta_weight().to(device)
        d2 = dW.pow(2).sum().item()
        A, B = m.effective_factors()
        eye = torch.eye(k, device=device)
        rows.append({
            "weight": key, "layer": int(key.split(".")[2]), "module": key.split(".")[-2],
            "d_out": W0.shape[0], "d_in": W0.shape[1], "k": k,
            "dW_fro": math.sqrt(d2), "W0_fro": W0.norm().item(),
            "eta": math.sqrt(d2) / W0.norm().item(),
            "rho_out": (Uk.T @ dW).pow(2).sum().item() / d2 if d2 > 0 else float("nan"),
            "rho_in": (dW @ Vk).pow(2).sum().item() / d2 if d2 > 0 else float("nan"),
            "rho_out_chance": k / W0.shape[0], "rho_in_chance": k / W0.shape[1],
            "B_leak": ((Uk.T @ B.float()).norm() / B.float().norm().clamp_min(1e-30)).item(),
            "A_leak": ((A.float() @ Vk).norm() / A.float().norm().clamp_min(1e-30)).item(),
            "Uk_orth_err": (Uk.T @ Uk - eye).abs().max().item(),
            "Vk_orth_err": (Vk.T @ Vk - eye).abs().max().item(),
        })
    return rows
