"""Config loading, paths, seeding, I/O and environment helpers shared by Task 1 scripts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


# ----------------------------------------------------------------------------- config
class Cfg(dict):
    """dict with attribute access (recursive)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def __setattr__(self, k, v):
        self[k] = v


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> Cfg:
    """Load common.yaml, merge the method config, then apply `key.sub=value` overrides
    (values parsed as YAML, e.g. `train.num_updates=8`, `eval.limit=20`)."""
    common = yaml.safe_load((CONFIG_DIR / "common.yaml").read_text())
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path.name
    method = yaml.safe_load(path.read_text()) or {}
    cfg = _deep_merge(common, method)
    for ov in overrides or []:
        k, _, v = ov.partition("=")
        _set_dotted(cfg, k.strip(), yaml.safe_load(v))
    # Environment overrides for paths (handy outside Kaggle).
    if os.environ.get("T1_OUT_ROOT"):
        cfg["paths"]["out_root"] = os.environ["T1_OUT_ROOT"]
    if os.environ.get("T1_SCRATCH_ROOT"):
        cfg["paths"]["scratch_root"] = os.environ["T1_SCRATCH_ROOT"]
    if os.environ.get("T1_ARTIFACTS_ROOT"):
        cfg["paths"]["artifacts_root"] = os.environ["T1_ARTIFACTS_ROOT"]
    return Cfg(cfg)


def base_argparser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", required=True, help="method config, e.g. task1/configs/full_ft.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="config overrides, e.g. train.num_updates=8 eval.limit=20")
    return p


# ----------------------------------------------------------------------------- paths
class Paths:
    """All filesystem locations used by Task 1. Small results live under out_root
    (persisted, zipped); large artefacts live under scratch_root."""

    def __init__(self, cfg: Cfg):
        self.out = Path(cfg.paths.out_root)
        self.scratch = Path(cfg.paths.scratch_root)
        self.artifacts = Path(cfg.paths.artifacts_root)
        self.run_name = cfg.get("run_name", "run")

    # shared (method-independent)
    @property
    def data_out(self) -> Path:  # selected IDs, filter stats, probe set
        return self.out / "data"

    @property
    def data_scratch(self) -> Path:  # tokenised training cache
        return self.scratch / "data"

    @property
    def env_out(self) -> Path:
        return self.out / "env"

    def eval_dir(self, condition: str) -> Path:
        return self.out / "eval" / condition

    # run specific
    @property
    def run_out(self) -> Path:
        return self.out / "runs" / self.run_name

    @property
    def run_scratch(self) -> Path:
        return self.scratch / "runs" / self.run_name

    @property
    def ckpt_dir(self) -> Path:
        return self.run_scratch / "checkpoints"

    def export_dir(self, stage: str) -> Path:
        """FP16 inference export. Persisted in the Kaggle output (not zipped), so a failure after
        training only costs a re-run of the evaluation, not of the training."""
        return self.artifacts / "exports" / self.run_name / stage

    def delta_file(self, stage: str) -> Path:
        """FP32 down_proj deltas (layers 0/13/27). Persisted in the Kaggle output, not zipped."""
        return self.artifacts / "deltas" / self.run_name / f"{stage}.pt"

    @property
    def diag_out(self) -> Path:
        return self.out / "diagnostics"

    @property
    def report_dir(self) -> Path:
        return self.out / "report"


def condition_name(run_name: str, stage: str) -> str:
    return f"{run_name}_{stage}"


# ----------------------------------------------------------------------------- misc
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_json(path: str | os.PathLike, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    tmp.replace(path)


def read_json(path: str | os.PathLike) -> Any:
    return json.loads(Path(path).read_text())


def write_jsonl(path: str | os.PathLike, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size_bytes(path: str | os.PathLike) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_model_path(cfg: Cfg) -> tuple[str, str | None]:
    """Download (or reuse) the pinned base checkpoint; return (local_path, commit_hash)."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(cfg.model.name, revision=cfg.model.revision,
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt",
                                              "vocab.json", "tokenizer*"])
    commit = Path(local).name if len(Path(local).name) == 40 else None
    return local, commit


def env_info() -> dict:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for mod in ["torch", "transformers", "datasets", "lm_eval", "accelerate", "peft", "numpy",
                "safetensors", "huggingface_hub"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
        info["git_dirty"] = bool(subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True).strip())
    except Exception:
        info["git_commit"] = None
    return info


def save_run_config(cfg: Cfg, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(json.loads(json.dumps(cfg)), sort_keys=False))


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0
