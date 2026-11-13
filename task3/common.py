"""Config loading and the fixed question split for Task 3."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml

from task1.common import (Cfg, Timer, _deep_merge, _set_dotted, base_argparser, env_info, log,  # noqa: F401
                          read_json, read_jsonl, write_json, write_jsonl)

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_config(path, overrides=None) -> Cfg:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path.name
    cfg = yaml.safe_load(path.read_text())
    for ov in overrides or []:
        k, _, v = ov.partition("=")
        _set_dotted(cfg, k.strip(), yaml.safe_load(v))
    for env, key in [("T3_OUT_ROOT", "out_root"), ("T3_PRIOR_SEARCH", "prior_search")]:
        if os.environ.get(env):
            cfg["paths"][key] = os.environ[env]
    return Cfg(cfg)


def make_split(question_ids, seed: int = 6304, dev_size: int = 200) -> dict:
    """Sort IDs lexicographically, permute with default_rng(seed); first dev_size = development."""
    ids = sorted({str(q) for q in question_ids})
    perm = [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]
    return {"seed": seed, "n": len(ids), "order": perm, "dev": perm[:dev_size], "eval": perm[dev_size:]}
