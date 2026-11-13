"""Config loading and paths for Task 2 (generic helpers are shared with Task 1)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from task1.common import (Cfg, Timer, _deep_merge, _set_dotted, base_argparser, dir_size_bytes,  # noqa: F401
                          env_info, log, read_json, read_jsonl, save_run_config, seed_everything,
                          sha256_bytes, sha256_file, write_json, write_jsonl)

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> Cfg:
    common = yaml.safe_load((CONFIG_DIR / "common.yaml").read_text())
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path.name
    cfg = _deep_merge(common, yaml.safe_load(path.read_text()) or {})
    for ov in overrides or []:
        k, _, v = ov.partition("=")
        _set_dotted(cfg, k.strip(), yaml.safe_load(v))
    for env, key in [("T2_OUT_ROOT", "out_root"), ("T2_SCRATCH_ROOT", "scratch_root"),
                     ("T2_ARTIFACTS_ROOT", "artifacts_root"), ("T2_PRIOR_SEARCH", "prior_search")]:
        if os.environ.get(env):
            cfg["paths"][key] = os.environ[env]
    return Cfg(cfg)


class Paths:
    def __init__(self, cfg: Cfg):
        self.out = Path(cfg.paths.out_root)
        self.scratch = Path(cfg.paths.scratch_root)
        self.artifacts = Path(cfg.paths.artifacts_root)
        self.run_name = cfg.get("run_name", "run")

    # shared across runs
    @property
    def data_out(self) -> Path:      # selected IDs, pair lists, split checks (small, zipped)
        return self.out / "data"

    @property
    def raw(self) -> Path:           # downloaded annotation files
        return self.scratch / "raw"

    @property
    def images(self) -> Path:        # downloaded COCO images
        return self.scratch / "images"

    @property
    def features(self) -> Path:      # FP16 SigLIP features (+ index), kept in the output, not zipped
        return self.artifacts / "features"

    @property
    def thumbs(self) -> Path:        # evaluation-image thumbnails (small, zipped)
        return self.out / "thumbnails"

    @property
    def env_out(self) -> Path:
        return self.out / "env"

    # per run
    @property
    def run_out(self) -> Path:
        return self.out / "runs" / self.run_name

    @property
    def run_art(self) -> Path:
        return self.artifacts / "runs" / self.run_name

    @property
    def ckpt_dir(self) -> Path:
        return self.run_art / "checkpoints"

    @property
    def stage_weights(self) -> Path:
        return self.run_art / "stage_weights"

    @property
    def eval_out(self) -> Path:
        return self.run_out / "final_eval"

    @property
    def report_dir(self) -> Path:
        return self.out / "report"


def dump_yaml(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False))
