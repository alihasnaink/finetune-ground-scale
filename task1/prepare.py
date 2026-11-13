"""Step 0 for every Task 1 notebook: download the base model, build/verify the training
data, the activation probe set and the qualitative prompt list, and prefetch the
evaluation datasets.

    python -m task1.prepare --config task1/configs/full_ft.yaml
"""

from __future__ import annotations

from pathlib import Path

from .common import (Paths, base_argparser, env_info, load_config, log, read_json,
                     resolve_model_path, save_run_config, seed_everything, write_json)
from .data import build_probe_set, build_training_data, load_qualitative_prompts


def main():
    args = base_argparser(__doc__).parse_args()
    cfg = load_config(args.config, args.set)
    paths = Paths(cfg)
    seed_everything(int(cfg.seed))
    paths.out.mkdir(parents=True, exist_ok=True)
    paths.scratch.mkdir(parents=True, exist_ok=True)

    model_path, commit = resolve_model_path(cfg)
    info = env_info() | {"base_model": cfg.model.name, "base_model_commit": commit,
                         "base_model_path": model_path}
    write_json(paths.env_out / f"env_{cfg.run_name}.json", info)
    save_run_config(cfg, paths.run_out / "config.yaml")
    log(f"base model {cfg.model.name} @ {commit} -> {model_path}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)

    cache = paths.data_scratch / "train_tokens.npz"
    stats_path = paths.data_out / "data_stats.json"
    if cache.exists() and stats_path.exists() and \
            read_json(stats_path).get("num_selected") == int(cfg.data.num_examples):
        log(f"training data cache found ({read_json(stats_path)['data_sha256'][:12]}); skipping rebuild")
    else:
        build_training_data(cfg, tok, paths)

    if not (paths.data_out / "probe_set.json").exists():
        build_probe_set(cfg, tok, paths)
    write_json(paths.data_out / "qualitative_prompts.json", load_qualitative_prompts(cfg))

    # Prefetch evaluation datasets so parallel evaluation workers do not race on downloads.
    from lm_eval.tasks import TaskManager, get_task_dict

    tasks = get_task_dict([cfg.eval.gsm8k_task, *cfg.eval.mc_tasks], TaskManager())
    for name, t in tasks.items():
        docs = t.test_docs() if t.has_test_docs() else t.validation_docs()
        log(f"eval task {name}: {len(list(docs))} docs")
    log("prepare done")


if __name__ == "__main__":
    main()
