"""Import results of earlier Task 1 notebooks (e.g. the shared baseline and Full FT) so that this
notebook's report can show changes from baseline and cross-method comparisons.

    python -m task1.import_prior --config task1/configs/l2sp.yaml --search /kaggle/input

Searches `--search` for earlier Task 1 output folders (any directory containing
data/data_stats.json) and for task1 report zips. A prior output is used only if its training-data
SHA-256 equals this notebook's (same 40k examples, same tokenisation). Copies, without overwriting:
  eval/<condition>/                 every condition not produced by the current run
  runs/<run>/                       every run other than the current one
  diagnostics/*_<run>.csv           diagnostics of other runs
  env/*.json
Records what was imported in data/imported_from.json.

Own run: if an attached output already holds a *finished* training of the current run
(runs/<run>/train_summary.json plus the FP16 exports of every stage in the sibling
`<out>_artifacts/exports/<run>/` folder), that run is imported too: runs/<run>, its exports and
deltas, and its finished evaluations. task1.train then skips training and only the missing
evaluations run. This recovers a notebook that failed after training. Model files of *other* runs
are never imported.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

from .common import Paths, base_argparser, load_config, log, read_json, write_json


def find_roots(search: Path, extract_to: Path) -> list[Path]:
    roots = [p.parent.parent for p in search.rglob("data/data_stats.json")]
    for z in search.rglob("*.zip"):
        if "task1" not in z.name:
            continue
        dest = extract_to / z.stem
        if not dest.exists():
            with zipfile.ZipFile(z) as zf:
                names = [n for n in zf.namelist() if not n.endswith((".pt", ".safetensors", ".bin"))]
                zf.extractall(dest, members=names)
        roots += [p.parent.parent for p in dest.rglob("data/data_stats.json")]
    return sorted(set(roots))


def _copy(src: Path, dest: Path):
    if dest.exists():
        shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest) if src.is_dir() else shutil.copy2(src, dest)


def import_own_run(root: Path, art: Path, paths: Paths, run: str, stages, imported: list):
    """Finished training of the current run from an earlier output (replaces any partial local state)."""
    _copy(root / "runs" / run, paths.run_out)
    imported.append(f"runs/{run}")
    for st in stages:
        _copy(art / "exports" / run / st, paths.export_dir(st))
        imported.append(f"artifacts/exports/{run}/{st}")
    if (art / "deltas" / run).exists():
        _copy(art / "deltas" / run, paths.delta_file(stages[0]).parent)
        imported.append(f"artifacts/deltas/{run}")
    for d in sorted((root / "eval").glob(f"{run}_*")):
        if (d / "metrics.json").exists() and not (paths.out / "eval" / d.name).exists():
            _copy(d, paths.out / "eval" / d.name)
            imported.append(f"eval/{d.name}")


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--search", default="/kaggle/input")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    paths = Paths(cfg)
    search = Path(args.search)
    own_stats = paths.data_out / "data_stats.json"
    assert own_stats.exists(), "run task1.prepare first"
    own_sha = read_json(own_stats)["data_sha256"]
    record = {"search": str(search), "data_sha256": own_sha, "sources": []}
    if not search.exists():
        log(f"{search} does not exist; nothing to import")
        write_json(paths.data_out / "imported_from.json", record)
        return

    stages = list(cfg.get("stages", ["final"]))
    own_done = (paths.run_out / "train_summary.json").exists() and all(
        (paths.export_dir(st) / "EXPORT_DONE.json").exists() for st in stages)
    extract = Path(tempfile.mkdtemp(prefix="t1_import_", dir=str(paths.scratch)
                                    if paths.scratch.exists() else None))
    run = cfg.run_name
    for root in find_roots(search, extract):
        if root.resolve() == paths.out.resolve():
            continue
        sha = read_json(root / "data" / "data_stats.json").get("data_sha256")
        src = {"path": str(root), "data_sha256": sha, "imported": []}
        if sha != own_sha:
            src["skipped"] = "training-data hash differs (e.g. a smoke-test output)"
            log(f"skip {root}: data hash {str(sha)[:12]} != {own_sha[:12]}")
            record["sources"].append(src)
            continue
        art = root.parent / f"{root.name}_artifacts"
        if not own_done and (root / "runs" / run / "train_summary.json").exists() and all(
                (art / "exports" / run / st / "EXPORT_DONE.json").exists() for st in stages):
            import_own_run(root, art, paths, run, stages, src["imported"])
            own_done = True
            log(f"imported finished training of '{run}' from {root}: training will be skipped")
        for d in sorted((root / "eval").glob("*")):
            dest = paths.out / "eval" / d.name
            if d.is_dir() and (d / "metrics.json").exists() and not d.name.startswith(f"{run}_") \
                    and not dest.exists():
                shutil.copytree(d, dest)
                src["imported"].append(f"eval/{d.name}")
        for d in sorted((root / "runs").glob("*")):
            dest = paths.out / "runs" / d.name
            if d.is_dir() and d.name != run and not dest.exists():
                shutil.copytree(d, dest)
                src["imported"].append(f"runs/{d.name}")
        for f in sorted((root / "diagnostics").glob("*.csv")):
            dest = paths.out / "diagnostics" / f.name
            if not f.stem.endswith(f"_{run}") and not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                src["imported"].append(f"diagnostics/{f.name}")
        for f in sorted((root / "env").glob("*.json")):
            dest = paths.env_out / f.name
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                src["imported"].append(f"env/{f.name}")
        log(f"imported from {root}: {src['imported'] or 'nothing new'}")
        record["sources"].append(src)
    shutil.rmtree(extract, ignore_errors=True)
    write_json(paths.data_out / "imported_from.json", record)
    have = sorted(p.name for p in (paths.out / "eval").glob("*")) if (paths.out / "eval").exists() else []
    log(f"evaluation conditions now available: {have}")
    if "baseline" not in have:
        log("WARNING: no baseline found - the report will show scores without changes from baseline. "
            "Attach the Full FT notebook output as an input to get them.")


if __name__ == "__main__":
    main()
