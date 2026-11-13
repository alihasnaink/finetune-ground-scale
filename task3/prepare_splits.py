"""Fixed question splits and the live MATH subset (CPU, run once locally; outputs are committed).

    python -m task3.prepare_splits [--data-root task3_data]

For each dataset: sort the 1,000 question IDs lexicographically, permute with
numpy.random.default_rng(6304); the first 200 are development, the remaining 800 held-out evaluation.
The live subset is the first 50 MATH evaluation questions. Writes, under task3/data/:
  split_math.json, split_mathvista.json   ordered dev / eval IDs, seed, source file SHA-256
  live_questions.json                     the 50 live questions (ID, problem text, gold answer)
The raw supplied data (task3_data/) is not committed; these small files make the live run on
Kaggle independent of it.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from .common import make_split, write_json

OUT = Path(__file__).resolve().parent / "data"
SEED, DEV, N_LIVE = 6304, 200, 50


def sha256(f: Path) -> str:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="task3_data")
    args = ap.parse_args()
    root = Path(args.data_root)
    for name, folder in [("math", "qwen25-math"), ("mathvista", "qwen25vl-mathvista")]:
        f = root / folder / "generations.parquet"
        df = pd.read_parquet(f, columns=["dataset", "sample_id", "rollout_id", "question_text", "gold_answer"])
        assert df.groupby("sample_id")["rollout_id"].nunique().eq(32).all(), "expected 32 rollouts per question"
        assert df.groupby("sample_id")[["question_text", "gold_answer"]].nunique().le(1).all().all()
        split = make_split(df["sample_id"].unique(), SEED, DEV)
        assert split["n"] == 1000 and len(set(split["dev"]) & set(split["eval"])) == 0
        write_json(OUT / f"split_{name}.json",
                   {"dataset": name, "seed": SEED, "rule": "sorted IDs permuted with numpy default_rng(6304); "
                    "first 200 development, remaining 800 evaluation", "n": split["n"],
                    "source": str(f), "source_sha256": sha256(f), "dev": split["dev"], "eval": split["eval"]})
        print(f"{name}: {len(split['dev'])} dev / {len(split['eval'])} eval -> {OUT / f'split_{name}.json'}")
        if name == "math":
            q = df.drop_duplicates("sample_id").set_index("sample_id")
            live = [{"live_index": i, "qid": s, "problem": q.at[s, "question_text"], "gold": q.at[s, "gold_answer"],
                     "dataset": q.at[s, "dataset"]} for i, s in enumerate(split["eval"][:N_LIVE])]
            write_json(OUT / "live_questions.json", live)
            print(f"live subset: {len(live)} questions, first {live[0]['qid']} -> {OUT / 'live_questions.json'}")


if __name__ == "__main__":
    main()
