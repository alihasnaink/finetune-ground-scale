"""Released MathVista testmini question metadata needed for answer normalisation (CPU, run once; output committed).

    python -m task3.prepare_mathvista_meta --parquet <testmini-00000-of-00001-725687bf7a18d64b.parquet>

The parquet is AI4Math/MathVista at dataset revision REVISION (data/testmini-00000-of-00001-725687bf7a18d64b.parquet,
https://huggingface.co/datasets/AI4Math/MathVista). Only the metadata columns are read (no images, no question
text). Writes task3/data/mathvista_meta.json: per pid the question type, answer type, precision, unit, choices and
the released answer, plus the source revision and file SHA-256. The released answers are used to check that
the supplied gold answers agree (option letters for multiple choice).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pyarrow.parquet as pq

from .common import write_json
from .prepare_splits import sha256

OUT = Path(__file__).resolve().parent / "data" / "mathvista_meta.json"
REVISION = "2b6ad69445fbb5695c9b165475e8decdbeb97747"
COLS = ["pid", "question_type", "answer_type", "precision", "unit", "choices", "answer"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", required=True)
    args = ap.parse_args()
    f = Path(args.parquet)
    df = pq.read_table(f, columns=COLS).to_pandas()
    meta = {}
    for r in df.itertuples(index=False):
        prec = None if r.precision is None or (isinstance(r.precision, float) and math.isnan(r.precision)) \
            else int(r.precision)
        unit = None if r.unit is None or (isinstance(r.unit, float) and math.isnan(r.unit)) else str(r.unit)
        meta[str(r.pid)] = {"question_type": r.question_type, "answer_type": r.answer_type, "precision": prec,
                            "unit": unit, "choices": None if r.choices is None else [str(c) for c in r.choices],
                            "answer": str(r.answer)}
    write_json(OUT, {"dataset": "AI4Math/MathVista", "split": "testmini", "revision": REVISION,
                     "file": "data/testmini-00000-of-00001-725687bf7a18d64b.parquet", "file_sha256": sha256(f),
                     "n": len(meta), "questions": meta})
    print(f"{len(meta)} questions -> {OUT}")


if __name__ == "__main__":
    main()
