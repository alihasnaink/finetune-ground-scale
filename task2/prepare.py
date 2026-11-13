"""Task 2 data: download annotations, select every split, fetch images, cache SigLIP features.

    python -m task2.prepare --config task2/configs/vlm196.yaml [--steps download select images features]

Selection (seed 6304; train2014 for training, val2014 for development and final test):
  captions      6,000 train images x 3 captions; 96 dev images x 1 caption
  grounding     template presence / counting pairs from COCO instances: training pools (images of
                the caption-training set, sampled during training), 24 dev pairs, 64 test pairs
  VQAv2         complementary pairs with >= 8/10 agreement on each image and different normalised
                consensus answers: train 640/256/640, dev 12/8/12, test 36/24/36 (yes/no/number/other)
All development and final-test items are image-disjoint from one another (across suites and
splits) and from training. Selected IDs are saved under data/, the checks in data/split_checks.json.

Features: SigLIP-base vision tower, deterministic preprocessing, the 196 final patch tokens in
14x14 row-major order with the final layer norm applied exactly once (SiglipVisionModel's
last_hidden_state = post_layernorm(encoder output); verified on the first batch), cached in FP16.
Thumbnails of evaluation images are saved for the qualitative figures.
"""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .common import (Paths, Timer, base_argparser, env_info, load_config, log, read_json,
                     write_json, write_jsonl)
from .vqa_norm import consensus, normalize_answer, original_string_for

COCO_FILES = ["captions_train2014.json", "captions_val2014.json", "instances_train2014.json",
              "instances_val2014.json"]
IRREGULAR_PLURALS = {"person": "people", "mouse": "mice", "sheep": "sheep", "skis": "skis",
                     "scissors": "scissors", "knife": "knives", "broccoli": "broccoli"}
NO_ARTICLE = {"skis", "scissors", "broccoli"}


def plural(name: str) -> str:
    if name in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[name]
    head, _, last = name.rpartition(" ")
    if last in IRREGULAR_PLURALS:
        p = IRREGULAR_PLURALS[last]
    elif last.endswith(("s", "sh", "ch", "x")):
        p = last + "es"
    else:
        p = last + "s"
    return f"{head} {p}".strip()


def with_article(name: str) -> str:
    if name in NO_ARTICLE:
        return name
    return ("an " if name[0] in "aeiou" else "a ") + name


def presence_question(name: str) -> str:
    return f"Does this image contain {with_article(name)}?"


def counting_question(name: str) -> str:
    return f"How many {plural(name)} can you see?"


def img_key(split: str, image_id: int) -> str:
    return f"{split}/{int(image_id)}"


# ----------------------------------------------------------------------------- download
def _fetch(url: str, dest: Path, tries: int = 4) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise RuntimeError(f"download failed: {url}: {e}") from e
            time.sleep(2 * (k + 1))
    return dest


def download(cfg, paths: Paths) -> dict:
    raw = paths.raw
    raw.mkdir(parents=True, exist_ok=True)
    t = Timer()
    rec = {}
    if not all((raw / f).exists() for f in COCO_FILES):
        z = _fetch(cfg.sources.coco_annotations, raw / "annotations_trainval2014.zip")
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if Path(n).name in COCO_FILES:
                    (raw / Path(n).name).write_bytes(zf.read(n))
        z.unlink()
    for f in cfg.sources.vqa_files:
        z = raw / f
        marker = raw / (f + ".done")
        if marker.exists():
            continue
        _fetch(f"{cfg.sources.vqa_base}/{f}", z)
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.endswith(".json"):
                    (raw / Path(n).name).write_bytes(zf.read(n))
        z.unlink()
        marker.write_text("ok")
    for p in sorted(raw.glob("*.json")):
        rec[p.name] = p.stat().st_size
    log(f"annotations ready in {t.elapsed():.0f}s: {sorted(rec)}")
    return {"files": rec, "download_s": t.elapsed()}


# ----------------------------------------------------------------------------- COCO helpers
def load_instances(path: Path):
    """Compact view (only the fields used); the raw JSON is freed immediately."""
    with open(path) as f:
        j = json.load(f)
    cats = {c["id"]: c["name"] for c in j["categories"]}
    images = {im["id"] for im in j["images"]}
    per = defaultdict(lambda: defaultdict(list))  # image -> category -> [(area, iscrowd)]
    for a in j["annotations"]:
        per[a["image_id"]][a["category_id"]].append((float(a["area"]), int(a["iscrowd"])))
    del j
    gc.collect()
    return cats, images, per


def load_captions(path: Path):
    with open(path) as f:
        j = json.load(f)
    rows = sorted((a["id"], a["image_id"], a["caption"]) for a in j["annotations"])
    del j
    caps = defaultdict(list)
    for _, im, c in rows:
        c = " ".join(c.strip().split())
        if c:
            caps[im].append(c)
    return caps


class GroundingIndex:
    """Eligibility of images for presence / counting questions under the label-quality rules."""

    def __init__(self, cats, per, universe, gcfg):
        self.cats = cats
        self.min_area = float(gcfg.min_instance_area)
        self.excl_crowd = bool(gcfg.exclude_crowd)
        self.cmin, self.cmax = int(gcfg.count_min), int(gcfg.count_max)
        self.pos = defaultdict(list)            # category -> images that clearly contain it
        self.neg_ok = {}                        # image -> set of annotated categories
        self.count = defaultdict(lambda: defaultdict(list))  # category -> n -> images
        for im in sorted(universe):
            anns = per.get(im, {})
            self.neg_ok[im] = set(anns)
            for c, inst in anns.items():
                crowd = any(ic for _, ic in inst)
                if crowd and self.excl_crowd:
                    continue
                big = [a for a, ic in inst if not ic and a >= self.min_area]
                if big:
                    self.pos[c].append(im)
                if not crowd and len(big) == len(inst) and self.cmin <= len(inst) <= self.cmax:
                    self.count[c][len(inst)].append(im)
        self.universe = sorted(universe)

    def negatives(self, c):
        return [im for im in self.universe if c not in self.neg_ok[im]]


def sample_grounding(gi: GroundingIndex, kind: str, n: int, rng, split: str, used: set | None,
                     prefix: str) -> list[dict]:
    """n pairs; with `used` given, every image is used at most once (and is added to `used`)."""
    out, seen, tries = [], set(), 0
    cats = sorted(gi.cats)
    neg_cache = {}
    while len(out) < n:
        tries += 1
        if tries > 200 * n + 1000:
            raise RuntimeError(f"could not sample {n} {kind} pairs ({len(out)} found)")
        c = cats[rng.integers(len(cats))]
        name = gi.cats[c]
        if kind == "presence":
            pos = [im for im in gi.pos.get(c, []) if not used or im not in used]
            if c not in neg_cache:
                neg_cache[c] = gi.negatives(c)
            neg = [im for im in neg_cache[c] if not used or im not in used]
            if not pos or not neg:
                continue
            a, b = int(pos[rng.integers(len(pos))]), int(neg[rng.integers(len(neg))])
            items = [(a, "yes"), (b, "no")]
            q = presence_question(name)
        else:
            avail = {k: [im for im in v if not used or im not in used] for k, v in gi.count[c].items()}
            ks = sorted(k for k, v in avail.items() if v)
            if len(ks) < 2:
                continue
            k1, k2 = rng.choice(ks, size=2, replace=False)
            a = int(avail[k1][rng.integers(len(avail[k1]))])
            b = int(avail[k2][rng.integers(len(avail[k2]))])
            if a == b:
                continue
            items = [(a, str(int(k1))), (b, str(int(k2)))]
            q = counting_question(name)
        sig = (kind, c, items[0][0], items[1][0])
        if sig in seen or (used is not None and (items[0][0] in used or items[1][0] in used)):
            continue
        seen.add(sig)
        if used is not None:
            used.update([items[0][0], items[1][0]])
        pid = f"{prefix}_{kind[:4]}_{len(out):05d}"
        out.append({"pair_id": pid, "suite": "grounding", "qtype": kind, "category": name,
                    "question": q,
                    "items": [{"image": img_key(split, im), "answer": ans, "canonical": ans,
                               "refs": [ans]} for im, ans in items]})
    return out


# ----------------------------------------------------------------------------- VQAv2
def vqa_eligible(raw: Path, split: str, tok, cfg) -> dict[str, list[dict]]:
    tag = "train2014" if split == "train2014" else "val2014"
    with open(raw / f"v2_OpenEnded_mscoco_{tag}_questions.json") as f:
        qs = {q["question_id"]: {"question": q["question"]} for q in json.load(f)["questions"]}
    gc.collect()
    with open(raw / f"v2_mscoco_{tag}_annotations.json") as f:
        anns = {a["question_id"]: {"answer_type": a["answer_type"], "image_id": a["image_id"],
                                   "answers": [{"answer": x["answer"]} for x in a["answers"]]}
                for a in json.load(f)["annotations"]}
    gc.collect()
    with open(raw / f"v2_mscoco_{tag}_complementary_pairs.json") as f:
        pairs = json.load(f)
    caps, agree = cfg.data.caps, int(cfg.data.vqa.min_agreement)
    out = defaultdict(list)
    stats = defaultdict(int)
    for q1, q2 in pairs:
        stats["pairs"] += 1
        if q1 not in qs or q2 not in qs or q1 not in anns or q2 not in anns:
            stats["missing"] += 1
            continue
        a1, a2 = anns[q1], anns[q2]
        if qs[q1]["question"].strip().lower() != qs[q2]["question"].strip().lower():
            stats["question_differs"] += 1
            continue
        if a1["answer_type"] != a2["answer_type"] or a1["image_id"] == a2["image_id"]:
            stats["type_or_image"] += 1
            continue
        items = []
        for qid, a in ((q1, a1), (q2, a2)):
            refs = [x["answer"] for x in a["answers"]]
            cons, n = consensus(refs)
            items.append((qid, a, refs, cons, n))
        if any(n < agree or not cons for *_, cons, n in items):
            stats["low_agreement"] += 1
            continue
        if items[0][3] == items[1][3]:
            stats["same_consensus"] += 1
            continue
        question = qs[q1]["question"].strip()
        tgt = [original_string_for(refs, cons) for _, _, refs, cons, _ in items]
        if len(tok(question, add_special_tokens=False)["input_ids"]) > caps.question or any(
                len(tok(t, add_special_tokens=False)["input_ids"]) > caps.answer_target for t in tgt):
            stats["over_token_cap"] += 1
            continue
        stats["eligible"] += 1
        out[a1["answer_type"]].append({
            "pair_id": f"vqa_{q1}_{q2}", "suite": "vqa", "qtype": a1["answer_type"],
            "question": question, "question_ids": [q1, q2],
            "items": [{"image": img_key(split, a["image_id"]), "question_id": qid, "answer": t,
                       "canonical": cons, "refs": refs, "agreement": n}
                      for (qid, a, refs, cons, n), t in zip(items, tgt)]})
    log(f"VQAv2 {split}: {dict(stats)}; eligible by type { {k: len(v) for k, v in out.items()} }")
    return out, dict(stats)


def pick_vqa(elig: dict, counts: dict, rng, used: set | None, split_name: str) -> list[dict]:
    out = []
    for qtype, n in counts.items():
        cand = elig.get(qtype, [])
        order = rng.permutation(len(cand))
        got = 0
        for i in order:
            p = cand[int(i)]
            ims = [it["image"] for it in p["items"]]
            if used is not None and any(im in used for im in ims):
                continue
            if used is not None:
                used.update(ims)
            out.append(p)
            got += 1
            if got == n:
                break
        if got < n:
            raise RuntimeError(f"VQAv2 {split_name}: only {got}/{n} {qtype} pairs available")
    return out


def mismatch_map(pairs: list[dict], rng) -> dict:
    """Fixed unrelated-image mapping: each question gets the same-position image of another pair
    of the suite (a derangement over pairs, so neither image of its own pair is used)."""
    n = len(pairs)
    while True:
        perm = rng.permutation(n)
        if all(perm[i] != i for i in range(n)):
            break
    out = {}
    for i, p in enumerate(pairs):
        other = pairs[int(perm[i])]
        for m in range(2):
            assert other["items"][m]["image"] not in {it["image"] for it in p["items"]}
            out[f"{p['pair_id']}#{m}"] = other["items"][m]["image"]
    return out


# ----------------------------------------------------------------------------- selection
def select(cfg, paths: Paths) -> dict:
    from transformers import AutoTokenizer

    rng = np.random.default_rng(int(cfg.seed))
    raw, d = paths.raw, cfg.data
    tok = AutoTokenizer.from_pretrained(cfg.models.language)
    log("loading COCO annotations")
    cats_tr, imgs_tr, per_tr = load_instances(raw / "instances_train2014.json")
    cats_va, imgs_va, per_va = load_instances(raw / "instances_val2014.json")
    caps_tr = load_captions(raw / "captions_train2014.json")
    caps_va = load_captions(raw / "captions_val2014.json")
    used_eval: set = set()  # every val2014 image used by any dev / test item (image keys)

    # 1) evaluation sets first (most constrained), all image-disjoint
    vqa_va, vqa_va_stats = vqa_eligible(raw, "val2014", tok, cfg)
    vqa_test = pick_vqa(vqa_va, dict(d.vqa.test), rng, used_eval, "test")
    vqa_dev = pick_vqa(vqa_va, dict(d.vqa.dev), rng, used_eval, "dev")
    g = d.grounding
    va_universe = [im for im in sorted(imgs_va) if im in caps_va]
    gi_va = GroundingIndex(cats_va, per_va, va_universe, g)
    used_ids = set()  # integer ids mirror for the grounding sampler

    def sync():
        used_ids.update(int(k.split("/")[1]) for k in used_eval)

    sync()
    g_test = (sample_grounding(gi_va, "presence", g.test_presence, rng, "val2014", used_ids, "gtest")
              + sample_grounding(gi_va, "counting", g.test_counting, rng, "val2014", used_ids, "gtest"))
    g_dev = (sample_grounding(gi_va, "presence", g.dev_presence, rng, "val2014", used_ids, "gdev")
             + sample_grounding(gi_va, "counting", g.dev_counting, rng, "val2014", used_ids, "gdev"))
    for p in g_test + g_dev:
        used_eval.update(it["image"] for it in p["items"])
    free_va = [im for im in va_universe if im not in used_ids]
    dev_cap_ims = [free_va[int(i)] for i in rng.choice(len(free_va), size=int(d.caption_dev_images),
                                                        replace=False)]
    cap_dev = [{"image": img_key("val2014", im),
                "caption": caps_va[im][int(rng.integers(len(caps_va[im])))]} for im in dev_cap_ims]
    used_eval.update(c["image"] for c in cap_dev)

    # 2) training sets (train2014 only)
    tr_ok = [im for im in sorted(imgs_tr) if len(caps_tr.get(im, [])) >= int(d.captions_per_train_image)]
    cap_ims = sorted(tr_ok[int(i)] for i in rng.choice(len(tr_ok), size=int(d.caption_train_images),
                                                         replace=False))
    cap_train = []
    for im in cap_ims:
        sel = rng.choice(len(caps_tr[im]), size=int(d.captions_per_train_image), replace=False)
        cap_train += [{"image": img_key("train2014", im), "caption": caps_tr[im][int(j)]} for j in sorted(sel)]
    gi_tr = GroundingIndex(cats_tr, per_tr, cap_ims, g)
    g_pres = sample_grounding(gi_tr, "presence", g.train_pool_presence, rng, "train2014", None, "gtrain")
    g_cnt = sample_grounding(gi_tr, "counting", g.train_pool_counting, rng, "train2014", None, "gtrain")
    vqa_tr, vqa_tr_stats = vqa_eligible(raw, "train2014", tok, cfg)
    vqa_train = pick_vqa(vqa_tr, dict(d.vqa.train), rng, None, "train")

    mm_test = {"grounding": mismatch_map(g_test, rng), "vqa": mismatch_map(vqa_test, rng)}

    # 3) checks
    def ims_of(pairs):
        return [it["image"] for p in pairs for it in p["items"]]

    eval_sets = {"vqa_test": ims_of(vqa_test), "vqa_dev": ims_of(vqa_dev),
                 "grounding_test": ims_of(g_test), "grounding_dev": ims_of(g_dev),
                 "caption_dev": [c["image"] for c in cap_dev]}
    train_sets = {"caption_train": [c["image"] for c in cap_train],
                  "grounding_train": ims_of(g_pres + g_cnt), "vqa_train": ims_of(vqa_train)}
    all_eval = [im for v in eval_sets.values() for im in v]
    train_all = {im for v in train_sets.values() for im in v}
    checks = {
        "eval_images_unique_across_all_dev_test_sets": len(all_eval) == len(set(all_eval)),
        "eval_disjoint_from_training": not (set(all_eval) & train_all),
        "eval_all_val2014": all(im.startswith("val2014/") for im in all_eval),
        "train_all_train2014": all(im.startswith("train2014/") for im in train_all),
        "pairs_have_distinct_images": all(p["items"][0]["image"] != p["items"][1]["image"]
                                          for p in vqa_test + vqa_dev + vqa_train + g_test + g_dev
                                          + g_pres + g_cnt),
        "pairs_have_different_answers": all(p["items"][0]["canonical"] != p["items"][1]["canonical"]
                                            for p in vqa_test + vqa_dev + vqa_train + g_test + g_dev
                                            + g_pres + g_cnt),
        "vqa_min_agreement_ok": all(it["agreement"] >= int(d.vqa.min_agreement)
                                    for p in vqa_test + vqa_dev + vqa_train for it in p["items"]),
        "mismatch_maps_exclude_own_pair": True,  # asserted in mismatch_map
        "counts": {k: len(v) for k, v in {**eval_sets, **train_sets}.items()},
        "pairs": {"vqa_train": len(vqa_train), "vqa_dev": len(vqa_dev), "vqa_test": len(vqa_test),
                  "grounding_train_presence_pool": len(g_pres), "grounding_train_counting_pool": len(g_cnt),
                  "grounding_dev": len(g_dev), "grounding_test": len(g_test),
                  "caption_train_examples": len(cap_train), "caption_dev_examples": len(cap_dev)},
    }
    bad = [k for k, v in checks.items() if v is False]
    assert not bad, f"split checks failed: {bad}"

    out = paths.data_out
    files = {"captions_train.jsonl": cap_train, "captions_dev.jsonl": cap_dev,
             "grounding_train_presence.jsonl": g_pres, "grounding_train_counting.jsonl": g_cnt,
             "grounding_dev.jsonl": g_dev, "grounding_test.jsonl": g_test,
             "vqa_train.jsonl": vqa_train, "vqa_dev.jsonl": vqa_dev, "vqa_test.jsonl": vqa_test}
    h = hashlib.sha256()
    for name, rows in files.items():
        write_jsonl(out / name, rows)
        h.update((out / name).read_bytes())
    write_json(out / "mismatch_test.json", mm_test)
    h.update((out / "mismatch_test.json").read_bytes())
    needed = sorted(train_all | set(all_eval), key=lambda k: (k.split("/")[0], int(k.split("/")[1])))
    write_json(out / "images_needed.json", {"train": sorted(train_all), "eval": sorted(set(all_eval)),
                                            "all": needed})
    summary = {"selection_sha256": h.hexdigest(), "seed": int(cfg.seed), "checks": checks,
               "vqa_filter_stats": {"val2014": vqa_va_stats, "train2014": vqa_tr_stats},
               "grounding_rules": {"min_instance_area": g.min_instance_area,
                                   "exclude_crowd": g.exclude_crowd,
                                   "count_range": [g.count_min, g.count_max],
                                   "presence_negative": "no annotation of the category at all",
                                   "presence_positive": "a non-crowd instance with area >= min_instance_area",
                                   "counting": "all instances of the category non-crowd and >= min area; "
                                               "pairs have different counts"},
               "n_images": len(needed)}
    write_json(out / "split_checks.json", summary)
    log(f"selection done: {len(needed)} images, sha {summary['selection_sha256'][:12]}; checks ok")
    return summary


# ----------------------------------------------------------------------------- images
def image_path(cfg, paths: Paths, key: str) -> Path:
    split, iid = key.split("/")
    name = f"COCO_{split}_{int(iid):012d}.jpg"
    for root in cfg.paths.get("coco_image_roots") or []:
        for cand in (Path(root) / split / name, Path(root) / name):
            if cand.exists():
                return cand
    return paths.images / split / name


def fetch_images(cfg, paths: Paths) -> dict:
    keys = read_json(paths.data_out / "images_needed.json")["all"]
    t = Timer()
    todo = [k for k in keys if not image_path(cfg, paths, k).exists()]
    log(f"images: {len(keys)} needed, {len(todo)} to download")

    def one(k):
        split, iid = k.split("/")
        _fetch(cfg.sources.coco_image_url.format(split=split, image_id=int(iid)), image_path(cfg, paths, k))

    with ThreadPoolExecutor(int(cfg.sources.download_threads)) as ex:
        for i, _ in enumerate(ex.map(one, todo)):
            if (i + 1) % 1000 == 0:
                log(f"  downloaded {i + 1}/{len(todo)}")
    return {"n_images": len(keys), "downloaded": len(todo), "fetch_s": t.elapsed()}


# ----------------------------------------------------------------------------- features
def find_prior_features(cfg, sha: str) -> Path | None:
    root = Path(cfg.paths.get("prior_search") or "/nonexistent")
    if not root.exists():
        return None
    for meta in root.rglob("features/meta.json"):
        try:
            if read_json(meta).get("selection_sha256") == sha:
                return meta.parent
        except Exception:  # noqa: BLE001
            continue
    return None


def cache_features(cfg, paths: Paths) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, SiglipVisionModel

    sha = read_json(paths.data_out / "split_checks.json")["selection_sha256"]
    fdir = paths.features
    if (fdir / "meta.json").exists() and read_json(fdir / "meta.json").get("selection_sha256") == sha:
        log("features already cached for this selection")
        return read_json(fdir / "meta.json")
    prior = find_prior_features(cfg, sha)
    if prior is not None:
        log(f"reusing features from {prior}")
        shutil.rmtree(fdir, ignore_errors=True)
        shutil.copytree(prior, fdir)
        meta = read_json(fdir / "meta.json") | {"copied_from": str(prior)}
        write_json(fdir / "meta.json", meta)
        return meta

    need = read_json(paths.data_out / "images_needed.json")
    keys, eval_keys = need["all"], set(need["eval"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(cfg.models.vision)
    enc = SiglipVisionModel.from_pretrained(cfg.models.vision, dtype=torch.float32).to(dev).eval()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    fdir.mkdir(parents=True, exist_ok=True)
    tmp = fdir / "features.npy.tmp"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(keys), 196, 768))
    paths.thumbs.mkdir(parents=True, exist_ok=True)
    bs = int(cfg.features.batch_size)
    t = Timer()
    ln_check = None
    with torch.no_grad():
        for s in range(0, len(keys), bs):
            batch = keys[s:s + bs]
            ims = []
            for k in batch:
                im = Image.open(image_path(cfg, paths, k)).convert("RGB")
                if k in eval_keys:
                    th = im.copy()
                    th.thumbnail((int(cfg.features.thumbnail_px),) * 2)
                    th.save(paths.thumbs / (k.replace("/", "_") + ".jpg"), quality=85)
                ims.append(im)
            px = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
            o = enc(pixel_values=px, output_hidden_states=ln_check is None)
            h = o.last_hidden_state
            assert h.shape[1:] == (196, 768), h.shape
            if ln_check is None:  # final LN applied exactly once
                pre = o.hidden_states[-1]
                post_ln = [m for n, m in enc.named_modules() if n.endswith("post_layernorm")]
                assert len(post_ln) == 1, "expected exactly one final layer norm"
                re_ln = post_ln[0](pre)
                ln_check = {"last_hidden_state_equals_post_layernorm_of_encoder_output":
                            bool(torch.allclose(re_ln, h, atol=1e-5)),
                            "last_hidden_state_differs_from_encoder_output":
                            bool(not torch.allclose(pre, h, atol=1e-3)),
                            "token_mean_abs_mean": float(h.mean(-1).abs().mean()),
                            "token_std_mean": float(h.std(-1).mean())}
                assert ln_check["last_hidden_state_equals_post_layernorm_of_encoder_output"]
            arr[s:s + len(batch)] = h.float().cpu().numpy().astype(np.float16)
            if (s // bs) % 20 == 0:
                log(f"  features {s + len(batch)}/{len(keys)}")
    arr.flush()
    del arr
    tmp.replace(fdir / "features.npy")
    meta = {"selection_sha256": sha, "model": cfg.models.vision, "n_images": len(keys),
            "shape": [len(keys), 196, 768], "dtype": "float16", "token_order": "14x14 row-major",
            "preprocessing": proc.to_dict(), "ln_check": ln_check,
            "cache_time_s": t.elapsed(), "images_per_s": len(keys) / max(t.elapsed(), 1e-9),
            "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else None,
            "bytes": (fdir / "features.npy").stat().st_size, "device": str(dev)}
    write_json(fdir / "index.json", keys)
    write_json(fdir / "meta.json", meta)
    del enc
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    log(f"features cached: {len(keys)} images in {t.elapsed() / 60:.1f} min "
        f"({meta['bytes'] / 2**30:.2f} GiB); vision encoder released")
    return meta


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--steps", nargs="*", default=["download", "select", "images", "features"])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    paths = Paths(cfg)
    paths.out.mkdir(parents=True, exist_ok=True)
    write_json(paths.env_out / "env_prepare.json", env_info())
    res = {}
    if "download" in args.steps:
        res["download"] = download(cfg, paths)
    if "select" in args.steps:
        res["select"] = select(cfg, paths)
    if "features" in args.steps:
        sha = read_json(paths.data_out / "split_checks.json")["selection_sha256"]
        have = ((paths.features / "meta.json").exists()
                and read_json(paths.features / "meta.json").get("selection_sha256") == sha)
        if not have and find_prior_features(cfg, sha) is None and "images" in args.steps:
            res["images"] = fetch_images(cfg, paths)
        res["features"] = cache_features(cfg, paths)
    elif "images" in args.steps:
        res["images"] = fetch_images(cfg, paths)
    prev = read_json(paths.data_out / "prepare_resources.json") if (paths.data_out / "prepare_resources.json").exists() else {}
    write_json(paths.data_out / "prepare_resources.json", prev | {k: (v if k != "select" else
                                                                     {"selection_sha256": v["selection_sha256"]})
                                                                 for k, v in res.items()})
    log("prepare done")


if __name__ == "__main__":
    main()
