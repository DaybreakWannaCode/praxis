"""Build the paired VIVA calibration slice: images for g_v, text situations for g_t.

VIVA (Hu et al., EMNLP 2024) ships `VIVA_annotation.json` on HuggingFace but *not* the
images — those are fetched from their original `image_url`s, which are third-party hosts
and therefore rot. Everything here is written around that: download what we can, verify
each file actually decodes, and build the pool only from items that survived, reporting
the yield rather than quietly shrinking.

The text channel
----------------
Default is VIVA's own `situation_description` — "A person improperly disposes of a plastic
bottle by throwing it out of a car window onto a scenic rural road." It is a scene
summary, it does not name or hint at the correct action, and critically it is *fixed
benchmark data*: independent of whichever checkpoint is being measured. That matters for
Task 1, where the measured model changes between checkpoints and the text channel must
not move with it.

`text_channel: self_caption` regenerates captions with `data.caption_model` instead. It is
kept because comparing the two answers a real question (does Lambda depend on where the
text channel came from?), but it is not the default: a caption written by the model under
test shares that model's visual encoding, which can inflate alignment.

Fields used from each record: image_url, image_file, index, action_list, answer,
situation_description, category.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import random
from typing import Any

from .config import rel
from .gradient import CalibItem
from .rewards import option_labels

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


# ------------------------------------------------------------------ annotation


def fetch_annotation(cfg) -> list[dict]:
    """Download VIVA_annotation.json from HuggingFace (cached by huggingface_hub)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=cfg.data.hf_repo,
        filename=cfg.data.annotation_file,
        repo_type="dataset",
    )
    with open(path) as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"unexpected annotation payload at {path}")
    return records


# ------------------------------------------------------------------ images


def _verify_image(path: str, min_side: int = 48) -> bool:
    """A file on disk is not an image until PIL has decoded it.

    Dead links commonly return an HTML error page or a 1x1 placeholder with a 200 status,
    so both a full decode and a size floor are needed.
    """
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
        return w >= min_side and h >= min_side
    except Exception:
        return False


def _download_one(rec: dict, out_dir: str, timeout: float, retries: int) -> tuple[int, bool, str]:
    import requests

    idx = rec.get("index")
    dest = os.path.join(out_dir, _image_name(rec))
    if os.path.exists(dest) and _verify_image(dest):
        return idx, True, "cached"

    url = rec.get("image_url")
    if not isinstance(url, str) or not url.strip():
        return idx, False, "no url"          # NaN floats appear here too
    last = "?"
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": _UA})
            if r.status_code != 200 or not r.content:
                last = f"http {r.status_code}"
                continue
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(r.content)
            if _verify_image(tmp):
                os.replace(tmp, dest)
                return idx, True, "downloaded"
            os.remove(tmp)
            last = "undecodable"
        except Exception as e:  # network flake, DNS, TLS, truncated body
            last = type(e).__name__
    return idx, False, last


def download_images(cfg, records: list[dict], *, workers: int = 16, timeout: float = 15.0,
                    retries: int = 2, limit: int | None = None) -> dict[int, bool]:
    """Fetch every image we might need. Returns {index: ok}."""
    out_dir = rel(cfg.data.images_dir)
    os.makedirs(out_dir, exist_ok=True)
    todo = records if limit is None else records[:limit]

    status: dict[int, bool] = {}
    reasons: dict[str, int] = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_download_one, r, out_dir, timeout, retries) for r in todo]
        for k, fut in enumerate(cf.as_completed(futs), 1):
            idx, ok, why = fut.result()
            status[idx] = ok
            reasons[why] = reasons.get(why, 0) + 1
            if k % 100 == 0 or k == len(futs):
                got = sum(status.values())
                print(f"  images {k}/{len(futs)}  ok={got} ({got/max(k,1):.0%})", flush=True)

    print(f"  outcomes: {dict(sorted(reasons.items(), key=lambda kv: -kv[1]))}")
    return status


# ------------------------------------------------------------------ calibration set


def _clean_str(v) -> str:
    """Coerce a JSON field to a string by TYPE, never by truthiness.

    VIVA's annotation carries NaN floats where some string fields are absent, and
    `x or ""` does not catch that: float('nan') is truthy, so the NaN sails through and
    blows up on the first .strip(). Type-checking is the only safe test here.
    """
    return v.strip() if isinstance(v, str) else ""


def _image_name(rec: dict) -> str:
    f = rec.get("image_file")
    return f if isinstance(f, str) and f.strip() else f"{rec.get('index')}.jpg"


def _usable(rec: dict, images_dir: str, text_field: str) -> tuple[bool, str]:
    opts = rec.get("action_list")
    if not isinstance(opts, list) or len(opts) < 2:
        return False, "action_list missing or too short"
    if not all(isinstance(o, str) and o.strip() for o in opts):
        return False, "action_list has non-string entries"
    labels = option_labels(opts)
    ans = _clean_str(rec.get("answer")).upper()
    if ans not in set(labels):
        return False, "answer not among option labels"
    if len(set(labels)) != len(labels):
        return False, "duplicate option labels"
    if len(_clean_str(rec.get(text_field))) < 20:
        return False, f"{text_field} missing or too short"
    if not os.path.exists(os.path.join(images_dir, _image_name(rec))):
        return False, "image missing"
    return True, ""


def build_calibration(cfg, *, records: list[dict] | None = None,
                      download: bool = True, limit: int | None = None) -> str:
    """Assemble and write data/viva_calibration.json. Returns the path."""
    images_dir = rel(cfg.data.images_dir)
    os.makedirs(images_dir, exist_ok=True)
    if records is None:
        records = fetch_annotation(cfg)
    print(f"VIVA annotation: {len(records)} records")

    if download:
        download_images(cfg, records, limit=limit)

    text_field = "situation_description"
    kept, rejected = [], {}
    for rec in records:
        ok, why = _usable(rec, images_dir, text_field)
        if ok:
            kept.append(rec)
        else:
            rejected[why] = rejected.get(why, 0) + 1

    print(f"usable items: {len(kept)} / {len(records)}")
    if rejected:
        print(f"  rejected: {dict(sorted(rejected.items(), key=lambda kv: -kv[1]))}")

    n_pool = int(cfg.data.n_pool)
    if len(kept) < n_pool:
        raise RuntimeError(
            f"only {len(kept)} usable VIVA items but data.n_pool={n_pool}. Either lower "
            f"n_pool (and study.n_values with it) or supply pre-downloaded images in "
            f"{images_dir} — the repo's Google Drive mirror is the reliable route when "
            f"the original image_urls have rotted."
        )

    # Deterministic pool: sort by index, then shuffle under data.seed.
    kept.sort(key=lambda r: int(r.get("index", 0)))
    rng = random.Random(int(cfg.data.seed))
    rng.shuffle(kept)
    pool = kept[:n_pool]

    out = []
    for rec in pool:
        fname = _image_name(rec)
        out.append({
            "index": int(rec["index"]),
            "image_file": fname,
            "image_path": os.path.join(images_dir, fname),
            "situation_description": _clean_str(rec.get("situation_description")),
            "action_list": [o.strip() for o in rec["action_list"]],
            "answer": _clean_str(rec.get("answer")).upper(),
            "category": _clean_str(rec.get("category")),
        })

    path = rel(cfg.data.calibration_file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {path} ({len(out)} items)")
    _report_pool(out)
    return path


def _report_pool(items: list[dict]) -> None:
    from collections import Counter

    ans = Counter(i["answer"] for i in items)
    n_opt = Counter(len(i["action_list"]) for i in items)
    cats = Counter(i["category"] for i in items)
    print(f"  gold-answer distribution: {dict(sorted(ans.items()))}")
    print(f"  options per item: {dict(sorted(n_opt.items()))}")
    print(f"  categories: {len(cats)} distinct, top {cats.most_common(3)}")
    lens = sorted(len(i["situation_description"]) for i in items)
    print(f"  situation_description chars: p10={lens[len(lens)//10]} "
          f"median={lens[len(lens)//2]} p90={lens[9*len(lens)//10]}")


# ------------------------------------------------------------------ self-captioning


CAPTION_PROMPT = (
    "Describe the situation shown in this image in two or three sentences. Describe only "
    "what is visibly happening, the people involved, and the setting. Do not suggest or "
    "evaluate any course of action."
)


def add_self_captions(cfg, calibration_path: str | None = None, *, device: str = "cuda") -> str:
    """Caption every pooled image with data.caption_model and store under `caption`.

    Greedy decoding, so the captions are a deterministic function of (model, image,
    prompt) and re-running does not perturb the measurement. The prompt forbids
    recommending an action, which is what keeps the text channel from leaking the label.
    """
    import torch
    from .load_model import load_processor

    path = calibration_path or rel(cfg.data.calibration_file)
    with open(path) as f:
        items = json.load(f)

    name = cfg.get_path("data.caption_model", cfg.model.name)
    print(f"captioning {len(items)} images with {name} (greedy)")

    from .load_model import _import_model_class

    cls, _ = _import_model_class()
    dtype = getattr(torch, str(cfg.model.dtype))
    try:
        model = cls.from_pretrained(name, dtype=dtype)
    except TypeError:
        model = cls.from_pretrained(name, torch_dtype=dtype)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    proc = load_processor(name, min_pixels=cfg.get_path("model.min_pixels"),
                          max_pixels=cfg.get_path("model.max_pixels"))

    from PIL import Image

    for k, it in enumerate(items, 1):
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": CAPTION_PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img = Image.open(it["image_path"]).convert("RGB")
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=120,
                                 temperature=None, top_p=None, top_k=None)
        gen = out[0, inputs["input_ids"].shape[1]:]
        it["caption"] = proc.tokenizer.decode(gen, skip_special_tokens=True).strip()
        if k % 25 == 0 or k == len(items):
            print(f"  captioned {k}/{len(items)}", flush=True)

    with open(path, "w") as f:
        json.dump(items, f, indent=1)
    print(f"updated {path} with captions")

    del model
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path


# ------------------------------------------------------------------ loading


def load_calibration(cfg, n: int | None = None) -> list[CalibItem]:
    """Read the calibration file into CalibItems, honouring data.text_channel."""
    path = rel(cfg.data.calibration_file)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `python -m src.run_task0 data` first."
        )
    with open(path) as f:
        raw = json.load(f)

    channel = cfg.get_path("data.text_channel", "situation_description")
    field = "caption" if channel == "self_caption" else "situation_description"

    items: list[CalibItem] = []
    for r in raw:
        text = (r.get(field) or "").strip()
        if not text:
            raise RuntimeError(
                f"item {r['index']} has no {field!r}. For text_channel=self_caption you "
                f"must run the captioning stage first."
            )
        if not os.path.exists(r["image_path"]):
            raise FileNotFoundError(f"missing image for item {r['index']}: {r['image_path']}")
        items.append(CalibItem(
            index=int(r["index"]), image_path=r["image_path"], situation=text,
            action_list=list(r["action_list"]), answer=str(r["answer"]).upper(),
            category=r.get("category", ""),
        ))
    return items if n is None else items[:n]


__all__ = ["fetch_annotation", "download_images", "build_calibration", "add_self_captions",
           "load_calibration", "CAPTION_PROMPT"]
