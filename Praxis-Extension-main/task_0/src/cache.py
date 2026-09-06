"""The per-(item, sample) sketched gradient cache.

What is stored, and why it is enough
------------------------------------
For every calibration item i, modality m, and sampled answer j we store three things:

    v_ij = S( grad_theta sum_t log pi_theta(a_ijt | s_i, a_ij<t) )   the sketched gradient
    L_ij = number of response tokens
    r_ij = binary reward

The advantage A_ij is a *scalar* determined entirely by the rewards of whichever samples
are grouped together, and every GRPO convention is a scalar reweighting of v_ij (see
gradient.aggregate_coeffs). So

    g_hat = scale * sum_ij coeff(A_ij, L_ij) * v_ij

for any n, any G, any resample, and either convention — all of it reconstructible from the
cache with no GPU work. That is what collapses the 12-cell x 30-resample sweep from ~1.26M
generate+backward passes to a single pass of 2 * n_pool * K of them.

Storage
-------
float16 with a per-vector float32 scale (v is stored as v / max|v|, which lands in [-1, 1]
where float16 carries ~11 bits of mantissa). At the defaults this is
2 * 400 * 32 * 262144 * 2 bytes = 13.4 GB, memory-mapped so the build never holds it all.

Resumability
------------
The build is on the order of a GPU-hour and Colab disconnects. Progress is recorded per
(modality, item) in `done.json` and the memmaps persist, so re-running skips finished work.
Items redone after a crash get fresh generations, which is harmless: they are i.i.d. draws
either way.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Sequence

MODALITIES = ("image", "text")


def cache_dir(cfg) -> str:
    """Where the sketched gradient arrays live.

    Separable from `results/` on purpose. The cache is a numpy memmap, and memmap over a
    network filesystem (notably Google Drive's FUSE mount in Colab) is slow and not
    reliably supported, so it wants fast local disk. The small outputs in `results/` want
    the opposite: durable storage that survives a runtime dying. Set `data.cache_dir` to
    an absolute local path when `results/` is pointed at network storage.
    """
    from .config import rel

    return rel(cfg.get_path("data.cache_dir") or os.path.join("results", "cache"))


# ------------------------------------------------------------------ writing


def fingerprint(cfg, items, sketch) -> dict:
    """Everything that has to match for a partially-built cache to be resumable.

    Without this, changing `sketch.m_target`, `data.n_pool`, `data.seed` or the model and
    re-running would quietly reuse the existing arrays: numpy's open_memmap ignores the
    requested dtype/shape in 'r+' mode and takes them from the file header, and done.json
    would still claim the old items were finished. The result would be a cache whose rows
    do not correspond to the items the study thinks it is using.
    """
    import hashlib

    fp = {
        "model": str(cfg.model.name),
        "n_items": len(items),
        "K": int(cfg.sampling.K),
        "m": int(sketch.m),
        "block_size": int(sketch.plan.block_size),
        "sketch_seed": int(cfg.get_path("sketch.seed", 0)),
        "data_seed": int(cfg.get_path("data.seed", 0)),
        "text_channel": str(cfg.get_path("data.text_channel")),
        "max_new_tokens": int(cfg.get_path("policy.max_new_tokens")),
        "temperature": float(cfg.get_path("policy.temperature")),
        "item_indices": [int(it.index) for it in items],
    }
    fp["hash"] = hashlib.sha256(
        json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
    return fp


class CacheWriter:
    def __init__(self, root: str, *, n_items: int, K: int, m: int, store_dtype: str = "float16"):
        import numpy as np

        self.np = np
        self.root = root
        self.n_items = int(n_items)
        self.K = int(K)
        self.m = int(m)
        self.dtype = np.dtype(store_dtype)
        os.makedirs(root, exist_ok=True)

        self.vecs = {}
        self.scales = {}
        for mod in MODALITIES:
            self.vecs[mod] = self._open(f"{mod}_vecs.npy", self.dtype,
                                        (self.n_items, self.K, self.m))
            self.scales[mod] = self._open(f"{mod}_scales.npy", np.float32,
                                          (self.n_items, self.K))

        self.meta_path = os.path.join(root, "samples.json")
        self.done_path = os.path.join(root, "done.json")
        self.meta = self._load(self.meta_path, {mod: {} for mod in MODALITIES})
        self.done = set(tuple(x) for x in self._load(self.done_path, []))

    def _open(self, name: str, dtype, shape):
        """Memory-map an array, verifying an existing file matches what we asked for."""
        import numpy as np

        path = os.path.join(self.root, name)
        if os.path.exists(path):
            arr = np.lib.format.open_memmap(path, mode="r+")
            if tuple(arr.shape) != tuple(shape) or arr.dtype != np.dtype(dtype):
                raise RuntimeError(
                    f"{path} exists with shape {arr.shape}/{arr.dtype} but this run needs "
                    f"{tuple(shape)}/{np.dtype(dtype)}. numpy takes shape and dtype from the "
                    f"file header in 'r+' mode, so reusing it would silently corrupt the "
                    f"cache. Delete {self.root} and rebuild."
                )
            return arr
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

    @staticmethod
    def _load(path, default):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return default

    def is_done(self, modality: str, i: int) -> bool:
        return (modality, i) in self.done

    def write_item(self, modality: str, i: int, vectors, lengths, rewards, parsed, texts) -> None:
        """vectors: (K, m) float32 torch tensor or numpy array."""
        np = self.np
        arr = vectors.detach().cpu().numpy() if hasattr(vectors, "detach") else np.asarray(vectors)
        arr = arr.astype(np.float32, copy=False)
        if arr.shape != (self.K, self.m):
            # A short group would leave zero rows that later draws could still select,
            # silently contributing a zero gradient with reward 0. Refuse instead.
            raise RuntimeError(
                f"expected {self.K} sampled gradients of dim {self.m} for {modality} item "
                f"{i}, got {arr.shape}"
            )
        scales = np.abs(arr).max(axis=1)
        safe = np.where(scales > 0, scales, 1.0).astype(np.float32)
        self.vecs[modality][i] = (arr / safe[:, None]).astype(self.dtype)
        self.scales[modality][i] = scales.astype(np.float32)
        self.meta[modality][str(i)] = {
            "lengths": [int(x) for x in lengths],
            "rewards": [float(x) for x in rewards],
            "parsed_ok": [bool(x) for x in parsed],
            "texts": list(texts),
        }
        self.done.add((modality, i))

    def flush(self) -> None:
        for mod in MODALITIES:
            self.vecs[mod].flush()
            self.scales[mod].flush()
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f)
        with open(self.done_path, "w") as f:
            json.dump(sorted(self.done), f)


def build_cache(cfg, extractor, sketch, items, *, root: str | None = None,
                flush_every: int = 5, verbose: bool = True) -> str:
    """Generate K answers per (item, modality) and cache their sketched gradients."""
    import torch

    from .config import stamp

    root = root or cache_dir(cfg)
    os.makedirs(root, exist_ok=True)
    K = int(cfg.sampling.K)
    fp = fingerprint(cfg, items, sketch)

    meta_path = os.path.join(root, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            old = json.load(f)
        old_fp = old.get("fingerprint", {})
        if old_fp.get("hash") != fp["hash"]:
            differing = sorted(k for k in fp
                               if k != "hash" and old_fp.get(k) != fp[k])
            raise RuntimeError(
                f"cache at {root} was built with different settings "
                f"(differing: {differing or 'unknown'}). Resuming would mix runs. "
                f"Delete {root} to rebuild, or pass a different cache root."
            )

    writer = CacheWriter(root, n_items=len(items), K=K, m=sketch.m,
                         store_dtype=str(cfg.get_path("sketch.store_dtype", "float16")))

    with open(meta_path, "w") as f:
        json.dump({
            "n_items": len(items), "K": K, "m": sketch.m, "d": sketch.d,
            "block_size": sketch.plan.block_size,
            "sketch_seed": int(cfg.get_path("sketch.seed", 0)),
            "item_indices": [it.index for it in items],
            "text_channel": cfg.get_path("data.text_channel"),
            "model": cfg.model.name,
            "fingerprint": fp,
            **stamp(cfg),
        }, f, indent=1)

    buf = torch.zeros(sketch.m, dtype=torch.float32, device=sketch.device)
    vecs = torch.zeros(K, sketch.m, dtype=torch.float32, device=sketch.device)
    total = len(items) * len(MODALITIES)
    done_count = sum(1 for mod in MODALITIES for i in range(len(items)) if writer.is_done(mod, i))
    t0 = time.time()
    step = 0

    for i, item in enumerate(items):
        for mod in MODALITIES:
            step += 1
            if writer.is_done(mod, i):
                continue
            seed = (int(cfg.get_path("data.seed", 0)) * 7919 + i) * 31 + (0 if mod == "image" else 1)
            grp = extractor.sample_group(item, mod, K, seed=seed)
            vecs.zero_()
            for j, seq in enumerate(grp.seqs):
                extractor.per_sample_gradient(seq, grp.prompt_len, grp.vis, sketch, out=buf)
                vecs[j].copy_(buf)
            writer.write_item(mod, i, vecs[:len(grp.seqs)], grp.lengths, grp.rewards,
                              grp.parsed_ok, grp.texts)
            done_count += 1
            if verbose and (done_count % 10 == 0 or step == total):
                el = time.time() - t0
                rate = max(done_count - 0, 1) / max(el, 1e-9)
                print(f"  cache {done_count}/{total}  "
                      f"mean_reward={sum(grp.rewards)/max(len(grp.rewards),1):.2f}  "
                      f"{el/60:.1f} min elapsed, ~{(total-done_count)/max(rate,1e-9)/60:.0f} min left",
                      flush=True)
        item.release()
        if (i + 1) % flush_every == 0:
            writer.flush()

    writer.flush()
    if verbose:
        print(f"cache complete: {root}")
    return root


# ------------------------------------------------------------------ reading


@dataclass
class ModalityCache:
    vectors: object      # (N, K, m) float16 tensor on `device`
    scales: object       # (N, K) float32
    rewards: object      # (N, K) float32
    lengths: object      # (N, K) int64
    parsed_ok: object    # (N, K) bool


class GradientCache:
    """The cache loaded for the study. Everything downstream is arithmetic on this."""

    def __init__(self, root: str, *, device: str = "cpu"):
        import numpy as np
        import torch

        self.root = root
        self.torch = torch
        with open(os.path.join(root, "meta.json")) as f:
            self.meta = json.load(f)
        with open(os.path.join(root, "samples.json")) as f:
            sample_meta = json.load(f)
        with open(os.path.join(root, "done.json")) as f:
            done = set(tuple(x) for x in json.load(f))

        self.N = int(self.meta["n_items"])
        self.K = int(self.meta["K"])
        self.m = int(self.meta["m"])
        self.item_indices = list(self.meta.get("item_indices", range(self.N)))

        missing = [(mod, i) for mod in MODALITIES for i in range(self.N) if (mod, i) not in done]
        if missing:
            raise RuntimeError(
                f"cache at {root} is incomplete: {len(missing)} (modality, item) pairs never "
                f"written, e.g. {missing[:4]}. Re-run the cache stage; it resumes."
            )

        if device == "auto":
            device = self._auto_device()
        self.device = torch.device(device)

        self.mod: dict[str, ModalityCache] = {}
        for m_ in MODALITIES:
            v = np.load(os.path.join(root, f"{m_}_vecs.npy"), mmap_mode="r")
            s = np.load(os.path.join(root, f"{m_}_scales.npy"), mmap_mode="r")
            rew = np.zeros((self.N, self.K), dtype=np.float32)
            ln = np.ones((self.N, self.K), dtype=np.int64)
            ok = np.zeros((self.N, self.K), dtype=bool)
            for i in range(self.N):
                rec = sample_meta[m_][str(i)]
                k = len(rec["rewards"])
                rew[i, :k] = rec["rewards"]
                ln[i, :k] = rec["lengths"]
                ok[i, :k] = rec["parsed_ok"]
            self.mod[m_] = ModalityCache(
                vectors=torch.from_numpy(np.ascontiguousarray(v)).to(self.device),
                scales=torch.from_numpy(np.ascontiguousarray(s)).to(self.device),
                rewards=torch.from_numpy(rew).to(self.device),
                lengths=torch.from_numpy(ln).to(self.device),
                parsed_ok=torch.from_numpy(ok).to(self.device),
            )

    def _auto_device(self) -> str:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        need = 2 * self.N * self.K * self.m * 2 * 1.2   # both modalities, fp16, 20% headroom
        free_b, _ = torch.cuda.mem_get_info()
        return "cuda" if free_b > need else "cpu"

    def nbytes(self) -> int:
        return sum(int(c.vectors.numel()) * 2 for c in self.mod.values())

    def describe(self) -> str:
        lines = [f"cache {self.root}",
                 f"  N={self.N} items, K={self.K} samples, m={self.m:,} "
                 f"({self.nbytes()/1e9:.1f} GB fp16 on {self.device})"]
        for m_ in MODALITIES:
            c = self.mod[m_]
            r = c.rewards
            deg = float(((r == r[:, :1]).all(dim=1)).float().mean())
            lines.append(f"  {m_:>5}: mean reward {float(r.mean()):.3f}, "
                         f"parse-ok {float(c.parsed_ok.float().mean()):.3f}, "
                         f"all-K-identical items {deg:.2%}, "
                         f"median response {int(c.lengths.float().median())} tok")
        return "\n".join(lines)

    def gather(self, modality: str, item_idx, sample_idx):
        """Dequantised (n, G, m) float32 block for the given item/sample index tensors."""
        c = self.mod[modality]
        v = c.vectors[item_idx.unsqueeze(-1), sample_idx]           # (n, G, m) fp16
        s = c.scales[item_idx.unsqueeze(-1), sample_idx]            # (n, G)
        return v.float() * s.unsqueeze(-1)

    def meta_for(self, modality: str, item_idx, sample_idx):
        c = self.mod[modality]
        r = c.rewards[item_idx.unsqueeze(-1), sample_idx]
        l = c.lengths[item_idx.unsqueeze(-1), sample_idx]
        return r, l


__all__ = ["MODALITIES", "cache_dir", "CacheWriter", "build_cache", "GradientCache",
           "ModalityCache"]
