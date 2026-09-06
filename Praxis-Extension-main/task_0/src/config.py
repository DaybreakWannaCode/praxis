"""config.yaml loading, dotted access, validation, and provenance stamping."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any


class Cfg(dict):
    """dict with attribute access, recursively. `cfg.study.n_values` reads better than
    `cfg["study"]["n_values"]` in the hot paths, and typos raise instead of returning None."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(f"no config key {k!r} (have: {sorted(self)})") from e
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cur: Any = self
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value


def _coerce(text: str) -> Any:
    """Parse a CLI override value as JSON, falling back to the raw string."""
    try:
        return json.loads(text)
    except Exception:
        return text


def load_config(path: str, overrides: list[str] | None = None) -> Cfg:
    """Load config.yaml and apply `a.b.c=value` overrides (value parsed as JSON)."""
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = Cfg(raw)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like a.b.c=value, got {ov!r}")
        key, _, val = ov.partition("=")
        cfg.set_path(key.strip(), _coerce(val.strip()))
    validate(cfg)
    return cfg


def validate(cfg: Cfg) -> None:
    """Fail loudly on the config mistakes that would silently corrupt the measurement."""
    problems: list[str] = []

    n_values = cfg.get_path("study.n_values", [])
    G_values = cfg.get_path("study.G_values", [])
    n_pool = cfg.get_path("data.n_pool", 0)
    K = cfg.get_path("sampling.K", 0)

    if n_values and n_pool < max(n_values):
        problems.append(
            f"data.n_pool={n_pool} < max(study.n_values)={max(n_values)}: the largest sweep "
            f"cell would have to draw more items than the calibration pool holds."
        )
    if G_values and K < max(G_values):
        problems.append(
            f"sampling.K={K} < max(study.G_values)={max(G_values)}: cannot draw a group of "
            f"G from a pool of K generations."
        )
    if G_values and K < 2 * max(G_values):
        problems.append(
            f"sampling.K={K} is under 2*max(G)={2*max(G_values)}. Bootstrapping a group of "
            f"G from a pool of K underestimates answer-sampling variance by roughly a "
            f"factor (1 - 1/K); keep K comfortably above G."
        )

    anchor_n = cfg.get_path("study.anchor.n")
    anchor_G = cfg.get_path("study.anchor.G")
    if anchor_n is not None and n_values and anchor_n not in n_values:
        problems.append(f"study.anchor.n={anchor_n} is not in study.n_values={n_values}")
    if anchor_G is not None and G_values and anchor_G not in G_values:
        problems.append(f"study.anchor.G={anchor_G} is not in study.G_values={G_values}")

    # The single most dangerous silent bias: sampling from a truncated distribution while
    # differentiating the full softmax. Task 0 forces these, but catch a hand-edit.
    if cfg.get_path("policy.top_p", 1.0) != 1.0 or cfg.get_path("policy.top_k", 0) not in (0, None):
        problems.append(
            "policy.top_p must be 1.0 and policy.top_k must be 0. The score function "
            "grad log pi is computed against the untruncated softmax; sampling from a "
            "truncated distribution makes the gradient estimate biased, not just noisy."
        )
    if cfg.get_path("policy.repetition_penalty", 1.0) != 1.0:
        problems.append("policy.repetition_penalty must be 1.0 for the same reason as top_p/top_k.")

    conv = cfg.get_path("grpo.convention")
    if conv not in ("verl", "spec"):
        problems.append(f"grpo.convention must be 'verl' or 'spec', got {conv!r}")
    if cfg.get_path("grpo.loss_agg") not in ("batch_token_mean", "seq_mean", "seq_sum"):
        problems.append(f"grpo.loss_agg invalid: {cfg.get_path('grpo.loss_agg')!r}")
    if cfg.get_path("study.estimator") not in ("crossfit", "paired"):
        problems.append(f"study.estimator invalid: {cfg.get_path('study.estimator')!r}")
    if cfg.get_path("data.text_channel") not in ("situation_description", "self_caption"):
        problems.append(f"data.text_channel invalid: {cfg.get_path('data.text_channel')!r}")

    if problems:
        raise ValueError("config validation failed:\n  - " + "\n  - ".join(problems))


def resolve_grpo(cfg: Cfg) -> dict:
    """Turn `grpo.convention` into the concrete knobs the estimator uses.

    `convention` is a shorthand that pins loss_agg and normalize_by_std together; any key
    explicitly present in the config still wins, so you can hand-mix if you want to.
    """
    g = dict(cfg.get_path("grpo", {}))
    eps = float(g.get("adv_eps", 1e-6))
    floor = float(g.get("adv_std_floor", 0.0))
    if g.get("convention") == "spec":
        # The brief's literal wording: sequence-SUM log-prob, advantage centred only.
        return {"loss_agg": "seq_sum", "normalize_by_std": False,
                "eps": eps, "std_floor": floor}
    return {"loss_agg": g.get("loss_agg", "batch_token_mean"),
            "normalize_by_std": bool(g.get("normalize_adv_by_std", True)),
            "eps": eps, "std_floor": floor}


def both_conventions(cfg: Cfg) -> dict[str, dict]:
    """The two convention presets, so every result can be reported under both."""
    eps = float(cfg.get_path("grpo.adv_eps", 1e-6))
    floor = float(cfg.get_path("grpo.adv_std_floor", 0.0))
    return {
        "verl": {"loss_agg": "batch_token_mean", "normalize_by_std": True,
                 "eps": eps, "std_floor": floor},
        "spec": {"loss_agg": "seq_sum", "normalize_by_std": False,
                 "eps": eps, "std_floor": floor},
    }


def stamp(cfg: Cfg) -> dict:
    """Provenance block written next to every result file."""
    payload = json.dumps(cfg, sort_keys=True, default=str)
    out = {
        "config_sha256": hashlib.sha256(payload.encode()).hexdigest()[:16],
        "config": copy.deepcopy(dict(cfg)),
    }
    try:
        import torch
        import transformers

        out["torch"] = torch.__version__
        out["transformers"] = transformers.__version__
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["gpu_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except Exception:
        pass
    return out


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rel(path: str) -> str:
    """Resolve a config path relative to the task_0 folder unless it is absolute."""
    return path if os.path.isabs(path) else os.path.join(project_root(), path)


__all__ = ["Cfg", "load_config", "validate", "resolve_grpo", "both_conventions",
           "stamp", "project_root", "rel"]