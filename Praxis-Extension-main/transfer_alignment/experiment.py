"""Bounded optimizer branches with scores sealed before independent outcomes."""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import torch

from .core import (alignment, assert_frozen_unchanged, capture, clipped_loss,
                   coordinate_manifest, derived_seed, digest, displacement,
                   grpo_advantages, isolated_rng, loo_advantages, restore, trainables, weights)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append_json(path, value):
    with open(path, "a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")
        f.flush()


def load_trusted_checkpoint(path):
    # Local run artifacts only: never use this on an untrusted downloaded pickle.
    return torch.load(path, map_location="cpu", weights_only=False)


def reward_function(profile):
    if profile == "correctness_only":
        return lambda response, item: response.reward
    if profile != "released_code":
        raise ValueError(f"Unknown text reward profile {profile}")
    path = Path(__file__).resolve().parents[2] / "Praxis-VLM-main/verl/utils/reward_score/mcq.py"
    spec = importlib.util.spec_from_file_location("praxis_mcq_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return lambda response, item: float(module.mcq_compute_score(response.text, item.answer)["overall"])


def response_record(response, item, seed, kind):
    return {"item_id": item.id, "group_id": item.group_id, "split": item.split,
            "seed": seed, "kind": kind, "answer": response.answer,
            "correct": response.reward, "parsed": response.parsed,
            "length": response.length, "truncated": response.truncated, "text": response.text}


def visual_gradient(backend, items, group_size, seed):
    if not items or any(i.split != "score" for i in items):
        raise ValueError("Visual gradient requires nonempty V_score only")
    model = backend.model
    g = {n: torch.zeros_like(p, dtype=torch.float32, device="cpu") for n, p in trainables(model).items()}
    records = []
    degenerate = 0
    for item in items:
        item_seed = derived_seed(seed, "visual_gradient", item.id)
        samples = backend.sample(item, "image", group_size, item_seed)
        adv = loo_advantages([s.reward for s in samples])
        degenerate += int(not bool(adv.any()))
        for j, (sample, a) in enumerate(zip(samples, adv)):
            records.append(dict(response_record(sample, item, item_seed, "probe"), sample=j))
            if a.item() == 0:
                continue
            model.zero_grad(set_to_none=True)
            # Sequence sum; NO token averaging, std normalization, KL, or format bonus.
            loss = backend.logprobs(sample).sum() * (a.item() / (len(items) * group_size))
            loss.backward()
            for n, p in trainables(model).items():
                if p.grad is not None:
                    g[n].add_(p.grad.detach().float().cpu())
    model.zero_grad(set_to_none=True)
    if not all(torch.isfinite(x).all() for x in g.values()):
        raise FloatingPointError("Nonfinite visual gradient")
    return g, records, {"items": len(items), "group_size": group_size,
                        "degenerate_groups": degenerate, "estimator": "rloo_sequence_sum"}


def text_step(backend, optimizer, scheduler, items, cfg, seed):
    """One GRPO-style optimizer step with sequence groups and token-mean loss.

    This bounded backend averages across the whole candidate token batch. Praxis
    averages microbatch losses separately: do not claim trainer parity.
    """
    if not items or any(i.split != "train" for i in items):
        raise ValueError("Optimizer updates require text-training split only")
    score = reward_function(cfg["reward_profile"])
    model = backend.model
    before = weights(model)
    samples, log_records = [], []
    started = time.monotonic()
    for item in items:
        item_seed = derived_seed(seed, "text", item.id)
        group = backend.sample(item, "text", cfg["text_group_size"], item_seed)
        rewards = [score(s, item) for s in group]
        advantages = grpo_advantages(rewards)
        for j, (sample, adv, reward) in enumerate(zip(group, advantages, rewards)):
            with torch.no_grad():
                old = backend.logprobs(sample).detach().cpu()
                ref = backend.logprobs(sample, reference=True).detach().cpu()
            samples.append((sample, adv.item(), old, ref))
            log_records.append(dict(response_record(sample, item, item_seed, "text_train"),
                                    sample=j, training_reward=reward, advantage=adv.item()))
    tokens = sum(s.length for s, _, _, _ in samples)
    optimizer.zero_grad(set_to_none=True)
    loss_value = 0.0
    for sample, advantage, old, ref in samples:
        current = backend.logprobs(sample)
        if current.numel() != sample.length:
            raise AssertionError("Logprob/response token count mismatch")
        term = clipped_loss(current, old.to(current.device), ref.to(current.device), advantage,
                            clip=cfg["clip_ratio"], kl_coef=cfg["kl_coef"])
        loss = term.sum() / tokens
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite GRPO loss")
        loss.backward()
        loss_value += loss.item()
    norm = torch.nn.utils.clip_grad_norm_(list(trainables(model).values()), cfg["max_grad_norm"],
                                         error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    delta = displacement(model, before)
    if not all(torch.isfinite(d).all() for d in delta.values()):
        raise FloatingPointError("Nonfinite actual update")
    return delta, {"loss": loss_value, "preclip_gradient_norm": float(norm),
                   "tokens": tokens, "responses": len(samples), "seconds": time.monotonic()-started,
                   "text_correctness": sum(r["correct"] for r in log_records)/len(log_records),
                   "training_reward": sum(r["training_reward"] for r in log_records)/len(log_records),
                   "reward_profile": cfg["reward_profile"]}, log_records


def evaluate(backend, items, count, seed, label):
    if not items or any(i.split != "dev" for i in items):
        raise ValueError("Engineering runner evaluates V_dev only; V_test remains sealed")
    records, item_means = [], {}
    for item in items:
        item_seed = derived_seed(seed, "outcome", item.id)
        draws = backend.sample(item, "image", count, item_seed)
        item_means[item.id] = sum(s.reward for s in draws)/len(draws)
        for j, sample in enumerate(draws):
            records.append(dict(response_record(sample, item, item_seed, "sampled"),
                                sample=j, checkpoint=label))
        greedy = backend.sample(item, "image", 1, item_seed, greedy=True)[0]
        records.append(dict(response_record(greedy, item, item_seed, "greedy"), checkpoint=label))
    sampled = [r for r in records if r["kind"] == "sampled"]
    greedy = [r for r in records if r["kind"] == "greedy"]
    return {"sampled_correctness": sum(item_means.values())/len(items),
            "greedy_accuracy": sum(r["correct"] for r in greedy)/len(greedy),
            "parse_rate": sum(r["parsed"] for r in sampled)/len(sampled),
            "truncation_rate": sum(r["truncated"] for r in sampled)/len(sampled),
            "item_means": item_means}, records


def run(backend, items, candidates, cfg, output):
    """Score all candidates first, replay them exactly, then evaluate saved children."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = backend.model
    optimizer = torch.optim.AdamW(list(trainables(model).values()), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    score_items = [x for x in items if x.split == "score"]
    dev_items = [x for x in items if x.split == "dev"]
    seed = cfg["seed"]
    started = time.monotonic()
    source_files = list(Path(__file__).parent.glob("*.py")) + list(
        (Path(__file__).parents[1]/"task_0/src").glob("*.py"))
    metadata = {"status": "running", "evidence": backend.metadata["backend"],
                "config": cfg, "backend": backend.metadata, "torch": torch.__version__,
                "coordinates": coordinate_manifest(model),
                "items": [vars(x) for x in items], "dataset_digest": digest([vars(x) for x in items]),
                "source_digests": {str(p.relative_to(Path(__file__).parents[1])):
                                   digest(p.read_bytes()) for p in source_files},
                "candidates": [[x.id for x in b] for b in candidates]}
    write_json(output/"manifest.json", metadata)
    # Warm optimizer moments so restore tests do not only cover an empty Adam state.
    for step in range(cfg["warmup_steps"]):
        _, metrics, rows = text_step(backend, optimizer, scheduler, candidates[step % len(candidates)], cfg,
                                    derived_seed(seed, "warmup", step))
        append_json(output/"warmup.jsonl", metrics)
    parent = capture(model, optimizer, scheduler)
    parent_id = digest(parent)
    torch.save(parent, output/"parent.pt")
    try:
        gv, probe_rows, probe_info = visual_gradient(backend, score_items, cfg["probe_group_size"], seed)
    except BaseException:
        restore(parent, model, optimizer, scheduler)
        raise
    torch.save(gv, output/"visual_gradient.pt")
    write_json(output/"probe.json", probe_info)
    for r in probe_rows:
        append_json(output/"probe_responses.jsonl", r)

    scores = []
    try:
        for b, batch in enumerate(candidates):
            restore(parent, model, optimizer, scheduler)
            if digest(capture(model, optimizer, scheduler)) != parent_id:
                raise AssertionError("Parent restoration failed")
            before = weights(model)
            branch_records = []
            for step in range(cfg["branch_steps"]):
                delta, metrics, rows = text_step(backend, optimizer, scheduler, batch, cfg,
                                               derived_seed(seed, "branch", b, step))
                torch.save(delta, output/f"branch_{b}_step_{step}_delta.pt")
                # For step >0 this is explicitly a stale-parent-gradient diagnostic.
                append_json(output/"updates.jsonl", dict(branch=b, step=step, **metrics,
                            **alignment(gv, delta), gradient_checkpoint="parent"))
                branch_records.extend(dict(r, branch=b, step=step) for r in rows)
            net_delta = displacement(model, before)
            assert_frozen_unchanged(model, parent)
            child = capture(model, optimizer, scheduler, compact=True)
            torch.save({"parent_id": parent_id, "state": child}, output/f"branch_{b}.pt")
            score = {"branch": b, "parent_id": parent_id, "steps": cfg["branch_steps"],
                     "score_kind": "parent_gradient_net_window_displacement", **alignment(gv, net_delta)}
            # Independent replay from complete parent, including warm moments and RNG.
            restore(parent, model, optimizer, scheduler)
            for step in range(cfg["branch_steps"]):
                text_step(backend, optimizer, scheduler, batch, cfg, derived_seed(seed, "branch", b, step))
            replay = capture(model, optimizer, scheduler, compact=True)
            if digest(replay) != digest(child):
                raise AssertionError(f"Branch {b} not exactly reproducible; inspect deterministic kernels")
            assert_frozen_unchanged(model, parent)
            score["replay_exact"] = True
            scores.append(score)
            append_json(output/"scores.jsonl", score)
            for r in branch_records:
                append_json(output/"text_responses.jsonl", r)
            print(f"Scored branch {b}: A={score['alignment']:.6g}; replay exact", flush=True)

        # No child visual outcome is available until the score list has been sealed.
        score_digest = digest(scores)
        write_json(output/"scores_sealed.json", {"sha256": score_digest, "branches": len(scores)})
        restore(parent, model, optimizer, scheduler)
        base, rows = evaluate(backend, dev_items, cfg["eval_samples"],
                              derived_seed(seed, "eval_parent"), "parent")
        for r in rows:
            append_json(output/"evaluations.jsonl", r)
        outcomes = []
        for score in scores:
            b = score["branch"]
            restore(parent, model, optimizer, scheduler)
            checkpoint = load_trusted_checkpoint(output/f"branch_{b}.pt")
            if checkpoint["parent_id"] != parent_id:
                raise ValueError("Child belongs to a different parent")
            restore(checkpoint["state"], model, optimizer, scheduler)
            result, rows = evaluate(backend, dev_items, cfg["eval_samples"],
                                    derived_seed(seed, "eval_child", b), f"branch_{b}")
            for r in rows:
                append_json(output/"evaluations.jsonl", r)
            diffs = [result["item_means"][i.id] - base["item_means"][i.id] for i in dev_items]
            change = sum(diffs)/len(diffs)
            groups = {}
            for item, diff in zip(dev_items, diffs):
                groups.setdefault(item.group_id, []).append(diff)
            # Cluster-robust SE of the item-weighted mean; scenes/videos can contain
            # multiple questions. Three trajectories still require run-level analysis.
            m, n = len(groups), len(diffs)
            se = (m/(m-1) * sum(sum(d-change for d in ds)**2 for ds in groups.values()) / n**2)**0.5 if m>1 else None
            outcomes.append(dict(score, **result, visual_change=change,
                                 greedy_change=result["greedy_accuracy"]-base["greedy_accuracy"],
                                 paired_scene_cluster_se=se))
        write_json(output/"summary.json", {"evidence": backend.metadata["backend"],
                   "parent": base, "branches": outcomes, "scores_digest": score_digest,
                   "seconds": time.monotonic()-started,
                   "note": "Engineering V_dev outcomes; shared parent errors are correlated. No significance claim."})
        metadata["status"] = "complete"
        metadata["parent_id"] = parent_id
        write_json(output/"manifest.json", metadata)
        return outcomes
    finally:
        restore(parent, model, optimizer, scheduler)
