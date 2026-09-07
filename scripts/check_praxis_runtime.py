"""Import/ABI preflight only; passing this is not a training reproduction."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    report = {"status": "started", "checks": {}, "versions": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for name in ("torch", "vllm", "transformers", "ray", "tensordict", "torchdata", "flash-attn"):
            report["versions"][name] = importlib.metadata.version(name)
        for name in ("verl.trainer.main", "verl.workers.fsdp_workers", "torchdata.stateful_dataloader"):
            importlib.import_module(name)
            report["checks"][name] = True
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLFlashAttention2
        from verl.models.monkey_patch import apply_ulysses_patch
        apply_ulysses_patch("qwen2_5_vl")
        report["checks"]["qwen_attention_patch"] = True
        if args.config:
            from omegaconf import OmegaConf
            from verl.trainer.config import PPOConfig
            from verl.utils.dataset import RLHFDataset
            from verl.utils.tokenizer import get_processor, get_tokenizer
            config = OmegaConf.to_object(OmegaConf.merge(
                OmegaConf.structured(PPOConfig()), OmegaConf.load(args.config)))
            config.deep_post_init()
            model = config.worker.actor.model.model_path
            tokenizer = get_tokenizer(model, use_fast=True)
            processor = get_processor(model, use_fast=True)
            report["datasets"] = {}
            for name in ("train", "val"):
                dataset = RLHFDataset(
                    getattr(config.data, name + "_files"), tokenizer, processor,
                    prompt_key=config.data.prompt_key, answer_key=config.data.answer_key,
                    system_prompt=config.data.system_prompt,
                    max_prompt_length=config.data.max_prompt_length,
                    truncation="error")
                lengths = [int(dataset[i]["attention_mask"].sum()) for i in range(len(dataset))]
                report["datasets"][name] = {"count": len(dataset), "prompt_lengths": lengths}
            report["checks"]["bounded_config_and_data"] = True
        report["source_hashes"] = {
            str(path.relative_to(args.source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((args.source / "verl").rglob("*.py"))
            if not path.name.startswith("._")
        }
        if args.gpu:
            import torch
            from flash_attn import flash_attn_func
            q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            out = flash_attn_func(q, q, q, causal=True)
            out.float().square().mean().backward()
            torch.cuda.synchronize()
            assert torch.isfinite(out).all() and torch.isfinite(q.grad).all()
            report["checks"]["flash_attention_forward_backward"] = True
            report["gpu"] = torch.cuda.get_device_name()
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
