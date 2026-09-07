"""Visual-only full-parameter backend; text updates stay in original Praxis.

Not yet GPU validated. No optimizer is constructed here. The visual policy is
explicitly Hugging Face generation with FP32 master parameters and BF16 autocast;
both its sampling and its sequence scores use this same configuration.
"""
from pathlib import Path
from types import SimpleNamespace

import torch

from .backends import QwenBackend
from .core import trainables


class FullVisualBackend(QwenBackend):
    def __init__(self, cfg, parent_model, coordinate_manifest):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from task_0.src.config import Cfg
        from task_0.src.gradient import PolicyGradientExtractor
        from task_0.src.load_model import enable_gradient_checkpointing
        if not torch.cuda.is_available():raise RuntimeError("Production visual probe requires CUDA")
        if cfg["answer_parser"] not in ("explicit_final_v2","explicit_final_v3"):
            raise ValueError("Visual parser contract differs")
        self.answer_parser=cfg["answer_parser"]
        self.parent_state=torch.load(parent_model,map_location="cpu",mmap=True,weights_only=False)
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg["model"],torch_dtype=torch.float32,attn_implementation="flash_attention_2",
            local_files_only=True,trust_remote_code=False)
        expected={x["name"]:tuple(x["shape"]) for x in coordinate_manifest["segments"] if not x["padding"]}
        actual={n:tuple(p.shape) for n,p in self.model.named_parameters()}
        if actual!=expected:raise ValueError("Visual model and Praxis canonical coordinates differ")
        for alias,primary in coordinate_manifest["aliases"].items():
            if not torch.equal(self.parent_state[alias],self.parent_state[primary]):
                raise ValueError("Tied parent values differ")
        self.model.load_state_dict(self.parent_state,strict=True)
        self.model.to("cuda")
        enable_gradient_checkpointing(self.model)
        self.model.train()
        if any(isinstance(m,torch.nn.Dropout) and m.p for m in self.model.modules()):
            raise ValueError("Stochastic dropout violates frozen visual policy")
        self.buffers={n:b.detach().cpu().clone() for n,b in self.model.named_buffers()}
        proc=AutoProcessor.from_pretrained(cfg["model"],local_files_only=True,
                                          min_pixels=3136,max_pixels=cfg["max_pixels"])
        class Extractor(PolicyGradientExtractor):
            def _messages(self,item,modality):
                if modality!="image":raise ValueError("Production visual backend is image-only")
                body=item.question+"\n"+"\n".join(item.action_list)
                return [{"role":"system","content":[{"type":"text","text":self.system_prompt}]},
                        {"role":"user","content":[{"type":"image"},{"type":"text","text":body}]}]
        ex_cfg=Cfg({"policy":{"temperature":1.0,"top_p":1.0,"top_k":0,
                              "repetition_penalty":1.0,"max_new_tokens":cfg["max_new_tokens"],
                              "system_prompt":cfg["system_prompt"]},"sampling":{"gen_batch":1}})
        self.extractor=Extractor(self.model,proc,SimpleNamespace(params=list(trainables(self.model).values())),ex_cfg,device="cuda")
        self.metadata={"backend":"full_parameter_visual_hf_bf16_autocast","torch":torch.__version__,
                       "parent_model":str(parent_model),"canonical_numel":sum(p.numel() for p in self.model.parameters()),
                       "config":cfg,"text_optimizer":"none; updates come from original Praxis"}

    def sample(self,*args,**kwargs):
        with torch.autocast("cuda",dtype=torch.bfloat16):
            return super().sample(*args,**kwargs)

    def logprobs(self,response,*,reference=False):
        if reference:raise ValueError("Visual correctness gradient has no reference/KL objective")
        seq,plen,vis=response.payload
        with torch.autocast("cuda",dtype=torch.bfloat16):
            # A scalar sequence sum avoids materializing per-token log-softmax.
            return self.extractor.sequence_logprob(seq,plen,vis)[0]

    def restore_parent(self):
        self.model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n,p in self.model.named_parameters():p.copy_(self.parent_state[n])
            for n,b in self.model.named_buffers():b.copy_(self.buffers[n])
        if hasattr(self.model,"rope_deltas"):self.model.rope_deltas=None

    def apply_displacement(self, root, manifest):
        from .production_displacement import load_tensor
        params=dict(self.model.named_parameters())
        rows={r["name"]:r for r in manifest["parameters"]}
        if set(rows)!=set(params):raise ValueError("Child displacement coordinates differ")
        self.restore_parent()
        with torch.no_grad():
            for n,p in params.items():
                d=load_tensor(Path(root),rows[n])
                if d.shape!=p.shape:raise ValueError("Displacement shape differs")
                p.copy_(self.parent_state[n]+d)
