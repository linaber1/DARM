from dataclasses import dataclass
from typing import List, Dict, Optional
import yaml
import random
import torch

from .config import AppCfg
from .models.llm import LLM, LLM_GPT
from .models.diffusion import Diffusion
from .models.ours import OurDualModel
from .concatenate import stitch
from .utils import timestamp
from pts.constants import Pipelines

def to_dual_args(cfg: AppCfg):
    return {
        "llm_id": cfg.llm.model_id,
        "diff_id": cfg.diffusion.model_id,
        "max_new_tokens": cfg.llm.max_new_tokens,
        "max_new_latents": cfg.diffusion.max_new_tokens,
        "temperature": cfg.llm.temperature,
        "do_sample": cfg.llm.do_sample
    }
    

@dataclass
class PTSPipeline:
    cfg: AppCfg
    llm: Optional[LLM]
    diffusion: Optional[Diffusion]
    dual: Optional[OurDualModel]

    @classmethod
    def from_yaml(cls, path: str, use_gpt=False):
        
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        cfg = AppCfg(**raw)
        # seeds
        random.seed(cfg.runtime.seed)
        torch.manual_seed(cfg.runtime.seed)
        llm, diffusion, dual = None, None, None
        
        if cfg.pipeline_type == "dual":
            
            dual = OurDualModel(**to_dual_args(cfg))
        else:
            diffusion = Diffusion(**cfg.diffusion.model_dump())
            if use_gpt:
                llm_gpt = LLM_GPT(**cfg.llm_gpt.model_dump())
                llm = llm_gpt
            else:
                llm = LLM(**cfg.llm.model_dump())
        return cls(cfg=cfg, diffusion=diffusion, llm=llm, dual=dual)

    def generate_plan(self, user_prompt: str, name_architecture: str) -> Dict:
        if name_architecture in Pipelines.llm_plan():
            return self.llm.generate(user_prompt=user_prompt)
        elif name_architecture in Pipelines.diffusion_plan():
            return self.diffusion.generate(prompt=user_prompt)
        raise ValueError(f"Unknown architecture name: {name_architecture}")

    def generate_answer(self, user_prompt: str, name_architecture: str):
        if name_architecture in Pipelines.diffusion_answer():
            return self.diffusion.generate(prompt=user_prompt)
        if name_architecture in Pipelines.llm_answer():
            return self.llm.generate(user_prompt=user_prompt)
        raise ValueError(f"Unknown architecture name: {name_architecture}")
    

    # Dual model generation with latents !
    def generate_dual(self, user_prompt: str, plan_prompt, with_latents: bool=True, stop_early: bool=False, percentage: int=100, use_latents: bool=True) -> Dict:
        if self.dual is None:
            raise ValueError("Dual model is not initialized.")
        latents, plan, diff_metadata = None, "", {}
        if with_latents:
            if stop_early:
                stop_at_ratio = percentage / 100.0
                diff_out = self.dual.generate_plan_early(prompt=plan_prompt, stop_at_ratio=stop_at_ratio )
            else:
                diff_out = self.dual.generate_plan(prompt=plan_prompt)
            latents = diff_out["latents"]
            plan = diff_out["text"]
            diff_metadata = diff_out["metadata"]
            
        answer = self.dual.generate_answer(user_prompt, latents=latents, use_latents=use_latents)
        return {
            "plan": plan,
            "plan_latents" : latents,
            "answer": answer['text'],
            "metadata": {
                **diff_metadata,
                **answer['metadata']
            }
        }
    

    def run(
        self,
        user_prompt: str,
        extra_text: Optional[List[str]] = None,
        refine_with_llm: bool = False,
    ) -> Dict:
        # 1) diffusion produces plan/structure
        plan = self.diffusion.generate(user_prompt)
        plan_text = plan["text"]

        # 2) (Optional) First-pass llm to refine the plan
        refined_text = ""
        if refine_with_llm:
            refined = self.llm.generate(
                system_prompt="You rewrite plans into crisp bullet points.",
                user_prompt=f"Rewrite the following plan more clearly:\n\n{plan_text}",
            )
            refined_text = refined["text"]

        # 3) Concatenate inputs for final llm call
        combined = stitch(
            plan=plan_text,
            refined=refined_text,
            extras=extra_text or [],
            headers=self.cfg.prompting,
        )

        final = self.llm.generate(
            system_prompt=self.cfg.prompting.final_system_msg, user_prompt=f"{combined}"
        )

        # 4) JSON result
        result = {
            "timestamp": timestamp(),
            "input": {"prompt": user_prompt, "extra_text": extra_text or []},
            "diffusion_output": {"text": plan_text, "metadata": plan["metadata"]},
            "llm_stage1_output": {
                "text": refined_text,
                "metadata": None
                if not refine_with_llm
                else {
                    "model_id": self.llm.model_id,
                    "max_new_tokens": self.llm.max_new_tokens,
                },
            },
            "combined_input": combined,
            "final_output": {"text": final["text"], "metadata": final["metadata"]},
            "models": {"diffusion": plan["metadata"], "llm": final["metadata"]},
        }
        # path = save_json(result, self.cfg.runtime.output_dir, self.cfg.runtime.json_prefix)
        # result["saved_path"] = path
        return result