from pydantic import BaseModel

class DiffusionCfg(BaseModel):
    model_id: str
    device: str = "cuda:0"
    max_new_tokens: int = 256

class LLMCfg(BaseModel):
    model_id: str
    device: str = "cuda:0"
    max_new_tokens: int = 512
    temperature: float = 0.2
    do_sample: bool = False

class RuntimeCfg(BaseModel):
    output_dir: str = "outputs"
    json_prefix: str = "run"
    seed: int = 42

class PromptingCfg(BaseModel):
    final_system_msg: str = "You are a precise assistant. Produce a clear, concise answer."
    concat_header_plan: str = "## PLAN"
    concat_header_refined: str = "## REFINED"
    concat_header_extra: str = "## CONTEXT"

class AppCfg(BaseModel):
    pipeline_type: str = "single"
    diffusion: DiffusionCfg
    llm: LLMCfg
    runtime: RuntimeCfg
    prompting: PromptingCfg
