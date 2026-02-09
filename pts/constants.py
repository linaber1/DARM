from dataclasses import dataclass

@dataclass
class Pipelines:
    diffusion_llm = "diffusion-llm"
    llm_diffusion = "llm-diffusion"
    diffusion_diffusion = "diffusion-diffusion"
    llm_llm = "llm-llm"
    llm_only = "llm-only"
    diffusion_only = "diffusion-only"
    ld_dual = "ld_dual"
    dl_dual = "dl_dual"
    
    @staticmethod
    def all_architectures():
        return [
            Pipelines.diffusion_llm,
            Pipelines.llm_diffusion,
            Pipelines.diffusion_diffusion,
            Pipelines.llm_llm,
            Pipelines.llm_only,
            Pipelines.diffusion_only,
            Pipelines.ld_dual,
            Pipelines.dl_dual
        ]
    
    @staticmethod       
    def diffusion_plan():
        return [
            Pipelines.diffusion_llm,
            Pipelines.diffusion_diffusion,
            Pipelines.dl_dual,
        ]
        
    @staticmethod
    def llm_plan():
        return [
            Pipelines.llm_diffusion,
            Pipelines.llm_llm,
            Pipelines.ld_dual,
        ]
        
    @staticmethod
    def llm_answer():
        return [
            Pipelines.diffusion_llm,
            Pipelines.llm_llm,
            Pipelines.llm_only,
            Pipelines.dl_dual
        ]
        
    @staticmethod
    def diffusion_answer():
        return [
            Pipelines.llm_diffusion,
            Pipelines.diffusion_diffusion,
            Pipelines.diffusion_only,
            Pipelines.ld_dual
        ]