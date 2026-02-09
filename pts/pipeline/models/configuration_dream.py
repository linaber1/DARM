"""
Dream model configuration
"""
from transformers import AutoConfig, PretrainedConfig


class DreamConfig(PretrainedConfig):
    """
    Configuration class for Dream diffusion models.
    Wraps the standard LLaMA config to ensure compatibility.
    """
    model_type = "dream"
    
    def __init__(self, **kwargs):
        # Dream uses standard LLaMA-based config
        super().__init__(**kwargs)


# Register the config class
AutoConfig.register("dream", DreamConfig)
