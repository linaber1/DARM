import torch
from peft import PeftModel
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm
import os
from torch import nn


# Define the add_latent_projector function from your training script
def add_latent_projector(model, config):
    """Add latent projector to a LlamaForCausalLM model"""
    model.bottleneck_dim = 1024
    model.latents_projector = nn.Sequential(
        nn.Linear(4096, model.bottleneck_dim),
        nn.GELU(approximate="tanh"),
        nn.Linear(model.bottleneck_dim, model.bottleneck_dim),
        nn.GELU(approximate="tanh"),
        nn.Linear(model.bottleneck_dim, config.hidden_size),
        LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps),
    ).to(model.device)
    return model

def init_latent_projector(model):
    """Initialize the latent projector weights"""
    with torch.no_grad():
        nn.init.xavier_uniform_(model.latents_projector[0].weight)
        nn.init.zeros_(model.latents_projector[0].bias)
        nn.init.xavier_uniform_(model.latents_projector[2].weight)
        nn.init.zeros_(model.latents_projector[2].bias)


print("Loading base model...")
base_model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu"  # Use CPU to avoid GPU memory issues
)

print("Adding latent projector...")
base_model = add_latent_projector(base_model, base_model.config)

print("Initializing latent projector...")
init_latent_projector(base_model)

print("Loading LoRA checkpoint...")
checkpoint_path = "/home/berrayan/planner/planner_executor_DiscreteDiffusion/models/pts-lora/checkpoint-4340"

model = PeftModel.from_pretrained(base_model, checkpoint_path)

# Check if projector exists
if hasattr(model.base_model, 'latents_projector'):
    print("✓ Found latents_projector!")
    projector = model.base_model.latents_projector
    print(f"  Projector type: {type(projector)}")
    print(f"  Projector state dict keys: {list(projector.state_dict().keys())}")
    
    # Save to final_merged directory
    output_path = "/home/berrayan/planner/planner_executor_DiscreteDiffusion/models/pts-lora/final_merged/latents_projector.pth"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.save(projector.state_dict(), output_path)
    print(f"✓ Saved projector to: {output_path}")
    
    # Verify the save
    loaded = torch.load(output_path)
    print(f"✓ Verified: {len(loaded)} weights saved")
    for k, v in loaded.items():
        print(f"    {k}: {v.shape}")
        
elif hasattr(model.base_model.model, 'latents_projector'):
    print("✓ Found latents_projector in model.base_model.model!")
    projector = model.base_model.model.latents_projector
    output_path = "/home/berrayan/planner/planner_executor_DiscreteDiffusion/models/pts-lora/final_merged/latents_projector.pth"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(projector.state_dict(), output_path)
    print(f"✓ Saved projector to: {output_path}")
else:
    print("✗ No projector found! Checking model structure...")
    print(f"  base_model attributes: {dir(model.base_model)}")
    print(f"  Looking for 'latents_projector' or similar...")
    for attr in dir(model.base_model):
        if 'latent' in attr.lower() or 'project' in attr.lower():
            print(f"    Found: {attr}")