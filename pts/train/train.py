from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm
import torch
from torch import nn
from datasets import Dataset, Features, Value, Array2D, concatenate_datasets
import os
import sys
import numpy as np
from tqdm import tqdm
from peft import LoraConfig
import json

# ============================================================================
# PROJECTOR SETUP
# ============================================================================
# The projector bridges two models:
# - Input: Diffusion model latents (4096 dimensions)
# - Output: LLM embeddings (3072 dimensions for Llama-3.2-3B)
# 
# Architecture: 4096 -> 1024 -> 1024 -> 3072
# - Uses bottleneck design to force learning of compact representations
# - Includes GELU activations for non-linearity
# - Ends with RMSNorm for scale normalization
# ============================================================================

def add_latent_projector(model, config):
    """
    Add a latent projector to transform diffusion latents to LLM embeddings.
    
    Args:
        model: LlamaForCausalLM model to which we'll attach the projector
        config: Model config containing hidden_size (3072 for Llama-3.2-3B)
    
    Returns:
        model: Same model with latents_projector attribute added
    
    The projector is a 6-layer Sequential network:
        0: Linear(4096 -> 1024)     - Compression layer
        1: GELU                      - Non-linear activation
        2: Linear(1024 -> 1024)     - Bottleneck processing
        3: GELU                      - Non-linear activation
        4: Linear(1024 -> 3072)     - Expansion to LLM dimension
        5: RMSNorm(3072)            - Normalization (matches LLM scale)
    """
    model.bottleneck_dim = 1024  # Compressed representation size
    
    model.latents_projector = nn.Sequential(
        # Layer 0: Compress 4096-dim diffusion latents to 1024-dim bottleneck
        nn.Linear(4096, model.bottleneck_dim),
        
        # Layer 1: GELU activation (smooth, non-linear transformation)
        nn.GELU(approximate="tanh"),
        
        # Layer 2: Process at bottleneck dimension (learns complex mappings)
        nn.Linear(model.bottleneck_dim, model.bottleneck_dim),
        
        # Layer 3: Another GELU activation
        nn.GELU(approximate="tanh"),
        
        # Layer 4: Expand from bottleneck to LLM hidden size (3072)
        nn.Linear(model.bottleneck_dim, config.hidden_size),
        
        # Layer 5: RMSNorm - normalizes scale to match text embeddings
        # This is critical for preventing latents from overwhelming text
        LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps),
    ).to(model.device)
    
    return model


def init_latent_projector(model):
    """
    Initialize projector weights with Xavier (Glorot) uniform initialization.
    
    Why Xavier initialization?
    - Keeps gradients stable during backpropagation
    - Prevents vanishing/exploding gradients
    - Works well for networks with linear layers + activations
    
    Xavier formula: weights ~ Uniform(-a, a) where a = sqrt(6/(fan_in + fan_out))
    
    We initialize:
    - All Linear layer weights with Xavier uniform
    - All Linear layer biases with zeros (standard practice)
    - Skip GELU layers (no parameters)
    - Skip RMSNorm (has its own initialization)
    """
    with torch.no_grad():  # Don't track gradients during initialization
        
        # Layer 0: Linear(4096 -> 1024)
        nn.init.xavier_uniform_(model.latents_projector[0].weight)
        nn.init.zeros_(model.latents_projector[0].bias)
        
        # Layer 1: GELU - no parameters, skip
        
        # Layer 2: Linear(1024 -> 1024)
        nn.init.xavier_uniform_(model.latents_projector[2].weight)
        nn.init.zeros_(model.latents_projector[2].bias)
        
        # Layer 3: GELU - no parameters, skip
        
        # Layer 4: Linear(1024 -> 3072) - CRITICAL FIX: This was missing!
        nn.init.xavier_uniform_(model.latents_projector[4].weight)
        nn.init.zeros_(model.latents_projector[4].bias)
        
        # Layer 5: RMSNorm - has its own initialization, skip


# ============================================================================
# MODEL AND TOKENIZER SETUP
# ============================================================================
# Load the base LLM and attach our custom projector
# ============================================================================

# Base model: Llama 3.2 3B Instruct
# - 3B parameters
# - Hidden size: 3072
# - Instruction-tuned for following prompts
model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Load tokenizer
# - Handles text -> token ID conversion
# - trust_remote_code=True allows custom tokenizers
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Set padding token (Llama doesn't have one by default)
# Using a special token from the reserved range
tokenizer.pad_token = "<|finetune_right_pad_id|>"
tokenizer.padding_side = "right"  # Add padding on the right side

# Load base model
# - bfloat16: Uses 16-bit brain floating point (faster, less memory)
# - device_map="auto": Automatically distributes model across available GPUs
model = LlamaForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    #attn_implementation="flash_attention_2",  # ← ADD THIS
)

# Add the projector to the model
# This modifies the model in-place, adding model.latents_projector
model = add_latent_projector(model, model.config)

# Initialize projector weights with good starting values
init_latent_projector(model)


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
# Set hyperparameters and paths
# ============================================================================

# Template for prompts (not used directly, just for documentation)
LLM_TEMPLATE = """You are an expert in solving multiple-choice questions.
Given the following plan or reasoning, please solve the question. If the plan contains any explicit answer or option letter, ignore it and solve from the hints + question only.
{question}"""

# Special token used to represent latent positions in the input
# During training, we prepend 128 of these tokens to mark where latents go
latent_token = "<|reserved_special_token_3|>"

# Training run configuration
run_name = "llama3b-pixart-4bs-2grad-lora_nb3"  # Name for this training run
data_folder = "outputs/train_5x"                # Where training data is stored
batch_size = 2                                  # Samples per GPU (reduced from 4)
gradient_accumulation = 2                       # Accumulate gradients over 2 steps
                                                # Effective batch size = 2 * 2 = 4
num_epochs = 3 #10                                 # Train for 10 full passes
learning_rate = 1e-4   #5e-4                            # Step size for optimizer
weight_decay = 0.01 #0.001                            # L2 regularization strength
warmup_steps = 300                              # Gradually increase LR for 300 steps
save_steps = 200                                # Save checkpoint every 100 steps
output_dir = f"models/{run_name}"               # Where to save checkpoints
eval_size_per_type = 10                         # Samples per dataset type in eval set

os.makedirs(output_dir, exist_ok=True)

# Dataset cache directories (using /scratch to avoid quota issues)
ds_root = "/scratch/izar/berrayan/cached_datasets"
os.makedirs(ds_root, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = "/scratch/izar/berrayan/hf_cache"
os.makedirs("/scratch/izar/berrayan/hf_cache", exist_ok=True)

cached_train_data_path = f"{ds_root}/train_data_{os.path.basename(model_id)}"
cached_eval_data_path = f"{ds_root}/eval_data_{os.path.basename(model_id)}"


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================
# Load pre-computed latents and create training/eval datasets
# 
# Data format:
# - prompt: Question text
# - answer: Ground truth answer
# - latents: Pre-computed diffusion model latents (128, 4096)
# - plan: Text generated by diffusion model
# - dataset: Source dataset name (arc_easy, arc_challenge, etc.)
# ============================================================================

if os.path.exists(cached_train_data_path) and os.path.exists(cached_eval_data_path):
    # Load from cache if available (much faster)
    train_data = Dataset.load_from_disk(cached_train_data_path)
    eval_data = Dataset.load_from_disk(cached_eval_data_path)
    print(f"Loaded cached data: {len(train_data)} training samples, {len(eval_data)} eval samples.")
    
else:
    # Load raw data and prepare datasets
    merged_file = f"{data_folder}/all_merged_latents.npy"
    print(f"Loading data from {merged_file}...")
    raw_data = np.load(merged_file, allow_pickle=True)
    print(f"Loaded {len(raw_data)} samples from merged file")
    
    # Extract fields from raw data
    questions, answers, latents, plans, ds_types = [], [], [], [], []
    for d in tqdm(raw_data, desc="Processing samples"):
        questions.append(d["prompt"])    # Question text
        answers.append(d["answer"])      # Ground truth
        latents.append(d["latents"])     # Pre-computed latents (128, 4096)
        plans.append(d["plan"])          # Diffusion model's plan text
        ds_types.append(d["dataset"])    # Dataset source
    
    print(f"Processed {len(questions)} samples")

    data = []       # Will hold training samples
    eval_data = {}  # Will hold eval samples (organized by dataset type)
    
    # Pattern to split prompt from completion in chat template
    split_pattern = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # Process each sample
    for question, answer, latent, plan, ds_type in tqdm(
        zip(questions, answers, latents, plans, ds_types), 
        desc="Preparing data"
    ):
        # Prepend 128 latent token placeholders to the question
        # Format: "<special_token><special_token>...<special_token>Question text"
        # During training, these tokens will be replaced with actual latent embeddings
        question = "Given these hints : \n".join([latent_token]*128) + question
        
        # Format as chat conversation
        conversations = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        
        # Apply chat template to get formatted string
        # Example output:
        # <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        # Question here<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        # Answer here<|eot_id|>
        msg = tokenizer.apply_chat_template(conversations, tokenize=False)
        
        # Split into prompt (input) and completion (target)
        prompt, completion = msg.split(split_pattern)
        prompt += split_pattern  # Include the assistant header in prompt
        
        # Convert latents to float16 to save memory
        latent = latent.astype(np.float16, copy=False)
        
        # Create sample dict
        sample = {
            "prompt": prompt,           # Text up to assistant header
            "completion": completion,   # Text to predict
            "ds_type": ds_type,        # Dataset source
            "latents": latent          # Pre-computed latents (128, 4096)
        }
        
        # Split into train/eval
        # Keep eval_size_per_type samples per dataset type for evaluation
        if len(eval_data.get(ds_type, [])) < eval_size_per_type:
            eval_data[ds_type] = eval_data.get(ds_type, [])
            eval_data[ds_type].append(sample)
        else:
            data.append(sample)

    # Define dataset features schema
    # This ensures consistent data types across shards
    features = Features({
        "prompt": Value("string"),
        "completion": Value("string"),
        "ds_type": Value("string"),
        "latents": Array2D(shape=(128, 4096), dtype="float16"),  # Fixed-size 2D array
    })

    def to_sharded_dataset(rows, features, shard_size=128):
        """
        Create dataset in shards to avoid OOM errors.
        
        Large datasets loaded all at once can cause memory issues.
        This function processes data in chunks (shards) of 128 samples.
        """
        shards = []
        for i in tqdm(range(0, len(rows), shard_size), desc="Sharding data"):
            # Get a chunk of samples
            shard = rows[i:i+shard_size]
            
            # Ensure latents are float16
            for s in shard:
                s["latents"] = s["latents"].astype(np.float16, copy=False)
            
            # Create a Dataset from this shard
            shards.append(Dataset.from_list(shard, features=features))
        
        # Concatenate all shards into one large dataset
        return concatenate_datasets(shards) if len(shards) > 1 else shards[0]
    
    # Free memory
    del raw_data, questions, answers, latents, plans, ds_types
    
    # Flatten eval_data dict into list
    eval_data = [item for sublist in eval_data.values() for item in sublist]
    
    # Create and save datasets
    eval_data = Dataset.from_list(eval_data, features=features)
    eval_data.save_to_disk(cached_eval_data_path)
    
    train_data = to_sharded_dataset(data, features, shard_size=128)
    train_data.save_to_disk(cached_train_data_path)
    
    print(f"Training on {len(train_data)} samples, evaluating on {len(eval_data)} samples.")


# ============================================================================
# TRAINING ARGUMENTS
# ============================================================================
# Configure the SFTTrainer with all hyperparameters
# ============================================================================

training_args = SFTConfig(
    do_train=True,                              # Enable training
    do_eval=True,                               # Enable evaluation
    do_predict=False,                           # No prediction phase
    
    # Sequence length configuration
    #max_length=512,                             # Maximum length of the tokenized sequence
    
    # Batch configuration
    per_device_train_batch_size=batch_size,     # 2 samples per GPU (reduced for memory)
    gradient_accumulation_steps=gradient_accumulation,  # Accumulate over 2 steps
                                                # Effective batch = 2 * 2 = 4
    
    # Evaluation strategy
    eval_strategy="steps",                      # Evaluate every N steps
    eval_steps=0,                             # Evaluate every 100 steps
    
    # Training duration
    num_train_epochs=num_epochs,                # Train for 10 epochs
    
    # Optimizer settings
    learning_rate=learning_rate,                # 5e-4
    weight_decay=weight_decay,                  # 0.001 (L2 regularization)
    optim="adamw_torch",                        # AdamW optimizer
    
    # Learning rate schedule
    lr_scheduler_type="cosine",                 # Cosine annealing
    warmup_steps=warmup_steps,                  # Warm up for 300 steps
    
    # Logging and saving
    logging_steps=2,                            # Log every 2 steps
    save_strategy="steps",                      # Save checkpoints by steps
    save_steps=save_steps,                      # Save every 500 steps
    save_total_limit=4,                         # Keep only 3 checkpoints
    
    # Best model tracking
    load_best_model_at_end=True,                # Load best checkpoint at end
    metric_for_best_model="eval_loss",          # Use eval loss to pick best
    greater_is_better=False,                    # Lower eval loss is better
    
    # Output and reporting
    output_dir=output_dir,                      # Where to save checkpoints
    run_name=run_name,                          # Name for W&B logging
    report_to="wandb",                          # Log to Weights & Biases
    
    # Precision and performance
    fp16=False,                                 # Don't use FP16
    bf16=True,                                  # Use BF16 (better for training)
    
    # Gradient stability
    max_grad_norm=1.0,                          # Clip gradients to prevent explosions
    
    # Data loading
    dataloader_num_workers=1,                   # Workers for data loading
    dataset_num_proc=1,                         # Processes for dataset mapping
    dataloader_pin_memory=True,                 # Pin memory for faster transfer
    remove_unused_columns=False,                # Keep all columns (need latents!)
    
    # Misc
    prediction_loss_only=False,                 # Compute all metrics
)


# ============================================================================
# FREEZE/UNFREEZE PARAMETERS
# ============================================================================
# Only train the projector and LoRA adapters, freeze everything else
# 
# Training budget:
# - LLM backbone: FROZEN (3B params) - too expensive to train
# - LLM head: FROZEN - already well-trained
# - Projector: TRAINABLE (~13M params) - this is what we're training!
# - LoRA adapters: TRAINABLE (~8M params) - efficient fine-tuning
# 
# Total trainable: ~21M params (~0.7% of full model)
# ============================================================================

for p in model.model.parameters():
    p.requires_grad = False  # Freeze entire LLM backbone (all transformer layers)

for p in model.latents_projector.parameters():
    p.requires_grad = True   # Train projector - THIS IS THE KEY COMPONENT!

for p in model.lm_head.parameters():
    p.requires_grad = False  # Freeze language modeling head
 
# Freeze the 2nd Linear layer in projector
# This reduces trainable params and may help generalization    
for p in model.latents_projector[2].parameters():
    p.requires_grad = False
    


# ============================================================================
# CREATE TRAINER
# ============================================================================
# SFTTrainer handles the training loop, including:
# - Forward passes with latent injection
# - Loss computation
# - Backpropagation
# - Optimizer steps
# - Logging and checkpointing
# ============================================================================

trainer = SFTTrainer(
    args=training_args,                         # Training configuration
    model=model,                                # Model with projector attached
    train_dataset=train_data,                   # Training samples
    eval_dataset=eval_data,                     # Evaluation samples
    processing_class=tokenizer,                 # Tokenizer for text processing
    
    # LoRA configuration - adds trainable low-rank adapters to attention layers
    # peft_config=LoraConfig(
    #     r=8,                                    # Rank of LoRA matrices (8 is typical)
    #     target_modules=[                        # Which modules to add LoRA to
    #         "q_proj",                           # Query projection
    #         "k_proj",                           # Key projection
    #         "v_proj",                           # Value projection
    #         "o_proj",                           # Output projection
    #         "gate_proj",                        # Gate projection (for GLU)
    #         "up_proj",                          # Up projection (MLP)
    #         "down_proj"                         # Down projection (MLP)
    #     ],
    #     lora_alpha=32,                          # LoRA scaling factor (alpha/r = 4x scaling)
    # )
    peft_config=LoraConfig(
        r=4,                    # ↓ from 8
        lora_alpha=16,          # keep alpha/r = 4
        target_modules=[
            "q_proj",
            "v_proj",
            "o_proj",
        ],
    )

)


# ============================================================================
# TRAINING
# ============================================================================
# Run the actual training loop
# 
# What happens during training:
# 1. Load batch of samples (prompt, completion, latents)
# 2. Tokenize the prompt text
# 3. Get text embeddings from LLM
# 4. Project latents: (128, 4096) -> (128, 3072)
# 5. Concatenate: [projected_latents, text_embeddings]
# 6. Feed combined embeddings to LLM
# 7. Compute loss on completion tokens
# 8. Backpropagate through projector (and LoRA adapters)
# 9. Update projector weights
# ============================================================================

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80 + "\n")

# Check if checkpoint exists to resume from
checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
if checkpoint_dirs:
    # Sort by checkpoint number and get the latest
    latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))[-1]
    checkpoint_path = os.path.join(output_dir, latest_checkpoint)
    print(f"📦 Resuming from checkpoint: {checkpoint_path}")
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    print("🆕 Starting training from scratch")
    trainer.train()

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80 + "\n")


# ============================================================================
# SAVE MODEL WITH PROJECTOR - CRITICAL SECTION
# ============================================================================
# This is the most important part for making the model reusable!
# 
# The problem:
# - trainer.model is a PEFT model (has LoRA adapters)
# - merge_and_unload() merges LoRA back into base model
# - BUT it doesn't save the projector! (not part of standard LlamaForCausalLM)
# 
# The solution:
# - Save projector BEFORE merging
# - Save as separate .pth file
# - Also save metadata for documentation
# ============================================================================

print("="*80)
print("SAVING MODEL AND PROJECTOR")
print("="*80 + "\n")

merged_dir = os.path.join(output_dir, "final_merged")
os.makedirs(merged_dir, exist_ok=True)

# Get the trained model (wrapped by PEFT)
peft_model = trainer.model

# Access the base model (which has the projector attached)
# PEFT wraps the model, so we need to unwrap it
if hasattr(peft_model, 'base_model'):
    base_with_projector = peft_model.base_model
elif hasattr(peft_model, 'model'):
    base_with_projector = peft_model.model
else:
    base_with_projector = peft_model

# ============================================================================
# SAVE THE PROJECTOR
# ============================================================================
# This is CRITICAL - without this, the projector weights are lost!
# ============================================================================

if hasattr(base_with_projector, 'latents_projector'):
    print("✓ Found latents_projector in trained model")
    
    # Get the state dict (all weights and biases)
    # This includes all 7 trainable parameters:
    # - Layer 0: weight (4096x1024), bias (1024)
    # - Layer 2: weight (1024x1024), bias (1024)
    # - Layer 4: weight (3072x1024), bias (3072)
    # - Layer 5: RMSNorm weight (3072)
    projector_state = base_with_projector.latents_projector.state_dict()
    
    projector_path = os.path.join(merged_dir, "latents_projector.pth")
    
    # Move to CPU before saving (in case it's on GPU)
    projector_state_cpu = {k: v.cpu() for k, v in projector_state.items()}
    
    # Save as PyTorch checkpoint file
    torch.save(projector_state_cpu, projector_path)
    
    print(f"✓ Saved trained projector to: {projector_path}")
    print(f"✓ Projector keys: {list(projector_state_cpu.keys())}")
    
    # Verify the save worked
    loaded = torch.load(projector_path)
    print(f"✓ Verified: {len(loaded)} weights saved")
    
    # Show statistics for first 3 weights
    for k, v in list(loaded.items())[:3]:
        # Check mean and std to ensure weights are not random/untrained
        # Trained weights should have specific distributions
        print(f"    {k}: {v.shape}, mean={v.float().mean():.6f}, std={v.float().std():.6f}")
        
else:
    # This should never happen, but good to check
    print("✗ WARNING: No latents_projector found in model!")
    print(f"  Available attributes: {[a for a in dir(base_with_projector) if not a.startswith('_')][:20]}")


# ============================================================================
# SAVE THE BASE MODEL (with LoRA merged)
# ============================================================================
# Merge LoRA adapters back into the base model and save
# ============================================================================

print("\nMerging LoRA adapters...")

# Merge LoRA weights into the base model
# This creates a standard LlamaForCausalLM without PEFT wrappers
merged = peft_model.merge_and_unload()

# Save the merged model (standard HuggingFace format)
# This saves:
# - model-00001-of-00002.safetensors (shard 1)
# - model-00002-of-00002.safetensors (shard 2)
# - config.json
# - generation_config.json
merged.save_pretrained(merged_dir)

# Save tokenizer files
# - tokenizer.json
# - tokenizer_config.json
# - special_tokens_map.json
tokenizer.save_pretrained(merged_dir)

print(f"✓ Merged model saved to {merged_dir}")


# ============================================================================
# SAVE METADATA
# ============================================================================
# Document the projector architecture for future reference
# ============================================================================

metadata = {
    "projector_architecture": "4096 -> 1024 -> 1024 -> 3072",
    "bottleneck_dim": 1024,
    "input_dim": 4096,                          # Diffusion latent dimension
    "output_dim": model.config.hidden_size,     # LLM hidden size (3072)
    "num_latent_tokens": 128,                   # Number of latent tokens
    "training_run": run_name,                   # Training run name
}

metadata_path = os.path.join(merged_dir, "projector_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Saved projector metadata to {metadata_path}")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TRAINING AND SAVING COMPLETE")
print("="*80)
print(f"\nModel directory: {merged_dir}")
print(f"\nFiles saved:")
print(f"  - Merged model (with LoRA): model-*.safetensors")
print(f"  - Trained projector: latents_projector.pth")  # ← CRITICAL FILE!
print(f"  - Metadata: projector_metadata.json")
print(f"  - Tokenizer: tokenizer*.json")
print(f"  - Config: config.json, generation_config.json")
print("\n" + "="*80 + "\n")

# ============================================================================
# WHAT HAPPENS NEXT (Inference)
# ============================================================================
# To use this trained model:
#
# 1. Load the base LLM:
#    model = LlamaForCausalLM.from_pretrained(merged_dir)
#
# 2. Recreate projector architecture:
#    model = add_latent_projector(model, model.config)
#
# 3. Load trained projector weights:
#    state_dict = torch.load(f"{merged_dir}/latents_projector.pth")
#    model.latents_projector.load_state_dict(state_dict)
#
# 4. Generate latents from diffusion model
#
# 5. Project latents and concatenate with text embeddings
#
# 6. Generate with LLM!
# ============================================================================