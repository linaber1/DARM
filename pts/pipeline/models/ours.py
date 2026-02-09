from typing import Dict

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm
import torch
import torch.nn.functional as F
import numpy as np
from .modeling_llada import LLaDAModelLM
from .modeling_dream import DreamModelLM
from torch import nn


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    
    # Store final representation as latents
    latent_representation = None

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                y = model(x_, output_hidden_states=True)
                hidden_states = y.hidden_states
                logits = y.logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                y = model(x, output_hidden_states=True)
                hidden_states = y.hidden_states
                logits = y.logits
            
            # Store hidden_states as latent representation
            if hidden_states is not None:
                latent_representation = hidden_states

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
    return x, latent_representation


@ torch.no_grad()
def generate_early(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                   cfg_scale=0., remasking='low_confidence', mask_id=126336, stop_at_ratio=0.7):
    """
    Generate with early stopping at a specific ratio of total steps.
    Returns partially denoised latents at stop_at_ratio * steps.
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    
    # Calculate stopping point
    total_steps = steps_per_block * num_blocks
    stop_at_step = int(total_steps * stop_at_ratio)
    current_step = 0

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            if current_step >= stop_at_step:
                # Stop early and return current state
                y = model(x, output_hidden_states=True)
                hidden_states = y.hidden_states
                return x, hidden_states
            
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                y = model(x_, output_hidden_states=True)
                hidden_states = y.hidden_states
                logits = y.logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                y = model(x, output_hidden_states=True)
                hidden_states = y.hidden_states
                logits = y.logits
            
            # Store hidden_states as latent representation
            if hidden_states is not None:
                latent_representation = hidden_states

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            
            current_step += 1
    
    # If we get here, we've completed all steps (shouldn't happen with stop_at_ratio < 1.0)
    return x, hidden_states

from transformers.models.llama.modeling_llama import LlamaRMSNorm
from torch import nn
import os

class OurDualModel:
    def add_latent_projector(self, model, config):
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
    
    def __init__(self, diff_id: str, llm_id: str, max_new_latents: int = 128, 
                 max_new_tokens=128, temperature: float = 0, do_sample: bool = False, **kwargs):
        self.diff_id = diff_id
        self.llm_id = llm_id
        self.max_new_latents = max_new_latents
        self.max_new_tokens = max_new_tokens
        
        self.temperature = temperature
        self.do_sample = do_sample
         
        self.diff_tokenizer = AutoTokenizer.from_pretrained(diff_id, trust_remote_code=True)
        
        # Dynamically load either LLaDA or Dream model based on model_id
        if "Dream" in diff_id or "dream" in diff_id:
            self.diff_model = DreamModelLM.from_pretrained(
                diff_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).to("cuda").eval()
        else:
            self.diff_model = LLaDAModelLM.from_pretrained(
                diff_id,
                trust_remote_code=True,
                dtype=torch.bfloat16,
            ).to("cuda").eval()
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=True, fix_mistral_regex=True)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        
        # Load base LLM model
        from transformers import LlamaForCausalLM
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        
        # Add projector architecture
        print("[INFO] Adding latent projector architecture...")
        self.llm_model = self.add_latent_projector(self.llm_model, self.llm_model.config)
        
        # Load trained projector weights
        projector_path = "/home/berrayan/planner/planner_executor_DiscreteDiffusion/models/llama3b-pixart-4bs-2grad-lora_nb3/final_merged/latents_projector.pth"
        if os.path.exists(projector_path):
            print(f"[INFO] Loading trained latents_projector from {projector_path}")
            state_dict = torch.load(projector_path, map_location="cpu")
            self.llm_model.latents_projector.load_state_dict(state_dict)
            
            # Move to correct device
            device = next(self.llm_model.parameters()).device
            self.llm_model.latents_projector = self.llm_model.latents_projector.to(device).to(torch.bfloat16)
            self.latents_projector = self.llm_model.latents_projector
            
            print(f"[INFO] ✓ Latents projector loaded successfully on {device}!")
            print(f"[INFO] ✓ Projector architecture: 4096 -> 1024 -> 1024 -> {self.llm_model.config.hidden_size}")
        else:
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
            print(hvygbor)
            print(f"[WARNING] Projector weights not found at {projector_path}")
            print("[WARNING] Using randomly initialized projector!")
            self.latents_projector = self.llm_model.latents_projector


    
    # def _load_or_create_latents_projector(self):
    #     """
    #     Try to load latents_projector from the checkpoint directory.
    #     The weights may exist in the checkpoint but not load due to them not being
    #     part of the standard LlamaForCausalLM architecture.
    #     """
    #     import os
    #     from safetensors.torch import load_file as load_safetensors
        
    #     model_path = self.llm_id
    #     latents_projector_weights = {}
    #     weights_found = False
        
    #     # Check for safetensors files in the model directory
    #     if os.path.isdir(model_path):
    #         # Look for model files (both single and sharded)
    #         for filename in sorted(os.listdir(model_path)):
    #             if ("model" in filename and filename.endswith(".safetensors")) or filename == "model.safetensors":
    #                 filepath = os.path.join(model_path, filename)
    #                 try:
    #                     # Use load_safetensors to load weights
    #                     full_state = load_safetensors(filepath)
    #                     for key, value in full_state.items():
    #                         if key.startswith("latents_projector."):
    #                             # Remove 'latents_projector.' prefix for state_dict
    #                             latents_projector_weights[key.replace("latents_projector.", "")] = value
    #                             weights_found = True
    #                     if weights_found:
    #                         break  # Found weights, no need to continue
    #                 except Exception as e:
    #                     print(f"[DEBUG] Could not load weights from {filename}: {e}")
    #                     continue
            
    #         # Also check pytorch_model.bin if safetensors not found
    #         pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
    #         if os.path.exists(pytorch_model_path) and not weights_found:
    #             try:
    #                 checkpoint = torch.load(pytorch_model_path, map_location="cpu")
    #                 for key in checkpoint.keys():
    #                     if key.startswith("latents_projector."):
    #                         latents_projector_weights[key.replace("latents_projector.", "")] = checkpoint[key]
    #                         weights_found = True
    #             except Exception as e:
    #                 print(f"[DEBUG] Could not load weights from pytorch_model.bin: {e}")
        
    #     if weights_found:
    #         try:
    #             # Reconstruct the latents_projector with the correct architecture
    #             hidden_size = self.llm_model.config.hidden_size
    #             bottleneck_dim = 1024
                
    #             # Get the device of the LLM model to ensure compatibility
    #             llm_device = next(self.llm_model.parameters()).device
                
    #             self.latents_projector = nn.Sequential(
    #                 nn.Linear(4096, bottleneck_dim),
    #                 nn.GELU(approximate="tanh"),
    #                 nn.Linear(bottleneck_dim, bottleneck_dim),
    #                 nn.GELU(approximate="tanh"),
    #                 nn.Linear(bottleneck_dim, hidden_size),
    #                 LlamaRMSNorm(hidden_size, eps=self.llm_model.config.rms_norm_eps),
    #             )
                
    #             # First load the weights (on CPU to avoid memory issues)
    #             self.latents_projector.load_state_dict(latents_projector_weights, strict=True)
                
    #             # Then move the entire module to the correct device and dtype
    #             self.latents_projector = self.latents_projector.to(llm_device).to(torch.bfloat16)
                
    #             print(f"[INFO] Successfully loaded latents_projector from checkpoint on device {llm_device}")
    #         except Exception as e:
    #             print(f"[WARNING] Failed to load latents_projector weights: {e}")
    #             self.latents_projector = None
    #             print("[WARNING] Model does not have latents_projector; skipping latents and using LLM only")
    #     else:
    #         self.latents_projector = None
    #         print("[WARNING] Model does not have latents_projector; skipping latents and using LLM only")

    def generate_plan(self, prompt: str, **kwargs) -> Dict:
        print(f"Generating Plan for {prompt}")
        m = [{"role": "user", "content": prompt}]
        prompt = self.diff_tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = self.diff_tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to("cuda").unsqueeze(0)
        input_length = input_ids.shape[1]
        text, latents = generate(self.diff_model, input_ids, steps=128, gen_length=self.max_new_latents, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
        
        # Ensure latents is a proper tensor
        if latents is None or (isinstance(latents, tuple)):
            print(f"[WARNING] latents is {type(latents)}, using zeros as fallback")
            latents = torch.zeros((1, self.max_new_latents, 4096), dtype=torch.bfloat16, device="cuda")
        elif not isinstance(latents, torch.Tensor):
            latents = torch.tensor(latents, dtype=torch.bfloat16, device="cuda")
        
        latents = latents[:, input_length:, :]
        text = text[:, input_length:]
        text = self.diff_tokenizer.decode(text[0], skip_special_tokens=True)
        print(text)
        return {
            "latents": latents,
            "text": text,
            "metadata": {
                "diff_id": self.diff_id,
                "max_new_latents": self.max_new_latents
            }
        }

    def generate_plan_early(self, prompt: str, stop_at_ratio: float = 0.7, **kwargs) -> Dict:
        """
        Generate plan but stop denoising at stop_at_ratio (default 70%) of total steps.
        Returns partially denoised latents.
        """
        print(f"Generating Plan (Early Stop at {stop_at_ratio*100}%) for {prompt}")
        m = [{"role": "user", "content": prompt}]
        prompt = self.diff_tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = self.diff_tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to("cuda").unsqueeze(0)
        input_length = input_ids.shape[1]
        text, latents = generate_early(
            self.diff_model, 
            input_ids, 
            steps=128, 
            gen_length=self.max_new_latents, 
            block_length=32, 
            temperature=0., 
            cfg_scale=0., 
            remasking='low_confidence',
            stop_at_ratio=stop_at_ratio
        )
        
        # Ensure latents is a proper tensor
        if latents is None or (isinstance(latents, tuple)):
            print(f"[WARNING] latents is {type(latents)}, using zeros as fallback")
            latents = torch.zeros((1, self.max_new_latents, 4096), dtype=torch.bfloat16, device="cuda")
        elif not isinstance(latents, torch.Tensor):
            latents = torch.tensor(latents, dtype=torch.bfloat16, device="cuda")

        latents = latents[:, input_length:, :]
        text = text[:, input_length:]
        text = self.diff_tokenizer.decode(text[0], skip_special_tokens=True)
        print(f"[Early Stop] Partially denoised text: {text}")
        return {
            "latents": latents,
            "text": text,
            "metadata": {
                "diff_id": self.diff_id,
                "max_new_latents": self.max_new_latents,
                "stop_at_ratio": stop_at_ratio,
                "early_stop": True
            }
        }


    
    
    def generate_answer(self, user_prompt: str, latents=None, system_prompt=None, use_latents=True, **kwargs) -> Dict:
        print(f"{user_prompt}\n{'='*50}")
        messages = []
        # Explicitly set empty system message to override tokenizer's default
        messages += [{"role": "system", "content": ""}]
        messages += [{"role": "user", "content": user_prompt}]
        prompt = self.llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        print(f"[DEBUG] Prompt:\n{prompt}\n[DEBUG ENDDDDDD]")
        # Tokenize prompt
        tokenized = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
        ).to("cuda")
        input_ids = tokenized["input_ids"]
        input_len = input_ids.shape[-1]
        # Build inputs_embeds by concatenating projected latents (if provided) before text embeddings
        input_embeds = self.llm_model.get_input_embeddings()(input_ids)
        attention_mask = torch.ones((input_embeds.shape[0], input_embeds.shape[1]), dtype=torch.long, device=input_embeds.device)
        
        # Track the total input length including latents
        total_input_len = input_len
        
        # Debug prints to check condition
        print(f"[DEBUG] Condition check - use_latents: {use_latents}")
        print(f"[DEBUG] Condition check - latents is not None: {latents is not None}")
        print(f"[DEBUG] Condition check - self.latents_projector is not None: {self.latents_projector is not None}")
        print(f"[DEBUG] Condition check - latents type: {type(latents)}")
        if latents is not None:
            print(f"[DEBUG] Condition check - latents shape: {getattr(latents, 'shape', 'No shape attribute')}")
        if self.latents_projector is not None:
            print(f"[DEBUG] Condition check - projector type: {type(self.latents_projector)}")
        
        # Handle latents concatenation
        if use_latents and latents is not None:
            print(f"[DEBUG] Using latents! Shape: {latents.shape}, device: {latents.device}, dtype: {latents.dtype}")
            print(f"[DEBUG] Latents stats - min: {latents.min().item():.4f}, max: {latents.max().item():.4f}, mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}")
            print(f"[DEBUG] Text embeddings device: {input_embeds.device}, dtype: {input_embeds.dtype}")
            print(f"[DEBUG] Text embeddings stats - min: {input_embeds.min().item():.4f}, max: {input_embeds.max().item():.4f}, mean: {input_embeds.mean().item():.4f}, std: {input_embeds.std().item():.4f}")
            
            # Ensure latents are on correct device/dtype
            latents = latents.to(input_embeds.device).to(input_embeds.dtype)
            
            # Check if we need to project latents externally
            if self.latents_projector is not None:
                print(f"[DEBUG] Using external latents_projector")
                print(f"[DEBUG] Projector device: {next(self.latents_projector.parameters()).device}")
                
                # Ensure latents are on the same device and dtype as the projector
                latents_device = next(self.latents_projector.parameters()).device
                latents = latents.to(latents_device).to(torch.bfloat16)
                
                # Project latents
                latents_proj = self.latents_projector(latents)
                print(f"[DEBUG] Projected latents shape: {latents_proj.shape}, device: {latents_proj.device}, dtype: {latents_proj.dtype}")
                print(f"[DEBUG] Projected latents stats - min: {latents_proj.min().item():.4f}, max: {latents_proj.max().item():.4f}, mean: {latents_proj.mean().item():.4f}, std: {latents_proj.std().item():.4f}")
                
                # Move projected latents to match text embeddings device
                latents_proj = latents_proj.to(input_embeds.device).to(input_embeds.dtype)
                print(f"[DEBUG] After moving - Projected latents device: {latents_proj.device}, dtype: {latents_proj.dtype}")
            else:
                print(f"[DEBUG] No external projector - using latents directly (LLM has internal projector)")
                print(wrong)
                latents_proj = latents
            
            # Check if scales are similar
            text_scale = input_embeds.std().item()
            latents_scale = latents_proj.std().item()
            print(f"[DEBUG] Scale comparison - Text std: {text_scale:.4f}, Latents std: {latents_scale:.4f}, Ratio: {latents_scale/text_scale:.2f}")
            
            # Concatenate latents before text embeddings to make sure the input embedding is not drowned out by the latents (since they are more informative and we want to ensure they have a strong influence on the generation)
            print(f"[DEBUG] Before concat - input_embeds shape: {input_embeds.shape}")
            input_embeds = torch.cat([latents_proj, input_embeds], dim=1)
            print(f"[DEBUG] After concat - input_embeds shape: {input_embeds.shape}")
            
            # Extend attention mask to match the concatenated embeddings (latents + text), ensuring the model can attend to both
            lat_mask = torch.ones((attention_mask.shape[0], latents_proj.shape[1]), dtype=torch.long, device=attention_mask.device)
            attention_mask = torch.cat([lat_mask, attention_mask], dim=1)
            
            # Update total input length to include latents
            total_input_len = input_embeds.shape[1]
            use_inputs_embeds = True
        else:
            print(f"[DEBUG] No latents used - using text only")
            use_inputs_embeds = False
        
        print(f"[DEBUG] Input embeddings device: {input_embeds.device}, dtype: {input_embeds.dtype}")
        print(f"[DEBUG] Attention mask device: {attention_mask.device}, dtype: {attention_mask.dtype}")
        
        # Generation
        with torch.no_grad():
            if use_inputs_embeds:
                print(f"[DEBUG] Generating WITH latents using inputs_embeds")
                outputs = self.llm_model.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else 1.0,
                    do_sample=self.do_sample,
                    repetition_penalty=1.2,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    output_scores=False,
                    return_dict_in_generate=False,
                )
            else:
                print(f"[DEBUG] Generating WITHOUT latents using input_ids")
                outputs = self.llm_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else 1.0,
                    do_sample=self.do_sample,
                    repetition_penalty=1.2,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                )
        
        print(f"[DEBUG] Generated output shape: {outputs.shape}")
        print(f"[DEBUG] Output length: {len(outputs[0])}")
        print(f"[DEBUG] Expected output length: ~{input_len + self.max_new_tokens} (prompt + new tokens)")
        print(f"[DEBUG] Full output IDs (last 30): {outputs[0][-30:]}")
        
        # When using inputs_embeds with max_new_tokens, we skip the original prompt length
        # (not total_input_len which includes latents, as latents don't produce token IDs)
        generated_token_ids = outputs[0][input_len:]
        print(f"[DEBUG] Generated token IDs ({len(generated_token_ids)} tokens): {generated_token_ids}")
        generated_text = self.llm_tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        print(f"[DEBUG] Generated text: '{generated_text}'")

        return {
            "text": generated_text.strip(),
            "metadata": {
                "llm_id": self.llm_id,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "do_sample": self.do_sample
            }
        }
        
        
    
    def generate_answer_old(self, user_prompt: str, latents=None, system_prompt=None, use_latents=True, **kwargs) -> Dict:
        print(f"{user_prompt}\n{'='*50}")
        messages = []
        # Explicitly set empty system message to override tokenizer's default
        messages += [{"role": "system", "content": ""}]
        messages += [{"role": "user", "content": user_prompt}]
        prompt = self.llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        print(f"[DEBUG] Prompt:\n{prompt}\n[DEBUG ENDDDDDD]")
        # Tokenize prompt
        tokenized = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
        ).to("cuda")
        input_ids = tokenized["input_ids"]
        input_len = input_ids.shape[-1]
        # Build inputs_embeds by concatenating projected latents (if provided) before text embeddings
        input_embeds = self.llm_model.get_input_embeddings()(input_ids)
        attention_mask = torch.ones((input_embeds.shape[0], input_embeds.shape[1]), dtype=torch.long, device=input_embeds.device)
        
        # Track the total input length including latents
        total_input_len = input_len
        
        # Debug prints to check why condition is always false
        print(f"[DEBUG] Condition check - use_latents: {use_latents}")
        print(f"[DEBUG] Condition check - latents is not None: {latents is not None}")
        print(f"[DEBUG] Condition check - self.latents_projector is not None: {self.latents_projector is not None}")
        print(f"[DEBUG] Condition check - latents type: {type(latents)}")
        if latents is not None:
            print(f"[DEBUG] Condition check - latents shape: {getattr(latents, 'shape', 'No shape attribute')}")
        if self.latents_projector is not None:
            print(f"[DEBUG] Condition check - projector type: {type(self.latents_projector)}")
            
            
        
        if use_latents and latents is not None and self.latents_projector is not None:
            print(f"[DEBUG] Using latents! Shape: {latents.shape}, device: {latents.device}, dtype: {latents.dtype}")
            print(f"[DEBUG] Latents stats - min: {latents.min().item():.4f}, max: {latents.max().item():.4f}, mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}")
            print(f"[DEBUG] Text embeddings device: {input_embeds.device}, dtype: {input_embeds.dtype}")
            print(f"[DEBUG] Text embeddings stats - min: {input_embeds.min().item():.4f}, max: {input_embeds.max().item():.4f}, mean: {input_embeds.mean().item():.4f}, std: {input_embeds.std().item():.4f}")
            print(f"[DEBUG] Projector device: {next(self.latents_projector.parameters()).device}")
            
            # Ensure latents are on the same device and dtype as the projector
            latents_device = next(self.latents_projector.parameters()).device
            latents = latents.to(latents_device).to(torch.bfloat16)
            
            # Project latents
            latents_proj = self.latents_projector(latents)
            print(f"[DEBUG] Projected latents shape: {latents_proj.shape}, device: {latents_proj.device}, dtype: {latents_proj.dtype}")
            print(f"[DEBUG] Projected latents stats - min: {latents_proj.min().item():.4f}, max: {latents_proj.max().item():.4f}, mean: {latents_proj.mean().item():.4f}, std: {latents_proj.std().item():.4f}")
            
            # Move projected latents to match text embeddings device
            latents_proj = latents_proj.to(input_embeds.device).to(input_embeds.dtype)
            print(f"[DEBUG] After moving - Projected latents device: {latents_proj.device}, dtype: {latents_proj.dtype}")
            
            # Check if scales are similar
            text_scale = input_embeds.std().item()
            latents_scale = latents_proj.std().item()
            print(f"[DEBUG] Scale comparison - Text std: {text_scale:.4f}, Latents std: {latents_scale:.4f}, Ratio: {latents_scale/text_scale:.2f}")
            
            # Concatenate
            print(f"[DEBUG] Before concat - input_embeds shape: {input_embeds.shape}")
            input_embeds = torch.cat([latents_proj, input_embeds], dim=1)
            print(f"[DEBUG] After concat - input_embeds shape: {input_embeds.shape}")
            lat_mask = torch.ones((attention_mask.shape[0], latents_proj.shape[1]), dtype=torch.long, device=attention_mask.device)
            attention_mask = torch.cat([lat_mask, attention_mask], dim=1)
            # Update total input length to include latents
            total_input_len = input_embeds.shape[1]
        else:
            print(f"[DEBUG] No latents used! latents is None: {latents is None}, projector is None: {self.latents_projector is None}")
            raise RuntimeError("Cannot generate answer without latents. Please provide valid latents from generate_plan().")
        
        print(f"[DEBUG] Input embeddings device: {input_embeds.device}, dtype: {input_embeds.dtype}")
        print(f"[DEBUG] Attention mask device: {attention_mask.device}, dtype: {attention_mask.dtype}")
        
        with torch.no_grad():
            # When not using latents, use input_ids directly for better generation
            if not use_latents or latents is None or self.latents_projector is None:
                print(f"[DEBUG] Generating WITHOUT latents using input_ids")
                outputs = self.llm_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else 1.0,
                    do_sample=self.do_sample,
                    repetition_penalty=1.2,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                )
            else:
                # When using latents with inputs_embeds
                print(f"[DEBUG] Generating WITH latents using inputs_embeds")
                outputs = self.llm_model.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.do_sample else 1.0,
                    do_sample=self.do_sample,
                    repetition_penalty=1.2,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    output_scores=False,
                    return_dict_in_generate=False,
                )
        print(f"[DEBUG] Generated output shape: {outputs.shape}")
        print(f"[DEBUG] Output length: {len(outputs[0])}")
        print(f"[DEBUG] Expected output length: ~{input_len + self.max_new_tokens} (prompt + new tokens)")
        print(f"[DEBUG] Full output IDs (last 30): {outputs[0][-30:]}")
        
        # When using inputs_embeds with max_new_tokens, we skip the original prompt length
        # (not total_input_len which includes latents, as latents don't produce token IDs)
        generated_token_ids = outputs[0][input_len:]
        print(f"[DEBUG] Generated token IDs ({len(generated_token_ids)} tokens): {generated_token_ids}")
        generated_text = self.llm_tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        print(f"[DEBUG] Generated text: '{generated_text}'")

        return {
            "text": generated_text.strip(),
            "metadata": {
                "llm_id": self.llm_id,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "do_sample": self.do_sample
            }
        }
        
        
        
    def execute_plan_WITH_noise(self, plan: Dict, prompt: str, **kwargs):
        # Corrupt the latents with random noise
        latents = plan["latents"]
        noisy_latents = latents + torch.randn_like(latents) * 10.0  # Large noise
        
        plan_corrupted = plan.copy()
        plan_corrupted["latents"] = noisy_latents
        
        return self.execute_plan(plan_corrupted, prompt, **kwargs)
#there's no projector (none) -> directly use llm finetuned