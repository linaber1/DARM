from typing import Dict, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import openai

class LLM:
    def __init__(self, model_id: str, device: str = "cuda:0", max_new_tokens: int = 512,
                 temperature: float = 0, do_sample: bool = False, **kwargs):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager").eval()
        

    def generate(self, user_prompt: str, system_prompt=None) -> Dict:
        messages = []
        if system_prompt: 
            messages += [{"role": "system", "content": system_prompt}]
        messages += [{"role": "user", "content": user_prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        return {
            "text": generated_text.strip(),
            "metadata": {
                "model_id": self.model_id,
                "device": self.device,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "do_sample": self.do_sample
            }
        }
        
        
class LLM_GPT:
    def __init__(self, model_id: str, max_new_tokens: int = 512,
                 temperature: float = 0.2, do_sample: bool = False, **kwargs):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

    def generate(self, system_prompt: Optional[str], user_prompt: str) -> Dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = openai.ChatCompletion.create(
            model=self.model_id,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        generated_text = response.choices[0].message.content.strip()

        return {
            "text": generated_text,
            "metadata": {
                "model_id": self.model_id,
                "device": self.device,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "do_sample": self.do_sample
            }
        }
