from transformers import AutoTokenizer

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import os
from datetime import datetime
from pts.pipeline.models.modeling_llada import LLaDAModelLM

dataset_names = ["arc_easy", "arc_challenge", "dart-1", "dart-2", "dart-3", "dart-4", "dart-5", "gsm8k"] 
ARC_QUESTION_PROMPT_TEMPLATE = """Question: {question}\n{choices_text}"""
ARC_QUESTION_POSTFIX = """\nAnswer with a single letter (A, B, C, or D) and no explanation. Your answer should start with "Answer: " and be followed by the letter of the answer you choose. Do not include any other text in your response."""

DART_QUESTION_PROMPT_TEMPLATE = """Question: {question}"""
DART_QUESTION_PREFIX = (
    """\nThe final answer MUST BE put in \\boxed{{}} and no explanation."""
)

DIFFUSION_HTNTS_TEMPLATE = """You are a careful problem-solving planner.

Task: Produce ONLY a short list of HINTS that help solve the question. 
Do NOT state or imply the final answer. Do NOT mention any option letter 
(A, B, C, or D). Do NOT quote any option text verbatim. 
If you find yourself about to reveal a specific option or an answer, 
replace it with “[HIDDEN]”.

Format:
- Key facts to recall (2–4 bullets)
- Reasoning steps or elimination rules (2–5 bullets)
- Useful equations or definitions (if relevant)
- Edge cases or common traps (optional)

Be concise (<=120 words). No “Answer:” line. No letters A–D. No option text.

Question (stem only):
{question}
"""
GSM8K_PROMPT_TEMPLATE = "{question}\n"

model_id = "GSAI-ML/LLaDA-8B-Instruct"
model = LLaDAModelLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16, 
).to("cuda").eval()



tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

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

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                y = model(x_)
                hidden_states = y.hidden_states
                logits = y.logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                y = model(x)
                hidden_states = y.hidden_states
                logits = y.logits
                

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

    return x, hidden_states

def generate_latents(prompt, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence'):
    m = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to("cuda").unsqueeze(0)
    out, latents = generate(
        model, 
        input_ids, 
        steps=steps,
        gen_length=gen_length, 
        block_length=block_length, 
        temperature=temperature, 
        cfg_scale=cfg_scale, 
        remasking=remasking,
    )
    input_length = input_ids.shape[1]
    text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    # latents are of shape [B, L, D], input_length are of shape [B, I]
    # I want the latents to have shape [B, L-I, D]
    latents = latents[:, input_length:, :]
    # since batch_size = 1, squeeze
    latents = latents.squeeze(0)
    return latents, text

def prepare_dart_sample(item):
    question = item["query"]
    answer_key = item["gt_ans"]
    input_text = DART_QUESTION_PROMPT_TEMPLATE.format(question=question)
    return {"input": input_text, "correct": answer_key}


def prepare_arc_sample(item):
    question = item["question"]
    choices = item["choices"]
    answer_key = item["answerKey"]
    choices_text = "\n".join(
        [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices["text"])]
    )
    input_text = ARC_QUESTION_PROMPT_TEMPLATE.format(
        question=question, choices_text=choices_text
    )
    labels = choices["label"]
    correct_idx = labels.index(answer_key)
    answer_key = chr(65 + correct_idx)  # Convert index to letter (A, B, C, D)
    return {"input": input_text, "correct": answer_key}

def prepare_gsm8k_sample(item: dict) -> dict:
    question = item["question"]
    answer = item["answer"]
    input_text = GSM8K_PROMPT_TEMPLATE.format(question=question)
    return {"input": input_text, "correct": answer}


def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--remasking", type=str, default="low_confidence")
    parser.add_argument("--dataset", type=str, choices=dataset_names, default="arc_easy")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--output_suffix", type=str, default="")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(f"Loading {args.dataset} ...")
    dataset = None
    ds_cache = os.path.join("/PATH/cached_datasets", args.dataset)
    if os.path.exists(ds_cache):
        dataset = load_from_disk(ds_cache)
    if args.dataset == "arc_easy":
        if dataset is None:
            dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        process_func = prepare_arc_sample
        prefix = ARC_QUESTION_POSTFIX
    elif args.dataset == "arc_challenge":
        if dataset is None:
            dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        process_func = prepare_arc_sample
        prefix = ARC_QUESTION_POSTFIX
    elif "dart" in args.dataset:
        if dataset is None:
            dataset = load_dataset("hkust-nlp/dart-math-pool-math", split="train")
        level = int(args.dataset.split("-")[-1])
        dataset = dataset.filter(
            lambda x: x["query_metadata"]["level"] == level, num_proc=32
        )
        process_func = prepare_dart_sample
        prefix = DART_QUESTION_PREFIX
    elif "gsm8k" in args.dataset:
        if dataset is None:
            dataset = load_dataset("gsm8k", "main", split="test")
        process_func = prepare_gsm8k_sample
        prefix = ""
    if args.num_samples > 0:
        start = max(args.start_index, 0)
        end = min(start + args.num_samples, len(dataset))
        dataset = dataset.select(range(start, end))
    
    # Prepare output directory and filename
    output_dir = "outputs/train_5x2"
    os.makedirs(output_dir, exist_ok=True)
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    base_output_path = f"{output_dir}/{args.dataset}_{time_stamp}{suffix}"
    
    # Check if there are existing partial files to resume from
    existing_files = sorted([f for f in os.listdir(output_dir) if f.startswith(f"{args.dataset}_{time_stamp}{suffix}_sample_")])
    start_idx = len(existing_files)
    
    if start_idx > 0:
        print(f"Found {start_idx} existing samples. Resuming from sample {start_idx}...")
    
    for idx, d in enumerate(tqdm(dataset, desc=f"Processing {args.dataset}"), start=start_idx):
        sample = process_func(d)
        prompt = sample["input"]
        hints_prompt = DIFFUSION_HTNTS_TEMPLATE.format(question=prompt)
        latents, plan = generate_latents(
            hints_prompt, 
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking
        )
        latents = latents.cpu().to(dtype=torch.float16).numpy()
        latent_sample = {
            "prompt": sample['input'] + prefix,
            "question": sample["input"],
            "latents": latents,
            "plan": plan,
            "postfix": prefix,
            "answer": sample["correct"],
            "dataset": args.dataset, 
            "model": model_id,
            "metadata": {
                "steps": args.steps,
                "gen_length": args.gen_length,
                "block_length": args.block_length,
                "temperature": args.temperature,
                "cfg_scale": args.cfg_scale,
                "remasking": args.remasking
            }
        }
        
        # Save each sample individually
        sample_path = f"{base_output_path}_sample_{idx:05d}.npy"
        np.save(sample_path, latent_sample)
    
    # After all samples are processed, merge them into a single file
    print(f"Merging all samples into final file...")
    all_samples = []
    for idx in range(len(dataset)):
        sample_path = f"{base_output_path}_sample_{idx:05d}.npy"
        if os.path.exists(sample_path):
            all_samples.append(np.load(sample_path, allow_pickle=True).item())
    
    final_output_path = f"{base_output_path}_latents.npy"
    np.save(final_output_path, all_samples)
    print(f"Saved final merged file to {final_output_path}")


if __name__ == "__main__":
    main()