from argparse import ArgumentParser
import sys
import inspect
from tqdm import tqdm
import re
import math
import json
from datetime import datetime
from functools import partial
import os

from datasets import load_dataset, load_from_disk
import torch
import torch.multiprocessing as mp
from mathruler.grader import extract_boxed_content, grade_answer

from pts.pipeline.orchestrator import PTSPipeline
from pts.pipeline.utils import read_yaml
from pts.constants import Pipelines
from pts.eval.compare_prepare.aime import compare_answers_aime, prepare_aime_sample
from sympy.parsing.latex import parse_latex


ARC_QUESTION_PROMPT_TEMPLATE = """Question: {question}\n{choices_text}"""
MCQ_QUESTION_POSTFIX = """\nAnswer with a single letter (A, B, C, or D) and no explanation. Your answer should start with "Answer: " and be followed by the letter of the answer you choose. Do not include any other text in your response."""

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

LLM_TEMPLATE = """You are an expert in solving multiple-choice questions.
Given the following plan or reasoning, please solve the question. If the plan contains any explicit answer or option letter, ignore it and solve from the hints + question only.
{question}"""


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


def compare_answers_mcq(predicted, correct):
    pred_answer = re.match(r"^(?:Answer:\s*)?([A-Da-d])\.?$", predicted.strip())
    if not pred_answer:
        return 0.0
    matched_group = pred_answer.group(1) or pred_answer.group(2)
    response = matched_group.strip()[0]
    return float(correct.lower().strip()[0] == response.lower())


def compare_answers_dart(pred, gt):
    if '\\boxed{' not in pred:
        pred = f"\\boxed{{{pred}}}"
    if '\\boxed{' not in gt:
        gt = f"\\boxed{{{gt}}}"
    pred_answer = extract_boxed_content(pred.strip())
    gt_answer = extract_boxed_content(gt.strip())
    try:
        x = parse_latex(pred_answer)
        y = parse_latex(gt_answer)
    except Exception:
        return float(grade_answer(pred_answer, gt_answer))
    return float(x.equals(y) or grade_answer(pred_answer, gt_answer))

# ------------------------------------------------------------------------------------


# Template used to prompt the model – GSM8K problems are just questions.
QSTANS_PROMPT_TEMPLATE = "Question: {question}\nAnswer:\n"
GSM8K_FEWSHOT_K = 2

def _format_gsm8k_example(q, a):
    return f"Question: {q}\nAnswer:\n{a}\n\n"
# FEW SHOTS: recommended : 5
def build_gsm8k_fewshot_prefix(train_ds, k=GSM8K_FEWSHOT_K):
    examples = train_ds.shuffle(seed=42).select(range(k))
    parts = []
    for ex in examples:
        parts.append(_format_gsm8k_example(ex["question"], ex["answer"]))
    return "".join(parts)



def prepare_gsm8k_sample(item: dict, fewshot_prefix: str = "") -> dict:
    question = item["question"]
    answer = item["answer"]
    input_text = fewshot_prefix + QSTANS_PROMPT_TEMPLATE.format(question=question)
    return {"input": input_text, "correct": answer}

def compare_answers_gsm8k(predicted: str, correct: str) -> float:
    def _extract_numeric_answer(text: str) -> str:
        match = re.search(r"####\s*([-+]?[0-9][\d,\.]*)", text)
        if match:
            answer = match.group(1)
        else:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
            answer = numbers[-1] if numbers else "" # take the last number found
        # cleaning
        answer = (
            answer.replace(",", "")  # remove commas
                .replace("$", "")  # remove dollar signs
                .strip()           # remove whitespace
        )
        # remove potential "#### " at the beginning 
        answer = re.sub(r"^####\s*", "", answer)
        # remove a trailing period
        answer = re.sub(r"\.$", "", answer)
        return answer
    
    pred_answer = _extract_numeric_answer(predicted)
    gold_answer = _extract_numeric_answer(correct)
    return 1.0 if pred_answer and (pred_answer == gold_answer) else 0.0



#---------------------------------------------------------------------
 
def prepare_mmlu_sample(item):
    question = item['question']
    choices = item['choices']
    prompt = f"{question.strip()}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
    answer_id = item['answer']
    answer = chr(ord('A')+answer_id)
    return {"input": prompt, "correct": answer}

# ------------------------------------------------------------------------------------
def worker_evaluate_dual(
    rank: int,
    samples,
    return_dict,
    configs,
    name_architecture,
    compare_func=compare_answers_mcq,
    postfix=MCQ_QUESTION_POSTFIX,
    stop_early: bool=False,
    percentage: int=100,
):
   
    torch.cuda.set_device(rank)
    
    pipeline = PTSPipeline.from_yaml(configs)
    print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    local_results = []
    
    for i, sample in enumerate(tqdm(samples, desc=f"[GPU {rank}] Answer")):
        
        user_prompt = sample["input"]
        llm_input = user_prompt + postfix
        diffusion_input = DIFFUSION_HTNTS_TEMPLATE.format(question=user_prompt)
        print("========================================")
        if "stop_early" in inspect.signature(pipeline.generate_dual).parameters:
            out = pipeline.generate_dual(
                llm_input,
                diffusion_input,
                with_latents=True,
                stop_early=stop_early,
                percentage = percentage,
            )
        else:
            out = pipeline.generate_dual(llm_input, diffusion_input, with_latents=True)
        
        print(f"[DEBUG] Generated answer: '{out['answer']}'")
        print(f"[DEBUG] Ground truth: '{sample['correct']}'")
        is_correct = compare_func(out["answer"], sample["correct"])
        print(f"[DEBUG] Is correct: {is_correct}")
        print(out["answer"])
        local_results.append(
            {
                "is_correct": is_correct,
                "question": user_prompt,
                "predicted_answer": out["answer"],
                "predicted_plan": out["plan"],
                "ground_truth": sample["correct"],
                "latent plan" : out["plan_latents"] #added to see the latents
            }
        )
    return_dict[rank] = local_results





def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="arc_easy")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--ds_cache", type=str, default="/home/berrayan/cached_datasets")
    parser.add_argument("--name_architecture", choices=Pipelines.all_architectures())
    parser.add_argument("--stop_early", action="store_true", help="Whether to stop the diffusion process early.")
    parser.add_argument("--percentage", default=100, type=int, help="Percentage of the model steps.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(f"Loading dataset {args.dataset}")
    cache_path = os.path.join(args.ds_cache, args.dataset)
    dataset = None
    if os.path.exists(cache_path):
        try:
            dataset = load_from_disk(cache_path)
        except (FileNotFoundError, Exception) as e:
            print(f"Failed to load cached dataset: {e}")
            dataset = None
    if args.dataset == "arc_easy":
        if not dataset:
            dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
            dataset.save_to_disk(cache_path)
        process_func = prepare_arc_sample
        compare_func = compare_answers_mcq
        prefix = MCQ_QUESTION_POSTFIX
    elif args.dataset == "arc_challenge":
        if not dataset:
            dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
            dataset.save_to_disk(cache_path)
        process_func = prepare_arc_sample
        compare_func = compare_answers_mcq
        prefix = MCQ_QUESTION_POSTFIX
    elif "dart" in args.dataset:
        if not dataset:
            dataset = load_dataset("hkust-nlp/dart-math-pool-math", split="train")
            dataset.save_to_disk(cache_path)
        level = int(args.dataset.split("-")[-1])
        dataset = dataset.filter(
            lambda x: x["query_metadata"]["level"] == level, num_proc=32
        )
        process_func = prepare_dart_sample
        compare_func = compare_answers_dart
        prefix = DART_QUESTION_PREFIX
    elif "gsm8k" in args.dataset:
        if not dataset:
            test_ds = load_dataset("gsm8k", "main", split="test")
            train_ds = load_dataset("gsm8k", "main", split="train")
            test_ds.save_to_disk(cache_path)  #cache the test data
        else:
            test_ds = dataset
            train_ds = load_dataset("gsm8k", "main", split="train")
            
        fewshot_prefix = build_gsm8k_fewshot_prefix(train_ds, k=GSM8K_FEWSHOT_K)
        process_func = partial(prepare_gsm8k_sample, fewshot_prefix=fewshot_prefix)
        compare_func = compare_answers_gsm8k  
        prefix = ""  # GSM8K does not need a postfix
        dataset = test_ds
    elif "mmlu" in args.dataset:
        if not dataset:
            dataset = load_dataset("cais/mmlu", "all", split="test")
            dataset.save_to_disk(cache_path)
        process_func = prepare_mmlu_sample
        compare_func = compare_answers_mcq
        prefix = MCQ_QUESTION_POSTFIX
    elif "aime2025" in args.dataset :
        if not dataset :
            dataset = load_dataset("yentinglin/aime_2025", split='train')
            dataset.save_to_disk(cache_path)
        process_func = prepare_aime_sample
        compare_func = compare_answers_aime
        prefix = " "
    elif "aime2024" in args.dataset :
        if not dataset :
            dataset = load_dataset("HuggingFaceH4/aime_2024", split='train')
            dataset.save_to_disk(cache_path)
        process_func = prepare_aime_sample
        compare_func = compare_answers_aime
        prefix = " "
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    dataset = dataset.shuffle(seed=42)
    dataset = (
        dataset.select(range(args.num_samples)) if args.num_samples > 0 else dataset
    )
    all_samples = [process_func(sample) for sample in dataset]

    world_size = torch.cuda.device_count()
    chunk_size = math.ceil(len(all_samples) / world_size)
    chunks = [
        all_samples[i : i + chunk_size] for i in range(0, len(all_samples), chunk_size)
    ]

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    worker_func = worker_evaluate_dual

    for rank in range(world_size):
        p = mp.Process(
            target=worker_func,
            args=(
                rank,
                chunks[rank],
                return_dict,
                args.config,
                args.name_architecture,
                compare_func,
                prefix,
                args.stop_early,
                args.percentage,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    all_results = []

    for rank in range(world_size):
        all_results.extend(return_dict[rank])

    acc = [result["is_correct"] for result in all_results]
    accuracy = sum(acc) * 100 / len(acc)
    yaml_config = read_yaml(args.config)
    all_results = {
        "latents" : "yes",
        "diffusion_model": yaml_config["diffusion"]["model_id"],
        "llm": yaml_config["llm"]["model_id"],
        "name_architecture": args.name_architecture,
        "dataset": args.dataset,
        "num_samples": len(all_samples),
        "accuracy": accuracy,
        "plan_template": DIFFUSION_HTNTS_TEMPLATE,
        "answer_template": LLM_TEMPLATE,
        "diffusion_max_new_tokens": yaml_config["diffusion"]["max_new_tokens"],
        "llm_max_new_tokens": yaml_config["llm"]["max_new_tokens"],
        "results": all_results,
    }
    print(f"Accuracy: {accuracy:.2f}")

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{yaml_config['runtime']['output_dir']}/{args.name_architecture}/{args.dataset}_evaluation_{time_stamp}_{args.name_architecture}_percentage{args.percentage}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)
    return 0


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)
    sys.exit(main())
