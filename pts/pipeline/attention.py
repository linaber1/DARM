

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
import os
from scipy import stats
from scipy import stats as scipy_stats
from datetime import datetime

# Helper function to analyze results across all workers
def analyze_attention_results(all_results, architecture, model_diffusion, model_llm, benchmark_name):
    """
    Analyze attention patterns across all workers' results
    
    Args:
        all_results: Combined results from all workers (list of result dicts)
    """
    # Extract samples with attention metrics
    attention_samples = [r for r in all_results if 'attention_metrics' in r and r['attention_metrics']]
    
    if not attention_samples:
        print("No attention metrics found in results")
        return
    
    # Extract metrics
    plan_attention_ratios = [r['attention_metrics']['plan_attention_ratio'] for r in attention_samples]
    question_attention_ratios = [r['attention_metrics']['question_attention_ratio'] for r in attention_samples]
    plan_to_question_ratios = [r['attention_metrics']['plan_to_question_ratio'] for r in attention_samples] #if less than 1 then more attention to question
    attention_entropies = [r['attention_metrics']['attention_entropy'] for r in attention_samples]
    
    # Analyze correlation with correctness
    correct_samples = [r for r in attention_samples if r['is_correct']]
    incorrect_samples = [r for r in attention_samples if not r['is_correct']]

    # Statistical significance test
    t_stat, p_value = scipy_stats.ttest_ind(
        [r['attention_metrics']['plan_attention_ratio'] for r in correct_samples] if correct_samples else [0],
        [r['attention_metrics']['plan_attention_ratio'] for r in incorrect_samples] if incorrect_samples else [0]
    )

    # Distribution analysis
    high_plan_attention = sum(1 for ratio in plan_attention_ratios if ratio > 0.3)
    plan_focused = sum(1 for ratio in plan_to_question_ratios if ratio > 1.0)

    # Prepare correctness analysis
    correctness_analysis = None
    if correct_samples and incorrect_samples:
        correct_plan_attention = float(np.mean([r['attention_metrics']['plan_attention_ratio'] for r in correct_samples]))
        incorrect_plan_attention = float(np.mean([r['attention_metrics']['plan_attention_ratio'] for r in incorrect_samples]))
        diff = correct_plan_attention - incorrect_plan_attention
        correct_attentions = [r['attention_metrics']['plan_attention_ratio'] for r in correct_samples]
        incorrect_attentions = [r['attention_metrics']['plan_attention_ratio'] for r in incorrect_samples]
        t_stat_corr, p_value_corr = scipy_stats.ttest_ind(correct_attentions, incorrect_attentions)
        correctness_analysis = {
            "correct_plan_attention_mean": correct_plan_attention,
            "incorrect_plan_attention_mean": incorrect_plan_attention,
            "difference": diff,
            "t_stat": float(t_stat_corr) if t_stat_corr is not None else None,
            "p_value": float(p_value_corr) if p_value_corr is not None else None,
            "significant": bool(p_value_corr < 0.05) if p_value_corr is not None else None,
            "num_correct": len(correct_samples),
            "num_incorrect": len(incorrect_samples)
        }
    stats = {
        "architecture": architecture,
        "model_diffusion": model_diffusion,
        "model_llm": model_llm,
        "num_samples": len(attention_samples),
        "plan_attention_ratio": {
            "mean": float(np.mean(plan_attention_ratios)),
            "std": float(np.std(plan_attention_ratios))
        },
        "question_attention_ratio": {
            "mean": float(np.mean(question_attention_ratios)),
            "std": float(np.std(question_attention_ratios))
        },
        "plan_to_question_ratio": {
            "mean": float(np.mean(plan_to_question_ratios)),
            "std": float(np.std(plan_to_question_ratios))
        },
        "attention_entropy": {
            "mean": float(np.mean(attention_entropies)),
            "std": float(np.std(attention_entropies))
        },
        "t_stat": float(t_stat) if t_stat is not None else None,
        "p_value": float(p_value) if p_value is not None else None,
        "high_plan_attention_count": high_plan_attention,
        "high_plan_attention_percent": 100 * high_plan_attention / len(attention_samples) if attention_samples else 0.0,
        "plan_focused_count": plan_focused,
        "plan_focused_percent": 100 * plan_focused / len(attention_samples) if attention_samples else 0.0,
        "correctness_analysis": correctness_analysis
    }

    
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), f'../../outputs/attention')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{benchmark_name}_attention_{time_stamp}.json')
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)





    # Attention vs. correctness analysis
    if correct_samples and incorrect_samples:
        correct_plan_attention = np.mean([r['attention_metrics']['plan_attention_ratio'] for r in correct_samples])
        incorrect_plan_attention = np.mean([r['attention_metrics']['plan_attention_ratio'] for r in incorrect_samples])
        
        
        # Statistical significance test
        correct_attentions = [r['attention_metrics']['plan_attention_ratio'] for r in correct_samples]
        incorrect_attentions = [r['attention_metrics']['plan_attention_ratio'] for r in incorrect_samples]
        
    
        t_stat, p_value = scipy_stats.ttest_ind(correct_attentions, incorrect_attentions)
        print(f"T-test p-value: {p_value:.6f} {'(significant)' if p_value < 0.05 else '(not significant)'}")
    
    # Distribution analysis
    high_plan_attention = sum(1 for ratio in plan_attention_ratios if ratio > 0.3)
    print(f"\nSamples with high plan attention (>30%): {high_plan_attention}/{len(attention_samples)} ({100*high_plan_attention/len(attention_samples):.1f}%)")
    
    plan_focused = sum(1 for ratio in plan_to_question_ratios if ratio > 1.0)
    print(f"Samples where plan gets more attention than question: {plan_focused}/{len(attention_samples)} ({100*plan_focused/len(attention_samples):.1f}%)")


    save_mean_output_token_attention("correct", correct_samples["attention_metrics"]["attention"], benchmark_name, correct_samples["attention_metrics"]["plan_tokens"], correct_samples["attention_metrics"]["question_tokens"], correct_samples["attention_metrics"]["output_tokens"])
    save_mean_output_token_attention("incorrect", incorrect_samples["attention_metrics"]["attention"], benchmark_name, incorrect_samples["attention_metrics"]["plan_tokens"], incorrect_samples["attention_metrics"]["question_tokens"], incorrect_samples["attention_metrics"]["output_tokens"])



def analyze_plan_attention(model, tokenizer, question: str, plan: str, output:str, template: str = None) -> Dict:
    """
    Function to analyze how much attention the LLM pays to the plan vs question.
    
    Args:
        model: The loaded LLM model (should have output_attentions=True capability)
        tokenizer: The tokenizer for the model
        question: The original question
        plan: The diffusion-generated plan
        template: Template for formatting input (should match your LLM_TEMPLATE)
    
    Returns:
        Dictionary with attention metrics
    """
    if template is None:
        template = "Question: {question}\nPlan: {plan}\nAnswer:"
    
    # Format input (use your existing template)
    formatted_input = template.format(question=question, plan=plan)
    input_output = formatted_input + "\n" + output
    
    # Tokenize
    sequence = tokenizer(input_output, return_tensors="pt", return_offsets_mapping=True)
    sequence_ids = sequence['input_ids'].to(model.device)
    offsets = sequence['offset_mapping'][0] if 'offset_mapping' in sequence else None
    offsets= offsets[2:] #attention sinks on special tokens so we remove them


    # Identify plan and question token positions
    plan_tokens, question_tokens, output_tokens = _identify_token_regions(formatted_input, question, plan, output, offsets)
    
    # Extract attention weights
    with torch.no_grad():
        # Set model to output attentions 
        original_output_attentions = getattr(model.config, 'output_attentions', False)
        model.config.output_attentions = True
        
        try:
            outputs = model(sequence_ids, output_attentions=True)
            attentions = outputs.attentions
            print("AAAAAAAAAAAAAAAAAAAAAAATTTTTTTTTTTTTTTTTEEEEEEEEEEEEEEENNNNNNNNNNNNTTTTTTTTTTTTTTTTTTTIIIIIIIIIIIIIOOOOOOOOOOONNNNNN")
            print(attentions  [0] )
        finally:
            # Restore original setting
            model.config.output_attentions = original_output_attentions
    
    # Compute attention metrics
    #metrics = _compute_attention_metrics(attentions, plan_tokens, question_tokens, output_tokens)
    metrics = {}
    metrics["plan_tokens"] = plan_tokens
    metrics["question_tokens"] = question_tokens
    metrics["output_tokens"] = output_tokens
    metrics["attentions"] = attentions  # Store raw attentions for further analysis if needed
    
    return metrics


def _identify_token_regions(formatted_input: str, question: str, plan: str, output: str, offsets) -> Tuple[List[int], List[int], List[int]]:
    """Identify which tokens belong to plan, question, and output regions."""
    plan_tokens = []
    question_tokens = []
    output_tokens = []

    question_start = formatted_input.find(question)
    question_end = question_start + len(question) if question_start != -1 else 0
    plan_start = formatted_input.find(plan)
    plan_end = plan_start + len(plan) if plan_start != -1 else 0
    output_start = formatted_input.find(output)
    output_end = output_start + len(output) if output_start != -1 else 0

    if offsets is None:
        # Fallback: rough estimation based on string positions
        total_length = len(formatted_input)
        seq_len = len(formatted_input.split())  # Very rough token count

        if question_start != -1:
            question_ratio_start = question_start / total_length
            question_ratio_end = question_end / total_length
            question_tokens = list(range(int(question_ratio_start * seq_len), int(question_ratio_end * seq_len)))
        if plan_start != -1:
            plan_ratio_start = plan_start / total_length
            plan_ratio_end = plan_end / total_length
            plan_tokens = list(range(int(plan_ratio_start * seq_len), int(plan_ratio_end * seq_len)))
        if output_start != -1:
            output_ratio_start = output_start / total_length
            output_ratio_end = output_end / total_length
            output_tokens = list(range(int(output_ratio_start * seq_len), int(output_ratio_end * seq_len)))
    else:
        for i, (start_char, end_char) in enumerate(offsets):
            if start_char is None or end_char is None:
                continue

            # Check if token overlaps with question
            if question_start != -1 and start_char < question_end and end_char > question_start:
                question_tokens.append(i)
            # Check if token overlaps with plan
            if plan_start != -1 and start_char < plan_end and end_char > plan_start:
                plan_tokens.append(i)
            # Check if token overlaps with output/answer
            if output_start != -1 and start_char >= output_start and end_char <= output_end:    
                output_tokens.append(i)

    return plan_tokens, question_tokens, output_tokens


def _compute_attention_metrics(attentions: List[torch.Tensor], 
                              plan_tokens: List[int], 
                              question_tokens: List[int],
                              output_tokens: List[int]) -> Dict:
    return {}
    # """Compute attention metrics from attention tensors."""
    
    # if not attentions or len(plan_tokens) == 0:
    #     return {
    #         'plan_attention_ratio': 0.0,
    #         'question_attention_ratio': 0.0,
    #         'plan_to_question_ratio': 0.0,
    #         'attention_entropy': 0.0,
    #         'layer_wise_plan_attention': []
    #     }
    
    # num_layers = len(attentions)
    # seq_len = attentions[0].shape[-1]
    
    # # Aggregate attention across layers and heads
    # total_attention = torch.zeros(seq_len, seq_len, device=attentions[0].device)
    # layer_wise_plan_attention = []
    
    # for attention_layer in attentions:
    #     # attention_layer shape: [batch=1, num_heads, seq_len, seq_len]
    #     layer_avg = attention_layer.squeeze(0).mean(dim=0)  # Average across heads
    #     total_attention += layer_avg
        
    #     # Calculate plan attention for this layer
    #     if plan_tokens:
    #         layer_plan_attention = layer_avg[:, plan_tokens].sum().item()
    #         layer_wise_plan_attention.append(layer_plan_attention)
    #     else:
    #         layer_wise_plan_attention.append(0.0)
    
    # # Average across layers
    # avg_attention = total_attention / num_layers
    
    # # Calculate attention sums
    # total_attention_sum = avg_attention.sum().item()
    # plan_attention_sum = avg_attention[:, plan_tokens].sum().item() if plan_tokens else 0.0
    # question_attention_sum = avg_attention[:, question_tokens].sum().item() if question_tokens else 0.0
    
    # # Calculate ratios
    # plan_attention_ratio = plan_attention_sum / total_attention_sum if total_attention_sum > 0 else 0.0
    # question_attention_ratio = question_attention_sum / total_attention_sum if total_attention_sum > 0 else 0.0
    

    # #ALSSO !!! Normalize by the number of tokens in each region to get per-token attention because nb of tokens can vary (for plan and question)
    # if plan_tokens:
    #     plan_attention_ratio /= len(plan_tokens)
    # if question_tokens:
    #     question_attention_ratio /= len(question_tokens)

    # plan_to_question_ratio = plan_attention_sum / question_attention_sum if question_attention_sum > 0 else 0.0


    # # Calculate attention entropy
    # attention_probs = avg_attention.sum(dim=0)  # Sum over query positions
    # attention_probs = attention_probs / attention_probs.sum()  # Normalize
    # attention_entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-12)).item() #1e-12 to avoid log(0)
    
    # return {
    #     'plan_attention_ratio': plan_attention_ratio,
    #     'question_attention_ratio': question_attention_ratio,
    #     'plan_to_question_ratio': plan_to_question_ratio,
    #     'attention_entropy': attention_entropy,
    #     'layer_wise_plan_attention': layer_wise_plan_attention,
    #     'num_plan_tokens': len(plan_tokens),
    #     'num_question_tokens': len(question_tokens)
    # }



def save_mean_output_token_attention(namecorrectness: str,
                                    attention_samples: List[Dict], 
                                    benchmark_name: str, 
                                    output_tokens: List[int], 
                                    plan_tokens: List[int], 
                                    question_tokens: List[int]):
    """
    Saves a list of mean normalized attention values from output_tokens to each plan token (then each question token).
    Each element in the list corresponds to the normalized attention for that token (plan first, then question).
    """

    print("ooooooooooooooooooOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO00000000000000000000000")
    extracted_list = []
    for sample in attention_samples:
        attentions = sample['attentions']  # List[torch.Tensor]
        num_layers = len(attentions)
        seq_len = attentions[0].shape[-1]
        total_attention = torch.zeros(seq_len, seq_len, device=attentions[0].device) # shape: [seq_len, seq_len]


        for attention_layer in attentions:
            layer_avg = attention_layer.squeeze(0).mean(dim=0)
            total_attention += layer_avg
        avg_attention = total_attention / num_layers

        
        plot_columns = plan_tokens + question_tokens
        extracted = avg_attention[output_tokens][:, plot_columns]  # shape: [len(output_tokens), len(plot_columns)]
        extracted_list.append(extracted.cpu().numpy())

    # Average across samples
    extracted_array = np.stack(extracted_list)  # [num_samples, len(output_tokens), len(plot_columns)]
    mean_attention = extracted_array.mean(axis=0)  # [len(output_tokens), len(plot_columns)]

    # Normalize by total attention for each output token
    normalized_attention = mean_attention / (mean_attention.sum(axis=1, keepdims=True) + 1e-12)  # [len(output_tokens), len(plot_columns)]

    # Flatten to a single list: for each token in plot_columns, mean over output_tokens
    # This gives a list where each element is the mean normalized attention to that token (plan first, then question)
    token_attention_list = normalized_attention.mean(axis=0).tolist()  # [len(plot_columns)]

    output_attention_stats = {
    "plan_tokens": plan_tokens,
    "question_tokens": question_tokens,
    "token_attention_list": token_attention_list,  # plan tokens first, then question tokens
    "plan_attention_list": token_attention_list[:len(plan_tokens)],  # attention for plan tokens
    "question_attention_list": token_attention_list[len(plan_tokens):]  # attention for question tokens
    }
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), f'../../outputs/attention')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{namecorrectness}_{benchmark_name}_PLOTWIST_{time_stamp}.json')
    with open(output_path, 'w') as f:
        json.dump(output_attention_stats, f, indent=2)