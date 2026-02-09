import evaluate as hf_evaluate
import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
compute_ = hf_evaluate.load("code_eval", cache_dir="/l/users/abdulrahman.mahmoud")





def prepare_humaneval_sample(item: dict) -> dict: # "input" and "setup"
    """
    Prepare a HumanEval sample for evaluation.
    
    Args:
        item: Raw sample from the HumanEval dataset containing:
            - task_id: identifier for the data sample
            - prompt: input for the model containing function header and docstrings
            - canonical_solution: solution for the problem in the prompt
            - test: contains function to test generated code for correctness
            - entry_point: entry point for test
    
    Returns:
        dict: Processed sample with 'input' and 'correct' fields
    """
    # The input is the prompt (function signature + docstring)
    user_prompt = item["prompt"]
    
    # For HumanEval, the "setup" includes the full test setup
    # We need the prompt + canonical solution + test cases for evaluation
    setup = {
        "task_id": item["task_id"],
        "canonical_solution": item["canonical_solution"],
        "test": item["test"],
        "entry_point": item["entry_point"],
        "prompt": item["prompt"]
    }
    
    return {
        "input": user_prompt,
        "setup": setup
    }
    

    
def run_humaneval(predicted: str, setup: dict) -> float:
    try:
        full_code = extract_code_between_backticks(predicted)
        test_cases = [setup["test"]]  # Ensure it's a list
        predictions = [[full_code]]
        results = compute_.compute(
            references=test_cases,
            predictions=predictions,
            k=[1]
        )
        return float(results[0]["pass@1"] > 0)
    except Exception as e:
        print(f"Error in code evaluation: {e}")
        import traceback
        traceback.print_exc()  # Add this for debugging
        return 0.0
    

def extract_code_between_backticks(s):
    # First try to find code blocks
    first = s.find("```")
    if first != -1:
        start = first + 3
        if s[start:start+6].lower() == 'python':
            start += 6
        if start < len(s) and s[start] == '\n':
            start += 1
        second = s.find("```", start)
        if second != -1:
            return s[start:second].strip()
        else:
            return s[start:].strip()
    
    # If no code blocks, look for function definitions
    lines = s.split('\n')
    code_lines = []
    in_function = False
    
    for line in lines:
        if line.strip().startswith('def ') or line.strip().startswith('return '):
            in_function = True
        if in_function:
            code_lines.append(line)
        # Stop if we hit explanatory text after code
        if in_function and line.strip() and not line.startswith(' ') and not line.strip().startswith('def') and not line.strip().startswith('return') and not line.strip().startswith('#'):
            break
    
    if code_lines:
        return '\n'.join(code_lines).strip()
    
    return s.strip()