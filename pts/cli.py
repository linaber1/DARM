import argparse
import sys
from datasets import load_dataset
from .pipeline.orchestrator import PTSPipeline


PLAN_PROMPT = """Create a concise plan for an essay on the following topic:  
"{topic}"

The plan should be brief and structured, suitable for an essay of 200–250 words.  
Include:  
- Introduction (with clear thesis statement)  
- 2–3 body paragraphs (each with one main argument and short explanation)  
- Conclusion (summarizing position)  

Write the plan in bullet points, no long sentences."""

ESSAY_PROMPT = """
Write a well-structured essay of about 200-250 words on the following topic:
"{topic}"

Follow this exact plan when writing the essay:  
{plan}  


Requirements:  
- Formal academic tone  
- Clear introduction, body, and conclusion  
- Logical flow with transitions between paragraphs 
- Stay within 200–250 words
"""

def main():
    ap = argparse.ArgumentParser(description="LLada → Llama pipeline")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--no-refine", action="store_true", help="Skip the stage-1 Llama refinement")
    ap.add_argument("--number_samples", type=int, default=10, help="Number of samples to generate")
    ap.add_argument("--name_architecture", type=str, default="diffusion-llm", help="Name of the architecture to use")
    args = ap.parse_args()

    pipe = PTSPipeline.from_yaml(args.config)
    
    ds = load_dataset("chillies/IELTS-writing-task-2-evaluation")
    unique_prompts = list(set([d["prompt"] for d in ds["train"]]))

    import os
    import json
    from datetime import datetime

    results = []
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs_essay/{args.name_architecture}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"essay_evaluation_{time_stamp}.json")

    for topic in unique_prompts[:args.number_samples]:
        plan_prompt = PLAN_PROMPT.format(topic=topic)
        essay_prompt = ESSAY_PROMPT.format(topic=topic, plan="{plan}")

        plan = pipe.generate_plan(user_prompt=plan_prompt, name_architecture=args.name_architecture)
        answer = pipe.generate_answer(user_prompt=essay_prompt.format(plan=plan['text']), name_architecture=args.name_architecture)
        result = {
            "topic": topic,
            "plan": plan["text"],
            "name architecture": args.name_architecture,
            "plan_metadata": plan.get("metadata"),
            "essay": answer["text"],
            "essay_metadata": answer.get("metadata"),
        }
        results.append(result)
        print(f"Topic: {topic}\nPlan: {plan['text']}\nEssay: {answer['text']}\n{'-'*40}")

        # Save results after each topic
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    sys.exit(main())
