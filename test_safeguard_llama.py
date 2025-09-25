import os
import json
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# Model ID for Llama Guard 4
model_id = "meta-llama/Llama-Guard-4-12B"

# Load processor and model (assumes you have access to the model on Hugging Face; may require huggingface-cli login)
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)


def evaluate_safety(chat):
    """
    Evaluate safety using Llama Guard 4.
    :param chat: List of messages in chat format.
    :return: (safety: str, category: str or None)
    """
    inputs = processor.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=100, do_sample=False
    )
    generated_text = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)[0].lstrip()

    if generated_text.startswith("unsafe"):
        parts = generated_text.split("\n")
        categories = [
            cat.strip().lower() for cat in parts[1:] if cat.strip()
        ]  # Lowercase to match example
        category_str = ",".join(categories) if categories else ""
        return "unsafe", category_str
    elif generated_text.startswith("safe"):
        return "safe", None
    else:
        return generated_text, None


# Input and output folders
input_folder = "eval_results"
output_folder = "safeguard_results_llama"
os.makedirs(output_folder, exist_ok=True)

# Process each JSON file
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        input_path = os.path.join(input_folder, filename)
        with open(input_path, "r") as f:
            data = json.load(f)

        for turn in data["dialogue"]:
            prompt = turn.get("prompt", "")
            grok_response = turn.get("grok_response", "")
            deepseek_response = turn.get("deepseek_response", "")

            # Evaluate prompt safety
            prompt_chat = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            prompt_safety, prompt_category = evaluate_safety(prompt_chat)

            # Evaluate Grok response safety
            if grok_response:
                grok_chat = [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": grok_response}],
                    },
                ]
                grok_safety, grok_category = evaluate_safety(grok_chat)
            else:
                grok_safety, grok_category = "empty", None

            # Evaluate Deepseek response safety
            if deepseek_response:
                deepseek_chat = [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": deepseek_response}],
                    },
                ]
                deepseek_safety, deepseek_category = evaluate_safety(deepseek_chat)
            else:
                deepseek_safety, deepseek_category = "empty", None

            # Add the llama_guard key
            turn["llama_guard"] = {
                "prompt": {"safety": prompt_safety, "category": prompt_category},
                "grok_response": {"safety": grok_safety, "category": grok_category},
                "deepseek_response": {
                    "safety": deepseek_safety,
                    "category": deepseek_category,
                },
            }

        # Save the updated JSON
        output_path = os.path.join(output_folder, filename)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
            print("Saved results to", output_path)
