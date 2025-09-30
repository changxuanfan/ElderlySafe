import os
import json
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm  # Import tqdm for progress bar

# Model ID for Llama Guard 4
model_id = "meta-llama/Llama-Guard-4-12B"

# Load processor and model
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
    generated_text = processor.batch_decode(
        outputs[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )[0].lstrip()

    if generated_text.startswith("unsafe"):
        parts = generated_text.split("\n")
        categories = [cat.strip().lower() for cat in parts[1:] if cat.strip()]
        category_str = ",".join(categories) if categories else ""
        return "unsafe", category_str
    elif generated_text.startswith("safe"):
        return "safe", None
    else:
        return generated_text, None


# Input and output folders
input_folder = "eval_results_test100"
output_folder = "safeguard_results_llama"
os.makedirs(output_folder, exist_ok=True)

# Process each JSON file
for filename in tqdm(sorted(os.listdir(input_folder))):
    if filename.endswith(".json"):
        input_path = os.path.join(input_folder, filename)
        with open(input_path, "r") as f:
            data = json.load(f)

        for turn in data["dialogue"]:
            prompt = turn.get("prompt", "")

            # Evaluate prompt safety
            prompt_chat = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            prompt_safety, prompt_category = evaluate_safety(prompt_chat)

            # Collect results in llama_guard dict
            turn["llama_guard"] = {
                "prompt": {"safety": prompt_safety, "category": prompt_category}
            }

            # Dynamically check all keys ending with "_response"
            for key, response in turn.items():
                if key.endswith("_response"):
                    if response:
                        chat = [
                            {"role": "user", "content": [{"type": "text", "text": prompt}]},
                            {"role": "assistant", "content": [{"type": "text", "text": response}]},
                        ]
                        safety, category = evaluate_safety(chat)
                    else:
                        safety, category = "empty", None

                    turn["llama_guard"][key] = {"safety": safety, "category": category}

        # Save the updated JSON
        output_path = os.path.join(output_folder, filename)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
            print("Saved results to", output_path)
