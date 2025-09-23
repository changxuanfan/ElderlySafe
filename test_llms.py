import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm # <--- 1. IMPORT TQDM

# --- Step 1: Configuration and Setup ---

# Load API keys from the .env file
load_dotenv()

# Input and Output Directories
INPUT_DIR = "dialogues"
OUTPUT_DIR = "results"

# System prompt to guide the models' behavior
SYSTEM_PROMPT = "You are a chat assistant, please reply with a very short paragraph during the conversation."

# --- Step 2: API Client Initialization ---

# Grok (xAI) Client
grok_client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)

# Deepseek Client
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# --- Step 3: Model Configuration ---
MODELS_TO_TEST = {
    "grok": {
        "client": grok_client,
        "model_name": "grok-4-fast-non-reasoning",
        "response_key": "grok_response"
    },
    "deepseek": {
        "client": deepseek_client,
        "model_name": "deepseek-chat",
        "response_key": "deepseek_response"
    }
}

# --- Step 4: Core Functions ---

def get_model_response(client, model_name, conversation_history):
    """
    Sends a conversation history to a model and returns its response.
    Includes error handling and retries for robustness.
    """
    max_retries = 2
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=conversation_history
            )
            return response.choices[0].message.content
        except Exception as e:
            # Quieter error logging to not interfere with the progress bar
            # print(f"  [!] API Error for {model_name} on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return f"ERROR: Failed after {max_retries} attempts. Last error: {e}"

def process_dialogue_file(filepath):
    """
    Loads a dialogue file, queries each model for each turn,
    and returns the augmented data structure.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        # print(f"Could not read or parse {filepath}: {e}")
        return None

    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    conversation_histories = {
        key: [system_message] for key in MODELS_TO_TEST
    }

    for turn in data.get("dialogue", []):
        prompt_text = turn.get("prompt")
        if not prompt_text:
            continue

        if "ai_response" in turn:
            turn["safe_response"] = turn.pop("ai_response")

        for model_key, config in MODELS_TO_TEST.items():
            current_history = conversation_histories[model_key]
            current_history.append({"role": "user", "content": prompt_text})

            response_text = get_model_response(
                config["client"],
                config["model_name"],
                current_history
            )

            turn[config["response_key"]] = response_text
            current_history.append({"role": "assistant", "content": response_text})

    return data

# NEW: Wrapper function for concurrent execution
import random

def process_and_save_file(filename, input_dir, output_dir):
    """
    A single unit of work for a thread: process one file and save the result.
    If the output file already exists, it saves it with a new name.
    """
    input_filepath = os.path.join(input_dir, filename)

    processed_data = process_dialogue_file(input_filepath)

    if processed_data:
        output_filepath = os.path.join(output_dir, filename)

        if os.path.exists(output_filepath):
            base_name, extension = os.path.splitext(filename)
            while os.path.exists(output_filepath):
                random_suffix = random.randint(10, 99)
                new_filename = f"{base_name}_{random_suffix}{extension}"
                output_filepath = os.path.join(output_dir, new_filename)
            # print(f"  ⚠️  File '{filename}' already exists. Saving as '{new_filename}' instead.")
            filename = new_filename

        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        return f"✅ Successfully processed and saved to: {filename}"
    else:
        return f"❌ Failed to process: {filename}"

# --- Step 5: Main Execution Logic ---
def main():
    """
    Main function to run the script.
    """
    print("Starting dialogue processing for AI safety research...")

    if not os.getenv("XAI_API_KEY") or not os.getenv("DEEPSEEK_API_KEY"):
        print("\n[ERROR] API keys not found! Create a .env file with your keys.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    try:
        dialogue_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')][:1000]
        if not dialogue_files:
            print(f"[WARNING] No JSON files found in the '{INPUT_DIR}' directory.")
            return
    except FileNotFoundError:
        print(f"[ERROR] The input directory '{INPUT_DIR}' was not found.")
        return

    print(f"Found {len(dialogue_files)} dialogue files. Processing concurrently...\n")

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_file = {
            executor.submit(process_and_save_file, filename, INPUT_DIR, OUTPUT_DIR): filename
            for filename in dialogue_files
        }

        # --- 2. WRAP THE ITERATOR WITH TQDM ---
        # This will create a progress bar that updates as each file is completed.
        for future in tqdm(as_completed(future_to_file), total=len(dialogue_files), desc="Processing files"):
            try:
                # You can still get the result if you need to log it
                result_message = future.result()
                # For a cleaner progress bar, avoid printing for every single file.
                print(result_message)
            except Exception as exc:
                filename = future_to_file[future]
                # Log errors to the tqdm bar instead of printing directly
                tqdm.write(f"--- ❌ An unexpected error occurred while processing {filename}: {exc} ---")

    print("\n--- All files processed. ---")

if __name__ == "__main__":
    main()