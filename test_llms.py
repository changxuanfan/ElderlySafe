import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
# import anthropic
from google import genai
from google.genai import types

# --- Step 1: Configuration and Setup ---
load_dotenv()

INPUT_DIR = "dialogues"
OUTPUT_DIR = "model_responses"
SYSTEM_PROMPT = "You are a chat assistant, please reply with a very short paragraph during the conversation."
max_workers = 10  # Adjust based on your system and API rate limits

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
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

qwen_client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

# claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Use either GOOGLE_API_KEY or GEMINI_API_KEY if present
gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    gemini_client = genai.Client(api_key=gemini_api_key)
else:
    gemini_client = genai.Client()  # will read environment variables if present

# --- Step 3: Model Configuration ---
MODELS_TO_TEST = {
    "grok": {
        "client": grok_client,
        "model_name": "grok-4-fast",
        "response_key": "grok_response",
        "type": "openai"
    },
    "deepseek": {
        "client": deepseek_client,
        "model_name": "deepseek-chat",
        "response_key": "deepseek_response",
        "type": "openai"
    },
    "openai": {
        "client": openai_client,
        "model_name": "gpt-5-nano",
        "response_key": "gpt5_response",
        "type": "openai"
    },
    "qwen": {
        "client": qwen_client,
        "model_name": "qwen-flash",
        "response_key": "qwen_response",
        "type": "openai"
    },
    "gemini": {
        "client": gemini_client,
        "model_name": "gemini-2.5-flash-lite",
        "response_key": "gemini_response",
        "type": "gemini"
    },
    # "claude": {
    #     "client": claude_client,
    #     "model_name": "claude-3-5-haiku-20241022",
    #     "response_key": "claude_response",
    #     "type": "anthropic"
    # },
}

# --- Step 4: Core Functions ---
def get_model_response(config, conversation_history):
    """
    Sends a conversation history to a model and returns its response.
    Includes error handling and retries for robustness.
    """
    max_retries = 2
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            model_type = config.get("type", "openai")
            model_name = config["model_name"]

            if model_type == "openai":
                response = config["client"].chat.completions.create(
                    model=model_name,
                    messages=conversation_history
                )
                return response.choices[0].message.content

            elif model_type == "anthropic":
                system = conversation_history[0]["content"] if conversation_history and conversation_history[0]["role"] == "system" else ""
                messages = conversation_history[1:] if conversation_history and conversation_history[0]["role"] == "system" else conversation_history
                response = config["client"].messages.create(
                    model=model_name,
                    system=system,
                    messages=messages,
                    max_tokens=2000
                )
                return response.content[0].text

            elif model_type == "gemini":
                # Separate system instruction (if present) from the rest of the turn contents
                system_instruction = None
                start_idx = 0
                if conversation_history and conversation_history[0]["role"] == "system":
                    system_instruction = conversation_history[0]["content"]
                    start_idx = 1

                # Build typed contents for the Gemini SDK.
                # Use types.Part(text=...) instead of types.Part.from_text(...) to avoid positional-arg issues.
                contents = []
                for msg in conversation_history[start_idx:]:
                    # Gemini/GenAI expects role to be 'user' or 'model' (not 'assistant')
                    role = "user" if msg["role"] == "user" else "model"
                    part = types.Part(text=msg["content"])  # robust form: avoid .from_text positional bug
                    contents.append(types.Content(role=role, parts=[part]))

                # Build config with optional system_instruction
                gen_config = types.GenerateContentConfig(
                    system_instruction=system_instruction
                ) if system_instruction else None

                # Call Gemini
                response = config["client"].models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=gen_config
                )

                # The SDK exposes a convenient .text aggregator for the response
                return getattr(response, "text", str(response))

        except Exception as e:
            # Quietly handle retries
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
        return None

    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    conversation_histories = {key: [system_message] for key in MODELS_TO_TEST}

    for turn in data.get("dialogue", []):
        prompt_text = turn.get("prompt")
        if not prompt_text:
            continue

        if "ai_response" in turn:
            turn["safe_response"] = turn.pop("ai_response")

        for model_key, config in MODELS_TO_TEST.items():
            current_history = conversation_histories[model_key]
            current_history.append({"role": "user", "content": prompt_text})

            response_text = get_model_response(config, current_history)

            turn[config["response_key"]] = response_text
            current_history.append({"role": "assistant", "content": response_text})

    return data

# Wrapper for concurrent execution
import random

def process_and_save_file(filename, input_dir, output_dir):
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
            filename = new_filename

        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        return f"✅ Successfully processed and saved to: {filename}"
    else:
        return f"❌ Failed to process: {filename}"

# --- Step 5: Main Execution Logic ---
def main():
    print("Starting dialogue processing for AI safety research...")

    required_keys = ["OPENAI_API_KEY", "QWEN_API_KEY", "ANTHROPIC_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    # ensure at least one Google/Gemini key is present
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        missing_keys.append("GOOGLE_API_KEY or GEMINI_API_KEY")

    if missing_keys:
        print(f"\n[ERROR] Missing API keys: {', '.join(missing_keys)}! Add them to your .env file.")
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

    with ThreadPoolExecutor(max_workers) as executor:
        future_to_file = {
            executor.submit(process_and_save_file, filename, INPUT_DIR, OUTPUT_DIR): filename
            for filename in dialogue_files
        }

        for future in tqdm(as_completed(future_to_file), total=len(dialogue_files), desc="Processing files"):
            try:
                result_message = future.result()
                print(result_message)
            except Exception as exc:
                filename = future_to_file[future]
                tqdm.write(f"--- ❌ An unexpected error occurred while processing {filename}: {exc} ---")

    print("\n--- All files processed. ---")

if __name__ == "__main__":
    main()
