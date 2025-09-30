import os
import json
from openai import OpenAI, RateLimitError, APIError
import time
from dotenv import load_dotenv

load_dotenv()
# --- Configuration ---
# Directory containing the original JSON files
INPUT_DIRECTORY = "eval_results"

# Directory where the updated files will be saved
OUTPUT_DIRECTORY = "safeguard_results"
# --- End of Configuration ---

# Initialize the OpenAI client
# It will automatically pick up the API key from the environment variable OPENAI_API_KEY
try:
    client = OpenAI()
except APIError as e:
    print("Error initializing OpenAI client. Is your API key set correctly as an environment variable?")
    print(f"Details: {e}")
    exit()

def get_moderation_result(text: str):
    """
    Sends text to the OpenAI Moderation API and returns a structured result.
    Handles empty or invalid input gracefully.
    """
    # The API throws an error for empty strings, but an empty string is not harmful.
    if not text or not isinstance(text, str) or text.strip() == "":
        return {"flagged": False, "categories": {}}

    try:
        response = client.moderations.create(input=text)
        result = response.results[0]
        
        # Convert the Pydantic model to a standard dictionary for JSON serialization
        categories_dict = {k: v for k, v in result.categories.__dict__.items()}
        
        return {
            "flagged": result.flagged,
            "categories": categories_dict
        }
    except RateLimitError:
        print("Rate limit reached. Waiting for 60 seconds before retrying...")
        time.sleep(60)
        return get_moderation_result(text) # Retry the request
    except APIError as e:
        print(f"An API error occurred: {e}")
        return {"error": str(e), "flagged": None, "categories": {}}
    except Exception as e:
        print(f"An unexpected error occurred for text: '{text[:50]}...'")
        print(f"Error details: {e}")
        return {"error": str(e), "flagged": None, "categories": {}}


def process_json_file(filepath: str):
    """
    Opens a JSON file, adds OpenAI moderation results, and saves it
    to the specified output directory.
    """
    print(f"Processing file: {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "dialogue" not in data or not isinstance(data["dialogue"], list):
            print(f"  - Skipping {filepath}: 'dialogue' key is missing or not a list.")
            return

        for turn in data["dialogue"]:
            # Keys to be moderated in each turn
            keys_to_moderate = ["prompt", "safe_response", "grok_response", "deepseek_response"]
            
            moderation_results = {}
            for key in keys_to_moderate:
                text_to_check = turn.get(key, "") # Use .get() for safety
                print(f"  - Moderating 'turn {turn.get('turn', '?')}' -> '{key}'...")
                moderation_results[key] = get_moderation_result(text_to_check)
            
            # Add the new 'openai_moderation' object to the turn
            turn["openai_moderation"] = moderation_results

        # --- Create the new file path for saving ---
        filename = os.path.basename(filepath)
        output_filepath = os.path.join(OUTPUT_DIRECTORY, filename)

        # Write the updated data to the new file in the output directory
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  -> Successfully saved results to {output_filepath}\n")

    except json.JSONDecodeError:
        print(f"  - Skipping {filepath}: Invalid JSON format.")
    except Exception as e:
        print(f"  - An error occurred while processing {filepath}: {e}")


def main():
    """
    Main function to find and process all JSON files, saving them to a new directory.
    """
    # Check if the input directory exists
    if not os.path.isdir(INPUT_DIRECTORY):
        print(f"Error: Input directory '{INPUT_DIRECTORY}' not found.")
        print("Please make sure the script is in the same parent directory as 'eval_results'.")
        return
        
    # Create the output directory if it doesn't already exist
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    print(f"Moderated files will be saved in the '{OUTPUT_DIRECTORY}' directory.")

    json_files = [f for f in sorted(os.listdir(INPUT_DIRECTORY)) if f.endswith('.json')]

    if not json_files:
        print(f"No JSON files found in '{INPUT_DIRECTORY}'.")
        return

    print(f"\nFound {len(json_files)} JSON file(s) to process.\n")
    for filename in json_files:
        filepath = os.path.join(INPUT_DIRECTORY, filename)
        process_json_file(filepath)
    
    print("--- All files processed. ---")


if __name__ == "__main__":
    main()