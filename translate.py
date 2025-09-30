import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from a .env file
load_dotenv()
api_key = os.getenv("XAI_API_KEY")

# Initialize xAI API client
client = OpenAI(base_url="https://api.x.ai/v1", api_key=api_key)

# Input and output directories
input_dir = "final_modified"
output_dir = "final_translated"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Model to use
model = "grok-4-fast-non-reasoning"

# Function to translate JSON using Grok API
def translate_json(json_data, filename):
    try:
        # Convert JSON to string for the prompt
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
        
        # Prompt to translate string values to natural Chinese
        prompt = f"""You are a helpful translator. Given the following JSON data, translate all string values into authentic, natural, idiomatic Chinese (Simplified Chinese preferred). 
        Keep the structure exactly the same, non-string values and keys unchanged, and ensure the output is valid JSON only (no extra text or explanations).

JSON to translate:
{json_str}"""
        
        # API call for chat completion
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise JSON translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent output
            max_tokens=150000  # Adjust based on expected size
        )
        
        # Extract the response content
        translated_text = response.choices[0].message.content.strip()
        
        # Parse the response as JSON
        translated_json = json.loads(translated_text)
        return filename, translated_json, None
    except json.JSONDecodeError as e:
        return filename, None, f"Error parsing translated JSON: {e}\nRaw response: {translated_text}"
    except Exception as e:
        return filename, None, f"Error processing {filename}: {e}"

# Function to process a single JSON file
def process_file(filename):
    if not filename.endswith(".json"):
        return filename, None, "Not a JSON file"
    
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)
    
    # --- CHANGE: Check if the output file already exists ---
    if os.path.exists(output_path):
        print(f"Skipping {filename}: Already exists in output directory.")
        # Return a special status to indicate the file was skipped
        return filename, output_path, "skipped"
    
    print(f"Processing {filename}...")
    
    # Load the original JSON
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        # Translate
        filename, translated_data, error = translate_json(json_data, filename)
        
        if error:
            return filename, None, error
        
        # Save the translated JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        return filename, output_path, None
    except Exception as e:
        return filename, None, f"Error reading {filename}: {e}"

# Main function to process files concurrently
def main():
    # Get list of JSON files
    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    
    if not json_files:
        print("No JSON files found in the input directory.")
        return
    
    # Process files concurrently using ThreadPoolExecutor
    max_workers = min(50, len(json_files))  # Limit concurrent threads to avoid overwhelming API
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_file, filename): filename for filename in json_files}
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                filename, output_path, result = future.result()
                # --- CHANGE: Handle the 'skipped' status ---
                if result == "skipped":
                    # Message was already printed in the worker thread, so do nothing.
                    pass
                elif result: # An error occurred
                    print(f"Failed to process {filename}: {result}")
                else: # Success
                    print(f"Saved translated {filename} to {output_path}")
            except Exception as e:
                print(f"Exception occurred for {filename}: {e}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()