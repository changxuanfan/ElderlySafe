import json
import os

# Define the folder path
folder_path = "final_final"

# Initialize an empty list to store all extracted entries
combined_extracted = []

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):  # Process only JSON files
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Check if "extracted" key exists and is a list
                if "extracted" in data and isinstance(data["extracted"], list):
                    combined_extracted.extend(data["extracted"])
                else:
                    print(f"Warning: No valid 'extracted' key in {filename}")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Create the combined JSON structure
combined_json = {"extracted": combined_extracted}

# Save the combined JSON to a new file
output_path = os.path.join(folder_path, "combined_extracted.json")
try:
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(combined_json, output_file, indent=4)
    print(f"Combined JSON saved to {output_path}")
except Exception as e:
    print(f"Error saving combined JSON: {e}")