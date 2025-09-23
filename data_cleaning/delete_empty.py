import os
import json

def find_empty_string_keys(data, keys_found_in_file):
    """
    Recursively traverses a JSON structure (dict or list) to find all keys
    that have an empty string value.

    Args:
        data: The current piece of JSON data (can be a dict or list).
        keys_found_in_file (set): A set to store the keys found in the current file.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if value == "":
                keys_found_in_file.add(key)
            elif isinstance(value, (dict, list)):
                find_empty_string_keys(value, keys_found_in_file)
    elif isinstance(data, list):
        for item in data:
            find_empty_string_keys(item, keys_found_in_file)

def analyze_json_folder_recursively(folder_path):
    """
    Scans a folder for JSON files, removes files containing any empty string values,
    and reports the deleted files and the keys with empty strings found in them.
    """
    deleted_files = []  # Track deleted files for reporting

    # --- 1. Check if the 'results' folder exists ---
    if not os.path.isdir(folder_path):
        print(f"❌ Error: The folder '{folder_path}' was not found.")
        return

    # --- 2. Loop through all files in the folder ---
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            keys_found_in_this_file = set()

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # --- 3. Recursively find all keys with empty strings ---
                find_empty_string_keys(data, keys_found_in_this_file)

                # --- 4. If empty strings were found, delete the file ---
                if keys_found_in_this_file:
                    os.remove(file_path)
                    deleted_files.append((filename, keys_found_in_this_file))
                    print(f"🗑️ Deleted '{filename}' because it contained empty strings for key(s): {', '.join(keys_found_in_this_file)}")

            except json.JSONDecodeError:
                print(f"⚠️ Warning: Could not read '{filename}'. It's not a valid JSON file. Skipping.")
            except Exception as e:
                print(f"An unexpected error occurred with '{filename}': {e}")

    # --- 5. Print the final results ---
    if not deleted_files:
        print(f"✅ No JSON files with empty strings found in the '{folder_path}' folder.")
    else:
        print(f"📊 Analysis complete. Deleted {len(deleted_files)} file(s) containing empty strings:\n")
        for filename, keys in deleted_files:
            print(f"- '{filename}': Contained empty strings for key(s): {', '.join(sorted(keys))}")

# --- Run the analysis ---
if __name__ == "__main__":
    analyze_json_folder_recursively('results')