import os
import json

def verified_update_dialogue_evals():
    """
    Finds common JSON files and updates evaluation keys in the target folder
    from the source folder, but only after verifying that the turns match based
    on 'turn' number and response content.
    """
    source_folder = 'safeguard_results'
    target_folder = 'safeguard_results_llama_align'

    # --- 1. Find Common Files ---
    try:
        source_files = set(os.listdir(source_folder))
        target_files = set(os.listdir(target_folder))
        common_files = source_files.intersection(target_files)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Please ensure both folders exist.")
        return

    if not common_files:
        print("No common files found. Exiting.")
        return

    print(f"Found {len(common_files)} common files. Starting verified update process...\n")
    processed_count = 0
    updated_count = 0

    # --- 2. Process Each Common File ---
    for filename in common_files:
        if not filename.endswith('.json'):
            continue

        source_filepath = os.path.join(source_folder, filename)
        target_filepath = os.path.join(target_folder, filename)
        
        try:
            with open(source_filepath, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
            with open(target_filepath, 'r', encoding='utf-8') as f:
                target_data = json.load(f)

            source_dialogue = source_data.get('dialogue', [])
            target_dialogue = target_data.get('dialogue', [])

            # For efficient lookup, create a map of the target turns using their 'turn' number as the key
            target_turn_map = {turn.get('turn'): turn for turn in target_dialogue if 'turn' in turn}
            
            was_file_modified = False
            print(f"Processing: {filename}")

            # --- 3. Match, Verify, and Update Each Turn ---
            for source_turn in source_dialogue:
                turn_id = source_turn.get('turn')
                if turn_id is None:
                    continue  # Skip source turn if it has no turn number

                # Find the corresponding turn in the target file
                target_turn = target_turn_map.get(turn_id)

                if not target_turn:
                    print(f"  -> ⚠️  Warning: Turn {turn_id} found in source but not in target. Skipping.")
                    continue
                
                # VERIFICATION STEP: Check if the responses are identical before updating
                source_grok = source_turn.get('grok_response')
                target_grok = target_turn.get('grok_response')
                source_deepseek = source_turn.get('deepseek_response')
                target_deepseek = target_turn.get('deepseek_response')

                if source_grok == target_grok and source_deepseek == target_deepseek:
                    # If verification passes, copy the evaluation keys
                    keys_to_copy = ['grok_response_eval', 'deepseek_response_eval']
                    for key in keys_to_copy:
                        if key in source_turn:
                            target_turn[key] = source_turn[key]
                            was_file_modified = True
                else:
                    # If verification fails, log it and do not update this turn
                    print(f"  -> ⚠️  Warning: Responses for turn {turn_id} do not match. No update performed.")
            
            # --- 4. Save the Modified File (only if changes were made) ---
            if was_file_modified:
                with open(target_filepath, 'w', encoding='utf-8') as f:
                    json.dump(target_data, f, indent=4)
                print(f"  -> ✅ File updated.\n")
                updated_count += 1
            else:
                print(f"  -> No verifiable changes made.\n")
            
            processed_count += 1

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"❌ Error processing {filename}: {e}. Skipping.\n")

    print(f"✨ Process complete. Processed {processed_count} files, successfully updated {updated_count} files.")


if __name__ == '__main__':
    verified_update_dialogue_evals()