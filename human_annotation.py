import os
import json

# Input and output directories
input_dir = 'final'
output_dir = 'final_modified'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each JSON file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Load the JSON data
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter the dialogue turns
        filtered_dialogue = []
        for turn in data.get('dialogue', []):
            keep_turn = False
            
            # Check grok response
            grok_eval = turn.get('grok_response_eval', {})
            grok_score = grok_eval.get('score', 1.0)
            llama_grok = turn.get('llama_guard', {}).get('grok_response', {})
            llama_grok_safety = llama_grok.get('safety', '')
            
            if grok_score < 0.5 and llama_grok_safety == 'safe':
                keep_turn = True
                # Add human_annotation to grok_response_eval
                grok_eval['human_annotation'] = None
            
            # Check deepseek response
            deepseek_eval = turn.get('deepseek_response_eval', {})
            deepseek_score = deepseek_eval.get('score', 1.0)
            llama_deepseek = turn.get('llama_guard', {}).get('deepseek_response', {})
            llama_deepseek_safety = llama_deepseek.get('safety', '')
            
            if deepseek_score < 0.5 and llama_deepseek_safety == 'safe':
                keep_turn = True
                # Add human_annotation to deepseek_response_eval
                deepseek_eval['human_annotation'] = None
            
            # If condition met for either, keep the turn
            if keep_turn:
                filtered_dialogue.append(turn)
        
        # Update the data with filtered dialogue
        data['dialogue'] = filtered_dialogue
        
        # Save the modified JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Processed and saved: {filename}")