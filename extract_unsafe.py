import json
import os

input_dir = 'final_judge'
output_dir = 'final_final'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file in os.listdir(input_dir):
    if file.endswith('.json'):
        input_path = os.path.join(input_dir, file)
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        extracted = []
        if 'dialogue' in data:
            for turn in data['dialogue']:
                turn_num = turn.get('turn')
                prompt = turn.get('prompt')
                for model in ['grok', 'deepseek']:
                    eval_key = f"{model}_response_eval"
                    if eval_key in turn and 'safety_judge' in turn[eval_key]:
                        safety_judge = turn[eval_key]['safety_judge']
                        if safety_judge in ['low-risk unsafe', 'high-risk unsafe']:
                            response = turn.get(f"{model}_response")
                            eval_extract = {
                                'safety_judge': safety_judge,
                                'category': turn[eval_key].get('category', ''),
                                'risk_reasoning': turn[eval_key].get('risk_reasoning', '')
                            }
                            extracted.append({
                                'prompt': prompt,
                                'model': model,
                                'response': response,
                                'eval': eval_extract
                            })
        
        if extracted:  # Only save if there is data
            output_path = os.path.join(output_dir, file)
            with open(output_path, 'w') as f:
                json.dump({'extracted': extracted}, f, indent=4)