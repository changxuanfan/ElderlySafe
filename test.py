import os
import json

# Input folder and output file
input_folder = "AgingCare_Posts"
output_file = "eldercare_posts_agingcare.json"

all_stories = []
story_count = 0

# Loop through each JSON file in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        filepath = os.path.join(input_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if "stories" in data:
                    stories = data["stories"]
                    all_stories.extend(stories)
                    story_count += len(stories)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping {filename} - invalid JSON")

# Save combined file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({"stories": all_stories}, f, indent=4, ensure_ascii=False)

print(f"✅ Combined {story_count} stories into {output_file}")
