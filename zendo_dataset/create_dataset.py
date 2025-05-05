import os
import json
import pickle
from pathlib import Path

# Define root data folders and corresponding CSV files
data_dirs = ["data5", "data6", "data7", "data8"]
csv_files = ["ground_truth5.csv", "ground_truth6.csv", "ground_truth7.csv", "ground_truth8.csv"]

# Initialize list of tasks and ground truth programs
tasks = []
programs = []

# Helper function to load a single scene
def load_scene(path):
    with open(path, "r") as f:
        return json.load(f)

# Helper function to group examples by rule
from collections import defaultdict
rule_to_examples = defaultdict(list)

# Step 1: collect all scenes and group them by rule
data_root = Path(".")

for data_dir, csv_file in zip(data_dirs, csv_files):
    csv_path = data_root / csv_file
    print(f"Processing {csv_path}...")
    with open(csv_path, "r") as f:
        print(f"Reading {csv_path}...")
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            scene_name = parts[0]  # e.g. 0_0
            rule_text = parts[2].strip()

            # Positive or negative is based on the filename
            is_negative = scene_name.endswith("_n")

            # Path to corresponding JSON file
            rule_idx = scene_name.split('_')[0]  # e.g., '10'
            scene_path = data_root / data_dir / rule_idx / (scene_name + ".json")

            if not scene_path.exists():
                print(f"Scene file {scene_path} does not exist. Skipping...")
                continue

            try:
                scene = load_scene(scene_path)
                label = 0 if is_negative else 1
                prolog_query = parts[3].strip()
                rule_to_examples[rule_text].append((scene, label, prolog_query))
            except Exception as e:
                print(f"Error reading {scene_path}: {e}")
                continue
print(f"Loaded {len(rule_to_examples)} examples")
# Step 2: Create dataset
for rule_text, examples in rule_to_examples.items():
    positives = [ex for ex in examples if ex[1] == 1][:5]
    negatives = [ex for ex in examples if ex[1] == 0][:5]

    if len(positives) >= 2 and len(negatives) >= 2:
        task_examples = positives + negatives  # each is (scene, label, query)
        tasks.append([(scene, label) for (scene, label, _) in task_examples])

        rule_query = next((q for (_, _, q) in task_examples if q.startswith("generate_valid_structure")), task_examples[0][2])
        programs.append(rule_query)
    else:
        print(f"Rule {rule_text} does not have enough examples: {len(positives)} positives, {len(negatives)} negatives")
print(f"Prepared {len(tasks)} tasks.")

# Step 3: Save to pickle
with open("zendo_dataset.pkl", "wb") as f:
    pickle.dump(tasks, f)

with open("zendo_programs.pkl", "wb") as f:
    pickle.dump(programs, f)

print("Saved zendo_dataset.pkl and zendo_programs.pkl")
