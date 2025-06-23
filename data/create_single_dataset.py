import re
import pickle
import torch
from pathlib import Path

def split_csv_line(line):
    """
    Split a CSV line, ignoring commas inside quotes.
    """
    # Regular expression to match values inside quotes and outside
    pattern = r'[^,"]+|"(?:\\.|[^"\\])*"'
    
    # Find all matches based on the pattern
    return re.findall(pattern, line)

def remove_generate_valid_structure(query):
    """Remove the 'generate_valid_structure' or 'generate_invalid_structure' wrapper if present."""
    
    # Match the pattern for 'generate_valid_structure([...], Structure)' and 'generate_invalid_structure([...], Structure)'
    match = re.match(r'\"generate_(valid|invalid)_structure\(\s*\[(.*)\]\s*,\s*Structure\s*\)\"', query.strip())
    
    if match:
        # Extract the inner part of the query (the part inside the brackets)
        return match.group(2).strip()  # Return the content inside the brackets without leading/trailing spaces
    else:
        print(f"Warning: Query '{query}' does not match expected format. Returning as is.")
        return query

# Define root data folders and corresponding CSV files
data_dirs = ["training23"]

# Initialize list of tasks and ground truth programs
tasks = []
programs = []

# Helper function to load a tensor file
def load_tensor(path):
    return torch.load(path)

# Helper function to group examples by rule
from collections import defaultdict
rule_to_examples = defaultdict(list)

# Step 1: collect all scenes and group them by rule
data_root = Path("../Master_thesis")
csv_root = Path("test-dataset")

for data_dir in data_dirs:
    csv_file = Path(data_dir) / "ground_truth.csv"
    csv_path = data_root / csv_root / csv_file
    print(f"Processing {csv_path}...")
    with open(csv_path, "r") as f:
        print(f"Reading {csv_path}...")
        header = f.readline()
        for line in f:
            parts = split_csv_line(line)
            scene_name = parts[0]  # e.g. 0_0
            identifier = data_dir + "_" + scene_name
            if len(rule_to_examples[identifier]) != 0:
                continue
            rule_text = parts[2].strip()

            # Positive or negative is based on the filename
            is_negative = scene_name.endswith("_n")

            # Path to corresponding tensor file
            rule_idx = scene_name.split('_')[0]  # e.g., '10'
            tensor_path = data_root / Path("rules_test") / Path(data_dir) / (scene_name + ".pt")

            if not tensor_path.exists():
                print(f"Scene tensor {tensor_path} does not exist. Skipping...")
                continue

            try:
                tensor = load_tensor(tensor_path)
                label = not is_negative
                prolog_query = parts[3]
                program_query = remove_generate_valid_structure(prolog_query)
                rule_to_examples[identifier].append((tensor, label, program_query))
                if tensor.shape[0] != 7:
                    print(f"Warning: Tensor {tensor_path} has unexpected shape {tensor.shape}, expected (7, ...), {rule_text}")	
            except Exception as e:
                print(f"Error reading {tensor_path}: {e}")
                continue

print(f"Loaded {len(rule_to_examples)} examples")
# Add the examples to a task list grouped by rule
task_list = defaultdict(list)
for identifier, example in rule_to_examples.items():
    parts = identifier.split('_')
    prefix = f"{parts[0]}_{parts[1]}"
    task_list[prefix].extend(example)

for rule_text, examples in task_list.items():
    print(f"Processing rule: {rule_text} with {len(examples)} examples")
    if len(examples) != 20:
        # print(f"Rule {rule_text} has less than 10 examples, skipping...")
        continue
    positives = [ex for ex in examples if ex[1] == 1][:10]
    negatives = [ex for ex in examples if ex[1] == 0][:10]

    if len(positives) != 10 or len(negatives) != 10:
        print(f"Rule {rule_text} does not have exactly 10 positives and 10 negatives, skipping...")
        continue
    task_examples = positives + negatives  # each is (tensor, label, query)
    rule_query = task_examples[0][2]
    tasks.append([rule_query, [(tensor, label) for (tensor, label, _) in task_examples]])
    programs.append(rule_query)

print(f"Prepared {len(tasks)} tasks.")

# Step 3: Save to pickle
with open("data/zendo_test_tensors.pkl", "wb") as f:
    pickle.dump(tasks, f)

print("Saved zendo_dataset_tensors.pkl and zendo_programs.pkl")
