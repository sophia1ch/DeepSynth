import os
import json
import pickle
from pathlib import Path

# Define root data folders and corresponding CSV files
data_dirs = ["../Master_thesis/json_files/dataset/training1", "../Master_thesis/json_files/dataset/training2", "../Master_thesis/json_files/dataset/training3", "../Master_thesis/json_files/dataset/training4", "../Master_thesis/json_files/dataset/training5", "../Master_thesis/json_files/dataset/training6",
        "../Master_thesis/json_files/dataset/training7", "../Master_thesis/json_files/dataset/training8", "../Master_thesis/json_files/dataset/training9", "../Master_thesis/json_files/dataset/training10_pointing", "../Master_thesis/json_files/dataset/training11_length",
        "../Master_thesis/json_files/dataset/training12_pointing", "../Master_thesis/json_files/dataset/training13_on_top", "../Master_thesis/json_files/dataset/training14", "../Master_thesis/json_files/dataset/training15","../Master_thesis/json_files/dataset/training16_touching"]

# Initialize list of tasks and ground truth programs
tasks = []


# Step 1: collect all scenes and group them by rule
data_root = Path(".")

for data_dir in data_dirs:
    csv_file = Path(data_dir) / "ground_truth.csv"
    csv_path = data_root / csv_file
    print(f"Processing {csv_path}...")
    with open(csv_path, "r") as f:
        print(f"Reading {csv_path}...")
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            scene_name = parts[0]  # e.g. 0_0
            rule_text = parts[2].strip()
            if rule_text not in tasks and "_n" in scene_name:
                tasks.append(rule_text)
print(f"Loaded {len(tasks)} examples")

