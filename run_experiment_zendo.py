from DSL import zendo
import dsl
import torch
import csv
import os
import pickle

from run_experiment import gather_data, list_algorithms
from model_loader import __buildintlist_zendo_model
from experiment_helper import task_set2zendodataset

# --- Configuration ---
dataset_name = "zendo"
save_folder = "."

# --- Load Dataset ---
def load_zendo_dataset(pkl_path="zendo_test_tensors.pkl"):
    with open(pkl_path, "rb") as f:
        tasks = pickle.load(f)
    return tasks

tasks = load_zendo_dataset()
print("Loaded", len(tasks))

base_symbols = ["red", "blue", "yellow", "pyramid", "wedge", "block", "upright", "flat", "upside_down", "cheesecake", "vertical"]
max_objects = 7
zendo_dsl = dsl.DSL(zendo.semantics, zendo.primitive_types, None)
cfg, model = __buildintlist_zendo_model(dsl=zendo_dsl, max_program_depth=15, size_max=11, size_hidden=64, embedding_output_dimension=77, number_layers_RNN=1, autoload=True)

# --- Convert Tasks to Dataset ---
print(len(tasks), "tasks loaded.")
dataset = task_set2zendodataset(tasks, model, zendo_dsl, cfg)


# --- Run Inference & Export Results ---
for algo_index in range(len(list_algorithms)):
    print("Running algorithm index:", algo_index)
    algo_name = list_algorithms[algo_index][1]
    if algo_name != "Heap Search":
        print(f"Skipping algorithm {algo_name} as it is not 'heap search'.")
        continue

    print("Starting...")
    for splits in [2]:
        filename = f"{save_folder}/algo_{algo_name} {splits} CPUs_dataset_{dataset_name}_results_semantic.csv"
        if os.path.exists(filename):
            print("Already exists:", filename)
            continue

        print(f"Running {algo_name} with {splits} CPUs...")
        data = gather_data(dataset, 0)
        col_names = ["task_name", "program", "search_time", "evaluation_time",
                     "nb_programs", "cumulative_probability", "probability"]

        processed_data = []
        for task_name, results in data:
            for result in results:
                program, search_time, evaluation_time, nb_programs, cumulative_probability, probability = result

                # Format each row as one program result for this task
                processed_data.append([
                    str(task_name),
                    str(program),
                    search_time,
                    evaluation_time,
                    nb_programs,
                    cumulative_probability,
                    probability
                ])

                print("Processed program result:", processed_data[-1])

        with open(filename, "w", newline='') as fd:
            writer = csv.writer(fd)
            writer.writerow(col_names)
            writer.writerows(processed_data)

        print("Saved results to", filename)
