from data.create_programs import convert_prolog_to_dsl
import grammar.dsl as dsl
from DSL import zendo
from experiment_helper import __get_type_request
import torch 
import logging
import argparse
import matplotlib.pyplot as plt
import signal
import sys

import pickle
from model_loader import __build_generic_zendo_model, get_model_name

logging_levels = {0: logging.INFO, 1: logging.DEBUG}

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', dest='verbose', default=0)
args, unknown = parser.parse_known_args()

verbosity = int(args.verbose)
logging.basicConfig(format='%(message)s', level=logging_levels[verbosity])

## HYPERPARAMETERS

dataset_name = "zendo"
dataset_size: int = 10000
nb_epochs: int = 40
batch_size: int = 16  # smaller batch size due to complex structures

def load_zendo_dataset(pkl_path="data/zendo_dataset_tensors.pkl", program_path="data/zendo_programs.pkl"):
    with open(pkl_path, "rb") as f:
        tasks = pickle.load(f)
    with open(program_path, "rb") as f:
        prolog_programs = pickle.load(f)
    if len(tasks) != len(prolog_programs):
        print(prolog_programs, len(tasks), "tasks and", len(prolog_programs), "programs found.")
        raise ValueError("Number of tasks and programs do not match.")
    return tasks, prolog_programs

def handle_sigint(signal_received, frame):
    print('\nSIGINT or CTRL-C detected. Exiting gracefully...', flush=True)
    sys.exit(0)

tasks, prolog_programs = load_zendo_dataset()
print("Loaded", len(tasks), "tasks and", len(prolog_programs), "programs.")

## MODEL INIT
base_symbols = ["red", "blue", "yellow", "pyramid", "wedge", "block", "upright", "flat", "upside_down", "cheesecake", "vertical"]
max_objects = 7
zendo_dsl = dsl.DSL(zendo.semantics, zendo.primitive_types, None)
cfg, model = __build_generic_zendo_model(dsl=zendo_dsl, max_program_depth=15, size_max=11, size_hidden=64, embedding_output_dimension=10, number_layers_RNN=1)
programs = [convert_prolog_to_dsl(p, cfg) for p in prolog_programs]
if len(tasks) != len(programs):
    print(programs, len(tasks), "tasks and", len(programs), "programs found.")
    raise ValueError("Number of tasks and programs do not match.")
print("Training model:", get_model_name(model), "on", dataset_name)

nb_examples_max: int = 20  # 10 positive + 10 negative examples

############################
######## TRAINING ##########
############################

def train(model, tasks, programs):
    signal.signal(signal.SIGINT, handle_sigint)
    savename = get_model_name(model) + "_zendo.weights"
    try:
        for epoch in range(nb_epochs):
            for i in range(0, len(tasks), batch_size):
                batch_IOs = []
                batch_programs = []
                batch = tasks[i:i + batch_size]
                for j, examples in enumerate(batch):
                    batch_IOs.append(examples)
                    batch_programs.append(programs[i + j])
                raw_predictions = model(batch_IOs)
                batch_grammars = model.reconstruct_grammars(raw_predictions, [model.cfg_dictionary.keys().__iter__().__next__()] * len(batch_programs))

                # Compute loss using grammar log-probabilities
                model.optimizer.zero_grad()
                loss_value = model.loss(batch_grammars, batch_programs)
                loss_value.backward()
                model.optimizer.step()
                print(f"Minibatch {i // batch_size}: loss={float(loss_value)}")

            print(f"Epoch {epoch}: loss={float(loss_value)}")
            torch.save(model.state_dict(), savename)
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model state.")
        torch.save(model.state_dict(), savename)
        return

def plot_embedding(embedding, labels, title):
    weights = embedding.weight.detach().cpu()
    x = weights[:, 0].numpy()
    y = weights[:, 1].numpy()

    plt.figure()
    plt.title(title)
    plt.plot(x, y, 'o')
    for i, label in enumerate(labels):
        plt.annotate(str(label), (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.grid(True)
    plt.show()

train(model, tasks, programs)
