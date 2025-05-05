import torch 
import logging
import argparse
import matplotlib.pyplot as plt
from types import SimpleNamespace

import pickle
from pathlib import Path
from Predictions.IOencodings import ZendoStructureEncoding
from Predictions.models import RulesPredictor, BigramsPredictor
from Predictions.dataset_sampler import Dataset
from Predictions.embeddings import ZendoRNNEmbedding
from Predictions.models import RulesPredictor
from type_system import Arrow, List, INT
from model_loader import get_model_name
from zendo_config import cfg

logging_levels = {0: logging.INFO, 1: logging.DEBUG}

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', dest='verbose', default=0)
args, unknown = parser.parse_known_args()

verbosity = int(args.verbose)
logging.basicConfig(format='%(message)s', level=logging_levels[verbosity])

## HYPERPARAMETERS

dataset_name = "zendo"
type_request = None  # For Zendo, we use None for generic models
dataset_size: int = 10000
nb_epochs: int = 20
batch_size: int = 16  # smaller batch size due to complex structures

############################
#### LOAD ZENDO DATASET ####
############################

def load_zendo_dataset(pkl_path="zendo_dataset/zendo_dataset.pkl", program_path="zendo_dataset/zendo_programs.pkl"):
    with open(pkl_path, "rb") as f:
        tasks = pickle.load(f)
    with open(program_path, "rb") as f:
        programs = pickle.load(f)
    return tasks, programs

tasks, programs = load_zendo_dataset()

## MODEL INIT
base_symbols = ["red", "blue", "yellow", "pyramid", "wedge", "block", "upright", "flat", "upside_down", "cheesecake", "vertical"]
max_objects = 7

IOEncoder = ZendoStructureEncoding(lexicon=base_symbols, max_objects=max_objects)

print("✅ Symbol count for embedding:", IOEncoder.lexicon_size)  # should now include ID_0–6, PAD, NONE

IOEmbedder = ZendoRNNEmbedding(
    IOEncoder=IOEncoder,
    output_dimension=32,
    size_hidden=64,
    number_layers_RNN=1
)
latent_encoder = torch.nn.Sequential(
    torch.nn.Linear(IOEmbedder.output_dimension, 64),
    torch.nn.Sigmoid(),
    torch.nn.Linear(64, 64),
    torch.nn.Sigmoid()
)

model = RulesPredictor(
    cfg=cfg,
    IOEncoder=IOEncoder,
    IOEmbedder=IOEmbedder,
    latent_encoder=latent_encoder
)

print("Training model:", get_model_name(model), "on", dataset_name)
print("Type Request:", type_request or "generic")

nb_examples_max: int = 10  # 5 positive + 5 negative examples

############################
######## TRAINING ##########
############################

def train(model, tasks, programs):
    savename = get_model_name(model) + "_zendo.weights"
    for epoch in range(nb_epochs):
        for i in range(0, len(tasks), batch_size):
            batch_IOs = []
            batch_programs = []
            batch = tasks[i:i + batch_size]
            for j, examples in enumerate(batch):
                batch_IOs.append(examples)
                batch_programs.append(programs[i + j])
            batch_predictions = model(batch_IOs)
            model.optimizer.zero_grad()
            targets = torch.stack([
                model.ProgramEncoder(program) for program in batch_programs
            ])
            loss_value = model.loss(batch_predictions, targets)
            loss_value.backward()
            model.optimizer.step()
            print(f"Minibatch {i // batch_size}: loss={float(loss_value)}")

        print(f"Epoch {epoch}: loss={float(loss_value)}")
        torch.save(model.state_dict(), savename)

def print_embedding(model):
    print(model.IOEmbedder.embedding.weight)
    x = [x.detach().numpy() for x in model.IOEmbedder.embedding.weight[:, 0]]
    y = [x.detach().numpy() for x in model.IOEmbedder.embedding.weight[:, 1]]
    label = [str(a) for a in model.IOEncoder.lexicon]
    plt.plot(x, y, 'o')
    for i, s in enumerate(label):
        plt.annotate(s, (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.show()

train(model, tasks, programs)
print_embedding(model)
