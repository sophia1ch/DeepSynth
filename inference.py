import torch
import pickle
from pathlib import Path
from Predictions.IOencodings import ZendoStructureEncoding
from Predictions.embeddings import ZendoRNNEmbedding
from Predictions.models import RulesPredictor
from zendo_config import cfg
from model_loader import get_model_name

# === Load Trained Model ===

base_symbols = ["red", "blue", "yellow", "pyramid", "wedge", "block", "upright", "flat", "upside_down", "cheesecake", "vertical"]
max_objects = 7

IOEncoder = ZendoStructureEncoding(lexicon=base_symbols, max_objects=max_objects)

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

model_path = get_model_name(model) + "_zendo.weights"
model.load_state_dict(torch.load(model_path))
model.eval()

# === Load One Task ===

def load_tasks(path="zendo_dataset.pkl"):
    with open(path, "rb") as f:
        tasks = pickle.load(f)
    return tasks

tasks = load_tasks()
task_id = 0  # choose any valid task index
task_examples = tasks[task_id]  # List of (encoded_structure, label)

# === Run Inference ===

with torch.no_grad():
    predicted_program_embedding = model([task_examples])  # batch of size 1
    print(f"Predicted embedding vector:\n{predicted_program_embedding.squeeze()}")

# === Optional: Decode the program (if supported) ===

if hasattr(model, "ProgramDecoder"):
    decoded_program = model.ProgramDecoder(predicted_program_embedding[0])
    print("\nDecoded DSL program:")
    print(decoded_program)
else:
    print("\nNote: ProgramDecoder not available.")
