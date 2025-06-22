
# Choose a task from the dataset
import pickle
from DSL import zendo
from data.create_programs import convert_prolog_to_dsl
from experiment_helper import task_set2zendodataset
from grammar import dsl
from model_loader import __buildintlist_zendo_model
from zendo.game import play_game
from zendo.game_master import GameMaster
from zendo.player import ZendoPlayer

# --- Load Dataset ---
def load_zendo_dataset(pkl_path="data/zendo_test_tensors.pkl"):
    with open(pkl_path, "rb") as f:
        tasks = pickle.load(f)
    return tasks

tasks = load_zendo_dataset()
print("Loaded", len(tasks))

base_symbols = ["red", "blue", "yellow", "pyramid", "wedge", "block", "upright", "flat", "upside_down", "cheesecake", "vertical"]
max_objects = 7
zendo_dsl = dsl.DSL(zendo.semantics, zendo.primitive_types, None)
cfg, model = __buildintlist_zendo_model(dsl=zendo_dsl, max_program_depth=15, size_max=11, size_hidden=64, embedding_output_dimension=77, number_layers_RNN=1, autoload=True)
name, examples = tasks[0]
print("Selected task:", name, "with", len(examples), "examples")
task_index = 0

gm = GameMaster(true_program=convert_prolog_to_dsl(name, cfg), dataset=examples, zendo_dsl=zendo_dsl, cfg=cfg)
player = ZendoPlayer(cfg=cfg, dsl=zendo_dsl, model=model)

play_game(gm, player)