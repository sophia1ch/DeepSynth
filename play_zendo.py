
# Choose a task from the dataset
import json
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
results = []

for task_index, (name, examples) in enumerate(tasks[:2]):
    print(f"\n=== Running task {task_index} ===")
    gm = GameMaster(true_program=convert_prolog_to_dsl(name, cfg), dataset=examples.copy(), zendo_dsl=zendo_dsl, cfg=cfg)
    player = ZendoPlayer(cfg=cfg, dsl=zendo_dsl, model=model)

    guesses, won = play_game(gm, player, return_guesses=True)
    print(f"Task {task_index} - Guesses: {len(guesses)}")
    results.append({
        "task_index": task_index,
        "true_program": str(gm.true_program),
        "guesses": [str(g) for g in guesses],
        "num_examples": len(player.examples),
        "solved": won
    })

# Save results to file
with open("zendo_play_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved results for {len(results)} tasks to 'zendo_play_results.json'")