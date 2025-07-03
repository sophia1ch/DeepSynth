
import json
import pickle
import os
import gc
import torch
from DSL import zendo
from data.create_programs import convert_prolog_to_dsl
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
output_path = "zendo_play_results_2.jsonl"  # Use JSONL (newline-delimited JSON)
already_done = set()

if os.path.exists(output_path):
    with open(output_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                already_done.add(entry["task_index"])
            except:
                continue

for task_index, (name, examples) in enumerate(tasks):
    if task_index in already_done:
        print(f"Skipping already processed task {task_index}")
        continue

    print(f"\n=== Running task {task_index} ===")

    try:
        gm = GameMaster(true_program=convert_prolog_to_dsl(name, cfg), dataset=examples.copy(), zendo_dsl=zendo_dsl, cfg=cfg)
        player = ZendoPlayer(player_id=0, cfg=cfg, dsl=zendo_dsl, model=model)

        guesses, won = play_game(gm, player, return_guesses=True)
        print(f"Task {task_index} - Guesses: {len(guesses)}")

        result = {
            "task_index": task_index,
            "true_program": str(gm.true_program),
            "guesses": [str(g) for g in guesses],
            "num_examples": len(player.examples),
            "solved": won
        }

        with open(output_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    except Exception as e:
        print(f"❌ Task {task_index} failed: {e}")
        continue

    # 💡 Clean up memory explicitly
    del gm
    del player
    del guesses
    torch.cuda.empty_cache()
    gc.collect()

print(f"\n✅ Appended all results to '{output_path}'")