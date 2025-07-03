
import pickle
import torch
from pathlib import Path
import json
from DSL import zendo
from data.create_programs import convert_prolog_to_dsl
from grammar import dsl
from model_loader import __buildintlist_zendo_model
from zendo.game import play_game_state
from zendo.game_master import ZendoStateGameMaster
from zendo.player import HeuristicZendoPlayer, ZendoPlayer

output_dir = Path("gamestates_heuristic_same_guessing")
output_dir.mkdir(parents=True, exist_ok=True)

# --- Load Dataset ---
def load_zendo_dataset(pkl_path="data/zendo_test_tensors.pkl"):
    with open(pkl_path, "rb") as f:
        tasks = pickle.load(f)
    return tasks

tasks = load_zendo_dataset()
print("Loaded", len(tasks))

base_symbols = ["red", "blue", "yellow", "pyramid", "wedge", "block", "upright", "flat", "upside_down", "cheesecake", "vertical", "doorstop"]
max_objects = 7
zendo_dsl = dsl.DSL(zendo.semantics, zendo.primitive_types, None)
cfg, model = __buildintlist_zendo_model(dsl=zendo_dsl, max_program_depth=15, size_max=11, size_hidden=64, embedding_output_dimension=77, number_layers_RNN=1, autoload=True)


for task_index, (name, examples) in enumerate([tasks[1]]):
    print(f"\n=== Running task {task_index} ===")
    state_path = output_dir / f"task_{task_index}_state.json"
    if state_path.exists():
        print(f"⏭️  Skipping task {task_index} — file already exists.")
        continue

    try:
        gm = ZendoStateGameMaster(true_program=convert_prolog_to_dsl(name, cfg), dataset=examples.copy(), zendo_dsl=zendo_dsl, cfg=cfg)
        player0 = HeuristicZendoPlayer(player_id=0, cfg=cfg, dsl=zendo_dsl, model=model, bar=5e-9)

        state = play_game_state(gm, [player0])
        with open(state_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    except Exception as e:
        print(f"❌ Task {task_index} failed: {e}")
        continue

print(f"\n✅ Finished")