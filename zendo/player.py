import json
from pathlib import Path
import subprocess
from data.create_prolog import dsl_to_prolog
from data.pieces2tensor import prolog_strings_to_tensor
from experiment_helper import task_set2zendodataset
from experiments.run_experiment import gather_data
import random
import re
from program import strip_trailing_var0
import torch
import time
from collections import Counter
import math

PREDICATE_TO_IDX_VAL = {
    "IS_RED":       (1, 0),
    "IS_BLUE":      (1, 1),
    "IS_YELLOW":    (1, 2),
    "IS_BLOCK":     (2, 0),
    "IS_WEDGE":     (2, 1),
    "IS_PYRAMID":   (2, 2),
    "IS_UPRIGHT":   (3, 0),
    "IS_UPSIDE_DOWN": (3, 1),
    "IS_FLAT":        (3, 2),
    "IS_CHEESECAKE":  (3, 3),
    "IS_HORIZONTAL":  (3, 2),
    "IS_VERTICAL":    (3, 0),
}

AMOUNT_PREDICATES = ["EVEN", "ODD", "EITHER_OR"]

def extract_predicates(program_str):
    preds = [word.rstrip(')') for word in program_str.split() if word.startswith("IS_")]
    if preds:
        return preds
    # If no IS_ predicates, extract the first word after '('
    match = re.search(r'\(\s*(\w+)', program_str)
    if match:
        return [match.group(1)]
    return []

def parse_either_or_args(rule_str: str):
    """
    Extracts the two integer values from an EITHER_OR rule like:
    (EITHER_OR 2 3 var0)
    Returns (2, 3)
    """
    print("Parsing EITHER_OR rule:", rule_str)
    match = re.search(r'\(EITHER_OR\s+(\d+)\s+(\d+)', rule_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def call_prolog_subprocess_with_retries(n, query, prolog_file, retries=10, delay=2):
    """
    Calls the Prolog subprocess to generate a scene, with retry mechanism on failure.

    :param n: Number of examples to generate
    :param query: Prolog query string
    :param prolog_file: Path to the Prolog file
    :param retries: Number of retry attempts
    :param delay: Delay between retries in seconds
    :return: JSON-parsed result or None
    """
    for attempt in range(retries):
        try:
            abs_path = Path(prolog_file).resolve().as_posix()
            result = subprocess.check_output(
                ['python3', 'call_generate_prolog.py', str(n), query, abs_path],
                timeout=6,
                stderr=subprocess.STDOUT  # capture stderr too
            )
            return json.loads(result)
        except subprocess.TimeoutExpired:
            print(f"Timeout on attempt {attempt + 1}/{retries}")
        except subprocess.CalledProcessError as e:
            print(f"Subprocess failed on attempt {attempt + 1}/{retries}:\n", e.output.decode())
        except json.JSONDecodeError as e:
            print(f"JSON decode failed on attempt {attempt + 1}/{retries}:", e)
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}/{retries}:", e)

        if attempt < retries - 1:
            time.sleep(delay)

    print("❌ All retry attempts failed.", query)
    return None

def call_prolog_subprocess(n, query, prolog_file):
    try:
        abs_path = Path(prolog_file).resolve().as_posix()
        result = subprocess.check_output(
            ['python3', 'call_generate_prolog.py', str(n), query, abs_path],
            timeout=6,
            stderr=subprocess.STDOUT  # capture stderr too
        )
        return json.loads(result)
    except subprocess.TimeoutExpired:
        print("Timeout: Prolog query took too long.")
    except subprocess.CalledProcessError as e:
        print("Subprocess failed. Output:\n", e.output.decode())
    except json.JSONDecodeError as e:
        print("JSON decode failed:", e)
    except Exception as e:
        print("Unexpected error:", e)
    return None

class ZendoPlayerInterface:
    def __init__(self, player_id, cfg, dsl, model=None):
        self.player_id = player_id

    def observe(self, example): ...
    def guess_rule(self): ...
    def guess_label(self, input_scene): ...
    def propose_input(self): ...
    def quiz_correct(self): ...

class ZendoPlayer:
    def __init__(self, player_id, model, dsl, cfg, bar=5e-7):
        self.id = player_id
        self.examples = []  # List[(tensor, label)]
        self.model = model
        self.dsl = dsl
        self.cfg = cfg
        self.pad_values = [7, 3, 3, 4, 7, 7, 7, 7, 7, 7, 7, -1, -1, -1, -1]
        self.guessing_stones = 0
        self.bar = bar  # Threshold for rule probability to consider it valid

    def observe(self, example):
        self.examples.append(example)

    def quiz_correct(self):
        self.guessing_stones += 1

    def guess_label(self, input_scene):
        dataset = task_set2zendodataset([["", self.examples]], self.model, self.dsl, self.cfg, use_model=True)
        data = gather_data(dataset, 0, True)
        top_rule = data[0][1][0][0]  # use top rule
        try:
            top_rule = strip_trailing_var0(top_rule)
            prog_fn = top_rule.eval(dsl=self.dsl, environment=(None, None), i=0)
            return prog_fn(input_scene)
        except:
            return False
        
    def decide_guess(self, state):
        if self.guessing_stones <= 0:
            return None
        rule = self.guess_rule()
        if rule is None:
            print(f"Player {self.id} could not find a rule")
            return None
        self.guessing_stones -= 1
        print(f"Player {self.id} guessed rule: {rule}")
        return {"type": "guess_rule", "rule": rule}

    def guess_rule(self):
        dataset = task_set2zendodataset([["", self.examples]], self.model, self.dsl, self.cfg, use_model=True)
        data = gather_data(dataset, 0, True)
        for program, _, _, _, _, prob in data[0][1]:
            if prob > self.bar:
                return program
        return None
    
    def react(self, state):
        # Only called during PROPOSE phase
        proposed_input, label, confidence = self.propose_input()
        if proposed_input is None:
            print("❌ Failed to propose input, returning None.")
            return {"type": "propose_input", "input": None, "mode": "TELL"}
        mode = "QUIZ" if confidence >= self.bar else "TELL"
        return {"type": "propose_input", "input": proposed_input, "mode": mode}

    def propose_input(self):
        print(f"Proposing input based on {len(self.examples)} current examples...")

        # === Step 1: Rule selection ===
        dataset = task_set2zendodataset([["", self.examples]], self.model, self.dsl, self.cfg, use_model=True)
        data = gather_data(dataset, 0)
        candidates = data[0][1]
        print(f"Found {candidates}")

        valid_candidates = [(prog, prob) for prog, *_, prob in candidates]
        if not valid_candidates:
            print("⚠️ All candidate rules are in wrong_rules.")
            return None, None, None

        top_rule, top_prob = valid_candidates[0]
        second_prob = valid_candidates[1][1] if len(valid_candidates) > 1 else 0.0

        propose_label = top_prob > 5e-9

        # === Step 2: One-candidate shortcut ===
        if len(valid_candidates) == 1:
            print("Only one valid candidate. Trying both valid and invalid scenes...")
            inner_query = dsl_to_prolog(top_rule)
            for validity, label in [("invalid", False), ("valid", True)]:
                prolog_str = f"generate_{validity}_structure([{inner_query}], Structure)"
                scene = call_prolog_subprocess_with_retries(1, prolog_str, "rules/rules.pl")[0]
                if scene is not None:
                    try:
                        new_input = prolog_strings_to_tensor([scene])[0]
                        return new_input, label, top_prob
                    except Exception as e:
                        print(f"❌ Failed to convert Prolog {validity} scene to tensor:", e)
                        return None, None, top_prob
            print("❌ Failed to generate both valid and invalid scenes.")
            return None, None, top_prob

        # === Step 3: Generate based on confidence bias ===
        confidence = top_prob - second_prob
        prob_valid = 1 / (1 + math.exp(-50 * confidence))  # sigmoid for soft bias
        prefer_valid = random.random() < prob_valid
        validity_order = ["valid", "invalid"] if prefer_valid else ["invalid", "valid"]

        print(f"🎲 Initial preference: '{validity_order[0]}' (bias: {prob_valid:.2f})")

        # === Step 4: Try both validities to find discriminating input ===
        inner_query = dsl_to_prolog(top_rule)

        for validity in validity_order:
            print(f"\n🔍 Trying to generate a '{validity}' scene...")
            try:
                for _ in range(30):
                    prolog_str = f"generate_{validity}_structure([{inner_query}], Structure)"
                    scene = call_prolog_subprocess_with_retries(1, prolog_str, "rules/rules.pl")[0]
                    if scene is None:
                        print("❌ Prolog returned None for scene generation.")
                        continue

                    try:
                        new_input = prolog_strings_to_tensor([scene])[0]
                    except Exception as e:
                        print("❌ Failed to convert scene:", e)
                        return None, None, top_prob

                    eval_results = []
                    for prog, _ in valid_candidates:
                        try:
                            strip_trailing_var0(prog)
                            prog_fn = prog.eval(dsl=self.dsl, environment=(None, None), i=0)
                            eval_results.append(prog_fn(new_input))
                        except Exception as e:
                            print("⚠️ Evaluation error:", e)
                            eval_results.append(False)

                    counts = Counter(eval_results)
                    _, most_common_count = counts.most_common(1)[0]
                    num_disagreeing = len(eval_results) - most_common_count

                    if len(valid_candidates) > 3 and num_disagreeing >= len(valid_candidates) // 2 - 1:
                        print(f"✅ Discriminating input found: {num_disagreeing} disagreements out of {len(valid_candidates)}")
                        return new_input, (validity == "valid" if propose_label else None), top_prob

                    elif len(valid_candidates) <= 3 and num_disagreeing >= 1:
                        print("✅ Input distinguishes among small candidate set.")
                        return new_input, (validity == "valid" if propose_label else None), top_prob

            except Exception as e:
                print("❌ Exception during input generation:", e)
                return None, None, top_prob

        print("⚠️ No discriminating input found from either validity. Falling back...")
        return None, None, top_prob
    
COLOR_IDX = 1
SHAPE_IDX = 2
ORIENT_IDX = 3
max_values = {
    COLOR_IDX: 3,  # red, blue, yellow
    SHAPE_IDX: 3,  # block, wedge, pyramid
    ORIENT_IDX: 4,  # upright, upside_down, flat, cheesecake
}

def random_piece_like(piece: torch.Tensor) -> torch.Tensor:
    """Return a modified copy of `piece` with one attribute changed to a different valid value."""
    attr_idx = random.choice([COLOR_IDX, SHAPE_IDX, ORIENT_IDX])
    current_val = int(piece[attr_idx].item())
    candidates = [v for v in range(max_values[attr_idx]) if v != current_val]
    new_val = random.choice(candidates)
    
    new_piece = piece.clone()
    new_piece[attr_idx] = new_val
    return new_piece

class HeuristicZendoPlayer(ZendoPlayer):
    PAD_VALS = torch.tensor([7, 3, 3, 4, 7, 7, 7, 7, 7, 7, 7, -1, -1, -1, -1], dtype=torch.int64)
    def is_padding(self, piece):
        return torch.all(piece == self.PAD_VALS)
    
    def non_padded_indices(self, structure):
        return [i for i, p in enumerate(structure) if not self.is_padding(p)]
    
    def react(self, state):
        # Only called during PROPOSE phase
        proposed_input = self.propose_input()
        if proposed_input is None:
            print("❌ Failed to propose input, returning None.")
            return {"type": "propose_input", "input": None, "mode": "TELL"}
        dataset = task_set2zendodataset([["", self.examples]], self.model, self.dsl, self.cfg, use_model=True)
        data = gather_data(dataset, 0, True)
        *_, prob = data[0][1][0]
        mode = "QUIZ" if prob >= self.bar else "TELL"
        return {"type": "propose_input", "input": proposed_input, "mode": mode}
    
    def propose_input(self):
        print(f"Proposing input based on {len(self.examples)} current examples...")
        base_structure, _ = random.choice(self.examples)
        print(f"Mutating structure: {base_structure}")
        mutation = random.choice(self.heuristics)
        print(f"Selected mutation: {mutation.__name__}")
        return mutation(base_structure)
   
    def reduce_by_one(self, structure):
        print("Reducing structure by one piece...")
        structure = structure.clone()
        print(self.non_padded_indices(structure))
        indices = self.non_padded_indices(structure)
        if len(indices) <= 1:
            return structure
        idx_to_remove = random.choice(indices)
        structure = torch.cat([structure[:idx_to_remove], structure[idx_to_remove+1:], self.PAD_VALS.unsqueeze(0)], dim=0)
        return structure

    def substitute_one_piece(self, structure):
        print("Substituting one piece in the structure...")
        structure = structure.clone()
        print(self.non_padded_indices(structure))
        indices = self.non_padded_indices(structure)
        if not indices:
            return structure
        i = random.choice(indices)
        print(f"Substituting piece at index {i}", structure[i])
        new_piece = self.random_piece_like(structure[i])
        structure[i] = new_piece
        return structure

    def homogenize_attribute(self, structure):
        print("Homogenizing one attribute in the structure...")
        structure = structure.clone()
        print(self.non_padded_indices(structure))
        indices = self.non_padded_indices(structure)
        if not indices:
            return structure
        attr_idx = random.choice([COLOR_IDX, SHAPE_IDX, ORIENT_IDX])
        val = random.choice([int(structure[i][attr_idx].item()) for i in indices])
        for i in indices:
            structure[i][attr_idx] = val
        return structure
    
    def single_piece_structure(self, structure):
        print("Creating single piece structure...")
        structure = structure.clone()
        print(self.non_padded_indices(structure))
        indices = self.non_padded_indices(structure)
        if not indices:
            return structure  # fallback: return original if all are padding

        i = random.choice(indices)
        selected_piece = structure[i].clone()

        padding = self.PAD_VALS.unsqueeze(0).repeat(6, 1)
        new_structure = torch.cat([selected_piece.unsqueeze(0), padding], dim=0)
        return new_structure
    
    def spread_structure(self, structure):
        print("Spreading structure...")
        structure = structure.clone()
        print(self.non_padded_indices(structure))
        indices = self.non_padded_indices(structure)
        for i in indices:
            structure[i][4:11] = 8  # indices 4 to 10 inclusive
        return structure

    def random_piece_like(self, piece: torch.Tensor) -> torch.Tensor:
        print("Generating random piece like:", piece)
        attr_idx = random.choice([COLOR_IDX, SHAPE_IDX, ORIENT_IDX])
        current_val = int(piece[attr_idx].item())
        max_values = {
            COLOR_IDX: 3,
            SHAPE_IDX: 3,
            ORIENT_IDX: 4
        }
        print(f"Current value for attribute {attr_idx}: {current_val}")
        candidates = [v for v in range(max_values[attr_idx]) if v != current_val]
        print("candidates:", candidates)
        new_val = random.choice(candidates)
        new_piece = piece.clone()
        new_piece[attr_idx] = new_val
        return new_piece

    @property
    def heuristics(self):
        return [
            self.reduce_by_one,
            self.substitute_one_piece,
            self.homogenize_attribute,
            self.single_piece_structure,
            self.spread_structure
        ]