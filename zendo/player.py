import csv
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
import sys
import time

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
                [sys.executable, 'call_generate_prolog.py', str(n), query, abs_path],
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
            [sys.executable, 'call_generate_prolog.py', str(n), query, abs_path],
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

class ZendoPlayer:
    def __init__(self, model, dsl, cfg):
        self.examples = []  # List[(tensor, label)]
        self.model = model
        self.dsl = dsl
        self.cfg = cfg
        self.pad_values = [7, 3, 3, 4, 7, 7, 7, 7, 7, 7, 7, -1, -1, -1, -1]
        self.guessing_stones = 0
        self.wrong_rules = []

    def observe(self, example):
        self.examples.append(example)

    def quiz_correct(self):
        """
        This method is used to indicate that the player has correctly guessed the label of an example
        """
        self.guessing_stones += 1
        print(f"✅ Player guessed correctly! Total correct guesses: {self.guessing_stones}")

    def guess_rule(self):
        if self.guessing_stones <= 1:
            print("Not allowed to guess rule yet")
            return None

        dataset = task_set2zendodataset([["", self.examples]], self.model, self.dsl, self.cfg, use_model=True)
        data = gather_data(dataset, 0, True)
        self.guessing_stones -= 1

        for program, *_ in data[0][1]:
            if str(program) not in self.wrong_rules:
                print("Guessing rule:", program)
                return program

        print("All candidates have already been guessed.")
        return None

    def wrong_guess(self, rule):
        """
        This method is called when the player guesses a rule that is incorrect.
        It adds the guessed rule to the list of wrong rules.
        """
        self.wrong_rules.append(str(rule))
        print(f"❌ Player guessed wrong! Added rule to wrong guesses: {rule}")

    def propose_input(self):
        print("Proposing input based on current examples...", len(self.examples), "examples")

        dataset = task_set2zendodataset([["", self.examples]], self.model, self.dsl, self.cfg, use_model=True)
        data = gather_data(dataset, 0)
        candidates = data[0][1]
        valid_candidates = [(prog, prob) for prog, *_, prob in candidates if str(prog) not in self.wrong_rules]

        if not valid_candidates:
            print("All candidate rules are in wrong_rules.")
            return None, None

        top_rule, top_prob = valid_candidates[0]
        second_prob = valid_candidates[1][1] if len(valid_candidates) > 1 else 0.0
        propose_label = False
        if top_prob > 1e-6:
            propose_label = True

        inner_query = dsl_to_prolog(top_rule)
        prolog_str = f"generate_valid_structure([{inner_query}], Structure)"

        try:
            json_scene = call_prolog_subprocess_with_retries(10, prolog_str, "rules/rules.pl")
        except Exception as e:
            print("Failed to generate scene from Prolog query:", e)
            return None, None
        if json_scene is None:
            top_rule, second_prob = valid_candidates[1]
            if second_prob > 1e-6:
                propose_label = True
            else:
                propose_label = False
            inner_query = dsl_to_prolog(top_rule)
            prolog_str = f"generate_valid_structure([{inner_query}], Structure)"
            try:
                json_scene = call_prolog_subprocess_with_retries(10, prolog_str, "rules/rules.pl")
            except Exception as e:
                print("Failed to generate scene from Prolog query:", e)
                return None, None
        try:
            new_inputs = prolog_strings_to_tensor(json_scene)
        except Exception as e:
            print("Failed to convert Prolog scene to tensor:", e)
            return None, None

        # Evaluate input on top 3 rules
        for new_input in new_inputs:
            eval_results = []
            for prog, _ in valid_candidates[:6]:
                try:
                    strip_trailing_var0(prog)
                    prog_fn = prog.eval(dsl=self.dsl, environment=(None, None), i=0)  # only build the lambda
                    out = prog_fn(new_input)
                    eval_results.append(out)
                except Exception as e:
                    print("Evaluation error:", e)
                    eval_results.append(False)

            # print("Evaluation results:", eval_results)
            if len(set(eval_results)) > 1:
                print("New input discriminates between top rules!")
                if propose_label:
                    return new_input, True
                return new_input, None

        print("New input did not discriminate. Trying next fallback strategy.")
        return None, None


    def propose_input_old(self):
        print("Proposing input based on current examples...", len(self.examples), "examples")
        
        dataset = task_set2zendodataset([["", self.examples]], self.model, self.dsl, self.cfg, use_model=True)
        data = gather_data(dataset, 0)
        candidates = data[0][1]
        valid_candidates = [(prog, prob) for prog, *_ , prob in candidates if str(prog) not in self.wrong_rules]

        if not valid_candidates:
            print("All candidate rules are in wrong_rules.")
            return None, None

        top_rule, top_prob = valid_candidates[0]
        second_prob = valid_candidates[1][1] if len(valid_candidates) > 1 else 0.0

        preds = extract_predicates(str(top_rule))
        
        ratio = top_prob / (second_prob + 1e-9)
        print("Confidence ratio:", ratio)
        if ratio > 10:
            # Strategy: Verification
            for pred in preds:
                ex = self.generate_verifying_input(self.examples, pred, top_rule)
                if ex is not None:
                    return ex, True

        elif ratio < 1.1:
            # Strategy: Disambiguation
            return self.find_most_disruptive_input(top_rule, top_prob, preds, False)

        else:
            # Strategy: Mix
            if random.random() < 0.5:
                return self.find_most_disruptive_input(top_rule, top_prob, preds, True)
            else:
                for pred in preds:
                    ex = self.generate_verifying_input(self.examples, pred, top_rule)
                    if ex is not None:
                        return ex, True

        return None
    
    def find_discriminative_input(self, rules: list, dsl, num_trials=100):
        """
        Generate random inputs and return the one that best splits the rule evaluations (ideally 3 True / 3 False).
        :param rules: List of top candidate programs (Program instances)
        :param dsl: The DSL object
        :param num_trials: Number of random inputs to sample
        :return: A tensor [7, 15] that best splits the rules
        """
        best_input = None
        best_score = -1

        for _ in range(num_trials):
            scene_tensor = self._generate_random_input_tensor()

            # Evaluate the input on all rules
            results = []
            for i, rule in enumerate(rules):
                try:
                    out = rule.eval(dsl=dsl, environment=(scene_tensor, None), i=i)
                    results.append(bool(out))
                except Exception as e:
                    print(f"Error evaluating rule {rule} on input {scene_tensor}: {e}")
                    results.append(False)  # fallback if evaluation fails

            # Score = number of rules where majority = 3-3 or 2-4 or 4-2 (max disagreement)
            true_count = sum(results)
            false_count = len(results) - true_count
            score = min(true_count, false_count)

            if score > best_score:
                best_score = score
                best_input = scene_tensor

        return best_input
    
    
    def find_most_disruptive_input(self, top_rule, top_prob, preds, output_label):
        if output_label:
            label = False
        else:
            label = None
        print("Strategy: Disambiguate competing rules")
        best_input = None
        best_diff = -1

        for pred in preds:
            for _ in range(3):  # try 5 perturbations per predicate
                candidate = self.generate_perturbed_input(self.examples, pred, top_rule)
                if candidate is None:
                    print(f"Could not generate perturbation for predicate {pred}")
                    continue

                # Simulate adding it as a negative example
                new_examples = self.examples + [(candidate, False)]
                dataset = task_set2zendodataset([["", new_examples]], self.model, self.dsl, self.cfg, use_model=True)
                data = gather_data(dataset, 0)

                # Look for top_rule again and check its new probability
                for rule, prob, *_ in data[0][1]:
                    if rule == top_rule:
                        diff = top_prob - prob
                        print(f"Candidate reduced top rule probability from {top_prob:.4e} to {prob:.4e}")
                        if diff > best_diff:
                            best_diff = diff
                            best_input = candidate
                        break

        if best_input is not None:
            print("Most disruptive input found with prob drop:", best_diff)
        else:
            print("No useful perturbation found.")

        return best_input, label

    def generate_verifying_input(self, task_data, pred, rule):
        print("Strategy: Verify top rule")
        if pred in AMOUNT_PREDICATES:
            if pred == "EVEN":
                target_count = 2
            elif pred == "ODD":
                target_count = 3
            elif pred == "EITHER_OR":
                n1, n2 = parse_either_or_args(str(rule))
                target_count = random.choice([n1, n2])
            else:
                return None
            
            negative_examples = [(t, l) for t, l in task_data if l is False]
            if not negative_examples:
                print("No suitable negative example to extend")
                return None

            for ex in negative_examples:
                tensor, _ = ex
                count = self._count_valid_pieces(tensor)
                if count == target_count:
                    print(f"Example already satisfies target count {target_count}, skipping")
                    continue  # Already satisfies amount

                # Pad additional valid vectors to reach target_count
                needed = target_count - count
                tensor_list = list(tensor)
                tensor_clone = tensor.clone()
                valid_indices = [i for i, row in enumerate(tensor_list) if row[0].item() != 7]
                if needed < 0:
                    to_remove = count - target_count
                    print(f"Padding over {to_remove} valid pieces to reduce count")

                    padded = torch.tensor(self.pad_values, dtype=tensor.dtype)
                    removed = 0
                    for i in valid_indices:
                        tensor_clone[i] = padded
                        removed += 1
                        if removed == to_remove:
                            return tensor_clone
                # Add valid (non-padding) pieces
                valid_piece = None
                for row in tensor_list:
                    if row[0].item() != 7:
                        valid_piece = row.clone()
                        break

                if valid_piece is None:
                    print("Could not find a valid piece to clone.")
                    continue
                valid_pieces = [row for row in tensor_list if row[0].item() != 7]
                pad_indices = [i for i, row in enumerate(tensor_list) if row[0].item() == 7]
                last_valid_id = valid_pieces[-1][0].item()

                for j in range(needed):
                    print(f"Adding valid piece {j+1}/{needed} to tensor")
                    idx = pad_indices[j]
                    new_id = last_valid_id + j + 1
                    tensor_clone[idx] = torch.tensor([
                        new_id,
                        random.randint(0, 2),
                        random.randint(0, 2),
                        random.randint(0, 3),
                        8, 8, 8, 8, 8, 8,
                        8,
                        -1, -1, -1, -1
                    ], dtype=tensor.dtype)

                new_tensor = torch.stack(tensor_list)
                return new_tensor

            print("No unique verifying example could be generated.")
            return None

        positive_examples = [ex for ex in task_data if ex[1] is True]
        if not positive_examples:
            print("No positive examples available.")
            return None

        for tensor, _ in positive_examples:
            tensor_clone = tensor.clone()

            # Choose an unpadded object
            valid_indices = [i for i in range(tensor_clone.shape[0]) if tensor_clone[i, 0].item() != 7]
            if not valid_indices:
                continue

            chosen_idx = random.choice(valid_indices)

            # Determine which fields are irrelevant to the current predicate
            attr_idx, _ = PREDICATE_TO_IDX_VAL[pred]
            if attr_idx in [1, 2]:       # color or shape
                irrelevant_indices = [3]  # orientation
            elif attr_idx == 3:          # orientation
                irrelevant_indices = [1, 2]  # color and shape
            else:
                irrelevant_indices = [1, 2, 3]  # default to all categorical

            field_to_change = random.choice(irrelevant_indices)

            if field_to_change == 1 or field_to_change == 2:
                max_val = 3
            elif field_to_change == 3:
                max_val = 4
            else:
                continue

            current_val = tensor_clone[chosen_idx, field_to_change].item()
            new_val_options = [v for v in range(max_val) if v != current_val]
            if not new_val_options:
                continue

            new_val = random.choice(new_val_options)
            tensor_clone[chosen_idx, field_to_change] = new_val

            if not any(torch.equal(tensor_clone, e[0]) for e in task_data):
                return tensor_clone

        print("No unique attribute-based verifying input could be generated.")
        return None
    
    def generate_perturbed_input(self, task_data, pred, rule):
        """
        task_data: list of (input_tensor, output_bool) pairs
        pred: string like "IS_RED"
        Returns: updated list with one perturbed (input, False) pair appended
        """
        if pred in AMOUNT_PREDICATES:
            # Handle amount predicates separately
            if pred == "EVEN" or pred == "ODD":
                return self.handle_even_odd(task_data)
            elif pred == "EITHER_OR":
                n1, n2 = parse_either_or_args(str(rule))
            if n1 is not None and n2 is not None:
                return self.handle_either_or(task_data, n1, n2)
            else:
                print("Could not parse n1 and n2 from EITHER_OR rule:", rule)
                return None
        attr_idx, target_val = PREDICATE_TO_IDX_VAL.get(pred, (None, None))
        if attr_idx is None:
            return None
        positive_examples = [inp for inp, label in task_data if label is True]
        print(f"Positive examples count: {len(positive_examples)}")
        if not positive_examples:
            return None

        base_tensor = random.choice(positive_examples)
        base_tensor = base_tensor.clone()

        matches = [
            i for i in range(base_tensor.shape[0])
            if base_tensor[i, 0].item() != 7 and base_tensor[i, attr_idx].item() == target_val
        ]
        if not matches:
            return None
        if attr_idx in [1, 2]:
            max_value = 3
        elif attr_idx == 3:
            max_value = 4
        else:
            max_value = 7
        chosen = random.choice(matches)
        new_piece = base_tensor[chosen].clone()
        new_val = random.choice([v for v in range(max_value) if v != target_val])
        new_piece[attr_idx] = new_val

        modified_tensor = base_tensor.clone()
        modified_tensor[chosen] = new_piece
        return modified_tensor
    
    def handle_even_odd(self, task_data):
        best_example = max(task_data, key=lambda ex: self._count_valid_pieces(ex[0]))
        new_example = self._remove_one_valid_piece(best_example)
        return new_example

    def handle_either_or(self, task_data, n1, n2):
        # Try removal strategy first
        for ex in task_data:
            tensor, label = ex
            count = self._count_valid_pieces(tensor)

            if count in {n1, n2} and count > 1:
                modified_tensor = self._remove_one_valid_piece(ex)
                new_count = self._count_valid_pieces(modified_tensor)
                if new_count not in {n1, n2}:
                    return modified_tensor
        for ex in task_data:
            tensor, label = ex
            count = self._count_valid_pieces(tensor)

            if count >= tensor.shape[0]:
                continue

            if (count + 1) not in {n1, n2}:
                tensor_clone = tensor.clone()
                pad_indices = [i for i in range(tensor.shape[0]) if tensor[i, 0].item() == 7]

                if not pad_indices:
                    continue

                insert_idx = pad_indices[0]
                new_id = max((tensor[i, 0].item() for i in range(tensor.shape[0]) if tensor[i, 0].item() != 7), default=-1) + 1
                tensor_clone[insert_idx] = torch.tensor([
                    new_id,
                    random.randint(0, 2),
                    random.randint(0, 2),
                    random.randint(0, 3),
                    8, 8, 8, 8, 8, 8,
                    8,
                    -1, -1, -1, -1
                ], dtype=tensor.dtype)

                if self._count_valid_pieces(tensor_clone) not in {n1, n2}:
                    return tensor_clone

        print("No example could be perturbed to break EITHER_OR.")
        return None

    def _count_valid_pieces(self, tensor):
        count = sum(1 for piece in tensor if piece[0].item() != 7)
        print("Counting valid pieces in tensor:", count)
        return count

    def _remove_one_valid_piece(self, example):
        tensor, label = example
        tensor_list = list(tensor)

        # Count how many valid pieces there are
        valid_indices = [i for i, row in enumerate(tensor_list) if row[0].item() != 7]
        if len(valid_indices) <= 1:
            print("Skipping removal: only one or zero valid pieces.")
            return tensor  # Avoid creating fully padded example

        for i in valid_indices:
            padded = torch.tensor(self.pad_values, dtype=tensor.dtype)
            tensor_list[i] = padded
            return torch.stack(tensor_list)

        return tensor
