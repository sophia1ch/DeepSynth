from data.create_programs import convert_prolog_to_dsl
from data.create_prolog import dsl_to_prolog
from data.pieces2tensor import prolog_strings_to_tensor
from experiments.run_experiment import canonicalize_program, normalize_program_structure
from program import Program, strip_trailing_var0
import torch
import random

from zendo.player import call_prolog_subprocess_with_retries

class GameMaster:
    def __init__(self, true_program: Program, dataset, zendo_dsl, cfg):
        self.remaining_examples = dataset
        print(true_program)
        self.true_program = true_program
        self.dsl = zendo_dsl
        self.cfg = cfg

    def initial_examples(self):
        positives = [ex for ex in self.remaining_examples if ex[1] is True]
        negatives = [ex for ex in self.remaining_examples if ex[1] is False]

        if not positives or not negatives:
            raise ValueError("Not enough positive and negative examples to start.")

        pos_example = random.choice(positives)
        neg_example = random.choice(negatives)

        def safe_remove(target):
            for i, (tensor, label) in enumerate(self.remaining_examples):
                if torch.equal(tensor, target[0]) and label == target[1]:
                    del self.remaining_examples[i]
                    return

        safe_remove(pos_example)
        safe_remove(neg_example)

        return [pos_example, neg_example]

    def get_next_example(self):
        print("Getting next example from remaining examples.", len(self.remaining_examples))
        if self.remaining_examples:
            next_ex = self.remaining_examples.pop(0)
            return next_ex
        else:
            return None

    def label_input(self, tensor):
        try:
            strip_trailing_var0(self.true_program)
            program = self.true_program.eval(
                dsl=self.dsl,
                environment=(tensor, None),
                i=0  # Adjust this index if needed
            )
            return program(tensor)
        except Exception as e:
            raise ValueError(f"Failed to evaluate input: {e}")
        
    def disprove_guess(self, guess):
        """
        Disprove a guess by returning an example which follow the guess but not the true program (1),
        or returning an example which follows the true program but not the guess (2).
        """
        for i, (tensor, _) in enumerate(self.remaining_examples):
            try:
                # Evaluate true program
                strip_trailing_var0(guess)
                strip_trailing_var0(self.true_program)
                true_val = self.true_program.eval(dsl=self.dsl, environment=(tensor, None), i=i)
                true_label = true_val(tensor)

                # Evaluate guessed program
                guess_val = guess.eval(dsl=self.dsl, environment=(tensor, None), i=i)
                guess_label = guess_val(tensor)

                # Disagreement case 1: guess is True, true is False
                if guess_label and not true_label:
                    return self.remaining_examples.pop(i)

                # Disagreement case 2: true is True, guess is False
                if not guess_label and true_label:
                    return self.remaining_examples.pop(i)

            except Exception as e:
                print(f"ERROR: Skipping example due to evaluation error: {e}")
                continue

        print("Guess could not be disproven with remaining examples.")
        return None  # No disproof found
    
    def disprove_guess_via_prolog(self, guess_program):

        true_prolog = dsl_to_prolog(self.true_program)
        guess_prolog = dsl_to_prolog(guess_program)

        true_query = f"generate_valid_structure([{true_prolog}], Structure)"
        guess_query = f"generate_valid_structure([{guess_prolog}], Structure)"

        # Try to find example accepted by guess but rejected by true_program
        print("Try to find example accepted by guess but rejected by true_program")
        strip_trailing_var0(guess_program)
        strip_trailing_var0(self.true_program)
        for _ in range(20):
            scene = call_prolog_subprocess_with_retries(1, guess_query, "rules/rules.pl")[0]
            if scene is not None:
                guess_input = prolog_strings_to_tensor([scene])[0]
                try:
                    out_true = self.true_program.eval(dsl=self.dsl, environment=(guess_input, None), i=0)(guess_input)
                    if not out_true:
                        return (guess_input, False)
                except:
                    continue
        # Try to find example accepted by true_program but rejected by guessed program
        print("Try to find example accepted by true_program but rejected by guessed program")
        for _ in range(40):
            scene = call_prolog_subprocess_with_retries(1, true_query, "rules/rules.pl")[0]
            if scene is not None:
                true_input = prolog_strings_to_tensor([scene])[0]
                try:
                    out_guess = guess_program.eval(dsl=self.dsl, environment=(true_input, None), i=0)(true_input)
                    if not out_guess:
                        return (true_input, True)
                except:
                    continue

        return None

class ZendoStateGameMaster:
    def __init__(self, true_program: Program, dataset, zendo_dsl, cfg):
        self.remaining_examples = dataset
        self.true_program = true_program
        self.dsl = zendo_dsl
        self.cfg = cfg

    def initial_examples(self):
        positives = [ex for ex in self.remaining_examples if ex[1] is True]
        negatives = [ex for ex in self.remaining_examples if ex[1] is False]

        if not positives or not negatives:
            raise ValueError("Not enough positive and negative examples to start.")

        pos_example = random.choice(positives)
        neg_example = random.choice(negatives)

        def safe_remove(target):
            for i, (tensor, label) in enumerate(self.remaining_examples):
                if torch.equal(tensor, target[0]) and label == target[1]:
                    del self.remaining_examples[i]
                    return

        safe_remove(pos_example)
        safe_remove(neg_example)

        return [pos_example, neg_example]
    
    def get_next_example(self):
        print("Getting next example from remaining examples.", len(self.remaining_examples))
        if self.remaining_examples:
            next_ex = self.remaining_examples.pop(0)
            return next_ex
        else:
            return None

    def label_input(self, tensor):
        try:
            strip_trailing_var0(self.true_program)
            program = self.true_program.eval(
                dsl=self.dsl,
                environment=(tensor, None),
                i=0
            )
            return program(tensor)
        except Exception as e:
            raise ValueError(f"Failed to evaluate input: {e}")
    
    def check_guess(self, guess):
        strip_trailing_var0(guess)
        strip_trailing_var0(self.true_program)
        norm_true_program = normalize_program_structure(self.true_program)
        canonical_true_program = canonicalize_program(norm_true_program)
        norm_guess = normalize_program_structure(guess)
        canonical_guess = canonicalize_program(norm_guess)
        print(f"Checking guess: {canonical_guess} against true program: {canonical_true_program}")
        return str(canonical_guess) == str(canonical_true_program)

    def react(self, state, player_action):
        if player_action["type"] == "propose_input":
            input_scene = player_action["input"]
            if input_scene is None:
                return {"type": "label", "label": None}
            label = self.label_input(input_scene)
            return {"type": "label", "label": label}

        elif player_action["type"] == "guess_rule":
            rule = player_action["rule"]
            is_correct = str(strip_trailing_var0(rule)) == str(strip_trailing_var0(self.true_program))
            if is_correct:
                return {"type": "guess_feedback", "correct": True}

            counter_example = self.disprove_guess(rule)
            if not counter_example:
                counter_example = self.disprove_guess_via_prolog(rule)
            return {"type": "guess_feedback", "correct": False, "counter_example": counter_example}

        return {"type": "noop"}

    def disprove_guess(self, guess):
        for i, (tensor, _) in enumerate(self.remaining_examples):
            try:
                strip_trailing_var0(guess)
                strip_trailing_var0(self.true_program)
                true_val = self.true_program.eval(dsl=self.dsl, environment=(tensor, None), i=i)
                true_label = true_val(tensor)

                guess_val = guess.eval(dsl=self.dsl, environment=(tensor, None), i=i)
                guess_label = guess_val(tensor)

                if guess_label and not true_label:
                    return self.remaining_examples.pop(i)

                if not guess_label and true_label:
                    return self.remaining_examples.pop(i)

            except Exception as e:
                print(f"ERROR: Skipping example due to evaluation error: {e}")
                continue

        print("Guess could not be disproven with remaining examples.")
        return None

    def disprove_guess_via_prolog(self, guess_program):
        true_prolog = dsl_to_prolog(self.true_program)
        guess_prolog = dsl_to_prolog(guess_program)

        true_query = f"generate_valid_structure([{true_prolog}], Structure)"
        guess_query = f"generate_valid_structure([{guess_prolog}], Structure)"

        print("Try to find example accepted by guess but rejected by true_program")
        strip_trailing_var0(guess_program)
        strip_trailing_var0(self.true_program)
        for _ in range(20):
            scene = call_prolog_subprocess_with_retries(1, guess_query, "rules/rules.pl")[0]
            if scene is not None:
                guess_input = prolog_strings_to_tensor([scene])[0]
                try:
                    out_true = self.true_program.eval(dsl=self.dsl, environment=(guess_input, None), i=0)(guess_input)
                    if not out_true:
                        return (guess_input, False)
                except:
                    continue

        print("Try to find example accepted by true_program but rejected by guessed program")
        for _ in range(40):
            scene = call_prolog_subprocess_with_retries(1, true_query, "rules/rules.pl")[0]
            if scene is not None:
                true_input = prolog_strings_to_tensor([scene])[0]
                try:
                    out_guess = guess_program.eval(dsl=self.dsl, environment=(true_input, None), i=0)(true_input)
                    if not out_guess:
                        return (true_input, True)
                except:
                    continue

        return None
