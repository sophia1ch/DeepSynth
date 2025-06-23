from data.create_programs import convert_prolog_to_dsl
from data.create_prolog import dsl_to_prolog
from data.pieces2tensor import prolog_strings_to_tensor
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
        guess_scenes = call_prolog_subprocess_with_retries(10, guess_query, "rules/rules.pl")
        guess_inputs = prolog_strings_to_tensor(guess_scenes)
        strip_trailing_var0(guess_program)
        strip_trailing_var0(self.true_program)

        for tensor in guess_inputs:
            try:
                out_true = self.true_program.eval(dsl=self.dsl, environment=(tensor, None), i=0)(tensor)
                if not out_true:
                    return (tensor, False)
            except:
                continue
        # Try to find example accepted by true_program but rejected by query
        true_scenes = call_prolog_subprocess_with_retries(10, true_query, "rules/rules.pl")
        true_inputs = prolog_strings_to_tensor(true_scenes)

        for tensor in true_inputs:
            try:
                out_guess = guess_program.eval(dsl=self.dsl, environment=(tensor, None), i=0)(tensor)
                if not out_guess:
                    return (tensor, True)
            except:
                continue

        return None
