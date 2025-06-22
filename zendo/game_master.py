from data.create_programs import convert_prolog_to_dsl
from program import Program
import torch
import random

class GameMaster:
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
        print("🧩 Getting next example from remaining examples.", len(self.remaining_examples))
        if self.remaining_examples:
            next_ex = self.remaining_examples.pop(0)
            return next_ex
        else:
            return None

    def label_input(self, tensor):
        try:
            program = self.true_program.eval(
                dsl=self.dsl,
                environment=(tensor, None),
                i=0  # Adjust this index if needed
            )
            return program(tensor)
        except Exception as e:
            raise ValueError(f"Failed to evaluate input: {e}")
