from experiments.run_experiment import canonicalize_program, normalize_program_structure
from program import strip_trailing_var0


def play_game(gm, player, return_guesses=False):
    # Start with 2 examples
    guesses = []
    initial_examples = gm.initial_examples()
    print("1: Gamemaster provided initial examples")
    for example in initial_examples:
        player.observe(example)
    won = False
    norm_true_program = normalize_program_structure(gm.true_program)
    canonical_true_program = canonicalize_program(norm_true_program)
    # Interaction rounds
    while len(player.examples) < 30:
        print("---------New Round---------")
        proposed_input, proposed_label = player.propose_input()
        if proposed_input is None:
            print("WARNING: Player couldn't generate a new example.")
            next_example = gm.get_next_example()
            if next_example:
                print("FIX: Gamemaster provides new example.")
                player.observe(next_example)
            continue
        print(f"2: Player proposed input and label")
        label = gm.label_input(proposed_input)
        print(f"3: Gamemaster says: {label}")
        if label == proposed_label:
            print("3a: Player's proposed label is correct.")
            player.quiz_correct()
        player.observe((proposed_input, label))
        guessed_rule = player.guess_rule()
        guessed_rule = strip_trailing_var0(guessed_rule)
        guesses.append(guessed_rule)
        norm_guess = normalize_program_structure(guessed_rule)
        canonical_guess = canonicalize_program(norm_guess)
        if str(canonical_guess) == str(canonical_true_program):
            print("4: Player guessed the rule correctly!")
            won = True
            break
        else:
            print(f"4: Player's guess was incorrect: {guessed_rule}, \nCorrect rule: {gm.true_program}")
            if guessed_rule is not None:
                player.wrong_guess(guessed_rule)
                example = gm.disprove_guess(guessed_rule)
                if example:
                    print("4a: Gamemaster disproved the guess with an example.")
                    player.observe(example)
                    continue
                else:
                    print("4b: Gamemaster is generating new input to disrove guess.")
                    example = gm.disprove_guess_via_prolog(guessed_rule)
                    if example:
                        print("4c: Gamemaster disproved the guess with a Prolog example.")
                        player.observe(example)
                        continue
                    else:
                        print("4d: No disproof found, player won!")
                        won = True
                        break
            
        next_example = gm.get_next_example()
        if next_example:
            print("5: Gamemaster provides new example.")
            player.observe(next_example)

    # Final rule guess
    if not won:
        print("Player did not guess the rule within 30 examples.")
    else:
        print(f"Player won the game with {len(player.examples)} examples!")
    if return_guesses:
        return guesses, won

