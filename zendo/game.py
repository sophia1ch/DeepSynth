def play_game(gm, player):
    # Start with 2 examples
    initial_examples = gm.initial_examples()
    for example in initial_examples:
        player.observe(example)

    # Interaction rounds
    while len(player.examples) < 20:
        proposed_input, proposed_label = player.propose_input()
        if proposed_input is None:
            print("❌ Player couldn't generate a new example.")
            continue
        print(f"🤖 Player proposed input and label")
        label = gm.label_input(proposed_input)
        print(f"🎲 Player proposed structure → Gamemaster says: {label}")
        if label == proposed_label:
            print("✅ Player's proposed label is correct.")
            player.quiz_correct()
        player.observe((proposed_input, label))
        guessed_rule = player.guess_rule()
        if str(guessed_rule) == str(gm.true_program):
            print("🎉 Player guessed the rule correctly!")
            break
        else:
            print(f"❌ Player's guess was incorrect: {guessed_rule}")
            if guessed_rule is not None:
                player.wrong_guess(guessed_rule)
            
        next_example = gm.get_next_example()
        if next_example:
            print("🎁 Gamemaster provides new example.")
            player.observe(next_example)

    # Final rule guess
    guess = player.guess_rule()
    print(f"🏁 Player's final rule guess:\n{guess} \nCorrect rule:\n{gm.true_program}")

