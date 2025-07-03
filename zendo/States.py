from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
import json

from zendo.game_master import GameMaster
from zendo.player import ZendoPlayerInterface

class Turn(Enum):
      PROPOSE = auto()
      LABEL = auto()
      GUESS = auto()
      END = auto()

def is_json_serializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False

def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items() if is_json_serializable(v)}
    elif isinstance(value, list):
        return [sanitize(v) for v in value if is_json_serializable(v)]
    elif is_json_serializable(value):
        return value
    return str(value)

@dataclass
class GameState:
      examples: list[tuple]
      examples_proposed: dict[int, int]
      guesses: dict[int, list[str]]
      player_guess_tokens: dict[int, int]  # player_id -> tokens
      current_turn: Turn
      last_action: dict | None
      input_scene: Any | None = None
      quiz_mode: bool = False
      player_label_guesses: dict[int, bool] = field(default_factory=dict)
      won: bool = False
      max_examples: int = 30
      game_over_reason: str = ""
      turn = 0
      player = 0
      correct_program = None
      def to_dict(self):
            return {
                  "correct_program": str(self.correct_program) if self.correct_program else None,
                  "turns": self.turn + 1,
                  "examples": len(self.examples),
                  "guesses": self.guesses,
                  "player_guess_tokens": self.player_guess_tokens,
                  "last_action": sanitize(self.last_action),
                  "player_label_guesses": self.player_label_guesses,
                  "won": self.won,
                  "max_examples": self.max_examples,
                  "game_over_reason": self.game_over_reason
            }

def step(state: GameState, players: list[ZendoPlayerInterface], gm: GameMaster) -> GameState:
      if state.current_turn == Turn.PROPOSE:
            state.turn += 1
            state.player = state.turn % len(players)
            print(f"========Turn: {state.turn}, Player: {state.player}========")
            proposer = players[state.player]
            action = proposer.react(state)
            state.last_action = action
            if action["input"] is None:
                  print(f"Player {proposer.id} proposed no input, skipping turn")
                  state.current_turn = Turn.PROPOSE
                  example = gm.get_next_example()
                  for i, p in enumerate(players):
                        p.observe(example)
                  return state

            if action["type"] == "propose_input":
                  state.input_scene = action["input"]
                  state.examples_proposed[proposer.id] = state.examples_proposed.get(proposer.id, 0) + 1
                  state.quiz_mode = action["mode"] == "QUIZ"
                  state.current_turn = Turn.LABEL

      elif state.current_turn == Turn.LABEL:
            label = gm.label_input(state.input_scene)
            state.examples.append((state.input_scene, label))

            if state.quiz_mode:
                  print("🧪 QUIZ mode: players guessing label")
                  num_players = len(players)
                  for i in range(num_players):
                        player_index = (state.player + i) % num_players
                        p = players[player_index]
                        guess = p.guess_label(state.input_scene)
                        correct = (guess == label)
                        state.player_label_guesses[p.id] = correct
                        if correct:
                              print(f"Player {i} guessed correctly: {guess}")
                              state.player_guess_tokens[p.id] = state.player_guess_tokens.get(p.id, 0) + 1
                              p.quiz_correct()
                        p.observe((state.input_scene, label))
                  if len(state.examples) >= state.max_examples:
                        state.current_turn = Turn.END
                        state.game_over_reason = "Max examples reached"
                  else:
                        state.current_turn = Turn.GUESS
            else:
                  print("📖 TELL mode: GM reveals label")
                  for p in players:
                        p.observe((state.input_scene, label))
             
                  if len(state.examples) >= state.max_examples:
                        state.current_turn = Turn.END
                        state.game_over_reason = "Max examples reached"
                  else:
                        state.current_turn = Turn.GUESS

      elif state.current_turn == Turn.GUESS:
            print("🤔 Players guessing rules")
            p = players[state.player]
            print(f"Player {p.id} has {state.player_guess_tokens.get(p.id, 0)} guess tokens")
            while state.player_guess_tokens.get(p.id, 0) > 0:
                  guess_action = p.decide_guess(state)
                  if guess_action is None:
                        break
                  rule = guess_action["rule"]
                  correct = gm.check_guess(rule)
                  print(f"Player {p.id} guessed: {rule}, correct: {correct}, correct rule: {gm.true_program}")
                  state.guesses[p.id].append(str(rule))
                  state.player_guess_tokens[p.id] -= 1

                  if correct:
                        state.won = True
                        state.current_turn = Turn.END
                        state.game_over_reason = f"Player {p.id} guessed rule correctly"
                        return state
                  else:
                        counter = gm.disprove_guess(rule)
                        if counter:
                              for _, ps in enumerate(players):
                                    ps.observe(counter)
                              state.examples.append(counter)
                        else:
                              counter = gm.disprove_guess_via_prolog(rule)
                              if counter:
                                    for _, ps in enumerate(players):
                                          ps.observe(counter)
                                    state.examples.append(counter)

            state.current_turn = Turn.PROPOSE

      return state