# Connect Four

The MicroGPT in `microgpt/` learns Connect Four from games written as text. It is the same model and training loop that learn Irish jigs in `jigai/`, and nothing else is shared with that package. `connect4/game.py` holds the rules, the bots, the data generator and the checks; `connect4/cli.py` is the `connect4` command. One game is one block:

```
X:c4
4453342271
W:1
```

The second line lists the moves, one digit per move: the column (1–7) the piece was dropped in, player 1 first. `W:` gives the winner (1 or 2), or 0 for a draw. A game is at most 51 characters, so the whole game always fits in the 256-character window.

Install from the repo root, as in the main [README](../README.md): `pip install -e ".[dev]"`. Run the commands below from the repo root too, since they read `data/` and write `runs/`. `python -m connect4 ...` works the same as `connect4 ...`.

```sh
connect4 data                                        # 100,000 distinct self-play games -> data/connect4.json (~30 s)
connect4 train --steps 600 --out runs/c4
connect4 train --games 2000 --steps 600 --out runs/c4-2k                  # fewer games, same steps
connect4 generate --model runs/c4 -n 64 --out runs/c4/games.txt
connect4 score runs/c4/games.txt --pool connect4     # does it write real games?
connect4 eval --model runs/c4 --opponent random      # can it play? (also --opponent smart)
```

- **The data** comes from a "smart" bot playing itself: it wins when it can, blocks when it must, avoids setting up the opponent's win, and otherwise prefers the centre. 10% of its moves are random (`--eps`) so games stay varied. `--bot random` gives games between random players, which makes a weaker training set to compare against.
- **`score`** checks games the model writes on its own. Score the file `generate` writes by default (one whole game per sample). With `--raw`, each sample ends mid-game and that cut-off game would count as a mistake.
  - `legal_move_pct`: share of moves that are legal. An illegal move is skipped and the replay carries on; moves written after the game has ended all count as illegal.
  - `clean_game_pct`: share of games that are legal and stop exactly when someone wins or the board fills.
  - `result_correct_pct`: share of clean games whose `W:` line names the right winner.
  - `copied_pct`: share of games identical to a training game.
- **`eval`** plays real games against a bot, with the model moving first in half of them. `top_choice_legal_pct` is how often the model's first choice was a legal column. The move it actually plays is its most likely legal column, so an illegal guess never ends a game early.

**First run** (100k games, 600 steps, final loss 1.44; 23 min on a busy M-series Mac via MPS, measured 2026-11-12). This is one seed with 200 games per row, so treat the numbers as rough:

| check | result |
|---|---|
| plays against a random player | wins 85.5% (86% moving first, 85% moving second) |
| plays against the smart bot | wins 10.5%, loses 88.5% |
| games it writes on its own: legal moves | 60% |
| games it writes on its own: clean games | 1 of 200 (0.5%) |
| first mistake in those 200 games | 90 played on after the game had ended, 69 stopped before the end, 40 dropped into a full column |
| copies of a training game | 0 |

It picks reasonable moves, but at 600 steps it hasn't learned to tell when a game is over. That needs win detection, which is harder than choosing a move. Longer training (6,000 steps, as for the strong jig model) is the next thing to try.
