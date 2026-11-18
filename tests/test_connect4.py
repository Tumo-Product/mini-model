"""Connect Four rules, self-play data, the generated-game checker and the model-vs-bot loop (untrained model)."""

import json
import random

import pytest
import torch

from connect4 import game as c4
from connect4.cli import main
from microgpt.blocks import split_blocks
from microgpt.model import ARCH, MicroGPT, Vocab


def brute_force_winner(moves):
    grid = [[0] * c4.ROWS for _ in range(c4.COLS)]
    for i, ch in enumerate(moves):
        col = int(ch) - 1
        grid[col][grid[col].index(0)] = i % 2 + 1
    for c in range(c4.COLS):
        for r in range(c4.ROWS):
            p = grid[c][r]
            for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                cells = [(c + k * dc, r + k * dr) for k in range(4)]
                if p and all(0 <= x < c4.COLS and 0 <= y < c4.ROWS and grid[x][y] == p for x, y in cells):
                    return p
    return None


def test_wins_in_every_direction():
    assert c4.Game("1122334").winner == 1          # horizontal
    assert c4.Game("1212121").winner == 1          # vertical
    assert c4.Game("12233434544").winner == 1      # diagonal /
    assert c4.Game("76655454344").winner == 1      # diagonal \
    assert c4.Game("112233").winner is None


def test_bitboards_match_brute_force_on_random_games():
    rng = random.Random(0)
    for _ in range(300):
        g = c4.Game()
        while not g.over:
            g.play(c4.random_move(g, rng))
            expected = brute_force_winner(g.text_moves())
            assert (g.winner if g.winner else None) == expected or (g.winner == 0 and expected is None)


def test_smart_bot_wins_blocks_and_beats_random():
    rng = random.Random(0)
    assert c4.smart_move(c4.Game("121212"), rng) == 0       # completes its own column
    assert c4.smart_move(c4.Game("17273"), rng) == 3        # blocks 1-2-3 on the bottom row
    wins = 0
    for i in range(100):
        g = c4.Game()
        while not g.over:
            g.play(c4.smart_move(g, rng) if g.turn == i % 2 else c4.random_move(g, rng))
        wins += g.winner == i % 2 + 1
    assert wins >= 90


def test_generated_games_are_distinct_legal_and_reproducible():
    games = c4.generate_games(300, seed=5)
    assert len(games) == len(set(games)) == 300
    assert games == c4.generate_games(300, seed=5)
    assert all(g.startswith("X:c4\n") for g in games)
    summary, _ = c4.score_text("\n\n".join(games), pool=games)
    assert summary["clean_game_pct"] == 100 and summary["result_correct_pct"] == 100
    assert summary["copied_pct"] == 100
    assert all(len(g) < ARCH["block_size"] for g in games)


def test_checker_catches_each_kind_of_mistake():
    assert c4.check_game("X:c4\n1122334\nW:1")["result_ok"]
    assert not c4.check_game("X:c4\n1122334\nW:2")["result_ok"]               # wrong result line
    assert c4.check_game("X:c4\n1122334\nW:1")["first_mistake"] is None
    r = c4.check_game("X:c4\n11223345\nW:1")
    assert r["legal"] == 7 and not r["clean"] and r["first_mistake"] == "played_after_end"
    r = c4.check_game("X:c4\n1111111\nW:0")
    assert r["legal"] == 6 and not r["clean"] and r["first_mistake"] == "full_column"
    r = c4.check_game("X:c4\n12893\nW:1")
    assert r["legal"] == 3 and r["first_mistake"] == "not_a_column"         # 8 and 9 skipped, the 3 still counts
    assert c4.check_game("X:c4\n1122\nW:1")["first_mistake"] == "stopped_early"
    assert c4.check_game("garbage")["legal"] == 0


def test_checker_counts_legal_moves_after_a_mistake_and_never_raises():
    r = c4.check_game("X:c4\n1111111234\nW:0")                               # one illegal move: the 7th in column 1
    assert r["legal"] == 9 and r["length"] == 10 and r["first_mistake"] == "full_column"
    r = c4.check_game("X:c4\n12²3\nW:1")                                     # '²'.isdigit() is True, int('²') raises
    assert r["legal"] == 3 and r["first_mistake"] == "not_a_column"
    s, _ = c4.score_text("X:c4\n1111111234\nW:0")
    assert s["legal_move_pct"] == 90.0


def test_score_text_splits_generated_file_and_counts_copies():
    text = "X:c4\n1122334\nW:1\n\nX:c4\n1212121\nW:1\n\nX:c4\n99\nW:"
    s, rows = c4.score_text(text, pool=["X:c4\n1122334\nW:1"])
    assert s["games"] == 3 and s["clean_game_pct"] == 66.7 and s["copied_pct"] == 33.3
    assert [r["copied"] for r in rows] == [True, False, False]


def test_play_vs_bot_runs_with_an_untrained_model():
    vocab = Vocab(sorted(set("".join(c4.generate_games(50)) + "\n")))
    torch.manual_seed(0)
    model = MicroGPT(len(vocab), **ARCH)
    out = c4.play_vs_bot(model, vocab, n=6, opponent="random", seed=1)
    assert out["as_first"]["games"] == out["as_second"]["games"] == 3
    o = out["overall"]
    assert abs(o["win_pct"] + o["draw_pct"] + o["loss_pct"] - 100) < 0.2
    assert 0 <= out["top_choice_legal_pct"] <= 100
    one = c4.play_vs_bot(model, vocab, n=1, opponent="random", seed=1)       # no game as second player
    assert one["as_first"]["games"] == 1 and one["as_second"] == {"games": 0, "win_pct": 0.0, "draw_pct": 0.0,
                                                                  "loss_pct": 0.0}


def test_c4_data_command_writes_a_corpus(tmp_path, capsys):
    main(["--data", str(tmp_path), "data", "--games", "40"])
    games = json.load(open(tmp_path / "connect4.json"))
    assert len(games) == 40
    assert json.loads(capsys.readouterr().out)["games"] == 40
    main(["--data", str(tmp_path), "data", "--games", "20"])              # replacing its own corpus is fine
    assert len(json.load(open(tmp_path / "connect4.json"))) == 20


def test_c4_data_command_will_not_replace_a_tune_corpus(tmp_path):
    tunes = ["X:1\nT:A jig\nK:D\nDFA dAF|"]
    (tmp_path / "jigs.json").write_text(json.dumps(tunes))
    with pytest.raises(SystemExit):
        main(["--data", str(tmp_path), "data", "--games", "5", "--name", "jigs"])
    assert json.load(open(tmp_path / "jigs.json")) == tunes


def test_data_command_follows_jigai_data(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JIGAI_DATA", str(tmp_path / "shared"))
    main(["data", "--games", "5"])
    assert len(json.load(open(tmp_path / "shared" / "connect4.json"))) == 5


def test_train_generate_score_and_eval_commands(tmp_path, capsys, monkeypatch):
    for key, value in {"n_layer": 1, "n_head": 2, "n_embd": 32, "block_size": 32}.items():
        monkeypatch.setitem(ARCH, key, value)
    data, run, games_file = str(tmp_path / "data"), str(tmp_path / "run"), str(tmp_path / "games.txt")
    main(["--data", data, "data", "--games", "80"])

    main(["--data", data, "--device", "cpu", "train", "--games", "60", "--steps", "3", "--samples", "2", "--out", run])
    cfg = json.load(open(tmp_path / "run" / "config.json"))
    assert cfg["blocks"] == cfg["songs"] == 60 and cfg["steps"] == 3 and len(cfg["samples"]) == 2
    assert len(split_blocks(open(tmp_path / "run" / "samples.txt").read())) == 2

    def generate(*extra):
        capsys.readouterr()
        main(["--device", "cpu", "generate", "--model", run, "-n", "3", "--max-chars", "40", *extra])
        return capsys.readouterr().out

    assert generate("--seed", "1") == generate("--seed", "1") != generate("--seed", "2")
    assert generate("--seed", "1", "--out", games_file) == f"wrote 3 games to {games_file}\n"
    written = split_blocks(open(games_file).read())
    assert len(written) == 3 and all(g.startswith(c4.HEADER) for g in written)   # every game keeps its header
    assert "\n\n".join(written) + "\n" == generate("--seed", "1")                # the file holds what it prints

    capsys.readouterr()
    main(["--data", data, "score", games_file, "--pool", "connect4"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["games"] == 3 and summary["copied_pct"] is not None
    with pytest.raises(SystemExit, match="connect4 data"):                       # a missing corpus says how to make it
        main(["--data", str(tmp_path / "empty"), "score", games_file, "--pool", "connect4"])

    main(["--device", "cpu", "eval", "--model", run, "-n", "4", "--seed", "2"])
    assert json.loads(capsys.readouterr().out)["overall"]["games"] == 4
