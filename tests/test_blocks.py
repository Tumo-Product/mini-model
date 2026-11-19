"""The shared X: block helpers that both the jig and the Connect Four code use."""

import json

import pytest

from microgpt import blocks


def test_split_first_and_write_round_trip(tmp_path):
    text = "\n X:1\nK:D\nDFA|\r\n\r\nX:c4\n1122334\nW:1\n"
    parts = blocks.split_blocks(text)
    assert parts == ["X:1\nK:D\nDFA|", "X:c4\n1122334\nW:1"]
    assert blocks.first_block("X:1\nK:D\nDFA|\n\nX:2\nK:G") == "X:1\nK:D\nDFA|\n"
    assert blocks.first_block("X:1\nK:D") == "X:1\nK:D"
    blocks.write_blocks(parts, str(tmp_path / "out" / "all.txt"))
    assert blocks.split_blocks(open(tmp_path / "out" / "all.txt").read()) == parts


def test_load_corpus_by_name_or_path(tmp_path, monkeypatch):
    (tmp_path / "games.json").write_text(json.dumps(["X:c4\n1\nW:0"]))
    assert blocks.load_corpus("games", str(tmp_path)) == ["X:c4\n1\nW:0"]
    assert blocks.load_corpus(str(tmp_path / "games.json")) == ["X:c4\n1\nW:0"]
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "connect4.json").write_text(json.dumps(["X:c4\n2\nW:0"]))
    (tmp_path / "connect4").mkdir()                       # a folder with the corpus's name, as in the repo root
    monkeypatch.chdir(tmp_path)
    assert blocks.load_corpus("connect4") == ["X:c4\n2\nW:0"]
    with pytest.raises(SystemExit, match="make it first"):
        blocks.load_corpus("missing", str(tmp_path), "make it first")
    for path in ("nope.json", str(tmp_path / "nope")):      # a mistyped path is reported as that path, without the hint
        with pytest.raises(SystemExit) as err:
            blocks.load_corpus(path, str(tmp_path), "make it first")
        assert str(err.value) == f"{path} not found."
