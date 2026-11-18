"""The shared training loop on a tiny model: validation loss is recorded and does not disturb training."""

import json
import random

from microgpt import train
from microgpt.model import ARCH


def test_train_records_validation_loss_without_changing_the_training_run(tmp_path, monkeypatch):
    for key, value in {"n_layer": 1, "n_head": 2, "n_embd": 32, "block_size": 32}.items():
        monkeypatch.setitem(ARCH, key, value)
    monkeypatch.setattr(train, "EVAL_BATCHES", 2)
    rng = random.Random(1)
    blocks = [f"X:{i}\n" + "".join(rng.choice("ABCDEFG|") for _ in range(30)) for i in range(60)]

    def losses(eval_every, out):
        monkeypatch.setattr(train, "EVAL_EVERY", eval_every)
        train.train(blocks, 4, str(tmp_path / out), device="cpu", num_samples=0, log=lambda _: None)
        return json.load(open(tmp_path / out / "config.json"))["loss_history"]

    often, rarely = losses(1, "often"), losses(1000, "rarely")
    assert [h["step"] for h in often] == [0, 3]
    assert all(h["val_loss"] > 0 and h["train_eval"] > 0 for h in often)
    assert [h["loss"] for h in often] == [h["loss"] for h in rarely]         # checking more often changes nothing

    train.train(blocks[:2], 2, str(tmp_path / "tiny"), device="cpu", num_samples=0, log=lambda _: None)
    tiny = json.load(open(tmp_path / "tiny" / "config.json"))["loss_history"]
    assert "val_loss" not in tiny[-1] and "train_eval" in tiny[-1]           # too little text to hold any back
