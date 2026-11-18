"""Model shape and checkpoint round-trip. No training: one forward pass only."""

import os

import torch

from jigai.finetune import build_finetune_text
from microgpt.model import ARCH, MicroGPT, Vocab, load_checkpoint, save_checkpoint


def test_param_count_and_forward():
    vocab = Vocab(sorted(set("X:1\nM:6/8\nK:D\n|:DFA dfe:|")))
    m = MicroGPT(len(vocab), **ARCH)
    n = sum(p.numel() for p in m.parameters())
    assert 10.6e6 < n < 10.9e6, n
    x = torch.zeros((2, 16), dtype=torch.long)
    logits, loss = m(x, x)
    assert logits.shape == (2, 16, len(vocab)) and loss.item() > 0


def test_checkpoint_round_trip(tmp_path):
    vocab = Vocab(list("X:1\nABCDEFGabcdefg|:236/"))
    m = MicroGPT(len(vocab), **ARCH)
    save_checkpoint(tmp_path, m, vocab)
    m2, v2, cfg = load_checkpoint(tmp_path, "cpu")
    assert v2.chars == vocab.chars
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)


def test_strong_weights_load_if_present():
    path = os.path.join(os.path.dirname(__file__), "..", "weights", "tsALL-s6000-v1")
    if not os.path.exists(os.path.join(path, "model.pt")):
        return
    m, vocab, cfg = load_checkpoint(path, "cpu")
    assert len(vocab) == 133


def test_finetune_text_mixes_and_tiles():
    vocab = Vocab(sorted(set("X:1\nM:6/8\nK:D\n|:DFA dfe:| ")))
    text, info = build_finetune_text(["X:1\nK:D\nDFA|"], ["X:2\nK:D\n|:DFA dfe:|"] * 10, 0.06, vocab, 0)
    assert info["extra"] == 1 and info["set_tunes"] == 10 and len(text) >= 3 * 256
