"""The 10.8M-parameter character-level GPT ("MicroGPT"), with checkpoints and sampling.

Same layer names and shapes as the original Jig AI Modal code, so its checkpoints load unchanged.
"""

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH = {"n_layer": 6, "n_head": 6, "n_embd": 384, "block_size": 256, "dropout": 0.2}


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.n_embd, self.n_head = n_embd, n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        hd = C // self.n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = self.attn_dropout(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class MicroGPT(nn.Module):
    def __init__(self, vocab_size, n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.tok_emb.weight = self.head.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        logits = self.head(self.ln_f(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
        return idx


def pick_device(want=None):
    if want:
        return want
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Vocab:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


def load_checkpoint(path, device=None):
    """path = a run folder holding model.pt + config.json (vocab + arch). Returns (model, vocab, config)."""
    device = pick_device(device)
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    vocab = Vocab(cfg["vocab"])
    model = MicroGPT(len(vocab), **cfg.get("arch", ARCH)).to(device)
    sd = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.", ""): v.float() if v.is_floating_point() else v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model, vocab, cfg


def save_checkpoint(path, model, vocab, extra=None):
    os.makedirs(path, exist_ok=True)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(raw.state_dict(), os.path.join(path, "model.pt"))
    cfg = {"vocab": vocab.chars, "arch": ARCH, **(extra or {})}
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)


def sample(model, vocab, n=4, temperature=0.8, max_new_tokens=500, device=None, start="X:"):
    """n generations, each beginning with `start` (the opening of a block)."""
    device = device or next(model.parameters()).device
    model.eval()
    ctx = torch.tensor([vocab.encode(start)] * n, dtype=torch.long, device=device)
    out = model.generate(ctx, max_new_tokens=max_new_tokens, temperature=temperature)
    return [vocab.decode(row.tolist()) for row in out]


def generate_from(path, n=8, temperature=0.8, max_new_tokens=500, seed=0, device=None, raw=False, start="X:"):
    """Load a checkpoint and sample n blocks. raw=True keeps each whole generation (several blocks run together).

    start is the text every generation begins with: "X:" alone, or a whole header line the corpus always uses."""
    from .blocks import first_block
    torch.manual_seed(seed)
    model, vocab, _ = load_checkpoint(path, device)
    out = sample(model, vocab, n=n, temperature=temperature, max_new_tokens=max_new_tokens, start=start)
    return out if raw else [first_block(t) for t in out]
