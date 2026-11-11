"""Training from scratch and fine-tuning. Same recipes as the Modal runs behind the activity.

train():    AdamW lr 1e-3, wd 0.1, cosine schedule, grad clip 1.0, batch 128 x 256 chars, 90% of the
            text for training, fp16 autocast on CUDA. 6,000 steps on ~14k jigs = the strong model.
finetune(): start from a trained model, keep training briefly on a style set (+ optional extra tunes).
            AdamW wd 0.1, cosine, batch 8-32, bf16 on Ampere+ GPUs. Defaults per set were tuned on the
            strong base; MORE STEPS MEMORISE (slip at 180 steps reproduced training tunes verbatim).
"""

import random
import time

import torch

from .model import ARCH, MicroGPT, Vocab, load_checkpoint, pick_device, sample, save_checkpoint

SET_DEFAULTS = {
    "dark": {"steps": 40, "lr": 2e-4, "set_size": 64, "extra_weight": 0.06},
    "slip": {"steps": 120, "lr": 3e-4, "set_size": 64, "extra_weight": 0.06},
    None: {"steps": 20, "lr": 1e-4, "set_size": 0, "extra_weight": 0.5},
}
STEP_CEILING = {"dark": 60, "slip": 120}


def _amp(device):
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        return torch.autocast("cuda", dtype=dtype), dtype
    return torch.autocast("cpu", enabled=False), None


def _log(msg):
    # flush so progress shows up when output goes to a file or a pipe
    print(msg, flush=True)


def train(blocks, steps, out_dir, vocab_chars=None, seed=0, device=None, compile_model=False, num_samples=8,
          dry_run=False, log=None):
    log = log or _log
    device = pick_device(device)
    text = "\n\n".join(blocks) + "\n"
    vocab = Vocab(vocab_chars or sorted(set(text)))
    data = torch.tensor(vocab.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n] if n > ARCH["block_size"] else data
    block = ARCH["block_size"]
    batch = min(128, max(1, len(train_data) // block))
    torch.manual_seed(seed)
    model = MicroGPT(len(vocab), **ARCH).to(device)
    log(f"train: {len(blocks)} tunes, {len(text):,} chars, vocab {len(vocab)}, batch {batch}x{block}, "
        f"{steps} steps on {device} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")
    offs = torch.arange(block)

    def get_batch():
        ix = torch.randint(max(1, len(train_data) - block - 1), (batch,))
        return train_data[ix[:, None] + offs].to(device), train_data[ix[:, None] + offs + 1].to(device)

    if dry_run:
        xb, yb = get_batch()
        with torch.no_grad():
            _, loss = model(xb, yb)
        log(f"dry run: one forward pass ok, loss {loss.item():.3f} (no optimizer step taken)")
        return None
    runner = torch.compile(model) if compile_model else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    use_fp16 = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    t0, history = time.time(), []
    for step in range(steps):
        xb, yb = get_batch()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            _, loss = runner(xb, yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 50 == 0 or step == steps - 1:
            history.append({"step": step, "loss": round(loss.item(), 4)})
            if step % 500 == 0 or step == steps - 1:
                log(f"  step {step + 1}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
    tunes = sample(model, vocab, n=num_samples) if num_samples else []
    save_checkpoint(out_dir, model, vocab, {"type": "train", "steps": steps, "songs": len(blocks),
                                            "corpus_chars": len(text), "loss_history": history,
                                            "seconds": round(time.time() - t0, 1), "samples": tunes})
    log(f"saved {out_dir} (final loss {history[-1]['loss']}, {time.time() - t0:.0f}s)")
    return tunes


def build_finetune_text(extra, set_blocks, extra_weight, vocab, seed):
    """Extra tunes (e.g. a learner's liked cards) are repeated until they are ~extra_weight of the
    characters, then mixed with the set sample. Tiny inputs are tiled to fill a few 256-char windows."""
    keep = lambda s: "".join(c for c in s.strip() if c in vocab.stoi)
    extra = [keep(t if t.startswith("X:") else "X:1\n" + t) for t in extra]
    extra = [t for t in extra if len(t) > 4]
    real = [keep(b) for b in set_blocks]
    ec, rc = sum(map(len, extra)) or 1, sum(map(len, real))
    repeat = max(1, round(extra_weight / (1 - extra_weight) * rc / ec)) if extra and real and extra_weight < 1 else 1
    blocks = extra * repeat + real
    random.Random(seed).shuffle(blocks)
    text = "\n\n".join(blocks) + "\n"
    while len(text) < 3 * ARCH["block_size"]:
        text = text + "\n" + text
    return text, {"extra": len(extra), "extra_repeat": repeat, "set_tunes": len(real), "chars": len(text)}


def finetune(base_dir, out_dir, set_blocks=(), set_name=None, extra=(), steps=None, lr=None, seed=0,
             device=None, num_samples=16, dry_run=False, log=None):
    log = log or _log
    d = SET_DEFAULTS.get(set_name, SET_DEFAULTS[None])
    steps = steps or d["steps"]
    lr = lr or d["lr"]
    ceiling = STEP_CEILING.get(set_name)
    if ceiling and steps > ceiling:
        log(f"warning: {steps} steps on '{set_name}' is past the tested ceiling ({ceiling}); "
            f"in our runs more steps started reproducing training tunes verbatim.")
    device = pick_device(device)
    model, vocab, base_cfg = load_checkpoint(base_dir, device)
    rng = random.Random(seed)
    sample_set = rng.sample(list(set_blocks), min(d["set_size"] or len(set_blocks), len(set_blocks))) if set_blocks else []
    text, info = build_finetune_text(list(extra), sample_set, d["extra_weight"], vocab, seed)
    data = torch.tensor(vocab.encode(text), dtype=torch.long, device=device)
    block = ARCH["block_size"]
    batch = min(32, max(8, len(data) // block))
    hi = max(1, len(data) - block - 1)
    offs = torch.arange(block, device=device)
    log(f"finetune: base {base_dir}, set {set_name or '-'} ({info['set_tunes']} tunes), extra {info['extra']} "
        f"x{info['extra_repeat']}, {steps} steps, lr {lr}, batch {batch}, on {device}")
    torch.manual_seed(seed)
    if dry_run:
        ix = torch.randint(hi, (batch,), device=device)
        with torch.no_grad():
            _, loss = model(data[ix[:, None] + offs], data[ix[:, None] + offs + 1])
        log(f"dry run: one forward pass ok, loss {loss.item():.3f} (no optimizer step taken)")
        return None
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    amp_ctx, _ = _amp(device)
    t0, history = time.time(), []
    for step in range(steps):
        ix = torch.randint(hi, (batch,), device=device)
        with amp_ctx:
            _, loss = model(data[ix[:, None] + offs], data[ix[:, None] + offs + 1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 10 == 0 or step == steps - 1:
            history.append({"step": step, "loss": round(loss.item(), 4)})
    train_s = time.time() - t0
    tunes = sample(model, vocab, n=num_samples) if num_samples else []
    save_checkpoint(out_dir, model, vocab, {"type": "finetune", "base": base_dir, "set": set_name, "steps": steps,
                                            "lr": lr, "corpus": info, "loss_history": history,
                                            "train_seconds": round(train_s, 1), "samples": tunes})
    log(f"saved {out_dir} (final loss {history[-1]['loss']}, trained in {train_s:.1f}s)")
    return tunes
