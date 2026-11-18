"""Training from scratch on text blocks. The same recipe as the Modal runs behind the Jig AI activity.

train(): AdamW lr 1e-3, wd 0.1, cosine schedule, grad clip 1.0, batch 128 x 256 chars, 90% of the text for
         training and the rest held back for a validation loss, fp16 autocast on CUDA.
"""

import time

import torch

from .model import ARCH, MicroGPT, Vocab, pick_device, sample, save_checkpoint

EVAL_EVERY = 100
EVAL_BATCHES = 8


def print_flush(msg):
    # flush so progress shows up when output goes to a file or a pipe
    print(msg, flush=True)


def train(blocks, steps, out_dir, vocab_chars=None, seed=0, device=None, compile_model=False, num_samples=8,
          dry_run=False, log=None, unit="blocks"):
    """Train on the blocks and save a checkpoint to out_dir. Returns num_samples raw generations.

    unit is only the word used in the log line ("tunes", "games")."""
    log = log or print_flush
    device = pick_device(device)
    text = "\n\n".join(blocks) + "\n"
    vocab = Vocab(vocab_chars or sorted(set(text)))
    data = torch.tensor(vocab.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n] if n > ARCH["block_size"] else data
    val_data = data[n:] if len(data) - n > ARCH["block_size"] + 1 else None
    block = ARCH["block_size"]
    batch = min(128, max(1, len(train_data) // block))
    torch.manual_seed(seed)
    model = MicroGPT(len(vocab), **ARCH).to(device)
    log(f"train: {len(blocks)} {unit}, {len(text):,} chars, vocab {len(vocab)}, batch {batch}x{block}, "
        f"{steps} steps on {device} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")
    offs = torch.arange(block)

    def get_batch(src=None, gen=None):
        src = train_data if src is None else src
        ix = torch.randint(max(1, len(src) - block - 1), (batch,), generator=gen)
        return src[ix[:, None] + offs].to(device), src[ix[:, None] + offs + 1].to(device)

    eval_gen = torch.Generator()

    @torch.no_grad()
    def estimate_loss(iters=EVAL_BATCHES):
        # Its own generator, re-seeded each time: every check scores the same batches, and the training
        # batches for a given seed are the same whether or not (and however often) this runs.
        eval_gen.manual_seed(seed)
        model.eval()
        out = {}
        for name, src in (("train_eval", train_data), ("val_loss", val_data)):
            if src is not None:
                out[name] = round(sum(model(*get_batch(src, eval_gen))[1].item() for _ in range(iters)) / iters, 4)
        model.train()
        return out

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
            entry = {"step": step, "loss": round(loss.item(), 4)}
            if step % EVAL_EVERY == 0 or step == steps - 1:
                entry.update(estimate_loss())
            history.append(entry)
            if step % 500 == 0 or step == steps - 1:
                val = f" val {entry['val_loss']:.4f}" if "val_loss" in entry else ""
                log(f"  step {step + 1}/{steps} loss {loss.item():.4f}{val} ({time.time() - t0:.0f}s)")
    samples = sample(model, vocab, n=num_samples) if num_samples else []
    # "songs" is the older name for "blocks": both are written so new runs still line up with earlier configs
    save_checkpoint(out_dir, model, vocab, {"type": "train", "steps": steps, "blocks": len(blocks),
                                            "songs": len(blocks), "corpus_chars": len(text), "loss_history": history,
                                            "seconds": round(time.time() - t0, 1), "samples": samples})
    log(f"saved {out_dir} (final loss {history[-1]['loss']}, {time.time() - t0:.0f}s)")
    return samples
