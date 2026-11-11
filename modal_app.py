"""Run training / fine-tuning / sampling on a Modal GPU instead of your machine.

Needs your own Modal account (`pip install modal && modal setup`). Your local data/ and weights/
folders are shipped with the job; results come back into runs/ on your machine. Examples:

  modal run modal_app.py::finetune --base tsALL-s6000-v1 --set slip --out runs/slip-1
  modal run modal_app.py::finetune --base tsALL-s6000-v1 --abc my_tunes.abc --steps 20 --out runs/mine
  modal run modal_app.py::train --songs 482 --steps 2400 --out runs/s482-x2400
  modal run modal_app.py::train --corpus jigs --steps 6000 --out runs/strong     # ~16 min on an L4
  modal run modal_app.py::generate --model tsALL-s6000-v1 --n 16 --out samples.abc

Run jobs one at a time: free/starter workspaces have a small GPU cap (ours was 10).
"""

import os

import modal

GPU = os.environ.get("JIGAI_GPU", "L4")
HERE = os.path.dirname(os.path.abspath(__file__))

image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.5.1", "numpy<2").add_local_python_source("jigai")
for folder in ("data", "weights"):
    if os.path.isdir(os.path.join(HERE, folder)):
        image = image.add_local_dir(os.path.join(HERE, folder), f"/root/{folder}")

app = modal.App("jig-ai-training", image=image)


def _collect(out_dir, tunes=None):
    if tunes:
        from jigai.score import first_tune
        with open(os.path.join(out_dir, "samples.abc"), "w") as f:
            f.write("\n\n".join(first_tune(t).strip() for t in tunes) + "\n")
    files = {}
    for name in os.listdir(out_dir):
        with open(os.path.join(out_dir, name), "rb") as f:
            files[name] = f.read()
    return files


def _save(files, out):
    os.makedirs(out, exist_ok=True)
    for name, blob in files.items():
        with open(os.path.join(out, name), "wb") as f:
            f.write(blob)
    print(f"saved {', '.join(sorted(files))} to {out}")


@app.function(gpu=GPU, timeout=3600)
def _train(songs: int, steps: int, corpus: str, seed: int, vocab_from: str):
    import json
    from jigai import data, train
    os.chdir("/root")
    blocks = data.songs_subset(songs, seed) if (songs and corpus == "jigs_first") else data.load(corpus)[: songs or None]
    vocab = None
    if vocab_from:
        with open(os.path.join("weights", vocab_from, "config.json")) as f:
            vocab = json.load(f)["vocab"]
    tunes = train.train(blocks, steps, "/tmp/out", vocab, seed, "cuda", compile_model=True)
    return _collect("/tmp/out", tunes)


@app.function(gpu=GPU, timeout=900)
def _finetune(base: str, set_name: str, extra: list, steps: int, lr: float, seed: int, samples: int):
    from jigai import data, train
    os.chdir("/root")
    set_blocks = data.load(set_name) if set_name else []
    tunes = train.finetune(os.path.join("weights", base), "/tmp/out", set_blocks, set_name or None, extra,
                           steps or None, lr or None, seed, "cuda", samples)
    return _collect("/tmp/out", tunes)


@app.function(gpu="T4", timeout=600)
def _generate(model: str, n: int, temperature: float, seed: int):
    import torch
    from jigai.model import load_checkpoint, sample
    torch.manual_seed(seed)
    m, vocab, _ = load_checkpoint(os.path.join("/root/weights", model), "cuda")
    return sample(m, vocab, n=n, temperature=temperature)


@app.local_entrypoint()
def train(songs: int = 0, steps: int = 2400, corpus: str = "jigs_first", seed: int = 0, vocab_from: str = "", out: str = ""):
    _save(_train.remote(songs, steps, corpus, seed, vocab_from), out or f"runs/s{songs or 'all'}-x{steps}")


@app.local_entrypoint()
def finetune(base: str = "tsALL-s6000-v1", set: str = "", abc: str = "", steps: int = 0, lr: float = 0.0,
             seed: int = 0, samples: int = 16, out: str = ""):
    from jigai.data import read_abc_files
    extra = read_abc_files([p for p in abc.split(",") if p])
    _save(_finetune.remote(base, set, extra, steps, lr, seed, samples), out or f"runs/ft-{set or 'abc'}")


@app.local_entrypoint()
def generate(model: str = "tsALL-s6000-v1", n: int = 8, temperature: float = 0.8, seed: int = 0, out: str = ""):
    from jigai.score import first_tune
    tunes = [first_tune(t) for t in _generate.remote(model, n, temperature, seed)]
    text = "\n\n".join(t.strip() for t in tunes) + "\n"
    if out:
        with open(out, "w") as f:
            f.write(text)
        print(f"wrote {len(tunes)} tunes to {out}")
    else:
        print(text)
