# Jig AI training

A small **character-level GPT (10.8M parameters)** that learns to write Irish jigs in ABC notation, plus everything we used to study it:
- training from scratch with a chosen number of songs and steps
- fine-tuning on a style
- sampling
- the same scoring the TUMO "Making a Jig AI" activity uses

It's the model side of that activity, packaged to run on its own: on your laptop, or optionally on a Modal GPU.

> **Read [NOTICE.md](NOTICE.md) first.** The tunes come from The Session's open data, whose licence includes a clause prohibiting use with Large Language Models. The tune data is in `data/`.

## What's here

```
jigai/model.py   the MicroGPT (6 layers, 6 heads, width 384, 256-char context), load/save, sampling
jigai/data.py    builds data/ from The Session dump: all 6/8 jigs, one-per-tune jigs, dark set, slip set
jigai/train.py   train from scratch; fine-tune with the tuned per-set defaults
jigai/score.py   Python port of the activity's scoring (checked against the JavaScript in tests/)
jigai/cli.py     `jigai prepare-data | train | finetune | generate | score`
modal_app.py     the same train / finetune / generate on a Modal GPU (optional)
weights/         two trained checkpoints, model.pt not in git (about 45 MB each, get them from the owner)
tests/           unit tests + the JS reference copies the scoring port is checked against
```

The two checkpoints:

| folder | trained on | steps | notes |
|---|---|---|---|
| `weights/tsALL-s6000-v1` | ~14,137 6/8 jig settings from The Session | 6,000 | the **strong** model: final loss 0.736, 975 s on one L4 |
| `weights/d482-s2400` | 482 jigs from an older 2,000-tune book | 2,400 | the model the activity first shipped with: copies more, resolves less |

## Quickstart

```sh
# 1. install (Python 3.10–3.12)
uv venv --python 3.12 && uv pip install -e ".[dev]"      # or: python -m venv .venv && pip install -e ".[dev]"
source .venv/bin/activate

# 2. put the checkpoints in weights/ (from the owner; the config.json files are already there)

# 3. generate from the strong model: no data download needed
jigai generate -n 8 --out samples.abc        # 16 tunes took 44 s on an M-series Mac (measured)

# 4. score them the way the activity does
jigai score samples.abc

# 5. fine-tune on slip jigs (the data ships in data/) and score again
jigai finetune --set slip --out runs/slip     # 120 steps, lr 3e-4: the tuned default
jigai score runs/slip/samples.abc --meter-aware --pool slip
```

**Listen:** `jigai play samples.abc` opens the jig player in your browser with those tunes loaded: sheet music, Play/Stop, and the same sound as the activity (flute at a relaxed jig tempo; fiddle, tin whistle and a faster speed in the menus). You can also drop any `.abc` file on the page. It works fully offline: abcjs and the instrument sounds ship in `jigai/player/`, and `jigai play` serves them from a small local server on your machine (Ctrl-C to stop). Opening `index.html` as a plain file shows the sheet music, but browsers block the sounds that way.

### All commands

```sh
jigai play [file.abc ...]                                       # listen in the browser (sheet music + playback)
jigai prepare-data [--source tunes.json]                        # optional: rebuild data/ from a fresh dump
jigai train --songs 482 --steps 2400 [--out runs/x]             # N one-per-tune jigs, from scratch
jigai train --corpus jigs --steps 6000 --vocab-from weights/tsALL-s6000-v1   # rebuild the strong model
jigai finetune --set dark|slip [--abc liked.abc ...] [--steps --lr] [--base weights/...]
jigai generate --model weights/... -n 16 [--temperature 0.8] [--raw]
jigai score file.abc [--pool jigs|jigs_first|dark|slip|file.json] [--meter-aware] [--verbose]
```

Add `--dry-run` to `train` or `finetune` to check data loading and one forward pass without training. Device is picked automatically (CUDA, then Apple MPS, then CPU); override with `--device`.

### On Modal instead

```sh
uv pip install modal && modal setup                               # your own Modal account
modal run modal_app.py::finetune --set slip --out runs/slip
modal run modal_app.py::train --corpus jigs --steps 6000 --vocab-from tsALL-s6000-v1 --out runs/strong
modal run modal_app.py::generate --n 16 --out samples.abc
```

Your local `data/` and `weights/` are uploaded with each job, and results come back into `runs/`. **Run one job at a time.** Small workspaces have a GPU cap, and we hit ours (10 GPUs) by running retrains in parallel.

## How long things take

| job | Modal (measured) | laptop: Apple M4 Pro, 24 GB, MPS (measured 2026-11-11) |
|---|---|---|
| generate 16 tunes | a few s on a warm T4 | 44 s |
| fine-tune on a style set (dark 40 / slip 120 steps) | 1–10 s training, 8–14 s per request warm, ~15 s cold | about 25 s of training (a dozen slip jigs + the slip set, 120 steps); under a minute end to end, ~1 GB memory |
| train 482 songs × 2,400 steps | ~7 min on an L4 (~$0.10) | not timed; expect roughly 35–40 min (same speed per step as below) |
| train the strong model (14k × 6,000) | 975 s (~16 min) on an L4, ~$0.40 | **1 h 33 min**, peak memory ~15.8 GB (GPU memory shares system RAM on Macs) |

CPU-only machines will be much slower. Try `--dry-run`, then a few hundred steps, before committing to a long local run.

**Is the laptop model as good?** Yes. The strong model trained locally (final loss 0.70) against the Modal-trained `tsALL-s6000-v1`, 48 tunes each from `jigai generate --seed 1`, scored with `--pool jigs_first`:

| | laptop | Modal |
|---|---|---|
| goodness | 81.4 | 74.4 |
| passes the floor rule | 69% | 40% |
| dark keys | 9 of 48 | 8 of 48 |
| copies ≥35% | 0 | 0 |

Same recipe on both, so read this as "equally good" rather than better: one seed and 48 tunes is a small sample.

**Fine-tuning a dozen tunes locally** (12 slip jigs as `--abc`, from the strong model; 16 outputs each):

| recipe | training time | slip jigs | copies ≥35% |
|---|---|---|---|
| the dozen only, 120 steps | 11 s | 12 of 16 | **16 of 16**: it memorised them |
| the dozen only, 30 steps | 2.5 s | 2 of 16 | 0 |
| the dozen + the slip set (`--set slip`), 120 steps | 25 s | **15 of 16** | **0** |

Mix a small personal set with a style set, or the model just recites it.

## What we found

All numbers come from our own runs on Modal, scored with this package's scorers, mostly with 12–48 tunes per setting. Treat differences under ~10 points as noise.

**1. Data is the lever, and too little data with lots of training copies.** In an 11-run sweep (Nov 2026):

| model | goodness | resolves | copied (verbatim, mean) |
|---|---|---|---|
| 482 songs × 2,400 steps (the first shipped model) | 42 | 55% | 21% |
| ~14k songs × 6,000 steps (the strong model) | 71 | 68% (real jigs: 67.5%) | 11% |
| 482 songs × **6,000** steps | **0** | n/a | **97%**: loss 0.06, pure memorisation |

- Quality rises up to ~2,000 songs and then plateaus. More data past that mainly lowers copying.
- A bigger model (25M) and more than 6,000 steps didn't help.
- Changing the representation (notes as interval-from-tonic plus duration) beat the plateau: goodness 74, resolves 73%. That code isn't included here.

**2. Fine-tuning on a learner's few liked tunes memorises them.** We retrained on three liked cards; one was itself a 100% copy of a real jig.

| model | goodness | ≥80% copy of a training card |
|---|---|---|
| base | 71 | 0 of 12 |
| 3 liked cards only, 2nd round | 0 | **9 of 12** (it recites them) |
| 3 liked cards + 24 real songs | 55–62 | 0 of 12 (max overlap 10%) |

The mix stops the copying, but it also stops any measurable lean toward the liked cards. Phrase overlap with them was 4–11%, against 5% for the base.

**3. Fine-tuning on a style set works, starting from the strong model.** For each setting, 64 set tunes plus any extra tunes:

| set | steps × lr | before → after (of 16 tunes) | quality | copying |
|---|---|---|---|---|
| dark (minor + dorian), 1,673 tunes | 40 × 2e-4 | dark keys 2 → 15 (about 60% also *sound* minor) | ~67, unchanged | none ≥35% |
| slip jigs (9/8), 813 tunes | 120 × 3e-4 | slip jigs 0 → 8–16 (63–81% of bars hold 9 eighths) | 50–60 meter-aware (real slip jigs: 56) | none ≥35% |

- Slip jigs at **180 steps started memorising**: two outputs reproduced training tunes verbatim. 120 is the ceiling, and the CLI warns past it.
- A "bright" (major) set changed almost nothing audible, because the base already writes ~80% major tunes.
- **Fine-tuning never measurably improved quality.** It changes the style, not how good the tunes are.

**4. Compared with folk-rnn** (Sturm et al. 2016, the classic thesession.org model), scored with the same functions, 30 tunes each:

| | folk-rnn v2 | strong model (14k × 6,000) | 482 × 2,400 |
|---|---|---|---|
| training data | 23,636 tunes of all types, 100 epochs | ~14k 6/8 jigs, ~60 epochs | 482 jigs, ~700 epochs |
| model | 5.6M-param LSTM | 10.8M-param transformer | same |
| passes the floor rule | 62–63% | 57% | 37% |
| goodness without the style term | 72–75 | 76 | 70 |
| mean verbatim copy | 11% | 12.5% | 14% |

At about n=30, roughly level. folk-rnn stops more cleanly (16–32 bars); ours sometimes runs on.

### What the scores mean

- `goodness` (0–100) is the activity's score, weighted as follows. It is zeroed when ≥35% of the tune is copied verbatim from the `--pool`.

  | component | weight |
  |---|---|
  | phrases that also occur in real jigs | 30% |
  | parts that end on the tonic | 25% |
  | 6/8 bars | 15% |
  | few big leaps | 12% |
  | sensible length | 10% |
  | repeats its own motifs | 8% |

  Without `--pool`, the style and copy terms are 0, so scores read about 30 points lower.
- `floor`: has a valid key, lands on its home note within its first part or two, and runs 6–24 bars.
- `dark` / `slip`: membership used for the before/after counts. Dark is a minor, dorian or aeolian key. Slip is 9/8 with ≥75% of inner bars holding 9 eighths.
- `--meter-aware` judges bars and length against each tune's own meter. Use it for slip jigs: the default assumes 6/8.

## Known gaps

- **Licence:** see NOTICE.md. It needs a decision before anyone trains on, or redistributes, the data or the weights.
- **9/8 scoring is new** and lightly tested. `goodness` and the floor rule were designed for 6/8.
- The 482-song checkpoint came from an older 2,000-tune book that isn't included. `train --songs 482` draws 482 jigs from The Session data instead, so it won't reproduce that checkpoint exactly.
- `score --pool jigs` (14k tunes) runs the copy check against ~3.7 MB of text and is slow, about seconds per tune.
- The laptop timings come from one Mac (M4 Pro); other machines will differ.
- Sample sizes in "What we found" are small; see each table.

## Tests

```sh
pytest -q        # model shape and checkpoint round-trip, data prep on a hand-made fixture, scoring port vs JS (needs node)
```

`tests/js_ref/` holds copies of the activity's JavaScript scorers (from `irish-music-ai`). The port test runs both on the same tunes and requires them to match exactly.
