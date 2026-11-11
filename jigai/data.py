"""Build the training data from The Session's open data dump (tunes.json).

The built files ship in data/. `jigai prepare-data` only rebuilds them: it downloads
tunes.json from github.com/adactio/TheSession-data (or reads a copy you already have) and writes:

  data/jigs.json        every 6/8 jig setting  -> the strong base model's corpus (~14k)
  data/jigs_first.json  first setting of each 6/8 jig (one per tune) -> the --songs N subsets
  data/dark.json        first settings of minor / dorian 6/8 jigs      -> fine-tune set "dark"
  data/slip.json        first settings of 9/8 slip jigs                 -> fine-tune set "slip"
  data/meta.json        counts and the source date
"""

import json
import os
import random
import urllib.request

TUNES_URL = "https://raw.githubusercontent.com/adactio/TheSession-data/main/json/tunes.json"
DATA_DIR = os.environ.get("JIGAI_DATA", "data")


def tune_block(t):
    """One tune as ABC, in the same shape the models were trained on."""
    body = (t.get("abc") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
    if not lines:
        return None
    body = lines[0] + "".join("\n " + ln for ln in lines[1:])
    return (f"X:{t['tune_id']}\nT:{t.get('name', '').strip()}\nR:{t.get('type', '').strip()}\n"
            f"M:{t.get('meter', '').strip()}\nL:1/8\nK:{t.get('mode', '').strip()}\n{body}")


def build(tunes):
    jigs, first_jigs, slip, seen = [], [], [], set()
    for t in tunes:
        typ, meter = (t.get("type") or "").strip(), (t.get("meter") or "").strip()
        if not ((typ == "jig" and meter == "6/8") or (typ == "slip jig" and meter == "9/8")):
            continue
        blk = tune_block(t)
        if not blk:
            continue
        first = t.get("tune_id") not in seen
        seen.add(t.get("tune_id"))
        if typ == "jig":
            jigs.append(blk)
            if first:
                first_jigs.append((blk, t.get("mode") or ""))
        elif first:
            slip.append(blk)
    dark = [b for b, mode in first_jigs if mode.lower().endswith(("minor", "dorian"))]
    return {"jigs": jigs, "jigs_first": [b for b, _ in first_jigs], "dark": dark, "slip": slip}


def prepare(source=None, out_dir=DATA_DIR):
    if source and os.path.exists(source):
        with open(source) as f:
            tunes = json.load(f)
        origin = os.path.abspath(source)
    else:
        url = source or TUNES_URL
        with urllib.request.urlopen(url, timeout=120) as r:
            tunes = json.load(r)
        origin = url
    sets = build(tunes)
    os.makedirs(out_dir, exist_ok=True)
    for name, blocks in sets.items():
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(blocks, f)
    meta = {"source": origin, "rows": len(tunes), **{k: len(v) for k, v in sets.items()},
            "jig_chars": sum(len(b) for b in sets["jigs"])}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def load(name, data_dir=DATA_DIR):
    path = os.path.join(data_dir, f"{name}.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run `jigai prepare-data` first (see README).")
    with open(path) as f:
        return json.load(f)


def songs_subset(n, seed=0, data_dir=DATA_DIR):
    """N distinct jigs (one setting each), a fixed random draw so runs are comparable."""
    pool = load("jigs_first", data_dir)
    if n >= len(pool):
        return pool
    return random.Random(seed).sample(pool, n)


def read_abc_files(paths):
    """Extra tunes for fine-tuning: .abc files, each may hold several tunes starting with X:."""
    tunes = []
    for p in paths or []:
        with open(p) as f:
            text = f.read().replace("\r\n", "\n")
        for c in text.split("\nX:"):
            c = c.strip()
            if c:
                tunes.append(c if c.startswith("X:") else "X:" + c)
    return tunes
