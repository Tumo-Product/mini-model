"""Text made of blocks that each start with "X:" (one tune, one game): split, cut, read and write. No torch."""

import json
import os


def split_blocks(text):
    """Text holding several X: blocks -> the blocks, each stripped and starting with X:."""
    blocks = []
    for c in text.replace("\r\n", "\n").split("\nX:"):
        c = c.strip()
        if c:
            blocks.append(c if c.startswith("X:") else "X:" + c)
    return blocks


def first_block(raw):
    """The first block of one generation. A 500-character sample usually runs on into a second and a third."""
    i = raw.find("\nX:", 2)
    return raw[:i] if i > 0 else raw


def write_blocks(blocks, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n\n".join(b.strip() for b in blocks) + "\n")


def emit(blocks, path=None, unit="blocks"):
    """Write the blocks to path, or print them when there is no path."""
    if path:
        write_blocks(blocks, path)
        print(f"wrote {len(blocks)} {unit} to {path}")
    else:
        print("\n\n".join(b.strip() for b in blocks))


def load_corpus(name_or_path, data_dir="data", hint=""):
    """A corpus is a JSON list of blocks: give its path, or its name for <data_dir>/<name>.json.

    Anything ending in .json or holding a folder separator is a path; a bare word is a name. Going by the
    spelling, not by what exists on disk, keeps the name "connect4" apart from the connect4/ folder and lets a
    mistyped path be reported as that path. hint says how to make a missing named corpus."""
    seps = [s for s in (os.sep, os.altsep) if s]
    is_path = name_or_path.endswith(".json") or any(s in name_or_path for s in seps)
    path = name_or_path if is_path else os.path.join(data_dir, f"{name_or_path}.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} not found" + (f": {hint}" if hint and not is_path else "."))
    with open(path) as f:
        return json.load(f)
