"""jigai command line: prepare-data, train, finetune, generate, score."""

import argparse
import json
import os
import sys



def _write_abc(tunes, path, raw=False):
    from .score import first_tune
    if not raw:
        tunes = [first_tune(t) for t in tunes]  # one tune per generation, as the activity scores them
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n\n".join(t.strip() for t in tunes) + "\n")


def cmd_prepare(a):
    from . import data
    meta = data.prepare(a.source, a.out)
    print(json.dumps(meta, indent=1))


def cmd_train(a):
    from . import data, train
    if a.songs and a.songs > 0 and a.corpus == "jigs_first":
        blocks = data.songs_subset(a.songs, a.seed, a.data)
    else:
        blocks = data.load(a.corpus, a.data)
        if a.songs:
            blocks = blocks[:a.songs]
    vocab = None
    if a.vocab_from:
        with open(os.path.join(a.vocab_from, "config.json")) as f:
            vocab = json.load(f)["vocab"]
    out = a.out or os.path.join("runs", f"s{len(blocks)}-x{a.steps}")
    tunes = train.train(blocks, a.steps, out, vocab, a.seed, a.device, a.compile, a.samples, a.dry_run)
    if tunes:
        _write_abc(tunes, os.path.join(out, "samples.abc"))


def cmd_finetune(a):
    from . import data, train
    set_blocks = data.load(a.set, a.data) if a.set else []
    extra = data.read_abc_files(a.abc)
    if not set_blocks and not extra:
        sys.exit("Give --set dark|slip and/or --abc files to fine-tune on.")
    out = a.out or os.path.join("runs", f"ft-{a.set or 'abc'}-{os.path.basename(os.path.normpath(a.base))}")
    tunes = train.finetune(a.base, out, set_blocks, a.set, extra, a.steps, a.lr, a.seed, a.device, a.samples, a.dry_run)
    if tunes:
        _write_abc(tunes, os.path.join(out, "samples.abc"))


def cmd_generate(a):
    import torch
    from .model import load_checkpoint, sample
    torch.manual_seed(a.seed)
    model, vocab, _ = load_checkpoint(a.model, a.device)
    tunes = sample(model, vocab, n=a.n, temperature=a.temperature, max_new_tokens=a.max_chars)
    if not a.raw:
        from .score import first_tune
        tunes = [first_tune(t) for t in tunes]
    if a.out:
        _write_abc(tunes, a.out, raw=True)
        print(f"wrote {len(tunes)} tunes to {a.out}")
    else:
        print("\n\n".join(t.strip() for t in tunes))


def _split_tunes(text):
    parts = [p for p in text.replace("\r\n", "\n").split("\nX:")]
    return [p if p.startswith("X:") else "X:" + p for p in parts if p.strip()]


def cmd_score(a):
    from . import data, score
    with open(a.abc) as f:
        tunes = _split_tunes(f.read())
    pool = []
    if a.pool:
        blocks = data.load_pool(a.pool, a.data)
        pool = [{"title": b.split("\n", 2)[1][2:] if b.count("\n") > 1 else None, "abc": b} for b in blocks]
    rows = []
    for t in tunes:
        first = score.first_tune(t)
        two = score.trim_to_two_parts(first)
        copied = score.plagiarism(first, pool)["frac"] if pool else 0.0
        g = (score.goodness_meter if a.meter_aware else score.goodness)(two, pool, copied)
        rows.append({"goodness": round(g["score"] * 100), "copied": round(copied * 100), "floor": score.passes_floor(t),
                     "dark": score.in_set(first, "dark"), "slip": score.in_set(first, "slip")})
    n = len(rows) or 1
    summary = {"tunes": len(rows), "mean_goodness": round(sum(r["goodness"] for r in rows) / n, 1),
               "floor_pass_pct": round(100 * sum(r["floor"] for r in rows) / n),
               "dark": score.set_count([score.first_tune(t) for t in tunes], "dark"),
               "slip": score.set_count([score.first_tune(t) for t in tunes], "slip"),
               "copies_35pct_plus": sum(r["copied"] >= 35 for r in rows),
               "pool": a.pool or "none (style and copy checks need --pool)"}
    print(json.dumps({"summary": summary, "tunes": rows} if a.verbose else summary, indent=1))


def cmd_c4_data(a):
    from . import connect4
    path = os.path.join(a.data, f"{a.name}.json")
    if os.path.exists(path):  # e.g. --name jigs: don't replace a tune corpus with games
        try:
            with open(path) as f:
                old = json.load(f)
        except ValueError:
            old = None
        if not isinstance(old, list) or (old and not str(old[0]).startswith(connect4.HEADER)):
            sys.exit(f"{path} already exists and isn't a Connect Four corpus: pick another --name.")
    blocks = connect4.generate_games(a.games, a.seed, a.bot, a.eps)
    os.makedirs(a.data, exist_ok=True)
    with open(path, "w") as f:
        json.dump(blocks, f)
    summary, _ = connect4.score_text("\n\n".join(blocks))
    print(json.dumps({"wrote": path, "games": len(blocks), "chars": sum(len(b) + 2 for b in blocks),
                      "bot": a.bot, "eps": a.eps, "seed": a.seed, "mean_moves": summary["mean_moves"],
                      "winners": summary["winners"]}, indent=1))
    if len(blocks) < a.games:
        print(f"note: only {len(blocks)} distinct games found; raise --eps for more variety", file=sys.stderr)


def cmd_c4_score(a):
    from . import connect4, data
    with open(a.file) as f:
        text = f.read()
    pool = data.load_pool(a.pool, a.data) if a.pool else None
    summary, rows = connect4.score_text(text, pool)
    print(json.dumps({"summary": summary, "games": rows} if a.verbose else summary, indent=1))


def cmd_c4_eval(a):
    import torch
    from . import connect4
    from .model import load_checkpoint
    torch.manual_seed(a.seed)
    model, vocab, _ = load_checkpoint(a.model, a.device)
    print(json.dumps(connect4.play_vs_bot(model, vocab, a.n, a.opponent, a.temperature, a.seed), indent=1))


def read_tunes(paths):
    import re
    tunes = []
    for path in paths:
        text = open(path, encoding="utf-8").read().replace("\r\n", "\n").strip()
        parts = [t.strip() for t in re.split(r"\n(?=X:)", text) if t.strip()]
        if len(parts) <= 1:
            parts = [t.strip() for t in re.split(r"\n\s*\n", text) if t.strip()]
        tunes += parts
    return tunes


def cmd_play(a):
    import functools
    import http.server
    import webbrowser
    tunes = json.dumps(read_tunes(a.abc)).encode("utf-8")
    root = os.path.join(os.path.dirname(__file__), "player")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] == "/tunes.json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(tunes)))
                self.end_headers()
                self.wfile.write(tunes)
                return
            super().do_GET()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), functools.partial(Handler, directory=root))
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"{len(json.loads(tunes))} tune(s) loaded. Player at {url}  (works offline; Ctrl-C to stop)")
    if not a.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv=None):
    p = argparse.ArgumentParser(prog="jigai", description="Train, fine-tune, sample and score the Jig AI models.")
    p.add_argument("--data", default=os.environ.get("JIGAI_DATA", "data"), help="data folder (default: data)")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare-data", help="rebuild data/ from The Session dump (data/ already ships with the repo)")
    s.add_argument("--source", help="path or URL of tunes.json (default: the TheSession-data GitHub dump)")
    s.add_argument("--out", default="data")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("train", help="train from scratch: --songs N --steps S")
    s.add_argument("--songs", type=int, default=0, help="N distinct jigs (0 = the whole corpus)")
    s.add_argument("--steps", type=int, default=2400)
    s.add_argument("--corpus", default="jigs_first", help="jigs_first (one setting per tune) or jigs (all settings, the strong model)")
    s.add_argument("--vocab-from", help="reuse a checkpoint's vocabulary so the new model is compatible with it")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--samples", type=int, default=8)
    s.add_argument("--compile", action="store_true")
    s.add_argument("--out")
    s.add_argument("--dry-run", action="store_true", help="load data and run one forward pass only")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("finetune", help="fine-tune a trained model on a style set and/or extra ABC files")
    s.add_argument("--base", default="weights/tsALL-s6000-v1")
    s.add_argument("--set", choices=["dark", "slip"])
    s.add_argument("--abc", nargs="*", default=[], help="extra .abc files (e.g. tunes you liked)")
    s.add_argument("--steps", type=int)
    s.add_argument("--lr", type=float)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--samples", type=int, default=16)
    s.add_argument("--out")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_finetune)

    s = sub.add_parser("generate", help="sample tunes from a trained model")
    s.add_argument("--model", default="weights/tsALL-s6000-v1")
    s.add_argument("-n", type=int, default=8)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--max-chars", type=int, default=500)
    s.add_argument("--raw", action="store_true", help="keep each whole 500-char generation (often 2-3 tunes run together)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out")
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("score", help="score generated tunes the way the activity does")
    s.add_argument("abc", help=".abc file of generated tunes")
    s.add_argument("--pool", help="jigs | jigs_first | dark | slip, or a JSON list of ABC strings: the real tunes "
                                  "to measure style and copying against (the activity used its 482 base jigs)")
    s.add_argument("--meter-aware", action="store_true", help="judge bars against each tune's own meter (slip jigs)")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("play", help="open the jig player in your browser with these tunes loaded")
    s.add_argument("abc", nargs="*", help=".abc files to load (none = an empty player you can drop files on)")
    s.add_argument("--port", type=int, default=0, help="local port for the player (default: any free port)")
    s.add_argument("--no-open", action="store_true", help="print the player address without opening a browser")
    s.set_defaults(func=cmd_play)

    s = sub.add_parser("c4-data", help="Connect Four: write self-play games to data/<name>.json for `train --corpus`")
    s.add_argument("--games", type=int, default=100_000)
    s.add_argument("--bot", choices=["smart", "random"], default="smart")
    s.add_argument("--eps", type=float, default=0.1, help="chance of a random move, keeps smart games varied")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--name", default="connect4", help="corpus name (default: connect4)")
    s.set_defaults(func=cmd_c4_data)

    s = sub.add_parser("c4-score", help="Connect Four: check generated games (legal moves, endings, copies)")
    s.add_argument("file", help="text file of generated games, e.g. from `jigai generate --out` (not --raw: "
                                "raw samples end mid-game, and the cut-off game would count as a mistake)")
    s.add_argument("--pool", help="training corpus name or JSON file, for the copy check (e.g. connect4)")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_c4_score)

    s = sub.add_parser("c4-eval", help="Connect Four: the model plays games against a bot")
    s.add_argument("--model", required=True)
    s.add_argument("-n", type=int, default=200, help="games (the model moves first in half)")
    s.add_argument("--opponent", choices=["random", "smart"], default="random")
    s.add_argument("--temperature", type=float, default=0.0, help="0 = always its most likely legal move")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_c4_eval)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
