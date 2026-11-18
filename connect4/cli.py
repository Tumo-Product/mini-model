"""connect4 command line: data, train, generate, score, eval."""

import argparse
import json
import os
import sys

from microgpt import blocks

from . import game

MISSING_HINT = "run `connect4 data` first (see connect4/README.md)."


def cmd_data(a):
    path = os.path.join(a.data, f"{a.name}.json")
    if os.path.exists(path):  # e.g. --name jigs: don't replace a tune corpus with games
        try:
            with open(path) as f:
                old = json.load(f)
        except ValueError:
            old = None
        if not isinstance(old, list) or (old and not str(old[0]).startswith(game.HEADER)):
            sys.exit(f"{path} already exists and isn't a Connect Four corpus: pick another --name.")
    games = game.generate_games(a.games, a.seed, a.bot, a.eps)
    os.makedirs(a.data, exist_ok=True)
    with open(path, "w") as f:
        json.dump(games, f)
    summary, _ = game.score_text("\n\n".join(games))
    print(json.dumps({"wrote": path, "games": len(games), "chars": sum(len(g) + 2 for g in games),
                      "bot": a.bot, "eps": a.eps, "seed": a.seed, "mean_moves": summary["mean_moves"],
                      "winners": summary["winners"]}, indent=1))
    if len(games) < a.games:
        print(f"note: only {len(games)} distinct games found; raise --eps for more variety", file=sys.stderr)


def cmd_train(a):
    from microgpt import train
    games = blocks.load_corpus(a.corpus, a.data, MISSING_HINT)
    if a.games:
        games = games[:a.games]
    out = a.out or os.path.join("runs", f"c4-g{len(games)}-x{a.steps}")
    samples = train.train(games, a.steps, out, seed=a.seed, device=a.device, compile_model=a.compile,
                          num_samples=a.samples, dry_run=a.dry_run, unit="games")
    if samples:
        blocks.write_blocks([blocks.first_block(s) for s in samples], os.path.join(out, "samples.txt"))


def cmd_generate(a):
    from microgpt.model import generate_from
    # start from the whole header line, so every game is headed X:c4 and the model's first character is a move
    games = generate_from(a.model, n=a.n, temperature=a.temperature, max_new_tokens=a.max_chars, seed=a.seed,
                          device=a.device, raw=a.raw, start=game.HEADER + "\n")
    blocks.emit(games, a.out, "games")


def cmd_score(a):
    with open(a.file) as f:
        text = f.read()
    pool = blocks.load_corpus(a.pool, a.data, MISSING_HINT) if a.pool else None
    summary, rows = game.score_text(text, pool)
    print(json.dumps({"summary": summary, "games": rows} if a.verbose else summary, indent=1))


def cmd_eval(a):
    import torch
    from microgpt.model import load_checkpoint
    torch.manual_seed(a.seed)
    model, vocab, _ = load_checkpoint(a.model, a.device)
    print(json.dumps(game.play_vs_bot(model, vocab, a.n, a.opponent, a.temperature, a.seed), indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog="connect4", description="Teach the MicroGPT Connect Four from games written as text.")
    # the same folder the jigai command uses, so JIGAI_DATA moves both
    p.add_argument("--data", default=os.environ.get("JIGAI_DATA", "data"), help="data folder (default: data)")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("data", help="write self-play games to data/<name>.json")
    s.add_argument("--games", type=int, default=100_000)
    s.add_argument("--bot", choices=["smart", "random"], default="smart")
    s.add_argument("--eps", type=float, default=0.1, help="chance of a random move, keeps smart games varied")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--name", default="connect4", help="corpus name (default: connect4)")
    s.set_defaults(func=cmd_data)

    s = sub.add_parser("train", help="train from scratch on a corpus of games: --games N --steps S")
    s.add_argument("--corpus", default="connect4", help="corpus name or JSON file (default: connect4)")
    s.add_argument("--games", type=int, default=0, help="use only the first N games (0 = the whole corpus)")
    s.add_argument("--steps", type=int, default=600)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--samples", type=int, default=8)
    s.add_argument("--compile", action="store_true")
    s.add_argument("--out")
    s.add_argument("--dry-run", action="store_true", help="load data and run one forward pass only")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("generate", help="have a trained model write games on its own")
    s.add_argument("--model", required=True)
    s.add_argument("-n", type=int, default=8)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--max-chars", type=int, default=500)
    s.add_argument("--raw", action="store_true", help="keep each whole 500-char generation (about ten games, the last cut off)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out")
    s.set_defaults(func=cmd_generate)

    s = sub.add_parser("score", help="check generated games (legal moves, endings, copies)")
    s.add_argument("file", help="text file of generated games, e.g. from `connect4 generate --out` (not --raw: "
                                "raw samples end mid-game, and the cut-off game would count as a mistake)")
    s.add_argument("--pool", help="training corpus name or JSON file, for the copy check (e.g. connect4)")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("eval", help="the model plays games against a bot")
    s.add_argument("--model", required=True)
    s.add_argument("-n", type=int, default=200, help="games (the model moves first in half)")
    s.add_argument("--opponent", choices=["random", "smart"], default="random")
    s.add_argument("--temperature", type=float, default=0.0, help="0 = always its most likely legal move")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_eval)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
