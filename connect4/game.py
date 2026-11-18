"""Connect Four as text for MicroGPT: generate games, score what the model writes, play it against bots.

One game is one training block. It starts with "X:" like every block the model trains on:

  X:c4
  4453342271
  W:1

The second line is the moves, one character per move: the column 1-7 the piece was dropped in, player 1 first.
W: is the result: 1 or 2 for the winner, 0 for a draw (full board). A whole game is at most 51 characters, so it
always fits in the model's 256-character window.

generate_games()  self-play games from a bot ("smart" by default), duplicates removed
score_text()      checks generated games: legal moves, clean endings, correct W: line, verbatim copies
play_vs_bot()     the model plays real games against a bot (random or smart) and reports win/draw/loss
"""

import random

from microgpt.blocks import split_blocks

ROWS, COLS = 6, 7
HEADER = "X:c4"
COLUMN_CHARS = "".join(str(c + 1) for c in range(COLS))
CENTER_WEIGHTS = (1, 2, 3, 4, 3, 2, 1)
MISTAKES = ("played_after_end", "stopped_early", "full_column", "not_a_column")


class Game:
    """Board as two bitboards, 7 bits per column (6 rows + a spare bit so shifts can't wrap between columns)."""

    def __init__(self, moves=""):
        self.boards = [0, 0]
        self.heights = [0] * COLS
        self.moves = []
        self.winner = None  # None while playing, then 1 / 2, or 0 for a draw
        for ch in moves:
            self.play(int(ch) - 1)

    @property
    def turn(self):
        return len(self.moves) % 2

    @property
    def over(self):
        return self.winner is not None

    def legal(self):
        return [] if self.over else [c for c in range(COLS) if self.heights[c] < ROWS]

    def is_legal(self, col):
        return not self.over and 0 <= col < COLS and self.heights[col] < ROWS

    def _bit(self, col):
        return 1 << (col * (ROWS + 1) + self.heights[col])

    def wins_with(self, col, player):
        return _has_four(self.boards[player] | self._bit(col))

    def play(self, col):
        if not self.is_legal(col):
            raise ValueError(f"illegal move {col + 1} after {self.text_moves()!r}")
        p = self.turn
        self.boards[p] |= self._bit(col)
        self.heights[col] += 1
        self.moves.append(col)
        if _has_four(self.boards[p]):
            self.winner = p + 1
        elif len(self.moves) == ROWS * COLS:
            self.winner = 0

    def text_moves(self):
        return "".join(str(c + 1) for c in self.moves)

    def record(self):
        return f"{HEADER}\n{self.text_moves()}\nW:{self.winner}"


def _has_four(b):
    for shift in (1, ROWS + 1, ROWS, ROWS + 2):  # vertical, horizontal, two diagonals
        m = b & (b >> shift)
        if m & (m >> (2 * shift)):
            return True
    return False


def random_move(game, rng):
    return rng.choice(game.legal())


def smart_move(game, rng, eps=0.0):
    """Win now, else block, else avoid handing the opponent a win on top, preferring the centre.

    With probability eps it plays a random legal move instead, which keeps self-play games varied."""
    legal = game.legal()
    if eps and rng.random() < eps:
        return rng.choice(legal)
    me, them = game.turn, 1 - game.turn
    for c in legal:
        if game.wins_with(c, me):
            return c
    blocks = [c for c in legal if game.wins_with(c, them)]
    if blocks:
        return rng.choice(blocks)
    safe = [c for c in legal if not _gives_win_above(game, c, them)] or legal
    return rng.choices(safe, weights=[CENTER_WEIGHTS[c] for c in safe])[0]


def _gives_win_above(game, col, them):
    if game.heights[col] + 1 >= ROWS:
        return False
    above = 1 << (col * (ROWS + 1) + game.heights[col] + 1)
    return _has_four(game.boards[them] | above)


def bot_move(name, game, rng, eps=0.0):
    if name == "random":
        return random_move(game, rng)
    if name == "smart":
        return smart_move(game, rng, eps)
    raise ValueError(f"unknown bot {name!r}: use random or smart")


def generate_games(n, seed=0, bot="smart", eps=0.1, max_tries=None):
    """n distinct self-play games as text blocks. Stops early (with fewer) if the bot can't find n distinct games."""
    rng = random.Random(seed)
    seen, blocks = set(), []
    tries, max_tries = 0, max_tries or n * 5
    while len(blocks) < n and tries < max_tries:
        tries += 1
        g = Game()
        while not g.over:
            g.play(bot_move(bot, g, rng, eps))
        moves = g.text_moves()
        if moves not in seen:
            seen.add(moves)
            blocks.append(g.record())
    return blocks


def check_game(block):
    """Replay one generated game. Never raises: anything that isn't a legal game is counted, not crashed on.

    An illegal move is skipped and the replay carries on, so `legal` counts every legal move in the line, not only
    the ones before the first mistake. Once the game has ended, every further move is illegal."""
    lines = block.strip().split("\n")
    moves = lines[1].strip() if len(lines) > 1 else ""
    claimed = next((ln[2:].strip() for ln in lines[2:] if ln.startswith("W:")), None)
    g, legal, mistake = Game(), 0, None
    for ch in moves:
        col = COLUMN_CHARS.find(ch)  # -1 for anything that isn't a column
        if g.is_legal(col):
            g.play(col)
            legal += 1
        elif mistake is None:
            mistake = "played_after_end" if g.over else "not_a_column" if col < 0 else "full_column"
    clean = legal == len(moves) and g.over
    if mistake is None and not clean:
        mistake = "stopped_early"
    return {"moves": moves, "length": len(moves), "legal": legal, "clean": clean, "first_mistake": mistake,
            "winner": g.winner if clean else None, "claimed": claimed,
            "result_ok": clean and claimed == str(g.winner)}


def score_text(text, pool=None):
    """Summary + per-game rows for every game in the text. pool = training blocks, for the copy check."""
    rows = [check_game(b) for b in split_blocks(text)]
    pool_moves = {b.split("\n")[1] for b in pool or [] if b.count("\n") >= 1}
    for r in rows:
        r["copied"] = r["moves"] in pool_moves if pool else None
    n = len(rows) or 1
    total_moves = sum(r["length"] for r in rows) or 1
    clean = [r for r in rows if r["clean"]]
    summary = {
        "games": len(rows),
        "legal_move_pct": round(100 * sum(r["legal"] for r in rows) / total_moves, 1),
        "clean_game_pct": round(100 * len(clean) / n, 1),
        "result_correct_pct": round(100 * sum(r["result_ok"] for r in clean) / (len(clean) or 1), 1),
        "mean_moves": round(sum(r["length"] for r in rows) / n, 1),
        "copied_pct": round(100 * sum(bool(r["copied"]) for r in rows) / n, 1) if pool else None,
        "winners": {w: sum(r["winner"] == w for r in clean) for w in (1, 2, 0)},
        "first_mistakes": {k: sum(r["first_mistake"] == k for r in rows) for k in MISTAKES},
    }
    return summary, rows


def play_vs_bot(model, vocab, n=100, opponent="random", temperature=0.0, seed=0, eps=0.0):
    """The model plays n games against a bot, first player in half of them. Games run in a batch, one ply at a time.

    Each model move: the raw top choice is recorded (was it a legal column?), then the move is picked among legal
    columns only, so an illegal guess never ends a game early."""
    import torch

    device = next(model.parameters()).device
    model.eval()
    rng = random.Random(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    col_ids = [vocab.stoi.get(str(c + 1)) for c in range(COLS)]
    if None in col_ids:
        raise SystemExit("this model's vocabulary has no column digits 1-7: was it trained on Connect Four games?")
    games = [Game() for _ in range(n)]
    model_side = [i % 2 for i in range(n)]
    raw_legal = raw_total = 0
    while not all(g.over for g in games):
        mine = [i for i, g in enumerate(games) if not g.over and g.turn == model_side[i]]
        for i, g in enumerate(games):
            if not g.over and g.turn != model_side[i]:
                g.play(bot_move(opponent, g, rng, eps))
        if not mine:
            continue
        ctx = [vocab.encode(f"{HEADER}\n{games[i].text_moves()}") for i in mine]
        with torch.no_grad():
            logits, _ = model(torch.tensor(ctx, dtype=torch.long, device=device))
        last = logits[:, -1, :].float().cpu()
        top = last.argmax(dim=-1).tolist()
        for row, i in enumerate(mine):
            g = games[i]
            legal = g.legal()
            raw_total += 1
            raw_legal += top[row] in [col_ids[c] for c in legal]
            scores = last[row, [col_ids[c] for c in legal]]
            if temperature > 0:
                pick = torch.multinomial(torch.softmax(scores / temperature, dim=-1), 1, generator=gen).item()
            else:
                pick = scores.argmax().item()
            g.play(legal[pick])
    out = {"games": n, "opponent": opponent, "top_choice_legal_pct": round(100 * raw_legal / (raw_total or 1), 1)}
    for label, side in (("as_first", 0), ("as_second", 1), ("overall", None)):
        gs = [(g, s) for g, s in zip(games, model_side) if side is None or s == side]
        won = sum(g.winner == s + 1 for g, s in gs)
        drew = sum(g.winner == 0 for g, _ in gs)
        total = len(gs) or 1  # with -n 1 the model never moves second
        out[label] = {"games": len(gs), "win_pct": round(100 * won / total, 1),
                      "draw_pct": round(100 * drew / total, 1),
                      "loss_pct": round(100 * (len(gs) - won - drew) / total, 1)}
    return out
