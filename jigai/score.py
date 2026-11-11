"""Python port of the activity's scoring (irish-music-ai: jigLogic.jsx, deck/abc.js, deck/data.js, deck/sets.js).

goodness()       0..1 "good original jig": style 30%, resolves 25%, 6/8 bars 15%, smooth 12%, length 10%,
                 motif repetition 8%; zeroed when >=35% of the tune is copied verbatim from the pool.
goodness_meter() the same, but bars and length are judged against the tune's own meter (slip jigs = 9/8).
plagiarism()     longest verbatim run shared with any pool tune / tune body length.
passes_floor()   reads as music (valid key), lands on its home note in its first part or two, 6-24 bars.
in_set/set_count dark = minor/dorian/aeolian key; slip = 9/8 with >=75% of inner bars holding 9 eighths.

tests/test_score_port.py checks every function against the JS originals via node.
"""

import re

_DIA = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
_STRIP = re.compile(r'"[^"]*"|\{[^}]*\}|![^!]*!')
_NOTE_RE = re.compile(r"[_^=]*([A-Ga-gz])[,']*(\d*)(/*)(\d*)")


def body_of(abc):
    m = re.search(r"^K:[^\n]*\n", abc, re.M)
    return abc[abc.index(m.group(0)) + len(m.group(0)):] if m else abc


def key_of(abc):
    m = re.search(r"^K:([^\n]*)", abc, re.M)
    return m.group(1).strip() if m else "D"


def meter_of(abc):
    m = re.search(r"^M:([^\n]*)", abc, re.M)
    return m.group(1).strip() if m else "6/8"


def _bar_durations(body):
    clean = _STRIP.sub("", body).replace("\n", " ")
    durs = []
    for chunk in re.split(r"\|+|:+", clean):
        d = 0.0
        for m in _NOTE_RE.finditer(chunk):
            v = float(m.group(2)) if m.group(2) else 1.0
            if m.group(3):
                v = v / (float(m.group(4)) if m.group(4) else 2 ** len(m.group(3)))
            d += v
        if d > 0:
            durs.append(d)
    return durs


def jig_rules(abc):
    body = body_of(abc)
    durs = _bar_durations(body)
    inner = durs[1:-1] if len(durs) > 2 else durs
    six = (sum(1 for d in inner if abs(d - 6) < 0.01) / len(inner)) if inner else 0
    parts = []
    for seg in re.split(r":\||\|\||\|:|::", body.replace("\n", " ")):
        bars = len([b for b in seg.split("|") if re.search(r"[A-Ga-g]", b)])
        if bars >= 4:
            parts.append(bars)
    eight = len(parts) >= 2 and sum(1 for p in parts if p in (8, 9, 16)) / len(parts) >= 0.5
    return {"sixOk": six >= 0.75, "eightBar": eight}


def plagiarism(abc, pool):
    """pool = list of {"title", "abc"}. Returns {"frac", "span", "title"}."""
    g = body_of(abc)
    n = len(g)
    if not n or not pool:
        return {"frac": 0.0, "span": "", "title": None}
    bodies = [(t.get("title"), body_of(t["abc"])) for t in pool]
    pool_text = "".join(b for _, b in bodies)

    def find_at(L):
        if L <= 0 or L > n:
            return None
        for i in range(0, n - L + 1):
            s = g[i:i + L]
            if s in pool_text:
                return s
        return None

    lo, hi, best = 1, n, ""
    while lo <= hi:
        mid = (lo + hi) >> 1
        s = find_at(mid)
        if s:
            best, lo = s, mid + 1
        else:
            hi = mid - 1
    title = next((t for t, b in bodies if best and best in b), None) if best else None
    return {"frac": len(best) / n, "span": best, "title": title}


def _pitch_seq(abc):
    body = _STRIP.sub("", body_of(abc))
    seq = []
    for m in re.finditer(r"[_^=]*([A-Ga-gz])([,']*)", body):
        letter = m.group(1)
        if letter in ("z", "Z"):
            continue
        lower = "a" <= letter <= "g"
        up, dn = m.group(2).count("'"), m.group(2).count(",")
        seq.append(_DIA[letter.upper()] + (7 if lower else 0) + 7 * up - 7 * dn)
    return seq


_KG = 8


def _interval_grams(abc):
    s = _pitch_seq(abc)
    iv = [s[i] - s[i - 1] for i in range(1, len(s))]
    grams = [",".join(str(x) for x in iv[i:i + _KG]) for i in range(0, len(iv) - _KG + 1)]
    return grams, len(iv)


_style_cache = {}


def _style_pool(pool):
    key = (id(pool), len(pool))
    if key not in _style_cache:
        grams = set()
        for t in pool:
            grams.update(_interval_grams(t["abc"])[0])
        _style_cache.clear()
        _style_cache[key] = grams
    return _style_cache[key]


def style_match(abc, pool):
    if not pool:
        return 0
    grams, iv_len = _interval_grams(abc)
    if iv_len < _KG:
        return 0
    have = _style_pool(pool)
    covered = [False] * iv_len
    for i, g in enumerate(grams):
        if g in have:
            for j in range(i, i + _KG):
                covered[j] = True
    return sum(covered) / iv_len


def _split_parts(body):
    parts, cur = [], ""
    for tk in re.split(r"(:\||\|\|)", body):
        cur += tk
        if tk in (":|", "||"):
            parts.append(cur)
            cur = ""
    if cur.strip():
        parts.append(cur)
    return [p for p in parts if len(re.findall(r"[A-Ga-g]", p)) > 3]


def trim_to_two_parts(abc):
    if not abc:
        return abc
    body = body_of(abc)
    header = abc[:abc.index(body)]
    b = body
    second = b.find("X:")
    if second > 0:
        b = b[:second]
    b = re.sub(r'"[^"]*"', "", b)
    parts = _split_parts(b)
    kept = "".join(parts[:2]) if parts else b
    kept = kept.rstrip()
    if not re.search(r"(:\||\|\|?)\Z", kept):
        kept += " :|"
    return (header + kept).rstrip()


def _cadence_score(abc):
    k = key_of(abc)
    tonic = _DIA.get((k[0] if k else "D").upper(), 1)
    parts = _split_parts(body_of(abc))
    if not parts:
        return 0
    hits = 0
    for p in parts:
        s = _pitch_seq(p)
        if not s:
            continue
        last = s[-1] % 7
        if last == tonic or last == (tonic + 4) % 7:
            hits += 1
    return hits / len(parts)


def _self_rep_score(abc):
    s = _pitch_seq(abc)
    win = [",".join(str(x) for x in s[i:i + 4]) for i in range(0, len(s) - 3)]
    if not win:
        return 0
    seen, rep = set(), 0
    for w in win:
        if w in seen:
            rep += 1
        else:
            seen.add(w)
    return rep / len(win)


def goodness(abc, pool, copied=None):
    style = style_match(abc, pool)
    cadence = _cadence_score(abc)
    six = 1 if jig_rules(abc)["sixOk"] else 0
    s = _pitch_seq(abc)
    iv = [abs(s[i] - s[i - 1]) for i in range(1, len(s))]
    big_leap = (sum(1 for a in iv if a >= 5) / len(iv)) if iv else 1
    smooth = max(0, min(1, 1 - big_leap / 0.10))
    n_bars = body_of(abc).count("|")
    len_ok = max(0, 1 - abs(min(n_bars, 40) - 18) / 18) if n_bars else 0
    rep = _self_rep_score(abc)
    if copied is None:
        copied = plagiarism(abc, pool)["frac"]
    original = 0 if copied >= 0.35 else 1
    score = original * (0.30 * style + 0.25 * cadence + 0.15 * six + 0.12 * smooth + 0.10 * len_ok + 0.08 * rep)
    return {"score": score, "style": style, "cadence": cadence, "six": six, "smooth": smooth,
            "lenOk": len_ok, "rep": rep, "copied": copied, "original": original}


# ---- meter-aware scoring and the style sets (deck/sets.js) ----

def eighths_per_bar(abc):
    m = re.match(r"^(\d+)\s*/\s*(\d+)$", meter_of(abc))
    return int(m.group(1)) * 8 / int(m.group(2)) if m else 6


def meter_fit(abc):
    want = eighths_per_bar(abc)
    durs = _bar_durations(body_of(abc))
    inner = durs[1:-1] if len(durs) > 2 else durs
    return (sum(1 for d in inner if abs(d - want) < 0.01) / len(inner)) if inner else 0


def goodness_meter(abc, pool, copied=None):
    g = goodness(abc, pool, copied)
    fit = 1 if meter_fit(abc) >= 0.75 else 0
    n_bars = body_of(abc).count("|")
    target = 16 if eighths_per_bar(abc) == 9 else 18
    len_ok = max(0, 1 - abs(min(n_bars, 40) - target) / target) if n_bars else 0
    score = g["original"] * (0.30 * g["style"] + 0.25 * g["cadence"] + 0.15 * fit + 0.12 * g["smooth"] + 0.10 * len_ok + 0.08 * g["rep"])
    return {**g, "score": score, "fit": fit, "lenOk": len_ok}


_DARK_MODE = re.compile(r"(m|min|minor|dor|dorian|aeo|aeolian)", re.I)


def in_set(abc, which):
    if not abc:
        return False
    if which == "dark":
        k = re.match(r"^[A-G][#b]?(.*)$", re.sub(r"\s+", "", key_of(abc)))
        return bool(k and _DARK_MODE.fullmatch(k.group(1)))
    if which == "slip":
        return eighths_per_bar(abc) == 9 and meter_fit(abc) >= 0.75
    return False


def set_count(abcs, which):
    lst = [a for a in (abcs or []) if a]
    return {"n": sum(1 for a in lst if in_set(a, which)), "of": len(lst)}


# ---- the floor rule (deck/abc.js + deck/data.js) ----

_VALID_KEY = re.compile(r"[A-G][b#]?(major|minor|dorian|mixolydian|aeolian|ionian|lydian|phrygian|locrian|maj|min|m|dor|mix)?", re.I)
_PART_END = re.compile(r":\||\|\]|\|\|")


def first_tune(raw):
    i = raw.find("\nX:", 2)
    return raw[:i] if i > 0 else raw


def parse_generated(raw):
    first = first_tune(raw)
    key_lines = list(re.finditer(r"^K:([^\n]*)\n?", first, re.M))
    keys = [m.group(1).strip() for m in key_lines if _VALID_KEY.fullmatch(m.group(1).strip())]
    last = key_lines[-1] if key_lines else None
    body = first[last.start() + len(last.group(0)):] if last else first
    return {"key": keys[-1] if keys else None, "body": body, "readable": bool(key_lines), "first": first}


def _flatten(body):
    flat = body.replace("\r", "")
    flat = re.sub(r"^\s*([A-Za-z]):([^\n]*)$", r"[\1:\2]", flat, flags=re.M)
    return re.sub(r"\n\s*", " ", flat).strip()


def a_part(body):
    flat = _flatten(body)
    m = _PART_END.search(flat)
    if m:
        seg = flat[:m.start()]
        e1m = re.search(r"\|\s*\[?1|\[1", seg)
        e1 = e1m.start() if e1m else -1
        if e1 >= 0 and m.group(0) == ":|":
            rest = flat[m.start() + 2:]
            e2 = re.match(r"\s*\[?2\s*", rest)
            if e2:
                after = rest[e2.end():]
                stop_m = re.search(r"\|\||\|\]|\|:|:\|", after)
                ending2 = after[:stop_m.start()] if stop_m else after
                ending2 = re.sub(r"\|\s*\Z", "", ending2)
                part = f"{seg[:e1]}| {ending2}"
            else:
                part = seg[:e1]
        else:
            part = seg
    else:
        part = "|".join([b for b in flat.split("|") if b.strip()][:8])
    part = part.replace("|:", "|").replace("::", "|")
    part = re.sub(r"^\s*\|\s*", "", part, count=1)
    part = re.sub(r"\|\s*\Z", "", part)
    return f"{part} |]"


def _first_part_end(flat):
    m = _PART_END.search(flat)
    if not m:
        return -1
    seg = flat[:m.start()]
    if m.group(0) == ":|" and re.search(r"\|\s*\[?1|\[1", seg):
        rest = flat[m.start() + 2:]
        e2 = re.match(r"\s*\[?2\s*", rest)
        if e2:
            after = rest[e2.end():]
            stop_m = re.search(r"\|\||\|\]|\|:|:\|", after)
            return m.start() + 2 + e2.end() + stop_m.start() if stop_m else len(flat)
    return m.end()


def _last_note_letter(part):
    p = re.sub(r'"[^"]*"', "", part)
    p = re.sub(r"![^!]*!", "", p)
    p = re.sub(r"\[[A-Za-z]:[^\]]*\]", "", p)
    notes = re.findall(r"[A-Ga-g](?=[,']*[0-9/]*[^A-Ga-g]*\Z)", p)
    return notes[-1].upper() if notes else None


def lands_home(part, key):
    return bool(key) and _last_note_letter(part) == key.strip()[0].upper()


def home_part(body, key):
    a = a_part(body)
    if not key or lands_home(re.sub(r"\|\]\s*\Z", "", a), key):
        return {"part": a, "parts": 1}
    flat = _flatten(body)
    end = _first_part_end(flat)
    if end < 0 or end >= len(flat):
        return {"part": a, "parts": 1}
    rest = re.sub(r"^\s*\|?\]?", "", flat[end:], count=1)
    if not re.search(r"[A-Ga-g]", rest):
        return {"part": a, "parts": 1}
    b = a_part(rest)
    if not lands_home(re.sub(r"\|\]\s*\Z", "", b), key):
        return {"part": a, "parts": 1}
    a_open = re.sub(r"\|\]\s*\Z", "", a)
    return {"part": f"{a_open}| {b}", "parts": 2}


def passes_floor(raw):
    g = parse_generated(raw)
    if not g["readable"] or not g["key"]:
        return False
    hp = home_part(g["body"], g["key"])
    bars = len([b for b in hp["part"].split("|") if re.search(r"[A-Ga-g]", b)])
    return lands_home(re.sub(r"\|\]\s*\Z", "", hp["part"]), g["key"]) and 6 <= bars <= 24
