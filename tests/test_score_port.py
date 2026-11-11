"""The Python scoring must agree with the activity's JavaScript (tests/js_ref/, copied from irish-music-ai).

Runs every tune through both and compares field by field. Needs `node` on PATH (skipped otherwise).
Add generated tunes to tests/fixtures/*.abc to widen the check.
"""

import glob
import json
import os
import shutil
import subprocess
import tempfile

import pytest

from jigai import score
from jigai.cli import _split_tunes

HERE = os.path.dirname(__file__)


def _tunes():
    tunes = []
    for p in sorted(glob.glob(os.path.join(HERE, "fixtures", "*.abc"))):
        with open(p) as f:
            tunes += _split_tunes(f.read())
    return tunes


def _pool():
    with open(os.path.join(HERE, "fixtures", "handmade.abc")) as f:
        return [{"title": f"t{i}", "abc": t} for i, t in enumerate(_split_tunes(f.read())[:3])]


def _python(tunes, pool):
    out = []
    for t in tunes:
        two = score.trim_to_two_parts(t)
        g = score.goodness(two, pool)
        gm = score.goodness_meter(two, pool)
        out.append({"two": two, "score": g["score"], "style": g["style"], "cadence": g["cadence"], "six": g["six"],
                    "smooth": g["smooth"], "lenOk": g["lenOk"], "rep": g["rep"], "copied": score.plagiarism(t, pool)["frac"],
                    "meter_score": gm["score"], "fit": gm["fit"], "floor": score.passes_floor(t),
                    "dark": score.in_set(t, "dark"), "slip": score.in_set(t, "slip"), "meterFit": score.meter_fit(t),
                    "sixOk": score.jig_rules(t)["sixOk"]})
    return {"out": out, "dark": score.set_count(tunes, "dark"), "slip": score.set_count(tunes, "slip")}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_python_matches_javascript():
    tunes, pool = _tunes(), _pool()
    assert len(tunes) >= 7
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"tunes": tunes, "pool": pool}, f)
        path = f.name
    js = json.loads(subprocess.check_output(["node", os.path.join(HERE, "js_ref", "check.mjs"), path], text=True))
    py = _python(tunes, pool)
    assert py["dark"] == js["dark"] and py["slip"] == js["slip"]
    for i, (a, b) in enumerate(zip(py["out"], js["out"])):
        for k, v in b.items():
            if isinstance(v, float) or isinstance(a[k], float):
                assert a[k] == pytest.approx(v, abs=1e-9), f"tune {i} field {k}: py {a[k]} js {v}"
            else:
                assert a[k] == v, f"tune {i} field {k}: py {a[k]!r} js {v!r}"


def test_set_membership_basics():
    assert score.in_set("X:1\nM:6/8\nK:Edorian\nEBB BAG|", "dark")
    assert score.in_set("X:1\nM:6/8\nK:Am\nABc|", "dark")
    assert not score.in_set("X:1\nM:6/8\nK:Dmajor\nDFA|", "dark")
    slip = "X:1\nM:9/8\nL:1/8\nK:G\n|GAB dBG AFD|GAB dBG A2G|GAB dBG AFD|BAG AFD G3|"
    assert score.in_set(slip, "slip")
    assert score.set_count([slip, "X:1\nK:D\nDFA|"], "slip") == {"n": 1, "of": 2}
