"""prepare-data on a tiny hand-made tunes.json with the same schema as The Session's dump."""

import json
import os

from jigai import data

HERE = os.path.dirname(__file__)


def test_prepare_builds_sets(tmp_path):
    meta = data.prepare(os.path.join(HERE, "fixtures", "tiny_tunes.json"), str(tmp_path))
    assert meta["jigs"] == 4          # every 6/8 jig setting (tune 1 has two settings)
    assert meta["jigs_first"] == 3    # one per tune
    assert meta["dark"] == 2          # Edorian + Aminor
    assert meta["slip"] == 1
    jigs = json.load(open(tmp_path / "jigs.json"))
    assert jigs[0].startswith("X:1\nT:Handmade One\nR:jig\nM:6/8\nL:1/8\nK:Dmajor\n")
    assert "\n DFA" in jigs[0]        # continuation lines are indented, as in training


def test_songs_subset_is_stable(tmp_path):
    data.prepare(os.path.join(HERE, "fixtures", "tiny_tunes.json"), str(tmp_path))
    a = data.songs_subset(2, seed=3, data_dir=str(tmp_path))
    b = data.songs_subset(2, seed=3, data_dir=str(tmp_path))
    assert a == b and len(a) == 2
