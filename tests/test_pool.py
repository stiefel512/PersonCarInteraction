"""Guards on the pooled artifact.

The pooled file is a concatenation, so the only things that can go wrong are
silent: a clip quietly missing, or interactions from two different configs
merged into one list that then reads as a single result. Both refusals are
tested here because neither would show up as an error anywhere else.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import pool_outputs  # noqa: E402

HASH_A = "sha256:aaa"
HASH_B = "sha256:bbb"


def write_clip(d: Path, clip_id: str, n: int = 1, chash: str = HASH_A,
               start: int = 10) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{clip_id}.json").write_text(json.dumps({
        "clip_id": clip_id,
        "clip_meta": {"fps": 30.0, "n_frames": 100, "width": 640, "height": 360},
        "config_hash": chash,
        "interactions": [{
            "interaction_id": f"{clip_id}__i{i:03d}",
            "event_group_id": f"{clip_id}__g{i:03d}",
            "type": "enter_vehicle",
            "frame_start": start + i, "frame_end": start + i + 5,
            "time_start_s": 0.0, "time_end_s": 1.0,
            "person": {"track_id": i, "description": "woman, white shirt"},
            "vehicle": {"track_id": 100 + i, "class": "car",
                        "description": "silver sedan, parked"},
            "note": "", "confidence": 0.9, "evidence": {"rule": "R1_dwell"},
        } for i in range(1, n + 1)],
    }) + "\n")


def inventory(tmp_path: Path, ids) -> Path:
    p = tmp_path / "clips.json"
    p.write_text(json.dumps({c: {"clip_id": c} for c in ids}))
    return p


def test_clip_id_is_lifted_into_every_record(tmp_path):
    out = tmp_path / "outputs"
    write_clip(out, "clipA", n=2)
    write_clip(out, "clipB", n=1)
    art = pool_outputs.pool(out, inventory(tmp_path, ["clipA", "clipB"]))

    assert art["n_clips"] == 2
    assert art["n_interactions"] == 3
    # The whole point of the pooled file: clip_id is per interaction, and first.
    for rec in art["interactions"]:
        assert next(iter(rec)) == "clip_id"
    assert [r["clip_id"] for r in art["interactions"]] == \
        ["clipA", "clipA", "clipB"]
    # Required fields survive the merge.
    r = art["interactions"][0]
    assert r["person"]["description"] and r["vehicle"]["description"]
    assert r["time_start_s"] is not None and r["frame_start"] is not None


def test_records_are_ordered_by_clip_then_start(tmp_path):
    out = tmp_path / "outputs"
    write_clip(out, "clipB", n=1, start=5)
    write_clip(out, "clipA", n=3, start=40)
    art = pool_outputs.pool(out, inventory(tmp_path, ["clipB", "clipA"]))
    keys = [(r["clip_id"], r["frame_start"]) for r in art["interactions"]]
    assert keys == sorted(keys)


def test_refuses_a_missing_clip(tmp_path):
    out = tmp_path / "outputs"
    write_clip(out, "clipA")
    with pytest.raises(SystemExit, match="clipB"):
        pool_outputs.pool(out, inventory(tmp_path, ["clipA", "clipB"]))


def test_allow_partial_records_the_gap_in_the_artifact(tmp_path):
    out = tmp_path / "outputs"
    write_clip(out, "clipA")
    art = pool_outputs.pool(out, inventory(tmp_path, ["clipA", "clipB"]),
                            allow_partial=True)
    assert art["INCOMPLETE_missing_clips"] == ["clipB"]
    assert art["n_clips"] == 1


def test_refuses_to_merge_two_configs(tmp_path):
    out = tmp_path / "outputs"
    write_clip(out, "clipA", chash=HASH_A)
    write_clip(out, "clipB", chash=HASH_B)
    with pytest.raises(SystemExit, match="different configs"):
        pool_outputs.pool(out, inventory(tmp_path, ["clipA", "clipB"]))
