"""Prompt construction and response parsing.

Response parsing is defensive on purpose: a 7B model ignores "JSON and nothing
else" often enough that a strict parser would drop real verdicts. Every
tolerated deviation here is one that was worth tolerating; anything that would
change the *meaning* of a verdict is still rejected.
"""
import numpy as np
import pytest

from pvi.judge import prompt as P
from pvi.propose.rules import Candidate
from pvi.schema import ClipMeta

META = ClipMeta(clip_id="t", width=640, height=480, fps=10.0, n_frames=100)


def cand(start=20, end=40):
    return Candidate(person_id=1, vehicle_id=2, vehicle_cls=2,
                     frame_start=start, frame_end=end, rules=["R1_dwell"])


def frames(idx, w=640, h=480):
    rng = np.random.default_rng(0)
    return {i: rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for i in idx}


# --- parsing ---

def test_parses_plain_json():
    out = P.parse_response('{"type":"enter_vehicle","confidence":0.8,'
                           '"person":"a","vehicle":"b","note":"c"}')
    assert out["type"] == "enter_vehicle" and out["confidence"] == pytest.approx(0.8)


def test_parses_fenced_json():
    out = P.parse_response('```json\n{"type":"pass_by","confidence":0.2,'
                           '"person":"a","vehicle":"b","note":"c"}\n```')
    assert out["type"] == "pass_by"


def test_parses_json_surrounded_by_prose():
    out = P.parse_response('Sure! Here is the answer:\n'
                           '{"type":"exit_vehicle","confidence":0.9,'
                           '"person":"a","vehicle":"b","note":"c"}\nHope that helps.')
    assert out["type"] == "exit_vehicle"


def test_rejects_a_type_outside_the_vocabulary():
    """An invented type must not reach the output contract, where it would raise
    far from its cause."""
    assert P.parse_response('{"type":"stealing_car","confidence":0.9}') is None


def test_rejects_unparseable_text():
    assert P.parse_response("I cannot tell from these frames.") is None


def test_rejects_empty():
    assert P.parse_response("") is None


def test_pass_by_is_parseable_but_is_not_a_reportable_type():
    """The judge must be able to SAY pass_by even though it can never be
    emitted; that is how a rejection gets recorded in the debug artifact."""
    assert "pass_by" in P.ALLOWED_TYPES
    from pvi.schema import INTERACTION_TYPES
    assert "pass_by" not in INTERACTION_TYPES


def test_missing_confidence_defaults_to_zero_not_one():
    """Defaulting high would let a malformed response pass the acceptance
    threshold."""
    out = P.parse_response('{"type":"attend_vehicle"}')
    assert out["confidence"] == 0.0


def test_confidence_is_clamped():
    assert P.parse_response('{"type":"attend_vehicle","confidence":5}')["confidence"] == 1.0
    assert P.parse_response('{"type":"attend_vehicle","confidence":-2}')["confidence"] == 0.0


def test_non_numeric_confidence_does_not_raise():
    assert P.parse_response('{"type":"attend_vehicle","confidence":"high"}')["confidence"] == 0.0


def test_missing_text_fields_become_empty_strings():
    out = P.parse_response('{"type":"attend_vehicle","confidence":0.5}')
    assert out["person"] == "" and out["vehicle"] == "" and out["note"] == ""


# --- prompt text ---

def test_prompt_names_the_box_colours():
    """Without the colour reference the crop is ambiguous in the multi-actor
    clips, which is the entire reason for visual prompting."""
    b = P.build(cand(), META, frames(range(15, 46)),
                {i: (100, 100, 140, 200) for i in range(15, 46)},
                {i: (200, 150, 400, 300) for i in range(15, 46)},
                6, 1.0, 0.25)
    assert "RED" in b.user and "BLUE" in b.user


def test_prompt_defines_pass_by_as_a_non_interaction():
    b = P.build(cand(), META, frames(range(15, 46)),
                {i: (100, 100, 140, 200) for i in range(15, 46)},
                {i: (200, 150, 400, 300) for i in range(15, 46)},
                6, 1.0, 0.25)
    assert "pass_by" in b.user and "NOT an interaction" in b.user


# --- bundle construction ---

def test_build_samples_at_most_n_frames():
    b = P.build(cand(20, 60), META, frames(range(0, 100)),
                {i: (100, 100, 140, 200) for i in range(100)},
                {i: (200, 150, 400, 300) for i in range(100)},
                8, 1.0, 0.25)
    assert len(b.images) <= 8 and len(b.images) == len(b.frame_indices)


def test_build_pads_the_span_by_context():
    """Context padding must actually widen the sampled range -- the frames just
    before and after are what show approach and departure."""
    idx = list(range(0, 100))
    b = P.build(cand(40, 50), META, frames(idx),
                {i: (100, 100, 140, 200) for i in idx},
                {i: (200, 150, 400, 300) for i in idx},
                12, 1.0, 0.25)
    assert min(b.frame_indices) < 40 and max(b.frame_indices) > 50


def test_crop_contains_both_actors():
    crop = P.crop_box([(100, 100, 140, 200)], [(300, 150, 500, 300)], 0.0, META)
    assert crop[0] <= 100 and crop[2] >= 500
    assert crop[1] <= 100 and crop[3] >= 300


def test_crop_is_clipped_to_the_frame():
    crop = P.crop_box([(0, 0, 40, 60)], [(600, 440, 640, 480)], 1.0, META)
    assert crop[0] >= 0 and crop[1] >= 0
    assert crop[2] <= META.width and crop[3] <= META.height


def test_cache_key_is_stable_for_identical_input():
    args = (cand(), META, frames(range(15, 46)),
            {i: (100, 100, 140, 200) for i in range(15, 46)},
            {i: (200, 150, 400, 300) for i in range(15, 46)}, 6, 1.0, 0.25)
    assert P.build(*args).cache_key() == P.build(*args).cache_key()


def test_cache_key_changes_when_the_boxes_move():
    """The key must track the rendered pixels; otherwise a cached verdict could
    be replayed against a different crop."""
    f = frames(range(15, 46))
    a = P.build(cand(), META, f, {i: (100, 100, 140, 200) for i in range(15, 46)},
                {i: (200, 150, 400, 300) for i in range(15, 46)}, 6, 1.0, 0.25)
    b = P.build(cand(), META, f, {i: (110, 100, 150, 200) for i in range(15, 46)},
                {i: (200, 150, 400, 300) for i in range(15, 46)}, 6, 1.0, 0.25)
    assert a.cache_key() != b.cache_key()
