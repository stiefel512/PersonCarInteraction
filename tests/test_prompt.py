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


# --- tracking evidence ---
#
# Measured failure this addresses: without it the VLM rejected exit_vehicle
# events as pass_by at 0.9 confidence, including one at tIoU 0.97 where all four
# proposal rules fired. A person already beside a car who then walks away looks
# exactly like a pass-by in isolation; that they were NEVER SEEN approaching is
# a whole-clip fact the crops cannot carry.

from pvi.judge.prompt import tracking_evidence
from pvi.propose.rules import (R1_DWELL, R2_DEATH_NEAR, R3_BIRTH_NEAR,
                               R4_DOOR_CHANGE)


def cand_with(rules, evidence=None):
    return Candidate(person_id=1, vehicle_id=2, vehicle_cls=2,
                     frame_start=20, frame_end=40, rules=rules,
                     evidence=evidence or {})


def test_birth_in_open_view_states_they_did_not_walk_in():
    t = tracking_evidence(cand_with([R3_BIRTH_NEAR],
                                    {"born_at_frame_edge": False}), META)
    assert "FIRST appearance" in t and "not walk in from off-camera" in t


def test_birth_at_the_frame_edge_says_the_OPPOSITE():
    """Track birth alone does not mean 'emerged from the vehicle' -- a
    pedestrian walking into shot beside a parked car satisfies it too.
    Asserting the emergence claim unconditionally recovered 4 true positives
    but made the VLM accept 4 of 7 labeled pass_by events."""
    t = tracking_evidence(cand_with([R3_BIRTH_NEAR],
                                    {"born_at_frame_edge": True}), META)
    assert "walked in from off-camera" in t
    assert "did not come out" in t


def test_death_in_open_view_states_they_did_not_walk_off():
    t = tracking_evidence(cand_with([R2_DEATH_NEAR],
                                    {"died_at_frame_edge": False}), META)
    assert "LAST appearance" in t and "not walk off-camera" in t


def test_death_at_the_frame_edge_says_the_OPPOSITE():
    t = tracking_evidence(cand_with([R2_DEATH_NEAR],
                                    {"died_at_frame_edge": True}), META)
    assert "walked off-camera" in t and "did not get into" in t


def test_door_rule_reports_the_open_door():
    assert "door" in tracking_evidence(cand_with([R4_DOOR_CHANGE]), META).lower()


def test_dwell_alone_adds_no_evidence():
    """R1 fires on nearly every candidate, so it carries no information the
    frames do not already show. Saying something for every candidate would
    dilute the lines that matter."""
    assert tracking_evidence(cand_with([R1_DWELL]), META) == ""


def test_several_rules_are_all_reported():
    t = tracking_evidence(cand_with([R3_BIRTH_NEAR, R2_DEATH_NEAR, R4_DOOR_CHANGE],
                                    {"born_at_frame_edge": False,
                                     "died_at_frame_edge": False}), META)
    assert t.count("\n- ") == 3


def test_evidence_never_names_a_type_or_an_answer():
    """It reports what the tracker saw, not what to conclude. Naming the type
    would make the VLM a rubber stamp for the geometric arm and destroy the
    ablation's meaning."""
    t = tracking_evidence(cand_with([R3_BIRTH_NEAR, R2_DEATH_NEAR, R4_DOOR_CHANGE],
                                    {"born_at_frame_edge": False,
                                     "died_at_frame_edge": False}), META)
    for name in ("exit_vehicle", "enter_vehicle", "pass_by", "open_close_door"):
        assert name not in t


def test_clip_start_is_disclosed_so_absence_is_not_read_as_evidence():
    t = tracking_evidence(cand_with([R3_BIRTH_NEAR],
                                    {"birth_at_clip_start": True}), META)
    assert "clip itself begins" in t


def test_evidence_appears_in_the_built_prompt():
    c = cand_with([R3_BIRTH_NEAR], {"born_at_frame_edge": False})
    b = P.build(c, META, frames(range(0, 100)),
                {i: (100, 100, 140, 200) for i in range(100)},
                {i: (200, 150, 400, 300) for i in range(100)}, 6, 1.0, 0.25)
    assert "FIRST appearance" in b.user


def test_prompt_keeps_a_strong_pass_by_default_and_a_narrow_exception():
    """Both halves matter, and the balance was measured. Weakening the pass_by
    rule to fix exit_vehicle recovered recall (0.556 -> 0.778) but let 4 of 7
    labeled pass_by events through and left F1 flat (+0.004). The default must
    stay strong; the exception must be narrow and conditioned on the evidence."""
    c = cand_with([R1_DWELL])
    b = P.build(c, META, frames(range(0, 100)),
                {i: (100, 100, 140, 200) for i in range(100)},
                {i: (200, 150, 400, 300) for i in range(100)}, 6, 1.0, 0.25)
    assert "When in doubt, answer pass_by" in b.user
    assert "does NOT apply" in b.user


# --- cache key salt ---
#
# The key hashes the rendered images and prompt. Anything else that changes the
# answer must be salted in, or the cache is silently wrong across exactly the
# comparisons this project intends to make.

def _bundle():
    return P.build(cand(), META, frames(range(15, 46)),
                   {i: (100, 100, 140, 200) for i in range(15, 46)},
                   {i: (200, 150, 400, 300) for i in range(15, 46)}, 6, 1.0, 0.25)


def test_salt_changes_the_key():
    b = _bundle()
    assert b.cache_key("model-a") != b.cache_key("model-b")


def test_same_salt_is_stable():
    b = _bundle()
    assert b.cache_key("x") == b.cache_key("x")


def test_switching_model_would_not_replay_the_other_models_verdicts():
    """design-plan §6.4 calls for reporting both Qwen2.5-VL-7B and 32B. With an
    unsalted key the 32B run would serve the 7B answers."""
    from pvi.judge.vlm import VLMJudge
    a = VLMJudge(model_id="Qwen/Qwen2.5-VL-7B-Instruct", load_model=False)
    b = VLMJudge(model_id="Qwen/Qwen2.5-VL-32B-Instruct", load_model=False)
    bundle = _bundle()
    assert bundle.cache_key(a.cache_salt) != bundle.cache_key(b.cache_salt)


def test_changing_the_pixel_budget_invalidates_the_entry():
    """The cap changes what the model is shown without changing the PIL image
    we hash, so it has to be in the salt."""
    from pvi.judge.vlm import VLMJudge
    a = VLMJudge(max_pixels_per_frame=401_408, load_model=False)
    b = VLMJudge(max_pixels_per_frame=200_704, load_model=False)
    bundle = _bundle()
    assert bundle.cache_key(a.cache_salt) != bundle.cache_key(b.cache_salt)


def test_revision_is_in_the_salt():
    from pvi.judge.vlm import VLMJudge
    a = VLMJudge(revision="aaa", load_model=False)
    b = VLMJudge(revision="bbb", load_model=False)
    assert _bundle().cache_key(a.cache_salt) != _bundle().cache_key(b.cache_salt)
