"""Description slots: response parsing, GT parsing, and the scoring rules.

The GT parse is pinned against every string in the frozen ground truth. A
silent mis-parse there (a colour attached to the wrong garment, an object read
as clothing) would bias every description number without failing anything.
"""
import json
from pathlib import Path

import pytest

from pvi.evaluate.descriptions import (ABSTAIN, CORRECT, WRONG, compare_colour,
                                       description_report, parse_gt_person,
                                       parse_gt_vehicle, score_pair)
from pvi.judge import describe as D
from pvi.schema import GTEvent, Interaction, PersonRef, VehicleRef

GT_PATH = Path(__file__).resolve().parents[1] / "data" / "ground_truth.json"


# --- response parsing ---

def test_parse_normalizes_and_rejects_off_vocabulary():
    raw = json.dumps({"person": {"sex": "Male", "age": "adult",
                                 "upper_color": "Gray", "lower_color": "navy",
                                 "upper_garment": "jacket", "lower_garment": "",
                                 "carrying": "Yes"},
                      "vehicle": {"color": "silver", "body_type": "SUV",
                                  "state": "parked"}})
    out = D.parse_response("```json\n" + raw + "\n```")
    p, v = out["person"], out["vehicle"]
    assert p["sex"] == "male" and p["upper_color"] == "grey"
    # Off-list is NOT mapped to a neighbour at parse time.
    assert p["lower_color"] == D.UNKNOWN
    assert p["lower_garment"] == D.UNKNOWN
    assert p["carrying"] == "yes"
    assert "carried_object" not in p
    assert v == {"color": "silver", "body_type": "suv", "state": "parked"}


def test_parse_returns_none_without_json():
    assert D.parse_response("I cannot tell.") is None
    assert D.parse_response('{"foo": 1}') is None


def test_render_is_derived_from_slots():
    p = {"sex": "female", "age": "adult", "upper_color": "white",
         "upper_garment": "top", "lower_color": D.UNKNOWN,
         "lower_garment": "trousers", "carrying": "no"}
    assert D.render_person(p) == "adult female, white top, trousers"
    assert D.render_person({**p, "carrying": "yes"}).endswith(
        ", carrying an object")
    v = {"color": D.UNKNOWN, "body_type": "sedan", "state": "moving"}
    assert D.render_vehicle(v) == "sedan, moving"


def test_carrying_off_vocabulary_is_unknown():
    """An object name in the carrying slot must not leak into the output."""
    out = D.parse_response('{"person": {"carrying": "gun"}, "vehicle": {}}')
    assert out["person"]["carrying"] == D.UNKNOWN


def test_describe_prompt_differs_from_judge_prompt():
    """Shared text would share a cache key and replay a verdict as a
    description."""
    from pvi.judge.prompt import PromptBundle
    b = PromptBundle(images=[], frame_indices=[1, 2], system="s", user="u",
                     geometry=((0, 0, 1, 1),))
    d = D.rebundle(b)
    assert d.cache_key("x") != b.cache_key("x")
    assert d.geometry == b.geometry and d.frame_indices == b.frame_indices


# --- GT parsing ---

@pytest.mark.parametrize("desc,expected", [
    ("adult man, dark shirt, dark pants",
     dict(sex="male", age="adult", upper_color={"dark"}, lower_color={"dark"},
          carrying=False)),
    ("adult woman, blue shirt, black shorts, black bag",
     dict(sex="female", age="adult", upper_color={"blue"},
          lower_color={"black"}, carrying=True)),
    ("young boy",
     dict(sex="male", age="child", upper_color="unknown",
          lower_color="unknown", carrying="unknown")),
    ("adult, dark clothes",
     dict(sex="unknown", age="adult", upper_color={"dark"},
          lower_color={"dark"}, carrying=False)),
    ("adult male, green shirt, black pants, papers",
     dict(sex="male", age="adult", upper_color={"green"},
          lower_color={"black"}, carrying=True)),
])
def test_parse_gt_person(desc, expected):
    got = parse_gt_person(desc)
    for k, v in expected.items():
        assert got[k] == (frozenset(v) if isinstance(v, set) else v), k


@pytest.mark.parametrize("desc,expected", [
    ("grey, sedan, parked", ({"grey"}, "sedan", "static")),
    ("white and black, hatchback, parked", ({"white", "black"}, "hatchback", "static")),
    ("dark sedan, driving", ({"dark"}, "sedan", "moving")),
    ("grey, sedan, starting to drive", ({"grey"}, "sedan", "moving")),
    ("black, sedan, stopped", ({"black"}, "sedan", "static")),
])
def test_parse_gt_vehicle(desc, expected):
    got = parse_gt_vehicle(desc)
    assert (got["color"], got["body_type"], got["state"]) == \
        (frozenset(expected[0]), expected[1], expected[2])


def test_every_gt_string_parses_fully():
    """Every positive GT event must yield sex-or-age, both clothing colours,
    and all three vehicle slots -- except the known terse labels."""
    terse = {"young boy"}
    events = json.loads(GT_PATH.read_text())["events"]
    for e in events:
        p, v = parse_gt_person(e["person_desc"]), parse_gt_vehicle(e["vehicle_desc"])
        assert v["color"] != "unknown" and v["body_type"] != "unknown" \
            and v["state"] != "unknown", e["event_id"]
        if e["person_desc"] in terse:
            continue
        assert p["age"] != "unknown", e["event_id"]
        assert p["upper_color"] != "unknown" and p["lower_color"] != "unknown", \
            e["event_id"]


# --- scoring rules ---

def test_luminance_only_gt_scores_lightness_not_hue():
    assert compare_colour(frozenset({"dark"}), "black") == (None, CORRECT)
    assert compare_colour(frozenset({"dark"}), "white") == (None, WRONG)


def test_grey_and_silver_are_one_hue():
    assert compare_colour(frozenset({"grey"}), "silver") == (CORRECT, CORRECT)


def test_unknown_is_abstention_not_error():
    assert compare_colour(frozenset({"red"}), "unknown") == (ABSTAIN, ABSTAIN)


def test_model_luminance_answer_abstains_on_hue_only():
    assert compare_colour(frozenset({"black"}), "dark") == (ABSTAIN, CORRECT)


def test_multi_colour_gt_accepts_either():
    assert compare_colour(frozenset({"white", "black"}), "black")[0] == CORRECT


def _pair(gt_person, gt_vehicle, pattrs, vattrs, pred_box=None):
    g = GTEvent(event_id="c__e001", event_group_id="c__e001", clip_id="c",
                type="enter_vehicle", frame_start=10, frame_end=20,
                time_start_s=1.0, time_end_s=2.0, person_desc=gt_person,
                vehicle_desc=gt_vehicle,
                person_anchor={"frame": 10, "x": 0.5, "y": 0.5})
    p = Interaction(interaction_id="c__i001", event_group_id="c__g001",
                    type="enter_vehicle", frame_start=10, frame_end=20,
                    time_start_s=1.0, time_end_s=2.0,
                    person=PersonRef(1, "", pattrs),
                    vehicle=VehicleRef(2, "car", "", vattrs),
                    evidence={"person_box_norm": pred_box or [0.4, 0.4, 0.6, 0.6]})
    return g, p


PA = {"sex": "female", "age": "adult", "upper_color": "grey",
      "lower_color": "yellow", "upper_garment": "top",
      "lower_garment": "trousers", "carrying": "no"}
VA = {"color": "silver", "body_type": "sedan", "state": "stopped"}


def test_score_pair_all_correct():
    g, p = _pair("adult woman, grey shirt, yellow pants", "grey, sedan, parked",
                 PA, VA)
    out = score_pair(g, p)
    assert set(out.values()) == {CORRECT}
    assert "person.carrying" in out and "vehicle.state" in out


def test_report_excludes_actor_mismatch():
    """A temporal match about a different person must not be graded."""
    g, p = _pair("adult woman, grey shirt, yellow pants", "grey, sedan, parked",
                 PA, VA, pred_box=[0.0, 0.0, 0.1, 0.1])
    rep = description_report([([p], [g])])
    assert rep["n_matched"] == 1 and rep["n_actor_mismatch"] == 1
    assert rep["n_scored"] == 0


def test_report_skips_predictions_without_attributes():
    g, p = _pair("adult woman, grey shirt, yellow pants", "grey, sedan, parked",
                 None, None)
    rep = description_report([([p], [g])])
    assert rep["n_without_attributes"] == 1 and rep["n_scored"] == 0


def test_carrying_scored_against_gt_presence():
    g, p = _pair("adult woman, grey shirt, yellow pants, black bag",
                 "grey, sedan, parked", {**PA, "carrying": "no"}, VA)
    assert score_pair(g, p)["person.carrying"] == WRONG
    g, p = _pair("adult woman, grey shirt, yellow pants, black bag",
                 "grey, sedan, parked", {**PA, "carrying": "unknown"}, VA)
    assert score_pair(g, p)["person.carrying"] == ABSTAIN
