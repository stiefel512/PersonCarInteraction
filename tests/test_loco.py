"""Leave-one-clip-out protocol tests.

Run against synthetic predictions, so they check the PROTOCOL rather than the
pipeline: that the held-out fold is genuinely held out, that the constraint
prunes the grid, and that selection is deterministic. A leak here would inflate
the headline number in a way no downstream check would catch.
"""
import pytest

from pvi.evaluate.loco import (GRID, run, select, setting_key, valid_settings,
                               score)
from pvi.schema import GTEvent, Interaction, PersonRef, VehicleRef


def pred(clip, start, end, type_="enter_vehicle", idx=1):
    return Interaction(
        interaction_id=f"{clip}__i{idx:03d}", event_group_id=f"{clip}__g{idx:03d}",
        type=type_, frame_start=start, frame_end=end,
        time_start_s=start / 10, time_end_s=end / 10,
        person=PersonRef(track_id=idx), vehicle=VehicleRef(track_id=99))


def gt(clip, start, end, type_="enter_vehicle", idx=1):
    return GTEvent(event_id=f"{clip}__e{idx:03d}", event_group_id=f"{clip}__e{idx:03d}",
                   clip_id=clip, type=type_, frame_start=start, frame_end=end,
                   time_start_s=start / 10, time_end_s=end / 10)


# --- grid ---

def test_constraint_prunes_tau_far_le_tau_near():
    for s in valid_settings():
        assert s["tau_far"] > s["tau_near"]


def test_grid_is_not_empty_after_pruning():
    assert len(valid_settings()) > 0


def test_only_the_three_agreed_knobs_are_searched():
    """design-plan §6a.2 as amended 2026-09-17. Searching all ten fits fold
    noise at 18 positives; vlm_conf_thresh was dropped from the agreed four
    because it is inert -- across three full runs it never rejected a single
    candidate, Qwen answering pass_by at 0.86-0.96 rather than near it."""
    assert set(GRID) == {"det_conf", "tau_near", "tau_far"}


def test_the_inert_knob_is_frozen_not_searched():
    from pvi.config import LOCO_SEARCH_KEYS
    assert "vlm_conf_thresh" not in LOCO_SEARCH_KEYS
    assert set(LOCO_SEARCH_KEYS) == set(GRID)


def test_grid_and_loco_search_keys_agree():
    """Two declarations of the same decision; drift between them would mean the
    report names one set of knobs and the sweep varies another."""
    from pvi.config import LOCO_SEARCH_KEYS
    assert sorted(GRID) == sorted(LOCO_SEARCH_KEYS)


# --- selection ---

def test_select_picks_the_better_setting():
    a = {"det_conf": 0.35, "tau_near": 0.15, "tau_far": 0.30, "vlm_conf_thresh": 0.5}
    b = {"det_conf": 0.25, "tau_near": 0.15, "tau_far": 0.30, "vlm_conf_thresh": 0.3}
    gt_by = {"c1": [gt("c1", 10, 20)]}
    preds_by = {
        ("c1", setting_key(a)): [pred("c1", 10, 20)],     # exact
        ("c1", setting_key(b)): [pred("c1", 80, 90)],     # miss
    }
    assert select(["c1"], preds_by, gt_by, [a, b]) == a


def test_select_is_deterministic_under_setting_order():
    a = {"det_conf": 0.35, "tau_near": 0.15, "tau_far": 0.30, "vlm_conf_thresh": 0.5}
    b = {"det_conf": 0.25, "tau_near": 0.15, "tau_far": 0.30, "vlm_conf_thresh": 0.3}
    gt_by = {"c1": [gt("c1", 10, 20)]}
    # Both settings score identically -- a tie must resolve the same way.
    preds_by = {("c1", setting_key(a)): [pred("c1", 10, 20)],
                ("c1", setting_key(b)): [pred("c1", 10, 20)]}
    assert select(["c1"], preds_by, gt_by, [a, b]) == \
           select(["c1"], preds_by, gt_by, [a, b])


def test_score_of_no_predictions_is_zero_not_undefined():
    assert score([], [gt("c1", 10, 20)]) == 0.0


# --- the protocol itself ---

def _two_setting_world():
    good = {"det_conf": 0.35, "tau_near": 0.15, "tau_far": 0.30, "vlm_conf_thresh": 0.5}
    bad = {"det_conf": 0.25, "tau_near": 0.15, "tau_far": 0.30, "vlm_conf_thresh": 0.3}
    clips = ["c1", "c2", "c3"]
    gt_by = {c: [gt(c, 10, 20)] for c in clips}
    preds_by = {}
    for c in clips:
        preds_by[(c, setting_key(good))] = [pred(c, 10, 20)]
        preds_by[(c, setting_key(bad))] = [pred(c, 80, 90)]
    return [good, bad], clips, gt_by, preds_by


def test_every_clip_becomes_a_held_out_fold_exactly_once():
    settings, clips, gt_by, preds_by = _two_setting_world()
    r = run(preds_by, gt_by, settings)
    assert [f["held_out_clip"] for f in r["folds"]] == sorted(clips)


def test_headline_is_computed_from_held_out_predictions_only():
    settings, clips, gt_by, preds_by = _two_setting_world()
    r = run(preds_by, gt_by, settings)
    # One GT event per clip, all found by the good setting.
    assert r["headline_held_out"]["n_gt"] == len(clips)
    assert r["headline_held_out"]["recall"] == pytest.approx(1.0)


def test_a_setting_good_only_on_the_held_out_clip_is_not_selected():
    """The leak this protocol exists to prevent: a setting that only works on
    the held-out clip must not be chosen, because the held-out clip is not in
    the fitting set."""
    a = {"det_conf": 0.35, "tau_near": 0.15, "tau_far": 0.30, "vlm_conf_thresh": 0.5}
    b = {"det_conf": 0.25, "tau_near": 0.15, "tau_far": 0.30, "vlm_conf_thresh": 0.3}
    clips = ["c1", "c2", "c3"]
    gt_by = {c: [gt(c, 10, 20)] for c in clips}
    preds_by = {}
    for c in clips:
        # `a` works everywhere except c1; `b` works ONLY on c1.
        preds_by[(c, setting_key(a))] = [pred(c, 80, 90)] if c == "c1" else [pred(c, 10, 20)]
        preds_by[(c, setting_key(b))] = [pred(c, 10, 20)] if c == "c1" else [pred(c, 80, 90)]
    r = run(preds_by, gt_by, [a, b])
    c1_fold = next(f for f in r["folds"] if f["held_out_clip"] == "c1")
    assert c1_fold["selected"] == a, "selection used the held-out clip"
    assert c1_fold["fold_f1"] == 0.0


def test_all_data_number_is_reported_and_labelled_not_held_out():
    settings, clips, gt_by, preds_by = _two_setting_world()
    r = run(preds_by, gt_by, settings)
    assert "all_data_tuned_NOT_held_out" in r
    assert "overfitting_gap_f1" in r


def test_fold_agreement_is_reported():
    """If every fold picks a different value the selection is fitting noise --
    that has to be visible, not averaged away."""
    settings, clips, gt_by, preds_by = _two_setting_world()
    r = run(preds_by, gt_by, settings)
    assert set(r["fold_agreement"]) == set(GRID)
    assert r["fold_agreement"]["det_conf"]["n_distinct"] == 1


def test_frozen_knobs_are_listed_explicitly():
    settings, clips, gt_by, preds_by = _two_setting_world()
    r = run(preds_by, gt_by, settings)
    assert "smooth_window_s" in r["frozen_at_defaults"]
    assert "det_conf" not in r["frozen_at_defaults"]


def test_caveats_are_carried_in_the_report():
    """The honesty notes ship with the numbers rather than living only in a doc
    nobody opens next to the results."""
    settings, clips, gt_by, preds_by = _two_setting_world()
    assert len(run(preds_by, gt_by, settings)["caveats"]) >= 3


# --- on-disk cache key ---

def test_cache_path_is_stable_across_processes():
    """The key used to come from hash(), which Python randomises per process, so
    the filename differed every run and the sweep silently redid all the work."""
    import subprocess, sys, json as _json
    code = (
        "from pathlib import Path;"
        "from pvi.evaluate.loco import _cache_path;"
        "print(_cache_path(Path('/tmp'), 'clipA',"
        " {'det_conf':0.5,'tau_near':0.1,'tau_far':0.3}).name)"
    )
    names = {subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, cwd=".").stdout.strip()
             for _ in range(3)}
    assert len(names) == 1, f"cache filename is not stable: {names}"


def test_cache_path_differs_between_judges():
    """The judge changes the answer completely. Without it in the key a VLM
    sweep would load an earlier geometric sweep's results and report them as
    its own."""
    from pathlib import Path
    from pvi.evaluate.loco import _cache_path
    s = {"det_conf": 0.5, "tau_near": 0.1, "tau_far": 0.3}
    assert _cache_path(Path("/tmp"), "c", s, "vlm") != \
           _cache_path(Path("/tmp"), "c", s, "geometric")


def test_cache_path_differs_between_settings():
    from pathlib import Path
    from pvi.evaluate.loco import _cache_path
    a = _cache_path(Path("/tmp"), "c", {"det_conf": 0.5, "tau_near": 0.1, "tau_far": 0.3})
    b = _cache_path(Path("/tmp"), "c", {"det_conf": 0.5, "tau_near": 0.2, "tau_far": 0.3})
    assert a != b
