"""Config validation tests.

The two structural constraints are tested rather than trusted because both are
silent failures: a degenerate hysteresis still runs and still emits spans, and
an over-large min_dwell_s just quietly never fires R1 on short events.
"""
import pytest
import yaml

from pvi.config import (Config, ConfigError, MIN_SMOOTH_FRAMES,
                        SHORTEST_POSITIVE_S, Tunables, config_hash,
                        dump_resolved, load, smooth_frames, validate)


def test_default_config_file_is_valid():
    """Guards the bug this project shipped once already: config/default.yaml had
    min_dwell_s = 1.0 against a validated constraint of < 0.75, so loading the
    default would have raised."""
    cfg = load("config/default.yaml")
    assert cfg.tunable.min_dwell_s < SHORTEST_POSITIVE_S


def test_defaults_in_code_match_the_shipped_yaml():
    """Two sources of truth for the same numbers; if they drift, a run using
    code defaults is not the run the YAML describes."""
    from_yaml = load("config/default.yaml").tunable
    assert from_yaml == Tunables()


def test_tau_far_must_exceed_tau_near():
    with pytest.raises(ConfigError, match="tau_far"):
        validate(Config(tunable=Tunables(tau_near=0.3, tau_far=0.2)))


def test_equal_taus_are_rejected():
    with pytest.raises(ConfigError, match="tau_far"):
        validate(Config(tunable=Tunables(tau_near=0.2, tau_far=0.2)))


def test_min_dwell_at_or_above_shortest_positive_is_rejected():
    with pytest.raises(ConfigError, match="min_dwell_s"):
        validate(Config(tunable=Tunables(min_dwell_s=SHORTEST_POSITIVE_S)))


def test_out_of_range_tunable_is_rejected():
    with pytest.raises(ConfigError, match="det_conf"):
        validate(Config(tunable=Tunables(det_conf=0.95)))


def test_nonzero_temperature_is_rejected():
    from pvi.config import VLMCfg
    with pytest.raises(ConfigError, match="temperature"):
        validate(Config(vlm=VLMCfg(temperature=0.7)))


def test_batching_is_rejected():
    from pvi.config import VLMCfg
    with pytest.raises(ConfigError, match="batch_size"):
        validate(Config(vlm=VLMCfg(batch_size=4)))


def test_with_tunables_validates():
    cfg = load("config/default.yaml")
    with pytest.raises(ConfigError):
        cfg.with_tunables(tau_near=0.9)


def test_with_tunables_rejects_unknown_name():
    cfg = load("config/default.yaml")
    with pytest.raises(ConfigError, match="unknown tunable"):
        cfg.with_tunables(tau_nearr=0.1)


def test_unknown_top_level_key_is_rejected():
    import tempfile, pathlib
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("seed: 0\nwidget: 3\n")
        p = f.name
    with pytest.raises(ConfigError, match="unknown config key"):
        load(p)
    pathlib.Path(p).unlink()


# --- smoothing floor ---

def test_smooth_window_floored_at_three_frames_at_low_fps():
    """0.4 s at 6 fps is 2.4 frames; the floor keeps it usable."""
    assert smooth_frames(0.4, 6.0) == MIN_SMOOTH_FRAMES


def test_smooth_window_scales_at_high_fps():
    assert smooth_frames(0.4, 30.0) == 12


def test_zero_window_still_respects_the_floor():
    assert smooth_frames(0.0, 30.0) == MIN_SMOOTH_FRAMES


# --- determinism of the resolved config ---

def test_config_hash_is_stable_across_calls():
    cfg = load("config/default.yaml")
    assert config_hash(cfg) == config_hash(cfg)


def test_config_hash_changes_with_a_tunable():
    cfg = load("config/default.yaml")
    assert config_hash(cfg) != config_hash(cfg.with_tunables(det_conf=0.4))


def test_dump_resolved_roundtrips(tmp_path):
    cfg = load("config/default.yaml")
    out = tmp_path / "resolved.yaml"
    dump_resolved(cfg, out)
    loaded = yaml.safe_load(out.read_text())
    assert loaded["_config_hash"] == config_hash(cfg)
    assert loaded["tunable"]["tau_near"] == cfg.tunable.tau_near


# --- det_floor / det_conf split ---
#
# Pre-filtering detections at det_conf before the tracker disables ByteTrack's
# second association pass, which is the mechanism that carries a track through
# detector flicker. Measured on the night and CIF clips: the subject's
# confidence oscillates between ~0.6 and ~0.1 frame to frame, and filtering at
# 0.35 split one person into 5-7 sequential tracks.

def test_det_conf_must_exceed_the_tracker_floor():
    from pvi.config import TrackerCfg
    with pytest.raises(ConfigError, match="det_floor"):
        validate(Config(tracker=TrackerCfg(det_floor=0.6),
                        tunable=Tunables(det_conf=0.5)))


def test_equal_floor_and_conf_are_rejected():
    from pvi.config import TrackerCfg
    with pytest.raises(ConfigError, match="det_floor"):
        validate(Config(tracker=TrackerCfg(det_floor=0.5),
                        tunable=Tunables(det_conf=0.5)))


def test_shipped_default_leaves_a_real_second_association_band():
    """The gap between floor and det_conf IS the second-association pool. If it
    collapses the fix is undone, silently."""
    cfg = load("config/default.yaml")
    assert cfg.tunable.det_conf - cfg.tracker.det_floor >= 0.2


def test_det_conf_range_stays_above_the_floor():
    """Every point the LOCO search can reach must satisfy the constraint, or the
    sweep will raise partway through."""
    from pvi.config import TUNABLE_RANGES
    lo, _ = TUNABLE_RANGES["det_conf"]
    assert lo > load("config/default.yaml").tracker.det_floor


# --- environment recording ---
#
# A config that pins model revisions but not the stack running them is half a
# record. Reinstalling resolved torch 2.14.0+cu130 where the original had
# +cu132, which moved detection boxes by ~0.05 px -- harmless to every reported
# metric, but enough to matter if a result is disputed.

def test_environment_is_yaml_serialisable():
    """torch.__version__ is a TorchVersion, not a str, and yaml refuses it."""
    import yaml as _yaml
    from pvi.config import environment
    _yaml.safe_dump(environment())


def test_environment_records_the_cuda_build():
    """importlib.metadata drops the local tag (+cu130), which is precisely the
    part that distinguishes the builds. torch.__version__ keeps it."""
    from pvi.config import environment
    env = environment()
    assert "torch_build" in env and "cuda" in env


def test_resolved_config_carries_the_environment(tmp_path):
    from pvi.config import dump_resolved, load
    out = tmp_path / "resolved.yaml"
    dump_resolved(load("config/default.yaml"), out)
    assert "_environment" in yaml.safe_load(out.read_text())
