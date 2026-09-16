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
