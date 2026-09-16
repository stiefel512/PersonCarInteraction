"""Fail-fast check on the VLM checkpoint.

An incomplete HF snapshot is not obviously broken: snapshot_download can exit 0
having fetched only some shards, and the directory looks populated. Observed
here with 2 of 5 Qwen shards present -- the run died only after a full
detect-and-track pass, and the library's error blamed the network.
"""
import json

import pytest

from pvi.judge import vlm


def make_snapshot(tmp_path, shards_present, n_shards=3):
    snap = tmp_path / "snap"
    snap.mkdir()
    weight_map = {f"layer.{i}": f"model-0000{i + 1}-of-0000{n_shards}.safetensors"
                  for i in range(n_shards)}
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}))
    for name in shards_present:
        (snap / name).write_bytes(b"x")
    return snap


@pytest.fixture
def patched(monkeypatch):
    def install(snap):
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download",
            lambda *a, **k: str(snap))
    return install


def test_complete_snapshot_passes(tmp_path, patched):
    all_shards = [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]
    patched(make_snapshot(tmp_path, all_shards))
    vlm.check_weights_available("fake/model", "abc123")   # must not raise


def test_missing_shard_raises_and_names_it(tmp_path, patched):
    patched(make_snapshot(tmp_path, ["model-00001-of-00003.safetensors"]))
    with pytest.raises(RuntimeError) as e:
        vlm.check_weights_available("fake/model", "abc123")
    msg = str(e.value)
    assert "incomplete" in msg
    assert "model-00002-of-00003.safetensors" in msg


def test_error_mentions_the_cache_not_the_network(tmp_path, patched):
    """The library's own message blames huggingface.co, which sends the reader
    to the wrong problem when the cache is simply truncated."""
    patched(make_snapshot(tmp_path, []))
    with pytest.raises(RuntimeError, match="local cache"):
        vlm.check_weights_available("fake/model", "abc123")


def test_single_file_checkpoint_without_an_index_is_accepted(tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: str(snap))
    vlm.check_weights_available("fake/model", "abc123")   # must not raise


def test_absent_from_cache_raises(monkeypatch):
    def boom(*a, **k):
        raise OSError("not cached")
    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    with pytest.raises(RuntimeError, match="not in the local cache"):
        vlm.check_weights_available("fake/model", "abc123")


# --- token budget ---
#
# A crop is the union of the person and vehicle boxes, so a spurious track far
# from the vehicle makes it enormous. With Qwen's default ~12.8M pixel cap and
# n_vlm_frames images per candidate, the token count is effectively unbounded:
# observed as a 4.5 GiB allocation failure on a 48 GB card once low-confidence
# detections started reaching the tracker.

def test_default_pixel_cap_is_configured_and_modest():
    from pvi.config import load
    cfg = load("config/default.yaml")
    assert cfg.vlm.max_pixels_per_frame == 401_408          # 512 * 28 * 28
    assert cfg.vlm.max_pixels_per_frame < 1_000_000


def test_cap_bounds_the_whole_candidate_not_just_one_frame():
    """The budget that matters is per candidate: cap x n_vlm_frames."""
    from pvi.config import load
    cfg = load("config/default.yaml")
    tokens_per_frame = cfg.vlm.max_pixels_per_frame / (28 * 28)
    total = tokens_per_frame * cfg.tunable.n_vlm_frames
    assert total <= 8192, f"{total:.0f} visual tokens per candidate is too many"


def test_judge_accepts_and_stores_the_cap():
    from pvi.judge.vlm import VLMJudge
    j = VLMJudge(max_pixels_per_frame=123_456, load_model=False)
    assert j.max_pixels_per_frame == 123_456


# --- model cache ---
#
# run_all processes 8 clips x 2 judges in ONE process. Constructing every model
# per run_clip meant 16 loads of a 16 GB VLM, which exhausted system RAM partway
# through the set.

def test_detector_is_loaded_once_per_checkpoint(monkeypatch):
    from pvi import cli
    from pvi.config import load

    cli.clear_model_cache()
    calls = []

    class FakeDetector:
        def __init__(self, model_id, **kw):
            calls.append(model_id)
        revision = "fake"

    monkeypatch.setattr(cli, "HFDetector", FakeDetector)
    cfg = load("config/default.yaml")
    for _ in range(5):
        cli.build_detector(cfg, tiled=False)
    assert len(calls) == 1, f"loaded the detector {len(calls)} times"
    cli.clear_model_cache()


def test_tiled_and_plain_share_the_same_weights(monkeypatch):
    """The sliced wrapper is a decorator; it must not force a second load."""
    from pvi import cli
    from pvi.config import load

    cli.clear_model_cache()
    calls = []

    class FakeDetector:
        def __init__(self, model_id, **kw):
            calls.append(model_id)
        revision = "fake"

    monkeypatch.setattr(cli, "HFDetector", FakeDetector)
    cfg = load("config/default.yaml")
    plain = cli.build_detector(cfg, tiled=False)
    tiled = cli.build_detector(cfg, tiled=True)
    assert len(calls) == 1
    assert tiled.inner is plain
    cli.clear_model_cache()


def test_cache_key_separates_different_checkpoints(monkeypatch):
    import dataclasses
    from pvi import cli
    from pvi.config import load

    cli.clear_model_cache()
    calls = []

    class FakeDetector:
        def __init__(self, model_id, **kw):
            calls.append(model_id)
        revision = "fake"

    monkeypatch.setattr(cli, "HFDetector", FakeDetector)
    cfg = load("config/default.yaml")
    cli.build_detector(cfg, tiled=False)
    other = dataclasses.replace(
        cfg, detector=dataclasses.replace(cfg.detector,
                                          model_id="ustc-community/dfine-xlarge-coco"))
    cli.build_detector(other, tiled=False)
    assert len(calls) == 2, "a different checkpoint must not reuse cached weights"
    cli.clear_model_cache()
