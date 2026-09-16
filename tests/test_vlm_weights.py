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
