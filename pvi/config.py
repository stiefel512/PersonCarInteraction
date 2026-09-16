"""Configuration loading, validation and resolved-config dumping.

The fixed/tunable split mirrors `config/schema.md`, which is the contract the
sweep tooling reads. Two constraints are validated here rather than left to a
comment, because a config that violates either is not merely suboptimal --
it is structurally unable to produce correct behaviour (see `validate`).
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# The shortest positive event in the frozen ground truth, in seconds. R1 cannot
# fire on it if min_dwell_s is >= this, so it bounds the search range.
SHORTEST_POSITIVE_S = 0.75

# Smoothing is floored at this many frames regardless of smooth_window_s,
# because at 6 fps a 0.4 s window is 2.4 frames (design-plan.md s4).
MIN_SMOOTH_FRAMES = 3

# The four thresholds leave-one-clip-out is allowed to vary (design-plan s6a.2).
# The other tunables stay at their defaults for every reported number.
LOCO_SEARCH_KEYS = ("det_conf", "tau_near", "tau_far", "vlm_conf_thresh")


class ConfigError(ValueError):
    """Raised when a config violates a structural constraint."""


@dataclass(frozen=True)
class Paths:
    videos: Path = Path("Videos")
    # Labeling/contact-sheet use only -- downscaled to max width 1600, so
    # gt1125_06 is 1600x900 here rather than 3840x2160. The pipeline decodes
    # from `videos` at native resolution and must never read this.
    frames: Path = Path("data/frames")
    ground_truth: Path = Path("data/ground_truth.json")
    outputs: Path = Path("outputs")
    vlm_cache: Path = Path("data/vlm_cache")


@dataclass(frozen=True)
class DetectorCfg:
    model_id: str = "Roboflow/rf-detr-large"
    revision: str | None = None  # pinned on first run
    # COCO ids: person + the vehicle set.
    classes: tuple[int, ...] = (0, 1, 2, 3, 5, 7)


@dataclass(frozen=True)
class VLMCfg:
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    revision: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.0
    batch_size: int = 1
    # Pixel cap per image handed to the VLM. Qwen's own default (~12.8M) leaves
    # the token count unbounded, because a crop is the union of the person and
    # vehicle boxes and a spurious distant track makes that enormous.
    # 401408 = 512 * 28 * 28, so ~512 visual tokens per frame.
    max_pixels_per_frame: int = 401_408


@dataclass(frozen=True)
class TrackerCfg:
    impl: str = "roboflow-trackers"
    name: str = "bytetrack"
    gmc: str = "orb"
    # Confidence floor for detections HANDED TO THE TRACKER. Deliberately far
    # below `det_conf`, because ByteTrack's whole contribution is a second
    # association pass over LOW-confidence detections, which is what carries a
    # track through a few frames of detector flicker. Pre-filtering at
    # `det_conf` deletes exactly that pool and disables the mechanism.
    # Measured: on the night and CIF clips the subject's confidence oscillates
    # between ~0.6 and ~0.1 frame to frame, and filtering at 0.35 split one
    # person into 5-7 sequential tracks.
    det_floor: float = 0.10


@dataclass(frozen=True)
class OpenVocabCfg:
    model_id: str = "IDEA-Research/grounding-dino-base"
    revision: str | None = None
    box_threshold: float = 0.27
    text_threshold: float = 0.25


@dataclass(frozen=True)
class Tunables:
    # Not a pre-filter. This is the tracker's high-confidence threshold: at or
    # above it a detection can start a track and joins the first association
    # pass; between `tracker.det_floor` and here it still feeds the second
    # association pass. See TrackerCfg.det_floor.
    det_conf: float = 0.50
    tau_near: float = 0.15
    tau_far: float = 0.30
    min_dwell_s: float = 0.5
    smooth_window_s: float = 0.4
    door_conf_thresh: float = 0.30
    context_pad_s: float = 1.0
    n_vlm_frames: int = 12
    crop_margin: float = 0.25
    vlm_conf_thresh: float = 0.5


# Declared ranges, kept next to the defaults so a sweep cannot drift from the
# documented schema. Inclusive on both ends.
TUNABLE_RANGES: dict[str, tuple[float, float]] = {
    "det_conf": (0.15, 0.80),
    "tau_near": (0.05, 0.40),
    "tau_far": (0.10, 0.80),
    "min_dwell_s": (0.3, 0.7),
    "smooth_window_s": (0.0, 0.8),
    "door_conf_thresh": (0.20, 0.70),
    "context_pad_s": (0.0, 3.0),
    "n_vlm_frames": (4, 24),
    "crop_margin": (0.0, 0.6),
    "vlm_conf_thresh": (0.0, 0.9),
}


@dataclass(frozen=True)
class Config:
    seed: int = 0
    device: str = "cuda:0"
    paths: Paths = field(default_factory=Paths)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    vlm: VLMCfg = field(default_factory=VLMCfg)
    tracker: TrackerCfg = field(default_factory=TrackerCfg)
    openvocab: OpenVocabCfg = field(default_factory=OpenVocabCfg)
    tunable: Tunables = field(default_factory=Tunables)

    def with_tunables(self, **overrides: float) -> "Config":
        """Return a copy with some tunables replaced. Used by the LOCO search."""
        unknown = set(overrides) - {f.name for f in dataclasses.fields(Tunables)}
        if unknown:
            raise ConfigError(f"unknown tunable(s): {sorted(unknown)}")
        new = dataclasses.replace(self.tunable, **overrides)
        cfg = dataclasses.replace(self, tunable=new)
        validate(cfg)
        return cfg


def validate(cfg: Config) -> None:
    """Enforce the two structural constraints plus the declared ranges.

    Both constraints are structural rather than stylistic:

    - `tau_far > tau_near` -- with tau_far <= tau_near the hysteresis degenerates
      and a person hovering at the boundary yields a burst of fragment spans
      instead of one span.
    - `min_dwell_s < 0.75` -- the shortest positive event in the frozen GT is
      0.75 s, so a larger dwell makes R1 structurally unable to fire on it.
    """
    t = cfg.tunable
    if t.tau_far <= t.tau_near:
        raise ConfigError(
            f"tau_far ({t.tau_far}) must exceed tau_near ({t.tau_near}); "
            "otherwise the near/far hysteresis degenerates into a single threshold"
        )
    if t.det_conf <= cfg.tracker.det_floor:
        raise ConfigError(
            f"det_conf ({t.det_conf}) must exceed tracker.det_floor "
            f"({cfg.tracker.det_floor}); det_conf is the confidence at which a "
            "detection is trusted enough to start a track and to enter the "
            "first association pass, and the floor is what feeds the second pass"
        )
    if t.min_dwell_s >= SHORTEST_POSITIVE_S:
        raise ConfigError(
            f"min_dwell_s ({t.min_dwell_s}) must be < {SHORTEST_POSITIVE_S} s, the "
            "shortest positive event in the ground truth, or R1 cannot fire on it"
        )
    for name, (lo, hi) in TUNABLE_RANGES.items():
        v = getattr(t, name)
        if not lo <= v <= hi:
            raise ConfigError(f"{name}={v} is outside its declared range [{lo}, {hi}]")
    if cfg.vlm.temperature != 0.0:
        raise ConfigError("vlm.temperature must be 0.0; greedy decoding is required "
                          "for the determinism guarantee")
    if cfg.vlm.batch_size != 1:
        raise ConfigError("vlm.batch_size must be 1; batching perturbs outputs "
                          "even at temperature 0")


def smooth_frames(smooth_window_s: float, fps: float) -> int:
    """Smoothing window in frames for a clip, floored at MIN_SMOOTH_FRAMES.

    Separate from the rule logic so the floor is testable on its own: it is the
    one place a seconds-based threshold is allowed to become a frame count.
    """
    return max(int(round(smooth_window_s * fps)), MIN_SMOOTH_FRAMES)


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _build(d: dict[str, Any]) -> Config:
    def sub(cls, key):
        raw = dict(d.get(key, {}))
        if cls is Paths:
            raw = {k: Path(v) for k, v in raw.items()}
        if cls is DetectorCfg and "classes" in raw:
            raw["classes"] = tuple(raw["classes"])
        return cls(**raw)

    known = {"seed", "device", "paths", "detector", "vlm", "tracker",
             "openvocab", "tunable"}
    unknown = set(d) - known
    if unknown:
        raise ConfigError(f"unknown config key(s): {sorted(unknown)}")
    return Config(
        seed=d.get("seed", 0),
        device=d.get("device", "cuda:0"),
        paths=sub(Paths, "paths"),
        detector=sub(DetectorCfg, "detector"),
        vlm=sub(VLMCfg, "vlm"),
        tracker=sub(TrackerCfg, "tracker"),
        openvocab=sub(OpenVocabCfg, "openvocab"),
        tunable=sub(Tunables, "tunable"),
    )


def load(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load YAML, apply overrides, validate. Raises ConfigError on violation."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        raw = _merge(raw, overrides)
    cfg = _build(raw)
    validate(cfg)
    return cfg


def to_dict(cfg: Config) -> dict[str, Any]:
    d = asdict(cfg)

    def norm(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, tuple):
            return list(x)
        if isinstance(x, dict):
            return {k: norm(v) for k, v in x.items()}
        return x

    return norm(d)


def config_hash(cfg: Config) -> str:
    """Stable hash of the resolved config, written into every output record.

    sort_keys so the hash depends on values, not on field declaration order.
    """
    blob = json.dumps(to_dict(cfg), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def dump_resolved(cfg: Config, path: str | Path) -> None:
    """Write the fully-resolved config next to an output, per the determinism
    requirement in problem-definition.md s4."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = to_dict(cfg)
    payload["_config_hash"] = config_hash(cfg)
    p.write_text(yaml.safe_dump(payload, sort_keys=True, default_flow_style=False))
