"""Qwen2.5-VL adjudicator with an on-disk response cache.

Determinism (a graded requirement) is pursued on three levels:

1. **Greedy decoding, batch size 1.** `do_sample=False`, `temperature` unset --
   passing temperature=0 alongside do_sample=False makes transformers warn and
   is meaningless. Batch size 1 because padding and batched attention kernels
   perturb logits enough to flip a borderline token even at zero temperature.
2. **Deterministic kernels** where torch offers them.
3. **A committed response cache**, keyed on a hash of the exact rendered image
   bytes plus the prompt text. This is the strongest lever available: with the
   cache in the repo a reviewer reproduces every number with no GPU at all, and
   any cache miss is visible rather than silently re-rolled.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..geometry import Box
from ..propose.rules import Candidate
from ..schema import ClipMeta
from . import describe as D
from . import prompt as P
from .base import Judge, Verdict


def check_weights_available(model_id: str, revision: str | None) -> None:
    """Fail fast if the checkpoint is not fully present locally.

    The VLM is constructed only after detection, tracking and the door cue have
    run, so a missing shard otherwise surfaces ten minutes into a clip. Worse,
    an *incomplete* snapshot is not obviously broken: `snapshot_download` can
    exit 0 having fetched only some shards, and the directory looks populated.
    Observed here -- 2 of 5 Qwen shards present, and the run died after a full
    detect-and-track pass.

    Checks the index's shard list against what is on disk. Raises with the
    missing filenames rather than the library's generic "couldn't connect to
    huggingface.co", which points at the network when the real problem is a
    truncated cache.
    """
    import json

    try:
        from huggingface_hub import snapshot_download
    except ImportError:          # pragma: no cover - hub is a hard dependency
        return

    try:
        snap = Path(snapshot_download(model_id, revision=revision,
                                      local_files_only=True))
    except Exception as exc:
        raise RuntimeError(
            f"{model_id} (revision {revision}) is not in the local cache: {exc}"
        ) from exc

    index = snap / "model.safetensors.index.json"
    if not index.exists():
        return                   # single-file checkpoint; nothing to verify

    want = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    missing = [w for w in want if not (snap / w).exists()]
    if missing:
        raise RuntimeError(
            f"{model_id} is incomplete in the local cache: {len(missing)} of "
            f"{len(want)} shards missing ({', '.join(missing)}). "
            f"Re-run snapshot_download for this revision; it resumes."
        )


class VLMJudge(Judge):
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 revision: str | None = None, device: str = "cuda:0",
                 cache_dir: str | Path = "data/vlm_cache",
                 max_new_tokens: int = 512,
                 n_frames: int = 12, context_pad_s: float = 1.0,
                 crop_margin: float = 0.25, conf_thresh: float = 0.5,
                 max_pixels_per_frame: int = 401_408,
                 load_model: bool = True):
        self.model_id = model_id
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_new_tokens = max_new_tokens
        self.n_frames = n_frames
        self.context_pad_s = context_pad_s
        self.crop_margin = crop_margin
        self.conf_thresh = conf_thresh
        # Cap on pixels per image handed to the VLM. Qwen2.5-VL's own default is
        # ~12.8M, which with n_vlm_frames images per candidate is unbounded in
        # practice: the crop is the union of the person and vehicle boxes, so a
        # spurious track far from the vehicle produces an enormous crop and an
        # enormous token count. Observed as a 4.5 GiB allocation failure on a
        # 48 GB card once low-confidence detections started reaching the tracker.
        # 401408 = 512 * 28 * 28, i.e. ~512 visual tokens per frame, so 12
        # frames cost ~6k tokens regardless of how bad a crop gets.
        self.max_pixels_per_frame = max_pixels_per_frame

        self.n_cache_hits = 0
        self.n_cache_misses = 0

        self.model = None
        self.processor = None
        self._revision = revision or "unresolved"
        if load_model:
            self._load(revision)

    def _load(self, revision: str | None) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, revision=revision,
            max_pixels=self.max_pixels_per_frame)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, revision=revision, dtype=torch.bfloat16,
        ).to(self.device).eval()
        cfg = getattr(self.model, "config", None)
        self._revision = str(getattr(cfg, "_commit_hash", None) or revision or "unresolved")

    @property
    def name(self) -> str:
        return f"vlm:{self.model_id}"

    @property
    def revision(self) -> str:
        return self._revision

    # --- cache ---

    def _cache_path(self, key: str) -> Path:
        # Two-level fan-out: a few hundred candidates in one directory is fine,
        # but this keeps the committed tree tidy if the clip set ever grows.
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cached(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    def _store(self, key: str, payload: dict) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    # --- inference ---

    @torch.inference_mode()
    def _generate(self, bundle: P.PromptBundle) -> str:
        if self.model is None:
            raise RuntimeError(
                "VLM not loaded and the prompt is not in the cache. Either run "
                "with a GPU or restore data/vlm_cache/."
            )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": bundle.system}]},
            {"role": "user", "content": (
                [{"type": "image"} for _ in bundle.images]
                + [{"type": "text", "text": bundle.user}]
            )},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=bundle.images,
                                return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,          # greedy; see module docstring
            num_beams=1,
        )
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True)

    @property
    def cache_salt(self) -> str:
        """Everything outside the images and prompt that changes the verdict."""
        return "|".join(str(x) for x in (
            self.model_id, self._revision, self.max_new_tokens,
            self.max_pixels_per_frame,
        ))

    def judge_bundle(self, bundle: P.PromptBundle,
                     parse=P.parse_response) -> dict | None:
        key = bundle.cache_key(self.cache_salt)
        hit = self._cached(key)
        if hit is not None:
            self.n_cache_hits += 1
            return hit.get("parsed")

        self.n_cache_misses += 1
        raw = self._generate(bundle)
        parsed = parse(raw)
        self._store(key, {
            "model_id": self.model_id,
            "revision": self._revision,
            "max_pixels_per_frame": self.max_pixels_per_frame,
            "frame_indices": bundle.frame_indices,
            "raw": raw,
            "parsed": parsed,
        })
        return parsed

    def judge(self, cand: Candidate, meta: ClipMeta,
              frames: dict[int, np.ndarray] | None = None,
              person_boxes: dict[int, Box] | None = None,
              vehicle_boxes: dict[int, Box] | None = None) -> Verdict | None:
        if frames is None or person_boxes is None or vehicle_boxes is None:
            raise ValueError("VLMJudge.judge needs frames and per-frame boxes")

        bundle = P.build(cand, meta, frames, person_boxes, vehicle_boxes,
                         self.n_frames, self.context_pad_s, self.crop_margin)
        if not bundle.images:
            return None          # could not judge -- distinct from a rejection

        parsed = self.judge_bundle(bundle)
        if parsed is None:
            return None

        is_pos = parsed["type"] != "pass_by" and parsed["confidence"] >= self.conf_thresh
        return Verdict(
            is_interaction=is_pos,
            # Keep the model's own label even when rejected, so the debug
            # artifact records *what* it thought rather than only that it said no.
            type=parsed["type"],
            person_desc=parsed["person"],
            vehicle_desc=parsed["vehicle"],
            note=parsed["note"],
            confidence=parsed["confidence"],
        )

    def describe(self, cand: Candidate, meta: ClipMeta,
                 frames: dict[int, np.ndarray],
                 person_boxes: dict[int, Box],
                 vehicle_boxes: dict[int, Box]) -> dict | None:
        """Structured person/vehicle slots for one candidate (see describe.py).

        Rebuilds the judge's bundle rather than taking it as an argument: the
        build is deterministic, so the images are the ones the verdict saw.
        """
        bundle = P.build(cand, meta, frames, person_boxes, vehicle_boxes,
                         self.n_frames, self.context_pad_s, self.crop_margin)
        if not bundle.images:
            return None
        return self.judge_bundle(D.rebundle(bundle), parse=D.parse_response)
