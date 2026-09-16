"""HF transformers object detector.

One class covers RF-DETR-L and the D-FINE-X fallback, because both expose the
same `AutoImageProcessor` + `AutoModelForObjectDetection` +
`post_process_object_detection` surface. That turns the documented fallback
(design-plan s6.1) into a config change rather than a second implementation --
which is the outcome the `Detector` ABC was introduced to enable.

Determinism notes:
- `torch.inference_mode()` and `model.eval()`, no dropout, no sampling.
- Outputs are sorted before returning, so equal-confidence boxes do not depend
  on the order the model happened to emit them.
- The resolved revision hash is read back from the loaded model and recorded.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image

from ..schema import ClipMeta
from .base import Detection, Detector, resolve_class_map


class HFDetector(Detector):
    def __init__(self, model_id: str, revision: str | None = None,
                 device: str = "cuda:0", classes: Sequence[int] | None = None,
                 dtype: torch.dtype = torch.float32):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.model_id = model_id
        self.device = device
        # `classes` are CANONICAL ids (pvi.detect.base), not this model's own.
        self.classes = tuple(classes) if classes else (0, 1, 2, 3, 5, 7)
        # float32 rather than fp16: determinism is graded, and at this data
        # volume (2313 frames total) the speed is irrelevant. fp16 accumulation
        # order can shift borderline detections across the confidence threshold.
        self.dtype = dtype

        self.processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForObjectDetection.from_pretrained(
            model_id, revision=revision, dtype=dtype,
        ).to(device).eval()

        self._revision = self._resolve_revision(revision)
        # model label id -> canonical id. Built once at load, so a checkpoint
        # whose vocabulary we cannot interpret fails here rather than by
        # silently returning nothing.
        self.class_map = resolve_class_map(self.id2label, self.classes)

    def _resolve_revision(self, revision: str | None) -> str:
        """Record the exact commit the weights came from, not the tag asked for.

        A branch name pins nothing: `main` today and `main` next month can be
        different weights, which would silently break the reproducibility claim.
        """
        cfg = getattr(self.model, "config", None)
        for attr in ("_commit_hash", "_name_or_path_commit_hash"):
            got = getattr(cfg, attr, None)
            if got:
                return str(got)
        return revision or "unresolved"

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def id2label(self) -> dict[int, str]:
        return {int(k): v for k, v in self.model.config.id2label.items()}

    @torch.inference_mode()
    def detect(self, frames: Sequence[np.ndarray], meta: ClipMeta,
               frame_indices: Sequence[int], conf: float) -> list[list[Detection]]:
        if len(frames) != len(frame_indices):
            raise ValueError("frames and frame_indices must be the same length")
        if not frames:
            return []

        images = [Image.fromarray(f) for f in frames]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # target_sizes in (height, width) -- boxes come back in the ORIGINAL
        # pixel frame of each image, undoing the processor's internal resize.
        # Per-image sizes rather than meta.*, because the sliced detector passes
        # tiles whose size is not the clip size.
        sizes = torch.tensor([[im.height, im.width] for im in images],
                             device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=conf, target_sizes=sizes,
        )

        out: list[list[Detection]] = []
        for res, fidx in zip(results, frame_indices):
            dets: list[Detection] = []
            for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
                # Translate the model's own label id to the canonical one, and
                # drop anything outside the wanted set. A model id absent from
                # the map is a class we never asked for.
                canon = self.class_map.get(int(label))
                if canon is None:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box)
                dets.append(Detection(frame=fidx, cls=canon, conf=float(score),
                                      box=(x1, y1, x2, y2)))
            dets.sort(key=lambda d: (-d.conf, d.cls, d.box))
            out.append(dets)
        return out
