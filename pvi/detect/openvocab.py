"""Grounding DINO open-vocabulary detection -- probe-only.

Status: this is arm (c) of the `gt1125_06` probe (design-plan s6.2/s6.7). It is
NOT wired into the pipeline. The question it answers is whether the R4 door cue
should stay a pixel-change heuristic (which works only on a static camera with a
parked vehicle, so it is dead on the drone clip and on the moving sedan in
`iMGR`) or become a text prompt that works everywhere.

The argument for adopting it is strong enough to be suspicious of: Grounding
DINO's documented weaknesses are small objects and aerial imagery, which is
exactly where the heuristic already fails. Hence a probe rather than a design
change.

Prompt format is not free-form. Grounding DINO expects lowercase, period-
terminated phrases; packing many phrases into one prompt measurably degrades
each. `format_prompt` enforces that so a caller cannot get it subtly wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from ..geometry import Box


def format_prompt(phrases: Sequence[str]) -> str:
    """Lowercase, strip, and dot-terminate each phrase.

    Grounding DINO's text encoder was trained on this format and silently
    performs worse without it -- there is no error, just fewer boxes.
    """
    out = []
    for p in phrases:
        p = p.strip().lower().rstrip(".").strip()
        if p:
            out.append(p + ".")
    return " ".join(out)


@dataclass(frozen=True)
class OVDetection:
    frame: int
    label: str
    box: Box
    conf: float


class OpenVocabDetector:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base",
                 revision: str | None = None, device: str = "cuda:0"):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, revision=revision,
        ).to(device).eval()
        self.device = device
        cfg = getattr(self.model, "config", None)
        self._revision = str(getattr(cfg, "_commit_hash", None) or revision or "unresolved")

    @property
    def revision(self) -> str:
        return self._revision

    @torch.inference_mode()
    def detect(self, frames: Sequence[np.ndarray], frame_indices: Sequence[int],
               phrases: Sequence[str], box_threshold: float = 0.27,
               text_threshold: float = 0.25) -> list[list[OVDetection]]:
        images = [Image.fromarray(f) for f in frames]
        text = format_prompt(phrases)
        inputs = self.processor(images=images, text=[text] * len(images),
                                return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        sizes = torch.tensor([[im.height, im.width] for im in images],
                             device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_threshold,
            text_threshold=text_threshold, target_sizes=sizes,
        )

        out: list[list[OVDetection]] = []
        for res, fidx in zip(results, frame_indices):
            dets = [
                OVDetection(frame=fidx, label=str(lbl), conf=float(score),
                            box=tuple(float(v) for v in box))
                for score, lbl, box in zip(res["scores"], res["text_labels"], res["boxes"])
            ]
            dets.sort(key=lambda d: (-d.conf, d.label, d.box))
            out.append(dets)
        return out
