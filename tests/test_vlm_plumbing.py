"""VLM message/image plumbing, checked against the real processor.

The processor is a few MB and cached alongside the weights, so this runs without
a GPU and without the 16 GB checkpoint. It covers the part of the VLM path that
unit tests on our own code cannot: that the chat template emits one image
placeholder per image we pass, and that the processor accepts the multi-image
message shape. A mismatch there is silent at build time and fails only inside
generate().

Skipped when the processor is not cached, so the suite still runs on a clean
checkout.
"""
import numpy as np
import pytest

from pvi.judge import prompt as P
from pvi.propose.rules import Candidate
from pvi.schema import ClipMeta

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
META = ClipMeta("t", 640, 480, 10.0, 100)


@pytest.fixture(scope="module")
def processor():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoProcessor.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True)
    except Exception as exc:                       # not cached on this machine
        pytest.skip(f"{MODEL} processor not cached locally: {type(exc).__name__}")


def bundle(n_frames=8):
    rng = np.random.default_rng(0)
    frames = {i: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
              for i in range(100)}
    cand = Candidate(person_id=1, vehicle_id=2, vehicle_cls=2,
                     frame_start=20, frame_end=40, rules=["R1_dwell"])
    return P.build(cand, META, frames,
                   {i: (100, 100, 140, 200) for i in range(100)},
                   {i: (200, 150, 400, 300) for i in range(100)},
                   n_frames, 1.0, 0.25)


def render(processor, b):
    msgs = [{"role": "system", "content": [{"type": "text", "text": b.system}]},
            {"role": "user", "content": [{"type": "image"} for _ in b.images]
                                        + [{"type": "text", "text": b.user}]}]
    return processor.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)


def test_chat_template_emits_one_placeholder_per_image(processor):
    """Too few placeholders and the model silently sees fewer frames than the
    span it is being asked about."""
    for n in (4, 8):
        b = bundle(n)
        text = render(processor, b)
        assert text.count("<|vision_start|>") == len(b.images) == n


def test_processor_accepts_the_multi_image_message(processor):
    b = bundle(8)
    inputs = processor(text=[render(processor, b)], images=b.images,
                       return_tensors="pt")
    assert {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"} \
        <= set(inputs)
    assert inputs["image_grid_thw"].shape[0] == len(b.images)


def test_patch_count_scales_with_frame_count(processor):
    """n_vlm_frames is a tunable; its cost has to be visible and linear, since
    12 frames per candidate across a clip is the VLM's whole budget."""
    small = bundle(4)
    large = bundle(8)
    a = processor(text=[render(processor, small)], images=small.images,
                  return_tensors="pt")["pixel_values"].shape[0]
    b = processor(text=[render(processor, large)], images=large.images,
                  return_tensors="pt")["pixel_values"].shape[0]
    assert b == pytest.approx(2 * a, rel=0.1)


def test_prompt_text_survives_templating(processor):
    """The type definitions are load-bearing -- attend_vehicle vs pass_by cannot
    be tuned at n=2 and has to come from this wording."""
    text = render(processor, bundle(4))
    assert "pass_by" in text and "NOT an interaction" in text
    assert "RED" in text and "BLUE" in text
