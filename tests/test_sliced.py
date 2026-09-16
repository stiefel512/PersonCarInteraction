"""Tiling geometry and NMS merging.

Runs without a GPU or a model: the inner Detector is a stub. The point is that
box coordinate mapping across tiles is arithmetic, and arithmetic that is wrong
by a tile origin produces detections in plausible-but-wrong places, which no
downstream stage would flag.
"""
import numpy as np
import pytest

from pvi.detect.base import Detection, Detector, nms
from pvi.detect.sliced import SlicedDetector, tile_origins
from pvi.schema import ClipMeta


def meta(w, h):
    return ClipMeta(clip_id="t", width=w, height=h, fps=30.0, n_frames=1)


# --- tiling geometry ---

def test_tile_origins_cover_the_full_extent():
    origins = tile_origins(3840, 704, 0.25)
    assert origins[0] == 0
    assert origins[-1] + 704 == 3840   # last tile flush with the edge


def test_tile_origins_single_tile_when_frame_is_smaller():
    assert tile_origins(352, 704, 0.25) == [0]


def test_tile_origins_have_no_gaps():
    tile, origins = 704, tile_origins(2160, 704, 0.25)
    for a, b in zip(origins, origins[1:]):
        assert b <= a + tile, "consecutive tiles must overlap or touch, never gap"


def test_tile_origins_respect_overlap_fraction():
    # 25% overlap on a 704 tile -> step 528.
    assert tile_origins(4000, 704, 0.25)[1] == 528


def test_zero_overlap_tiles_abut():
    assert tile_origins(2000, 500, 0.0)[:3] == [0, 500, 1000]


# --- NMS ---

def d(x1, y1, x2, y2, conf=0.9, cls=0, frame=0):
    return Detection(frame=frame, cls=cls, box=(x1, y1, x2, y2), conf=conf)


def test_nms_suppresses_duplicate_of_same_class():
    out = nms([d(0, 0, 10, 10, 0.9), d(1, 1, 11, 11, 0.8)], 0.5)
    assert len(out) == 1 and out[0].conf == 0.9


def test_nms_keeps_different_classes_even_when_coincident():
    """A person standing against a car produces heavily overlapping boxes of
    different classes; suppressing across classes would delete one of them."""
    out = nms([d(0, 0, 10, 10, cls=0), d(0, 0, 10, 10, cls=2)], 0.5)
    assert len(out) == 2


def test_nms_keeps_disjoint_boxes():
    assert len(nms([d(0, 0, 10, 10), d(100, 100, 110, 110)], 0.5)) == 2


def test_nms_is_deterministic_under_input_reordering():
    """Determinism is a graded requirement; equal-confidence boxes must not let
    input order decide the survivor."""
    a = [d(0, 0, 10, 10, 0.8), d(1, 1, 11, 11, 0.8), d(50, 50, 60, 60, 0.8)]
    assert [x.box for x in nms(a, 0.5)] == [x.box for x in nms(list(reversed(a)), 0.5)]


# --- composition ---

class _CornerStub(Detector):
    """Emits one box at a fixed offset inside whatever frame it is given."""

    @property
    def revision(self) -> str:
        return "stub"

    def detect(self, frames, meta_, frame_indices, conf):
        return [[Detection(frame=fi, cls=0, box=(5.0, 6.0, 15.0, 26.0), conf=0.9)]
                for fi in frame_indices]


def test_sliced_maps_tile_boxes_back_into_frame_coordinates():
    """The stub reports (5,6,15,26) inside every tile, so the merged output must
    contain boxes at each tile origin plus that offset -- never a box still in
    tile-local coordinates."""
    inner = _CornerStub()
    sl = SlicedDetector(inner=inner, tile=100, overlap=0.0,
                        include_full_frame=False, merge_iou=0.99)
    m = meta(300, 100)
    frame = np.zeros((100, 300, 3), np.uint8)
    out = sl.detect([frame], m, [0], 0.3)[0]
    xs = sorted(det.box[0] for det in out)
    assert xs == [5.0, 105.0, 205.0]


def test_sliced_reports_the_frame_index_it_was_given():
    sl = SlicedDetector(inner=_CornerStub(), tile=100, overlap=0.0,
                        include_full_frame=False)
    out = sl.detect([np.zeros((100, 100, 3), np.uint8)], meta(100, 100), [42], 0.3)[0]
    assert all(det.frame == 42 for det in out)


class _HalfExtentStub(Detector):
    """Emits one box covering the centred half of whatever frame it is given.

    Centred and half-size rather than full-size so a tile detection is not
    trivially contained in the full-frame one -- otherwise containment merging
    collapses them and the test cannot see whether the full-frame pass ran.
    """

    @property
    def revision(self) -> str:
        return "stub"

    def detect(self, frames, meta_, frame_indices, conf):
        out = []
        for f, fi in zip(frames, frame_indices):
            h, w = f.shape[:2]
            out.append([Detection(frame=fi, cls=0, conf=0.9,
                                  box=(w * 0.25, h * 0.25, w * 0.75, h * 0.75))])
        return out


def test_sliced_includes_full_frame_pass_by_default():
    """Tiling alone loses objects bigger than a tile -- the box truck in
    gt1125_06 is exactly that case, so the full-frame pass must still run."""
    sl = SlicedDetector(inner=_HalfExtentStub(), tile=100, overlap=0.0,
                        merge_iou=0.99)
    out = sl.detect([np.zeros((100, 200, 3), np.uint8)], meta(200, 100), [0], 0.3)[0]
    widths = sorted(round(d.box[2] - d.box[0]) for d in out)
    assert 100 in widths, "the full-frame detection was lost"
    assert widths == [50, 50, 100]


def test_sliced_without_full_frame_pass_only_tiles():
    sl = SlicedDetector(inner=_HalfExtentStub(), tile=100, overlap=0.0,
                        merge_iou=0.99, include_full_frame=False)
    out = sl.detect([np.zeros((100, 200, 3), np.uint8)], meta(200, 100), [0], 0.3)[0]
    assert sorted(round(d.box[2] - d.box[0]) for d in out) == [50, 50]


def test_sliced_passes_through_inner_revision():
    sl = SlicedDetector(inner=_CornerStub())
    assert sl.revision == "stub"


# --- canonical class resolution ---
#
# Hard-coded COCO ids were a live bug: RF-DETR-L ships 91 labels (person=1)
# while D-FINE-X ships 80 (person=0) with Pascal-style names. Filtering RF-DETR
# by the 80-class id list selected N/A, airplane and train while dropping
# motorcycle, bus and truck.

from pvi.detect.base import resolve_class_map

RF_DETR_LABELS = {0: "N/A", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
                  5: "airplane", 6: "bus", 7: "train", 8: "truck"}
DFINE_LABELS = {0: "person", 1: "bicycle", 2: "car", 3: "motorbike",
                4: "aeroplane", 5: "bus", 6: "train", 7: "truck"}


def test_resolve_maps_rf_detr_91_class_ids_to_canonical():
    m = resolve_class_map(RF_DETR_LABELS)
    assert m[1] == 0    # person
    assert m[3] == 2    # car
    assert m[4] == 3    # motorcycle
    assert m[6] == 5    # bus
    assert m[8] == 7    # truck


def test_resolve_excludes_classes_never_asked_for():
    """airplane, train and N/A must not survive into the canonical map."""
    m = resolve_class_map(RF_DETR_LABELS)
    assert 5 not in m and 7 not in m and 0 not in m


def test_resolve_maps_dfine_80_class_ids_and_pascal_names():
    m = resolve_class_map(DFINE_LABELS)
    assert m[0] == 0 and m[2] == 2 and m[7] == 7
    assert m[3] == 3, "motorbike must resolve to canonical motorcycle"


def test_both_detectors_agree_on_canonical_ids_after_mapping():
    """The point of the whole exercise: swapping the detector must not change
    what a downstream class id means."""
    a, b = resolve_class_map(RF_DETR_LABELS), resolve_class_map(DFINE_LABELS)
    assert sorted(a.values()) == sorted(b.values())


def test_resolve_raises_when_person_is_absent():
    with pytest.raises(ValueError, match="person"):
        resolve_class_map({0: "car", 1: "truck"})


# --- containment (IoS) merging ---
#
# Measured on gt1125_06: IoU-only merging roughly doubled the vehicle count
# (22 -> 46 on frame 239) through tile-boundary fragments, which would have
# become duplicate tracks and duplicate reported events.

from pvi.detect.base import ios


def test_ios_of_a_contained_fragment_is_near_one():
    full = (0.0, 0.0, 100.0, 100.0)
    half = (0.0, 0.0, 50.0, 100.0)      # truncated at a tile edge
    assert ios(full, half) == pytest.approx(1.0)


def test_that_same_fragment_survives_iou_thresholding():
    """Why IoU alone is not enough: the fragment scores 0.5, under a 0.55 NMS."""
    from pvi.geometry import iou as box_iou
    assert box_iou((0, 0, 100, 100), (0, 0, 50, 100)) == pytest.approx(0.5)


def test_ios_is_symmetric():
    a, b = (0.0, 0.0, 100.0, 100.0), (0.0, 0.0, 50.0, 100.0)
    assert ios(a, b) == pytest.approx(ios(b, a))


def test_ios_of_disjoint_boxes_is_zero():
    assert ios((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0


def test_nms_suppresses_a_tile_fragment_that_iou_would_keep():
    full = d(0, 0, 100, 100, conf=0.9, cls=2)
    frag = d(0, 0, 50, 100, conf=0.8, cls=2)
    assert len(nms([full, frag], iou_thresh=0.55)) == 1


def test_nms_keeps_the_fragment_when_containment_is_disabled():
    full = d(0, 0, 100, 100, conf=0.9, cls=2)
    frag = d(0, 0, 50, 100, conf=0.8, cls=2)
    assert len(nms([full, frag], iou_thresh=0.55, ios_thresh=None)) == 2


def test_the_containing_box_survives_when_confidences_tie():
    """The fragment must never be the survivor: a truncated tile box in place of
    the whole vehicle would mis-place every downstream d_norm."""
    full = d(0, 0, 100, 100, conf=0.9, cls=2)
    frag = d(0, 0, 50, 100, conf=0.9, cls=2)
    out = nms([frag, full], iou_thresh=0.55)
    assert len(out) == 1 and out[0].box == (0, 0, 100, 100)


def test_containment_never_suppresses_across_classes():
    """A person standing against a car is almost entirely contained in the car's
    box. Suppressing that would delete the person from every event."""
    car = d(0, 0, 200, 100, conf=0.95, cls=2)
    person = d(10, 10, 40, 90, conf=0.8, cls=0)
    assert len(nms([car, person], iou_thresh=0.55)) == 2


def test_two_genuinely_adjacent_cars_are_both_kept():
    """Parked cars touch but do not contain each other -- the rule must not
    collapse a parking row into one detection."""
    a = d(0, 0, 100, 60, conf=0.9, cls=2)
    b = d(105, 0, 205, 60, conf=0.9, cls=2)
    assert len(nms([a, b], iou_thresh=0.55)) == 2


# --- per-frame detection cap ---
#
# Lowering the tracker's detection floor to 0.10 multiplied per-frame counts,
# and on the 4K clip that runs 28 tiles per frame the accumulated detections
# exhausted system RAM.

from pvi.detect.base import cap_per_frame


def test_cap_keeps_the_highest_confidence_detections():
    dets = [d(0, 0, 10, 10, conf=c / 100, cls=0) for c in range(1, 21)]
    out = cap_per_frame(dets, limit=5)
    assert len(out) == 5
    assert min(x.conf for x in out) == pytest.approx(0.16)


def test_cap_is_per_class_so_cars_cannot_crowd_out_people():
    """On the aerial clip vehicles outnumber persons about five to one. An
    overall cap would delete the people, which are the whole point."""
    cars = [d(i, 0, i + 5, 10, conf=0.9, cls=2) for i in range(0, 300, 6)]
    people = [d(i, 50, i + 4, 70, conf=0.2, cls=0) for i in range(0, 30, 6)]
    out = cap_per_frame(cars + people, limit=10)
    assert sum(1 for x in out if x.cls == 0) == 5
    assert sum(1 for x in out if x.cls == 2) == 10


def test_cap_is_a_no_op_below_the_limit():
    dets = [d(0, 0, 10, 10, conf=0.5, cls=0), d(50, 50, 60, 60, conf=0.4, cls=0)]
    assert len(cap_per_frame(dets, limit=100)) == 2


def test_cap_is_deterministic_under_input_reordering():
    dets = [d(i, 0, i + 5, 10, conf=0.5, cls=0) for i in range(0, 100, 5)]
    assert [x.box for x in cap_per_frame(dets, limit=7)] == \
           [x.box for x in cap_per_frame(list(reversed(dets)), limit=7)]


def test_default_limit_is_well_above_a_realistic_scene():
    """gt1125_06 peaks around 25 vehicles and 8 people per frame; the cap must
    not bite on real content, only on the noise tail."""
    from pvi.detect.base import MAX_DETS_PER_FRAME_PER_CLASS
    assert MAX_DETS_PER_FRAME_PER_CLASS >= 50
