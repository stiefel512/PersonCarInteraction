"""GMC correctness tests on synthetic transforms with known ground truth.

These exist because a wrong homography raises nothing -- it silently degrades
tracking, and shows up only as worse metrics on one clip. Synthetic warps give
an exact expected answer, which real footage never does.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pvi.track.gmc import (IDENTITY, GMCResult, estimate, to_gray, warp_box,
                           _plausible)


def textured_frame(w=640, h=480, seed=0):
    """Deterministic high-frequency texture: ORB needs corners to find."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(h, w), dtype=np.uint8)
    # Blur slightly so features are stable under interpolation rather than
    # being pure per-pixel noise.
    return cv2.GaussianBlur(img, (5, 5), 0)


def shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


# --- estimation ---

def test_pure_translation_is_recovered():
    a = textured_frame()
    b = shift(a, 12, -7)
    r = estimate(a, b)
    assert r.ok, r.reason
    assert r.H[0, 2] == pytest.approx(12, abs=1.0)
    assert r.H[1, 2] == pytest.approx(-7, abs=1.0)


def test_identical_frames_give_near_identity():
    a = textured_frame()
    r = estimate(a, a.copy())
    assert r.ok, r.reason
    assert np.allclose(r.H, IDENTITY, atol=0.05)


def test_featureless_frames_degrade_to_identity_and_say_so():
    """A blank frame must not produce a confident garbage homography."""
    blank = np.full((480, 640), 128, np.uint8)
    r = estimate(blank, blank.copy())
    assert not r.ok
    assert r.is_identity
    assert r.reason


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="shape mismatch"):
        estimate(textured_frame(640, 480), textured_frame(320, 240))


def test_translation_recovered_under_moderate_rotation():
    a = textured_frame()
    M = cv2.getRotationMatrix2D((320, 240), 3.0, 1.0)
    b = cv2.warpAffine(a, M, (640, 480))
    r = estimate(a, b)
    assert r.ok, r.reason
    # 3 degrees about the centre: the linear part should be a near-rotation.
    assert np.linalg.det(r.H[:2, :2]) == pytest.approx(1.0, abs=0.1)


# --- plausibility gate ---

def test_plausible_accepts_identity():
    ok, _ = _plausible(IDENTITY)
    assert ok


def test_plausible_rejects_gross_scale():
    H = np.diag([5.0, 5.0, 1.0])
    ok, why = _plausible(H)
    assert not ok and "scale" in why


def test_plausible_rejects_reflection():
    H = np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    ok, why = _plausible(H)
    assert not ok and "determinant" in why


def test_plausible_rejects_non_finite():
    H = IDENTITY.copy()
    H[0, 0] = np.nan
    ok, why = _plausible(H)
    assert not ok and "non-finite" in why


# --- box warping ---

def test_warp_box_under_identity_is_unchanged():
    box = (10.0, 20.0, 30.0, 40.0)
    assert warp_box(box, IDENTITY) == pytest.approx(box)


def test_warp_box_translation():
    assert warp_box((10, 20, 30, 40),
                    np.array([[1, 0, 5.0], [0, 1, -3.0], [0, 0, 1]])) \
        == pytest.approx((15, 17, 35, 37))


def test_warp_box_uses_all_four_corners_under_rotation():
    """A 90-degree rotation maps the box to a box of swapped extent. Using only
    two corners would give a degenerate or mirrored result."""
    theta = np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                  [np.sin(theta), np.cos(theta), 0.0],
                  [0.0, 0.0, 1.0]])
    x1, y1, x2, y2 = warp_box((0.0, 0.0, 10.0, 20.0), R)
    assert (x2 - x1) == pytest.approx(20.0)
    assert (y2 - y1) == pytest.approx(10.0)


def test_warp_box_45_degrees_grows_the_axis_aligned_bound():
    """An axis-aligned bounding box of a rotated box is larger than the original
    -- if this returns the same size, corners are not being transformed."""
    t = np.pi / 4
    R = np.array([[np.cos(t), -np.sin(t), 0.0],
                  [np.sin(t), np.cos(t), 0.0],
                  [0.0, 0.0, 1.0]])
    x1, y1, x2, y2 = warp_box((0.0, 0.0, 10.0, 10.0), R)
    assert (x2 - x1) == pytest.approx(10 * np.sqrt(2), abs=1e-6)


def test_warp_box_degenerate_projective_divide_returns_input():
    """A homography sending the box to the horizon must not emit inf/nan."""
    H = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 0.0]])
    box = (10.0, 20.0, 30.0, 40.0)
    assert warp_box(box, H) == box


# --- helpers ---

def test_to_gray_passes_through_2d():
    g = np.zeros((10, 10), np.uint8)
    assert to_gray(g).shape == (10, 10)


def test_to_gray_reduces_rgb():
    assert to_gray(np.zeros((10, 10, 3), np.uint8)).shape == (10, 10)
