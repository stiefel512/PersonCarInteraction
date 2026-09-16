"""Global motion compensation: frame-to-frame homography from ORB + RANSAC.

Why this is hand-written rather than a library call: BoT-SORT's GMC is an OpenCV
C++ VideoStab module with no Python binding upstream, and its own authors
recommend ORB or ECC in Python instead (design-plan s6.6). The earlier plan
treated GMC as nearly free; it is not.

Why it exists at all: `gt1125_06` is the one moving-camera clip. Without
compensation a ByteTrack Kalman filter reads ego-motion as object motion, which
inflates every velocity estimate and breaks the R2/R3 birth/death-near rules
that depend on where a track actually starts and ends. Running it
unconditionally means the other 7 clips take the same code path and simply
produce a near-identity homography.

The danger this module is built around: **a wrong homography raises nothing.**
It silently degrades tracking. So every estimate is checked for plausibility
before it is returned, and an implausible one degrades to identity rather than
being trusted -- explicitly, via `GMCResult.ok`, so the caller can count how
often it happened rather than discovering it in the metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# An estimate is rejected if fewer than this many inliers survive RANSAC.
# Four is the algebraic minimum for a homography; well above it is required for
# the estimate to mean anything.
MIN_INLIERS = 12

# Reject homographies that scale or shear implausibly between adjacent frames.
# Consecutive frames of real footage are 33 ms apart; a 2x scale change in that
# time is an estimation failure, not a camera move.
MAX_SCALE_RATIO = 1.5
MIN_SCALE_RATIO = 1.0 / MAX_SCALE_RATIO

IDENTITY = np.eye(3, dtype=np.float64)


@dataclass(frozen=True)
class GMCResult:
    H: np.ndarray          # 3x3 homography mapping previous-frame -> current-frame
    ok: bool               # False when the estimate was rejected and H is identity
    n_inliers: int
    n_matches: int
    reason: str = ""

    @property
    def is_identity(self) -> bool:
        return bool(np.allclose(self.H, IDENTITY))


def _plausible(H: np.ndarray) -> tuple[bool, str]:
    """Reject homographies that cannot describe one inter-frame camera move.

    Checks the 2x2 linear part only: translation is unbounded (a fast pan is
    legitimate) but scale and shear are not.
    """
    if not np.all(np.isfinite(H)):
        return False, "non-finite"
    A = H[:2, :2]
    det = float(np.linalg.det(A))
    if det <= 0:
        return False, "non-positive determinant (reflection or degenerate)"
    scale = float(np.sqrt(det))
    if not MIN_SCALE_RATIO <= scale <= MAX_SCALE_RATIO:
        return False, f"implausible scale {scale:.3f}"
    return True, ""


def estimate(prev_gray: np.ndarray, curr_gray: np.ndarray,
             n_features: int = 2000, ransac_thresh: float = 3.0,
             mask: np.ndarray | None = None) -> GMCResult:
    """Estimate the homography mapping `prev_gray` onto `curr_gray`.

    `mask` optionally excludes image regions from feature extraction -- pass the
    union of detected object boxes, because features on a moving car are exactly
    the ones that corrupt an estimate of *camera* motion. This is the single
    most effective thing that can be done for quality here and it is why the
    parameter exists rather than being left to a caller to remember.
    """
    import cv2

    if prev_gray.shape != curr_gray.shape:
        raise ValueError(f"frame shape mismatch: {prev_gray.shape} vs {curr_gray.shape}")

    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(prev_gray, mask)
    kp2, des2 = orb.detectAndCompute(curr_gray, mask)

    if des1 is None or des2 is None or len(kp1) < MIN_INLIERS or len(kp2) < MIN_INLIERS:
        return GMCResult(IDENTITY.copy(), False, 0, 0, "too few keypoints")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des1, des2), key=lambda m: (m.distance, m.queryIdx))
    if len(matches) < MIN_INLIERS:
        return GMCResult(IDENTITY.copy(), False, 0, len(matches), "too few matches")

    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh,
                                        maxIters=2000, confidence=0.995)
    if H is None:
        return GMCResult(IDENTITY.copy(), False, 0, len(matches), "no homography")

    n_in = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if n_in < MIN_INLIERS:
        return GMCResult(IDENTITY.copy(), False, n_in, len(matches),
                         f"only {n_in} inliers")

    good, why = _plausible(H)
    if not good:
        return GMCResult(IDENTITY.copy(), False, n_in, len(matches), why)

    return GMCResult(H.astype(np.float64), True, n_in, len(matches))


def warp_box(box: tuple[float, float, float, float], H: np.ndarray
             ) -> tuple[float, float, float, float]:
    """Map an axis-aligned box through a homography, returning its bounding box.

    All four corners are transformed, not just two. Under a homography the
    top-left and bottom-right corners do not remain the extremes -- using only
    them silently shrinks or skews the box whenever there is any rotation or
    perspective, which is precisely the case GMC exists to handle.
    """
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1, 1.0], [x2, y1, 1.0],
                        [x2, y2, 1.0], [x1, y2, 1.0]]).T
    out = H @ corners
    w = out[2]
    # Guard the projective divide: a point on the horizon has w -> 0.
    if np.any(np.abs(w) < 1e-9):
        return box
    xs, ys = out[0] / w, out[1] / w
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def to_gray(frame: np.ndarray) -> np.ndarray:
    """RGB (or already-gray) frame -> uint8 grayscale.

    `pvi.video` yields RGB, not BGR -- ORB is unaffected by channel order, but
    being explicit here stops a future caller assuming OpenCV's convention.
    """
    import cv2

    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
