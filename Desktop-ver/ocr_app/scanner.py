"""Document scanning helpers: find a page in a camera frame and flatten it."""
from __future__ import annotations

import cv2
import numpy as np


def find_document(frame_bgr: np.ndarray, min_area_ratio: float = 0.2) -> np.ndarray | None:
    """Return the 4 corners (float32, shape (4, 2)) of the largest quadrilateral
    covering at least ``min_area_ratio`` of the frame, or None."""
    h, w = frame_bgr.shape[:2]
    scale = 500.0 / max(h, w)
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale) if scale < 1 else frame_bgr
    scale = min(scale, 1.0)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_ratio * small.shape[0] * small.shape[1]
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(cnt) < min_area:
            break
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32) / scale
    return None


def _unique_count(pts: np.ndarray, tol: float = 1.0) -> int:
    kept: list[np.ndarray] = []
    for point in pts:
        if all(np.linalg.norm(point - other) > tol for other in kept):
            kept.append(point)
    return len(kept)


def _quad_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _best_rotation(ring: np.ndarray) -> np.ndarray | None:
    """Pick the cyclic start whose top edge is highest and points right."""
    best = None
    best_key = None
    for i in range(4):
        ordered = np.roll(ring, -i, axis=0)
        tl, tr, br, bl = ordered
        top = tr - tl
        if float(top[0]) <= 1e-3:
            continue
        top_y = float(tl[1] + tr[1]) * 0.5
        bot_y = float(bl[1] + br[1]) * 0.5
        if top_y >= bot_y:
            continue
        # Prefer the higher edge, then the one closer to horizontal.
        horiz = abs(float(top[1])) / (abs(float(top[0])) + 1e-6)
        key = (top_y, horiz)
        if best_key is None or key < best_key:
            best_key = key
            best = ordered
    if best is None:
        return None
    return np.ascontiguousarray(best, dtype=np.float32)


def _order_corners(pts: np.ndarray) -> np.ndarray | None:
    """Order as top-left, top-right, bottom-right, bottom-left.

    Points are walked around their centroid so every corner is used once.
    The old sum/diff ordering picks the same corner twice when a page is
    rotated near 45 degrees, and the warp then comes out blank.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4 or _unique_count(pts) < 4:
        return None
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ring = pts[np.argsort(angles)]
    ordered = _best_rotation(ring)
    if ordered is None:
        ordered = _best_rotation(ring[::-1])
    if ordered is None or _quad_area(ordered) < 1.0:
        return None
    return ordered


def quad_drift(a: np.ndarray, b: np.ndarray) -> float:
    """Largest corner movement, comparing x and y independently of order."""
    a = np.asarray(a, dtype=np.float32).reshape(4, 2)
    b = np.asarray(b, dtype=np.float32).reshape(4, 2)
    return float(np.abs(np.sort(a, axis=0) - np.sort(b, axis=0)).max())


def warp_document(frame_bgr: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
    """Perspective-correct the page delimited by ``corners``.

    Returns None when the quad is degenerate. OpenCV's warp with a 0-pixel
    side silently returns the source image, which looks like a successful scan.
    """
    rect = _order_corners(corners)
    if rect is None:
        return None
    tl, tr, br, bl = rect
    width = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    if width < 2 or height < 2:
        return None
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(rect, dst)
    if not np.isfinite(matrix).all():
        return None
    return cv2.warpPerspective(frame_bgr, matrix, (width, height))


def draw_boxes(image_bgr: np.ndarray, boxes, color=(0, 200, 0), thickness=2) -> np.ndarray:
    out = image_bgr.copy()
    for box in boxes:
        cv2.polylines(out, [np.asarray(box, dtype=np.int32).reshape(-1, 1, 2)],
                      True, color, thickness)
    return out
