"""Visual artifacts for human review.

Three views, in increasing order of how much they actually settle the question:

*Indexed overlay* — every slot drawn on its frame with index and confidence.
Good for "is the model looking at the court at all?".

*Contact sheet* — all inspected frames at a glance. Good for spotting which
frames are usable and which are replays or crowd shots.

*Per-slot filmstrip* — one slot cropped across every inspected frame, side by
side. This is the artifact that actually decides semantics: if slot 13 is the
same physical corner every time, twenty-four crops of the same corner make it
obvious, and if it wanders between two corners, that is equally obvious.

Nothing is ever hidden. Below-gate points are drawn hollow and small rather than
dropped, because a slot that is *consistently* low-confidence in a particular
place is itself evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from contracts.calibration_types import RawKeypointFrame

FONT = cv2.FONT_HERSHEY_SIMPLEX
CROP_SIZE = 140
FILMSTRIP_CAPTION_H = 22
CONTACT_TILE_W = 400


def confidence_color(confidence: float) -> tuple[int, int, int]:
    """Continuous red → amber → green ramp in BGR.

    Continuous, not bucketed, so the picture carries the same information the
    record does.
    """
    confidence = min(max(confidence, 0.0), 1.0)
    if confidence < 0.5:
        t = confidence / 0.5
        return (0, int(90 + 165 * t), 255)  # red -> amber
    t = (confidence - 0.5) / 0.5
    return (0, 255, int(255 * (1.0 - t)))  # amber -> green


def draw_overlay(
    frame_bgr: np.ndarray,
    record: RawKeypointFrame,
    gate: float,
    header_lines: Sequence[str] = (),
) -> np.ndarray:
    """Indexed keypoint overlay for one frame."""
    canvas = frame_bgr.copy()

    if not record.detected:
        _draw_banner(canvas, "NO DETECTION", (0, 0, 255))
    else:
        for observation in sorted(record.observations, key=lambda o: o.confidence):
            x, y = int(round(observation.x)), int(round(observation.y))
            color = confidence_color(observation.confidence)
            above = observation.confidence >= gate

            if above:
                cv2.circle(canvas, (x, y), 7, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, (x, y), 8, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, (x, y), 5, color, 1, cv2.LINE_AA)

            if observation.clamped:
                cv2.rectangle(canvas, (x - 11, y - 11), (x + 11, y + 11), (255, 0, 255), 1)

            label = f"{observation.slot_index}:{observation.confidence:.2f}"
            scale = 0.5 if above else 0.38
            thickness = 2 if above else 1
            cv2.putText(canvas, label, (x + 10, y - 8), FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (x + 10, y - 8), FONT, scale, color, thickness, cv2.LINE_AA)

    draw_header(canvas, header_lines)
    _draw_legend(canvas, gate)
    return canvas


def draw_header(canvas: np.ndarray, lines: Sequence[str]) -> None:
    if not lines:
        return
    height = 8 + 20 * len(lines)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (10, 18 + 20 * i), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_legend(canvas: np.ndarray, gate: float) -> None:
    text = f"filled = conf >= {gate:g}   hollow = below gate   magenta box = clamped to frame edge"
    y = canvas.shape[0] - 10
    cv2.putText(canvas, text, (10, y), FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (10, y), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_banner(canvas: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    height, width = canvas.shape[:2]
    size = cv2.getTextSize(text, FONT, 1.4, 3)[0]
    origin = ((width - size[0]) // 2, (height + size[1]) // 2)
    cv2.putText(canvas, text, origin, FONT, 1.4, (0, 0, 0), 7, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, FONT, 1.4, color, 3, cv2.LINE_AA)


def write_overlay(
    path: Path | str,
    frame_bgr: np.ndarray,
    record: RawKeypointFrame,
    gate: float,
    header_lines: Sequence[str] = (),
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), draw_overlay(frame_bgr, record, gate, header_lines))
    return path


def build_contact_sheet(images: Sequence[np.ndarray], columns: int = 4) -> np.ndarray:
    """Grid of downscaled overlays, padded to a full rectangle."""
    if not images:
        raise ValueError("no images for contact sheet")

    tiles = []
    for image in images:
        scale = CONTACT_TILE_W / image.shape[1]
        tiles.append(cv2.resize(image, (CONTACT_TILE_W, max(1, int(image.shape[0] * scale)))))

    tile_h = max(t.shape[0] for t in tiles)
    columns = max(1, min(columns, len(tiles)))
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.zeros((rows * tile_h, columns * CONTACT_TILE_W, 3), dtype=np.uint8)

    for i, tile in enumerate(tiles):
        row, column = divmod(i, columns)
        y, x = row * tile_h, column * CONTACT_TILE_W
        sheet[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    return sheet


def build_slot_filmstrip(
    frames_bgr: Sequence[np.ndarray],
    records: Sequence[RawKeypointFrame],
    slot_index: int,
    gate: float,
    crop_size: int = CROP_SIZE,
) -> np.ndarray:
    """One slot, cropped from every inspected frame, laid out left to right.

    Each tile is captioned with its frame index and confidence, and tinted by a
    border: green-ish above gate, red-ish below, grey when the frame had no
    detection at all. Reading one strip top to bottom answers "is this slot a
    stable physical landmark?" faster than any table can.
    """
    tiles = []
    for frame_bgr, record in zip(frames_bgr, records):
        tiles.append(_slot_tile(frame_bgr, record, slot_index, gate, crop_size))
    if not tiles:
        raise ValueError("no frames for filmstrip")
    return np.hstack(tiles)


def _slot_tile(
    frame_bgr: np.ndarray,
    record: RawKeypointFrame,
    slot_index: int,
    gate: float,
    crop_size: int,
) -> np.ndarray:
    tile = np.zeros((crop_size + FILMSTRIP_CAPTION_H, crop_size, 3), dtype=np.uint8)
    observation = record.observation(slot_index) if record.detected else None

    if observation is None:
        tile[:] = (40, 40, 40)
        caption = f"f{record.frame_index} none"
        border = (90, 90, 90)
    else:
        crop = _centered_crop(frame_bgr, observation.x, observation.y, crop_size)
        tile[:crop_size] = crop
        # Crosshair marks the exact prediction, which sits at the crop centre.
        centre = crop_size // 2
        color = confidence_color(observation.confidence)
        cv2.line(tile, (centre - 12, centre), (centre - 4, centre), color, 1, cv2.LINE_AA)
        cv2.line(tile, (centre + 4, centre), (centre + 12, centre), color, 1, cv2.LINE_AA)
        cv2.line(tile, (centre, centre - 12), (centre, centre - 4), color, 1, cv2.LINE_AA)
        cv2.line(tile, (centre, centre + 4), (centre, centre + 12), color, 1, cv2.LINE_AA)
        cv2.circle(tile, (centre, centre), 2, color, -1, cv2.LINE_AA)

        flag = "" if not observation.clamped else " C"
        caption = f"f{record.frame_index} {observation.confidence:.2f}{flag}"
        border = (0, 200, 0) if observation.confidence >= gate else (0, 90, 200)

    cv2.putText(
        tile,
        caption,
        (4, crop_size + 15),
        FONT,
        0.4,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(tile, (0, 0), (crop_size - 1, crop_size + FILMSTRIP_CAPTION_H - 1), border, 1)
    return tile


def _centered_crop(frame_bgr: np.ndarray, x: float, y: float, size: int) -> np.ndarray:
    """Crop ``size x size`` around ``(x, y)``, zero-padded past the frame edge.

    Padding rather than shifting keeps the prediction exactly at the tile centre,
    so a reviewer scanning a strip compares like with like even for points that
    sit on or beyond the border.
    """
    height, width = frame_bgr.shape[:2]
    half = size // 2
    cx, cy = int(round(x)), int(round(y))

    crop = np.zeros((size, size, 3), dtype=frame_bgr.dtype)
    src_x0, src_y0 = max(0, cx - half), max(0, cy - half)
    src_x1, src_y1 = min(width, cx - half + size), min(height, cy - half + size)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return crop

    dst_x0, dst_y0 = src_x0 - (cx - half), src_y0 - (cy - half)
    crop[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = frame_bgr[
        src_y0:src_y1, src_x0:src_x1
    ]
    return crop


def write_image(path: Path | str, image: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return path
