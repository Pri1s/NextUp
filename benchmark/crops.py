"""Magnified crops, and the arithmetic the annotator must never do.

A court line is two to four pixels wide near the baseline and about one pixel at
midcourt. Asking any annotator, human or model, to name the centre of that to
within a pixel while looking at a 1280x720 frame is asking for a guess. Magnifying
an 80 px window sixteen times turns the same judgement into "which of these large
squares is the middle one", and one crop pixel then represents a sixteenth of a
frame pixel -- far finer than the precision anyone needs.

The window is deliberately small. A wide window is the tempting mistake, because one
crop can then cover a whole marking and be read in a single pass; ``MIN_MEASURING_SCALE``
exists because the probe showed exactly what that costs.

The second half is quieter and matters as much. Having read a coordinate off the
crop, someone has to convert it back to the frame. If the annotator does that
multiplication, a single slip produces a coordinate that is wrong by an arbitrary
amount, is perfectly well-formed, and passes every structural check. So the crop
records its own exact transform and ``resolve_annotation`` applies it. The
annotator never sees a frame coordinate and is never asked to compute one.

Nearest-neighbour magnification is the default deliberately. Smooth interpolation
invents a gradient across a stripe edge, and where that invented gradient puts its
midpoint is exactly the question being asked; blocky pixels are honest about what
was actually measured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from contracts.annotations import (
    AnnotatedFeature,
    AnnotatedJunction,
    FrameAnnotation,
    SamplePoint,
)

CROP_SCHEMA_VERSION = "benchmark-crop-1.0.0"

DEFAULT_SIZE_PX = 80
DEFAULT_SCALE = 16
DEFAULT_INTERP = "nearest"

#: A gridded crop is a measuring instrument, and below this magnification it is not
#: an accurate one. Measured on the synthetic probe: features read from tight 16-24x
#: crops came in at 0.27-0.94 px median error, while every feature read from a wide
#: 5-7x crop landed between 1.09 and 9.04 px -- the 9 px case being a stripe crossing
#: floor-logo lettering, where a low-magnification view cannot separate paint from
#: paint-coloured noise. The floor is enforced rather than recommended because the
#: cheap mistake is to widen the window to cover a whole feature in one reading, and
#: the resulting error is smooth, plausible and invisible to a second pass that does
#: the same thing. Looking at a wide view is still fine -- pass ``grid=False``.
MIN_MEASURING_SCALE = 12

#: Grid spacing, in *frame* pixels. Labelled lines every 10 frame pixels give the
#: annotator a number to anchor on; faint lines every 2 subdivide it. Anything
#: finer competes with the nearest-neighbour pixel blocks, which already show
#: single-frame-pixel structure -- and a grid dense enough to hide the paint
#: defeats the point of magnifying it.
MAJOR_STEP_FRAME_PX = 10
MINOR_STEP_FRAME_PX = 5

_INTERPOLATIONS = {"nearest": cv2.INTER_NEAREST, "lanczos": cv2.INTER_LANCZOS4}

#: Magenta reads against hardwood, painted lane colour and white paint alike.
#: Court markings are never magenta, so nothing on the floor can be mistaken for
#: the grid or the grid for a marking.
_MINOR_ALPHA = 0.20
_MAJOR_ALPHA = 0.45
_GRID_COLOR = (255, 0, 255)


@dataclass(frozen=True, slots=True)
class CropTransform:
    """A crop and the exact affine that produced it.

    ``crop = (frame - origin) * scale``. Stored rather than recomputed, so a crop
    written by one process resolves identically in another with no shared state.
    """

    crop_id: str
    frame_id: str
    origin_x: float
    origin_y: float
    scale: int
    size_px: int
    interp: str
    path: str

    def to_frame(self, crop_x: float, crop_y: float) -> tuple[float, float]:
        return (crop_x / self.scale + self.origin_x, crop_y / self.scale + self.origin_y)

    def to_crop(self, frame_x: float, frame_y: float) -> tuple[float, float]:
        return ((frame_x - self.origin_x) * self.scale, (frame_y - self.origin_y) * self.scale)

    @property
    def frame_bounds(self) -> tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` of the frame region this crop covers."""
        return (
            self.origin_x,
            self.origin_y,
            self.origin_x + self.size_px,
            self.origin_y + self.size_px,
        )

    def as_dict(self) -> dict:
        return {
            "schema_version": CROP_SCHEMA_VERSION,
            "crop_id": self.crop_id,
            "frame_id": self.frame_id,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "scale": self.scale,
            "size_px": self.size_px,
            "interp": self.interp,
            "path": self.path,
            "frame_bounds": list(self.frame_bounds),
            "crop_px_per_frame_px": self.scale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CropTransform":
        schema = data.get("schema_version")
        if schema is not None and schema != CROP_SCHEMA_VERSION:
            raise ValueError(f"unsupported crop schema {schema!r}, expected {CROP_SCHEMA_VERSION!r}")
        return cls(
            crop_id=str(data["crop_id"]),
            frame_id=str(data["frame_id"]),
            origin_x=float(data["origin_x"]),
            origin_y=float(data["origin_y"]),
            scale=int(data["scale"]),
            size_px=int(data["size_px"]),
            interp=str(data.get("interp", DEFAULT_INTERP)),
            path=str(data["path"]),
        )


_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.44


def _label(canvas: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    """Draw text with a dark halo, so it survives both light wood and dark paint."""
    for colour, thickness in (((0, 0, 0), 4), (_GRID_COLOR, 1)):
        cv2.putText(canvas, text, origin, _FONT, _FONT_SCALE, colour, thickness, cv2.LINE_AA)


def _centred_label(canvas: np.ndarray, text: str, axis: str, position: int) -> None:
    """Place a label *centred on* the line it names.

    Offsetting a label to one side of its gridline is ambiguous, and the ambiguity
    is not academic: during development it caused readings a whole major cell out,
    every one of them in the same direction. That is the worst possible failure
    shape here, because a bias shared by both annotation passes is invisible to
    the agreement check that is supposed to catch mistakes -- only the synthetic
    ground-truth probe saw it. Centring removes the ambiguity at the source.
    """
    height, width = canvas.shape[:2]
    (text_w, text_h), _ = cv2.getTextSize(text, _FONT, _FONT_SCALE, 1)
    if axis == "x":
        x = int(min(max(position - text_w // 2, 2), width - text_w - 2))
        _label(canvas, text, (x, text_h + 6))
    else:
        y = int(min(max(position + text_h // 2, text_h + 2), height - 4))
        _label(canvas, text, (4, y))


def _draw_grid(canvas: np.ndarray, transform: CropTransform) -> None:
    """Overlay a coordinate grid labelled in crop pixels.

    The labels are what turn estimation into reading. Without them an annotator
    still has to judge "about 60% across", which reintroduces exactly the error
    magnification was meant to remove. With them the judgement becomes "two faint
    lines past 320", which is bounded and checkable.

    Minor and major lines are composited at different opacities in two passes, so
    the anchor lines stay findable while the subdivisions stay out of the way of
    the paint.
    """
    height, width = canvas.shape[:2]
    minor = MINOR_STEP_FRAME_PX * transform.scale
    major = MAJOR_STEP_FRAME_PX * transform.scale

    # Major lines are drawn thicker as well as more opaque, so the line a label
    # belongs to is identifiable from the line alone.
    for step, alpha, thickness in ((minor, _MINOR_ALPHA, 1), (major, _MAJOR_ALPHA, 2)):
        if step <= 0:
            continue
        overlay = canvas.copy()
        for x in range(0, width + 1, step):
            cv2.line(overlay, (x, 0), (x, height), _GRID_COLOR, thickness)
        for y in range(0, height + 1, step):
            cv2.line(overlay, (0, y), (width, y), _GRID_COLOR, thickness)
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

    # Labels last, at full opacity: a faint number is a misread number.
    for x in range(0, width + 1, major):
        _centred_label(canvas, str(x), "x", x)
    for y in range(major, height + 1, major):
        _centred_label(canvas, str(y), "y", y)


def crop_id_for(frame_id: str, origin_x: int, origin_y: int, size_px: int, scale: int) -> str:
    """Deterministic id, so requesting the same crop twice reuses one file."""
    return f"{frame_id}__x{origin_x}_y{origin_y}_s{size_px}_z{scale}"


def make_crop(
    frames_dir: Path | str,
    frame_id: str,
    center: tuple[float, float],
    size_px: int = DEFAULT_SIZE_PX,
    scale: int = DEFAULT_SCALE,
    interp: str = DEFAULT_INTERP,
    out_dir: Path | str | None = None,
    grid: bool = True,
) -> CropTransform:
    """Write a magnified, grid-labelled crop and return its exact transform.

    The window is clamped to the image rather than padded. A padded crop would put
    invented black pixels next to real ones at the same magnification, and an
    annotator has no way to tell which is which.
    """
    if size_px <= 0 or scale <= 0:
        raise ValueError(f"size_px and scale must be positive, got {size_px} and {scale}")
    if grid and scale < MIN_MEASURING_SCALE:
        raise ValueError(
            f"scale {scale} is too low to read a coordinate from: a gridded crop is a "
            f"measuring instrument and needs at least {MIN_MEASURING_SCALE}x. Sample the "
            f"marking with several tight crops instead of covering it with one wide one, "
            f"or pass grid=False if you only want to look."
        )
    if interp not in _INTERPOLATIONS:
        raise ValueError(f"unknown interpolation {interp!r}; choose from {sorted(_INTERPOLATIONS)}")

    frames_dir = Path(frames_dir)
    image_path = frames_dir / "images" / f"{frame_id}.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"no exported frame {frame_id!r} at {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"could not read {image_path}")
    height, width = image.shape[:2]

    size_px = min(size_px, width, height)
    origin_x = int(round(center[0] - size_px / 2.0))
    origin_y = int(round(center[1] - size_px / 2.0))
    origin_x = max(0, min(origin_x, width - size_px))
    origin_y = max(0, min(origin_y, height - size_px))

    window = image[origin_y : origin_y + size_px, origin_x : origin_x + size_px]
    magnified = cv2.resize(
        window, (size_px * scale, size_px * scale), interpolation=_INTERPOLATIONS[interp]
    )

    crop_id = crop_id_for(frame_id, origin_x, origin_y, size_px, scale)
    out_dir = Path(out_dir) if out_dir is not None else frames_dir.parent / "crops"
    crop_dir = out_dir / frame_id
    crop_dir.mkdir(parents=True, exist_ok=True)
    path = crop_dir / f"{crop_id}.png"

    transform = CropTransform(
        crop_id=crop_id,
        frame_id=frame_id,
        origin_x=float(origin_x),
        origin_y=float(origin_y),
        scale=scale,
        size_px=size_px,
        interp=interp,
        path=str(path),
    )

    if grid:
        _draw_grid(magnified, transform)
    if not cv2.imwrite(str(path), magnified):
        raise OSError(f"failed to write {path}")

    with open(crop_dir / f"{crop_id}.json", "w", encoding="utf-8") as handle:
        json.dump(transform.as_dict(), handle, indent=2)
        handle.write("\n")

    return transform


def load_crops(crops_dir: Path | str) -> dict[str, CropTransform]:
    """Every crop transform written under ``crops_dir``, keyed by crop id."""
    crops_dir = Path(crops_dir)
    transforms: dict[str, CropTransform] = {}
    if not crops_dir.is_dir():
        return transforms
    for path in sorted(crops_dir.rglob("*.json")):
        with open(path, encoding="utf-8") as handle:
            transform = CropTransform.from_dict(json.load(handle))
        transforms[transform.crop_id] = transform
    return transforms


class UnknownCrop(KeyError):
    """A point references a crop that was never written."""


def _resolve_point(point: SamplePoint, crops: dict[str, CropTransform], frame_id: str) -> SamplePoint:
    if point.crop_id is None:
        raise ValueError("cannot resolve a point that carries no crop_id")
    transform = crops.get(point.crop_id)
    if transform is None:
        raise UnknownCrop(
            f"{point.crop_id!r} is not among the crops on disk; the annotation "
            f"references a crop that was never written"
        )
    if transform.frame_id != frame_id:
        raise ValueError(
            f"crop {point.crop_id!r} belongs to frame {transform.frame_id!r}, "
            f"but the annotation is for {frame_id!r}"
        )
    x, y = transform.to_frame(point.x, point.y)
    return SamplePoint(x=x, y=y, crop_id=None)


def resolve_annotation(
    annotation: FrameAnnotation, crops: dict[str, CropTransform]
) -> FrameAnnotation:
    """Map every crop coordinate into frame space.

    Refuses rather than guesses on a missing or mismatched crop. A silently dropped
    point would leave a shorter feature that still looks plausible, and a point
    resolved against the wrong crop would land somewhere entirely else with no
    outward sign.
    """
    if annotation.coordinate_space == "frame":
        return annotation

    features = tuple(
        AnnotatedFeature(
            feature=item.feature,
            points=tuple(_resolve_point(p, crops, annotation.frame_id) for p in item.points),
            visibility=item.visibility,
            uncertainty_px=item.uncertainty_px,
            occluded_after=item.occluded_after,
            notes=item.notes,
        )
        for item in annotation.features
    )
    junctions = tuple(
        AnnotatedJunction(
            junction=item.junction,
            point=_resolve_point(item.point, crops, annotation.frame_id),
            uncertainty_px=item.uncertainty_px,
            notes=item.notes,
        )
        for item in annotation.junctions
    )

    return FrameAnnotation(
        frame_id=annotation.frame_id,
        clip=annotation.clip,
        frame_index=annotation.frame_index,
        image_width=annotation.image_width,
        image_height=annotation.image_height,
        image_sha256=annotation.image_sha256,
        annotator_id=annotation.annotator_id,
        pass_id=annotation.pass_id,
        prompt_version=annotation.prompt_version,
        coordinate_space="frame",
        features=features,
        junctions=junctions,
        skipped=annotation.skipped,
        notes=annotation.notes,
    )
