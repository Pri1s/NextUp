"""Deterministic frame selection.

A Phase-0 verdict has to be reproducible: the same clip and the same arguments
must inspect the same frames, or a second reviewer cannot check the first. So
sampling is evenly spaced and index-based — no randomness, no seeds to lose.
"""

from __future__ import annotations

DEFAULT_NUM_FRAMES = 24  # the design doc's floor is 20


def even_indices(frame_count: int, num_frames: int) -> list[int]:
    """``num_frames`` indices spread across ``0 .. frame_count - 1``.

    Spacing places samples at bin centres rather than at both endpoints, which
    keeps the first and last samples off the frames most likely to be a fade or a
    cut. Requesting more frames than exist simply returns every frame.
    """
    if frame_count <= 0:
        return []
    if num_frames <= 0:
        return []
    if num_frames >= frame_count:
        return list(range(frame_count))

    step = frame_count / num_frames
    indices = sorted({int(step * (i + 0.5)) for i in range(num_frames)})
    return [min(i, frame_count - 1) for i in indices]


def parse_frame_list(spec: str, frame_count: int | None = None) -> list[int]:
    """Parse ``"0,40,80"`` into sorted unique indices.

    Out-of-range indices are dropped rather than clamped — silently moving a
    reviewer's explicitly requested frame to a different one would be worse than
    telling them it wasn't there.
    """
    indices: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as error:
            raise ValueError(f"not a frame index: {token!r}") from error
        if value < 0:
            raise ValueError(f"negative frame index: {value}")
        if frame_count is not None and value >= frame_count:
            continue
        indices.add(value)
    if not indices:
        raise ValueError(f"no usable frame indices in {spec!r}")
    return sorted(indices)


def resolve_indices(
    frame_count: int,
    num_frames: int = DEFAULT_NUM_FRAMES,
    explicit: str | None = None,
) -> list[int]:
    """Explicit list when given, otherwise even spacing."""
    if explicit:
        return parse_frame_list(explicit, frame_count)
    return even_indices(frame_count, num_frames)
