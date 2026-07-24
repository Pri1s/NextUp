"""Cheap scene-cut detection for shot segmentation.

A full broadcast cuts between the game camera, replays, bench/crowd shots, and
ads. ByteTrack cannot carry identities across such a cut, so the pipeline starts
a new *shot* (and resets tracking) at each detected cut. Detection is a
downscaled grayscale-histogram correlation between consecutive *processed*
frames: a hard cut drops the correlation sharply. This is a deliberately cheap
heuristic (a few microseconds/frame), not shot-boundary science — it exists so
downstream consumers can trust ``(shot_id, track_id)`` as a join key and filter
non-gameplay frames, not to be perfect.
"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_SCENE_THRESHOLD = 0.5  # histogram correlation below this == a cut
DEFAULT_HIST_BINS = 64


class SceneCutDetector:
    """Flags the first frame of a new shot via consecutive-frame histogram drop."""

    def __init__(self, threshold: float = DEFAULT_SCENE_THRESHOLD, hist_bins: int = DEFAULT_HIST_BINS):
        self._threshold = threshold
        self._bins = hist_bins
        self._prev_hist: np.ndarray | None = None

    @staticmethod
    def _histogram(frame_bgr: np.ndarray, bins: int) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
        cv2.normalize(hist, hist)
        return hist

    def is_cut(self, frame_bgr: np.ndarray) -> bool:
        """True when ``frame_bgr`` is a hard cut from the previous processed frame.

        The very first frame primes the detector and returns ``False`` (there is
        nothing to compare against — the runner marks the first frame of a run as
        a shot start on its own). The rolling histogram updates every call, so the
        detector is *not* reset on a cut; only the tracker is.
        """
        hist = self._histogram(frame_bgr, self._bins)
        if self._prev_hist is None:
            self._prev_hist = hist
            return False
        correlation = float(cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL))
        self._prev_hist = hist
        return correlation < self._threshold

    def reset(self) -> None:
        """Drop the rolling reference (e.g. for a hard restart)."""
        self._prev_hist = None
