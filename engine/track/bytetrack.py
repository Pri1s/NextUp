"""Persistent player IDs via supervision's ByteTrack.

Ported in spirit from ``basketball_analysis/trackers/player_tracker.py`` but kept
framework-neutral: it consumes ``engine.detect.Detection`` rows and returns
``Detection`` rows carrying a ``track_id``, so nothing downstream ever touches a
supervision object.

ByteTrack is stateful in-process (Kalman filters, track ages, an id counter) and
cannot preserve ids across a camera cut or a cold restart. The pipeline therefore
calls :meth:`reset` at every shot boundary (scene cut) and at a ``--resume``
boundary; ids are only continuous *within* a shot. It is constructed with the
*processed* frame rate (``frame_rate == target_fps``), not the source fps, so the
lost-track buffer expires in the right wall-clock window despite the stride.

``supervision`` is imported lazily so the rest of the engine (and the unit tests,
which inject a fake tracker) do not require it to be installed.
"""

from __future__ import annotations

import numpy as np

from engine.detect.players import Detection


class PlayerTracker:
    """Assign shot-scoped persistent track ids to per-frame detections."""

    def __init__(self, frame_rate: float = 5.0, **bytetrack_kwargs):
        self._frame_rate = max(1, round(frame_rate))
        self._bytetrack_kwargs = bytetrack_kwargs
        self._tracker = self._new_tracker()

    def _new_tracker(self):
        from supervision import ByteTrack

        return ByteTrack(frame_rate=self._frame_rate, **self._bytetrack_kwargs)

    def reset(self) -> None:
        """New shot / resume boundary: drop all track state and restart ids."""
        self._tracker = self._new_tracker()

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Return the tracked subset of ``detections``, each with a ``track_id``.

        Detections ByteTrack drops (low score / unmatched) are not returned — the
        output is the set of players with a stable identity this frame. Class
        names are carried over from the inputs by class id.
        """
        from supervision import Detections

        if not detections:
            self._tracker.update_with_detections(Detections.empty())
            return []

        name_by_cls = {d.cls: d.name for d in detections}
        sv_dets = Detections(
            xyxy=np.array([d.xyxy for d in detections], dtype=float),
            confidence=np.array([d.conf for d in detections], dtype=float),
            class_id=np.array([d.cls for d in detections], dtype=int),
        )
        tracked = self._tracker.update_with_detections(sv_dets)

        out: list[Detection] = []
        for i in range(len(tracked)):
            tid = tracked.tracker_id[i] if tracked.tracker_id is not None else None
            cls = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
            out.append(
                Detection(
                    xyxy=tuple(float(v) for v in tracked.xyxy[i]),
                    cls=cls,
                    name=name_by_cls.get(cls, str(cls)),
                    conf=conf,
                    track_id=int(tid) if tid is not None else None,
                )
            )
        return out
