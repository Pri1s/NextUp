"""The pipeline's output contract: per-frame records + a resumable run manifest.

Records stream to append-only JSONL shards (``records/seg_%05d.jsonl``), one
shard per segment, so nothing accumulates in memory and a crash costs at most one
in-flight segment. The run manifest (``run.json``) records which segments are
``done`` and enough config to detect a stale reuse; it is written atomically at
every segment boundary as the checkpoint.

Frame-record schema (``frames-1.0.0``) — one JSON object per processed frame:

    frame_index   int    native (unstrided) index — the resume/join anchor
    timestamp_s   float
    shot_id       int    increments on scene cut / resume boundary
    scene_cut     bool   true on the first frame of a new shot
    court         {visible_keypoints, homography|null, used_keypoint_ids, keypoints|null}
    players       [{track_id|null, cls, name, conf, bbox, foot_px, court_ft|null}]
    ball          {conf, bbox, center_px, court_ft|null} | null
    error         str|null   set when the frame failed; other fields are minimal

Downstream join contract: ``track_id`` is unique and continuous ONLY within a
``shot_id``. Join a player across frames on ``(shot_id, track_id)``; never assume
continuity across a ``scene_cut`` or a ``--resume`` boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "frames-1.0.0"
MANIFEST_SCHEMA_VERSION = "run-1.0.0"
RECORDS_SUBDIR = "records"
MANIFEST_NAME = "run.json"


# --------------------------------------------------------------------------- #
# Record builders (kept here so rounding/shape is consistent across the code).
# --------------------------------------------------------------------------- #

def _round_pt(pt, ndigits: int = 1):
    return [round(float(pt[0]), ndigits), round(float(pt[1]), ndigits)]


def player_dict(det, court_ft) -> dict:
    x1, y1, x2, y2 = det.xyxy
    return {
        "track_id": int(det.track_id) if det.track_id is not None else None,
        "cls": int(det.cls),
        "name": det.name,
        "conf": round(float(det.conf), 4),
        "bbox": [round(float(v), 1) for v in (x1, y1, x2, y2)],
        "foot_px": _round_pt(det.foot_point()),
        "court_ft": _round_pt(court_ft, 2) if court_ft is not None else None,
    }


def ball_dict(det, court_ft) -> dict:
    x1, y1, x2, y2 = det.xyxy
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return {
        "conf": round(float(det.conf), 4),
        "bbox": [round(float(v), 1) for v in (x1, y1, x2, y2)],
        "center_px": _round_pt(center),
        "court_ft": _round_pt(court_ft, 2) if court_ft is not None else None,
    }


def court_dict(visible_keypoints, homography, used_keypoint_ids, keypoints=None) -> dict:
    return {
        "visible_keypoints": int(visible_keypoints),
        "homography": (
            [[round(float(v), 6) for v in row] for row in np.asarray(homography)]
            if homography is not None
            else None
        ),
        "used_keypoint_ids": list(used_keypoint_ids),
        "keypoints": (
            [[round(float(v), 2) for v in row] for row in np.asarray(keypoints)]
            if keypoints is not None
            else None
        ),
    }


@dataclass
class FrameRecord:
    frame_index: int
    timestamp_s: float
    shot_id: int
    scene_cut: bool
    court: dict
    players: list = field(default_factory=list)
    ball: dict | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "frame_index": int(self.frame_index),
            "timestamp_s": float(self.timestamp_s),
            "shot_id": int(self.shot_id),
            "scene_cut": bool(self.scene_cut),
            "court": self.court,
            "players": self.players,
            "ball": self.ball,
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# Shard writer.
# --------------------------------------------------------------------------- #

class RecordSink:
    """Append-only JSONL shard writer that opens one file per segment index.

    The runner decides *when* to roll (based on ``segment_frames``); the sink owns
    file lifecycle. Each write is flushed. Opening a shard uses ``"w"`` so
    re-running an un-checkpointed segment overwrites its partial shard rather than
    appending duplicates (idempotent resume).
    """

    def __init__(self, records_dir: Path | str, start_segment: int = 0):
        self._dir = Path(records_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._segment_index = start_segment
        self._handle = None
        self._path: Path | None = None
        self.n_in_segment = 0

    @property
    def segment_index(self) -> int:
        return self._segment_index

    @property
    def path(self) -> Path | None:
        return self._path

    def write(self, record: FrameRecord | dict) -> None:
        if self._handle is None:
            self._path = self._dir / f"seg_{self._segment_index:05d}.jsonl"
            self._handle = open(self._path, "w", encoding="utf-8")
            self.n_in_segment = 0
        payload = record.to_dict() if isinstance(record, FrameRecord) else record
        json.dump(payload, self._handle)
        self._handle.write("\n")
        self._handle.flush()
        self.n_in_segment += 1

    def roll(self) -> None:
        """Close the current shard and advance to the next segment index."""
        self.close()
        self._segment_index += 1

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


# --------------------------------------------------------------------------- #
# Run manifest.
# --------------------------------------------------------------------------- #

def run_dir_for(out_dir: Path | str, video_stem: str, profile_name: str) -> Path:
    return Path(out_dir) / f"{video_stem}.{profile_name}"


def resume_key(video_path: Path | str, config: dict) -> str:
    """Cheap staleness key: file size + mtime + a hash of the run config.

    Deliberately NOT a content hash. Tradeoff (write it down): this can
    false-invalidate a byte-identical copy (new mtime) and, less likely,
    false-reuse an in-place edit that preserves size+mtime. It still fixes the
    reference pipeline's ``len == frame_count``-only staleness bug, but it is not
    collision-proof.
    """
    stat = os.stat(video_path)
    config_blob = json.dumps(config, sort_keys=True, default=str)
    digest = hashlib.sha1(config_blob.encode("utf-8")).hexdigest()[:12]
    return f"{stat.st_size}-{int(stat.st_mtime)}-{digest}"


def new_run_manifest(video_path, profile_name, video_info, config, key) -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "record_schema_version": SCHEMA_VERSION,
        "video": str(video_path),
        "resume_key": key,
        "profile": profile_name,
        "video_info": video_info,
        "config": config,
        "last_shot_id": -1,
        "segments": [],
        "summary": {},
        "status": "running",
    }


def load_run_manifest(run_dir: Path | str) -> dict | None:
    path = Path(run_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_run_manifest(run_dir: Path | str, data: dict) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".run-", suffix=".json", dir=run_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        os.replace(temp_path, run_dir / MANIFEST_NAME)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def done_segments(manifest: dict) -> list[dict]:
    return [seg for seg in manifest.get("segments", []) if seg.get("status") == "done"]


@dataclass
class ResumePoint:
    """Where a resumed run picks up. All frame numbers are native (unstrided)."""

    start_index: int      # first native frame to process
    start_segment: int    # next shard index to write
    start_shot: int       # shot id for the first frame after resume


def resume_point(manifest: dict | None) -> ResumePoint:
    """Compute the resume anchor from a manifest's finalized segments.

    ``start_index`` is the native frame *after* the last done segment's
    ``end_frame`` (``iter_frames``'s ``% stride`` gate then lands on the next
    frame we would have processed). A resume always begins a fresh shot, so the
    tracker restarts and ``(shot_id, track_id)`` never collides across the seam.
    """
    if not manifest:
        return ResumePoint(start_index=0, start_segment=0, start_shot=0)
    segs = done_segments(manifest)
    if not segs:
        return ResumePoint(start_index=0, start_segment=0, start_shot=0)
    last = max(segs, key=lambda s: s["index"])
    return ResumePoint(
        start_index=int(last["end_frame"]) + 1,
        start_segment=int(last["index"]) + 1,
        start_shot=int(manifest.get("last_shot_id", -1)) + 1,
    )
