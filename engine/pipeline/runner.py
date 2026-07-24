"""The streaming pipeline orchestrator.

One pass over the clip, batched for detector throughput, sequential for tracking:

    iter_frames(stride, start_index) ── batches of N ──▶
      per frame (in order):
        scene-cut? ─▶ new shot_id + reset tracker
        court model ─▶ canonical kpts ─▶ homography
        player/ball detectors ─▶ ByteTrack ─▶ project to court feet
        FrameRecord ─▶ RecordSink shard
      every `segment_frames`: checkpoint (atomic manifest write) + roll shard

Nothing holds more than one batch of frames; the run is resumable at segment
granularity. Each frame's stage chain is isolated so one bad frame (corrupt
decode, model hiccup) is recorded as an ``error`` and the run continues; only a
long run of consecutive failures aborts.

``run_pipeline`` builds its stages from a profile by default, but accepts an
injected :class:`Stages` so tests can drive the whole loop with fakes (no real
weights, no ``supervision`` install).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from contracts.court_template import load_template
from engine.geometry.homography import project, solve_homography
from engine.io.video import iter_frames, probe
from engine.profiles import build_ball_detector, build_court_model, build_detector, get_profile
from engine.pipeline.records import (
    MANIFEST_NAME,
    RECORDS_SUBDIR,
    FrameRecord,
    RecordSink,
    ball_dict,
    court_dict,
    load_run_manifest,
    new_run_manifest,
    player_dict,
    resume_key,
    resume_point,
    run_dir_for,
    save_run_manifest,
)
from engine.pipeline.scene import DEFAULT_SCENE_THRESHOLD, SceneCutDetector

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "engine_out"


@dataclass
class PipelineConfig:
    target_fps: float = 5.0
    segment_frames: int = 1000
    batch: int = 8
    device: str | None = None
    ball_conf: float = 0.3
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD
    max_frames: int | None = None
    keep_keypoints: bool = False
    fmt: str = "jsonl"
    resume: bool = True
    max_consecutive_errors: int = 100


@dataclass
class Stages:
    """The per-frame processing units. Injectable for testing.

    ``tracker`` must expose ``update(list[Detection]) -> list[Detection]`` and
    ``reset()``. ``player_detector``/``ball_detector`` may be ``None``.
    """

    court_model: object
    template: object
    tracker: object
    player_detector: object | None = None
    ball_detector: object | None = None
    player_classes: tuple[str, ...] | None = None  # keep only these names (None = all)
    ball_classes: tuple[str, ...] | None = None


# --------------------------------------------------------------------------- #
# Per-frame processing.
# --------------------------------------------------------------------------- #

def _filter_classes(dets, names: tuple[str, ...] | None):
    if names is None:
        return list(dets or [])
    keep = set(names)
    return [d for d in (dets or []) if d.name in keep]


def _pick_ball(ball_dets, min_conf: float):
    candidates = [d for d in (ball_dets or []) if d.conf >= min_conf]
    return max(candidates, key=lambda d: d.conf) if candidates else None


def _process_frame(frame_index, ts, shot_id, scene_cut, frame, pd, bd, stages, config) -> FrameRecord:
    """Run the full stage chain for one frame. May raise — the caller isolates it."""
    canonical = np.asarray(stages.court_model.predict(frame))
    visible = int((canonical[:, 2] > 0).sum())
    homography, used = solve_homography(canonical, stages.template)

    if pd is None:
        pd = stages.player_detector.detect(frame) if stages.player_detector is not None else []
    pd = _filter_classes(pd, stages.player_classes)
    tracked = stages.tracker.update(pd) if stages.tracker is not None else pd
    foot_pts = [d.foot_point() for d in tracked]
    court_pts = project(homography, foot_pts) if (homography is not None and foot_pts) else None
    players = [
        player_dict(d, court_pts[i] if court_pts is not None else None)
        for i, d in enumerate(tracked)
    ]

    if bd is None:
        bd = stages.ball_detector.detect(frame) if stages.ball_detector is not None else []
    bd = _filter_classes(bd, stages.ball_classes)
    ball = _pick_ball(bd, config.ball_conf)
    ball_rec = None
    if ball is not None:
        cx = (ball.xyxy[0] + ball.xyxy[2]) / 2.0
        cy = (ball.xyxy[1] + ball.xyxy[3]) / 2.0
        ball_ct = project(homography, [(cx, cy)])[0] if homography is not None else None
        ball_rec = ball_dict(ball, ball_ct)

    keypoints = canonical if config.keep_keypoints else None
    court = court_dict(visible, homography, used, keypoints)
    return FrameRecord(frame_index, ts, shot_id, scene_cut, court, players, ball_rec)


def _safe_batch_detect(detector, frames):
    """Batched detect for throughput; ``None`` signals a per-frame fallback.

    When there is no detector, returns empty rows so the frame simply has no
    such detections. When a batch call raises, returns ``None`` so each frame
    re-detects inside its own try/except (error isolation on the slow path).
    """
    if detector is None:
        return [[] for _ in frames]
    try:
        return detector.detect_batch(frames)
    except Exception:  # noqa: BLE001 - fall back to per-frame detection
        return None


# --------------------------------------------------------------------------- #
# Segment / manifest bookkeeping.
# --------------------------------------------------------------------------- #

def _new_seg(index: int) -> dict:
    return {"index": index, "start_frame": None, "end_frame": None, "n_records": 0, "errors": 0}


def _account(seg: dict, stats: dict, record: FrameRecord, frame_index: int) -> None:
    if seg["start_frame"] is None:
        seg["start_frame"] = frame_index
    seg["end_frame"] = frame_index
    seg["n_records"] += 1
    stats["processed"] += 1
    stats["shots"].add(record.shot_id)
    if record.error:
        seg["errors"] += 1
        stats["errors"] += 1
        return
    if record.court.get("homography") is not None:
        stats["with_homography"] += 1
    for p in record.players:
        if p.get("track_id") is not None:
            stats["tracks"].add((record.shot_id, p["track_id"]))


def _summary(stats: dict) -> dict:
    return {
        "frames_processed": stats["processed"],
        "frames_with_homography": stats["with_homography"],
        "errors": stats["errors"],
        "shots": len(stats["shots"]),
        "unique_tracks": len(stats["tracks"]),
    }


def _finalize_segment(manifest: dict, seg: dict, current_shot: int, stats: dict, run_dir: Path) -> None:
    entry = dict(seg)
    entry["path"] = f"{RECORDS_SUBDIR}/seg_{seg['index']:05d}.jsonl"
    entry["status"] = "done"
    manifest["segments"].append(entry)
    manifest["last_shot_id"] = current_shot
    manifest["summary"] = _summary(stats)
    manifest["status"] = "running"
    save_run_manifest(run_dir, manifest)  # the checkpoint


# --------------------------------------------------------------------------- #
# Setup helpers.
# --------------------------------------------------------------------------- #

def _info_dict(info) -> dict:
    return {"fps": info.fps, "width": info.width, "height": info.height, "frame_count": info.frame_count}


def _config_dict(config: PipelineConfig, stride: int, info, profile) -> dict:
    data = {
        "target_fps": config.target_fps,
        "stride": stride,
        "segment_frames": config.segment_frames,
        "batch": config.batch,
        "device": config.device,
        "ball_conf": config.ball_conf,
        "scene_threshold": config.scene_threshold,
        "keep_keypoints": config.keep_keypoints,
        "format": config.fmt,
        "source_fps": info.fps,
    }
    if profile is not None:
        data["court_weights"] = str(profile.court_weights)
        data["detector_weights"] = str(profile.detector_weights) if profile.detector_weights else None
        data["ball_weights"] = str(profile.ball_weights) if profile.ball_weights else None
        data["court_template"] = profile.court_template
    return data


def _build_stages(profile, config: PipelineConfig) -> Stages:
    from engine.track.bytetrack import PlayerTracker

    return Stages(
        court_model=build_court_model(profile, device=config.device),
        template=load_template(profile.court_template),
        tracker=PlayerTracker(frame_rate=config.target_fps),
        player_detector=build_detector(profile, device=config.device),
        ball_detector=build_ball_detector(profile, device=config.device),
        player_classes=profile.player_classes,
        ball_classes=profile.ball_classes,
    )


def _clear_run(run_dir: Path) -> None:
    records = run_dir / RECORDS_SUBDIR
    if records.exists():
        shutil.rmtree(records)
    manifest = run_dir / MANIFEST_NAME
    if manifest.exists():
        manifest.unlink()


def _batched(iterable, batch: int):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= batch:
            yield buf
            buf = []
    if buf:
        yield buf


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #

def run_pipeline(
    video: Path | str,
    profile_name: str = "nba",
    config: PipelineConfig | None = None,
    out_dir: Path | str = DEFAULT_OUT,
    *,
    stages: Stages | None = None,
) -> dict:
    """Process ``video`` into sharded per-frame records; return the run manifest."""
    config = config or PipelineConfig()
    if config.fmt != "jsonl":
        raise NotImplementedError(f"format {config.fmt!r} not supported yet (jsonl only)")
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    info = probe(video)
    stride = max(1, round(info.fps / config.target_fps)) if config.target_fps > 0 else 1

    profile = None
    if stages is None:
        profile = get_profile(profile_name)
        stages = _build_stages(profile, config)

    run_dir = run_dir_for(out_dir, video.stem, profile_name)
    config_dict = _config_dict(config, stride, info, profile)
    key = resume_key(video, config_dict)

    existing = load_run_manifest(run_dir) if config.resume else None
    if existing is not None and existing.get("resume_key") != key:
        existing = None  # config / model / video changed → fresh run

    if existing is None:
        _clear_run(run_dir)
        manifest = new_run_manifest(video, profile_name, _info_dict(info), config_dict, key)
        rp = resume_point(None)
    else:
        manifest = existing
        rp = resume_point(manifest)

    sink = RecordSink(run_dir / RECORDS_SUBDIR, start_segment=rp.start_segment)
    scene = SceneCutDetector(config.scene_threshold)
    # The tracker is reset on the first processed frame (and every scene cut),
    # so no separate pre-loop reset is needed.

    current_shot = rp.start_shot
    first_frame = True
    consecutive_errors = 0
    seg = _new_seg(sink.segment_index)
    stats = {"processed": 0, "with_homography": 0, "errors": 0, "shots": set(), "tracks": set()}
    aborted = False

    frames_iter = iter_frames(
        video, stride=stride, max_frames=config.max_frames, start_index=rp.start_index
    )
    try:
        for batch in _batched(frames_iter, config.batch):
            frames = [f for (_, _, f) in batch]
            player_batch = _safe_batch_detect(stages.player_detector, frames)
            ball_batch = _safe_batch_detect(stages.ball_detector, frames)
            for i, (frame_index, ts, frame) in enumerate(batch):
                cut = scene.is_cut(frame)
                if first_frame:
                    scene_cut, first_frame = True, False
                    if stages.tracker is not None:
                        stages.tracker.reset()
                elif cut:
                    current_shot += 1
                    scene_cut = True
                    if stages.tracker is not None:
                        stages.tracker.reset()
                else:
                    scene_cut = False

                pd = player_batch[i] if player_batch is not None else None
                bd = ball_batch[i] if ball_batch is not None else None
                try:
                    record = _process_frame(
                        frame_index, ts, current_shot, scene_cut, frame, pd, bd, stages, config
                    )
                    consecutive_errors = 0
                except Exception as exc:  # noqa: BLE001 - isolate one frame's failure
                    record = FrameRecord(
                        frame_index=frame_index,
                        timestamp_s=ts,
                        shot_id=current_shot,
                        scene_cut=scene_cut,
                        court=court_dict(0, None, []),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    consecutive_errors += 1

                sink.write(record)
                _account(seg, stats, record, frame_index)

                if consecutive_errors >= config.max_consecutive_errors:
                    aborted = True
                    break
                if sink.n_in_segment >= config.segment_frames:
                    _finalize_segment(manifest, seg, current_shot, stats, run_dir)
                    sink.roll()
                    seg = _new_seg(sink.segment_index)
            if aborted:
                break
    finally:
        if seg["n_records"] > 0:
            _finalize_segment(manifest, seg, current_shot, stats, run_dir)
        sink.close()

    manifest["status"] = "aborted" if aborted else "done"
    manifest["summary"] = _summary(stats)
    save_run_manifest(run_dir, manifest)
    return manifest
