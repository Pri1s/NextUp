"""The blind frame set.

An annotation is only independent evidence if the annotator never saw what the
model predicted. That sounds like a rule about prompts, but the durable version is
a rule about *files*: the tree an annotator can read contains raw frames, a
glossary and nothing else. There is no projected overlay to be influenced by,
because none was written there.

This matters more than it first appears. The engine's reprojected images in
``engine_out/calibration/*/interim/reprojected/`` are the natural thing to reach
for -- they are already on disk, already one per frame, already named by index. They
are also the exact contamination §7.2 warns about: an annotator shown the current
projection would place the free-throw circle where the projection puts it, and the
benchmark would certify the drift it exists to detect. ``assert_blind`` refuses
that mistake rather than trusting nobody makes it.

Frames are written as PNG. JPEG re-encoding would move stripe edges by a fraction
of a pixel, which is small compared to most things and not small compared to a
2 px annotation-agreement target.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from calibration.io.video import VideoFrameSource

MANIFEST_SCHEMA_VERSION = "benchmark-manifest-1.0.0"
EXPORT_SCHEMA_VERSION = "benchmark-export-1.0.0"

#: Directories whose contents are model-derived. Nothing from these may reach the
#: annotator's tree, and ``assert_blind`` checks the exported tree for traces.
FORBIDDEN_SOURCES = ("engine_out", "models", "runs", "dataset")

#: Filename fragments that betray a leaked overlay or prediction file.
FORBIDDEN_NAME_FRAGMENTS = (
    "reproject",
    "overlay",
    "prediction",
    "keypoint",
    "calibration",
    "slot",
    "detect",
)


@dataclass(frozen=True, slots=True)
class ManifestFrame:
    """One frame the benchmark has committed to annotating.

    ``stratum`` records *why* this frame is in the set -- clean, occluded, blurred,
    graphics-heavy, currently-failing. §8.1 asks for coverage of each; recording
    the intent makes a gap visible instead of leaving it to be noticed later.
    """

    frame_id: str
    clip: str
    frame_index: int
    stratum: str
    pilot: bool = False
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "clip": self.clip,
            "frame_index": self.frame_index,
            "stratum": self.stratum,
            "pilot": self.pilot,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ManifestFrame":
        return cls(
            frame_id=str(data["frame_id"]),
            clip=str(data["clip"]),
            frame_index=int(data["frame_index"]),
            stratum=str(data["stratum"]),
            pilot=bool(data.get("pilot", False)),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    """The locked frame set. Locked before the detector is tuned against it (§8.1)."""

    manifest_id: str
    layout_id: str
    description: str
    frames: tuple[ManifestFrame, ...] = ()

    @property
    def pilot_frames(self) -> tuple[ManifestFrame, ...]:
        return tuple(f for f in self.frames if f.pilot)

    def as_dict(self) -> dict:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "manifest_id": self.manifest_id,
            "layout_id": self.layout_id,
            "description": self.description,
            "frames": [f.as_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkManifest":
        schema = data.get("schema_version")
        if schema != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest schema {schema!r}, expected {MANIFEST_SCHEMA_VERSION!r}"
            )
        return cls(
            manifest_id=str(data["manifest_id"]),
            layout_id=str(data["layout_id"]),
            description=str(data.get("description", "")),
            frames=tuple(ManifestFrame.from_dict(f) for f in data["frames"]),
        )


MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"


def load_manifest(path_or_id: Path | str) -> BenchmarkManifest:
    path = Path(path_or_id)
    if not path.is_file():
        candidate = MANIFEST_DIR / f"{path_or_id}.json"
        if not candidate.is_file():
            available = ", ".join(sorted(p.stem for p in MANIFEST_DIR.glob("*.json"))) or "none"
            raise FileNotFoundError(f"unknown manifest {path_or_id!r}; available: {available}")
        path = candidate
    with open(path, encoding="utf-8") as handle:
        return BenchmarkManifest.from_dict(json.load(handle))


def frame_id_for(clip: Path | str, frame_index: int) -> str:
    """``video_2.mp4`` frame 119 -> ``video_2_000119``."""
    return f"{Path(clip).stem}_{frame_index:06d}"


def sha256_of(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ExportedFrame:
    """A frame on disk, pinned by content hash.

    The hash is what lets two annotation passes be *proven* to have seen the same
    bytes rather than assumed to. Cheap to compute, and the alternative is
    discovering after a re-export that half the labels refer to a different image.
    """

    frame_id: str
    clip: str
    frame_index: int
    stratum: str
    pilot: bool
    path: str
    sha256: str
    width: int
    height: int

    def as_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "clip": self.clip,
            "frame_index": self.frame_index,
            "stratum": self.stratum,
            "pilot": self.pilot,
            "path": self.path,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExportedFrame":
        return cls(
            frame_id=str(data["frame_id"]),
            clip=str(data["clip"]),
            frame_index=int(data["frame_index"]),
            stratum=str(data.get("stratum", "")),
            pilot=bool(data.get("pilot", False)),
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )


def export_frames(
    manifest: BenchmarkManifest,
    out_dir: Path | str,
    repo_dir: Path | str = ".",
    progress=None,
) -> list[ExportedFrame]:
    """Write every manifest frame as a lossless PNG, grouped by clip for one decode.

    Frames are grouped so each clip is opened once. ``VideoFrameSource`` seeks by
    index and falls back to sequential reads where the container's index lies, so
    repeated random access across a 240-frame clip is not free.
    """
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(repo_dir)

    by_clip: dict[str, list[ManifestFrame]] = {}
    for frame in manifest.frames:
        by_clip.setdefault(frame.clip, []).append(frame)

    exported: list[ExportedFrame] = []
    for clip, entries in sorted(by_clip.items()):
        clip_path = repo_dir / clip
        source = VideoFrameSource(clip_path)
        try:
            for entry in sorted(entries, key=lambda f: f.frame_index):
                image = source.frame_at(entry.frame_index)
                path = images_dir / f"{entry.frame_id}.png"
                if not cv2.imwrite(str(path), image):
                    raise OSError(f"failed to write {path}")
                exported.append(
                    ExportedFrame(
                        frame_id=entry.frame_id,
                        clip=clip,
                        frame_index=entry.frame_index,
                        stratum=entry.stratum,
                        pilot=entry.pilot,
                        path=str(path.relative_to(out_dir)),
                        sha256=sha256_of(path),
                        width=int(image.shape[1]),
                        height=int(image.shape[0]),
                    )
                )
                if progress:
                    progress(f"{entry.frame_id} ({entry.stratum})")
        finally:
            source.close()

    return sorted(exported, key=lambda f: f.frame_id)


def write_export_index(
    out_dir: Path | str,
    manifest: BenchmarkManifest,
    exported: list[ExportedFrame],
    glossary_markdown: str,
    glossary_version: str,
) -> Path:
    """Write the annotator-facing index plus the glossary beside the images."""
    out_dir = Path(out_dir)
    (out_dir / "GLOSSARY.md").write_text(glossary_markdown, encoding="utf-8")

    index = out_dir / "frames.json"
    payload = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "manifest_id": manifest.manifest_id,
        "layout_id": manifest.layout_id,
        "glossary_version": glossary_version,
        "frame_count": len(exported),
        "frames": [f.as_dict() for f in exported],
    }
    with open(index, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return index


def read_export_index(out_dir: Path | str) -> list[ExportedFrame]:
    with open(Path(out_dir) / "frames.json", encoding="utf-8") as handle:
        payload = json.load(handle)
    schema = payload.get("schema_version")
    if schema != EXPORT_SCHEMA_VERSION:
        raise ValueError(f"unsupported export schema {schema!r}, expected {EXPORT_SCHEMA_VERSION!r}")
    return [ExportedFrame.from_dict(f) for f in payload["frames"]]


def verify_export(out_dir: Path | str) -> list[str]:
    """Re-hash every exported image. Empty list means the tree is intact."""
    out_dir = Path(out_dir)
    problems: list[str] = []
    for frame in read_export_index(out_dir):
        path = out_dir / frame.path
        if not path.is_file():
            problems.append(f"{frame.frame_id}: {frame.path} is missing")
            continue
        actual = sha256_of(path)
        if actual != frame.sha256:
            problems.append(
                f"{frame.frame_id}: content changed since export "
                f"({actual[:12]} != {frame.sha256[:12]})"
            )
    return problems


ALLOWED_EXPORT_NAMES = frozenset({"frames.json", "GLOSSARY.md", "README.md"})


def assert_blind(out_dir: Path | str) -> list[str]:
    """Check that nothing model-derived reached the annotator's tree.

    Two ways contamination arrives: a file copied in from a model output directory,
    or a file whose name announces it is a projection. Both are caught by
    inspection rather than by trusting the export path, because the tree is also
    somewhere a person can drop a file by hand.
    """
    out_dir = Path(out_dir)
    problems: list[str] = []

    for path in sorted(out_dir.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(out_dir)
        name = path.name.lower()

        if relative.parts[0] == "images":
            if path.suffix.lower() != ".png":
                problems.append(f"{relative}: images/ must hold PNG frames only")
            continue

        if path.name in ALLOWED_EXPORT_NAMES:
            continue

        problems.append(f"{relative}: unexpected file in the blind frame tree")
        for fragment in FORBIDDEN_NAME_FRAGMENTS:
            if fragment in name:
                problems.append(
                    f"{relative}: name suggests model-derived content ({fragment!r})"
                )
                break

    index = out_dir / "frames.json"
    if index.is_file():
        text = index.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_SOURCES:
            if f'"{forbidden}/' in text or f"/{forbidden}/" in text:
                problems.append(f"frames.json references {forbidden}/, a model-output tree")

    return problems


#: The six-frame core deliberately covers these axes. The conditional expansion
#: adds pan/blur, a second graphics case, and an existing implausible result.
REQUIRED_STRATA = frozenset(
    {
        "clean",
        "occluded",
        "graphics_tight",
        "logo_midcourt",
    }
)

MIN_BENCHMARK_FRAMES = 6
MAX_BENCHMARK_FRAMES = 9
CORE_FRAME_COUNT = 6


def validate_manifest(manifest: BenchmarkManifest, repo_dir: Path | str = ".") -> list[str]:
    """Structural and coverage checks, before anything is exported or annotated."""
    problems: list[str] = []
    repo_dir = Path(repo_dir)

    seen: set[str] = set()
    for frame in manifest.frames:
        if frame.frame_id in seen:
            problems.append(f"{frame.frame_id}: listed more than once")
        seen.add(frame.frame_id)

        expected = frame_id_for(frame.clip, frame.frame_index)
        if frame.frame_id != expected:
            problems.append(f"{frame.frame_id}: id does not match clip/index (expected {expected})")

        if not (repo_dir / frame.clip).is_file():
            problems.append(f"{frame.frame_id}: clip {frame.clip} not found")

        if frame.frame_index < 0:
            problems.append(f"{frame.frame_id}: negative frame index")

        if not frame.notes.strip():
            problems.append(f"{frame.frame_id}: no note saying why it is in the set")

    if not manifest.frames:
        problems.append("manifest lists no frames")
        return problems

    if len(manifest.frames) < MIN_BENCHMARK_FRAMES:
        problems.append(
            f"benchmark has {len(manifest.frames)} frames; Milestone 1 requires the "
            f"{MIN_BENCHMARK_FRAMES}-frame core"
        )
    if len(manifest.frames) > MAX_BENCHMARK_FRAMES:
        problems.append(
            f"benchmark has {len(manifest.frames)} frames; Milestone 1 is capped at "
            f"{MAX_BENCHMARK_FRAMES}"
        )

    missing = REQUIRED_STRATA - {f.stratum for f in manifest.frames}
    if missing:
        problems.append(f"no frame covers stratum: {', '.join(sorted(missing))}")

    # A benchmark drawn from one clip measures one floor, one broadcast style and
    # one camera. Splitting by clip is what §8.1 asks for, so every clip has to be
    # represented -- and the pilot has to sample every clip too, or a clip-specific
    # annotation failure surfaces only after the full spend.
    by_clip: dict[str, list[ManifestFrame]] = {}
    for frame in manifest.frames:
        by_clip.setdefault(frame.clip, []).append(frame)
    if len(by_clip) != 3:
        problems.append(
            f"benchmark covers {len(by_clip)} clips; the locked design requires all three"
        )

    for clip, entries in sorted(by_clip.items()):
        pilot_count = sum(f.pilot for f in entries)
        if pilot_count != 2:
            problems.append(
                f"{clip}: expected the locked easy/hard pilot pair, found {pilot_count} frames"
            )

    pilots = manifest.pilot_frames
    if len(pilots) != CORE_FRAME_COUNT:
        problems.append(
            f"expected exactly {CORE_FRAME_COUNT} locked pilot frames, found {len(pilots)}"
        )

    return problems
