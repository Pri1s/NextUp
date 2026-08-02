"""Frozen-prompt and isolated-run records for annotation work.

The prompts are model-neutral, but a reproducible annotation must still record
which exact OpenAI model the runtime selected. This module creates that boundary:
prompt content is hash-pinned in source control, while model IDs are required only
when a fresh run namespace is initialized.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .frames import MAX_BENCHMARK_FRAMES, MIN_BENCHMARK_FRAMES, read_export_index

PROMPT_MANIFEST_SCHEMA_VERSION = "benchmark-prompts-1.0.0"
RUN_SCHEMA_VERSION = "benchmark-annotation-run-1.0.0"
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_MANIFEST = PROMPT_DIR / "prompts.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prompt_manifest(path: Path | str = PROMPT_MANIFEST) -> dict:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    schema = payload.get("schema_version")
    if schema != PROMPT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported prompt manifest {schema!r}, expected "
            f"{PROMPT_MANIFEST_SCHEMA_VERSION!r}"
        )
    return payload


def verify_prompts(path: Path | str = PROMPT_MANIFEST) -> list[str]:
    """Verify that every frozen prompt still has its declared version and hash."""
    manifest_path = Path(path)
    payload = load_prompt_manifest(manifest_path)
    prompt_dir = manifest_path.parent
    problems: list[str] = []
    roles: set[str] = set()

    for entry in payload.get("prompts", ()):
        role = str(entry.get("role", ""))
        version = str(entry.get("version", ""))
        roles.add(role)
        prompt_path = prompt_dir / str(entry.get("path", ""))
        if not prompt_path.is_file():
            problems.append(f"{role or '<missing role>'}: prompt file is missing")
            continue
        actual = _sha256(prompt_path)
        expected = str(entry.get("sha256", ""))
        if actual != expected:
            problems.append(
                f"{role}: prompt hash changed ({actual[:12]} != {expected[:12]}); "
                "review and version the prompt before annotation"
            )
        text = prompt_path.read_text(encoding="utf-8")
        if version not in text:
            problems.append(f"{role}: declared version {version!r} is absent from the prompt")

    required = {"annotator", "adjudicator"}
    if roles != required:
        problems.append(
            f"prompt roles are {sorted(roles)}, expected exactly {sorted(required)}"
        )
    return problems


def initialize_run(
    run_dir: Path | str,
    *,
    run_id: str,
    frames_dir: Path | str,
    pass_a_model: str,
    pass_b_model: str,
    adjudicator_model: str,
    reasoning_effort: str | None = None,
    image_detail: str | None = None,
    prompt_manifest: Path | str = PROMPT_MANIFEST,
) -> Path:
    """Create an empty, isolated run namespace without invoking any model."""
    identifiers = {
        "run_id": run_id,
        "pass_a_model": pass_a_model,
        "pass_b_model": pass_b_model,
        "adjudicator_model": adjudicator_model,
    }
    missing = [name for name, value in identifiers.items() if not value.strip()]
    if missing:
        raise ValueError(f"runtime identifiers must be explicit: {', '.join(missing)}")
    allowed_runtime_settings = {
        "reasoning_effort": (reasoning_effort, {"low", "medium", "high"}),
        "image_detail": (image_detail, {"low", "high", "original"}),
    }
    invalid = [
        f"{name}={value!r}"
        for name, (value, allowed) in allowed_runtime_settings.items()
        if value is not None and value not in allowed
    ]
    if invalid:
        raise ValueError(f"unsupported runtime settings: {', '.join(invalid)}")

    prompt_problems = verify_prompts(prompt_manifest)
    if prompt_problems:
        raise ValueError("prompts are not frozen:\n  " + "\n  ".join(prompt_problems))

    frames_dir = Path(frames_dir)
    frames = read_export_index(frames_dir)
    if not MIN_BENCHMARK_FRAMES <= len(frames) <= MAX_BENCHMARK_FRAMES:
        raise ValueError(
            f"run has {len(frames)} frames; expected {MIN_BENCHMARK_FRAMES} to "
            f"{MAX_BENCHMARK_FRAMES}"
        )
    with open(frames_dir / "frames.json", encoding="utf-8") as handle:
        export = json.load(handle)

    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"{run_dir} is not empty; start with a fresh run ID so old labels cannot mix"
        )

    prompts = load_prompt_manifest(prompt_manifest)
    prompt_by_role = {entry["role"]: entry for entry in prompts["prompts"]}
    payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "runtime_settings": {
            name: value
            for name, (value, _) in allowed_runtime_settings.items()
            if value is not None
        },
        "manifest_id": export["manifest_id"],
        "frame_count": len(frames),
        "frames": [
            {"frame_id": frame.frame_id, "sha256": frame.sha256} for frame in frames
        ],
        "prompts": {
            role: {
                "version": entry["version"],
                "path": str((Path(prompt_manifest).parent / entry["path"]).resolve()),
                "sha256": entry["sha256"],
            }
            for role, entry in prompt_by_role.items()
        },
        "passes": [
            {
                "pass_id": "pass_a",
                "model_id": pass_a_model,
                "output_dir": "pass_a",
                "isolation": "fresh_session_no_prior_annotations",
            },
            {
                "pass_id": "pass_b",
                "model_id": pass_b_model,
                "output_dir": "pass_b",
                "isolation": "fresh_session_no_prior_annotations",
            },
        ],
        "adjudication": {
            "model_id": adjudicator_model,
            "output_dir": "adjudication",
            "start_condition": "nonempty_consensus/adjudication_queue.json",
            "isolation": "fresh_session_no_annotation_pass_history",
        },
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("pass_a", "pass_b", "adjudication", "consensus"):
        (run_dir / child).mkdir()
    destination = run_dir / "run.json"
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return destination
