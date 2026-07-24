"""Profiles — the registry that wires a court model into its seat.

A ``Profile`` bundles the models, keypoint adapter, and court template for one
kind of footage. Selecting a profile is the entire "plug it in" action:

* ``nba`` — the NBA detector + NBA court model, adapted into canonical order,
  on the NBA 94x50 template. In use today.
* ``hs``  — the HS pose model (identity adapter) on the NFHS 84x50 template.
  Points at the existing ``court_pose_v1`` checkpoint, so it runs today with the
  undertrained weights and simply improves as the real model finishes training.
  **No engine code changes when the finished HS model arrives — just retrain and
  the same profile picks up the new best.pt.**
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from engine.court.adapters import (
    IdentityAdapter,
    KeypointAdapter,
    NBA_COURT_KEYPOINT_MAP,
    NbaCourtAdapter,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


@dataclass(frozen=True)
class Profile:
    name: str
    court_weights: Path
    court_adapter_factory: Callable[[], KeypointAdapter]
    court_template: str
    detector_weights: Path | None = None
    ball_weights: Path | None = None
    # Class names to keep from each detector (these models also emit Hoop, Ref,
    # Scoreboard, Overlay, Clock). None keeps everything.
    player_classes: tuple[str, ...] | None = None
    ball_classes: tuple[str, ...] | None = None


def _nba() -> Profile:
    return Profile(
        name="nba",
        detector_weights=MODELS_DIR / "player_detector.pt",
        ball_weights=MODELS_DIR / "ball_detector_model.pt",
        court_weights=MODELS_DIR / "court_keypoint_detector.pt",
        court_adapter_factory=lambda: NbaCourtAdapter(NBA_COURT_KEYPOINT_MAP),
        court_template="nba_94x50",
        player_classes=("Player",),
        ball_classes=("Ball",),
    )


def _hs() -> Profile:
    return Profile(
        name="hs",
        detector_weights=None,  # no HS player/ball model yet — court model only
        court_weights=REPO_ROOT / "runs" / "pose" / "court_pose_v1" / "weights" / "best.pt",
        court_adapter_factory=IdentityAdapter,
        court_template="nfhs_84x50",
    )


_BUILDERS: dict[str, Callable[[], Profile]] = {"nba": _nba, "hs": _hs}


def profile_names() -> list[str]:
    return sorted(_BUILDERS)


def get_profile(name: str, **overrides) -> Profile:
    if name not in _BUILDERS:
        raise KeyError(f"unknown profile {name!r}; choose from {profile_names()}")
    profile = _BUILDERS[name]()
    overrides = {key: value for key, value in overrides.items() if value is not None}
    return replace(profile, **overrides) if overrides else profile


def build_court_model(profile: Profile, device: str | None = None):
    from engine.court.yolo_pose import YoloPoseCourtModel

    return YoloPoseCourtModel(profile.court_weights, profile.court_adapter_factory(), device=device)


def build_detector(profile: Profile, device: str | None = None):
    """Returns the player ``Detector`` or ``None`` when the profile has none."""
    if profile.detector_weights is None:
        return None
    from engine.detect.players import Detector

    return Detector(profile.detector_weights, device=device)


def build_ball_detector(profile: Profile, device: str | None = None):
    """Returns the (separate) ball ``Detector`` or ``None`` when the profile has none.

    The reference models keep player and ball detection in two checkpoints, so
    the ball runs as its own detector rather than a class of the player model.
    """
    if profile.ball_weights is None:
        return None
    from engine.detect.players import Detector

    return Detector(profile.ball_weights, device=device)
