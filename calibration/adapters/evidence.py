"""Turn model slots into pooled landmark evidence.

The detector emits redundant slots. Phase-0 inspection found six pairs whose
members predict the same physical point a few pixels apart and split their
confidence between themselves, summing to about 1.0. Picking the more confident
member of a pair throws away half the signal and — worse — makes an even split
look like two weak observations rather than one strong one, so a landmark the
model is *certain* about can fall below a fixed gate.

This layer therefore pools. A landmark's confidence is the summed confidence of
every slot that voted for it, and its position is the confidence-weighted mean of
those votes. Which slot "won" is recorded but never acted on.

Slots the review could not settle are simply absent from the map. That is the
point: an ambiguous slot with no landmark id contributes nothing, rather than
contributing something plausible and wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from contracts.calibration_types import LandmarkEvidence, RawKeypointFrame

MAP_SCHEMA_VERSION = "evidence-map-1.0.0"
MAP_DIR = Path(__file__).resolve().parent / "maps"

#: Tiers a map entry may declare, most trusted first.
TIER_ORDER = ("confident", "probable", "speculative")


@dataclass(frozen=True, slots=True)
class EvidenceGroup:
    """The slots that vote for one landmark."""

    landmark_id: str
    slots: tuple[int, ...]
    tier: str
    evidence: str = ""

    def __post_init__(self) -> None:
        if not self.slots:
            raise ValueError(f"{self.landmark_id} has no slots")
        if self.tier not in TIER_ORDER:
            raise ValueError(f"unknown tier {self.tier!r} for {self.landmark_id}")


@dataclass(frozen=True, slots=True)
class EvidenceMap:
    """A provisional slot-to-landmark evidence map.

    Loading refuses any map that claims to be verified, because no such map has
    been earned yet and the failure mode of using one is silent.
    """

    map_id: str
    status: str
    keypoint_count: int
    weights_sha256: str
    groups: tuple[EvidenceGroup, ...]
    rejected: tuple[dict, ...] = ()

    @property
    def mapped_slots(self) -> set[int]:
        return {slot for group in self.groups for slot in group.slots}

    def groups_for_tiers(self, tiers: tuple[str, ...]) -> tuple[EvidenceGroup, ...]:
        return tuple(g for g in self.groups if g.tier in tiers)

    def rejection_reason(self, slot: int) -> str | None:
        for entry in self.rejected:
            if slot in entry.get("slots", ()):
                return entry.get("reason", "rejected")
        return None


def load_evidence_map(path: Path | str) -> EvidenceMap:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    schema = data.get("schema_version")
    if schema != MAP_SCHEMA_VERSION:
        raise ValueError(f"unsupported evidence-map schema {schema!r}")

    # A map asserting verified status would mean someone cleared the Phase-0 gate
    # without the sign-off this codebase requires. Refuse rather than trust it.
    if data.get("is_verified_map"):
        raise ValueError(
            f"{data.get('map_id')!r} claims to be a verified map. Verified slot "
            "semantics require human sign-off recorded in a slot_review, and this "
            "loader deliberately will not consume such a claim."
        )
    if data.get("status") != "provisional":
        raise ValueError(f"evidence maps must declare status 'provisional', got {data.get('status')!r}")

    groups = tuple(
        EvidenceGroup(
            landmark_id=entry["landmark_id"],
            slots=tuple(int(s) for s in entry["slots"]),
            tier=entry.get("tier", "confident"),
            evidence=entry.get("evidence", ""),
        )
        for entry in data.get("groups", ())
    )

    seen: dict[int, str] = {}
    for group in groups:
        for slot in group.slots:
            if slot in seen:
                raise ValueError(
                    f"slot {slot} appears in both {seen[slot]!r} and {group.landmark_id!r}"
                )
            seen[slot] = group.landmark_id

    return EvidenceMap(
        map_id=data["map_id"],
        status=data["status"],
        keypoint_count=int(data["keypoint_count"]),
        weights_sha256=data.get("weights_sha256", ""),
        groups=groups,
        rejected=tuple(data.get("rejected", ())),
    )


def load_registered_map(map_id: str) -> EvidenceMap:
    path = MAP_DIR / f"{map_id}.json"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in MAP_DIR.glob("*.json"))) or "none"
        raise FileNotFoundError(f"unknown evidence map {map_id!r}; available: {available}")
    return load_evidence_map(path)


def pool_evidence(
    detection: RawKeypointFrame,
    evidence_map: EvidenceMap,
    tiers: tuple[str, ...] = ("confident",),
    min_confidence: float = 0.25,
    max_pair_spread_px: float = 40.0,
) -> tuple[tuple[LandmarkEvidence, ...], tuple[tuple[str, str], ...]]:
    """Pool slot votes into per-landmark evidence.

    Returns the evidence that cleared ``min_confidence``, plus ``(identifier,
    reason)`` pairs for everything dropped.

    ``max_pair_spread_px`` guards the pooling assumption itself: members of a
    group are supposed to agree on position to within a few pixels, so if they
    disagree wildly the "same physical point" premise has broken down for this
    frame and averaging them would invent a point that neither slot predicted.
    """
    if not detection.detected:
        return (), (("frame", "no_detection"),)

    kept: list[LandmarkEvidence] = []
    dropped: list[tuple[str, str]] = []

    for group in evidence_map.groups:
        if group.tier not in tiers:
            dropped.append((group.landmark_id, f"tier_excluded:{group.tier}"))
            continue

        votes = [
            observation
            for slot in group.slots
            if (observation := detection.observation(slot)) is not None
        ]
        if not votes:
            dropped.append((group.landmark_id, "no_observation"))
            continue

        total_confidence = sum(v.confidence for v in votes)
        if total_confidence <= 0.0:
            dropped.append((group.landmark_id, "zero_confidence"))
            continue

        # Only points carrying real weight should steer the pooled position.
        contributing = [v for v in votes if v.confidence > 0.01 * total_confidence]
        if len(contributing) > 1:
            spread = max(
                ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
                for i, a in enumerate(contributing)
                for b in contributing[i + 1 :]
            )
            if spread > max_pair_spread_px:
                dropped.append((group.landmark_id, f"pair_disagreement:{spread:.0f}px"))
                continue

        pooled_confidence = min(total_confidence, 1.0)
        if pooled_confidence < min_confidence:
            dropped.append((group.landmark_id, f"below_confidence:{pooled_confidence:.3f}"))
            continue

        weight_sum = sum(v.confidence for v in contributing) or 1.0
        x = sum(v.x * v.confidence for v in contributing) / weight_sum
        y = sum(v.y * v.confidence for v in contributing) / weight_sum

        kept.append(
            LandmarkEvidence(
                landmark_id=group.landmark_id,
                x=x,
                y=y,
                confidence=pooled_confidence,
                source_slots=tuple(v.slot_index for v in contributing),
                tier=group.tier,
            )
        )

    for slot in sorted(set(range(evidence_map.keypoint_count)) - evidence_map.mapped_slots):
        reason = evidence_map.rejection_reason(slot) or "unmapped"
        dropped.append((f"slot_{slot}", reason))

    return tuple(kept), tuple(dropped)
