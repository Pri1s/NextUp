import copy
import json
import unittest
from pathlib import Path

from validate_schema import SchemaError, flip_idx_from_schema, load_schema, validate

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_DIR / "dataset" / "schemas" / "court_keypoints.v3.json"


def load() -> dict:
    return load_schema(SCHEMA_PATH)


class SchemaV3Tests(unittest.TestCase):
    def test_schema_and_svg_validate(self):
        schema = load()
        svg_path = PROJECT_DIR / schema["reference_diagram"]["asset"]
        flip_idx = validate(schema, svg_path)
        self.assertEqual(len(flip_idx), len(schema["keypoints"]))

    def test_flip_idx_pinned(self):
        # Regression pin: reordering or re-pairing keypoints must be a loud,
        # reviewed change — a silently different flip_idx corrupts training.
        self.assertEqual(
            flip_idx_from_schema(load()),
            [18, 17, 16, 15, 14, 13, 20, 19, 21, 9, 10, 11, 12, 5, 4, 3, 2, 1, 0, 7, 6, 8],
        )

    def test_validator_rejects_double_pair_membership(self):
        schema = copy.deepcopy(load())
        schema["mirror_pairs"]["north_south"][1][1] = "south_baseline_sideline_east"
        with self.assertRaises(SchemaError):
            validate(schema)

    def test_validator_rejects_diagram_drift(self):
        schema = copy.deepcopy(load())
        schema["keypoints"][0]["diagram_xy"][0] += 3
        with self.assertRaises(SchemaError):
            validate(schema)


class HorizontalFlipSemanticsTests(unittest.TestCase):
    """Prove flip_idx against the labeling convention, not against itself.

    A synthetic side camera renders the court with north at image-left and the
    west (near) sideline at image-bottom — exactly the canonical assignment of
    the image_left_basket convention. The image is then mirrored horizontally
    and labeled afresh by the same convention. YOLO's fliplr produces
    flipped[i] = mirror(original[flip_idx[i]]); that must equal the fresh
    convention labeling of the mirrored image for every keypoint.
    """

    IMAGE_W = 1920.0
    LEFT, SCALE_ALONG = 300.0, 16.0   # x = LEFT + 16 * feet-from-north-baseline
    BOTTOM, SCALE_ACROSS = 1000.0, 14.0  # y = BOTTOM - 14 * feet-from-west-sideline

    def project(self, along: float, across: float) -> tuple[float, float]:
        return (self.LEFT + self.SCALE_ALONG * along, self.BOTTOM - self.SCALE_ACROSS * across)

    def convention_labels(self, schema: dict, court_left_x: float) -> list[tuple[float, float]]:
        """Label a court whose north baseline sits at image x == court_left_x."""
        labels = []
        for point in schema["keypoints"]:
            along, across = point["court_xy_ft"]
            labels.append((court_left_x + self.SCALE_ALONG * along, self.BOTTOM - self.SCALE_ACROSS * across))
        return labels

    def test_horizontal_flip_matches_convention(self):
        schema = load()
        flip_idx = flip_idx_from_schema(schema)
        length = schema["reference_court"]["length"]

        original = self.convention_labels(schema, court_left_x=self.LEFT)
        # Mirrored image: the court now spans [W - LEFT - length*s, W - LEFT];
        # its image-left end is the physical south end, relabeled north.
        flipped_truth = self.convention_labels(
            schema, court_left_x=self.IMAGE_W - self.LEFT - self.SCALE_ALONG * length
        )

        for i, truth in enumerate(flipped_truth):
            source_x, source_y = original[flip_idx[i]]
            augmented = (self.IMAGE_W - source_x, source_y)
            self.assertAlmostEqual(augmented[0], truth[0], places=6, msg=schema["keypoints"][i]["id"])
            self.assertAlmostEqual(augmented[1], truth[1], places=6, msg=schema["keypoints"][i]["id"])

    def test_east_west_pairs_fail_fliplr(self):
        # The v2 export used east/west pairs for fliplr; show that mapping
        # breaks the convention (it is the vertical-flip pairing).
        schema = load()
        index_of = {point["id"]: point["index"] - 1 for point in schema["keypoints"]}
        east_west_flip = list(range(len(schema["keypoints"])))
        for id_a, id_b in schema["mirror_pairs"]["east_west"]:
            a, b = index_of[id_a], index_of[id_b]
            east_west_flip[a], east_west_flip[b] = b, a

        length = schema["reference_court"]["length"]
        original = self.convention_labels(schema, court_left_x=self.LEFT)
        flipped_truth = self.convention_labels(
            schema, court_left_x=self.IMAGE_W - self.LEFT - self.SCALE_ALONG * length
        )
        mismatches = sum(
            1
            for i, truth in enumerate(flipped_truth)
            if abs(self.IMAGE_W - original[east_west_flip[i]][0] - truth[0]) > 1e-6
            or abs(original[east_west_flip[i]][1] - truth[1]) > 1e-6
        )
        self.assertGreater(mismatches, 0)


if __name__ == "__main__":
    unittest.main()
