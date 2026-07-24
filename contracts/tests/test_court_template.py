import unittest

from contracts.court_schema import canonical_ids
from contracts.court_template import (
    TemplateError,
    available_templates,
    load_template,
)


class CourtTemplateTests(unittest.TestCase):
    def test_both_templates_available(self):
        self.assertEqual(set(available_templates()) >= {"nba_94x50", "nfhs_84x50"}, True)

    def test_templates_match_canonical_ids_and_validate(self):
        ids = set(canonical_ids())
        for name in ("nba_94x50", "nfhs_84x50"):
            template = load_template(name)  # raises TemplateError if inconsistent
            self.assertEqual(set(template.court_xy_ft), ids, msg=name)

    def test_dimensions(self):
        self.assertEqual((load_template("nba_94x50").length, load_template("nba_94x50").width), (94.0, 50.0))
        self.assertEqual((load_template("nfhs_84x50").length, load_template("nfhs_84x50").width), (84.0, 50.0))

    def test_known_landmarks(self):
        nba = load_template("nba_94x50").court_xy_ft
        # NBA apex sits 29 ft from the baseline (5.25 ft basket + 23.75 ft arc).
        self.assertEqual(nba["north_three_point_apex"], (29.0, 25.0))
        # 16 ft lane -> edges at 17 and 33 across the 50 ft width.
        self.assertEqual(nba["north_lane_baseline_west"], (0.0, 17.0))
        self.assertEqual(nba["north_lane_baseline_east"], (0.0, 33.0))

    def test_missing_template_raises(self):
        with self.assertRaises(TemplateError):
            load_template("does_not_exist")


if __name__ == "__main__":
    unittest.main()
