import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from review_homography_labels import build_app

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "homography_results"
SUMMARY_PATH = RESULTS_DIR / "batch_summary.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(SUMMARY_PATH.is_file(), "homography_results/batch_summary.json is unavailable")
class ReviewUiTests(unittest.TestCase):
    def setUp(self):
        self.reviews_dir = Path(tempfile.mkdtemp()) / "reviews"
        self.addCleanup(shutil.rmtree, self.reviews_dir.parent, ignore_errors=True)
        app = build_app(RESULTS_DIR, self.reviews_dir)
        app.testing = True
        self.client = app.test_client()

        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.record = next(
            record for record in summary["frames"] if record.get("review_candidate_available")
        )

    def test_batch_endpoint_flags_review_candidate_availability(self):
        response = self.client.get("/api/batch")
        self.assertEqual(response.status_code, 200)
        record = next(
            item for item in response.get_json()["frames"] if item["result"] == self.record["result"]
        )
        self.assertTrue(record["review_candidate_available"])

    def test_result_endpoint_exposes_review_candidate(self):
        response = self.client.get(f"/api/result?path={self.record['result']}")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["solution"]["status"], "rejected")
        candidate = payload["solution"]["review_candidate"]
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["status"], "review_required")
        self.assertGreaterEqual(candidate["matched_keypoints"], 4)

    def test_approving_a_candidate_writes_a_separate_artifact_without_touching_pipeline_output(self):
        result_path = RESULTS_DIR / self.record["result"]
        before_result_hash = sha256(result_path)
        before_summary_hash = sha256(SUMMARY_PATH)

        result_payload = self.client.get(f"/api/result?path={self.record['result']}").get_json()
        candidate = result_payload["solution"]["review_candidate"]
        decision_payload = {
            "points": {
                "north_lane_west": {"status": "unchanged"},
                "north_baseline": {"status": "moved", "pixel_xy": [12.0, 34.0]},
            },
            "source_result_path": self.record["result"],
            "automatic_status": result_payload["solution"]["status"],
            "automatic_reason": result_payload["solution"]["reason"],
            "review_candidate": {
                "proposal_type": candidate["proposal"]["type"],
                "proposal_source": candidate["proposal"]["source"],
                "homography": candidate["homography"],
            },
            "decision": "approved",
            "decided_at": "2026-07-20T00:00:00Z",
        }
        response = self.client.post(
            f"/api/review?path={self.record['result']}",
            data=json.dumps(decision_payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

        # A separate review artifact was created...
        review_file = self.reviews_dir / Path(self.record["result"]).with_suffix(".review.json")
        self.assertTrue(review_file.is_file())
        saved = json.loads(review_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["decision"], "approved")
        self.assertEqual(saved["automatic_status"], "rejected")
        self.assertEqual(saved["points"]["north_baseline"]["status"], "moved")
        self.assertEqual(saved["points"]["north_lane_west"]["status"], "unchanged")

        # ...and the generated pipeline result / batch summary were untouched.
        self.assertEqual(sha256(result_path), before_result_hash)
        self.assertEqual(sha256(SUMMARY_PATH), before_summary_hash)

    def test_get_review_returns_empty_points_when_no_override_saved(self):
        response = self.client.get(f"/api/review?path={self.record['result']}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"points": {}})


if __name__ == "__main__":
    unittest.main()
