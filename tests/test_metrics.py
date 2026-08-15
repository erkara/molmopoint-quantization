import unittest

from molmo_quant.metrics import compare_to_baseline, matched_point_distances


class MetricsTests(unittest.TestCase):
    def test_exact_agreement(self):
        baseline = {
            "variant": "bf16",
            "generated_token_ids": [1, 2, 3],
            "points": [{"x": 10.0, "y": 20.0}],
            "parse_success": True,
        }
        candidate = {
            "variant": "int4",
            "generated_token_ids": [1, 2, 3],
            "points": [{"x": 10.0, "y": 20.0}],
            "parse_success": True,
        }
        result = compare_to_baseline(baseline, candidate)
        self.assertTrue(result["exact_token_agreement"])
        self.assertTrue(result["point_count_agreement"])
        self.assertEqual(result["max_matched_point_distance_px"], 0.0)

    def test_matching_is_order_independent(self):
        baseline = [{"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 10.0}]
        candidate = [{"x": 10.0, "y": 10.0}, {"x": 3.0, "y": 4.0}]
        self.assertEqual(matched_point_distances(baseline, candidate), [5.0, 0.0])


if __name__ == "__main__":
    unittest.main()
