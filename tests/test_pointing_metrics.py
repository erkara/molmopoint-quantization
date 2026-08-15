import unittest

import numpy as np

from molmo_quant.pointing_metrics import absolute_gt_points, score_pointing_prediction


class PointingMetricsTests(unittest.TestCase):
    def test_one_to_one_mask_scoring(self):
        masks = np.zeros((2, 10, 10), dtype=bool)
        masks[0, 1:4, 1:4] = True
        masks[1, 6:9, 6:9] = True
        gt = np.asarray([[2, 2], [7, 7]], dtype=float)
        score = score_pointing_prediction([{"x": 7, "y": 7}, {"x": 2, "y": 2}], gt, masks)
        self.assertAlmostEqual(score["precision"], 1.0)
        self.assertAlmostEqual(score["recall"], 1.0)
        self.assertAlmostEqual(score["f1"], 1.0)

    def test_negative_example(self):
        masks = np.zeros((1, 5, 5), dtype=bool)
        self.assertEqual(
            score_pointing_prediction([], np.empty((0, 2)), masks),
            {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        )
        self.assertEqual(
            score_pointing_prediction([{"x": 2, "y": 2}], np.empty((0, 2)), masks)["f1"],
            0.0,
        )

    def test_normalized_ground_truth_conversion(self):
        result = absolute_gt_points([{"x": 25, "y": 50}], width=200, height=80)
        np.testing.assert_allclose(result, [[50, 40]])


if __name__ == "__main__":
    unittest.main()
