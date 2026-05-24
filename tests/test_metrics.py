import math
import unittest

from mer.evaluation import (
    classification_metrics,
    mcnemar_from_two_preds,
    reliability_ece,
    selective_accuracy_curve,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_reliability_ece_returns_expected_shapes(self):
        ece, bin_acc, bin_conf, bin_count = reliability_ece(
            confidence=[0.1, 0.8, 0.9],
            correct=[0, 1, 1],
            n_bins=2,
        )

        self.assertGreaterEqual(ece, 0.0)
        self.assertEqual(len(bin_acc), 2)
        self.assertEqual(len(bin_conf), 2)
        self.assertEqual(len(bin_count), 2)

    def test_reliability_ece_handles_empty_finite_confidence(self):
        ece, _, _, _ = reliability_ece([float("nan")], [1], n_bins=3)
        self.assertTrue(math.isnan(ece))

    def test_selective_accuracy_curve_orders_by_confidence(self):
        coverages, accuracies, risk = selective_accuracy_curve(
            confidence=[0.9, 0.8, 0.1],
            correct=[1, 1, 0],
            n_points=3,
        )

        self.assertEqual(len(coverages), 3)
        self.assertEqual(accuracies[0], 1.0)
        self.assertGreaterEqual(risk, 0.0)

    def test_mcnemar_counts_disagreements(self):
        result = mcnemar_from_two_preds(
            y_true=["A", "A", "B", "B"],
            y_pred_a=["A", "B", "B", "B"],
            y_pred_b=["B", "A", "B", "A"],
        )

        self.assertEqual(result["n01_a_wrong_b_right"], 1)
        self.assertEqual(result["n10_a_right_b_wrong"], 2)
        self.assertEqual(result["total_divergent"], 3)

    def test_classification_metrics_includes_core_metrics(self):
        result = classification_metrics(
            y_true=["Angry", "Happy", "Sad", "Neutral"],
            y_pred=["Angry", "Happy", "Neutral", "Neutral"],
            confidence=[0.9, 0.8, 0.6, 0.7],
            latencies_ms=[10, 20, 30, 40],
        )

        self.assertEqual(result["num_samples"], 4)
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertIn("balanced_accuracy", result)
        self.assertIn("f1_macro", result)
        self.assertIn("mcc", result)
        self.assertIn("kappa", result)
        self.assertIn("ece_10bins", result)
        self.assertEqual(result["latency_ms_p50"], 25.0)


if __name__ == "__main__":
    unittest.main()
