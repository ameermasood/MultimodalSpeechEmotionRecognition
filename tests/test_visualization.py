import unittest
from mer.visualization import (
    plot_confusion_matrices,
    plot_correctness_overlap,
    plot_dominant_confusions,
    plot_gender_comparison,
    plot_global_metric_comparison,
    plot_per_class_metric_comparison,
    plot_transcript_length_analysis,
    set_premium_plot_style,
)


class VisualizationHelperTests(unittest.TestCase):
    def test_set_premium_plot_style_no_crash(self):
        # Verify function runs without raising any exception, even if matplotlib is not installed.
        try:
            set_premium_plot_style()
        except Exception as e:
            self.fail(f"set_premium_plot_style raised an unexpected exception: {e}")

    def test_plotting_helpers_fallback_or_succeed(self):
        # We supply valid mock datasets to check that functions don't crash and either return a Figure/Grid or None (fallback)
        y_true = ["Angry", "Happy", "Sad", "Neutral"]
        y_pred_audio = ["Angry", "Neutral", "Sad", "Neutral"]
        y_pred_both = ["Angry", "Happy", "Sad", "Neutral"]
        labels = ["Angry", "Happy", "Sad", "Neutral"]

        # Call confusion matrix plot
        res = plot_confusion_matrices(y_true, y_pred_audio, y_pred_both, labels)
        # Should be None if matplotlib is missing, or a Figure object if it is installed
        if res is not None:
            import matplotlib.pyplot as plt
            self.assertIsInstance(res, plt.Figure)

        # Call global metric comparison plot
        res = plot_global_metric_comparison(y_true, y_pred_audio, y_pred_both, labels)
        if res is not None:
            import matplotlib.pyplot as plt
            self.assertIsInstance(res, plt.Figure)

        # Call per-class metric comparison plot
        res = plot_per_class_metric_comparison(y_true, y_pred_audio, y_pred_both, labels)
        if res is not None:
            import matplotlib.pyplot as plt
            self.assertIsInstance(res, plt.Figure)

        # Call correctness overlap plot
        res = plot_correctness_overlap(y_true, y_pred_audio, y_pred_both)
        if res is not None:
            import matplotlib.pyplot as plt
            self.assertIsInstance(res, plt.Figure)

        # Call dominant confusion plot
        res = plot_dominant_confusions(y_true, y_pred_audio, y_pred_both, labels)
        if res is not None:
            import matplotlib.pyplot as plt
            self.assertIsInstance(res, plt.Figure)


if __name__ == "__main__":
    unittest.main()
