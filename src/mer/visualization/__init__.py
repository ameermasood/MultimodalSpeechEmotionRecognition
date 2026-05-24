"""Reusable visualization helpers for notebooks and reports."""

from mer.visualization.plots import (
    plot_confusion_matrices,
    plot_correctness_overlap,
    plot_dominant_confusions,
    plot_gender_comparison,
    plot_global_metric_comparison,
    plot_per_class_metric_comparison,
    plot_transcript_length_analysis,
    set_premium_plot_style,
)

__all__ = [
    "plot_confusion_matrices",
    "plot_correctness_overlap",
    "plot_dominant_confusions",
    "plot_gender_comparison",
    "plot_global_metric_comparison",
    "plot_per_class_metric_comparison",
    "plot_transcript_length_analysis",
    "set_premium_plot_style",
]
