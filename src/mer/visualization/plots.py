"""Premium, reusable plotting helpers for emotion recognition analysis."""

from __future__ import annotations

import os
from typing import Any, Sequence


def set_premium_plot_style() -> None:
    """Configure matplotlib RC parameters for premium, publication-quality aesthetics."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            "figure.figsize": (7, 5),
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.8,
            "legend.fontsize": 9,
            "legend.title_fontsize": 10,
        })
    except ImportError:
        pass


def plot_confusion_matrices(
    y_true: Sequence[Any],
    y_pred_audio: Sequence[Any],
    y_pred_both: Sequence[Any],
    labels: Sequence[Any],
    title_audio: str = "Audio Only",
    title_both: str = "Audio + Transcript",
) -> Any:
    """Plot side-by-side row-normalized confusion matrix heatmaps using Seaborn."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("Required libraries (matplotlib, numpy, seaborn, sklearn) not found. Skipping plot.")
        return None

    set_premium_plot_style()
    cm_audio = confusion_matrix(y_true, y_pred_audio, labels=labels)
    cm_both = confusion_matrix(y_true, y_pred_both, labels=labels)

    # Prevent division by zero
    sum_audio = cm_audio.sum(axis=1, keepdims=True)
    sum_audio = np.where(sum_audio == 0, 1, sum_audio)
    cm_audio_norm = cm_audio.astype(float) / sum_audio

    sum_both = cm_both.sum(axis=1, keepdims=True)
    sum_both = np.where(sum_both == 0, 1, sum_both)
    cm_both_norm = cm_both.astype(float) / sum_both

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        cm_audio_norm,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=axes[0],
        cbar=True,
        linewidths=0.5,
    )
    axes[0].set_title(f"{title_audio}\n(Row-Normalized)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_both_norm,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Greens",
        ax=axes[1],
        cbar=True,
        linewidths=0.5,
    )
    axes[1].set_title(f"{title_both}\n(Row-Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    return fig


def plot_global_metric_comparison(
    y_true: Sequence[Any],
    y_pred_audio: Sequence[Any],
    y_pred_both: Sequence[Any],
    labels: Sequence[Any] | None = None,
    title: str = "Global Metrics: Audio vs Audio+Transcript",
) -> Any:
    """Plot a grouped bar chart of global scores comparing audio-only and audio+text modalities."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    except ImportError:
        print("Required libraries (matplotlib, pandas, seaborn, sklearn) not found. Skipping plot.")
        return None

    set_premium_plot_style()

    scores_audio = {
        "Accuracy": accuracy_score(y_true, y_pred_audio),
        "Macro F1": f1_score(y_true, y_pred_audio, labels=labels, average="macro", zero_division=0),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred_audio),
    }
    scores_both = {
        "Accuracy": accuracy_score(y_true, y_pred_both),
        "Macro F1": f1_score(y_true, y_pred_both, labels=labels, average="macro", zero_division=0),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred_both),
    }

    rows = []
    for metric in scores_audio:
        rows.append({"metric": metric, "modality": "audio_only", "score": scores_audio[metric]})
        rows.append({"metric": metric, "modality": "audio_plus_text", "score": scores_both[metric]})

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=df, x="metric", y="score", hue="modality", ax=ax, palette=["#3b82f6", "#10b981"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_per_class_metric_comparison(
    y_true: Sequence[Any],
    y_pred_audio: Sequence[Any],
    y_pred_both: Sequence[Any],
    labels: Sequence[Any],
    metric: str = "f1-score",
    title: str | None = None,
    ylabel: str | None = None,
) -> Any:
    """Plot a bar plot comparing per-class precision, recall, or F1 scores across modalities."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        from sklearn.metrics import classification_report
    except ImportError:
        print("Required libraries (matplotlib, pandas, seaborn, sklearn) not found. Skipping plot.")
        return None

    set_premium_plot_style()
    rep_audio = classification_report(y_true, y_pred_audio, labels=labels, output_dict=True, zero_division=0)
    rep_both = classification_report(y_true, y_pred_both, labels=labels, output_dict=True, zero_division=0)

    rows = []
    for label in labels:
        rows.append({
            "label": label,
            "modality": "audio_only",
            "score": rep_audio[str(label)][metric],
        })
        rows.append({
            "label": label,
            "modality": "audio_plus_text",
            "score": rep_both[str(label)][metric],
        })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=df, x="label", y="score", hue="modality", ax=ax, palette=["#3b82f6", "#10b981"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Emotion Label")
    ax.set_ylabel(ylabel or metric.replace("-", " ").title())
    ax.set_title(title or f"Per-Class {metric.replace('-', ' ').title()} Comparison")
    plt.tight_layout()
    return fig


def plot_correctness_overlap(
    y_true: Sequence[Any],
    y_pred_audio: Sequence[Any],
    y_pred_both: Sequence[Any],
    title: str = "Overlap of Correctness (Audio vs Audio+Text)",
) -> Any:
    """Plot a bar chart breaking down correctness agreement and disagreement regimes."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
    except ImportError:
        print("Required libraries (matplotlib, numpy, seaborn) not found. Skipping plot.")
        return None

    set_premium_plot_style()
    y_t = np.asarray(y_true)
    y_a = np.asarray(y_pred_audio)
    y_b = np.asarray(y_pred_both)

    correct_a = y_a == y_t
    correct_b = y_b == y_t

    both_correct = np.sum(correct_a & correct_b)
    audio_only_correct = np.sum(correct_a & ~correct_b)
    text_only_correct = np.sum(~correct_a & correct_b)
    both_wrong = np.sum(~correct_a & ~correct_b)

    categories = ["Both Correct", "Audio Only Correct", "Audio+Text Only", "Both Wrong"]
    counts = [int(both_correct), int(audio_only_correct), int(text_only_correct), int(both_wrong)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=categories, y=counts, ax=ax, palette=["#10b981", "#3b82f6", "#f59e0b", "#ef4444"])
    ax.set_ylabel("Number of Samples")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_gender_comparison(
    df: Any,
    gender_col: str,
    label_col: str,
    true_col: str,
    pred_audio_col: str,
    pred_both_col: str,
    labels: Sequence[Any],
) -> Any:
    """Plot per-class F1 performance sliced by gender and modality."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        from sklearn.metrics import classification_report
    except ImportError:
        print("Required libraries (matplotlib, pandas, seaborn, sklearn) not found. Skipping plot.")
        return None

    set_premium_plot_style()

    rows = []
    for g in sorted(df[gender_col].unique()):
        mask = df[gender_col] == g
        sub = df[mask]
        y_true_g = sub[true_col].tolist()
        y_a_g = sub[pred_audio_col].tolist()
        y_b_g = sub[pred_both_col].tolist()

        rep_a_g = classification_report(y_true_g, y_a_g, labels=labels, output_dict=True, zero_division=0)
        rep_b_g = classification_report(y_true_g, y_b_g, labels=labels, output_dict=True, zero_division=0)

        for label in labels:
            rows.append({
                "gender": g,
                "label": label,
                "modality": "audio_only",
                "F1": rep_a_g[str(label)]["f1-score"],
            })
            rows.append({
                "gender": g,
                "label": label,
                "modality": "audio_plus_text",
                "F1": rep_b_g[str(label)]["f1-score"],
            })

    df_gender_f1 = pd.DataFrame(rows)

    g_grid = sns.FacetGrid(df_gender_f1, col="gender", height=4.5, aspect=1.1, sharey=True)
    g_grid.map_dataframe(sns.barplot, x="label", y="F1", hue="modality", palette=["#3b82f6", "#10b981"])
    g_grid.add_legend(title="Modality")
    g_grid.set_titles(col_template="Gender = {col_name}")

    for ax in g_grid.axes.flat:
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Label")
        ax.set_ylabel("F1-Score")

    plt.tight_layout()
    return g_grid


def plot_dominant_confusions(
    y_true: Sequence[Any],
    y_pred_audio: Sequence[Any],
    y_pred_both: Sequence[Any],
    labels: Sequence[Any],
    title: str = "Most Frequent Confusions per Emotion",
) -> Any:
    """Plot the counts of the most frequent confusions for each emotion across modalities."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
    except ImportError:
        print("Required libraries (matplotlib, pandas, seaborn, sklearn) not found. Skipping plot.")
        return None

    set_premium_plot_style()
    cm_audio = confusion_matrix(y_true, y_pred_audio, labels=labels)
    cm_both = confusion_matrix(y_true, y_pred_both, labels=labels)

    def get_dominant_confusions_df(cm: Any, prefix: str) -> pd.DataFrame:
        rows = []
        for i, true_label in enumerate(labels):
            row = cm[i].copy()
            row[i] = 0  # Ignore correct prediction
            if row.sum() == 0:
                rows.append({"true": true_label, "confused_with": None, "count": 0})
            else:
                j = row.argmax()
                rows.append({
                    "true": true_label,
                    "confused_with": labels[j],
                    "count": int(row[j]),
                })
        df = pd.DataFrame(rows)
        df["modality"] = prefix
        return df

    df_audio = get_dominant_confusions_df(cm_audio, "audio_only")
    df_both = get_dominant_confusions_df(cm_both, "audio_plus_text")
    df_all = pd.concat([df_audio, df_both], ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=df_all, x="true", y="count", hue="modality", palette=["#3b82f6", "#10b981"], ax=ax)
    ax.set_ylabel("Misclassification Count")
    ax.set_xlabel("True Emotion")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_transcript_length_analysis(
    df: Any,
    transcript_col: str,
    true_col: str,
    pred_audio_col: str,
    pred_both_col: str,
    labels: Sequence[Any],
    title: str = "Delta Macro-F1 per Transcript-Length Bucket (Audio+Text - Audio)",
) -> Any:
    """Plot delta Macro-F1 (audio+text - audio) categorized by transcript length buckets."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        from sklearn.metrics import f1_score
    except ImportError:
        print("Required libraries (matplotlib, pandas, seaborn, sklearn) not found. Skipping plot.")
        return None

    set_premium_plot_style()
    df_tx = df.copy()
    df_tx["text_len"] = df_tx[transcript_col].fillna("").str.split().str.len()

    q1 = df_tx["text_len"].quantile(0.33)
    q2 = df_tx["text_len"].quantile(0.66)

    def txt_bucket(n: float) -> str:
        if n <= q1:
            return "Short Text"
        elif n <= q2:
            return "Medium Text"
        else:
            return "Long Text"

    df_tx["txt_bucket"] = df_tx["text_len"].apply(txt_bucket)

    rows = []
    for b in ["Short Text", "Medium Text", "Long Text"]:
        sub = df_tx[df_tx["txt_bucket"] == b]
        if len(sub) == 0:
            continue

        y_true_b = sub[true_col].values
        y_pa_b = sub[pred_audio_col].values
        y_pb_b = sub[pred_both_col].values

        f1_a = f1_score(y_true_b, y_pa_b, labels=labels, average="macro", zero_division=0)
        f1_b = f1_score(y_true_b, y_pb_b, labels=labels, average="macro", zero_division=0)

        rows.append({
            "bucket": b,
            "ΔF1": f1_b - f1_a,
            "f1_audio": f1_a,
            "f1_both": f1_b,
        })

    df_metrics = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=df_metrics, x="bucket", y="ΔF1", ax=ax, palette="coolwarm", hue="bucket", legend=False)
    ax.axhline(y=0, color="#ef4444", linestyle="--", linewidth=1.2)
    ax.set_ylabel("Delta Macro-F1")
    ax.set_xlabel("Transcript Length Bucket")
    ax.set_title(title)
    plt.tight_layout()
    return fig
