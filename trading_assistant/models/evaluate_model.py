"""
Model evaluation and comparison utilities.

Generates confusion matrices, per-class metric charts, walk-forward
accuracy plots, and a head-to-head XGBoost vs LSTM comparison table.
All outputs saved as JSON + plot images for the Streamlit dashboard.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from trading_assistant.config import PATHS

logger = logging.getLogger(__name__)
RESULTS_DIR = PATHS.MODEL_DIR / "eval_results"


# ── Confusion Matrix ────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """Generate an annotated confusion matrix heatmap.

    Args:
        y_true: Ground truth labels (integer-encoded).
        y_pred: Predicted labels.
        labels: Class names, e.g. ['DOWN', 'SIDEWAYS', 'UP'].
        title:  Plot title.
        save_path: If provided, saves the figure as HTML.

    Returns:
        plotly Figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Normalize to percentages
    cm_pct = (cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100).round(1)

    text = [[f"{cm[i][j]}\n({cm_pct[i][j]}%)"
             for j in range(len(labels))]
            for i in range(len(labels))]

    fig = ff.create_annotated_heatmap(
        z=cm, x=labels, y=labels,
        annotation_text=text, colorscale="Blues",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted", yaxis_title="Actual",
        width=500, height=450,
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        logger.info("Confusion matrix saved to %s", save_path)

    return fig


# ── Per-Class Metrics Chart ─────────────────────────────────────

def plot_classification_metrics(
    report: Dict,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """Create a grouped bar chart of precision / recall / F1 per class.

    Args:
        report: Output of sklearn classification_report(output_dict=True).
        model_name: Label for the chart title.
        save_path: Optional HTML save path.

    Returns:
        plotly Figure.
    """
    classes = [k for k in report if k not in
               ("accuracy", "macro avg", "weighted avg")]

    metrics_df = pd.DataFrame({
        "Class": classes * 3,
        "Score": ([report[c]["precision"] for c in classes] +
                  [report[c]["recall"] for c in classes] +
                  [report[c]["f1-score"] for c in classes]),
        "Metric": (["Precision"] * len(classes) +
                   ["Recall"] * len(classes) +
                   ["F1-Score"] * len(classes)),
    })

    fig = px.bar(
        metrics_df, x="Class", y="Score", color="Metric",
        barmode="group", title=f"{model_name} — Per-Class Metrics",
        range_y=[0, 1],
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))

    return fig


# ── Walk-Forward Accuracy Plot ──────────────────────────────────

def plot_walkforward_accuracy(
    fold_metrics: List[Dict],
    model_name: str = "Model",
    save_path: Optional[Path] = None,
) -> go.Figure:
    """Line chart of accuracy across walk-forward CV folds.

    Args:
        fold_metrics: List of dicts, each with 'fold' and 'accuracy' keys.
        model_name: Label for chart title.

    Returns:
        plotly Figure.
    """
    folds = [m["fold"] for m in fold_metrics]
    accs = [m["accuracy"] for m in fold_metrics]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=folds, y=accs, mode="lines+markers",
        name=model_name, line=dict(width=2),
    ))
    fig.update_layout(
        title=f"{model_name} — Walk-Forward Accuracy",
        xaxis_title="Fold", yaxis_title="Accuracy",
        yaxis_range=[0, 1],
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))

    return fig


# ── Model Comparison ────────────────────────────────────────────

def compare_models(
    results: Dict,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build a comparison table between all trained models.

    Args:
        results: Dict from run_training_pipeline / run_lstm_pipeline,
                 keyed by model name, each having 'avg_accuracy' and
                 'fold_metrics'.

    Returns:
        DataFrame with columns [Model, Avg Accuracy, Best Fold Acc, Worst Fold Acc].
    """
    rows = []
    for name, data in results.items():
        if name in ("best_model",):
            continue
        if not isinstance(data, dict) or "avg_accuracy" not in data:
            continue

        fold_accs = [m["accuracy"] for m in data.get("fold_metrics", [])]
        rows.append({
            "Model": name,
            "Avg Accuracy": round(data["avg_accuracy"], 4),
            "Best Fold": round(max(fold_accs), 4) if fold_accs else 0.0,
            "Worst Fold": round(min(fold_accs), 4) if fold_accs else 0.0,
            "Folds": len(fold_accs),
        })

    df = pd.DataFrame(rows).sort_values("Avg Accuracy", ascending=False)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(save_path), index=False)
        logger.info("Comparison table saved to %s", save_path)

    return df


# ── Full Evaluation Report ──────────────────────────────────────

def generate_full_report(
    results: Dict,
    save_dir: Optional[Path] = None,
) -> Dict:
    """Generate all evaluation artefacts for the dashboard.

    Creates and saves:
      - Confusion matrices per model (last fold)
      - Per-class metric charts per model
      - Walk-forward accuracy plots
      - Model comparison table

    Args:
        results: Combined dict from all training pipelines.
        save_dir: Directory for saving artefacts (defaults to eval_results/).

    Returns:
        Dict of {artifact_name: plotly Figure or DataFrame}.
    """
    save_dir = save_dir or RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    artefacts = {}

    for name, data in results.items():
        if not isinstance(data, dict) or "fold_metrics" not in data:
            continue

        fold_metrics = data["fold_metrics"]
        if not fold_metrics:
            continue

        # Walk-forward accuracy plot
        fig_wf = plot_walkforward_accuracy(
            fold_metrics, model_name=name,
            save_path=save_dir / f"{name}_walkforward.html",
        )
        artefacts[f"{name}_walkforward"] = fig_wf

        # Last fold classification report → per-class chart
        last_report = fold_metrics[-1].get("report", {})
        if last_report:
            fig_cls = plot_classification_metrics(
                last_report, model_name=name,
                save_path=save_dir / f"{name}_metrics.html",
            )
            artefacts[f"{name}_metrics"] = fig_cls

    # Comparison table
    comp_df = compare_models(results, save_path=save_dir / "comparison.csv")
    artefacts["comparison"] = comp_df

    # Save summary JSON
    summary = {
        name: {"avg_accuracy": data.get("avg_accuracy", 0)}
        for name, data in results.items()
        if isinstance(data, dict) and "avg_accuracy" in data
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Full evaluation report saved to %s", save_dir)
    return artefacts
