"""
Train XGBoost and Random Forest classifiers for intraday direction prediction.

Target: 15-min-ahead price direction → UP (>+0.3%), DOWN (<-0.3%), SIDEWAYS.
Uses walk-forward (expanding window) validation — never random splits on
time-series data.

Features: technical indicators + sentiment_score + time-of-day encodings.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from trading_assistant.config import MODEL, PATHS, TRADING

logger = logging.getLogger(__name__)

# Label mapping
LABELS = {0: "DOWN", 1: "SIDEWAYS", 2: "UP"}


# ── Feature Engineering ─────────────────────────────────────────

def create_target(df: pd.DataFrame) -> pd.Series:
    """Create the target variable: price direction 15 minutes ahead.

    Computes the percentage change of 'close' one row forward (next
    15-min bar), then bins into UP / DOWN / SIDEWAYS.

    IMPORTANT: the forward return is computed and then the LAST row
    is dropped (it has no future to predict) — no look-ahead bias.
    """
    future_return = df["close"].pct_change(periods=1).shift(-1)

    target = pd.Series("SIDEWAYS", index=df.index, name="target")
    target[future_return > MODEL.UP_THRESHOLD] = "UP"
    target[future_return < MODEL.DOWN_THRESHOLD] = "DOWN"

    return target


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append hour, minute, and day-of-week columns.

    These capture intraday seasonality (e.g., volatility near open/close).
    """
    df = df.copy()
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["day_of_week"] = df.index.dayofweek
    return df


def prepare_features(
    df: pd.DataFrame,
    sentiment_col: str = "sentiment_score",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build X (features) and y (target) from an indicator-enriched DataFrame.

    Steps:
      1. Add time-of-day features.
      2. Create forward-looking target (no lookahead).
      3. Drop the last row (no future target).
      4. Drop non-feature columns (open/high/low/close/volume/symbol/interval).
      5. Fill any remaining NaN with 0 (e.g., missing sentiment).

    Returns:
        (X, y) — aligned DataFrames/Series ready for model.fit().
    """
    df = add_time_features(df)
    target = create_target(df)

    # Drop last row (no future target) and any remaining NaN
    valid = target.notna() & (target != "SIDEWAYS") | target.eq("SIDEWAYS")
    valid.iloc[-1] = False  # last row has NaN future return
    df = df.loc[valid]
    target = target.loc[valid]

    # Drop raw OHLCV + metadata columns (keep only features)
    drop_cols = ["open", "high", "low", "close", "volume",
                 "symbol", "interval", "target"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].fillna(0)
    y = target

    logger.info("Prepared %d samples with %d features", len(X), X.shape[1])
    return X, y


# ── Walk-Forward Validation ─────────────────────────────────────

def walk_forward_split(
    X: pd.DataFrame,
    n_splits: int = MODEL.WF_N_SPLITS,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate expanding-window train/test index pairs.

    Uses sklearn TimeSeriesSplit — train always precedes test
    chronologically, with no data leakage.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X))
    logger.info("Walk-forward: %d folds, last train=%d / test=%d",
                len(splits), len(splits[-1][0]), len(splits[-1][1]))
    return splits


# ── Model Training ──────────────────────────────────────────────

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    """Train an XGBoost classifier with sensible defaults.

    Uses GPU (tree_method='hist', device='cuda') when available.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device="cuda",
        eval_metric="mlogloss",
        random_state=42,
        use_label_encoder=False,
    )
    model.fit(X_train, y_enc)
    model._label_encoder = le  # attach for inverse_transform
    logger.info("XGBoost trained on %d samples", len(X_train))
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_enc)
    model._label_encoder = le
    logger.info("Random Forest trained on %d samples", len(X_train))
    return model


# ── Evaluation ──────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
) -> Dict:
    """Evaluate a trained model and return metrics dict.

    Returns dict with accuracy, per-class precision/recall/F1,
    and the full classification report string.
    """
    le = model._label_encoder
    y_true = le.transform(y_test)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=le.classes_,
        output_dict=True,
    )
    report_str = classification_report(
        y_true, y_pred,
        target_names=le.classes_,
    )

    logger.info("%s accuracy: %.4f\n%s", model_name, acc, report_str)
    return {"accuracy": acc, "report": report, "report_str": report_str}


# ── Full Pipeline ───────────────────────────────────────────────

def run_training_pipeline(
    df: pd.DataFrame,
    save: bool = True,
) -> Dict:
    """End-to-end training: features → walk-forward CV → save best model.

    Args:
        df: Indicator-enriched DataFrame (output of compute_all_indicators).
        save: Persist best model to PATHS.MODEL_DIR.

    Returns:
        Dict with 'xgboost' and 'random_forest' metrics, plus 'best_model' name.
    """
    X, y = prepare_features(df)
    splits = walk_forward_split(X)

    results = {}
    for name, train_fn in [("xgboost", train_xgboost),
                           ("random_forest", train_random_forest)]:
        fold_metrics: list = []
        best_model = None

        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            model = train_fn(X_tr, y_tr)
            metrics = evaluate_model(model, X_te, y_te, f"{name}_fold{fold}")
            metrics["fold"] = fold
            fold_metrics.append(metrics)
            best_model = model  # keep last fold model (most data)

        avg_acc = np.mean([m["accuracy"] for m in fold_metrics])
        results[name] = {"fold_metrics": fold_metrics, "avg_accuracy": avg_acc,
                         "model": best_model}
        logger.info("%s avg accuracy across %d folds: %.4f",
                    name, len(fold_metrics), avg_acc)

    # Determine best model
    best_name = max(results, key=lambda k: results[k]["avg_accuracy"])
    results["best_model"] = best_name

    if save:
        PATHS.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = PATHS.MODEL_DIR / f"{best_name}_best.pkl"
        joblib.dump(results[best_name]["model"], path)
        logger.info("Saved best model (%s) to %s", best_name, path)

    return results
