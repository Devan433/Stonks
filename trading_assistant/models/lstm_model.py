"""
LSTM time-series classifier for intraday direction prediction (PyTorch).

Architecture: Input → LSTM(64) → Dropout → LSTM(32) → Dropout →
              Linear(16, ReLU) → Linear(3, Softmax)

Uses walk-forward validation, early stopping, and model checkpointing.
Runs on GPU (RTX 4050) via PyTorch CUDA.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from trading_assistant.config import MODEL, PATHS
from trading_assistant.models.train_model import prepare_features, walk_forward_split

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Sequence Construction ───────────────────────────────────────

def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = MODEL.LSTM_SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert flat feature rows into overlapping sequences for LSTM.

    For each index i, builds window X[i-seq_len:i] as one sample
    with y[i] as the label.  First seq_len rows are consumed as
    context — no data leakage.

    Returns:
        (X_seq, y_seq) of shapes (n-seq_len, seq_len, features) and (n-seq_len,).
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len: i])
        y_seq.append(y[i])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)


# ── PyTorch Model ───────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """2-layer LSTM with dropout → dense → softmax (3-class)."""

    def __init__(self, n_features: int, n_classes: int = 3):
        super().__init__()
        u1, u2 = MODEL.LSTM_UNITS  # (64, 32)

        self.lstm1 = nn.LSTM(n_features, u1, batch_first=True)
        self.drop1 = nn.Dropout(MODEL.LSTM_DROPOUT)
        self.lstm2 = nn.LSTM(u1, u2, batch_first=True)
        self.drop2 = nn.Dropout(MODEL.LSTM_DROPOUT)
        self.fc1 = nn.Linear(u2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])   # last timestep only
        out = self.relu(self.fc1(out))
        return self.fc2(out)              # raw logits (CrossEntropyLoss handles softmax)


# ── Training Loop ───────────────────────────────────────────────

def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Tuple[LSTMClassifier, List[float]]:
    """Train the LSTM with early stopping.

    Args:
        X_train: Sequences (n, seq_len, features).
        y_train: Integer-encoded labels (n,).
        X_val / y_val: Optional validation set for early stopping.

    Returns:
        (trained_model, list of validation losses per epoch)
    """
    n_features = X_train.shape[2]
    model = LSTMClassifier(n_features).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_dl = DataLoader(train_ds, batch_size=MODEL.LSTM_BATCH_SIZE, shuffle=False)

    val_dl = None
    if X_val is not None and y_val is not None:
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        val_dl = DataLoader(val_ds, batch_size=MODEL.LSTM_BATCH_SIZE)

    # Early stopping state
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    val_losses: List[float] = []

    for epoch in range(1, MODEL.LSTM_EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        # --- Validate ---
        val_loss = train_loss  # fallback if no val set
        if val_dl:
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    vl += criterion(model(xb), yb).item() * len(xb)
            val_loss = vl / len(val_ds)

        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            logger.info("Epoch %3d/%d — train_loss=%.4f  val_loss=%.4f",
                        epoch, MODEL.LSTM_EPOCHS, train_loss, val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= MODEL.LSTM_PATIENCE:
                logger.info("Early stopping at epoch %d (patience=%d)",
                            epoch, MODEL.LSTM_PATIENCE)
                break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    logger.info("LSTM training complete — %d epochs, best val_loss=%.4f",
                len(val_losses), best_val_loss)
    return model, val_losses


# ── Full Pipeline ───────────────────────────────────────────────

def run_lstm_pipeline(
    df: pd.DataFrame,
    save: bool = True,
) -> Dict:
    """End-to-end LSTM training with walk-forward validation.

    1. Prepare features & target (same as XGBoost pipeline).
    2. Scale features with StandardScaler (per fold, fit on train only).
    3. Build sequences of length LSTM_SEQ_LEN.
    4. Walk-forward CV across N folds.
    5. Save best model.

    Returns:
        Dict with fold metrics, average accuracy, and model.
    """
    X, y = prepare_features(df)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    splits = walk_forward_split(X)
    fold_metrics: List[Dict] = []
    best_model = None

    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        logger.info("LSTM fold %d/%d", fold, len(splits))

        # Scale features (fit on train only — no leakage)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X.iloc[train_idx])
        X_te_scaled = scaler.transform(X.iloc[test_idx])

        # Build sequences
        X_tr_seq, y_tr_seq = create_sequences(X_tr_scaled, y_enc[train_idx])
        X_te_seq, y_te_seq = create_sequences(X_te_scaled, y_enc[test_idx])

        if len(X_tr_seq) < 50 or len(X_te_seq) < 10:
            logger.warning("Fold %d: too few sequences (%d train, %d test), skipping",
                           fold, len(X_tr_seq), len(X_te_seq))
            continue

        model, _ = train_lstm(X_tr_seq, y_tr_seq, X_te_seq, y_te_seq)

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_te_seq).to(DEVICE))
            y_pred = logits.argmax(dim=1).cpu().numpy()

        acc = accuracy_score(y_te_seq, y_pred)
        report = classification_report(
            y_te_seq, y_pred, target_names=le.classes_, output_dict=True,
        )
        fold_metrics.append({"fold": fold, "accuracy": acc, "report": report})
        best_model = model
        logger.info("Fold %d accuracy: %.4f", fold, acc)

    avg_acc = np.mean([m["accuracy"] for m in fold_metrics]) if fold_metrics else 0.0
    logger.info("LSTM avg accuracy: %.4f across %d folds", avg_acc, len(fold_metrics))

    if save and best_model:
        PATHS.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = PATHS.MODEL_DIR / "lstm_best.pt"
        torch.save(best_model.state_dict(), path)
        logger.info("Saved LSTM model to %s", path)

    return {
        "fold_metrics": fold_metrics,
        "avg_accuracy": avg_acc,
        "model": best_model,
        "label_encoder": le,
    }
