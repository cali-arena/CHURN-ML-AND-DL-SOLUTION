"""
Keras MLP for Telco Customer Churn prediction (binary classification).

Architecture: simple feedforward MLP with Dropout for regularization.
Output: probability of churn (sigmoid), not a hard decision.

Training: binary crossentropy, Adam, EarlyStopping on val_loss,
class weighting for imbalance, save best model.

Requires Python 3.10–3.12 for TensorFlow; Python 3.13 is not yet supported.
"""

import sys

if sys.version_info >= (3, 13):
    sys.exit(
        "TensorFlow does not support Python 3.13 yet. Use Python 3.10, 3.11, or 3.12.\n"
        "Example: py -3.12 -m pip install -r requirements.txt  then  py -3.12 src/train.py"
    )

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# -----------------------------------------------------------------------------
# Layer-choice explanations (for explainability)
# -----------------------------------------------------------------------------
#
# Input (Dense input_dim)
#   - One neuron per preprocessed feature (e.g. 45 after StandardScaler + OneHot).
#   - Features are already scaled and encoded; no extra embedding needed.
#
# Hidden layers (Dense + ReLU + Dropout)
#   - Dense: linear combinations of inputs; ReLU adds nonlinearity so the model
#     can learn interactions (e.g. tenure + contract type).
#   - ReLU: simple, avoids vanishing gradients, and is easy to interpret as
#     "activate when signal is positive."
#   - Units: modest widths (e.g. 64, 32) keep the model small and interpretable;
#     enough capacity for tabular churn patterns without overfitting.
#   - Dropout (after each hidden block): randomly zeros a fraction of activations
#     at training time, so the model cannot rely on single neurons. Reduces
#     overfitting and improves generalization; no effect at inference.
#
# Output (Dense(1) + Sigmoid)
#   - Single unit: one output per sample.
#   - Sigmoid: maps logits to P(Churn=1) in [0,1]. Matches binary cross-entropy
#     and gives a probability for downstream use (e.g. ranking, thresholding).
#   - We do not apply a threshold inside the model; the business can choose
#     a cutoff (e.g. 0.5) when making decisions.
# -----------------------------------------------------------------------------


def build_churn_mlp(
    input_dim: int,
    hidden_units: tuple[int, ...] = (64, 32),
    dropout_rate: float = 0.3,
    *,
    seed: int | None = 42,
) -> Model:
    """
    Build a Keras MLP for binary churn classification.

    Binary classification, sigmoid output (probability of churn), Dropout
    after each hidden layer. Architecture is kept simple and explainable.

    Parameters
    ----------
    input_dim : int
        Number of input features (e.g. X_train.shape[1] after preprocessing).
    hidden_units : tuple of int, default (64, 32)
        Sizes of hidden layers. Small sizes keep the model interpretable.
    dropout_rate : float, default 0.3
        Fraction of units to drop after each hidden layer (training only).
    seed : int or None, default 42
        Random seed for reproducibility (weights init, dropout mask).

    Returns
    -------
    keras.Model
        Compiled model: loss=binary_crossentropy, optimizer=adam, metrics=[accuracy].
    """
    if seed is not None:
        keras.utils.set_random_seed(seed)

    inputs = layers.Input(shape=(input_dim,), name="inputs")

    x = inputs
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="relu", name=f"hidden_{i}")(x)
        x = layers.Dropout(dropout_rate, seed=seed, name=f"dropout_{i}")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="churn_mlp")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_architecture_explanation() -> str:
    """
    Short explanation of each layer choice for documentation and reports.
    """
    return """
=== Churn MLP – layer choice rationale ===

1) Input
   - Shape (input_dim,): one dimension per preprocessed feature.
   - No extra layer: inputs are already scaled (StandardScaler) and
     one‑hot encoded; the first Dense layer learns linear combinations.

2) Hidden layers (Dense → ReLU → Dropout)
   - Dense: learns weighted combinations of features; supports interactions
     (e.g. tenure × contract type).
   - ReLU: adds nonlinearity with minimal vanishing gradients; easy to
     explain as “fire when positive.”
   - Small widths (e.g. 64, 32): enough capacity for tabular churn signals
     without unnecessary complexity.
   - Dropout: regularizes by dropping a fraction of activations at training
     time; reduces overfitting, no change at inference.

3) Output (Dense(1) + Sigmoid)
   - Single unit: one score per customer.
   - Sigmoid: outputs P(Churn=1) in [0, 1], suitable for binary cross‑entropy
     and for ranking or thresholding by the business.
   - No built‑in decision rule; probability is the deliverable.
""".strip()


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """
    Compute class weights for imbalanced binary targets (Keras format).

    Uses sklearn 'balanced': weight = n_samples / (n_classes * n_samples_per_class).
    Returns {0: w0, 1: w1} for use in model.fit(class_weight=...).
    """
    y_flat = np.asarray(y).ravel()
    classes = np.unique(y_flat)
    weights = compute_class_weight(
        "balanced", classes=classes, y=y_flat.astype(int)
    )
    return dict(zip(classes.astype(int), weights.tolist()))


def train_churn_mlp(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_path: str | Path,
    *,
    epochs: int = 200,
    batch_size: int = 32,
    patience: int = 10,
    seed: int | None = 42,
) -> tuple[keras.callbacks.History, dict[str, Any]]:
    """
    Train the churn MLP with EarlyStopping, class weighting, and best-model save.

    Uses binary_crossentropy and Adam (already set on model). EarlyStopping
    monitors val_loss; best model is saved to save_path and optionally
    restored into model (restore_best_weights=True).

    Returns
    -------
    history : keras.callbacks.History
        Training history (history.history has loss, val_loss, accuracy, val_accuracy).
    log : dict
        Keys: "history" (history.history), "epoch_stopped", "best_epoch", "save_path".
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    class_weight = compute_class_weights(y_train)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    checkpoint = ModelCheckpoint(
        save_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    if seed is not None:
        keras.utils.set_random_seed(seed)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=1,
    )

    epoch_stopped = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1  # 1-based

    log = {
        "history": history.history,
        "epoch_stopped": epoch_stopped,
        "best_epoch": best_epoch,
        "save_path": str(save_path),
    }
    return history, log


def format_training_log(log: dict[str, Any]) -> str:
    """Format training log (history + epoch stopped) for printing."""
    lines = [
        "=== Training log ===",
        f"Epoch where training stopped: {log['epoch_stopped']}",
        f"Best epoch (by val_loss): {log['best_epoch']}",
        f"Best model saved to: {log['save_path']}",
        "",
        "Training history (final epoch):",
    ]
    h = log["history"]
    if h:
        last = {k: v[-1] for k, v in h.items()}
        for k, v in sorted(last.items()):
            try:
                lines.append(f"  {k}: {float(v):.4f}")
            except (TypeError, ValueError):
                lines.append(f"  {k}: {v}")
    return "\n".join(lines)


if __name__ == "__main__":
    from preprocess import (
        load_and_clean,
        split_stratified,
        fit_preprocessor,
        save_preprocessor,
        transform_with_preprocessor,
        TARGET_COL,
    )

    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    ids_path = project_root / "data" / "processed" / "customer_ids.csv"
    preprocessor_path = project_root / "models" / "preprocessor" / "preprocessor.joblib"
    model_path = project_root / "models" / "best_model" / "churn_mlp.keras"

    df_clean, _ = load_and_clean(csv_path, save_customer_ids_path=ids_path)
    df_train, df_val, df_test = split_stratified(
        df_clean, target_col=TARGET_COL, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
    )
    preprocessor, X_train, y_train = fit_preprocessor(df_train)
    save_preprocessor(preprocessor, preprocessor_path)
    X_val = transform_with_preprocessor(df_val, preprocessor)
    y_val = np.asarray(df_val[TARGET_COL].values, dtype=np.float32)

    model = build_churn_mlp(input_dim=X_train.shape[1], hidden_units=(64, 32), dropout_rate=0.3, seed=42)
    print("Model summary:")
    model.summary()

    history, log = train_churn_mlp(
        model, X_train, y_train, X_val, y_val,
        save_path=model_path,
        epochs=200,
        batch_size=32,
        patience=10,
        seed=42,
    )
    print("\n" + format_training_log(log))
    print("\nFull training history (per epoch):", log["history"])
