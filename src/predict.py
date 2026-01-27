"""
Inference pipeline for Telco Customer Churn.

Loads saved model and preprocessor, accepts new customer data (CSV),
outputs churn probability per customer, and exports a ranked CSV with
customerID | churn_probability | risk_level.

Risk levels: Low < 0.3; Medium 0.3–0.6; High > 0.6.

Requires Python 3.10–3.12 for TensorFlow; Python 3.13 is not yet supported.
"""

import argparse
import json
import sys
from pathlib import Path

if sys.version_info >= (3, 13):
    sys.exit(
        "TensorFlow (required to load the model) does not support Python 3.13 yet. "
        "Use Python 3.10, 3.11, or 3.12."
    )

import numpy as np
import pandas as pd
from tensorflow import keras

from preprocess import (
    load_and_clean_for_inference,
    load_preprocessor,
    transform_with_preprocessor,
)


# Risk band thresholds (inclusive boundaries for Medium)
RISK_LOW_MAX = 0.3
RISK_MEDIUM_MAX = 0.6


def _risk_level(proba: float) -> str:
    """Map churn probability to risk level."""
    if proba < RISK_LOW_MAX:
        return "Low"
    if proba <= RISK_MEDIUM_MAX:
        return "Medium"
    return "High"


def run_inference(
    input_csv: str | Path,
    model_path: str | Path,
    preprocessor_path: str | Path,
    output_csv: str | Path,
    *,
    total_charges_empty_fill: float = 0.0,
) -> pd.DataFrame:
    """
    Load model and preprocessor, score new customers, and export ranked results.

    Reads input CSV (Telco schema: customerID + feature columns; Churn optional),
    applies the same cleaning as training, runs the saved MLP, and writes
    customerID | churn_probability | risk_level to output_csv, sorted by
    churn_probability descending (highest risk first).

    Parameters
    ----------
    input_csv : str or Path
        Path to customer CSV (same columns as training data).
    model_path : str or Path
        Path to saved Keras model (e.g. churn_mlp.keras).
    preprocessor_path : str or Path
        Path to saved preprocessor (e.g. preprocessor.joblib).
    output_csv : str or Path
        Path for output CSV: customerID, churn_probability, risk_level.
    total_charges_empty_fill : float, default 0.0
        Value used to fill missing TotalCharges (same as training).

    Returns
    -------
    pandas.DataFrame
        DataFrame with customerID, churn_probability, risk_level, ranked by risk.
    """
    model_path = Path(model_path)
    preprocessor_path = Path(preprocessor_path)
    output_csv = Path(output_csv)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

    model = keras.models.load_model(model_path)
    preprocessor = load_preprocessor(preprocessor_path)

    df_features, customer_ids = load_and_clean_for_inference(
        input_csv, total_charges_empty_fill=total_charges_empty_fill
    )
    X = transform_with_preprocessor(df_features, preprocessor, target_col=None)
    proba = np.asarray(model.predict(X, verbose=0)).ravel()

    risk = np.array([_risk_level(p) for p in proba])
    out = pd.DataFrame({
        "customerID": customer_ids.values,
        "churn_probability": np.round(proba.astype(float), 6),
        "risk_level": risk,
    })
    out = out.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def save_predictions_json(df: pd.DataFrame, path: str | Path) -> Path:
    """Write predictions DataFrame to JSON (list of records, readable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert for JSON: churn_probability as float, rest as-is
    records = df.to_dict(orient="records")
    for r in records:
        r["churn_probability"] = float(r["churn_probability"])
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    return path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    default_csv = project_root / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    default_model = project_root / "models" / "best_model" / "churn_mlp.keras"
    default_preprocessor = project_root / "models" / "preprocessor" / "preprocessor.joblib"
    default_output = project_root / "reports" / "churn_predictions.csv"

    parser = argparse.ArgumentParser(
        description="Run churn inference: load model + preprocessor, score customers, export ranked CSV."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        default=default_csv,
        help=f"Input customer CSV (default: {default_csv})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model,
        help=f"Path to saved Keras model (default: {default_model})",
    )
    parser.add_argument(
        "--preprocessor",
        type=Path,
        default=default_preprocessor,
        help=f"Path to saved preprocessor (default: {default_preprocessor})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=default_output,
        help=f"Output CSV path (default: {default_output})",
    )
    parser.add_argument(
        "--total-charges-fill",
        type=float,
        default=0.0,
        help="Value to fill missing TotalCharges (default: 0.0)",
    )
    default_json = project_root / "reports" / "churn_predictions.json"
    parser.add_argument(
        "--json",
        "-j",
        type=Path,
        default=default_json,
        metavar="PATH",
        help=f"Also save predictions to JSON (default: {default_json})",
    )
    args = parser.parse_args()

    result = run_inference(
        args.input_csv,
        args.model,
        args.preprocessor,
        args.output,
        total_charges_empty_fill=args.total_charges_fill,
    )
    save_predictions_json(result, args.json)

    print(f"Inference complete: {len(result)} customers scored.")
    print(f"CSV written to: {args.output}")
    print(f"JSON written to: {args.json}")
    print("\nRisk distribution:")
    print(result["risk_level"].value_counts().sort_index().to_string())
