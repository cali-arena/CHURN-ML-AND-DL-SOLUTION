"""
Data loading and cleaning for the Telco Customer Churn dataset (Kaggle – blastchar).
Preprocessing pipeline: numerical (StandardScaler) + categorical (OneHotEncoder) via ColumnTransformer.
"""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Expected categorical values for validation (Churn only; other cats left flexible)
VALID_CHURN = {"Yes", "No"}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
TARGET_COL = "Churn"
ID_COL = "customerID"

# Feature columns for the preprocessing pipeline (exclude target)
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# --- Proxy justifications (for interpretability and documentation) ---
# Service usage: columns that proxy “how much and which services the customer uses”.
# Rationale: churn is tied to product stickiness; service breadth and type (phone, internet,
# add-ons like streaming, security, backup) capture engagement. Low or narrow usage
# is a known churn signal; OneHotEncoder preserves each choice (e.g. Fiber vs DSL).
SERVICE_USAGE_PROXY_COLS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Payment behavior: columns that proxy “how the customer pays and commits”.
# Rationale: contract length (month-to-month vs term) and payment method (e.g. automatic
# vs manual) are strong churn predictors; paperless billing reflects channel preference.
# These are standard operational proxies for payment reliability and commitment.
PAYMENT_BEHAVIOR_PROXY_COLS = [
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def load_and_clean(
    csv_path: str | Path,
    *,
    save_customer_ids_path: str | Path | None = None,
    total_charges_empty_fill: float = 0.0,
) -> Tuple[pd.DataFrame, dict]:
    """
    Load the Telco Customer Churn CSV, clean it, and validate.

    Steps:
    1. Load CSV.
    2. Record missing-values summary (before cleaning).
    3. Save customerID for later inference, then drop it from the dataframe.
    4. Convert TotalCharges to numeric (empty/whitespace -> NaN, then filled safely).
    5. Encode target: Churn Yes->1, No->0.
    6. Validate: no negative values, no invalid Churn categories.

    Parameters
    ----------
    csv_path : str or Path
        Path to the raw CSV file.
    save_customer_ids_path : str or Path, optional
        If set, customerID is saved here (e.g. data/processed/customer_ids.csv)
        for later join at inference. If None, IDs are only preserved in the
        returned summary and not written to disk.
    total_charges_empty_fill : float, default 0.0
        Value used to fill TotalCharges after converting empty/whitespace to NaN.
        Use 0.0 for "new customer, no charges yet".

    Returns
    -------
    df_clean : pandas.DataFrame
        Cleaned dataframe without customerID; Churn is 0/1; TotalCharges numeric.
    summary : dict
        Keys: "missing_before", "missing_after", "customer_ids_path", "rows_dropped".
        "missing_before" / "missing_after" are dicts {column_name: missing_count}.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    # --- Missing before cleaning (nulls + whitespace-only for object cols) ---
    missing_before = {c: int(v) for c, v in df.isnull().sum().items()}
    for c in df.select_dtypes(include=["object"]).columns:
        if c in df.columns:
            empty_str = (df[c].astype(str).str.strip() == "").sum()
            if empty_str > 0:
                missing_before[c] = missing_before.get(c, 0) + int(empty_str)
    missing_before = {k: v for k, v in missing_before.items() if v > 0}

    # --- Save customerID for later inference, then drop ---
    customer_ids = df[ID_COL].copy()
    if save_customer_ids_path is not None:
        out_path = Path(save_customer_ids_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        customer_ids.to_csv(out_path, index=False, header=[ID_COL])
    df = df.drop(columns=[ID_COL])

    # --- TotalCharges: safe numeric conversion ---
    # Replace empty/whitespace with NaN, then convert
    tc = df["TotalCharges"].astype(str).str.strip()
    df["TotalCharges"] = pd.to_numeric(tc.replace("", pd.NA), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(total_charges_empty_fill)

    # --- Target encoding: Churn Yes->1, No->0 ---
    is_obj_or_cat = (
        df[TARGET_COL].dtype == object
        or getattr(df[TARGET_COL].dtype, "name", "") in ("category", "string", "str")
    )
    if is_obj_or_cat:
        _validate_churn_categories(df[TARGET_COL])
        df[TARGET_COL] = (df[TARGET_COL].astype(str).str.strip().str.lower() == "yes").astype(int)
    else:
        # Already numeric? Ensure 0/1
        unique_vals = set(df[TARGET_COL].dropna().unique())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(f"Churn has unexpected values: {unique_vals}")

    # --- Ensure numeric columns are numeric ---
    for col in ["tenure", "SeniorCitizen", "MonthlyCharges"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Validation: no negatives ---
    _validate_no_negatives(df)
    # --- Validation: Churn already validated above ---

    # --- Missing after cleaning ---
    missing_after = {k: int(v) for k, v in df.isnull().sum().items() if v > 0}

    summary = {
        "missing_before": missing_before,
        "missing_after": missing_after,
        "customer_ids_path": str(save_customer_ids_path) if save_customer_ids_path else None,
        "rows_dropped": 0,
        "n_rows": len(df),
    }
    return df, summary


def _validate_churn_categories(series: pd.Series) -> None:
    """Raise if Churn contains values other than Yes/No (case-insensitive, stripped)."""
    vals = set(series.astype(str).str.strip().str.lower().dropna().unique())
    bad = vals - {"yes", "no"}
    if bad:
        raise ValueError(f"Invalid Churn categories: {bad}. Expected only Yes/No.")


def _validate_no_negatives(df: pd.DataFrame) -> None:
    """Raise if any of tenure, MonthlyCharges, TotalCharges, SeniorCitizen are negative."""
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        neg = (s < 0).sum()
        if neg > 0:
            raise ValueError(
                f"Column '{col}' has {int(neg)} negative value(s). "
                "Validation requires no negative values."
            )


def load_and_clean_for_inference(
    csv_path: str | Path,
    *,
    total_charges_empty_fill: float = 0.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load Telco-style CSV and prepare feature DataFrame for inference.

    Applies the same cleaning as load_and_clean for numeric/TotalCharges so
    the feature DataFrame matches what the preprocessor was fitted on.
    Churn may be absent or present; it is not used. customerID is required
    and returned separately for joining to predictions.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV (same schema as training: customerID + feature columns).
    total_charges_empty_fill : float, default 0.0
        Value used to fill TotalCharges after converting empty/whitespace to NaN.

    Returns
    -------
    df_features : pandas.DataFrame
        DataFrame with columns NUMERIC_FEATURES + CATEGORICAL_FEATURES only.
    customer_ids : pandas.Series
        customerID for each row (same length as df_features).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) | {ID_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    customer_ids = df[ID_COL].copy()
    df = df.copy()

    # TotalCharges: same logic as load_and_clean
    tc = df["TotalCharges"].astype(str).str.strip()
    df["TotalCharges"] = pd.to_numeric(tc.replace("", pd.NA), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(total_charges_empty_fill)

    for col in ["tenure", "SeniorCitizen", "MonthlyCharges"]:
        if col in df.columns and (df[col].dtype == object or getattr(df[col].dtype, "name", "") == "category"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _validate_no_negatives(df)

    df_features = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    return df_features, customer_ids


def get_cleaning_summary_report(summary: dict) -> str:
    """
    Format the cleaning summary for logging/printing.

    Parameters
    ----------
    summary : dict
        The summary dict returned by load_and_clean.

    Returns
    -------
    str
        Human-readable report of missing values before/after and basic stats.
    """
    lines = [
        "=== Data Cleaning Summary ===",
        "",
        "Missing values BEFORE cleaning (per column):",
    ]
    before = summary.get("missing_before") or {}
    if not before:
        lines.append("  (none)")
    else:
        for col, n in sorted(before.items()):
            lines.append(f"  {col}: {n}")

    lines.extend([
        "",
        "Missing values AFTER cleaning (per column):",
    ])
    after = summary.get("missing_after") or {}
    if not after:
        lines.append("  (none)")
    else:
        for col, n in sorted(after.items()):
            lines.append(f"  {col}: {n}")

    lines.extend([
        "",
        f"Rows in cleaned DataFrame: {summary.get('n_rows', '—')}",
        f"Customer IDs saved to: {summary.get('customer_ids_path') or '—'}",
    ])
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Stratified train / validation / test split
# -----------------------------------------------------------------------------

# Why stratification matters here:
# Churn is imbalanced (e.g. ~27% Yes). A random split can leave one fold with
# far fewer churners, so validation/test metrics (e.g. recall, AUC) become
# unstable and misleading. Stratifying by Churn keeps the 0/1 ratio similar
# in train, validation, and test, so metrics are comparable and the model
# sees the same class balance during training and evaluation.
STRATIFICATION_NOTE = (
    "Stratification keeps the Churn (0/1) ratio similar in train/val/test, "
    "so metrics are stable and evaluation is fair for this imbalanced target."
)


def split_stratified(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int | None = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train / validation / test with stratification by target.

    Ratios must sum to 1. Splits are reproducible when random_state is fixed.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError(
            f"Ratios must sum to 1, got train={train_ratio} val={val_ratio} test={test_ratio}"
        )
    if target_col not in df.columns:
        raise ValueError(f"Target column missing: {target_col}")

    y = df[target_col]
    # First split: train vs (val+test)
    df_train, df_rest = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        stratify=y,
        random_state=random_state,
    )
    # Second split: val vs test, stratify on remainder
    y_rest = df_rest[target_col]
    test_of_rest = test_ratio / (val_ratio + test_ratio)  # 0.5 for 15/15
    df_val, df_test = train_test_split(
        df_rest,
        test_size=test_of_rest,
        stratify=y_rest,
        random_state=random_state,
    )
    return df_train, df_val, df_test


def print_split_class_distribution(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str = TARGET_COL,
    *,
    class_names: dict[int, str] | None = None,
) -> None:
    """Print class (Churn) distribution for train, validation, and test splits."""
    if class_names is None:
        class_names = {0: "No churn (0)", 1: "Churn (1)"}
    for name, d in [("Train", df_train), ("Validation", df_val), ("Test", df_test)]:
        vc = d[target_col].value_counts().sort_index()
        n = len(d)
        print(f"\n{name} (n={n}):")
        for k, count in vc.items():
            label = class_names.get(int(k), str(k))
            pct = 100.0 * count / n
            print(f"  {label}: {count} ({pct:.1f}%)")


# -----------------------------------------------------------------------------
# Preprocessing pipeline (ColumnTransformer → NumPy, save/load)
# -----------------------------------------------------------------------------


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer for numerical (StandardScaler) and categorical
    (OneHotEncoder, handle_unknown="ignore") features.
    Output is a dense NumPy array suitable for neural networks.
    """
    num = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    cat = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), num),
            (
                "cat",
                OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore"),
                cat,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def fit_preprocessor(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    preprocessor: ColumnTransformer | None = None,
) -> Tuple[ColumnTransformer, np.ndarray, np.ndarray]:
    """
    Fit the preprocessor on feature columns and return (fitted preprocessor, X, y).

    X is a dense NumPy array ready for neural networks; y is the target as 1D array.
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing feature columns: {sorted(missing)}")
    if target_col not in df.columns:
        raise ValueError(f"Target column missing: {target_col}")
    X_df = df.drop(columns=[target_col])
    y = df[target_col].values.astype(np.float32)
    X = preprocessor.fit_transform(X_df)
    if not isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=np.float32)
    return preprocessor, X, y


def transform_with_preprocessor(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    target_col: str | None = TARGET_COL,
) -> np.ndarray:
    """
    Transform features using a fitted preprocessor. Returns dense NumPy array.
    If target_col is in df and provided, it is dropped before transform.
    """
    if target_col and target_col in df.columns:
        df = df.drop(columns=[target_col])
    X = preprocessor.transform(df)
    if not isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=np.float32)
    return X


def save_preprocessor(preprocessor: ColumnTransformer, path: str | Path) -> Path:
    """Save the fitted preprocessor to disk (joblib)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    return path


def load_preprocessor(path: str | Path) -> ColumnTransformer:
    """Load a preprocessor from disk."""
    return joblib.load(Path(path))


def get_proxy_justification_report() -> str:
    """
    Return a short justification for the proxy feature groups used in the pipeline.
    Use for documentation or reports.
    """
    return """
=== Proxy justifications ===

1) Service usage (columns: {service})
   These columns proxy “how much and which services the customer uses”.
   Churn is strongly tied to product stickiness: narrow or low usage is a known
   risk. Phone/internet type (e.g. Fiber vs DSL) and add-ons (streaming, security,
   backup, tech support) capture engagement. OneHotEncoder keeps each choice
   explicit for the model.

2) Payment behavior (columns: {payment})
   These columns proxy “how the customer pays and commits”.
   Contract length (month-to-month vs one/two year) and payment method (e.g.
   automatic vs manual) are standard operational predictors of churn and
   commitment. PaperlessBilling reflects channel preference and is often
   correlated with payment reliability.
""".format(
        service=", ".join(SERVICE_USAGE_PROXY_COLS),
        payment=", ".join(PAYMENT_BEHAVIOR_PROXY_COLS),
    ).strip()


if __name__ == "__main__":
    import sys

    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    ids_path = project_root / "data" / "processed" / "customer_ids.csv"
    preprocessor_path = project_root / "models" / "preprocessor" / "preprocessor.joblib"

    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        ids_path = Path(sys.argv[2])
    if len(sys.argv) > 3:
        preprocessor_path = Path(sys.argv[3])

    df_clean, summary = load_and_clean(
        csv_path,
        save_customer_ids_path=ids_path,
    )
    print(get_cleaning_summary_report(summary))

    # Stratified 70/15/15 split (reproducible)
    df_train, df_val, df_test = split_stratified(
        df_clean,
        target_col=TARGET_COL,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )
    print("\n--- Stratified split (70/15/15) ---")
    print("Why stratification: " + STRATIFICATION_NOTE)
    print_split_class_distribution(df_train, df_val, df_test, target_col=TARGET_COL)

    # Fit preprocessor on train only, then transform all splits
    preprocessor, X_train, y_train = fit_preprocessor(df_train)
    save_preprocessor(preprocessor, preprocessor_path)
    X_val = transform_with_preprocessor(df_val, preprocessor)
    y_val = df_val[TARGET_COL].values.astype(np.float32)
    X_test = transform_with_preprocessor(df_test, preprocessor)
    y_test = df_test[TARGET_COL].values.astype(np.float32)

    print("\nPreprocessor fitted on train and saved to:", preprocessor_path)
    print("Shapes: X_train", X_train.shape, "y_train", y_train.shape)
    print("       X_val  ", X_val.shape, "  y_val  ", y_val.shape)
    print("       X_test ", X_test.shape, " y_test ", y_test.shape)

    print("\n" + get_proxy_justification_report())
