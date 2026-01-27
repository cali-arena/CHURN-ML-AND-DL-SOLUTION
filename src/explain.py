"""
Model explainability for the churn MLP: permutation importance, top features, business interpretation.

Uses permutation importance (no SHAP dependency). Identifies the top 10 features
that most influence churn predictions and explains them in plain business language.
"""

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 13):
    sys.exit(
        "TensorFlow (required to load the model) does not support Python 3.13 yet. "
        "Use Python 3.10, 3.11, or 3.12."
    )

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


def _keras_roc_auc_scorer(estimator, X, y):
    """Scorer for permutation_importance: ROC AUC from Keras model probabilities."""
    probs = np.asarray(estimator.predict(X, verbose=0)).ravel()
    return roc_auc_score(y, probs)


def compute_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_repeats: int = 5,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute permutation importance (drop in ROC AUC when each feature is shuffled).

    Returns (importances, std) for each feature, in order of columns of X.
    Higher importance = feature matters more for predictions.
    """
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=_keras_roc_auc_scorer,
        n_jobs=1,
    )
    return result.importances_mean, result.importances_std


def get_top_k_features(
    importances: np.ndarray,
    feature_names: list[str],
    k: int = 10,
) -> list[tuple[str, float]]:
    """Return top k (name, importance) pairs, sorted by importance descending."""
    order = np.argsort(-np.asarray(importances))
    out = []
    for i in order[:k]:
        if i < len(feature_names):
            out.append((feature_names[i], float(importances[i])))
    return out


def _humanize_feature_name(name: str) -> str:
    """Turn pipeline feature names into short, readable labels for plots and text."""
    s = str(name).strip()
    for prefix in ("num__", "cat__"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    # Underscore to space; keep e.g. "Month-to-month" readable
    s = s.replace("_", " ").strip()
    return s.title() if s else name


def plot_feature_importance(
    top_features: list[tuple[str, float]],
    save_path: str | Path,
    *,
    title: str = "Top features influencing churn (permutation importance)",
    max_display: int = 10,
) -> Path:
    """Plot horizontal bar chart of top feature importances and save to save_path."""
    import matplotlib.pyplot as plt

    display = top_features[:max_display]
    if not display:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).touch()
        return Path(save_path)

    names = [_humanize_feature_name(n) for n, _ in display]
    vals = [v for _, v in display]

    fig, ax = plt.subplots(figsize=(8, max(4, len(display) * 0.4)))
    bars = ax.barh(range(len(names)), vals, color="steelblue", alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Importance (drop in ROC AUC when feature is shuffled)")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def write_business_interpretation(
    top_features: list[tuple[str, float]],
    save_path: str | Path,
) -> str:
    """
    Write a short, business-language interpretation of the top driving factors for churn.

    Uses plain language: what each feature represents and why it matters for retention.
    """
    # Map common pipeline names to business-friendly explanations (plain language)
    hints = {
        "contract": "Contract type (e.g. month-to-month vs annual). Short or month-to-month contracts are often tied to higher churn.",
        "tenure": "How long the customer has been with the company. Shorter tenure often means higher churn risk.",
        "monthlycharges": "Monthly bill amount. Very high charges can push some customers to leave.",
        "totalcharges": "Total amount paid so far. Reflects both tenure and spend; low totals can mean new or low-engagement customers.",
        "electroniccheck": "Paying by electronic check (vs automatic or mailed). Often linked to higher churn, e.g. more manual steps or different segment.",
        "mailedcheck": "Paying by mailed check. Manual payment methods are often associated with higher churn than automatic.",
        "banktransfer": "Bank transfer (automatic) payment. Automatic payments tend to go with lower churn.",
        "creditcard": "Credit card (automatic) payment. Same as above; less friction and more commitment.",
        "paymentmethod": "How the customer pays. Automatic payments are typically associated with lower churn.",
        "internetservice": "Type of internet (e.g. fiber, DSL, none). Fiber and quality of service affect satisfaction and churn.",
        "fiber": "Fiber internet. Service quality and expectations; can drive both satisfaction and churn depending on experience.",
        "dsl": "DSL internet. Often a different segment from fiber; engagement and expectations differ.",
        "onlinesecurity": "Whether they have online security add-ons. Having add-ons often means higher engagement and lower churn.",
        "techsupport": "Tech support add-on. Can reduce churn by helping resolve issues.",
        "streaming": "Streaming TV or movies. Usage of add-ons is a signal of engagement.",
        "paperless": "Paperless billing. Preference and habit; often correlates with other behaviors.",
        "partner": "Having a partner (e.g. household). Can reflect segment and stability.",
        "dependents": "Having dependents. Often tied to household type and commitment.",
        "gender": "Gender. Demographic signal; importance depends on segment mix.",
        "senior": "Senior citizen flag. Age segment can have different churn patterns.",
        "phoneservice": "Having phone service. Part of the product bundle and engagement.",
        "multiplelines": "Multiple phone lines. Another signal of bundle size and engagement.",
        "deviceprotection": "Device protection add-on. Add-ons tend to correlate with lower churn.",
        "onlinebackup": "Online backup add-on. Same as above.",
        "nointernetservice": "No internet service. Reflects product mix; different risk profile.",
        "nophoneservice": "No phone service. Same idea; bundle and engagement signal.",
    }

    def explain_one(name: str) -> str:
        key = name.lower().replace(" ", "").replace("-", "").replace("_", "")
        for k, v in hints.items():
            if k in key or key.startswith(k):
                return v
        return "Influences churn risk; see data and domain experts for context."

    lines = [
        "=== What drives churn? Top factors in plain language ===\n",
        "The model's most influential inputs (by permutation importance) and what they mean for retention:\n",
    ]
    for i, (feat, imp) in enumerate(top_features[:10], 1):
        short = _humanize_feature_name(feat)
        expl = explain_one(feat)
        lines.append(f"{i}. {short}")
        lines.append(f"   {expl}\n")

    lines.append(
        "These rankings show which levers (contracts, add-ons, payment type, tenure, etc.) "
        "matter most for predicting churn. Use them to focus retention efforts and product decisions."
    )
    text = "\n".join(lines)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(text)
    return text


def run_explain(
    model_path: str | Path,
    preprocessor_path: str | Path,
    csv_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    top_k: int = 10,
    n_repeats: int = 5,
    random_state: int | None = 42,
) -> dict[str, Any]:
    """
    Load model and data, compute permutation importance, plot top features, write interpretation.

    Returns dict with top_features, importance_plot_path, interpretation_path, feature_names, importances.
    """
    import joblib
    from tensorflow import keras

    from preprocess import (
        TARGET_COL,
        load_and_clean,
        split_stratified,
        transform_with_preprocessor,
    )

    project_root = Path(__file__).resolve().parents[1]
    model_path = Path(model_path)
    preprocessor_path = Path(preprocessor_path)
    csv_path = Path(csv_path)

    model = keras.models.load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)

    df_clean, _ = load_and_clean(csv_path, save_customer_ids_path=None)
    _, _, df_test = split_stratified(
        df_clean, target_col=TARGET_COL, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
    )
    X_test = transform_with_preprocessor(df_test, preprocessor)
    y_test = np.asarray(df_test[TARGET_COL].values, dtype=np.float32)

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]

    importances, std = compute_permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=random_state
    )
    top_features = get_top_k_features(importances, feature_names, k=top_k)

    out_dir = Path(output_dir) if output_dir else project_root / "reports"
    out_dir = out_dir.resolve()
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_path = figures_dir / "feature_importance.png"
    plot_feature_importance(
        top_features, plot_path,
        title="Top features influencing churn (permutation importance)",
        max_display=top_k,
    )

    interp_path = out_dir / "feature_importance_interpretation.txt"
    interpretation_text = write_business_interpretation(top_features, interp_path)

    return {
        "top_features": top_features,
        "importances": importances.tolist(),
        "feature_names": feature_names,
        "importance_plot_path": str(plot_path),
        "interpretation_path": str(interp_path),
        "interpretation": interpretation_text,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    preprocessor_path = project_root / "models" / "preprocessor" / "preprocessor.joblib"
    model_path = project_root / "models" / "best_model" / "churn_mlp.keras"
    output_dir = project_root / "reports"

    if not model_path.exists():
        print(f"Model not found: {model_path}. Run train.py first.")
        sys.exit(1)

    out = run_explain(
        model_path,
        preprocessor_path,
        csv_path,
        output_dir=output_dir,
        top_k=10,
        n_repeats=5,
        random_state=42,
    )

    print("Top 10 features influencing churn (permutation importance):")
    for i, (name, imp) in enumerate(out["top_features"], 1):
        print(f"  {i}. {name}: {imp:.4f}")
    print(f"\nFeature importance plot saved to: {out['importance_plot_path']}")
    print(f"Written interpretation saved to: {out['interpretation_path']}")
    print("\n--- Written interpretation (excerpt) ---\n")
    print(out["interpretation"])
