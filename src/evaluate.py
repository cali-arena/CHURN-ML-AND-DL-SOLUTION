"""
Evaluate the trained churn MLP on the test set.

Metrics: ROC AUC, Precision, Recall, F1-score, Confusion matrix.
Outputs: ROC curve plot, metrics report, and business meaning of FP vs FN.
"""

import sys
from pathlib import Path

if sys.version_info >= (3, 13):
    sys.exit(
        "TensorFlow (required to load the model) does not support Python 3.13 yet. "
        "Use Python 3.10, 3.11, or 3.12."
    )

import json
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow import keras


# -----------------------------------------------------------------------------
# Business meaning: False positives vs False negatives
# -----------------------------------------------------------------------------
#
# In churn prediction, we predict "will churn" (positive) vs "will stay" (negative).
#
# - False Positive (FP): We predicted churn, but the customer stayed.
#   Business impact: Wasted retention effort and cost (discounts, calls, offers)
#   on customers who would not have left. Can also irritate low-risk customers.
#   Tolerable if retention is cheap; costly if offers are expensive.
#
# - False Negative (FN): We predicted stay, but the customer churned.
#   Business impact: Missed chance to retain a leaver. Lost revenue and possible
#   recovery cost. Often the costlier error when losing a customer is expensive.
#
# Choosing a threshold (e.g. 0.5) trades off FP vs FN: lower threshold → more
# predicted churners (higher recall, more FP); higher threshold → fewer
# predicted churners (higher precision, more FN). ROC and precision–recall
# curves help pick a threshold that matches business costs.
# -----------------------------------------------------------------------------

FP_FN_BUSINESS_NOTE = """
=== Business meaning: False positives vs False negatives ===

  Positive = predicted churn | Negative = predicted stay
  Actual churn = true positive / false negative | Actual stay = false positive / true negative

  False Positive (FP): Predicted churn, but the customer STAYED.
    -> Wasted retention effort (offers, calls, discounts) on customers who would
      not have left. Can annoy low-risk customers. Cost depends on how expensive
      retention actions are.

  False Negative (FN): Predicted stay, but the customer CHURNED.
    -> Missed chance to retain a leaver; lost revenue and recovery opportunity.
    -> Often the costlier error when losing a customer is expensive.

  Trade-off: A lower score threshold flags more customers as at-risk (fewer FN,
  more FP). A higher threshold is more selective (fewer FP, more FN). Use ROC
  and cost assumptions to choose a threshold that fits your retention budget.
""".strip()


def evaluate_churn_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict:
    """
    Compute test-set metrics: ROC AUC, precision, recall, F1, confusion matrix.

    Predictions use the given threshold for binary metrics; ROC AUC uses
    raw probabilities.
    """
    y_prob = np.asarray(model.predict(X_test, verbose=0)).ravel()
    y_pred = (y_prob >= threshold).astype(np.int32)
    y_test = np.asarray(y_test).ravel().astype(np.int32)

    roc_auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "y_test": y_test,
    }


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path,
    *,
    title: str = "ROC curve (test set)",
) -> Path:
    """Plot ROC curve and save to save_path."""
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def format_metrics_report(metrics: dict) -> str:
    """Format metrics and confusion matrix for printing."""
    lines = [
        "=== Test-set metrics ===",
        f"Threshold: {metrics['threshold']}",
        f"ROC AUC:   {metrics['roc_auc']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall:    {metrics['recall']:.4f}",
        f"F1-score:  {metrics['f1_score']:.4f}",
        "",
        "Confusion matrix (rows=true, cols=predicted):",
        "                 Predicted=0 (stay)  Predicted=1 (churn)",
    ]
    cm = np.array(metrics["confusion_matrix"])
    if cm.shape == (2, 2):
        # sklearn confusion_matrix: rows=true, cols=pred; order 0,1 => [[tn,fp],[fn,tp]]
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        lines.append(f"  Actual=0 (stay)   {tn:>12}  {fp:>18}")
        lines.append(f"  Actual=1 (churn)  {fn:>12}  {tp:>18}")
    else:
        lines.append(f"  {cm}")
    return "\n".join(lines)


def threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> list[dict]:
    """
    Compute Precision, Recall, F1, and confusion counts for thresholds from 0.2 to 0.8.

    Returns a list of dicts, one per threshold, with keys: threshold, precision,
    recall, f1_score, tn, fp, fn, tp, n_pred_pos.
    """
    y_true = np.asarray(y_true).ravel().astype(np.int32)
    y_prob = np.asarray(y_prob).ravel()

    if thresholds is None:
        thresholds = np.arange(0.20, 0.81, 0.05)

    rows = []
    for t in np.atleast_1d(thresholds):
        t = float(t)
        y_pred = (y_prob >= t).astype(np.int32)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        else:
            tn = fp = fn = tp = 0
        n_pred_pos = int(np.sum(y_pred))
        rows.append({
            "threshold": t,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "n_pred_pos": n_pred_pos,
        })
    return rows


def format_threshold_table(rows: list[dict]) -> str:
    """Format the threshold-analysis rows as a text table."""
    if not rows:
        return "No threshold data."
    lines = [
        "=== Threshold analysis (0.2–0.8) — Precision vs Recall trade-off ===",
        "",
    ]
    header = "threshold  precision  recall   f1_score     tn    fp    fn    tp  n_pred_pos"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(
            f"    {r['threshold']:.2f}      {r['precision']:.4f}   {r['recall']:.4f}   {r['f1_score']:.4f}   "
            f"{r['tn']:>4}  {r['fp']:>4}  {r['fn']:>4}  {r['tp']:>4}  {r['n_pred_pos']:>10}"
        )
    return "\n".join(lines)


def recommend_threshold_churn_prevention(rows: list[dict]) -> tuple[float, str]:
    """
    Recommend an optimal threshold for churn prevention.

    For churn prevention, missing a churner (FN) is usually costlier than
    wasting effort on a stayer (FP). We favor recall while keeping precision
    acceptable. Strategy: among thresholds with recall >= 0.60, pick the one
    with the highest F1 (balance). If none, pick the one with highest recall.
    """
    if not rows:
        return 0.5, "Insufficient data for recommendation."

    # Require minimum recall so we catch a reasonable share of churners
    min_recall = 0.60
    candidates = [r for r in rows if r["recall"] >= min_recall]

    if candidates:
        best = max(candidates, key=lambda r: (r["f1_score"], r["recall"]))
        th = best["threshold"]
        just = (
            f"For churn prevention, missing churners (FN) is usually costlier than "
            f"wasted retention (FP). Among thresholds with recall >= {min_recall:.0%}, "
            f"{th:.2f} gives the best F1 ({best['f1_score']:.4f}) while keeping recall "
            f"{best['recall']:.4f} and precision {best['precision']:.4f}. "
            f"Recommended threshold: {th:.2f}."
        )
        return th, just

    # Fallback: highest recall above 0.5
    best = max(rows, key=lambda r: (r["recall"], r["f1_score"]))
    th = best["threshold"]
    just = (
        f"No threshold reached recall >= {min_recall:.0%}. "
        f"Using {th:.2f} (recall {best['recall']:.4f}, precision {best['precision']:.4f}) "
        f"to prioritize catching churners. Consider lowering the threshold if retention "
        f"capacity allows more outreach."
    )
    return th, just


def run_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
    save_path: str | Path | None = None,
    save_recommendation_path: str | Path | None = None,
) -> tuple[list[dict], float, str]:
    """
    Run threshold analysis, optionally save CSV and recommendation, return (rows, recommended_threshold, justification).
    """
    rows = threshold_analysis(y_true, y_prob, thresholds=thresholds)
    rec_th, justification = recommend_threshold_churn_prevention(rows)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(save_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    if save_recommendation_path and rows:
        rec_path = Path(save_recommendation_path)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rec_path, "w") as f:
            f.write(f"Recommended threshold (churn prevention): {rec_th:.2f}\n\n")
            f.write(justification)

    return rows, rec_th, justification


def run_evaluation(
    model_path: str | Path,
    preprocessor_path: str | Path,
    csv_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    threshold: float = 0.5,
) -> dict:
    """
    Load model and data, evaluate on test set, save ROC plot and metrics JSON.

    Returns the metrics dict (including ROC plot path if output_dir is set).
    """
    import joblib

    from preprocess import (
        TARGET_COL,
        load_and_clean,
        load_preprocessor,
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

    metrics = evaluate_churn_model(model, X_test, y_test, threshold=threshold)

    out_dir = Path(output_dir) if output_dir else project_root / "reports"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    metrics_dir = out_dir / "metrics"

    roc_path = figures_dir / "roc_curve.png"
    plot_roc_curve(metrics["y_test"], metrics["y_prob"], roc_path, title="Churn model – ROC curve (test set)")
    metrics["roc_plot_path"] = str(roc_path)

    fpr, tpr, _ = roc_curve(metrics["y_test"], metrics["y_prob"])
    cm = np.array(metrics["confusion_matrix"])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total else 0.0

    # Top-N capture and Decile Lift/Gain
    y_true = np.asarray(metrics["y_test"]).ravel()
    y_prob = np.asarray(metrics["y_prob"]).ravel()
    n_total = len(y_true)
    n_churners = int(np.sum(y_true))
    top_pcts = [5, 10, 15, 20, 25, 30, 40, 50]
    top_n_capture = []
    decile_gain = []
    if n_churners > 0 and n_total > 0:
        order = np.argsort(-y_prob)
        for pct in top_pcts:
            k = max(1, int(round(n_total * pct / 100)))
            captured = int(np.sum(y_true[order[:k]]))
            capture_pct = 100.0 * captured / n_churners
            top_n_capture.append({"top_pct": pct, "capture_pct": round(capture_pct, 2), "n_captured": captured})

        # Decile lift/gain: 10 deciles, % churn per decile, cumulative churn captured
        n_decile = max(1, n_total // 10)
        cum_churn = 0
        for d in range(1, 11):
            start = (d - 1) * n_decile
            end = n_total if d == 10 else d * n_decile
            idx = order[start:end]
            churn_in_decile = int(np.sum(y_true[idx]))
            cum_churn += churn_in_decile
            pct_decile = 100.0 * churn_in_decile / n_churners
            cum_pct = 100.0 * cum_churn / n_churners
            decile_gain.append({
                "decile": d,
                "pct_contacted": d * 10,
                "churn_in_decile": churn_in_decile,
                "pct_churn_decile": round(pct_decile, 2),
                "cumulative_pct_churn": round(cum_pct, 2),
            })

    report = {k: v for k, v in metrics.items() if k not in ("y_prob", "y_pred", "y_test")}
    report["fpr"] = np.asarray(fpr).tolist()
    report["tpr"] = np.asarray(tpr).tolist()
    report["accuracy"] = float(accuracy)
    report["top_n_capture"] = top_n_capture
    report["decile_gain"] = decile_gain
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "test_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    return metrics


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    preprocessor_path = project_root / "models" / "preprocessor" / "preprocessor.joblib"
    model_path = project_root / "models" / "best_model" / "churn_mlp.keras"
    output_dir = project_root / "reports"

    if not model_path.exists():
        print(f"Model not found: {model_path}. Run train.py first (e.g. py -3.12 src/train.py).")
        sys.exit(1)

    metrics = run_evaluation(
        model_path,
        preprocessor_path,
        csv_path,
        output_dir=output_dir,
        threshold=0.5,
    )

    print(format_metrics_report(metrics))
    print(f"\nROC curve saved to: {metrics['roc_plot_path']}")
    print("\n" + FP_FN_BUSINESS_NOTE)

    # Threshold analysis (0.2–0.8): Precision vs Recall trade-off
    thresh_rows, rec_thresh, rec_just = run_threshold_analysis(
        metrics["y_test"],
        metrics["y_prob"],
        thresholds=np.arange(0.20, 0.81, 0.05),
        save_path=output_dir / "metrics" / "threshold_analysis.csv",
        save_recommendation_path=output_dir / "metrics" / "threshold_recommendation.txt",
    )
    print("\n" + format_threshold_table(thresh_rows))
    print("\n=== Recommended threshold for churn prevention ===")
    print(f"Threshold: {rec_thresh:.2f}")
    print(f"Justification: {rec_just}")
