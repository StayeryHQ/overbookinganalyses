# ---------------------------------------------------------------------------
# src/diagnostics.py
# Reusable, model-agnostic SCORING DIAGNOSTICS for the cancellation models.
#
# Everything here works on pooled out-of-sample predictions (y_true, y_prob) that
# the model notebooks already compute via the walk-forward helpers  so the notebook
# cells just hand those arrays in and get the full "standard" battery:
#
#   * baseline (mean-rate) reference        -> is the model beating "always guess p0"?
#   * headline metrics: ROC-AUC, PR-AP, Brier, log-loss
#   * confusion matrix + precision/recall/F1/accuracy/specificity at a threshold
#   * ROC curve, precision-recall curve
#   * precision / recall / F1 vs. threshold sweep
#   * SHAP (global bar of mean|value| + native beeswarm) on the tree model
#
# Design notes
#   * Heavy deps (sklearn / plotly / shap) are imported lazily inside functions, so
#     `import src.diagnostics` stays cheap and never hard-requires shap.
#   * Plotly-first (project standard). SHAP's beeswarm is inherently matplotlib 
#     that's the documented "a library genuinely can't do it in Plotly" exception.
#   * SHAP helpers are defensive: if a model can't be unwrapped/explained they warn
#     and return None instead of blowing up a long notebook run.
# ---------------------------------------------------------------------------

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


# ---- brand colours (fall back to safe defaults if the config isn't available) ----
def _colors() -> dict[str, str]:
    try:
        from .utils import load_brand_config
        b = load_brand_config()
        return {**b["colors"]["core"], **b["colors"]["supporting"]}
    except Exception:  # noqa: BLE001
        return {"blue": "#1E4BA1", "orange": "#EB6E14", "green": "#08A064",
                "red": "#E62828", "yellow": "#FFE650", "purple": "#6E32C8",
                "pink": "#F0B4C8", "black": "#000000", "white": "#FFFFFF"}


def _prep(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    return y, p


# =============================================================================
# Metrics
# =============================================================================
def baseline_metrics(y_true) -> dict:
    """The mean-rate BASELINE: predict the base rate p0 for every booking. It has no
    ranking power (ROC-AUC = 0.5), its AP equals the prevalence p0, and its Brier is
    the irreducible p0(1-p0). This is the bar a useful model must clear."""
    y = np.asarray(y_true).astype(int)
    p0 = float(y.mean())
    return {"model": "baseline (mean rate)", "auc": 0.5, "ap": p0,
            "brier": p0 * (1 - p0), "log_loss": _safe_logloss(y, np.full_like(y, p0, dtype=float)),
            "base_rate": p0}


def _safe_logloss(y, p) -> float:
    from sklearn.metrics import log_loss
    p = np.clip(p, 1e-9, 1 - 1e-9)
    try:
        return float(log_loss(y, p, labels=[0, 1]))
    except Exception:  # noqa: BLE001
        return float("nan")


def headline_metrics(y_true, y_prob, model_name: str = "model") -> dict:
    """Threshold-free headline metrics: ROC-AUC, PR-AP, Brier, log-loss, base rate."""
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y, p = _prep(y_true, y_prob)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    return {"model": model_name, "auc": float(auc),
            "ap": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": _safe_logloss(y, p), "base_rate": float(y.mean())}


def confusion_at(y_true, y_prob, threshold: float) -> dict:
    """Confusion counts + point metrics at a decision threshold."""
    y, p = _prep(y_true, y_prob)
    pred = (p >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1))); fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1))); tn = int(np.sum((pred == 0) & (y == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "specificity": spec, "f1": f1, "accuracy": acc}


def threshold_sweep(y_true, y_prob, n: int = 91) -> pd.DataFrame:
    """precision / recall / F1 across a grid of thresholds (for the sweep plot and to
    read off the F1-optimal point)."""
    y, p = _prep(y_true, y_prob)
    grid = np.linspace(0.02, 0.98, n)
    rows = [confusion_at(y, p, t) for t in grid]
    return pd.DataFrame(rows)[["threshold", "precision", "recall", "f1", "accuracy"]]


def metrics_table(y_true, y_prob, model_name: str = "model",
                  threshold: float | None = None) -> pd.DataFrame:
    """Side-by-side table: the model vs. the mean-rate baseline, plus the at-threshold
    confusion metrics. `threshold` None -> 0.5. Returns a tidy DataFrame."""
    thr = 0.5 if threshold is None else float(threshold)
    hm = headline_metrics(y_true, y_prob, model_name)
    bm = baseline_metrics(y_true)
    cm = confusion_at(y_true, y_prob, thr)
    rows = [
        {"metric": "ROC-AUC", model_name: round(hm["auc"], 4), "baseline": round(bm["auc"], 4)},
        {"metric": "PR-AP", model_name: round(hm["ap"], 4), "baseline": round(bm["ap"], 4)},
        {"metric": "Brier", model_name: round(hm["brier"], 4), "baseline": round(bm["brier"], 4)},
        {"metric": "log-loss", model_name: round(hm["log_loss"], 4), "baseline": round(bm["log_loss"], 4)},
        {"metric": f"precision @ {thr:.2f}", model_name: round(cm["precision"], 4), "baseline": ""},
        {"metric": f"recall @ {thr:.2f}", model_name: round(cm["recall"], 4), "baseline": ""},
        {"metric": f"F1 @ {thr:.2f}", model_name: round(cm["f1"], 4), "baseline": ""},
        {"metric": f"accuracy @ {thr:.2f}", model_name: round(cm["accuracy"], 4), "baseline": ""},
        {"metric": "base rate (p0)", model_name: round(hm["base_rate"], 4), "baseline": round(bm["base_rate"], 4)},
    ]
    return pd.DataFrame(rows)


def print_report(y_true, y_prob, model_name: str = "model",
                 threshold: float | None = None) -> pd.DataFrame:
    """Print the metrics table (model vs baseline + confusion metrics) and return it."""
    tbl = metrics_table(y_true, y_prob, model_name, threshold)
    thr = 0.5 if threshold is None else float(threshold)
    cm = confusion_at(y_true, y_prob, thr)
    print(f"=== Scoring diagnostics: {model_name}  (n={len(np.asarray(y_true)):,}, "
          f"threshold={thr:.3f}) ===")
    print(tbl.to_string(index=False))
    print(f"\nConfusion @ {thr:.3f}:  TP={cm['tp']}  FP={cm['fp']}  FN={cm['fn']}  TN={cm['tn']}")
    return tbl


# =============================================================================
# Plotly figures
# =============================================================================
def _brand_fig(fig):
    try:
        from dash_app.theme import brand_figure
        return brand_figure(fig)
    except Exception:  # noqa: BLE001  notebooks may run without the dash_app package
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        return fig


def roc_curve_fig(y_true, y_prob, model_name: str = "model"):
    from sklearn.metrics import roc_curve, roc_auc_score
    import plotly.graph_objects as go
    y, p = _prep(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    c = _colors()
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{model_name} (AUC={auc:.3f})",
                    line=dict(color=c["blue"], width=2))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                    line=dict(color="grey", dash="dash"))
    fig.update_layout(title=f"ROC curve  {model_name}", xaxis_title="False positive rate",
                      yaxis_title="True positive rate", height=360)
    return _brand_fig(fig)


def pr_curve_fig(y_true, y_prob, model_name: str = "model"):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import plotly.graph_objects as go
    y, p = _prep(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    p0 = float(y.mean())
    c = _colors()
    fig = go.Figure()
    fig.add_scatter(x=rec, y=prec, mode="lines", name=f"{model_name} (AP={ap:.3f})",
                    line=dict(color=c["green"], width=2))
    fig.add_hline(y=p0, line=dict(color="grey", dash="dash"),
                  annotation_text=f"baseline (p0={p0:.3f})")
    fig.update_layout(title=f"Precision–Recall  {model_name}", xaxis_title="Recall",
                      yaxis_title="Precision", height=360)
    return _brand_fig(fig)


def confusion_fig(y_true, y_prob, threshold: float, model_name: str = "model"):
    import plotly.graph_objects as go
    cm = confusion_at(y_true, y_prob, threshold)
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    labels = [[f"TN<br>{cm['tn']}", f"FP<br>{cm['fp']}"], [f"FN<br>{cm['fn']}", f"TP<br>{cm['tp']}"]]
    c = _colors()
    fig = go.Figure(go.Heatmap(
        z=z, x=["pred: keep", "pred: cancel"], y=["actual: keep", "actual: cancel"],
        text=labels, texttemplate="%{text}", colorscale=[[0, "#FFFFFF"], [1, c["yellow"]]],
        showscale=False))
    fig.update_layout(title=f"Confusion @ {threshold:.2f}  {model_name}",
                      height=340, yaxis_autorange="reversed")
    return _brand_fig(fig)


def threshold_sweep_fig(y_true, y_prob, operating_threshold: float | None = None):
    import plotly.graph_objects as go
    sweep = threshold_sweep(y_true, y_prob)
    c = _colors()
    fig = go.Figure()
    fig.add_scatter(x=sweep["threshold"], y=sweep["precision"], name="precision",
                    line=dict(color=c["blue"]))
    fig.add_scatter(x=sweep["threshold"], y=sweep["recall"], name="recall",
                    line=dict(color=c["orange"]))
    fig.add_scatter(x=sweep["threshold"], y=sweep["f1"], name="F1",
                    line=dict(color=c["green"], width=3))
    if operating_threshold is not None:
        fig.add_vline(x=float(operating_threshold), line=dict(color="grey", dash="dash"),
                      annotation_text=f"operating {operating_threshold:.2f}")
    fig.update_layout(title="Precision / recall / F1 vs. threshold",
                      xaxis_title="threshold", yaxis_title="score", height=360)
    return _brand_fig(fig)


# =============================================================================
# SHAP
# =============================================================================
def tree_estimator(fitted: Any):
    """Unwrap a fitted object down to the underlying TREE estimator SHAP can explain.

    Handles sklearn Pipeline (uses the final step) and CalibratedClassifierCV (returns
    the calibrated base estimator). Returns the object unchanged if already a bare
    estimator. Returns None if nothing tree-like can be found.
    """
    est = fitted
    try:
        # sklearn Pipeline -> final step
        if hasattr(est, "named_steps"):
            est = list(est.named_steps.values())[-1]
        # CalibratedClassifierCV -> a fitted base estimator
        if est.__class__.__name__ == "CalibratedClassifierCV":
            if getattr(est, "calibrated_classifiers_", None):
                cc = est.calibrated_classifiers_[0]
                est = getattr(cc, "estimator", getattr(cc, "base_estimator", est))
        return est
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"tree_estimator: could not unwrap ({e})")
        return None


def pipeline_design_matrix(pipeline: Any, X: pd.DataFrame):
    """Transform raw features through a fitted Pipeline's preprocessing step, returning
    (design_df, feature_names). For a bare estimator, returns X unchanged."""
    if not hasattr(pipeline, "named_steps"):
        return X, list(X.columns)
    steps = list(pipeline.named_steps.items())
    prep = steps[0][1]
    design = prep.transform(X)
    try:
        names = list(prep.get_feature_names_out())
    except Exception:  # noqa: BLE001
        names = [f"f{i}" for i in range(design.shape[1])]
    if hasattr(design, "toarray"):
        design = design.toarray()
    return pd.DataFrame(design, columns=names, index=X.index), names


def shap_values(tree_model: Any, X_design: pd.DataFrame, max_samples: int = 2000):
    """Compute SHAP values for a tree model on X_design. Returns a shap.Explanation
    (or None on failure). Samples down to `max_samples` rows for speed."""
    try:
        import shap
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"shap not available ({e}); skipping SHAP.")
        return None
    if tree_model is None:
        warnings.warn("shap_values: no tree model; skipping SHAP.")
        return None
    Xs = X_design.sample(n=min(max_samples, len(X_design)), random_state=0) \
        if len(X_design) > max_samples else X_design
    try:
        explainer = shap.TreeExplainer(tree_model)
        return explainer(Xs)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"shap_values: TreeExplainer failed ({e}); skipping SHAP.")
        return None


def shap_bar_fig(tree_model: Any, X_design: pd.DataFrame, feature_names=None,
                 title: str = "SHAP (mean |value|)", max_samples: int = 2000,
                 top: int = 20):
    """Plotly bar of global feature importance = mean |SHAP value| per feature.
    Returns a plotly figure, or None if SHAP is unavailable."""
    import plotly.graph_objects as go
    sv = shap_values(tree_model, X_design, max_samples=max_samples)
    if sv is None:
        return None
    vals = sv.values
    if getattr(vals, "ndim", 2) == 3:      # (n, features, classes) -> positive class
        vals = vals[:, :, -1]
    mean_abs = np.abs(vals).mean(axis=0)
    names = list(feature_names) if feature_names is not None else \
        (list(sv.feature_names) if getattr(sv, "feature_names", None) else
         [f"f{i}" for i in range(len(mean_abs))])
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs}) \
        .sort_values("mean_abs_shap", ascending=False).head(top)
    fig = go.Figure(go.Bar(x=imp["mean_abs_shap"][::-1], y=imp["feature"][::-1],
                           orientation="h", marker_color=_colors()["purple"]))
    fig.update_layout(title=title, xaxis_title="mean |SHAP value|", height=460)
    return _brand_fig(fig)


def shap_beeswarm(tree_model: Any, X_design: pd.DataFrame, feature_names=None,
                  max_samples: int = 2000, max_display: int = 15) -> None:
    """SHAP's native beeswarm (matplotlib  the sanctioned non-Plotly exception).
    No-op with a warning if SHAP is unavailable."""
    try:
        import shap
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"shap/matplotlib not available ({e}); skipping beeswarm.")
        return
    sv = shap_values(tree_model, X_design, max_samples=max_samples)
    if sv is None:
        return
    try:
        if feature_names is not None:
            sv.feature_names = list(feature_names)
        shap.plots.beeswarm(sv, max_display=max_display, show=True)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"shap beeswarm failed ({e}).")
