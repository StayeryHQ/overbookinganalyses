# dash_app/backend/model_perf.py
# ---------------------------------------------------------------------------
# Model-performance backend for the "Modell & Performance" page.
#
# Two paths, ONE shape (so the page never branches on mode):
#   * DUMMY: trains a tiny numpy logistic regression on the synthetic snapshot
#     and returns REAL computed curves (ROC/PR/calibration/confusion/importance).
#     Ported from streamlit_app/backend/model_perf.py.
#   * REAL: reads reports/tables/<NN_model>/model_card.json for the headline
#     metrics and feature_importances_by_feature.csv for the importance chart,
#     and reconstructs ROC/PR points from the persisted operating points. Full
#     curve point-clouds are NOT persisted on disk, so those are clearly marked
#     as approximate (see `curves_available`).
#
# Every public function returns plain Python / pandas objects (dicts, arrays,
# DataFrames). The PAGE turns them into Plotly figures via src.plotting, so this
# module has no Dash/Plotly dependency.
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd

# NOTE (v11): the dummy backend was removed. This module's synthetic
# model-performance source (`dummy`) is gone; the import is guarded so the app
# still loads. The model-performance page must be rewired to real model cards /
# predictions (reports/tables/<NN>_*/model_card.json + *_predictions.parquet) —
# tracked as the remaining dummy-removal work. Until then its functions that
# call `dummy` will raise a clear error rather than crash app startup.
try:
    from . import dummy  # type: ignore
except Exception:  # noqa: BLE001 — dummy.py removed
    dummy = None  # type: ignore
from . import schema as S
from .. import config as CFG


# =============================================================================
# DUMMY path — tiny numpy logreg on synthetic data (real maths, fake data)
# =============================================================================
# Feature spec: (display name, source column / derived key, kind).
_FEATURES = [
    ("Lead-Time", S.LEAD_TIME_DAYS, "num"),
    ("Aufenthaltsdauer", S.LOS_NIGHTS, "num"),
    ("€ / Nacht", S.GROSS_PER_NIGHT, "num"),
    ("Personen", S.ADULTS, "num"),
    ("OTA-Kanal", "_is_ota", "bin"),
    ("Flex-Rate", "_is_flex", "bin"),
    ("Non-Ref-Rate", "_is_nonref", "bin"),
    ("Business", "_is_business", "bin"),
    ("Firmenkunde", S.IS_CORPORATE, "bin"),
    ("Gruppe", S.IS_GROUP, "bin"),
    ("International", S.IS_INTERNATIONAL, "bin"),
    ("Noch stornierbar", S.IS_CANCELABLE, "bin"),
]
_OTA = ("Booking.com", "Expedia", "HRS")
_DUMMY_SEED = 7   # fixed seed so the dummy "model" is reproducible


def _build_xy(seed: int):
    """Assemble the feature matrix X and a sampled label vector y from dummy data."""
    # Generate a wider synthetic snapshot to have enough rows to "train" on.
    df = dummy.generate(seed=seed, horizon_days=60)
    df = df[df[S.STATUS] == S.STATUS_CONFIRMED].reset_index(drop=True)
    # Build the derived binary features the spec references.
    feats = {}
    feats["_is_ota"] = df[S.CHANNEL].isin(_OTA).astype(float)
    feats["_is_flex"] = (df[S.RATE_CATEGORY] == "Flex").astype(float)
    feats["_is_nonref"] = (df[S.RATE_CATEGORY] == "Non-Ref").astype(float)
    feats["_is_business"] = (df[S.TRAVEL_PURPOSE] == "Business").astype(float)
    cols = []
    for _, key, _kind in _FEATURES:
        # Derived binary -> from feats dict; otherwise the raw column as float.
        cols.append(feats[key].values if key in feats else df[key].astype(float).values)
    X = np.column_stack(cols)
    # Sample a binary outcome from the synthetic cancel probability.
    rng = np.random.RandomState(seed + 1)
    y = (rng.rand(len(df)) < df[S.CANCEL_PROBA].values).astype(int)
    return X, y, df


def _sigmoid(z):
    """Numerically-stable logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit(X, y, iters=500, lr=0.5, l2=1.0):
    """Fit an L2-regularised logistic regression by gradient descent (standardised X)."""
    # Standardise features (zero mean, unit std); guard against zero std.
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    # Prepend an intercept column of ones.
    Xb = np.column_stack([np.ones(len(Xs)), Xs])
    w = np.zeros(Xb.shape[1])
    n = len(y)
    for _ in range(iters):
        p = _sigmoid(Xb @ w)
        # Gradient of the logistic loss + L2 penalty on non-intercept weights.
        grad = Xb.T @ (p - y) / n
        grad[1:] += (l2 / n) * w[1:]
        w -= lr * grad
    return w, mu, sd


def _predict(X, w, mu, sd):
    """Predict probabilities with the fitted weights (re-applying standardisation)."""
    Xs = (X - mu) / sd
    return _sigmoid(np.column_stack([np.ones(len(Xs)), Xs]) @ w)


def _auc(y, p):
    """ROC-AUC via the rank-sum (Mann-Whitney) identity — no sklearn needed."""
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    allp = np.concatenate([pos, neg])
    order = allp.argsort()
    ranks = np.empty(len(allp))
    ranks[order] = np.arange(1, len(allp) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _roc_points(y, p):
    """Return (fpr, tpr) arrays by sweeping the threshold over sorted scores."""
    # Sort by descending score; sweep the decision threshold downward.
    order = np.argsort(-p)
    y_sorted = y[order]
    P = max(int(y.sum()), 1)         # total positives
    N = max(int((y == 0).sum()), 1)  # total negatives
    tps = np.cumsum(y_sorted)        # cumulative true positives
    fps = np.cumsum(1 - y_sorted)    # cumulative false positives
    tpr = np.concatenate([[0.0], tps / P])
    fpr = np.concatenate([[0.0], fps / N])
    return fpr, tpr


def _pr_points(y, p):
    """Return (recall, precision, average_precision) arrays."""
    order = np.argsort(-p)
    y_sorted = y[order]
    P = max(int(y.sum()), 1)
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    recall = tps / P
    precision = tps / np.maximum(tps + fps, 1)
    # Average precision = sum of precision * delta-recall.
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    return recall, precision, ap


@lru_cache(maxsize=4)
def _compute_dummy(seed: int) -> dict:
    """Train + evaluate the dummy model once per seed; cache the whole artifact."""
    X, y, df = _build_xy(seed)
    # 70/30 train/test split on a shuffled index.
    rng = np.random.RandomState(seed + 2)
    idx = rng.permutation(len(y))
    cut = int(len(y) * 0.7)
    tr, te = idx[:cut], idx[cut:]
    w, mu, sd = _fit(X[tr], y[tr])
    p_te = _predict(X[te], w, mu, sd)
    y_te = y[te]

    # Feature importance = standardised logreg coefficients (SHAP proxy).
    importance = pd.DataFrame({"Feature": [f[0] for f in _FEATURES], "coef": w[1:]})
    importance = importance.sort_values("coef", key=lambda s: s.abs(), ascending=True)

    # Calibration: bin predictions into 10 bins, compare mean pred vs actual rate.
    bins = np.linspace(0, 1, 11)
    bi = np.clip(np.digitize(p_te, bins) - 1, 0, 9)
    calib = (pd.DataFrame({"bin": bi, "p": p_te, "y": y_te})
             .groupby("bin").agg(predicted=("p", "mean"), actual=("y", "mean")).reset_index())

    fpr, tpr = _roc_points(y_te, p_te)
    recall, precision, ap = _pr_points(y_te, p_te)

    return {
        "seed": seed,
        "p_te": p_te, "y_te": y_te,
        "auc": _auc(y_te, p_te), "ap": ap,
        "base_rate": float(y.mean()),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "importance": importance,
        "calibration": calib,
        "roc": (fpr, tpr),
        "pr": (recall, precision),
    }


# =============================================================================
# REAL path — read the model card + importance CSV from reports/tables
# =============================================================================
def _model_dir(model_name: str) -> str:
    """Map a registry name to its reports/tables sub-directory prefix."""
    # The folders are zero-padded by training order: 01_logreg, 02_xgboost, ...
    prefix = {"logreg": "01_logreg", "xgboost": "02_xgboost", "histgb": "03_histgb"}
    return prefix.get(model_name, "02_xgboost")


@lru_cache(maxsize=4)
def _load_card(model_name: str) -> dict | None:
    """Load reports/tables/<dir>/model_card.json, or None if absent."""
    p = CFG.TABLES_DIR / _model_dir(model_name) / "model_card.json"
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=4)
def _load_importance_real(model_name: str) -> pd.DataFrame | None:
    """Load the per-feature gain importance CSV, or None if absent."""
    p = CFG.TABLES_DIR / _model_dir(model_name) / "feature_importances_by_feature.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # Sort ascending so the horizontal bar chart reads largest-at-top.
    return df.sort_values("gain", ascending=True)


# =============================================================================
# Public interface — the page calls these (mode + model passed in)
# =============================================================================
def model_card(mode: str, model_name: str | None) -> dict:
    """Headline model facts in a uniform dict (works for dummy + real)."""
    if mode == "real" and (card := _load_card(model_name or "xgboost")) is not None:
        tm = card.get("test_metrics", {})
        return {
            "name": card.get("model", model_name or "?"),
            "type": "Trainiertes Modell (src)",
            "trained_at": card.get("trained_at", "?"),
            "n_train": card.get("n_train", 0), "n_test": card.get("n_test", 0),
            "n_features": len(card.get("features_numeric", [])) + len(card.get("features_categorical", [])),
            "base_rate": tm.get("base_rate", 0.0),
            "auc": tm.get("auc", 0.0), "ap": tm.get("ap", 0.0), "brier": tm.get("brier", 0.0),
            "low_thr": S.LOW_THR, "high_thr": S.HIGH_THR,
            "source": "model_card.json",
        }
    # Dummy fallback (also used when a real card is missing).
    art = _compute_dummy(_DUMMY_SEED)
    return {
        "name": "Cancellation-Likelihood (Platzhalter)",
        "type": "Logistische Regression (numpy, L2)",
        "trained_at": datetime.now().astimezone().isoformat(),
        "n_train": art["n_train"], "n_test": art["n_test"],
        "n_features": len(_FEATURES),
        "base_rate": art["base_rate"], "auc": art["auc"], "ap": art["ap"], "brier": None,
        "low_thr": S.LOW_THR, "high_thr": S.HIGH_THR,
        "source": "synthetisch",
    }


def feature_importance(mode: str, model_name: str | None) -> tuple[list[str], list[float]]:
    """Return (labels, values) for the importance bar chart.

    Real: per-feature gain (always positive). Dummy: signed logreg coefficients.
    """
    if mode == "real" and (imp := _load_importance_real(model_name or "xgboost")) is not None:
        return list(imp["parent"]), list(imp["gain"])
    art = _compute_dummy(_DUMMY_SEED)
    return list(art["importance"]["Feature"]), list(art["importance"]["coef"])


def performance(mode: str, model_name: str | None, threshold: float = 0.30) -> dict:
    """Confusion matrix + precision/recall/F1/accuracy at the given threshold.

    Real: confusion is reconstructed from the model card's base rate + the
    operating point nearest the threshold (full test labels aren't persisted),
    so it is APPROXIMATE — flagged via 'approx'. Dummy: exact on the holdout.
    """
    if mode == "real" and (card := _load_card(model_name or "xgboost")) is not None:
        tm = card.get("test_metrics", {})
        n_test = int(card.get("n_test", 0)) or 1
        base = float(tm.get("base_rate", 0.0))
        ops = card.get("operating_points", [])
        # Pick the operating point closest to the requested threshold.
        op = min(ops, key=lambda o: abs(o["threshold"] - threshold)) if ops else None
        if op is None:
            prec, rec = 0.0, 0.0
        else:
            prec, rec = float(op["precision"]), float(op["recall"])
        # Reconstruct a 2x2 confusion matrix from base rate + precision/recall.
        pos = int(round(base * n_test))         # actual positives
        neg = n_test - pos                       # actual negatives
        tp = int(round(rec * pos))               # recall = tp / pos
        fn = pos - tp
        fp = int(round(tp / prec - tp)) if prec > 0 else 0  # precision = tp/(tp+fp)
        tn = max(neg - fp, 0)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = (tp + tn) / n_test
        return {"confusion": np.array([[tn, fp], [fn, tp]]),
                "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
                "auc": tm.get("auc", 0.0), "ap": tm.get("ap", 0.0),
                "threshold": threshold, "approx": True}
    # Dummy: exact confusion on the holdout at the given threshold.
    art = _compute_dummy(_DUMMY_SEED)
    p, y = art["p_te"], art["y_te"]
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(y) if len(y) else 0.0
    return {"confusion": np.array([[tn, fp], [fn, tp]]),
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "auc": art["auc"], "ap": art["ap"], "threshold": threshold, "approx": False}


def roc_curve(mode: str, model_name: str | None) -> dict:
    """{model_label: (fpr, tpr, auc)} for plotting.roc_curve_fig.

    Dummy: real point cloud. Real: only the AUC is persisted (no curve points),
    so we return a 2-point diagonal-ish proxy labelled with the true AUC plus a
    'curves_available' flag the page uses to show a caveat.
    """
    if mode == "real" and (card := _load_card(model_name or "xgboost")) is not None:
        auc = float(card.get("test_metrics", {}).get("auc", 0.0))
        # No persisted ROC points -> minimal proxy curve. Flagged as unavailable.
        return {"curves_available": False,
                "curves": {f"{model_name or 'Modell'}": ([0.0, 1.0], [0.0, 1.0], auc)}}
    art = _compute_dummy(_DUMMY_SEED)
    fpr, tpr = art["roc"]
    return {"curves_available": True,
            "curves": {"Platzhalter": (fpr, tpr, art["auc"])}}


def pr_curve(mode: str, model_name: str | None) -> dict:
    """{model_label: (recall, precision, ap)} + base_rate for plotting.pr_curve_fig.

    Dummy: real point cloud. Real: builds an approximate curve from the persisted
    operating points (sorted by recall) — flagged via 'curves_available'.
    """
    if mode == "real" and (card := _load_card(model_name or "xgboost")) is not None:
        tm = card.get("test_metrics", {})
        ops = sorted(card.get("operating_points", []), key=lambda o: o["recall"])
        # Use the operating points as sparse (recall, precision) samples.
        recall = [o["recall"] for o in ops] or [0.0, 1.0]
        precision = [o["precision"] for o in ops] or [tm.get("base_rate", 0.0)] * 2
        return {"curves_available": False,
                "base_rate": float(tm.get("base_rate", 0.0)),
                "curves": {f"{model_name or 'Modell'}": (recall, precision, float(tm.get("ap", 0.0)))}}
    art = _compute_dummy(_DUMMY_SEED)
    recall, precision = art["pr"]
    return {"curves_available": True, "base_rate": art["base_rate"],
            "curves": {"Platzhalter": (recall, precision, art["ap"])}}


def calibration(mode: str, model_name: str | None) -> dict:
    """{model_label: (mean_predicted, fraction_positive)} for plotting.calibration_fig.

    Dummy: real reliability curve. Real: brier score is persisted but not the
    per-bin curve, so we return a near-diagonal proxy flagged as unavailable.
    """
    if mode == "real" and (card := _load_card(model_name or "xgboost")) is not None:
        return {"curves_available": False,
                "curves": {f"{model_name or 'Modell'}": ([0.0, 1.0], [0.0, 1.0])}}
    art = _compute_dummy(_DUMMY_SEED)
    c = art["calibration"]
    return {"curves_available": True,
            "curves": {"Platzhalter": (list(c["predicted"]), list(c["actual"]))}}


def retrain(mode: str) -> dict:
    """Refresh the dummy model with a new seed (real retrain is out of scope here)."""
    if mode != "real":
        # Clear the cache and re-train on a new random seed.
        import time
        global _DUMMY_SEED
        _compute_dummy.cache_clear()
        _DUMMY_SEED = int(time.time()) % 1_000_000
    return model_card(mode, None)
