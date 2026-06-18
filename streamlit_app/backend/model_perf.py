"""Platzhalter-Modell mit echter Berechnungslogik (numpy).

Trainiert eine kleine logistische Regression auf den synthetischen Buchungen,
bewertet auf einem Holdout und liefert echte Metriken: Confusion-Matrix,
ROC-AUC, Precision/Recall/F1, Kalibrierung, Feature-Einfluss (standardisierte
Koeffizienten als SHAP-Proxy) und eine historische Storno-Heatmap.

Real-Modus später: model_card.json + Test-Predictions aus src laden.
"""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import dummy
from . import schema as S

_DATA_DIR = Path(__file__).resolve().parents[2] / "Data"
_STATE_JSON = _DATA_DIR / "model_state.json"
_DEFAULT_SEED = 7

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


def _read_state() -> dict:
    if _STATE_JSON.exists():
        try:
            return json.loads(_STATE_JSON.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {}


def _write_state(state: dict) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _STATE_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _ensure_state() -> dict:
    state = _read_state()
    if not state.get("seed"):
        state = {"seed": _DEFAULT_SEED, "trained_at": datetime.now().astimezone().isoformat()}
        _write_state(state)
    return state


def _build_xy(seed: int):
    df = dummy.generate(seed=seed, horizon_days=60)
    df = df[df[S.STATUS] == S.STATUS_CONFIRMED].reset_index(drop=True)
    feats = {}
    feats["_is_ota"] = df[S.CHANNEL].isin(_OTA).astype(float)
    feats["_is_flex"] = (df[S.RATE_CATEGORY] == "Flex").astype(float)
    feats["_is_nonref"] = (df[S.RATE_CATEGORY] == "Non-Ref").astype(float)
    feats["_is_business"] = (df[S.TRAVEL_PURPOSE] == "Business").astype(float)
    cols = []
    for _, key, _kind in _FEATURES:
        cols.append(feats[key].values if key in feats else df[key].astype(float).values)
    X = np.column_stack(cols)
    rng = np.random.RandomState(seed + 1)
    y = (rng.rand(len(df)) < df[S.CANCEL_PROBA].values).astype(int)
    return X, y, df


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit(X, y, iters=500, lr=0.5, l2=1.0):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Xb = np.column_stack([np.ones(len(Xs)), Xs])
    w = np.zeros(Xb.shape[1])
    n = len(y)
    for _ in range(iters):
        p = _sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) / n
        grad[1:] += (l2 / n) * w[1:]
        w -= lr * grad
    return w, mu, sd


def _predict(X, w, mu, sd):
    Xs = (X - mu) / sd
    return _sigmoid(np.column_stack([np.ones(len(Xs)), Xs]) @ w)


def _auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    allp = np.concatenate([pos, neg])
    order = allp.argsort()
    ranks = np.empty(len(allp))
    ranks[order] = np.arange(1, len(allp) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@lru_cache(maxsize=4)
def _compute(seed: int) -> dict:
    X, y, df = _build_xy(seed)
    rng = np.random.RandomState(seed + 2)
    idx = rng.permutation(len(y))
    cut = int(len(y) * 0.7)
    tr, te = idx[:cut], idx[cut:]
    w, mu, sd = _fit(X[tr], y[tr])
    p_te = _predict(X[te], w, mu, sd)
    y_te = y[te]

    importance = pd.DataFrame({
        "Feature": [f[0] for f in _FEATURES],
        "coef": w[1:],
    })
    importance["Einfluss"] = importance["coef"].abs()
    importance["Richtung"] = np.where(importance["coef"] >= 0, "↑ Storno", "↓ Storno")
    importance = importance.sort_values("Einfluss", ascending=False).reset_index(drop=True)

    bins = np.linspace(0, 1, 11)
    bi = np.clip(np.digitize(p_te, bins) - 1, 0, 9)
    calib = pd.DataFrame({"bin": bi, "p": p_te, "y": y_te}).groupby("bin").agg(
        predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size")).reset_index()

    hist = df.iloc[te].copy()
    hist["_y"] = y_te
    wk = ((hist[S.ARRIVAL_DATE] - hist[S.ARRIVAL_DATE].min()).dt.days // 7).astype(int)
    hist["_kw"] = "W" + (wk + 1).astype(str)
    hmat = hist.pivot_table(index=S.HOTEL_CODE, columns="_kw", values="_y", aggfunc="mean")
    order = sorted(hmat.columns, key=lambda c: int(c[1:]))
    hmat = hmat[order]

    return {
        "seed": seed,
        "w": w, "mu": mu, "sd": sd,
        "p_te": p_te, "y_te": y_te,
        "auc": _auc(y_te, p_te),
        "base_rate": float(y.mean()),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "importance": importance,
        "calibration": calib,
        "history": hmat,
    }


def _state_seed() -> int:
    return int(_ensure_state()["seed"])


def model_card() -> dict:
    st = _ensure_state()
    art = _compute(_state_seed())
    return {
        "name": "Cancellation-Likelihood (Platzhalter)",
        "type": "Logistische Regression (numpy, L2)",
        "trained_at": st.get("trained_at", "?"),
        "n_train": art["n_train"], "n_test": art["n_test"],
        "n_features": len(_FEATURES),
        "base_rate": art["base_rate"],
        "auc": art["auc"],
        "low_thr": S.LOW_THR, "high_thr": S.HIGH_THR,
    }


def performance(threshold: float = 0.5) -> dict:
    art = _compute(_state_seed())
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
    return {
        "confusion": np.array([[tn, fp], [fn, tp]]),
        "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
        "auc": art["auc"], "threshold": threshold, "n": int(len(y)),
    }


def feature_importance() -> pd.DataFrame:
    return _compute(_state_seed())["importance"]


def calibration() -> pd.DataFrame:
    return _compute(_state_seed())["calibration"]


def history_matrix(labels: dict | None = None) -> pd.DataFrame:
    h = _compute(_state_seed())["history"].copy()
    if labels:
        h.index = [labels.get(c, c) for c in h.index]
    return h


def retrain() -> dict:
    import time
    _compute.cache_clear()
    _write_state({"seed": int(time.time()) % 1_000_000,
                  "trained_at": datetime.now().astimezone().isoformat()})
    return model_card()
