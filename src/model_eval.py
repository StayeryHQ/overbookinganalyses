# ---------------------------------------------------------------------------
# src/model_eval.py
# Leak-free, PER-MODEL evaluation predictions for the XAI / Model-Performance page.
#
# Why this module exists
# ----------------------
# The page must compare each model to the naive historical-average baseline on the
# SAME estimand — P(cancel by arrival), scored at the decision horizon d = min(lead, H) —
# and break the result down per location. The four models emit different things natively
# (static classifiers -> predict_proba; hazard -> survival product), but src.scoring /
# src.hazard already collapse both to that one scalar. This module runs the decision-time
# walk-forward (src.walkforward.make_folds — the SINGLE eval regime for every model) once
# per model, refitting LEAK-FREE per fold with the FROZEN card hyperparameters (calibrated),
# and pools the out-of-time predictions into ONE artifact per model:
#
#     Data/model_eval_<model>.parquet
#       columns: fold, property_name, days_until_arrival,
#                y_true, y_prob, base_global, base_property
#     Data/model_eval_<model>_folds.parquet   (per-fold train-vs-test metrics for 4.4)
#       columns: fold, train_auc, test_auc, train_ap, test_ap, train_brier, test_brier,
#                n_train, n_test
#
# base_global / base_property are the LEAK-FREE naive baseline: each test booking's
# baseline "prediction" is the TRAIN base rate of its fold (overall, and for its own
# property). The historical-average baseline is a CONSTANT predictor, so it is only
# meaningful for calibration / Brier / log-loss / cost — NEVER for ROC-AUC (0.5 by
# construction). The page enforces that distinction (baseline shown at 4.2/4.3/4.9, not 4.1).
#
# DRY note: modeling logic is NOT duplicated here. The fold loop reuses the exact tested
# building blocks — training.build_pipeline / _card_hp / _family_feature_lists and
# hazard.fit_hazard / survival_cancel_proba — mirroring training.bakeoff_walk_forward.
#
# Compute: one artifact per model, cached. Static models are cheap; the HAZARD refit
# (person-period RandomizedSearch per fold) is the slow one — pre-warm it offline
# (`python main.py eval --model hazard`) so the running app only ever READS the parquet.
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
from typing import Final

import numpy as np
import pandas as pd

from . import walkforward as wf
from .paths import data_dir

SEED: Final[int] = 42
HORIZON_DAYS: Final[int] = 14            # project-wide decision horizon (matches WINDOW_DAYS)
DEFAULT_N_FOLDS: Final[int] = 6          # matches training.walk_forward_eval's default
STEP_DAYS: Final[int] = 14
BASELINE_MIN_N: Final[int] = 100         # min train bookings for a per-property baseline rate

# Every model the page can evaluate. Names mirror src.scoring.MODEL_REGISTRY.
EVAL_MODELS: Final[tuple[str, ...]] = ("hazard", "xgboost", "histgb", "logreg")

EVAL_COLS: Final[tuple[str, ...]] = (
    "fold", "property_name", "days_until_arrival",
    "y_true", "y_prob", "base_global", "base_property",
)


# =============================================================================
# Cache locations
# =============================================================================
def eval_cache_path(model_name: str):
    """Path of a model's pooled evaluation-predictions parquet."""
    return data_dir() / f"model_eval_{model_name}.parquet"


def fold_metrics_path(model_name: str):
    """Path of a model's per-fold train-vs-test metrics parquet (for 4.4)."""
    return data_dir() / f"model_eval_{model_name}_folds.parquet"


def eval_meta_path(model_name: str):
    """Path of the sidecar provenance JSON for a model's eval artifact."""
    return data_dir() / f"model_eval_{model_name}.json"


def eval_available(model_name: str) -> bool:
    return eval_cache_path(model_name).exists()


# =============================================================================
# Leak-free naive baseline (pure — unit-testable without sklearn)
# =============================================================================
def property_baseline(y_train, prop_train, prop_test,
                      *, min_n: int = BASELINE_MIN_N) -> tuple[float, np.ndarray]:
    """(global_train_rate, per_test_property_rate).

    The per-property rate is the TRAIN cancel rate for that property; properties with
    fewer than `min_n` train bookings (or unseen at train time) fall back to the global
    train rate. Everything is derived from TRAIN only, so it is leak-free by construction.
    Pure numpy/pandas — testable without the model stack.
    """
    y = np.asarray(y_train, dtype=float)
    global_rate = float(y.mean()) if y.size else float("nan")
    s = pd.Series(y, index=pd.Index(np.asarray(prop_train), name="property_name"))
    grp = s.groupby(level=0)
    rate = grp.mean().where(grp.size() >= min_n, global_rate)
    mapped = (pd.Series(np.asarray(prop_test), name="property_name")
              .map(rate).fillna(global_rate).to_numpy(dtype=float))
    return global_rate, mapped


# =============================================================================
# Metric + scoring helpers
# =============================================================================
def _fold_metrics(y_true, y_prob) -> dict:
    """AUC / AP / Brier on one split; NaN-safe for single-class slices."""
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    out = {"auc": float("nan"), "ap": float("nan"), "brier": float("nan"), "n": int(len(y))}
    if len(y) == 0:
        return out
    try:
        if np.unique(y).size > 1:
            out["auc"] = float(roc_auc_score(y, p))
            out["ap"] = float(average_precision_score(y, p))
        out["brier"] = float(brier_score_loss(y, p))
    except Exception:  # noqa: BLE001
        pass
    return out


def _decision_horizon(rows: pd.DataFrame, horizon_days: int) -> np.ndarray:
    """Per-booking d = min(lead, H), clipped to >= 1 day."""
    arr = pd.to_datetime(rows["arrival"], utc=True, errors="coerce")
    cre = pd.to_datetime(rows["created"], utc=True, errors="coerce")
    lead = (arr - cre) / pd.Timedelta(days=1)
    return np.minimum(lead, horizon_days).clip(lower=1).to_numpy(), lead.to_numpy()


def _hazard_score(hz_mod, hzm, rows: pd.DataFrame, horizon_days: int):
    """(prob, d) for a set of bookings via the survival product at d = min(lead, H)."""
    d, lead = _decision_horizon(rows, horizon_days)
    r = rows.copy()
    r["lead"] = lead
    r[hz_mod.AXIS] = d
    p = hz_mod.survival_cancel_proba(r, hz_mod.hazard_fn(hzm), hzm["num"], hzm["cat"],
                                     hzm["cat_dtypes"], snaps=hzm.get("snap"))
    return np.asarray(p, dtype=float), np.asarray(d, dtype=float)


# =============================================================================
# Per-model decision-time walk-forward (leak-free, frozen card hp)
# =============================================================================
def _predict_one_model(model_name: str, *, n_folds: int, horizon_days: int,
                       step_days: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pooled_predictions, per_fold_metrics) for ONE model, leak-free. Mirrors
    training.bakeoff_walk_forward's per-fold recipe (frozen card hp, calibrated static
    pipelines; hazard via the survival product at d = min(lead, H)), scoring every model
    on the SAME decision-time rows + label, and attaching property_name + the naive
    baseline. Also records train-vs-test AUC/AP/Brier per fold (for the 4.4 view)."""
    from . import load_clean_reservations
    from .training import _target  # shared, tested 0/1 target helper

    df = wf.add_outcome_known_date(load_clean_reservations())
    folds = wf.make_folds(df, n_folds=n_folds, horizon_days=horizon_days,
                          step_days=step_days, scheme="expanding")
    y = _target(df)
    prop = (df["property_name"].astype("string")
            if "property_name" in df.columns
            else pd.Series(["(unknown)"] * len(df), index=df.index, dtype="string"))

    is_hazard = model_name == "hazard"
    if is_hazard:
        from . import hazard as hz
    else:
        from .training import _family_feature_lists, build_pipeline, _card_hp
        num, cat = _family_feature_lists(model_name)   # family-correct (linear vs tree)
        X = df[num + cat]

    parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    for f in folds:
        if f.n_train < 500 or f.n_test < 50 or y.iloc[f.train_idx].nunique() < 2:
            continue
        ytr = y.iloc[f.train_idx].to_numpy()
        yte = y.iloc[f.test_idx].to_numpy().astype(int)
        prop_te = prop.iloc[f.test_idx].to_numpy()
        base_global, base_prop = property_baseline(
            ytr, prop.iloc[f.train_idx].to_numpy(), prop_te)
        te = df.iloc[f.test_idx]

        if is_hazard:
            hzm = hz.fit_hazard(df.iloc[f.train_idx], seed=seed)
            p_te, d_te = _hazard_score(hz, hzm, te, horizon_days)
            p_tr, _ = _hazard_score(hz, hzm, df.iloc[f.train_idx], horizon_days)
        else:
            pipe = build_pipeline(model_name, _card_hp(model_name), num, cat,
                                  calibrate=True, seed=seed)
            pipe.fit(X.iloc[f.train_idx], ytr)
            p_te = pipe.predict_proba(X.iloc[f.test_idx])[:, 1]
            p_tr = pipe.predict_proba(X.iloc[f.train_idx])[:, 1]
            d_te, _ = _decision_horizon(te, horizon_days)

        parts.append(pd.DataFrame({
            "fold": int(f.k),
            "property_name": prop_te,
            "days_until_arrival": np.asarray(d_te, dtype=float),
            "y_true": yte,
            "y_prob": np.asarray(p_te, dtype=float),
            "base_global": float(base_global),
            "base_property": np.asarray(base_prop, dtype=float),
        }))
        mt, mtr = _fold_metrics(yte, p_te), _fold_metrics(ytr, p_tr)
        fold_rows.append({
            "fold": int(f.k),
            "train_auc": mtr["auc"], "test_auc": mt["auc"],
            "train_ap": mtr["ap"], "test_ap": mt["ap"],
            "train_brier": mtr["brier"], "test_brier": mt["brier"],
            "n_train": mtr["n"], "n_test": mt["n"],
        })

    pred = (pd.concat(parts, ignore_index=True) if parts
            else pd.DataFrame(columns=list(EVAL_COLS)))
    foldm = pd.DataFrame(fold_rows)
    return pred, foldm


# =============================================================================
# Public entry points: cached, per-model
# =============================================================================
def model_eval(model_name: str, *, refresh: bool = False, n_folds: int = DEFAULT_N_FOLDS,
               horizon_days: int = HORIZON_DAYS, step_days: int = STEP_DAYS,
               seed: int = SEED, persist: bool = True) -> pd.DataFrame:
    """Pooled leak-free evaluation predictions for `model_name`, cached to parquet.

    Reads Data/model_eval_<model>.parquet if it exists (and `refresh` is False),
    otherwise recomputes the decision-time walk-forward and (if `persist`) writes the
    predictions parquet, the per-fold train-vs-test metrics parquet, and a provenance
    sidecar JSON. The Dash page calls this READ-ONLY (heavy compute is pre-warmed offline
    via `python main.py eval`).
    """
    if model_name not in EVAL_MODELS:
        raise KeyError(f"unknown model '{model_name}'. Known: {EVAL_MODELS}")
    path = eval_cache_path(model_name)
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    out, foldm = _predict_one_model(model_name, n_folds=n_folds, horizon_days=horizon_days,
                                    step_days=step_days, seed=seed)
    if persist and len(out):
        out.to_parquet(path, index=False)
        if len(foldm):
            foldm.to_parquet(fold_metrics_path(model_name), index=False)
        meta = {
            "model": model_name,
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "estimand": "P(cancel at/before arrival) at d = min(lead, horizon_days)",
            "horizon_days": int(horizon_days),
            "n_folds_requested": int(n_folds),
            "n_folds_used": int(out["fold"].nunique()),
            "n_pooled": int(len(out)),
            "pooled_base_rate": float(out["y_true"].mean()),
            "pooled_mean_pred": float(out["y_prob"].mean()),
        }
        eval_meta_path(model_name).write_text(json.dumps(meta, indent=2))
    return out


def load_fold_metrics(model_name: str) -> pd.DataFrame:
    """Per-fold train-vs-test metrics for a model (empty frame if not generated yet)."""
    p = fold_metrics_path(model_name)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def model_eval_meta(model_name: str) -> dict | None:
    """Provenance sidecar for a model's eval artifact, or None if not generated yet."""
    p = eval_meta_path(model_name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
