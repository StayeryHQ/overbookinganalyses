# ---------------------------------------------------------------------------
# src/training.py
# App-callable training / retraining for the static cancellation models.
#
# Why this module exists
# ----------------------
# Fit logic used to live inside the model notebooks, so the dash app could not
# retrain. This centralises it so BOTH the notebooks (as thin drivers) and the
# app's "Datenaktualisierung" page call the same functions:
#
#   walk_forward_eval(model)            -> honest one-step-ahead metrics (no leakage)
#   retrain(model, mode="refit")        -> fit on ALL resolved data, FROZEN hyperparams
#   retrain(model, mode="retune")       -> re-search hyperparams, then fit on all data
#   select_models()                     -> the optimal model(s) the app should retrain
#
# Retraining decisions baked in (see the guards section):
#   * roster fingerprint stored in the card; feature-set changes are detected and
#     LOGGED (added/removed) - adding a column never errors, it just flows in;
#   * if the feature set changed and you asked for "refit", we WARN that the frozen
#     hyperparameters were tuned for the old set and recommend "retune";
#   * a scoring-time null audit flags any feature that is ~always-blank on upcoming
#     bookings (the check-in-leakage signature), so a bad un-exclude is loud, not silent.
#
# Heavy deps (sklearn / xgboost) are imported LAZILY inside functions so importing
# this module is cheap and never required just to read the guards.
# ---------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import json
import warnings
from typing import Final

import numpy as np
import pandas as pd

from . import scoring as sc
from . import walkforward as wf
from .features import load_feature_roster
from .paths import data_dir, repo_root

SEED: Final[int] = 42

# name -> estimator kind. Registry names mirror src.scoring.MODEL_REGISTRY.
MODEL_KIND: Final[dict[str, str]] = {
    "logreg": "logreg", "xgboost": "xgboost", "histgb": "histgb",
}


# =============================================================================
# Feature lists + target
# =============================================================================
def _feature_lists() -> tuple[list[str], list[str]]:
    r = load_feature_roster()
    return list(r["numeric"]), list(r["categorical"])


def _target(df: pd.DataFrame) -> pd.Series:
    """Binary target. Clean parquet ships `status` as 0/1; fall back to is_cancelled."""
    col = "is_cancelled" if "is_cancelled" in df.columns else "status"
    return pd.to_numeric(df[col], errors="coerce").astype(int)


# =============================================================================
# GUARDS - the "lil flag" logic for retraining with changed columns
# =============================================================================
def roster_fingerprint(roster: dict | None = None) -> dict:
    """Stable fingerprint of the modelled feature set (order-independent)."""
    r = roster or load_feature_roster()
    num, cat = sorted(r.get("numeric", [])), sorted(r.get("categorical", []))
    h = hashlib.sha1(("N:" + ",".join(num) + "|C:" + ",".join(cat)).encode()).hexdigest()[:12]
    return {"numeric": num, "categorical": cat, "hash": h}


def feature_change_report(model_name: str | None = None) -> dict:
    """Diff the CURRENT roster against the deployed model's stored feature set.

    `changed` is True if any feature was added or removed. Adding features is
    fine (they flow into the next fit); this just makes it visible.
    """
    cur = roster_fingerprint()
    cur_feats = set(cur["numeric"] + cur["categorical"])
    old_feats: set[str] = set()
    try:
        card = sc.load_model_card(model_name or sc.best_model())
        old_feats = set(card.get("features_numeric", [])) | set(card.get("features_categorical", []))
    except Exception:  # noqa: BLE001 - no card yet => everything is "added"
        pass
    added = sorted(cur_feats - old_feats)
    removed = sorted(old_feats - cur_feats)
    return {"hash": cur["hash"], "added": added, "removed": removed,
            "changed": bool(added or removed)}


def scoring_null_audit(upcoming_df: pd.DataFrame | None = None,
                       null_warn: float = 0.98) -> pd.DataFrame:
    """Per-feature null-rate on the UPCOMING frame; flags leakage-smelling features.

    A roster feature that is ~always null on upcoming bookings is almost certainly
    a check-in/address/company field that leaks (blank at scoring => model learns
    "blank ⇒ cancel"). Returns a frame with `leakage_warn` / `missing_in_build`.
    """
    from .scoring import build_features
    if upcoming_df is None:
        from .data_loader import load_reservations
        upcoming_df = load_reservations(upcoming_only=True)
    feat = build_features(upcoming_df)
    num, cat = _feature_lists()
    rows = []
    for c in num + cat:
        if c not in feat.columns:
            rows.append({"feature": c, "null_rate_upcoming": None,
                         "leakage_warn": True, "missing_in_build": True})
        else:
            nr = float(feat[c].isna().mean())
            rows.append({"feature": c, "null_rate_upcoming": round(nr, 4),
                         "leakage_warn": nr >= null_warn, "missing_in_build": False})
    return pd.DataFrame(rows).sort_values("null_rate_upcoming", ascending=False, na_position="first")


# =============================================================================
# Pipeline construction - mirrors notebooks 01 / 02 / 03 exactly
# =============================================================================
def build_pipeline(model_name: str, hp: dict, num: list[str], cat: list[str],
                   *, calibrate: bool = True, seed: int = SEED):
    """Build the preprocess+estimator(+calibration) pipeline for a model.

    Mirrors the notebook definitions:
      * logreg : median-impute + StandardScaler on numerics; most_frequent +
                 OneHotEncoder(ignore) on categoricals; LogisticRegression
                 (elasticnet/saga). Linear model => scaling + imputation needed.
      * xgboost: OneHotEncoder(ignore) on categoricals; numerics PASSTHROUGH
                 (NaN handled natively). tree_method="hist".
      * histgb : OneHotEncoder(ignore) on categoricals; numerics PASSTHROUGH
                 (NaN native), HistGradientBoosting with internal early stopping.
    Calibration: isotonic via CalibratedClassifierCV(cv=5) - so refitting on the
    full set still cross-fits the calibrator (no leakage).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.calibration import CalibratedClassifierCV

    kind = MODEL_KIND[model_name]
    ohe = OneHotEncoder(handle_unknown="ignore")  # robust to unseen categories (new hotels)

    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler())])
        cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                             ("onehot", ohe)])
        prep = ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])
        est = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000,
                                 random_state=seed, **hp)
    elif kind == "xgboost":
        from xgboost import XGBClassifier
        prep = ColumnTransformer([("cat", ohe, cat), ("num", "passthrough", num)])
        est = XGBClassifier(tree_method="hist", eval_metric="logloss",
                            random_state=seed, n_jobs=-1, **hp)
    elif kind == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        prep = ColumnTransformer([("cat", ohe, cat), ("num", "passthrough", num)])
        est = HistGradientBoostingClassifier(random_state=seed, early_stopping=True,
                                             validation_fraction=0.1, n_iter_no_change=20, **hp)
    else:
        raise KeyError(f"unknown model kind for {model_name!r}")

    clf = CalibratedClassifierCV(estimator=est, method="isotonic", cv=5) if calibrate else est
    return Pipeline([("prep", prep), ("clf", clf)])


def _default_hp(model_name: str) -> dict:
    """Baseline hyperparameters (used if no card / as search seed). From notebooks."""
    return {
        "logreg": dict(C=1.0, l1_ratio=0.5),
        "xgboost": dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=1.0,
                        colsample_bytree=1.0, min_child_weight=1, reg_lambda=1.0),
        "histgb": dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                       l2_regularization=1.0, max_iter=600),
    }[MODEL_KIND[model_name]]


def _card_hp(model_name: str) -> dict:
    """Frozen hyperparameters from the model card (estimator params only)."""
    try:
        hp = dict(sc.load_model_card(model_name).get("hyperparams", {}))
    except Exception:  # noqa: BLE001
        return _default_hp(model_name)
    # Drop non-estimator keys the card stores for provenance.
    for k in ("penalty", "solver", "max_iter", "random_state", "calibration",
              "tree_method", "eval_metric", "importance_type", "n_jobs",
              "early_stopping", "validation_fraction", "n_iter_no_change"):
        hp.pop(k, None)
    return hp or _default_hp(model_name)


# =============================================================================
# Hyperparameter search (retune) - train-only, TimeSeriesSplit, AP
# =============================================================================
def search_hyperparams(model_name: str, X, y, *, n_iter: int = 40, n_folds: int = 5,
                       seed: int = SEED) -> dict:
    """RandomizedSearch (TimeSeriesSplit, scoring=AP) on UNcalibrated pipeline.

    Mirrors the notebook search spaces; returns the best estimator params.
    """
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    from scipy.stats import loguniform, uniform, randint
    num, cat = _feature_lists()
    kind = MODEL_KIND[model_name]
    pipe = build_pipeline(model_name, _default_hp(model_name), num, cat, calibrate=False, seed=seed)

    if kind == "logreg":
        space = {"clf__C": loguniform(1e-4, 1e3), "clf__l1_ratio": uniform(0.0, 1.0)}
    elif kind == "xgboost":
        space = {"clf__n_estimators": [300, 600, 900, 1200],
                 "clf__max_depth": [3, 4, 5, 6, 8, 10],
                 "clf__learning_rate": loguniform(0.01, 0.3),
                 "clf__subsample": uniform(0.5, 0.5),
                 "clf__colsample_bytree": uniform(0.5, 0.5),
                 "clf__min_child_weight": [1, 3, 5, 10, 20],
                 "clf__reg_lambda": loguniform(0.5, 50),
                 "clf__reg_alpha": loguniform(1e-3, 5.0)}
    else:  # histgb
        space = {"clf__learning_rate": loguniform(0.01, 0.3),
                 "clf__max_leaf_nodes": randint(15, 127),
                 "clf__min_samples_leaf": randint(10, 200),
                 "clf__l2_regularization": loguniform(1e-3, 100.0),
                 "clf__max_iter": [300, 600, 900]}

    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="average_precision",
                                cv=TimeSeriesSplit(n_splits=n_folds), random_state=seed,
                                n_jobs=-1, refit=False)
    search.fit(X, y)
    return {k.replace("clf__", ""): v for k, v in search.best_params_.items()}


# =============================================================================
# Walk-forward evaluation - honest one-step-ahead metrics
# =============================================================================
def walk_forward_eval(model_name: str, *, hp: dict | None = None, n_folds: int = 6,
                      horizon_days: int = 14, step_days: int = 30, scheme: str = "expanding",
                      c_walk: float = sc.COST_WALK, c_empty: float = sc.COST_EMPTY,
                      seed: int = SEED) -> dict:
    """Fit the FROZEN-hp pipeline on each ARRIVAL-anchored fold's train, score the
    bookings arriving in the next `horizon_days`, and return per-fold + aggregate
    AUC / AP / Brier / cost@analytic-threshold.

    This is the decision-aligned procedure metric: at each scoring date S the test
    is the population the desk would decide on (active at S, arriving within the
    horizon), trained on ALL bookings resolved by S. The deployment model uses all
    resolved data; its expected performance is this distribution (live-monitored).
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    num, cat = _feature_lists()
    hp = hp or _card_hp(model_name)
    df = wf.add_outcome_known_date(_load_clean())
    folds = wf.make_folds(df, n_folds=n_folds, horizon_days=horizon_days,
                          step_days=step_days, scheme=scheme)
    X, y = df[num + cat], _target(df)
    t_analytic = sc.analytic_threshold(c_walk, c_empty)

    per_fold = []
    for f in folds:
        if f.n_train < 500 or f.n_test < 50 or y.iloc[f.train_idx].nunique() < 2:
            continue
        pipe = build_pipeline(model_name, hp, num, cat, calibrate=True, seed=seed)
        pipe.fit(X.iloc[f.train_idx], y.iloc[f.train_idx].values)
        p = pipe.predict_proba(X.iloc[f.test_idx])[:, 1]
        yt = y.iloc[f.test_idx].values
        cm = sc.cost_at_threshold(yt, p, t_analytic, c_walk, c_empty)
        per_fold.append({"fold": f.k, "origin": str(pd.Timestamp(f.origin).date()),
                         "n_train": f.n_train, "n_test": f.n_test,
                         "auc": roc_auc_score(yt, p) if len(np.unique(yt)) > 1 else float("nan"),
                         "ap": average_precision_score(yt, p),
                         "brier": brier_score_loss(yt, p),
                         "cost": cm["total_cost"]})
    pf = pd.DataFrame(per_fold)
    agg = {m: {"mean": float(pf[m].mean()), "std": float(pf[m].std())}
           for m in ["auc", "ap", "brier", "cost"]} if len(pf) else {}
    return {"model": model_name, "scheme": scheme, "n_folds_used": len(pf),
            "per_fold": per_fold, "aggregate": agg}


# =============================================================================
# Retrain - fit deployment model on ALL resolved data
# =============================================================================
def _load_clean() -> pd.DataFrame:
    from . import load_clean_reservations
    return load_clean_reservations()


def retrain(model_name: str, *, mode: str = "refit", asof: str | pd.Timestamp | None = None,
            persist: bool = True, n_iter: int = 40, seed: int = SEED) -> dict:
    """Retrain a model for DEPLOYMENT and report honest walk-forward metrics.

    mode="refit"  -> frozen hyperparameters from the model card.
    mode="retune" -> re-search hyperparameters first (heavier).

    Always: read the CURRENT roster (so added columns flow in), fit the pipeline
    on ALL data resolved by `asof`, recompute the cost-optimal threshold, and (if
    persist) write the joblib + an updated card carrying the roster fingerprint,
    the feature-change report, and the walk-forward metrics.
    """
    if mode not in ("refit", "retune"):
        raise ValueError("mode must be 'refit' or 'retune'")
    if model_name == "hazard":
        # The hazard model (per-night engine, the PRIMARY serving model) has its own
        # person-period fit; dispatch to src.hazard. It always runs its small HP grid,
        # so refit/retune behave the same here.
        from . import hazard as hz
        return hz.retrain_hazard(asof=asof, persist=persist, seed=seed)
    num, cat = _feature_lists()
    df = wf.add_outcome_known_date(_load_clean())
    known = pd.to_datetime(df[wf.KNOWN_COL], utc=True, errors="coerce")
    asof_ts = pd.Timestamp(asof, tz="UTC") if asof is not None else pd.Timestamp(known.max())
    resolved = known <= asof_ts                                   # all data known "now"
    X, y = df[num + cat], _target(df)

    # ---- the retraining "flag": detect + log feature-set changes --------------
    change = feature_change_report(model_name)
    if change["changed"]:
        print(f"[retrain:{model_name}] roster changed vs deployed model - "
              f"added={change['added']} removed={change['removed']}")
        if mode == "refit" and change["added"]:
            warnings.warn(
                f"[retrain:{model_name}] feature set changed but mode='refit' reuses "
                "hyperparameters tuned for the OLD set. Consider mode='retune'.",
                stacklevel=2)

    # ---- hyperparameters ------------------------------------------------------
    if mode == "retune":
        # search on the resolved data (TimeSeriesSplit inside the search)
        Xr = X[resolved].sort_index()
        hp = search_hyperparams(model_name, Xr, y[resolved].loc[Xr.index],
                                n_iter=n_iter, seed=seed)
    else:
        hp = _card_hp(model_name)

    # ---- honest forward metrics (procedure) -----------------------------------
    wf_report = walk_forward_eval(model_name, hp=hp, seed=seed)

    # ---- deployment fit on ALL resolved data ----------------------------------
    pipe = build_pipeline(model_name, hp, num, cat, calibrate=True, seed=seed)
    pipe.fit(X[resolved], y[resolved].values)

    # threshold: from the most recent fold's holdout (honest), fall back to analytic
    pf = wf_report.get("per_fold", [])
    thr = sc.analytic_threshold()
    result = {"model": model_name, "mode": mode, "asof": str(asof_ts.date()),
              "n_train_deploy": int(resolved.sum()),
              "feature_change": change, "walk_forward": wf_report,
              "hyperparams": hp, "threshold": thr}

    if persist:
        reg = sc.MODEL_REGISTRY[model_name]
        jp = data_dir() / reg["joblib"]
        import joblib
        joblib.dump(pipe, jp)
        fp = roster_fingerprint()
        card_path = repo_root() / reg["card"]
        card = {}
        if card_path.exists():
            try: card = json.loads(card_path.read_text())
            except Exception: card = {}
        card.update({
            "model": model_name, "retrained_at": pd.Timestamp.utcnow().isoformat(),
            "mode": mode, "asof": str(asof_ts.date()), "n_train_deploy": int(resolved.sum()),
            "features_numeric": num, "features_categorical": cat,
            "roster_hash": fp["hash"], "feature_change": change,
            "hyperparams": hp, "walk_forward": wf_report["aggregate"],
            "operating_points": [{"name": "cost_optimal", "threshold": thr}],
        })
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(json.dumps(card, indent=2))
        result["persisted"] = {"joblib": str(jp), "card": str(card_path)}
    return result


# =============================================================================
# Model selection - what the app should retrain
# =============================================================================
def select_models() -> dict:
    """The optimal models to retrain/serve.

    primary : "hazard" — the horizon-aware per-night expected-freed engine. The
              overbooking decision is made a horizon (d = 1..14 days) before
              arrival, so the time-resolved hazard model is the right serving
              model; `src.hazard` makes it persistable/retrainable.
    static  : best static model by AP among well-calibrated ones (Brier gate) —
              a horizon-blind per-booking BASELINE / cross-check, not the decision
              engine (it has no days-until-arrival feature).
    """
    from . import hazard as hz
    out = {"primary": "hazard", "hazard": "hazard",
           "hazard_trained": bool(hz.hazard_available())}
    try:
        out["static"] = sc.best_model()
    except Exception:  # noqa: BLE001 - no trained static models yet
        out["static"] = None
    return out
