# ---------------------------------------------------------------------------
# Training / retraining for the static cancellation models (logreg, xgboost,
# histgb). Notebooks and the app's Update page call the SAME functions:
#
#   walk_forward_eval(model)      -> honest out-of-time metrics (no leakage)
#   retrain(model, mode="refit")  -> fit on all resolved data, frozen hyperparams
#   retrain(model, mode="retune") -> re-search hyperparams first
#   select_models()               -> which models the app should retrain/serve
#
# Guards built into retrain(): roster-fingerprint diff (feature changes are
# logged, refit warns), no-card refits escalate to retune (never deploy an
# un-tuned fixed-tree model), and scoring_null_audit() catches check-in leakage.
# Heavy deps (sklearn/xgboost) import lazily inside functions.
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
    """Raw-roster SUPERSET (all candidate cols, both raw and `_log` twins). Presence
    guard / null-audit only - NOT a model's trained set. Single accessor:
    src.features.roster_features."""
    from .features import roster_features
    return roster_features()


def _family_feature_lists(model_name: str) -> tuple[list[str], list[str]]:
    """Family-aware feature view for a model (linear -> `_log` twins; tree -> raw
    columns; decided 2026-06-30). Thin delegate to the ONE source of truth,
    src.features.model_feature_lists - so every fit/score site shares one list."""
    from .features import model_feature_lists
    return model_feature_lists(model_name)


def _target(df: pd.DataFrame) -> pd.Series:
    """Binary target  delegates to the one shared accessor (wf.target_series)."""
    return wf.target_series(df)


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
    """Diff a model's CURRENT feature set against the one its deployed card was trained on.

    `changed` is True if any feature was added or removed. Adding features is fine (they
    flow into the next fit); this just makes it visible.

    Compares LIKE-FOR-LIKE via the model's FAMILY-CORRECT list (`_family_feature_lists`:
    tree -> raw columns; linear -> `_log` twins). Using the raw-roster SUPERSET here made the
    four `_log` twins look permanently "added" on every TREE-model refit (they are linear-
    family features the tree card never lists) -> a spurious "feature set changed" warning
    even when the roster hash is identical. The stable roster hash is still reported.
    """
    cur = roster_fingerprint()                       # kept for the order-independent roster hash
    name = model_name or sc.best_model()
    cur_num, cur_cat = _family_feature_lists(name)
    cur_feats = set(cur_num) | set(cur_cat)
    old_feats: set[str] = set()
    try:
        card = sc.load_model_card(name)
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
    # dense output: HistGradientBoosting rejects sparse; robust to unseen categories.
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

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
    """Baseline hyperparameters (structural search seed only). NB: xgboost carries
    NO fixed `n_estimators` on purpose - the tree count is set by early stopping in the
    HP search (`src.hp_search`), and `retrain` forces a retune when no card exists, so we
    never silently deploy a fixed-tree XGBoost (the old n_estimators=600 trap)."""
    return {
        "logreg": dict(C=1.0, l1_ratio=0.5),
        "xgboost": dict(max_depth=6, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0),
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
# Walk-forward evaluation - honest one-step-ahead metrics
# =============================================================================
def iter_decision_folds(df: pd.DataFrame, *, n_folds: int = wf.N_FOLDS, horizon_days: int = 14,
                        step_days: int = 14, scheme: str = "expanding",
                        min_train: int = 500, min_test: int = 50):
    """Yield the USABLE decision-time folds for `df` (must already carry
    outcome_known_date): src.walkforward.make_folds + the shared usability guard
    (enough train/test rows AND a two-class train target). THE single fold iterator
    behind every walk-forward surface - walk_forward_eval / walk_forward_predict /
    bakeoff_walk_forward / model_eval all loop over this, so the fold generation and
    the skip-guard live in exactly one place (was copy-pasted four times)."""
    y = _target(df)
    for f in wf.make_folds(df, n_folds=n_folds, horizon_days=horizon_days,
                           step_days=step_days, scheme=scheme):
        if f.n_train < min_train or f.n_test < min_test or y.iloc[f.train_idx].nunique() < 2:
            continue
        yield f


def walk_forward_eval(model_name: str, *, hp: dict | None = None, n_folds: int = wf.N_FOLDS,
                      horizon_days: int = 14, step_days: int = 14, scheme: str = "expanding",
                      c_walk: float = sc.COST_WALK, c_empty: float = sc.COST_EMPTY,
                      seed: int = SEED, collect_predictions: bool = False) -> dict:
    """Decision-time walk-forward for one static model: fit the frozen-hp pipeline
    on each fold's train, score its test bookings, return per-fold + aggregate
    AUC / AP / Brier / cost.

    Cost is measured at the TRAIN-derived cost-optimal threshold APPLIED to the held-out
    test fold (out-of-sample)  an honest "operated well" estimate, NOT the optimistic
    in-sample argmin on the very rows it is scored on. (At the fixed analytic threshold
    ~0.79 no calibrated model flags anything, so that metric was identical across models.)

    `collect_predictions=True` also returns the pooled out-of-time predictions
    under "predictions" [fold, y_true, y_prob]  retrain() persists them so the
    serving thresholds can be derived from real data.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    num, cat = _family_feature_lists(model_name)
    hp = hp or _card_hp(model_name)
    df = wf.add_outcome_known_date(_load_clean())
    X, y = df[num + cat], _target(df)

    per_fold, parts = [], []
    for f in iter_decision_folds(df, n_folds=n_folds, horizon_days=horizon_days,
                                 step_days=step_days, scheme=scheme):
        pipe = build_pipeline(model_name, hp, num, cat, calibrate=True, seed=seed)
        pipe.fit(X.iloc[f.train_idx], y.iloc[f.train_idx].values)
        p = pipe.predict_proba(X.iloc[f.test_idx])[:, 1]
        yt = y.iloc[f.test_idx].values
        # Operating point chosen OUT-OF-SAMPLE: fit the cost-optimal threshold on the TRAIN
        # predictions, then APPLY it to the held-out test fold  honest cost, not the
        # optimistic in-sample argmin (M1 fix).
        ytr = y.iloc[f.train_idx].values
        p_tr = pipe.predict_proba(X.iloc[f.train_idx])[:, 1]
        t_fold = sc.cost_threshold_from_scores(ytr, p_tr, c_walk, c_empty)
        cm = sc.cost_at_threshold(yt, p, t_fold, c_walk, c_empty)
        per_fold.append({"fold": f.k, "origin": str(pd.Timestamp(f.origin).date()),
                         "n_train": f.n_train, "n_test": f.n_test,
                         "auc": roc_auc_score(yt, p) if len(np.unique(yt)) > 1 else float("nan"),
                         "ap": average_precision_score(yt, p),
                         "brier": brier_score_loss(yt, p),
                         "cost": cm["total_cost"], "cost_threshold": t_fold})
        if collect_predictions:
            parts.append(pd.DataFrame({"fold": f.k, "y_true": yt, "y_prob": p}))
    pf = pd.DataFrame(per_fold)
    agg = {m: {"mean": float(pf[m].mean()), "std": float(pf[m].std())}
           for m in ["auc", "ap", "brier", "cost"]} if len(pf) else {}
    out = {"model": model_name, "scheme": scheme, "n_folds_used": len(pf),
           "per_fold": per_fold, "aggregate": agg}
    if collect_predictions:
        out["predictions"] = (pd.concat(parts, ignore_index=True) if parts
                              else pd.DataFrame(columns=["fold", "y_true", "y_prob"]))
    return out


def walk_forward_predict(model_name: str, *, hp: dict | None = None, n_folds: int = wf.N_FOLDS,
                         horizon_days: int = 14, step_days: int = 14,
                         scheme: str = "expanding", seed: int = SEED) -> pd.DataFrame:
    """Pooled out-of-time predictions across the decision-time folds: fit the
    frozen-hp CALIBRATED pipeline on each fold's train, predict its test, and return
    a long DataFrame [fold, y_true, y_prob]. One place to get honest OOS scores for
    reliability curves, cost-optimal thresholds and pooled metrics (notebooks 01-03).
    """
    # DRY: the pooled [fold, y_true, y_prob] frame is EXACTLY what walk_forward_eval already
    # collects on the same folds / same guard / same frozen-hp calibrated pipeline. Derive it
    # from there instead of re-implementing an identical fold loop (output is bit-for-bit the
    # same; walk_forward_eval also computes per-fold cost, which this caller simply ignores).
    out = walk_forward_eval(model_name, hp=hp, n_folds=n_folds, horizon_days=horizon_days,
                            step_days=step_days, scheme=scheme, seed=seed,
                            collect_predictions=True)
    return out["predictions"]


# The matched bake-off is expensive (fits 3 static + hazard per fold, ~hours). Persist
# its per-row predictions so notebook 05 - and any later diagnosis - can slice metrics
# (pooled / per-fold / by horizon d) in seconds instead of re-fitting everything.
BAKEOFF_PATH: Final[str] = "bakeoff_predictions.parquet"
BAKEOFF_META_PATH: Final[str] = "bakeoff_predictions_meta.json"


def _persist_bakeoff(frame: pd.DataFrame, *, n_folds: int, horizon_days: int,
                     step_days: int, seed: int) -> None:
    """Write the matched bake-off frame + a small provenance JSON to the Data dir."""
    frame.to_parquet(data_dir() / BAKEOFF_PATH, index=False)
    meta = {"generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "n_rows": int(len(frame)), "n_folds": int(n_folds),
            "horizon_days": int(horizon_days), "step_days": int(step_days), "seed": int(seed),
            "base_rate": float(frame["y_true"].mean()) if len(frame) else None,
            "models": [c[2:] for c in frame.columns if c.startswith("p_")]}
    (data_dir() / BAKEOFF_META_PATH).write_text(json.dumps(meta, indent=2))


class StaleArtifact(RuntimeError):
    """A cached artifact is older than an input it was built from (must recompute)."""


def _bakeoff_deps() -> list:
    """Files the bake-off predictions are built FROM: the CLEAN data + each model's card
    (the frozen hyperparameters every fold refits with). If any is newer than the cache,
    the cache no longer reflects the current models/data and is stale."""
    from . import scoring as sc
    from .data_loader import CLEAN_CACHE_FILE
    deps = [data_dir() / CLEAN_CACHE_FILE]
    for name in ("logreg", "xgboost", "histgb", "hazard"):
        try:
            deps.append(repo_root() / sc.MODEL_REGISTRY[name]["card"])
        except Exception:  # noqa: BLE001
            pass
    return [p for p in deps if p.exists()]


def bakeoff_cache_status() -> str:
    """'missing' | 'stale' | 'fresh' for Data/bakeoff_predictions.parquet - compares its
    mtime against the clean data + the model cards it was built from. This is the bake-off's
    equivalent of retrain(refresh_eval=True): after a retrain (which rewrites the cards) or a
    data refresh, the cache reports 'stale' so consumers recompute instead of showing old
    numbers."""
    p = data_dir() / BAKEOFF_PATH
    if not p.exists():
        return "missing"
    cache_mtime = p.stat().st_mtime
    return "stale" if any(d.stat().st_mtime > cache_mtime for d in _bakeoff_deps()) else "fresh"


def load_bakeoff(*, require_fresh: bool = True) -> pd.DataFrame:
    """Load the persisted matched bake-off frame (columns: fold, y_true, d, p_<model>).

    Raises FileNotFoundError if it was never generated, or StaleArtifact if it predates
    the clean data / a model card (so a retrain or data refresh forces a recompute rather
    than silently serving old predictions). Pass `require_fresh=False` to read a stale
    cache anyway (e.g. quick inspection)."""
    status = bakeoff_cache_status()
    if status == "missing":
        raise FileNotFoundError(
            f"{data_dir() / BAKEOFF_PATH} not found - run training.bakeoff_walk_forward() "
            "(notebook 05) first.")
    if require_fresh and status == "stale":
        raise StaleArtifact(
            f"{BAKEOFF_PATH} is older than the clean data or a model card - recompute via "
            "bakeoff_walk_forward() so the comparison reflects the current models/data.")
    return pd.read_parquet(data_dir() / BAKEOFF_PATH)


def bakeoff_walk_forward(*, n_folds: int = wf.N_FOLDS, horizon_days: int = 14, step_days: int = 14,
                         seed: int = SEED, persist: bool = True) -> pd.DataFrame:
    """FAIR matched bake-off (notebook 05). On each decision-time fold, fit LogReg,
    XGBoost, HistGB (family-correct, frozen card hp, calibrated) AND the hazard model
    on the SAME train, then score the SAME test rows. Pool. Returns a wide frame
    [fold, y_true, d, p_logreg, p_xgboost, p_histgb, p_hazard] - every model on the
    identical rows + label, so AUC / AP / Brier / confusion are directly comparable.
    Hazard uses the survival product at d = min(lead, H) (its decision horizon).

    `persist=True` (default) writes the frame to Data/bakeoff_predictions.parquet (+ a
    provenance JSON) so downstream metrics are a cheap slice, not an hours-long refit;
    read it back with `load_bakeoff()`.
    """
    from . import hazard as hz
    df = wf.add_outcome_known_date(_load_clean())
    tgt = _target(df)
    lin_num, lin_cat = _family_feature_lists("logreg")
    tree_num, tree_cat = _family_feature_lists("xgboost")   # xgboost == histgb family
    statics = {"logreg": (lin_num, lin_cat), "xgboost": (tree_num, tree_cat),
               "histgb": (tree_num, tree_cat)}
    parts = []
    for f in iter_decision_folds(df, n_folds=n_folds, horizon_days=horizon_days,
                                 step_days=step_days):
        ytr = tgt.iloc[f.train_idx].to_numpy()
        te = df.iloc[f.test_idx].copy()
        arr = pd.to_datetime(te["arrival"], utc=True); cre = pd.to_datetime(te["created"], utc=True)
        te["lead"] = (arr - cre) / pd.Timedelta(days=1)
        te[hz.AXIS] = np.minimum(te["lead"], horizon_days).clip(lower=1)   # remaining decision horizon d
        out = {"fold": f.k, "y_true": tgt.iloc[f.test_idx].to_numpy(),
               "d": te[hz.AXIS].to_numpy()}
        for m, (num, cat) in statics.items():
            pipe = build_pipeline(m, _card_hp(m), num, cat, calibrate=True, seed=seed)
            pipe.fit(df[num + cat].iloc[f.train_idx], ytr)
            out[f"p_{m}"] = pipe.predict_proba(df[num + cat].iloc[f.test_idx])[:, 1]
        hzm = hz.fit_hazard(df.iloc[f.train_idx], seed=seed, fixed_hp=hz.card_hp())
        out["p_hazard"] = hz.survival_cancel_proba(te, hz.hazard_fn(hzm), hzm["num"], hzm["cat"],
                                                   hzm["cat_dtypes"], snaps=hzm.get("snap"))
        parts.append(pd.DataFrame(out))
    frame = (pd.concat(parts, ignore_index=True) if parts
             else pd.DataFrame(columns=["fold", "y_true", "d",
                                        "p_logreg", "p_xgboost", "p_histgb", "p_hazard"]))
    if persist and len(frame):
        _persist_bakeoff(frame, n_folds=n_folds, horizon_days=horizon_days,
                         step_days=step_days, seed=seed)
    return frame


# =============================================================================
# Retrain - fit deployment model on ALL resolved data
# =============================================================================
def _load_clean() -> pd.DataFrame:
    from . import load_clean_reservations
    return load_clean_reservations()


def retrain(model_name: str, *, mode: str = "refit", asof: str | pd.Timestamp | None = None,
            persist: bool = True, n_iter: int = 90, seed: int = SEED,
            refresh_eval: bool = False) -> dict:
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
        # person-period fit; dispatch to src.hazard. refit reuses the frozen card HP
        # (fast); retune runs the shared TPE search (slower) — same semantics as the
        # static models, so the app's "Re-estimate hyperparameters" toggle is honoured.
        from . import hazard as hz
        return hz.retrain_hazard(mode=mode, asof=asof, persist=persist, seed=seed,
                                 refresh_eval=refresh_eval)
    num, cat = _family_feature_lists(model_name)
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

    # ---- safety: never deploy an UN-TUNED model -------------------------------
    # A 'refit' with no existing card would fall back to the structural defaults;
    # for xgboost that is a fixed tree count with no early stopping. Escalate to a
    # retune so the tree count is always chosen by early stopping.
    if mode == "refit":
        try:
            sc.load_model_card(model_name)
        except Exception:  # noqa: BLE001 - no card yet
            warnings.warn(f"[retrain:{model_name}] no model card -> switching refit to retune "
                          "(never deploy an un-tuned / fixed-tree model).", stacklevel=2)
            mode = "retune"

    # ---- hyperparameters ------------------------------------------------------
    tune_report = None
    if mode == "retune":
        # search on the resolved data, TIME-ORDERED by `created` (so xgboost's
        # early-stopping val is a true temporal tail, and TimeSeriesSplit is honest).
        # ONE shared TPE engine (src.hp_search) for every model; captures the tuning
        # report (best PR-AUC + expected cost) so the card records what the search found.
        from . import hp_search as hps
        order = pd.to_datetime(df.loc[resolved, "created"], utc=True, errors="coerce").sort_values().index
        Xr, yr = X.loc[order], y.loc[order]
        hp, tune_report = hps.tune_static(model_name, Xr, yr, n_trials=n_iter, seed=seed)
    else:
        hp = _card_hp(model_name)

    # ---- honest forward metrics + pooled out-of-time predictions --------------
    wf_report = walk_forward_eval(model_name, hp=hp, seed=seed, collect_predictions=True)
    preds = wf_report.pop("predictions", None)   # DataFrame; kept out of the JSON card

    # ---- deployment fit on ALL resolved data ----------------------------------
    pipe = build_pipeline(model_name, hp, num, cat, calibrate=True, seed=seed)
    pipe.fit(X[resolved], y[resolved].values)

    # ---- decision-time recalibration -------------------------------------------
    # The pipeline's isotonic calibration is learned on the RESOLVED population
    # (base rate ~20%), but the model is consumed on the DECISION-TIME population
    # ("still open at the decision date", base ~12%), where survivorship selection
    # makes it overpredict ~2x. Fit a second isotonic map on the pooled out-of-time
    # decision-time predictions; scoring.cancel_proba applies it when present.
    # Verified on held-out later folds: reliability −98%, Brier −19%, BSS −0.13→+0.09,
    # ranking unchanged (monotone map).
    recal = None
    if preds is not None and len(preds) >= 500 and preds["y_true"].nunique() > 1:
        from sklearn.isotonic import IsotonicRegression
        recal = IsotonicRegression(out_of_bounds="clip").fit(preds["y_prob"], preds["y_true"])
        preds = preds.assign(y_prob_raw=preds["y_prob"],
                             y_prob=recal.predict(preds["y_prob"].to_numpy()))

    # Decision threshold: cost-optimal on the pooled predictions, on the SAME
    # (recalibrated) scale the served probabilities use; analytic value only when
    # no fold was usable.
    if preds is not None and len(preds):
        thr = sc.cost_threshold_from_scores(preds["y_true"], preds["y_prob"])
    else:
        thr = sc.analytic_threshold()
    result = {"model": model_name, "mode": mode, "asof": str(asof_ts.date()),
              "n_train_deploy": int(resolved.sum()),
              "feature_change": change, "walk_forward": wf_report,
              "hyperparams": hp, "threshold": thr, "tuning": tune_report}

    if persist:
        reg = sc.MODEL_REGISTRY[model_name]
        jp = data_dir() / reg["joblib"]
        import joblib
        joblib.dump(pipe, jp)
        # Persist the pooled (recalibrated) predictions  scoring's threshold
        # helpers (cost_optimal_threshold) read this file.
        if preds is not None and len(preds):
            pred_path = data_dir() / reg["joblib"].replace("_model.joblib",
                                                           "_predictions.parquet")
            preds.to_parquet(pred_path, index=False)
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
            "decision_time_recalibrated": bool(recal is not None),
            "operating_points": [{"name": "cost_optimal", "threshold": thr}],
            "tuning": tune_report,   # None on refit; {metric,best_ap,best_cost,...} on retune
        })
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(json.dumps(card, indent=2))
        persisted = {"joblib": str(jp), "card": str(card_path)}
        if preds is not None and len(preds):
            persisted["predictions"] = str(pred_path)
        if recal is not None:
            cal_path = data_dir() / reg["joblib"].replace("_model.joblib",
                                                          "_calibration.joblib")
            joblib.dump(recal, cal_path)
            persisted["calibration"] = str(cal_path)
        result["persisted"] = persisted

    # Keep the Model-Performance page's eval artifact in sync with the new model.
    if refresh_eval:
        try:
            from . import model_eval as _me
            _me.model_eval(model_name, refresh=True)
            result["eval_refreshed"] = True
        except Exception as e:  # noqa: BLE001
            result["eval_refresh_error"] = str(e)[:120]
    return result


# =============================================================================
# Model selection - what the app should retrain
# =============================================================================
def select_models() -> dict:
    """The optimal models to retrain/serve.

    primary : "hazard"  the horizon-aware per-night expected-freed engine. The
              overbooking decision is made a horizon (d = 1..14 days) before
              arrival, so the time-resolved hazard model is the right serving
              model; `src.hazard` makes it persistable/retrainable.
    static  : best static model by AP among well-calibrated ones (Brier gate) 
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
