# ---------------------------------------------------------------------------
# src/model_eval.py
# Leak-free, PER-MODEL evaluation predictions for the XAI / Model-Performance page.
#
# Why this module exists
# ----------------------
# The page must compare each model to the naive historical-average baseline on the
# SAME estimand  P(cancel by arrival), scored at the decision horizon d = min(lead, H) 
# and break the result down per location. The four models emit different things natively
# (static classifiers -> predict_proba; hazard -> survival product), but src.scoring /
# src.hazard already collapse both to that one scalar. This module runs the decision-time
# walk-forward (src.walkforward.make_folds  the SINGLE eval regime for every model) once
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
# meaningful for calibration / Brier / log-loss / cost  NEVER for ROC-AUC (0.5 by
# construction). The page enforces that distinction (baseline shown at 4.2/4.3/4.9, not 4.1).
#
# DRY note: modeling logic is NOT duplicated here. The fold loop reuses the exact tested
# building blocks  training.build_pipeline / _card_hp / _family_feature_lists and
# hazard.fit_hazard / survival_cancel_proba  mirroring training.bakeoff_walk_forward.
#
# Compute: one artifact per model, cached. Static models are cheap; the HAZARD refit
# (person-period fit per fold, frozen card HP) is the slow one  pre-warm it offline
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
DEFAULT_N_FOLDS: Final[int] = wf.N_FOLDS  # the single shared fold budget (walkforward.N_FOLDS)
STEP_DAYS: Final[int] = 14
BASELINE_MIN_N: Final[int] = 100         # min train bookings for a per-property baseline rate
TRAIN_METRIC_SAMPLE: Final[int] = 30000  # cap for the (diagnostic) train-metric scoring

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
# Leak-free naive baseline (pure  unit-testable without sklearn)
# =============================================================================
def property_baseline(y_train, prop_train, prop_test,
                      *, min_n: int = BASELINE_MIN_N) -> tuple[float, np.ndarray]:
    """(global_train_rate, per_test_property_rate).

    The per-property rate is the TRAIN cancel rate for that property; properties with
    fewer than `min_n` train bookings (or unseen at train time) fall back to the global
    train rate. Everything is derived from TRAIN only, so it is leak-free by construction.
    Pure numpy/pandas  testable without the model stack.
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


def _hazard_hp() -> dict | None:
    """Frozen hazard hyperparameters  delegates to hazard.card_hp() (the SINGLE source),
    so the Model-Performance page and the notebook bake-offs fit the hazard identically.
    None (=> full search) if no card exists yet."""
    from . import hazard as hz
    return hz.card_hp()


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
    from .training import _target, iter_decision_folds  # shared fold iterator + 0/1 target

    df = wf.add_outcome_known_date(load_clean_reservations())
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
    for f in iter_decision_folds(df, n_folds=n_folds, horizon_days=horizon_days,
                                 step_days=step_days):
        ytr = y.iloc[f.train_idx].to_numpy()
        yte = y.iloc[f.test_idx].to_numpy().astype(int)
        prop_te = prop.iloc[f.test_idx].to_numpy()
        base_global, base_prop = property_baseline(
            ytr, prop.iloc[f.train_idx].to_numpy(), prop_te)
        te = df.iloc[f.test_idx]

        if is_hazard:
            hzm = hz.fit_hazard(df.iloc[f.train_idx], seed=seed, fixed_hp=_hazard_hp())
            p_te, d_te = _hazard_score(hz, hzm, te, horizon_days)
            # Train-metric (diagnostic only) on a SAMPLE  scoring all train via the survival
            # product is costly and the overfitting gap doesn't need the full set.
            tr_rows = df.iloc[f.train_idx]
            if len(tr_rows) > TRAIN_METRIC_SAMPLE:
                tr_rows = tr_rows.sample(TRAIN_METRIC_SAMPLE, random_state=seed)
            p_tr, _ = _hazard_score(hz, hzm, tr_rows, horizon_days)
            ytr_metric = _target(tr_rows).to_numpy()
        else:
            pipe = build_pipeline(model_name, _card_hp(model_name), num, cat,
                                  calibrate=True, seed=seed)
            pipe.fit(X.iloc[f.train_idx], ytr)
            p_te = pipe.predict_proba(X.iloc[f.test_idx])[:, 1]
            p_tr = pipe.predict_proba(X.iloc[f.train_idx])[:, 1]
            d_te, _ = _decision_horizon(te, horizon_days)
            ytr_metric = ytr

        parts.append(pd.DataFrame({
            "fold": int(f.k),
            "property_name": prop_te,
            "days_until_arrival": np.asarray(d_te, dtype=float),
            "y_true": yte,
            "y_prob": np.asarray(p_te, dtype=float),
            "base_global": float(base_global),
            "base_property": np.asarray(base_prop, dtype=float),
        }))
        mt, mtr = _fold_metrics(yte, p_te), _fold_metrics(ytr_metric, p_tr)
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


# ===========================================================================
# Honest comparison metrics (notebook 05 / promotion decision)
# ---------------------------------------------------------------------------
# A pooled ROC-AUC over (booking x horizon) rows is partly MECHANICAL: it rewards
# a model for knowing "how close is arrival" (far-out bookings have both higher risk
# AND higher predictions). `within_horizon_auc` strips that out by measuring AUC at a
# FIXED days-until-arrival - the honest per-booking ranking skill. And a promotion
# decision needs a real statistical test on the SAME rows, not "mean(delta) > its own
# std": `paired_delta_ci` + `promotion_report` give a paired bootstrap CI on the
# metrics the project actually cares about (PR-AUC at low prevalence, and cost).
# ===========================================================================
def within_horizon_auc(frame: pd.DataFrame, *, prob_col: str, y_col: str = "y_true",
                       day_col: str = "d", max_day: int = HORIZON_DAYS) -> tuple[pd.DataFrame, float]:
    """Per-horizon (fixed integer days-until-arrival) discrimination = the honest
    per-booking skill, stripped of the between-horizon separation that inflates a
    pooled AUC. Returns (table[day, n, base, p_std, auc], n_weighted_mean_auc).
    Days are ceil()'d and capped at `max_day`; a horizon with no score spread or one
    class gets auc=NaN (and is dropped from the weighted mean)."""
    from sklearn.metrics import roc_auc_score
    d = np.ceil(pd.to_numeric(frame[day_col], errors="coerce").clip(upper=max_day)).astype("Int64")
    g = frame.assign(_d=d).dropna(subset=["_d"])
    rows = []
    for day, grp in g.groupby("_d"):
        yy = grp[y_col].to_numpy(); pp = grp[prob_col].to_numpy()
        auc = (roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 and pp.std() > 0 else np.nan)
        rows.append({"day": int(day), "n": int(len(grp)), "base": float(yy.mean()),
                     "p_std": float(pp.std()), "auc": auc})
    tab = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    ok = tab.dropna(subset=["auc"])
    wmean = float(np.average(ok["auc"], weights=ok["n"])) if len(ok) else float("nan")
    return tab, wmean


def paired_delta_ci(y, p_challenger, p_baseline, *, metric: str = "ap",
                    n_boot: int = 2000, seed: int = SEED, alpha: float = 0.05) -> dict:
    """Paired bootstrap CI for (challenger - baseline) on the SAME rows. `metric` in
    {'ap' (PR-AUC), 'auc' (ROC-AUC)}. Resamples ROWS with replacement (preserving the
    pairing), recomputes the metric DIFFERENCE, returns the point estimate, the
    (alpha/2, 1-alpha/2) percentile CI, and the share of resamples where the challenger
    wins. A CI whose lower bound is >= 0 is evidence the challenger is not worse."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    scorer = average_precision_score if metric == "ap" else roc_auc_score
    y = np.asarray(y); pc = np.asarray(p_challenger, float); pb = np.asarray(p_baseline, float)
    point = float(scorer(y, pc) - scorer(y, pb))
    rng = np.random.default_rng(seed); n = len(y); deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        deltas.append(scorer(yy, pc[idx]) - scorer(yy, pb[idx]))
    deltas = np.asarray(deltas)
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"metric": metric, "delta": point, "lo": float(lo), "hi": float(hi),
            "ci": f"[{lo:+.4f}, {hi:+.4f}]", "p_challenger_better": float((deltas > 0).mean())}


def promotion_report(frame: pd.DataFrame, *, challenger: str = "hazard",
                     baseline: str = "histgb", n_boot: int = 2000, seed: int = SEED) -> dict:
    """Decision-relevant promotion test on the MATCHED bake frame (paired rows;
    columns y_true, p_<challenger>, p_<baseline>). Reports paired-bootstrap CIs for
    dPR-AUC and dROC-AUC + the delta in expected COST at each model's own cost-optimal
    threshold. Verdict (replaces the old mean(dAUC)>std heuristic): promote the
    challenger only if it does NOT lose on cost AND its dPR-AUC CI lower bound >= 0
    (wins/ties on the low-prevalence metric that actually matters here)."""
    from . import scoring as sc
    y = frame["y_true"].to_numpy()
    pc = frame[f"p_{challenger}"].to_numpy(); pb = frame[f"p_{baseline}"].to_numpy()
    d_ap = paired_delta_ci(y, pc, pb, metric="ap", n_boot=n_boot, seed=seed)
    d_auc = paired_delta_ci(y, pc, pb, metric="auc", n_boot=n_boot, seed=seed)
    t_c = sc.cost_threshold_from_scores(y, pc); t_b = sc.cost_threshold_from_scores(y, pb)
    cost_c = sc.cost_at_threshold(y, pc, t_c)["total_cost"]
    cost_b = sc.cost_at_threshold(y, pb, t_b)["total_cost"]
    d_cost = cost_c - cost_b
    promote = bool(d_ap["lo"] >= 0 and d_cost <= 0)
    return {"challenger": challenger, "baseline": baseline, "n": int(len(y)),
            "delta_pr_auc": d_ap, "delta_roc_auc": d_auc,
            "cost_challenger": cost_c, "cost_baseline": cost_b, "delta_cost": d_cost,
            "promote": promote,
            "reason": (f"dPR-AUC {d_ap['ci']}, dcost {d_cost:+,.0f}: "
                       + ("promote" if promote else "hold - not a clear win on PR-AUC and cost"))}
