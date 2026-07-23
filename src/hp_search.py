# ---------------------------------------------------------------------------
# src/hp_search.py
# ONE standardized hyperparameter search for every model AND every surface.
#
# Why this module exists
# ----------------------
# Tuning used to be MIRRORED in two hand-rolled places that could drift apart:
#   * training._search_xgboost_earlystop + a RandomizedSearchCV branch (statics)
#   * an inline random-candidate loop inside hazard.fit_hazard (hazard)
# They used different strategies (RandomizedSearch vs manual random draws), wired
# the metric differently, and carried no shared cost view. This module replaces BOTH
# with a single Tree-structured Parzen Estimator (TPE, via Optuna) engine that:
#   * optimizes PR-AUC (average precision) - the repo's declared honest metric at
#     ~12% prevalence - and ALSO reports the expected overbooking cost at the
#     cost-optimal threshold (src.scoring.cost_at_threshold), so every retune carries
#     both numbers for the model-choice decision;
#   * preserves the temporal discipline the old code had: TimeSeriesSplit CV for
#     logreg/histgb; a most-recent-by-created validation tail + XGBoost/hazard early
#     stopping (no fixed tree count) for the boosted models;
#   * warm-starts each study by ENQUEUEING the known-good baseline config, so a
#     retune can never land worse than the previous hand-tuned default;
#   * returns hyperparameters in the SAME schema the model cards already store, so
#     training.retrain / hazard.retrain_hazard / _card_hp / build_pipeline all keep
#     working unchanged - only the SEARCH changed, not the frozen-HP fit/eval paths.
#
# Public API
#   run_study(suggest, evaluate, *, n_trials, seed, ...) -> (best_params, report)
#       the shared TPE engine (sampler + median pruner + baseline warm-start).
#   tune_static(model_name, X, y, ...) -> (hp, report)   logreg / xgboost / histgb
#   suggest_space(model_name, trial) -> params           per-model space (ONE source)
#   pr_auc(y, p) / expected_cost(y, p)                    shared metric + cost helpers
#
# The hazard model owns its person-period candidate fit (src.hazard) and calls
# run_study with suggest_space("hazard", .); everything else about the search -
# sampler, pruner, metric, cost, warm-start - lives HERE, so the search is identical
# everywhere it runs (notebooks, CLI retrain, the app's Update page).
# Heavy deps (optuna / sklearn / xgboost) are imported lazily inside functions.
# ---------------------------------------------------------------------------
from __future__ import annotations

import logging
from typing import Callable, Final

import numpy as np

SEED: Final[int] = 42
N_TRIALS_DEFAULT: Final[int] = 90          # the standard budget (user-chosen)
VAL_FRAC_DEFAULT: Final[float] = 0.15      # most-recent temporal tail for early-stopping models
N_FOLDS_DEFAULT: Final[int] = 5            # TimeSeriesSplit folds for logreg/histgb

logger = logging.getLogger(__name__)


# =============================================================================
# Shared metric + cost (single definition, reused by every model)
# =============================================================================
def pr_auc(y, p) -> float:
    """Average precision (PR-AUC). NaN if a fold/val slice is single-class."""
    from sklearn.metrics import average_precision_score
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def expected_cost(y, p, *, c_walk: float | None = None, c_empty: float | None = None) -> float:
    """Expected overbooking cost at the cost-optimal threshold, using the SAME shared
    rule the app/notebooks pick the operating point by (src.scoring). Reported next to
    PR-AUC so a retune shows both the ranking metric and the money metric."""
    from . import scoring as sc
    cw = sc.COST_WALK if c_walk is None else c_walk
    ce = sc.COST_EMPTY if c_empty is None else c_empty
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if y.size == 0 or np.unique(y).size < 2:
        return float("nan")
    thr = sc.cost_threshold_from_scores(y, p, cw, ce)
    return float(sc.cost_at_threshold(y, p, thr, cw, ce)["total_cost"])


# =============================================================================
# Per-model search space (ONE source of truth for the tuning ranges)
# =============================================================================
def suggest_space(model_name: str, trial) -> dict:
    """Sample one hyperparameter config for `model_name` from `trial`. Ranges mirror
    the previous hand-rolled searches exactly (so behaviour is comparable), just driven
    by Optuna/TPE instead of RandomizedSearch / manual draws."""
    from .training import MODEL_KIND
    kind = MODEL_KIND.get(model_name, model_name)   # "hazard" passes through
    if kind == "logreg":
        return {"C": trial.suggest_float("C", 1e-4, 1e3, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)}
    if kind == "histgb":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 100.0, log=True),
                "max_iter": trial.suggest_categorical("max_iter", [300, 600, 900])}
    if kind == "xgboost":
        # Tightened, calibration-friendlier space (LR<=0.12, depth<=6, min_child_weight>=5,
        # subsample/colsample<1): the old wide space landed on LR~0.245/depth-4 which won AP
        # but calibrated worse (Brier 0.121 vs histgb 0.106). Tree count via early stopping.
        return {"max_depth": trial.suggest_categorical("max_depth", [3, 4, 5, 6]),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
                "min_child_weight": trial.suggest_categorical("min_child_weight", [5, 10, 20, 30, 50]),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True)}
    if kind == "hazard":
        return {"max_depth": trial.suggest_categorical("max_depth", [4, 5, 6, 7, 8]),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "min_child_weight": trial.suggest_categorical("min_child_weight", [5, 10, 20, 30, 50]),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 15.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.95)}
    raise KeyError(f"no search space defined for {model_name!r}")


# =============================================================================
# The shared TPE engine (the single search mechanism)
# =============================================================================
def run_study(suggest: Callable, evaluate: Callable, *, n_trials: int = N_TRIALS_DEFAULT,
              seed: int = SEED, direction: str = "maximize",
              enqueue: list[dict] | None = None, prunable: bool = False) -> tuple[dict, dict]:
    """Run ONE TPE study and return (best_params, report).

    `suggest(trial) -> params`      : the model's search space (see suggest_space).
    `evaluate(trial, params) -> float` : fit+score a candidate. It MAY stash, via
        trial.set_user_attr(...): 'derived' (dict of extra HP to merge into the winner,
        e.g. an early-stopped 'n_estimators'), 'ap' and 'cost' (for the report).
    `enqueue` warm-starts the study with known-good configs (tried first, so the
    result can't be worse than the baseline). `prunable=True` enables MedianPruner
    for evaluators that report per-fold intermediate values (CV models).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1) if prunable else optuna.pruners.NopPruner()
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    for cfg in (enqueue or []):
        study.enqueue_trial(cfg, skip_if_exists=True)

    study.optimize(lambda tr: evaluate(tr, suggest(tr)), n_trials=n_trials, gc_after_trial=True)

    best = study.best_trial
    params = dict(best.params)
    params.update(best.user_attrs.get("derived", {}))
    states = [t.state.name for t in study.trials]
    n_complete = states.count("COMPLETE")
    n_pruned = states.count("PRUNED")
    report = {"metric": "average_precision", "sampler": "TPE",
              "pruner": "median" if prunable else "none",
              # count only trials that actually ran (enqueue_trial pre-creates WAITING
              # trials that never execute when n_trials is smaller than the warm-start set)
              "n_trials": n_complete + n_pruned, "n_complete": n_complete, "n_pruned": n_pruned,
              "best_ap": float(best.user_attrs.get("ap", best.value)),
              "best_cost": best.user_attrs.get("cost")}
    return params, report


# =============================================================================
# Static-model tuning (logreg / xgboost / histgb) - fully self-contained here
# =============================================================================
def tune_static(model_name: str, X, y, *, n_trials: int = N_TRIALS_DEFAULT,
                n_folds: int = N_FOLDS_DEFAULT, seed: int = SEED,
                val_frac: float = VAL_FRAC_DEFAULT,
                c_walk: float | None = None, c_empty: float | None = None) -> tuple[dict, dict]:
    """Tune a static model on time-ordered (X, y). Returns (hp, report). `hp` is in the
    exact schema build_pipeline / the model card expect (xgboost incl. the early-stopped
    n_estimators). `X` must be sorted by `created` (retrain passes such rows)."""
    from .training import MODEL_KIND, _default_hp, _family_feature_lists
    kind = MODEL_KIND[model_name]
    num, cat = _family_feature_lists(model_name)
    y = np.asarray(y)

    if kind == "xgboost":
        evaluate = _xgb_earlystop_eval(num, cat, X, y, seed=seed, val_frac=val_frac,
                                       c_walk=c_walk, c_empty=c_empty)
        # Warm-start with the known-good baseline (must lie inside the tightened space above:
        # min_child_weight=5 not the _default_hp's 1, and reg_alpha at the log floor 1e-3
        # rather than 0.0 which is invalid for a log-uniform range). Tried first => can't regress.
        enqueue = [dict(max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                        min_child_weight=5, reg_lambda=1.0, reg_alpha=1e-3)]
        return run_study(lambda t: suggest_space("xgboost", t), evaluate,
                         n_trials=n_trials, seed=seed, enqueue=enqueue, prunable=False)

    # logreg / histgb: TimeSeriesSplit CV, mean PR-AUC (histgb early-stops natively).
    evaluate = _cv_eval(model_name, num, cat, X, y, n_folds=n_folds, seed=seed,
                        c_walk=c_walk, c_empty=c_empty)
    return run_study(lambda t: suggest_space(model_name, t), evaluate,
                     n_trials=n_trials, seed=seed, enqueue=[_default_hp(model_name)], prunable=True)


def _xgb_earlystop_eval(num, cat, X, y, *, seed, val_frac, c_walk, c_empty):
    """XGBoost candidate evaluator: ONE early-stopped fit on a temporal val tail, scored
    by PR-AUC; records the early-stopped tree count as derived n_estimators."""
    from xgboost import XGBClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    cut = int(len(X) * (1 - val_frac))
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                              ("num", "passthrough", num)])
    Xtr = prep.fit_transform(X.iloc[:cut]); Xva = prep.transform(X.iloc[cut:])
    ytr = np.asarray(y[:cut]); yva = np.asarray(y[cut:])

    def evaluate(trial, params):
        m = XGBClassifier(n_estimators=2000, early_stopping_rounds=50, eval_metric="aucpr",
                          tree_method="hist", objective="binary:logistic", n_jobs=-1,
                          random_state=seed, **params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        p = m.predict_proba(Xva)[:, 1]
        ap = pr_auc(yva, p)
        trial.set_user_attr("derived", {"n_estimators": int(getattr(m, "best_iteration", 0) or 0) + 1})
        trial.set_user_attr("ap", ap)
        trial.set_user_attr("cost", expected_cost(yva, p, c_walk=c_walk, c_empty=c_empty))
        return ap
    return evaluate


def _cv_eval(model_name, num, cat, X, y, *, n_folds, seed, c_walk, c_empty):
    """logreg/histgb evaluator: mean PR-AUC over a TimeSeriesSplit, with per-fold
    intermediate reporting so MedianPruner can stop clearly-bad trials early."""
    import optuna
    from sklearn.model_selection import TimeSeriesSplit
    from .training import build_pipeline
    splits = list(TimeSeriesSplit(n_splits=n_folds).split(X))

    def evaluate(trial, params):
        aps, ys, ps = [], [], []
        for step, (tr, va) in enumerate(splits):
            pipe = build_pipeline(model_name, params, num, cat, calibrate=False, seed=seed)
            pipe.fit(X.iloc[tr], y[tr])
            p = pipe.predict_proba(X.iloc[va])[:, 1]
            aps.append(pr_auc(y[va], p)); ys.append(np.asarray(y[va])); ps.append(p)
            trial.report(float(np.nanmean(aps)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        yall = np.concatenate(ys); pall = np.concatenate(ps)
        trial.set_user_attr("ap", float(np.nanmean(aps)))
        trial.set_user_attr("cost", expected_cost(yall, pall, c_walk=c_walk, c_empty=c_empty))
        return float(np.nanmean(aps))
    return evaluate
