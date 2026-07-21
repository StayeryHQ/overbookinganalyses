# dash_app/backend/explain.py
# Explainability layer for the XAI / Model-Performance page: SHAP (global beeswarm +
# per-feature importance + single-booking waterfall), partial dependence / ICE, and the
# boosting iteration curve.
#
# Design decision (freigegeben, "B1"): SHAP is computed MODEL-AGNOSTICALLY on the SAME
# scalar the four models are compared on  P(cancel by arrival) via src.scoring.cancel_proba
# (for the hazard model that is the survival-product output). ONE code path for all models:
#   * it keeps the hazard explanation on the same estimand as the classifiers (comparable
#     beeswarms), which is exactly why SurvSHAP is NOT needed here  the decision target is
#     already a scalar;
#   * it never applies the classifiers' SHAP code unchanged to the hazard model.
# Trade-off: the model-agnostic explainer is slow, so global SHAP is PRE-COMPUTED per model
# (`python main.py explain`) and cached to Data/shap_<model>.parquet; the single-booking
# waterfall is light enough to run live.
#
# NB (verify in your env): the exact shap API is pinned by the installed version
# (shap 0.49/0.52). This uses shap.Explainer(f, masker=shap.maskers.Independent(...)),
# shap.maskers.Independent and the Explanation.values / .base_values attributes  please
# confirm these against `shap.__version__` if a call signature has drifted.

from __future__ import annotations

import numpy as np
import pandas as pd

import src
from src import scoring as sc
from src import model_eval as me

# Sample sizes  kept modest so the agnostic explainer stays tractable; raise for a
# sharper global picture at the cost of compute (pre-warm handles the wait).
BG_SAMPLE = 100        # background rows for the masker
GLOBAL_SAMPLE = 300    # rows whose SHAP values populate the beeswarm
SEED = 42


# ---------------------------------------------------------------------------
# Cache location
# ---------------------------------------------------------------------------
def shap_cache_path(model: str):
    return src.data_dir() / f"shap_{model}.parquet"


def shap_available(model: str) -> bool:
    return shap_cache_path(model).exists()


# ---------------------------------------------------------------------------
# Feature set + numeric encoding for the agnostic explainer
# ---------------------------------------------------------------------------
def _feature_lists(model: str) -> tuple[list[str], list[str], list[str]]:
    """(numeric, categorical, extra) feature columns the model consumes. `extra` carries
    the hazard day-axis (days_until_arrival), a genuine feature of that model."""
    if model == "hazard":
        from src import hazard as hz
        h = hz.load_hazard()
        return list(h["num"]), list(h["cat"]), [hz.AXIS]
    num, cat = sc.model_feature_lists()
    return num, cat, []


def _encode(feat: pd.DataFrame, num, cat, extra):
    """Numeric matrix (categoricals -> integer codes) + the decode maps. The agnostic
    masker samples ACTUAL background rows, so codes stay valid integers under perturbation."""
    cols = list(num) + list(extra) + list(cat)
    mats, cat_maps = [], {}
    for c in num + extra:
        mats.append(pd.to_numeric(feat[c], errors="coerce").astype(float).to_numpy())
    for c in cat:
        s = feat[c].astype("category")
        cat_maps[c] = list(s.cat.categories)
        mats.append(s.cat.codes.astype(float).to_numpy())
    return np.column_stack(mats), cols, cat_maps


def _decoder(cols, cat_maps, num, extra):
    n_num = len(num) + len(extra)

    def decode(X: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(X, columns=cols)
        for c, cats in cat_maps.items():
            codes = np.rint(pd.to_numeric(df[c], errors="coerce").to_numpy())
            codes = np.where(np.isnan(codes), -1, codes).astype(int)
            codes = np.clip(codes, -1, len(cats) - 1)
            df[c] = [cats[i] if i >= 0 else np.nan for i in codes]
        for c in cols[:n_num]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    return decode


def explainable_features(model: str) -> list[str]:
    """Model features available for PDP/ICE on the current scored frame (numeric + day-axis
    first, then categoricals). Empty if nothing is scored yet."""
    num, cat, extra = _feature_lists(model)
    frame = _explain_frame(1)
    have = set(frame.columns) if not frame.empty else set()
    return [c for c in list(num) + list(extra) + list(cat) if (not have or c in have)]


def _predict_fn(model: str, decode):
    """f: numeric-matrix -> P(cancel) via the unified adapter (survival product for hazard)."""
    def f(X: np.ndarray) -> np.ndarray:
        return np.asarray(sc.cancel_proba(model, decode(np.atleast_2d(X))), dtype=float)
    return f


# ---------------------------------------------------------------------------
# Explanation sample (upcoming scored bookings  what the model does "now")
# ---------------------------------------------------------------------------
def _explain_frame(n: int | None = None) -> pd.DataFrame:
    """Rows to explain: the cached scored upcoming bookings (already carry the engineered
    features). Falls back to empty if nothing is scored yet."""
    from dash_app.backend import data_access as da
    df = da.load_scored()
    if df.empty:
        return df
    if n is not None and len(df) > n:
        df = df.sample(n, random_state=SEED).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Global SHAP (beeswarm + importance)  pre-computed, cached
# ---------------------------------------------------------------------------
def compute_global_shap(model: str, *, refresh: bool = False,
                        bg: int = BG_SAMPLE, sample: int = GLOBAL_SAMPLE) -> pd.DataFrame:
    """Long frame [feature, shap, fval_norm] for the beeswarm, cached per model. Uses the
    model-agnostic explainer over the scalar adapter. Heavy → pre-warm offline."""
    path = shap_cache_path(model)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    import shap

    frame = _explain_frame(max(bg, sample))
    if frame.empty:
        return pd.DataFrame(columns=["feature", "shap", "fval_norm"])
    num, cat, extra = _feature_lists(model)
    have = [c for c in num + cat + extra if c in frame.columns]
    missing = [c for c in num + cat + extra if c not in frame.columns]
    if missing:
        # Never fabricate: explain only what the scored frame actually carries.
        num = [c for c in num if c in have]; cat = [c for c in cat if c in have]
        extra = [c for c in extra if c in have]

    X, cols, cat_maps = _encode(frame, num, cat, extra)
    decode = _decoder(cols, cat_maps, num, extra)
    f = _predict_fn(model, decode)

    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(len(X), size=min(bg, len(X)), replace=False)
    ex_idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    masker = shap.maskers.Independent(X[bg_idx])
    explainer = shap.Explainer(f, masker)
    sv = explainer(X[ex_idx])                      # Explanation: .values (n, n_features)
    vals = np.asarray(sv.values)

    # normalise each feature's value to [0,1] for the beeswarm colour (per column min-max).
    Xe = X[ex_idx]
    long = []
    for j, c in enumerate(cols):
        col = Xe[:, j].astype(float)
        rng_j = np.nanmax(col) - np.nanmin(col)
        norm = (col - np.nanmin(col)) / rng_j if rng_j > 0 else np.full_like(col, 0.5)
        long.append(pd.DataFrame({"feature": c, "shap": vals[:, j], "fval_norm": norm}))
    out = pd.concat(long, ignore_index=True)
    if not out.empty:
        out.to_parquet(path, index=False)
    return out


def importance_from_shap(model: str) -> pd.DataFrame:
    """mean |SHAP| per feature (the common cross-model importance basis)."""
    bee = compute_global_shap(model) if shap_available(model) else pd.DataFrame()
    if bee is None or bee.empty:
        return pd.DataFrame(columns=["feature", "importance"])
    imp = (bee.assign(a=bee["shap"].abs()).groupby("feature")["a"].mean()
           .reset_index().rename(columns={"a": "importance"}))
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def global_beeswarm(model: str) -> pd.DataFrame:
    return compute_global_shap(model) if shap_available(model) else pd.DataFrame()


# ---------------------------------------------------------------------------
# Single-booking SHAP (live)  the reusable unit (see components/shap_explain)
# ---------------------------------------------------------------------------
def single_contribution(model: str, booking: pd.Series | pd.DataFrame,
                        *, bg: int = BG_SAMPLE) -> dict:
    """Waterfall contribution for ONE booking against a background sample. Returns
    {'base', 'pred', 'items':[{feature,value,shap}]}. Light enough to run live."""
    import shap
    row = booking.to_frame().T if isinstance(booking, pd.Series) else booking.iloc[[0]]
    num, cat, extra = _feature_lists(model)
    have = [c for c in num + cat + extra if c in row.columns]
    if not have:
        return {}
    num = [c for c in num if c in have]; cat = [c for c in cat if c in have]
    extra = [c for c in extra if c in have]

    frame = _explain_frame(bg)
    if frame.empty:
        return {}
    Xb, cols, cat_maps = _encode(frame, num, cat, extra)
    Xr, _, _ = _encode(row, num, cat, extra)
    decode = _decoder(cols, cat_maps, num, extra)
    f = _predict_fn(model, decode)

    masker = shap.maskers.Independent(Xb)
    explainer = shap.Explainer(f, masker)
    sv = explainer(Xr)
    vals = np.asarray(sv.values)[0]
    base = float(np.asarray(sv.base_values).ravel()[0])
    pred = float(f(Xr)[0])
    items = [{"feature": c, "value": _fmt_val(row.iloc[0].get(c)), "shap": float(vals[j])}
             for j, c in enumerate(cols)]
    return {"base": base, "pred": pred, "items": items}


def _fmt_val(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


# ---------------------------------------------------------------------------
# Partial dependence / ICE  manual sweep through the adapter (uniform, all models)
# ---------------------------------------------------------------------------
def partial_dependence(model: str, feature: str, *, n_rows: int = 120,
                       n_grid: int = 20, ice: bool = True) -> dict:
    """Average P(cancel) as `feature` is swept across its observed range, holding the other
    features at each sampled row's values (manual PD  correct for every model via the
    adapter, no sklearn estimator required for the hazard model). Optional ICE lines."""
    frame = _explain_frame(n_rows)
    if frame.empty or feature not in frame.columns:
        return {}
    num, cat, extra = _feature_lists(model)
    keep = [c for c in num + cat + extra if c in frame.columns]
    base = frame[keep].reset_index(drop=True)
    col = base[feature]
    if pd.api.types.is_numeric_dtype(col):
        lo, hi = np.nanpercentile(pd.to_numeric(col, errors="coerce"), [5, 95])
        grid = np.linspace(lo, hi, n_grid) if hi > lo else np.array([lo])
    else:
        grid = pd.Series(col.astype("category").cat.categories).tolist()
    pd_vals, ice_lines = [], []
    per_row = np.zeros((len(base), len(grid)))
    for gi, v in enumerate(grid):
        tmp = base.copy()
        tmp[feature] = v
        p = np.asarray(sc.cancel_proba(model, tmp), dtype=float)
        per_row[:, gi] = p
        pd_vals.append(float(np.mean(p)))
    if ice:
        ice_lines = per_row.tolist()
    x = [float(g) if isinstance(g, (int, float, np.floating)) else str(g) for g in grid]
    return {"feature": feature, "x": x, "pd": pd_vals, "ice": ice_lines if ice else None}


# ---------------------------------------------------------------------------
# Cached PDP (built on retrain, read read-only on the predictions page)  so the
# global explanations are NOT recomputed over many bookings on every rescore.
# ---------------------------------------------------------------------------
def pdp_cache_path(model: str):
    return src.data_dir() / f"pdp_{model}.json"


def pdp_available(model: str) -> bool:
    return pdp_cache_path(model).exists()


def compute_all_pdp(model: str, *, refresh: bool = False, ice_sample: int = 30) -> dict:
    """Pre-compute partial dependence for every explainable feature and cache it to
    Data/pdp_<model>.json, so the predictions page never recomputes PDP live. Called on
    retrain (alongside the SHAP rebuild). Returns {feature: {feature, x, pd, ice}}."""
    import json
    path = pdp_cache_path(model)
    if path.exists() and not refresh:
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            pass
    out: dict = {}
    for feat in explainable_features(model):
        try:
            d = partial_dependence(model, feat)
        except Exception:  # noqa: BLE001  one bad feature must not sink the whole cache
            continue
        if not d:
            continue
        ice = d.get("ice")
        if ice and len(ice) > ice_sample:               # bound the cache size
            idx = np.linspace(0, len(ice) - 1, ice_sample).astype(int)
            ice = [ice[i] for i in idx]
        out[feat] = {"feature": feat, "x": d.get("x"), "pd": d.get("pd"), "ice": ice}
    try:
        path.write_text(json.dumps(out))
    except Exception:  # noqa: BLE001
        pass
    return out


def cached_pdp(model: str, feature: str) -> dict:
    """One feature's cached PDP (built on retrain). {} if nothing cached yet."""
    import json
    path = pdp_cache_path(model)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()).get(feature, {})
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# Boosting iteration curve (XGBoost / HistGB only)  best-effort, labelled
# ---------------------------------------------------------------------------
def _itercurve_path(model: str):
    return src.data_dir() / f"itercurve_{model}.json"


def iteration_curve(model: str, *, refresh: bool = False) -> dict:
    """Train/validation loss vs boosting iteration, CACHED to JSON (computing it refits a
    model, so we never redo it on every page interaction). Only meaningful for the boosting
    models (xgboost, histgb); returns {} for logreg (no iterations) and for hazard (its curve
    is person-period logloss on a different scale  not shown next to the others)."""
    import json
    p = _itercurve_path(model)
    if p.exists() and not refresh:
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            pass
    if model not in ("xgboost", "histgb"):
        return {}
    try:
        curve = _xgb_iteration_curve() if model == "xgboost" else _histgb_iteration_curve()
    except Exception:  # noqa: BLE001  never fabricate a curve; fall through to empty
        return {}
    if curve:
        try:
            p.write_text(json.dumps(curve))
        except Exception:  # noqa: BLE001
            pass
    return curve


def _train_val_split():
    from src import training as tr
    from src import walkforward as wf
    df = wf.add_outcome_known_date(src.load_clean_reservations())
    order = pd.to_datetime(df["created"], utc=True, errors="coerce").sort_values().index
    df = df.loc[order]
    cut = int(len(df) * 0.85)
    return df.iloc[:cut], df.iloc[cut:]


def _xgb_iteration_curve() -> dict:
    from xgboost import XGBClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from src import training as tr
    num, cat = tr._family_feature_lists("xgboost")
    trn, val = _train_val_split()
    ytr = tr._target(trn).to_numpy(); yva = tr._target(val).to_numpy()
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                              ("num", "passthrough", num)])
    Xtr = prep.fit_transform(trn[num + cat]); Xva = prep.transform(val[num + cat])
    hp = tr._card_hp("xgboost")
    m = XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=-1,
                      random_state=SEED, **hp)
    m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
    ev = m.evals_result()
    tr_loss = ev["validation_0"]["logloss"]; va_loss = ev["validation_1"]["logloss"]
    return {"iters": list(range(1, len(tr_loss) + 1)), "train": tr_loss, "valid": va_loss,
            "metric": "logloss", "note": "XGBoost train/validation logloss per boosting round"}


def _histgb_iteration_curve() -> dict:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import HistGradientBoostingClassifier
    from src import training as tr
    num, cat = tr._family_feature_lists("histgb")
    trn, val = _train_val_split()
    ytr = tr._target(trn).to_numpy()
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
                              ("num", "passthrough", num)])
    Xtr = prep.fit_transform(trn[num + cat])
    hp = tr._card_hp("histgb")
    m = HistGradientBoostingClassifier(random_state=SEED, early_stopping=True,
                                        validation_fraction=0.15, n_iter_no_change=1000, **hp)
    m.fit(Xtr, ytr)
    tr_score = list(getattr(m, "train_score_", []))
    va_score = list(getattr(m, "validation_score_", []))
    if not tr_score:
        return {}
    return {"iters": list(range(1, len(tr_score) + 1)), "train": tr_score, "valid": va_score or None,
            "metric": "neg loss (higher = better)",
            "note": "HistGB internal train/validation score per iteration"}
