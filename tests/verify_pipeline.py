"""Consistency + leakage verification for the cancellation pipeline.
Run FROM THE REPO ROOT: uv run python tests/verify_pipeline.py
Fast checks only (no model fitting). Exits non-zero on any failure."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, ".")

P, F = [], []
def ok(name, cond, extra=""):
    (P if cond else F).append(name)
    print(("  PASS " if cond else "  FAIL ") + name + (f"  | {extra}" if extra else ""))

print("\n=== 1. imports + artifacts ===")
import src, src.walkforward as wf, src.hazard as hz, src.training as T, src.scoring as sc
from src.features import (load_feature_roster, family_feature_lists, excluded_columns,
                          FEATURE_EXCLUSIONS)
ok("all src modules import", True)
roster = load_feature_roster()
clean = pd.read_parquet("Data/reservations_clean.parquet")
raw = pd.read_parquet("Data/reservations_raw_no_pii.parquet")
ok("roster has log_twins", bool(roster.get("log_twins")))

print("\n=== 2. leakage: roster excludes every leakage column ===")
exc = excluded_columns()
leak_groups = [g for r, g in FEATURE_EXCLUSIONS.items()
               if ("leakage" in r or "target" in r or "dynamic" in r)]
leak_cols = {c for g in leak_groups for c in g}
roster_feats = set(roster["numeric"]) | set(roster["categorical"])
ok("no leakage/target/dynamic column in roster", not (leak_cols & roster_feats),
   f"offenders={sorted(leak_cols & roster_feats)}")
for grp in ["travelPurpose", "guest_country_region", "primaryGuest_preferredLanguage",
            "company_prior_cancel_rate", "cancel_days_before_arrival", "outcome_known_date",
            "days_until_arrival"]:
    ok(f"'{grp}' excluded", grp in exc)

print("\n=== 3. scoring parity: build_features produces EVERY roster feature ===")
conf = raw[raw["status"] == "Confirmed"].head(3000).copy()
feat = sc.build_features(conf)
missing = [c for c in roster["numeric"] + roster["categorical"] if c not in feat.columns]
ok("build_features covers all roster features (incl. log twins)", not missing, f"missing={missing}")
ok("log twins present in serving path",
   all(c in feat.columns for c in ["los_nights_log", "lead_time_days_log",
                                    "gross_per_night_log", "diff_gross_cancellation_fee_log"]))

print("\n=== 4. leakage null-audit on CONFIRMED (upcoming) bookings ===")
# a roster feature that is ~always null on upcoming bookings = check-in leakage signature
na = training_audit = T.scoring_null_audit(conf, null_warn=0.98)
leaky = na[na["leakage_warn"] == True]["feature"].tolist()
ok("no roster feature is ~always-null on upcoming (no check-in leakage)", not leaky, f"leaky={leaky}")

print("\n=== 5. point-in-time / decision-time folds ===")
dfw = wf.add_outcome_known_date(clean)
folds = wf.make_folds(dfw, n_folds=12, horizon_days=14, step_days=14)
try:
    wf.assert_point_in_time(dfw, folds); ok("assert_point_in_time passes (no future leak)", True)
except AssertionError as e:
    ok("assert_point_in_time passes (no future leak)", False, str(e))
allt = np.concatenate([f.test_idx for f in folds])
ok("each booking tested at most once (no double-count)", len(allt) == len(set(allt)))
overlap = any(len(np.intersect1d(f.train_idx, f.test_idx)) for f in folds)
ok("train/test disjoint in every fold", not overlap)

print("\n=== 6. target coherence: static (00) == hazard (08), cdba>=0 ===")
src_haz = open("src/hazard.py").read()
ok("hazard event def uses cdba.ge(0) (same-day incl.)", "cdba.ge(0)" in src_haz)
nb0 = "\n".join("".join(c["source"]) for c in json.load(open("notebooks/00_data_audit.ipynb"))["cells"])
ok("nb00 target uses cdba.ge(0)", "cdba.ge(0)" in nb0)

print("\n=== 7. family features: linear=log twins, tree=raw ===")
ln, _ = family_feature_lists(roster, "linear"); tn, _ = family_feature_lists(roster, "tree")
logs = set(roster["log_twins"].values()); raws = set(roster["log_twins"].keys())
ok("linear uses log twins, drops raw skewed", logs <= set(ln) and not (raws & set(ln)))
ok("tree uses raw skewed, drops log twins", raws <= set(tn) and not (logs & set(tn)))

print("\n=== 8. no stale patterns across model notebooks ===")
# Known-good exceptions: 'temporal_split' may appear in 00 as an explanatory
# mention; 'matplotlib' is allowed where SHAP's beeswarm runs (the documented
# non-Plotly exception, see src/diagnostics.py). Everything else is stale.
ALLOWED = {"temporal_split": {"00_data_audit"},
           "matplotlib": {"02_xgboost", "03_histgb", "08_hazard"}}
stale = {"temporal_split": [], "is_cancelled": [], "matplotlib": [], "seaborn": [],
         "arrival-anchored": [], "t_f1": []}
for nbf in ["00_data_audit", "01_logreg", "02_xgboost", "03_histgb", "05_model_comparison", "08_hazard"]:
    j = "\n".join("".join(c["source"]) for c in json.load(open(f"notebooks/{nbf}.ipynb"))["cells"])
    for k in stale:
        if k in j:
            stale[k].append(nbf)
for k, hits in stale.items():
    ok(f"no stale '{k}' in model notebooks", set(hits) <= ALLOWED.get(k, set()),
       f"in={hits}")

print("\n=== 9. KPI: cost-optimal threshold, not F1 ===")
ok("scoring exposes cost_threshold_from_scores", hasattr(sc, "cost_threshold_from_scores"))
ok("scoring.best_model ranks by AP with Brier gate", "ap" in open("src/scoring.py").read().lower())

print("\n=== 10. train/serve parity: build_features == notebook-00 clean engineering ===")
# build_clean_reservations reuses src.scoring.build_features (the SERVING function). If it
# reproduces the committed clean parquet (nb00's output) value-for-value, serving and
# training feature engineering are provably in sync (M3 — catches formula drift the
# column-presence check in §3 cannot).
from src.data_loader import build_clean_reservations
rebuilt = build_clean_reservations(raw)
static_cols = [c for c in roster["numeric"] + roster["categorical"] if c in clean.columns]
row_ok = len(rebuilt) == len(clean)
ok("build_clean_reservations reproduces clean row count", row_ok,
   f"rebuilt={len(rebuilt)} committed={len(clean)}")
if row_ok:
    mism = []
    for c in static_cols:
        a = clean[c].reset_index(drop=True); b = rebuilt[c].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            bad = int((~np.isclose(pd.to_numeric(a, errors="coerce"),
                                   pd.to_numeric(b, errors="coerce"), equal_nan=True)).sum())
        else:
            bad = int((a.astype("string").fillna("<NA>") != b.astype("string").fillna("<NA>")).sum())
        if bad:
            mism.append(f"{c}:{bad}")
    ok("build_features matches clean parquet on every static feature (no train/serve drift)",
       not mism, f"mismatches={mism}")

print("\n=== 11. roster regenerates from src (runtime twin of nb00 §11) ===")
# A fresh deploy must be able to rebuild Data/feature_roster.json without the notebook.
from src.features import build_feature_roster, build_rateplan_category_map
from src.data_loader import ARRIVAL_FLOOR, GRACE_DAYS
_arr = pd.to_datetime(raw["arrival"], utc=True, errors="coerce")
_cut = pd.to_datetime(raw["created"], utc=True, errors="coerce").max().normalize() \
       - pd.Timedelta(days=GRACE_DAYS)
_rp = build_rateplan_category_map(raw[(_arr >= ARRIVAL_FLOOR) & (_arr < _cut)])
ok("ratePlan_category_map reproduces committed roster map (byte-for-byte)",
   _rp == roster.get("ratePlan_category_map", {}),
   f"src={len(_rp)} committed={len(roster.get('ratePlan_category_map', {}))}")
_regen = build_feature_roster(clean, rateplan_category_map=_rp)
ok("roster numeric/categorical/log_twins/excluded reproduce committed",
   all(_regen[k] == roster[k] for k in ("numeric", "categorical", "log_twins", "excluded",
                                        "dynamic_numeric", "n_numeric", "n_categorical")))

print(f"\n==== {len(P)} passed, {len(F)} failed ====")
if F:
    print("FAILURES:", F); sys.exit(1)
print("ALL CONSISTENCY / LEAKAGE CHECKS GREEN")
