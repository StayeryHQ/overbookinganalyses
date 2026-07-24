"""CLI entry point for the overbooking analysis project.

Designed to be run from the repo root:

    uv run python main.py --help
    uv run python main.py refresh           # re-pull BigQuery and save parquet
    uv run python main.py update            # BigQuery refresh (both tables) + score next 14 days
    uv run python main.py bqcheck           # probe the BigQuery connection
    uv run python main.py score             # score upcoming arrivals (auto pick model)
    uv run python main.py score --model xgboost # score with a specific model
    uv run python main.py retrain --model hazard        # retrain (refit, frozen HP)
    uv run python main.py retrain --model xgboost --retune  # retrain + HP search
    uv run python main.py refresh-all       # FULL serving refresh: BigQuery → retrain ALL
                                            #   → bake-off → eval → SHAP/PDP → score (progress bar)
    uv run python main.py status            # show which models / parquets exist

For day-to-day work the notebooks are the primary interface; this CLI exists
so the daily scoring can be wired to a cron job or scheduled task without
opening Jupyter. `refresh-all` is the one-command "rebuild everything for serving"
pipeline (its logic lives in dash_app.backend.model_ops.full_serving_refresh, the
SAME function a future in-app button would call, so the app and CLI never drift).
"""

from __future__ import annotations

import argparse
import sys

from src import (
    data_dir,
    list_available_models,
    load_clean_reservations,
    load_property_performance,
    load_reservations,
    load_risk_buckets,
    resolve_model,
    score_upcoming,
)
from src.model_eval import EVAL_MODELS, model_eval
from src.scoring import SERVEABLE_MODELS


def cmd_refresh(args: argparse.Namespace) -> int:
    """Force a fresh BigQuery pull and rebuild BOTH parquet caches (reservations +
    property_performance_daily)."""
    print("Re-querying BigQuery: reservations (PII stripped before caching)…")
    resv = load_reservations(force_refresh=True, quiet=False)
    print(f"  reservations cache          : rows={len(resv):,}  cols={resv.shape[1]}")

    print("Re-querying BigQuery: property_performance_daily (operational columns only)…")
    perf = load_property_performance(force_refresh=True, quiet=False)
    print(f"  property_performance cache  : rows={len(perf):,}  cols={perf.shape[1]}")
    print(f"\nFresh caches written to {data_dir()}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Score upcoming arrivals with the chosen model (default: hazard, fallback xgboost)."""
    try:
        chosen = resolve_model(args.model)
    except (KeyError, FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Scoring upcoming arrivals with model '{chosen}'…")
    scored = score_upcoming(model_name=chosen, force_refresh=args.refresh, save=True)
    if scored.empty:
        print("No upcoming arrivals to score.")
        return 0

    rb = load_risk_buckets()                       # THE one risk scale: {high_cutoff, labels}
    high_cut = float(rb["high_cutoff"])            # fixed High cutoff (0.85)
    # Low/Medium boundary is the DYNAMIC cost-optimal threshold, carried on each scored row
    # (not a config value any more — see configs/risk_buckets.yaml / src.scoring).
    thr = (float(scored["cancel_threshold"].iloc[0])
           if "cancel_threshold" in scored.columns and len(scored) else None)
    n_high   = int((scored["risk_bucket"] == "high").sum())
    n_medium = int((scored["risk_bucket"] == "medium").sum())
    n_low    = int((scored["risk_bucket"] == "low").sum())
    print(f"\n  rows scored          : {len(scored):,}")
    print(f"  high (≥{high_cut:.0%})          : {n_high:,}")
    if thr is not None:
        print(f"  medium ({thr:.0%}–{high_cut:.0%})    : {n_medium:,}")
    else:
        print(f"  medium               : {n_medium:,}")
    print(f"  low                  : {n_low:,}")
    print(f"\nResult saved to: {data_dir() / 'scored_upcoming.parquet'}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show which models and parquet files exist on disk."""
    d = data_dir()
    print("Data folder:", d)
    parquets = sorted(d.glob("*.parquet"))
    if not parquets:
        print("  (no parquets yet)")
    for p in parquets:
        mb = p.stat().st_size / 1_048_576
        print(f"  {p.name:<45} {mb:>7.2f} MB")

    models = list_available_models()
    print(f"\nModels on disk          : {models or '(none)'}")
    if models:
        # Which model score/score_upcoming would actually use right now.
        print(f"Default scoring model   : {resolve_model()}")

    try:
        df = load_clean_reservations()
        # `status` in the cleaned dataset is the encoded int8 target (1 = cancel at/
        # before arrival, 0 otherwise) — compare to 1, NOT the raw 'Canceled' string.
        pos_share = (df["status"] == 1).mean()
        print(f"\nCleaned dataset         : {len(df):,} rows, "
              f"{pos_share:.2%} positive class")
    except FileNotFoundError:
        print("\nCleaned dataset         : (not built — run 00_data_audit.ipynb)")

    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Pre-warm the per-model evaluation artifact(s) the Model-Performance page reads.

    Runs the leak-free decision-time walk-forward once per model and writes
    Data/model_eval_<model>.parquet (+ provenance JSON). Static models are fast; the
    hazard refit is the slow one. Run this offline / in the Docker build so the app
    only ever READS the parquet.
    """
    if args.all:
        models = list(EVAL_MODELS)
    elif args.model:
        models = [args.model]
    else:
        print("ERROR: pass --model <name> or --all", file=sys.stderr)
        return 1

    for m in models:
        print(f"Evaluating '{m}' (leak-free decision-time walk-forward, {args.folds} folds)…")
        try:
            d = model_eval(m, refresh=args.refresh, n_folds=args.folds)
        except Exception as e:  # noqa: BLE001 — surface the reason, keep going
            print(f"  FAILED for '{m}': {e}", file=sys.stderr)
            continue
        if d.empty:
            print(f"  no usable folds/data for '{m}'.")
            continue
        print(f"  pooled n={len(d):,}  folds={d['fold'].nunique()}  "
              f"base_rate={d['y_true'].mean():.3f}  mean_pred={d['y_prob'].mean():.3f}")
        print(f"  saved → {data_dir() / f'model_eval_{m}.parquet'}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """Pre-warm the global explanation artifacts the Model-Performance / Occupancy pages read
    (SHAP beeswarm + importance, and the per-feature PDP cache). Model-agnostic over the scalar
    P(cancel) adapter, so it can be slow — run offline / in the Docker build. Requires a scored
    set (`main.py score`)."""
    from dash_app.backend.explain import (compute_all_pdp, compute_global_shap,
                                          iteration_curve, pdp_cache_path, shap_cache_path)

    if args.all:
        models = list(EVAL_MODELS)
    elif args.model:
        models = [args.model]
    else:
        print("ERROR: pass --model <name> or --all", file=sys.stderr)
        return 1

    for m in models:
        print(f"Computing global SHAP for '{m}' (model-agnostic; may take a while)…")
        try:
            d = compute_global_shap(m, refresh=args.refresh)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED for '{m}': {e}", file=sys.stderr)
            continue
        if d.empty:
            print(f"  nothing to explain for '{m}' (is anything scored? run `main.py score`).")
            continue
        print(f"  features={d['feature'].nunique()}  rows={len(d):,}  saved → {shap_cache_path(m)}")
        # Build the per-feature partial-dependence cache the Occupancy page reads read-only
        # (pdp_<model>.json). Same offline pre-warm as SHAP; the page never recomputes PDP live.
        try:
            pdp = compute_all_pdp(m, refresh=args.refresh)
            print(f"  PDP cache built for '{m}' ({len(pdp)} features) → {pdp_cache_path(m)}")
        except Exception as e:  # noqa: BLE001
            print(f"  (PDP skipped for '{m}': {e})", file=sys.stderr)
        if m in ("xgboost", "histgb"):                       # warm the boosting iteration curve too
            try:
                iteration_curve(m, refresh=args.refresh)
                print(f"  iteration curve warmed for '{m}'.")
            except Exception as e:  # noqa: BLE001
                print(f"  (iteration curve skipped for '{m}': {e})", file=sys.stderr)
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    """THE combined data update — CLI twin of the app's "Update data & scores" button.

    One strict BigQuery pull per table (reservations full history + property
    performance), then the next N days are scored from the fresh data. No cache
    fallback: a BigQuery failure exits non-zero so a scheduler notices.
    """
    from src import refresh_and_score

    def progress(msg: str, frac: float) -> None:
        print(f"  [{frac:>4.0%}] {msg}")

    try:
        res = refresh_and_score(args.model, days=args.days, progress=progress)
    except Exception as e:  # noqa: BLE001 — loud, non-zero exit for schedulers
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    b = res["buckets"]
    print(f"\n  reservations pulled  : {res['reservations_rows']:,} rows "
          f"(data as of {res['data_max_created']})")
    print(f"  performance pulled   : {res['perf_rows']:,} rows")
    print(f"  scored ({res['model_used']})      : {res['scored_rows']:,} bookings "
          f"— high {b['high']:,} / medium {b['medium']:,} / low {b['low']:,}")
    print(f"\nResult saved to: {data_dir() / 'scored_upcoming.parquet'}")
    return 0


def cmd_bqcheck(args: argparse.Namespace) -> int:
    """Full BigQuery diagnosis: credential resolution facts + a live probe."""
    from src import bigquery_healthcheck
    from src.data_loader import bigquery_diagnose

    print("--- credential resolution ---")
    for line in bigquery_diagnose():
        print(" ", line)
    print("--- live probe ---")
    res = bigquery_healthcheck()
    print(("OK   " if res["ok"] else "FAIL ") + res["detail"])
    if not res["ok"]:
        print("\nMost common fixes:\n"
              "  1) The sibling project probably uses a service-account key — reuse it:\n"
              "         export GCP_SERVICE_ACCOUNT_JSON_FILE=/path/to/key.json\n"
              "  2) Your gcloud quota/config project points at the DATA project. Point it\n"
              "     at YOUR OWN project (the one the sibling repo runs jobs in):\n"
              "         gcloud auth application-default set-quota-project <your-project>\n"
              "         gcloud config set project <your-project>")
    return 0 if res["ok"] else 1


def cmd_retrain(args: argparse.Namespace) -> int:
    """Retrain a model on all resolved data. Default 'refit' keeps the frozen card
    hyperparameters; --retune re-searches them. Delegates to src.training.retrain (which
    dispatches the hazard model to src.hazard.retrain_hazard) — no logic is duplicated here,
    so this CLI, the app button, and a future scheduler all share the same code path.
    """
    from src.training import retrain

    mode = "retune" if args.retune else "refit"
    print(f"Retraining '{args.model}' (mode={mode}"
          + (f", {args.trials} TPE trials" if mode == "retune" else "")
          + ") on all resolved data"
          + (f" as-of {args.asof}" if args.asof else "") + "…")
    try:
        res = retrain(args.model, mode=mode, asof=args.asof, persist=not args.dry_run,
                      n_iter=args.trials)
    except Exception as e:  # noqa: BLE001 — surface the reason, non-zero exit
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"  mode           : {res.get('mode', mode)}")
    if res.get("n_train_deploy") is not None:
        print(f"  trained on     : {res['n_train_deploy']:,} bookings (as-of {res.get('asof')})")
    elif res.get("n_books_resolved") is not None:                 # hazard result shape
        print(f"  trained on     : {res['n_books_resolved']:,} resolved bookings "
              f"({res.get('n_train_pp', '?'):,} person-periods)")
    agg = (res.get("walk_forward", {}) or {}).get("aggregate", {})
    auc = agg.get("auc", {}).get("mean") if isinstance(agg.get("auc"), dict) else None
    if auc is not None:
        print(f"  walk-forward AUC (mean): {auc:.3f}")
    if res.get("val_ap") is not None:                             # hazard reports val AP
        print(f"  val AP (person-period) : {res['val_ap']:.4f}")
    tune = res.get("tuning")                                       # TPE search report (retune only)
    if tune:
        line = (f"  HP search      : {tune.get('sampler', 'TPE')}, "
                f"{tune.get('n_complete', tune.get('n_trials'))} trials, "
                f"best PR-AUC {tune.get('best_ap'):.4f}")
        if tune.get("best_cost") is not None:
            line += f", val cost €{tune['best_cost']:,.0f}"
        print(line)
    fc = res.get("feature_change") or {}
    if fc.get("changed"):
        print(f"  feature change : added={fc.get('added')} removed={fc.get('removed')}")
    if res.get("persisted"):
        print(f"  saved          : {res['persisted'].get('joblib')}")
    elif args.dry_run:
        print("  (dry run — nothing persisted)")
    return 0


def cmd_build_roster(args: argparse.Namespace) -> int:
    """Regenerate Data/feature_roster.json from the raw cache — the runtime twin of
    notebook 00 §11. Lets a fresh deploy (or a cache rebuild) recreate the roster
    without opening Jupyter; build_clean_reservations also does this automatically when
    the roster is missing."""
    from src import (
        build_clean_reservations,
        data_dir,
        load_feature_roster,
        load_reservations,
    )

    print("Rebuilding the clean dataset from the raw cache and regenerating the feature roster…")
    try:
        raw = load_reservations()                       # cached parquet (no BigQuery)
        clean = build_clean_reservations(raw, write_roster=True)
    except FileNotFoundError as e:
        print(f"ERROR: {e}\n  Run `uv run python main.py refresh` first to build the raw cache.",
              file=sys.stderr)
        return 1
    r = load_feature_roster()
    print(f"  wrote {data_dir() / 'feature_roster.json'}")
    print(f"  clean rows           : {len(clean):,}")
    print(f"  numeric / categorical: {r['n_numeric']} / {r['n_categorical']}")
    print(f"  ratePlan map entries : {len(r.get('ratePlan_category_map', {}))}")
    return 0


def cmd_refresh_all(args: argparse.Namespace) -> int:
    """FULL serving refresh in ONE command: BigQuery history → retrain all serving models
    (HP search) → matched bake-off → eval → SHAP/PDP → score, with a staged progress bar.
    Uses the SAME code path (dash_app.backend.model_ops.full_serving_refresh) a future app
    button would use, so the app and the CLI can never drift apart."""
    from dash_app.backend.model_ops import full_serving_refresh

    def progress(msg: str, frac: float) -> None:
        filled = int(max(0.0, min(1.0, frac)) * 24)
        print(f"  [{'█' * filled}{'·' * (24 - filled)}] {frac:>4.0%}  {msg}")

    print("Full serving refresh — a multi-hour offline job with --retune (default) + bake-off.\n")
    try:
        res = full_serving_refresh(progress=progress, retune=not args.refit,
                                   bakeoff=not args.skip_bakeoff, days=args.days)
    except Exception as e:  # noqa: BLE001 — loud, non-zero exit for schedulers
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    b = res.get("buckets", {})
    print("\nDone.")
    print(f"  reservations pulled : {res['reservations_rows']:,}  (clean rows {res['clean_rows']:,})")
    print(f"  retrained ({res['mode']})   : {', '.join(res['trained'])}")
    print(f"  selected static     : {res.get('selected_static')}")
    print(f"  scored              : {res.get('scored_rows', 0):,} — high {b.get('high', 0)} / "
          f"medium {b.get('medium', 0)} / low {b.get('low', 0)}")
    if res.get("errors"):
        print(f"  non-fatal errors    : {res['errors']}")
    print(f"  elapsed             : {res['elapsed_s']}s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="overbooking-analyse",
        description="Stayery overbooking — load, score, status helpers.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("refresh", help="Re-pull BigQuery; rebuild parquet cache.")
    pr.set_defaults(func=cmd_refresh)

    ps = sub.add_parser("score", help="Score upcoming arrivals.")
    ps.add_argument("--model", choices=list(SERVEABLE_MODELS),
                    help="Which model to use (hazard / xgboost / histgb). Default: hazard "
                         "(falls back to xgboost if the hazard artifact is missing).")
    ps.add_argument("--refresh", action="store_true",
                    help="Re-pull BigQuery before scoring.")
    ps.set_defaults(func=cmd_score)

    pst = sub.add_parser("status", help="Show what's on disk and which model wins.")
    pst.set_defaults(func=cmd_status)

    pu = sub.add_parser("update", help="Combined update: BigQuery refresh (both tables) + score next N days.")
    pu.add_argument("--model", choices=list(SERVEABLE_MODELS),
                    help="Which model to score with (hazard / xgboost / histgb). "
                         "Default: hazard (fallback xgboost).")
    pu.add_argument("--days", type=int, default=14,
                    help="Forward arrival window in days (default 14).")
    pu.set_defaults(func=cmd_update)

    pb = sub.add_parser("bqcheck", help="Probe the BigQuery connection (credentials, quota project).")
    pb.set_defaults(func=cmd_bqcheck)

    pbr = sub.add_parser("build-roster",
                         help="Regenerate Data/feature_roster.json from the raw cache (nb00 §11 twin).")
    pbr.set_defaults(func=cmd_build_roster)

    prt = sub.add_parser("retrain", help="Retrain a model on all resolved data (refit/retune).")
    prt.add_argument("--model", required=True, choices=list(EVAL_MODELS),
                     help="Which model to retrain.")
    prt.add_argument("--retune", action="store_true",
                     help="Re-search hyperparameters with the shared TPE search (default: "
                          "keep the frozen card HP).")
    prt.add_argument("--trials", type=int, default=90,
                     help="Hyperparameter search budget (TPE trials) when --retune is set "
                          "(default 90). Ignored on a plain refit.")
    prt.add_argument("--asof", default=None,
                     help="Only use data resolved by this date (YYYY-MM-DD). Default: latest.")
    prt.add_argument("--dry-run", action="store_true",
                     help="Fit + report metrics but do NOT persist the artifact/card.")
    prt.set_defaults(func=cmd_retrain)

    pe = sub.add_parser("eval", help="Pre-warm the Model-Performance eval artifact(s).")
    pe.add_argument("--model", choices=list(EVAL_MODELS),
                    help="Which model to evaluate. Omit and pass --all for every model.")
    pe.add_argument("--all", action="store_true", help="Evaluate all four models.")
    pe.add_argument("--folds", type=int, default=6,
                    help="Walk-forward folds to pool (default 6). More = larger sample, slower.")
    pe.add_argument("--refresh", action="store_true",
                    help="Recompute even if the parquet already exists.")
    pe.set_defaults(func=cmd_eval)

    px = sub.add_parser("explain", help="Pre-warm global SHAP (beeswarm + importance) + PDP cache.")
    px.add_argument("--model", choices=list(EVAL_MODELS),
                    help="Which model to explain. Omit and pass --all for every model.")
    px.add_argument("--all", action="store_true", help="Explain all four models.")
    px.add_argument("--refresh", action="store_true",
                    help="Recompute even if the SHAP parquet already exists.")
    px.set_defaults(func=cmd_explain)

    pfa = sub.add_parser("refresh-all",
                         help="FULL serving refresh: BigQuery → retrain all → bake-off → eval → "
                              "SHAP/PDP → score, with a staged progress bar.")
    pfa.add_argument("--refit", action="store_true",
                     help="Reuse frozen card hyperparameters instead of a full HP re-search (faster).")
    pfa.add_argument("--skip-bakeoff", action="store_true",
                     help="Skip the matched bake-off (priciest stage); best_model then selects "
                          "from the freshly-retrained cards.")
    pfa.add_argument("--days", type=int, default=14,
                     help="Forward arrival window to score at the end (default 14).")
    pfa.set_defaults(func=cmd_refresh_all)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
