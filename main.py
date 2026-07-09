"""CLI entry point for the overbooking analysis project.

Designed to be run from the repo root:

    uv run python main.py --help
    uv run python main.py refresh           # re-pull BigQuery and save parquet
    uv run python main.py update            # fast path: next 14 days only + score
    uv run python main.py score             # score upcoming arrivals (auto pick model)
    uv run python main.py score --model xgboost # score with a specific model
    uv run python main.py retrain --model hazard        # retrain (refit, frozen HP)
    uv run python main.py retrain --model xgboost --retune  # retrain + HP search
    uv run python main.py status            # show which models / parquets exist

For day-to-day work the notebooks are the primary interface; this CLI exists
so the daily scoring can be wired to a cron job or scheduled task without
opening Jupyter.
"""

from __future__ import annotations

import argparse
import sys

from src import (
    HIGH_THR,
    LOW_THR,
    data_dir,
    list_available_models,
    load_clean_reservations,
    load_property_performance,
    load_reservations,
    resolve_model,
    score_upcoming,
)
from src.model_eval import EVAL_MODELS, model_eval


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

    n_high      = int((scored["risk_bucket"] == "high").sum())
    n_uncertain = int((scored["risk_bucket"] == "uncertain").sum())
    n_low       = int((scored["risk_bucket"] == "low").sum())
    print(f"\n  rows scored          : {len(scored):,}")
    print(f"  high risk (≥{HIGH_THR:.0%})    : {n_high:,}")
    print(f"  uncertain ({LOW_THR:.0%}–{HIGH_THR:.0%}) : {n_uncertain:,}")
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
    """Pre-warm the global SHAP artifact(s) the Model-Performance page reads (beeswarm +
    feature importance). Model-agnostic over the scalar P(cancel) adapter, so it can be
    slow — run offline / in the Docker build. Requires a scored set (`main.py score`)."""
    from dash_app.backend.explain import compute_global_shap, iteration_curve, shap_cache_path

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
        if m in ("xgboost", "histgb"):                       # warm the boosting iteration curve too
            try:
                iteration_curve(m, refresh=args.refresh)
                print(f"  iteration curve warmed for '{m}'.")
            except Exception as e:  # noqa: BLE001
                print(f"  (iteration curve skipped for '{m}': {e})", file=sys.stderr)
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    """FAST PATH: pull ONLY the next N days of arrivals from BigQuery and score them.

    This is the CLI twin of the "Update scores" button — same logic (src.data_loader
    windowed pull + src.scoring), so a scheduler can run it without the UI or any code
    duplication. Writes Data/scored_upcoming.parquet (the shared scored set).
    """
    from src import resolve_model, score_reservations
    from src.data_loader import load_upcoming_window

    try:
        chosen = resolve_model(args.model)
    except (KeyError, FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    source = "cache" if args.cache_only else "BigQuery (date-filtered)"
    print(f"Fast update: pulling next {args.days} days from {source}, scoring with '{chosen}'…")
    df = load_upcoming_window(days=args.days, force_refresh=not args.cache_only, quiet=False)
    if "status" in df.columns:
        df = df[df["status"].astype("string") != "Canceled"].copy()
    if df.empty:
        print("No upcoming arrivals in the window.")
        return 0

    scored = score_reservations(df, model_name=chosen, save_as="scored_upcoming.parquet")
    n_high = int((scored["risk_bucket"] == "high").sum())
    print(f"  scored {len(scored):,} bookings ({n_high:,} high-risk) "
          f"→ {data_dir() / 'scored_upcoming.parquet'}")
    return 0


def cmd_retrain(args: argparse.Namespace) -> int:
    """Retrain a model on all resolved data. Default 'refit' keeps the frozen card
    hyperparameters; --retune re-searches them. Delegates to src.training.retrain (which
    dispatches the hazard model to src.hazard.retrain_hazard) — no logic is duplicated here,
    so this CLI, the app button, and a future scheduler all share the same code path.
    """
    from src.training import retrain

    mode = "retune" if args.retune else "refit"
    print(f"Retraining '{args.model}' (mode={mode}) on all resolved data"
          + (f" as-of {args.asof}" if args.asof else "") + "…")
    try:
        res = retrain(args.model, mode=mode, asof=args.asof, persist=not args.dry_run)
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
    fc = res.get("feature_change") or {}
    if fc.get("changed"):
        print(f"  feature change : added={fc.get('added')} removed={fc.get('removed')}")
    if res.get("persisted"):
        print(f"  saved          : {res['persisted'].get('joblib')}")
    elif args.dry_run:
        print("  (dry run — nothing persisted)")
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
    ps.add_argument("--model", choices=["hazard", "xgboost"],
                    help="Which model to use. Default: hazard (falls back to xgboost "
                         "if the hazard artifact is missing).")
    ps.add_argument("--refresh", action="store_true",
                    help="Re-pull BigQuery before scoring.")
    ps.set_defaults(func=cmd_score)

    pst = sub.add_parser("status", help="Show what's on disk and which model wins.")
    pst.set_defaults(func=cmd_status)

    pu = sub.add_parser("update", help="Fast path: pull next N days of arrivals + score them.")
    pu.add_argument("--model", choices=["hazard", "xgboost"],
                    help="Which model to score with. Default: hazard (fallback xgboost).")
    pu.add_argument("--days", type=int, default=14,
                    help="Forward arrival window in days (default 14).")
    pu.add_argument("--cache-only", action="store_true",
                    help="Score from the cached parquet without a BigQuery pull.")
    pu.set_defaults(func=cmd_update)

    prt = sub.add_parser("retrain", help="Retrain a model on all resolved data (refit/retune).")
    prt.add_argument("--model", required=True, choices=list(EVAL_MODELS),
                     help="Which model to retrain.")
    prt.add_argument("--retune", action="store_true",
                     help="Re-search hyperparameters (default: keep the frozen card HP).")
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

    px = sub.add_parser("explain", help="Pre-warm global SHAP (beeswarm + importance).")
    px.add_argument("--model", choices=list(EVAL_MODELS),
                    help="Which model to explain. Omit and pass --all for every model.")
    px.add_argument("--all", action="store_true", help="Explain all four models.")
    px.add_argument("--refresh", action="store_true",
                    help="Recompute even if the SHAP parquet already exists.")
    px.set_defaults(func=cmd_explain)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
