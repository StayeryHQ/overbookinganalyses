"""CLI entry point for the overbooking analysis project.

Designed to be run from the repo root:

    uv run python main.py --help
    uv run python main.py refresh           # re-pull BigQuery and save parquet
    uv run python main.py score             # score upcoming arrivals (auto pick model)
    uv run python main.py score --model xgboost # score with a specific model
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

    pe = sub.add_parser("eval", help="Pre-warm the Model-Performance eval artifact(s).")
    pe.add_argument("--model", choices=list(EVAL_MODELS),
                    help="Which model to evaluate. Omit and pass --all for every model.")
    pe.add_argument("--all", action="store_true", help="Evaluate all four models.")
    pe.add_argument("--folds", type=int, default=6,
                    help="Walk-forward folds to pool (default 6). More = larger sample, slower.")
    pe.add_argument("--refresh", action="store_true",
                    help="Recompute even if the parquet already exists.")
    pe.set_defaults(func=cmd_eval)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
