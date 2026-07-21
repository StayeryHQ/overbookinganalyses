"""Diagnose the Model-Performance calibration/Brier vs the notebook.

Run in the project venv:   uv run python tests/diagnose_calibration.py [--model xgboost]

It answers ONE question: is the page's eval artifact doing the same thing as the
notebook's own decision-time walk-forward (src.training.walk_forward_predict), or does my
harness (src.model_eval) diverge?

For the chosen model it prints, side by side:
  A) src.training.walk_forward_predict(model, n_folds=6)   <- the notebook's function, 6 folds
  B) src.training.walk_forward_predict(model, n_folds=12)  <- the notebook's function, 12 folds
  C) Data/model_eval_<model>.parquet                       <- what the PAGE reads

For each: n, base_rate (observed), mean_pred, Brier, and the Murphy decomposition
(reliability = calibration error; resolution = skill; BSS vs base rate), plus a 10-bin
reliability table (predicted vs observed).

How to read it:
  * If C matches A almost exactly -> my harness == the notebook function; the page just used
    fewer/more-recent folds. Rebuild with `python main.py eval --model <m> --folds 12` to
    match the notebook, and compare A(6) vs B(12) to SEE the recency/calibration drift.
  * If C is clearly worse than A at the same fold count -> my harness has a real bug; send me
    this output and I'll fix src/model_eval.py.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src import model_eval as me
from src import scoring as sc
from src import training as tr


def _summary(tag: str, y, p) -> None:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    d = sc.brier_decomposition(y, p, n_bins=10)
    print(f"\n[{tag}]  n={len(y):,}")
    print(f"  observed base rate : {y.mean():.4f}")
    print(f"  mean prediction    : {p.mean():.4f}   (gap vs base: {p.mean() - y.mean():+.4f})")
    print(f"  Brier              : {brier_score_loss(y, p):.4f}")
    print(f"  reliability (calib): {d['reliability']:.4f}   resolution: {d['resolution']:.4f}"
          f"   uncertainty: {d['uncertainty']:.4f}")
    print(f"  Brier skill (BSS)  : {d['bss']:+.4f}")
    # 10-bin reliability table
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    print("   bin  pred   obs     n")
    for b in range(10):
        m = idx == b
        if m.any():
            print(f"   {b:>3}  {p[m].mean():.3f}  {y[m].mean():.3f}  {int(m.sum()):>5}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgboost", choices=list(me.EVAL_MODELS))
    args = ap.parse_args()
    m = args.model

    print(f"=== Calibration diagnosis for '{m}' ===")

    print("\n--- A) notebook function walk_forward_predict, n_folds=6 ---")
    try:
        a = tr.walk_forward_predict(m, n_folds=6) if m != "hazard" else _hazard_wf(6)
        _summary("A · walk_forward_predict n_folds=6", a["y_true"], a["y_prob"])
    except Exception as e:  # noqa: BLE001
        print("  FAILED:", e)

    print("\n--- B) notebook function walk_forward_predict, n_folds=12 ---")
    try:
        b = tr.walk_forward_predict(m, n_folds=12) if m != "hazard" else _hazard_wf(12)
        _summary("B · walk_forward_predict n_folds=12", b["y_true"], b["y_prob"])
    except Exception as e:  # noqa: BLE001
        print("  FAILED:", e)

    print("\n--- C) the page's artifact Data/model_eval_%s.parquet ---" % m)
    if me.eval_available(m):
        c = me.model_eval(m)
        _summary("C · page artifact", c["y_true"], c["y_prob"])
        meta = me.model_eval_meta(m)
        if meta:
            print(f"  (built with n_folds_requested={meta.get('n_folds_requested')}, "
                  f"n_folds_used={meta.get('n_folds_used')})")
    else:
        print("  not built yet  run `python main.py eval --model %s`." % m)

    print("\nInterpretation: compare C to A (same regime). Match => no harness bug, it's the "
          "fold window/recency (compare A vs B). Divergence => tell me, I'll fix the harness.")
    return 0


def _hazard_wf(n_folds: int) -> pd.DataFrame:
    from src import hazard as hz
    return hz.walk_forward_predict_hazard(n_folds=n_folds)


if __name__ == "__main__":
    raise SystemExit(main())
