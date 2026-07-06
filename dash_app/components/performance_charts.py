# dash_app/components/performance_charts.py
# Plotly figure builders for the XAI / Model-Performance page. Each takes an already-
# aggregated structure from dash_app.backend.model_performance (or explain) and returns a
# brand-styled go.Figure. Kept separate from the page so the page holds only layout +
# callbacks. Baseline discipline mirrors the backend: ROC carries NO baseline (a constant
# predictor is 0.5 by construction); reliability / threshold-cost DO.

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash_app import theme


def _empty(msg: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color="#9AA0A6", size=13))
    fig.update_layout(height=height, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return theme.brand_figure(fig)


# ---- 4.1 ROC (global) ------------------------------------------------------
def fig_roc(roc: dict, height: int = 340) -> go.Figure:
    if not roc:
        return _empty("Not enough data / artifact not built yet", height)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color="#9AA0A6", width=1, dash="dash"),
                             name="Chance (AUC 0.5)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=roc["fpr"], y=roc["tpr"], mode="lines", line=dict(color=theme.BLUE, width=2.5),
        name=f"Model (AUC {roc['auc']:.3f})", fill="tozeroy", fillcolor="rgba(40,90,200,0.08)",
        hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>"))
    fig.update_layout(
        height=height, xaxis_title="False-positive rate", yaxis_title="True-positive rate",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, x=0),
        margin=dict(l=55, r=20, t=20, b=60))
    return theme.brand_figure(fig)


# ---- 4.1 ROC-AUC per location ---------------------------------------------
def fig_roc_by_location(df: pd.DataFrame, height: int = 360) -> go.Figure:
    if df is None or df.empty:
        return _empty("No location has enough data for a reliable AUC", height)
    d = df.sort_values("auc")
    colors = [theme.GREEN if a >= 0.70 else theme.ORANGE if a >= 0.60 else theme.RED
              for a in d["auc"]]
    fig = go.Figure(go.Bar(
        x=d["auc"], y=d["property_name"], orientation="h", marker_color=colors,
        customdata=np.stack([d["n"], d["pos"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>AUC %{x:.3f}<br>Bookings %{customdata[0]:,} · "
                      "cancels %{customdata[1]:,}<extra></extra>"))
    fig.add_vline(x=0.5, line=dict(color="#9AA0A6", width=1, dash="dash"),
                  annotation_text="chance", annotation_position="bottom right",
                  annotation_font_size=10)
    fig.update_layout(height=max(height, 60 + 26 * len(d)), xaxis_title="ROC-AUC",
                      yaxis_title=None, xaxis=dict(range=[0.4, 1.0]),
                      margin=dict(l=140, r=20, t=20, b=40), bargap=0.25)
    return theme.brand_figure(fig)


# ---- 4.2 Precision / Recall / F1 over threshold + cost-optimal + baseline ---
def fig_pr_threshold(pr: dict, height: int = 360) -> go.Figure:
    if not pr:
        return _empty("Not enough data / artifact not built yet", height)
    t = pr["thr"]
    fig = go.Figure()
    for key, name, col in [("precision", "Precision", theme.BLUE),
                           ("recall", "Recall", theme.GREEN),
                           ("f1", "F1", theme.PURPLE)]:
        fig.add_trace(go.Scatter(x=t, y=pr[key], mode="lines", name=name,
                                 line=dict(color=col, width=2),
                                 hovertemplate=f"{name} %{{y:.2f}} @ thr %{{x:.2f}}<extra></extra>"))
    # cost-optimal operating point (shared cost definition).
    fig.add_vline(x=pr["t_cost"], line=dict(color=theme.BLACK, width=1.5, dash="dash"),
                  annotation_text=f"cost-opt thr {pr['t_cost']:.2f}",
                  annotation_position="top left", annotation_font_size=10)
    # baseline (constant = base rate) operating point at the same threshold.
    b = pr["baseline"]
    fig.add_trace(go.Scatter(
        x=[pr["t_cost"]], y=[b["f1"]], mode="markers", name=f"Baseline F1 {b['f1']:.2f}",
        marker=dict(color=theme.ORANGE, size=10, symbol="diamond"),
        hovertemplate=f"Baseline @ cost-opt thr<br>P {b['precision']:.2f} · R {b['recall']:.2f} · "
                      f"F1 {b['f1']:.2f}<extra></extra>"))
    fig.update_layout(
        height=height, xaxis_title="Decision threshold", yaxis_title="Score",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, x=0),
        margin=dict(l=50, r=20, t=25, b=60))
    return theme.brand_figure(fig)


# ---- 4.3 Reliability diagram + baseline sanity -----------------------------
def fig_reliability(rel: dict, height: int = 360) -> go.Figure:
    if not rel:
        return _empty("Not enough data / artifact not built yet", height)
    bins = rel["bins"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(color="#9AA0A6", width=1, dash="dash"), hoverinfo="skip"))
    if bins is not None and not bins.empty:
        fig.add_trace(go.Scatter(
            x=bins["pred"], y=bins["obs"], mode="lines+markers", name="Model",
            line=dict(color=theme.BLUE, width=2),
            marker=dict(size=bins["n"].clip(upper=400) / 25 + 5, color=theme.BLUE),
            customdata=bins["n"],
            hovertemplate="Predicted %{x:.2f}<br>Observed %{y:.2f}<br>n %{customdata:,}<extra></extra>"))
    # baseline: constant = base rate -> single point on the diagonal.
    br = rel["base_rate"]
    fig.add_trace(go.Scatter(
        x=[br], y=[br], mode="markers", name=f"Baseline (base rate {br:.2f})",
        marker=dict(color=theme.ORANGE, size=11, symbol="diamond"),
        hovertemplate=f"Historical-average baseline<br>base rate {br:.2f}<extra></extra>"))
    dec = rel.get("decomp", {})
    sub = (f"Brier {dec.get('brier', float('nan')):.3f} · reliability "
           f"{dec.get('reliability', float('nan')):.4f} · BSS {dec.get('bss', float('nan')):+.3f}")
    fig.update_layout(
        height=height, xaxis_title="Predicted probability", yaxis_title="Observed frequency",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        title=dict(text=sub, font=dict(size=11, color="#9AA0A6"), x=0, xanchor="left", y=0.98),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, x=0),
        margin=dict(l=55, r=20, t=30, b=60))
    return theme.brand_figure(fig)


# ---- 4.4 Train vs test (aggregate bars) ------------------------------------
def fig_train_test(tt: dict, metric: str = "auc", height: int = 320) -> go.Figure:
    if not tt or not tt.get("aggregate"):
        return _empty("Per-fold metrics not built yet", height)
    agg = tt["aggregate"]
    tr, te = f"train_{metric}", f"test_{metric}"
    if tr not in agg or te not in agg:
        return _empty("Metric unavailable", height)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Train", "Test"],
                         y=[agg[tr]["mean"], agg[te]["mean"]],
                         error_y=dict(type="data", array=[agg[tr]["std"], agg[te]["std"]]),
                         marker_color=[theme.BLUE, theme.YELLOW],
                         hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(height=height, yaxis_title=metric.upper(),
                      title=dict(text=f"Mean ± std over {tt['n_folds']} walk-forward folds",
                                 font=dict(size=11, color="#9AA0A6"), x=0, xanchor="left", y=0.98),
                      margin=dict(l=50, r=20, t=35, b=30), bargap=0.5)
    return theme.brand_figure(fig)


def fig_iteration_curve(curve: dict, height: int = 320) -> go.Figure:
    """Boosting train/validation loss vs iteration (XGBoost/HistGB only). `curve` =
    {'iters': [...], 'train': [...], 'valid': [...], 'metric': str, 'note': str}."""
    if not curve or not curve.get("iters"):
        return _empty("No iteration curve for this model type", height)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["iters"], y=curve["train"], mode="lines",
                             name="Train", line=dict(color=theme.BLUE, width=2)))
    if curve.get("valid"):
        fig.add_trace(go.Scatter(x=curve["iters"], y=curve["valid"], mode="lines",
                                 name="Validation", line=dict(color=theme.ORANGE, width=2)))
    note = curve.get("note", "")
    fig.update_layout(
        height=height, xaxis_title="Boosting iteration", yaxis_title=curve.get("metric", "loss"),
        title=dict(text=note, font=dict(size=11, color="#9AA0A6"), x=0, xanchor="left", y=0.98),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=55, r=20, t=35, b=40))
    return theme.brand_figure(fig)


# ---- 4.5 Feature importance (mean |SHAP|) ----------------------------------
def fig_importance(imp: pd.DataFrame, height: int = 420) -> go.Figure:
    """Horizontal bars of mean |SHAP| per feature (the common cross-model basis)."""
    if imp is None or imp.empty:
        return _empty("SHAP importance not built yet (run `main.py explain`)", height)
    d = imp.sort_values("importance").tail(20)
    fig = go.Figure(go.Bar(
        x=d["importance"], y=d["feature"], orientation="h", marker_color=theme.BLUE,
        hovertemplate="<b>%{y}</b><br>mean|SHAP| %{x:.4f}<extra></extra>"))
    fig.update_layout(height=max(height, 60 + 20 * len(d)),
                      xaxis_title="mean |SHAP| (impact on P(cancel))", yaxis_title=None,
                      margin=dict(l=170, r=20, t=20, b=40), bargap=0.2)
    return theme.brand_figure(fig)


# ---- 4.7 SHAP beeswarm -----------------------------------------------------
def fig_beeswarm(bee: pd.DataFrame, height: int = 460) -> go.Figure:
    """Beeswarm of SHAP values. `bee` long frame: [feature, shap, fval_norm] (fval_norm in
    [0,1], NaN for non-numeric). Features ordered by mean|SHAP|; y-jitter per feature."""
    if bee is None or bee.empty:
        return _empty("SHAP values not built yet (run `main.py explain`)", height)
    order = (bee.assign(a=bee["shap"].abs()).groupby("feature")["a"].mean()
             .sort_values().index.tolist())
    fig = go.Figure()
    for i, feat in enumerate(order):
        g = bee[bee["feature"] == feat]
        jitter = (np.random.default_rng(i).uniform(-0.32, 0.32, len(g)))
        fig.add_trace(go.Scatter(
            x=g["shap"], y=np.full(len(g), i) + jitter, mode="markers",
            marker=dict(size=5, color=g["fval_norm"], colorscale=[[0, theme.BLUE], [1, theme.RED]],
                        cmin=0, cmax=1, showscale=(i == len(order) - 1),
                        colorbar=dict(title="feature<br>value", thickness=10,
                                      tickvals=[0, 1], ticktext=["low", "high"])),
            name=str(feat), showlegend=False,
            hovertemplate=f"<b>{feat}</b><br>SHAP %{{x:.4f}}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#9AA0A6", width=1))
    fig.update_layout(height=max(height, 80 + 24 * len(order)),
                      xaxis_title="SHAP value (impact on P(cancel))",
                      yaxis=dict(tickmode="array", tickvals=list(range(len(order))),
                                 ticktext=order),
                      margin=dict(l=170, r=20, t=20, b=40))
    return theme.brand_figure(fig)


# ---- 4.8 Single-booking SHAP waterfall -------------------------------------
def fig_waterfall(contrib: dict, height: int = 460) -> go.Figure:
    """Waterfall for one booking. `contrib` = {'base': float, 'pred': float,
    'items': [{'feature','value','shap'}...]}. Top-k features shown; rest pooled."""
    if not contrib or not contrib.get("items"):
        return _empty("Select a booking to see its explanation", height)
    items = sorted(contrib["items"], key=lambda r: abs(r["shap"]), reverse=True)
    top = items[:12]
    rest = items[12:]
    labels = [f"{r['feature']} = {r['value']}" for r in top]
    values = [r["shap"] for r in top]
    if rest:
        labels.append(f"{len(rest)} other features")
        values.append(float(sum(r["shap"] for r in rest)))
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["relative"] * len(values),
        x=values, y=labels, base=contrib["base"],
        connector=dict(line=dict(color="#CCCCCC")),
        decreasing=dict(marker=dict(color=theme.GREEN)),
        increasing=dict(marker=dict(color=theme.RED)),
        hovertemplate="%{y}<br>SHAP %{x:+.4f}<extra></extra>"))
    fig.update_layout(
        height=height, xaxis_title="Contribution to P(cancel)", yaxis=dict(autorange="reversed"),
        title=dict(text=f"base {contrib['base']:.3f} → predicted {contrib['pred']:.3f}",
                   font=dict(size=11, color="#9AA0A6"), x=0, xanchor="left", y=0.99),
        margin=dict(l=220, r=20, t=35, b=40))
    return theme.brand_figure(fig)


# ---- 4.6 PDP / ICE ---------------------------------------------------------
def fig_pdp(pdp: dict, height: int = 340) -> go.Figure:
    """Partial dependence (+ optional ICE lines). `pdp` = {'x':[...], 'pd':[...],
    'ice': [[...],...] | None, 'feature': str}."""
    if not pdp or not pdp.get("x"):
        return _empty("Select a feature for its partial-dependence curve", height)
    fig = go.Figure()
    for line in (pdp.get("ice") or [])[:60]:
        fig.add_trace(go.Scatter(x=pdp["x"], y=line, mode="lines",
                                 line=dict(color="rgba(150,150,150,0.25)", width=1),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pdp["x"], y=pdp["pd"], mode="lines+markers", name="Partial dependence",
                             line=dict(color=theme.BLUE, width=3),
                             hovertemplate="%{x}<br>avg P(cancel) %{y:.3f}<extra></extra>"))
    fig.update_layout(height=height, xaxis_title=pdp.get("feature", "feature"),
                      yaxis_title="avg P(cancel)", showlegend=False,
                      margin=dict(l=55, r=20, t=20, b=45))
    return theme.brand_figure(fig)
