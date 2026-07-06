# XAI & Model Performance — Konzept zur Freigabe

Status: **Konzept, noch kein Code.** Erstellt nach dem Audit (Arbeitsweise Schritt 1).
Zur Freigabe, bevor Umsetzung beginnt (Schritt 2). Umsetzung dann schrittweise (Schritt 3).

Genauigkeits-Hinweis: Alle Zahlen unten stammen aus realen Artefakten im Repo
(`reports/tables/*/model_card.json`, `Data/run_summary.json`). Ich habe die Modelle
für dieses Konzept **nicht** live neu gescored. Wo ich unsicher über exakte
Bibliotheks-API-Namen bin, ist das markiert („in aktueller Doku verifizieren").

---

## 1. Audit-Ergebnisse (was schon existiert — das verändert den Plan erheblich)

### 1.1 Der Prediction-Adapter aus Abschnitt 3 existiert bereits
`src/scoring.py::cancel_proba(model_name, feat)` ist genau der geforderte einheitliche
Adapter. Er liefert für **alle vier Modelle** dieselbe skalare Zielgröße —
*P(Storno bis Anreise)* pro Buchung:

- `kind="static"` (xgboost, histgb, logreg): `pipeline.predict_proba(...)[:, 1]`
- `kind="hazard"`: Survival-Produkt `1 − Π(1 − h_s)` über das trainierte Snapshot-Grid
  (`src/hazard.py::score_upcoming_hazard`).

**Konsequenz für den wichtigsten Architektur-Punkt (Abschnitt 3):** Das Hazard-Modell
liefert bereits eine **skalare** Zielgröße, nicht nur eine Kurve. Die Vereinheitlichung
ist im Kern gelöst — wir bauen sie nicht neu, sondern nutzen `cancel_proba` als die eine
Quelle für jede Vorhersage auf dieser Seite.

### 1.2 Das Hazard-Modell im Detail
- **Library:** kein lifelines/scikit-survival. Es ist ein **Discrete-Time-Hazard-Modell
  mit XGBoost** als Basis-Lerner (`XGBClassifier`, `src/hazard.py`). Es lernt pro
  Person-Perioden-Zeile den Tages-/Fenster-Hazard `h_d` und aggregiert per Survival-Produkt
  zu P(Storno bis Anreise). Isotonische Kalibrierung getrennt pro Snapshot-Band (≤14 Tage
  täglich vs. grober Tail).
- **Estimand & Horizont:** Entscheidungszeitpunkt-Horizont `d = min(lead, 14)`; Label
  `status == 1` (Storno vor/bei Anreise). H = 14 Tage ist projektweit konsistent
  (`WINDOW_DAYS`, `walkforward_folds.json: horizon_days=14`). Damit ist **Offene Frage 3
  (projektweiter Horizont) durch das Audit beantwortet: 14 Tage.**

### 1.3 Kosten-/Threshold-/Kalibrierungs-Logik existiert ebenfalls schon
`src/scoring.py` enthält als Single Source of Truth: `analytic_threshold`,
`cost_threshold_from_scores`, `cost_at_threshold`, `cost_optimal_threshold`,
`serving_thresholds`, `brier_decomposition` (Murphy-Zerlegung + Brier Skill Score).
Diese Funktionen **wiederverwenden**, nicht neu implementieren.

### 1.4 Was fehlt / die echten Baustellen
1. **Keine persistierten Vorhersagen.** In `Data/` liegt **keine** `*_predictions.parquet`.
   `_val_predictions()` gibt für jedes Modell `None` zurück → Thresholds fallen aktuell auf
   den analytischen Bayes-Wert zurück. Für ROC/PR/Kalibrierung/Feature-Importance müssen wir
   Vorhersagen auf einem **gemeinsamen, leckfreien** Set erst erzeugen.
2. **Die vier Model-Cards sind nicht vergleichbar.** Die statischen Cards haben je einen
   eigenen `walk_forward`-Block (xgboost AUC ≈ 0.745, histgb ≈ 0.751, logreg ≈ 0.730 — real,
   aber jeweils aus dem eigenen Notebook-Lauf). Der Hazard-Card hat **nur** `val_ap ≈ 0.076`
   — und das ist die **Person-Perioden-AP**, eine völlig andere Größe als die Buchungs-AP der
   statischen Modelle. **Diese Zahlen dürfen nicht nebeneinandergestellt werden.** Ein fairer
   Vergleich braucht **eine** Harness, die alle vier über `cancel_proba` auf **denselben**
   Walk-Forward-Test-Folds, **derselben** Zeile, **demselben** Label scored.
3. **Seite ist ein Stub** (`pages/model_performance.py`, 15 Zeilen dbc). Muss wie die anderen
   Seiten auf dmc + `ui.chart_card`/`kpi_card`/`location_filter` + `theme.brand_figure` +
   Drawer-Pattern portiert werden (Design Guide, keine neue Designsprache).
4. **Cost-State ist noch nicht global.** `dcc.Store(id="cost-store", storage_type="local")`
   ist im **Occupancy-Page-Layout** deklariert und speichert pro `property|ISO-Woche` ein
   Dict `{walk, empty, high, mult}`. Es ist noch keine geteilte, seitenübergreifende Quelle
   (siehe Entscheidung C).

### 1.5 Dependencies — alle vorhanden
`shap` (0.49/0.52), `scikit-survival`, `lifelines`, `scikit-learn>=1.8`, `xgboost>=2.1`
sind installiert. Keine neue Abhängigkeit nötig.

---

## 2. Vorgeschlagene Architektur

### 2.1 Ein Eval-Harness (neu, offline vorberechnet)
Neues Modul `src/model_eval.py`: fitte pro Walk-Forward-Fold jedes der vier Modelle
leckfrei auf dem Fold-Train und score die **identischen** Test-Buchungen bei
`d = min(lead, 14)` via `cancel_proba` → poole zu einem Frame
`[model, fold, property_name, y_true, y_prob]`. Daraus **eine** Artefakt-Datei
`Data/model_eval.parquet` (+ kleines JSON mit Metadaten).

**Warum offline, nicht im Dash-Callback:** Das Hazard-Fit macht RandomizedSearch mit Early
Stopping auf ~1,2 Mio Person-Perioden pro Fold — über 12 Folds ist das Minuten- bis
Stundenbereich, kein Live-Callback. Vorschlag: ein CLI-Schritt (z. B. `main.py eval`), der
das Artefakt schreibt; die Seite liest nur das fertige Parquet (analog zur
Read-only-Parquet-Konvention der anderen Seiten). Das ist zugleich der Punkt, an dem die
Baseline (historischer Durchschnitt) mitberechnet wird.

Alle Kurven (ROC, PR, Kalibrierung, Baseline, KPI-Kacheln, Standort-AUC) speisen sich aus
diesem **einen** Frame → garantiert vergleichbar, kein Chart rechnet sein eigenes Set.

### 2.2 Datenschicht
Neues Backend `dash_app/backend/model_performance.py` als **einzige** Lese-/Aggregations-
schicht der Seite (liest `Data/model_eval.parquet`, aggregiert serverseitig, kleine Frames
raus) — konsistent mit `backend/cancellation_history.py`. Keine BigQuery, keine
Query-Duplikate. Modell-Laden/Scoring läuft ausschließlich über `src.scoring`/`src.hazard`.

### 2.3 SHAP-Einzelfall-Modul (4.8) als wiederverwendbare Komponente
`dash_app/components/shap_explain.py`: reine Funktion `booking_id + model → SHAP-Figure`,
zwei Darstellungsgrößen (voll auf dieser Seite, mini später in der Overbooking-Sidebar).
Keine Logik-Kopie auf der Overbooking-Seite.

### 2.4 Cost-State (Single Source of Truth)
`cost-store` aus dem Occupancy-Layout in das **globale** `app.layout` (app.py) heben, damit
alle Seiten dieselbe Store-Instanz teilen. Occupancy-Callbacks bleiben unverändert. Diese
Seite liest denselben Store — welche Kostenwerte sie für den Threshold nutzt, hängt an
Entscheidung C.

---

## 3. Pflicht-Visualisierungen — Verfahren & methodische Sonderbehandlung

| # | Chart | Verfahren | Baseline-Vergleich? | Sonderbehandlung Hazard |
|---|-------|-----------|---------------------|--------------------------|
| 4.1 | ROC-AUC global + je Standort | `roc_curve`/`roc_auc_score` auf `model_eval.parquet`, Gruppierung nach `property_name`; Min-n-Guard pro Standort | **Nein** (Baseline ≈ 0.5 per Definition — im UI nicht als AUC-Vergleich) | keine — läuft über den skalaren Adapter |
| 4.2 | Precision/Recall/F1 über Threshold + kostenoptimaler Punkt | Kurven aus `y_true,y_prob`; Marker am `cost_threshold_from_scores` mit dem geteilten Cost-Rate | **Ja** (konstanter Schätzer → feste P/R/F1) | keine |
| 4.3 | Kalibrierung (Reliability) | `sklearn.calibration.calibration_curve` *(Klassennamen in installierter sklearn verifizieren)* + Brier/Reliability aus `brier_decomposition` | **Ja** (Sanity-Check: Baseline nahe Diagonale) | keine |
| 4.4 | Train vs. Test | **Basis für alle vier:** Train- vs. Test-Metrik als Balken. **Iterationskurve nur** xgboost/histgb (`evals_result()` / `train_score_`+`validation_score_`) | — | Hazard-Iterationskurve ist Person-Perioden-LogLoss (andere Skala) → nur als „internes Trainingsdiagnostikum" gelabelt, **nicht** neben den anderen als gleiche Größe |
| 4.5 | Feature Importance | Native Importance je Typ + **SHAP als gemeinsame Vergleichsbasis** (siehe 4.7) | — | siehe 4.7 |
| 4.6 | PDP & ICE | Statisch: `sklearn.inspection` *(exakte Namen verifizieren)*. Hazard: eigenes PD über denselben Adapter (Feature-Grid durch `cancel_proba("hazard", ·)` mitteln) | — | ein PD-Pfad, über den Adapter — nicht in sklearn erzwingen |
| 4.7 | SHAP Beeswarm | statisch: TreeExplainer (xgb/histgb), LinearExplainer (logreg). Hazard: siehe Entscheidung B | — | **Kern-Entscheidung B** |
| 4.8 | SHAP Single Contribution | Tabelle gescorter Buchungen (`dash-ag-grid`) → Klick → Waterfall/Force; wiederverwendbares Modul | — | über den Adapter |
| 4.9 | Baseline- & Standort-KPIs oben | bester/schlechtester Standort vs. Baseline — auf **Kalibrierung / Brier / kostenbasiert**, **nicht** AUC | **Ja** | Baseline = historischer Durchschnitt, beim Drilldown Standort-Durchschnitt |

### Zur SHAP-Frage beim Hazard-Modell (4.7 — der methodische Knackpunkt)
Weil unsere **Entscheidungsgröße bereits skalar ist** (P(Storno bis Anreise) via Adapter),
brauchen wir **kein SurvSHAP**. SurvSHAP erklärt eine ganze Survival-**Kurve**; unser
Entscheidungsziel ist ein einzelner Wert. Zwei valide Wege für genau diesen Skalar:

- **B1 (empfohlen, vergleichbar):** modell-agnostisches SHAP auf die skalare Funktion
  `cancel_proba("hazard", ·)` (Permutation-/Kernel-Explainer, Hintergrund-Sample).
  → Werte auf **derselben** Größe wie die Klassifikationsmodelle → Beeswarm direkt
  vergleichbar. **Kosten:** langsam (jede f-Auswertung rechnet das Survival-Produkt) →
  offline vorberechnen, moderates Hintergrund-Sample, cachen.
- **B2 (schnell, aber NICHT vergleichbar):** TreeExplainer auf den Roh-Hazard bei einem
  festen Snapshot (z. B. d=14). Erklärt „den Tages-Hazard 14 Tage vor Anreise", eine
  **andere** Größe. Nur als klar separat gelabelte Ansicht — **nie** stillschweigend neben
  die Klassifikations-Beeswarms.

Ich werde den Standard-SHAP-Code der drei Klassifikationsmodelle **nicht** unverändert aufs
Hazard-Modell anwenden. Welche Variante gilt → Entscheidung B.

Quellen zur Einordnung (aus dem Prompt, real): SHAP-Cox-Beispiel-Notebook (shap.readthedocs.io)
und SurvSHAP (ResearchGate 368366978). Ich habe diese für das Konzept nicht erneut abgerufen.

---

## 4. Entscheidungen (freigegeben 2026-07-06)

- **A — Eval-Compute → pro aktivem Modell, gecacht, vorwärmbar.** Nicht alle vier auf
  einmal. Jedes Modell wird beim ersten Auswählen einmal berechnet und in
  `Data/model_eval_<modell>.parquet` gecacht; danach lädt die Seite nur die Datei. Für den
  Docker-Server per `python main.py eval --all` beim Deploy vorwärmen → in Produktion wartet
  niemand. („Live" = jedes Mal neu rechnen; „offline/gecacht" = einmal rechnen, danach nur
  lesen — wir nehmen gecacht.)
- **B — Hazard-SHAP → B1 (agnostisch, vergleichbar).** Modell-agnostisches SHAP auf die
  skalare Adapter-Funktion, vorberechnet. Kein SurvSHAP.
- **C — Cost-Rate-Scope → global + Standort-Override.** Geteilter `cost-store` als Default,
  optional pro Standort. Store wird ins globale `app.layout` gehoben.
- **D — Modellumfang → nur das aktuell aktive Modell** (Dropdown). Alle vier sind wählbar
  und werden bei Auswahl je einmal berechnet/gecacht; es werden nie alle gleichzeitig gerechnet.

Umgebungs-Hinweis: Die Modelle laufen nur in der Projekt-venv/Docker (Python ≥3.12 + ML-Stack),
nicht in der Cowork-Sandbox. Neuer Code wird gegen die bereits getesteten Funktionen gebaut;
Ausführung/Verifikation der Zahlen passiert in deiner venv bzw. im Docker-Build.

## 4a. Umsetzungsstand

- **Increment 1 — Eval-Grundlage: FERTIG (ungetestet in Sandbox).**
  - `src/model_eval.py`: `model_eval(model_name)` → leckfreies, gecachtes Vorhersage-Artefakt
    pro Modell mit `property_name` + leckfreier Baseline (global + pro Standort). Nutzt
    `training.build_pipeline/_card_hp/_family_feature_lists` und `hazard.fit_hazard/
    survival_cancel_proba` wieder (keine Modell-Logik dupliziert), Muster von
    `bakeoff_walk_forward`. Syntax geprüft; Baseline-Logik per Self-Test verifiziert.
  - `main.py eval [--model X | --all] [--folds N] [--refresh]`: Vorwärm-CLI für Docker.
  - Zu prüfen in deiner venv: `python main.py eval --model xgboost --folds 6`
    (schnell) und danach `--model hazard` (langsam, Person-Perioden-Refit pro Fold).

---

## 5. Umsetzungsreihenfolge (nach Freigabe, schrittweise)

1. Cost-State global heben (app.py) + verifizieren, dass Occupancy unverändert läuft.
2. `src/model_eval.py` + CLI-Schritt → `Data/model_eval.parquet` erzeugen; Zahlen prüfen.
3. `backend/model_performance.py` (Lese-/Aggregationsschicht).
4. Seite auf dmc portieren: KPI-Strip (4.9) + Modell-Dropdown + Standortfilter.
5. Charts 4.1 → 4.4 (Metrik-Kurven + Baseline dort, wo valide).
6. Feature Importance + SHAP Beeswarm (4.5/4.7) inkl. Hazard-Sonderweg.
7. PDP/ICE (4.6).
8. SHAP-Einzelfall-Modul (4.8), von Anfang an wiederverwendbar gebaut.
9. Verifikation: Leckfreiheit der Folds, Baseline-Sanity, Metrik-Konsistenz gegen Cards.
