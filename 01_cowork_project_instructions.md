# Hotel Overbooking & Occupancy Dashboard — Cowork Project Instructions

Paste this as the persistent instructions for this Cowork Project (Project settings → Instructions). This document stays constant across all phases. Phase-specific tasks are sent as separate messages.

## Role

You are building a production-grade Plotly Dash application for a small hotel chain with 11 properties. The primary user is a Revenue Manager who, every week, sets an overbooking allowance for each property based on predicted cancellations for the next 14 days.

Correctness and clarity take priority over speed of delivery. If something in the existing code or data contradicts this document, stop and report it — do not silently guess or work around it.

## Full information architecture (for context — build order is phased, see "Guardrails" below)

1. **Home** — navigation + short intro to the tool.
2. **Occupancy & Overbooking Dashboard (MAIN PAGE)** — 14-day focus, per-property and global filtering, KPI tiles, booking-level risk table with a detail side panel, room-type occupancy sub-view.
3. **Cancellation History** — global and per-property, historical cancellation rate (incl. pre/post-COVID if feasible), monthly granularity, breakdowns by channel and length-of-stay, a heatmap of average cancellation rate by property × month.
4. **Model Performance & XAI** — baseline (mean) model vs. XGBoost vs. hazard model: ROC/AUC, precision/recall/F1 vs. threshold, cost-threshold confusion matrix, calibration plot, AUC per property, train/test overfitting plot, feature importance, PDP/ICE, SHAP swarm + single-booking contribution, linked to a real booking picked from a table.

## Tech stack (decided — do not change without asking)

- **Framework:** Python, Dash. All charts in Plotly, except where a library genuinely can't do it (e.g. some SHAP visuals) — ask before deviating.
- **Tables:** dash-ag-grid for the booking list (not the basic DataTable) — needed for performance and row-selection → side panel interaction.
- **UI components:** dash-bootstrap-components and/or dash-mantine-components are acceptable for polish (tooltips, loading states, theming). Verify current install/API syntax against each library's own docs before using — do not assume method signatures.
- **Data source:** Google BigQuery. SQL queries already exist in the current codebase at `<PATH_TO_EXISTING_CODEBASE>` — locate and reuse them, do not rewrite from scratch unless they're broken.
- **Model artifact:** the current production model ("model 8") is already trained and serialized at `<PATH_TO_MODEL_8>` (.pkl/.joblib). Load it — do not retrain in this phase.
- **Caching (important):** BigQuery bills per data scanned and adds network latency. Do not query BigQuery live on every callback/filter interaction. Pull data into a local cache (e.g. Parquet files, or DuckDB) on a schedule or via a manual "refresh" trigger, and have the app read from that local cache. Use Dash Background Callbacks (DiskCache manager locally; Celery+Redis if this ever reaches production) for anything that takes more than ~1 second.

## Non-functional requirements (hard constraints)

- The app shell and KPI tiles must render immediately. Heavier charts/tables load progressively with proper loading states — never a blank white screen while BigQuery or model inference runs.
- Clean, commented, non-repetitive code. No unnecessary wrapper functions "just because."
- Code should be understandable and maintainable by other developers later.

## Deployment target (relevant from Phase 1 onward — the actual container build/rollout is NOT part of this project)

This app will run in a Docker container on a rented server reachable only internally/via VPN (not the public internet). A colleague owns a GitHub-based CI/CD pipeline that builds the image and rolls it out — that pipeline, the Dockerfile, and the server rollout itself are out of scope here. Do not create a Dockerfile, docker-compose file, or CI/CD config unless explicitly asked later.

What is in scope now, because it's much cheaper to get right from Phase 1 than to retrofit:

- All configuration and secrets (BigQuery credentials, possibly a `REDIS_URL` later) must come from environment variables — never hardcoded, never committed to source control.
- BigQuery authentication specifically must work in two contexts without any code change: locally via Application Default Credentials (a developer logged in with `gcloud auth application-default login`), and in the container via a service-account key referenced through the `GOOGLE_APPLICATION_CREDENTIALS` environment variable. If the existing code (found during the Phase 1 audit) hardcodes a specific local key file path instead, update it to this environment-driven pattern. Verify the exact current initialization call against the `google-cloud-bigquery`/`google-auth` documentation before implementing — do not assume a specific method signature from memory.
- Treat the local data cache as rebuildable/ephemeral, not as guaranteed-persistent state — the app should be able to rebuild it from BigQuery after a container restart.
- Implement Background Callbacks using the environment-driven pattern Dash documents for exactly this situation: a DiskCache manager when no `REDIS_URL` is set (local dev), a Celery manager backed by Redis when it is (possible future state). Verify current exact syntax against Dash's own Background Callbacks docs before implementing.
- Expose the underlying Flask server early as `server = app.server`, since a WSGI server (commonly gunicorn) will run it in the container. Costs nothing now, avoids a restructure later.
- Do not add authentication/login to the app itself — the server is internal/VPN-only, so this is low priority for now; revisit only if that changes.

## Roadmap (context only — each phase below is still dispatched as its own separate task)

1. Project steckbrief (this document)
2. Codebase audit + app foundation
3. Occupancy/Overbooking dashboard (priority page)
4. Cancellation history
5. Model performance & XAI
6. Room-type dashboard + UX polish
7. Any remaining container-readiness touch-ups if the colleague's pipeline needs something specific (Dockerfile/CI itself stays out of scope)

## Guardrails — do NOT build yet

- Do not build the Model Performance/XAI page, the room-type sub-view, authentication, or automated retraining until each is explicitly dispatched as its own task. Building ahead of the current phase creates untested, unreviewed surface area — this project is deliberately staged.
- If you discover more or fewer tables than expected, different column names, a different model file format, or anything else that doesn't match this document: stop, report the discrepancy, and wait for confirmation instead of adapting silently.

## Business logic notes for later phases (context only, not a build instruction yet)

- Overbooking cost parameters (cost of an oversold/"walked" guest vs. cost of an empty room) will be entered manually by the Revenue Manager, per property per week, with sensible pre-filled defaults and simple contextual toggles (e.g. "high-demand period") rather than fully automated external data feeds. Full spec will be provided when the Occupancy Dashboard phase is dispatched.
- Retraining (yearly hyperparameter search, periodic model refresh) is out of scope for the MVP; a later phase will add a local, click-to-run trigger.

## Working agreement

- Confirm your understanding of ambiguous instructions before writing code, rather than assuming the most likely interpretation.
- Prefer investigating the existing codebase over asking the user for information the code can answer directly (e.g. schema, existing auth pattern).
