# Phase 1 — Codebase Audit + App Foundation

Paste this as the first task/message inside the Cowork Project set up with `01_cowork_project_instructions.md`.

## Step 1 — Audit (do this first, do not skip, do not write app code yet)

1. Locate and read the existing notebooks/source files at `<PATH_TO_EXISTING_CODEBASE>`.
2. Find the existing BigQuery connection/authentication code and the existing SQL queries. Summarize: how the connection authenticates, and what each existing query currently returns.
3. Produce a data dictionary for the `reservations` table and the `property_performance_daily` table: column names, inferred types, a few sample values, and an order-of-magnitude row count. If you find a different number of tables than two, report exactly what you find — do not assume the plan below is still accurate.
4. Locate the model artifact at `<PATH_TO_MODEL_8>`. Confirm its file format, list its expected input features, and describe its output (e.g. cancellation probability, risk class, survival curve). Note any other model files you find (baseline/mean model, XGBoost model, hazard model) and their storage format.
5. Find the existing (currently non-functional) function intended to pull `property_performance_daily` and explain precisely why it doesn't work today (missing connection, wrong table name, never called, etc.).
6. Write all findings into a single `audit_findings.md` file. Stop after this step and wait for review before proceeding to Step 2.

## Step 2 — Foundation (only after the audit findings are approved)

- Multi-page Dash app skeleton: Home, Occupancy Dashboard (empty placeholder), Cancellation History (empty placeholder), Model Performance (empty placeholder), shared navigation shell.
- Data-access layer: functions that pull `reservations` and `property_performance_daily` from BigQuery using the existing queries found in Step 1, cache the result locally (e.g. Parquet), and expose a clean read function for the rest of the app. Add a simple manual "refresh cache" trigger.
- BigQuery auth: use Application Default Credentials so the same code works locally (developer's `gcloud auth application-default login`) and in the future container (a service-account key via the `GOOGLE_APPLICATION_CREDENTIALS` environment variable), with no code branching between the two. If the code found in Step 1 hardcodes a specific local credentials path, replace it with this environment-driven approach. Verify the exact current client-initialization call against the `google-cloud-bigquery` docs — don't assume a method signature from memory.
- Load the model_8 artifact and expose a function that scores a given set of reservations with it.
- No dashboard charts yet in this step — the goal is to confirm the whole pipeline runs end-to-end and the app shell loads instantly with placeholder pages.

Once Step 2 is done and confirmed working, the next message will contain the Phase 2 task: the full Occupancy/Overbooking dashboard, including the exact spec for the cost-parameter panel.
