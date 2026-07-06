# STAYERY dashboard — design system notes

The Cancellation History page (`pages/cancellation_history.py`) is the reference
implementation for the app's visual language. This note records the decisions so the
Home, Occupancy and XAI pages can adopt them 1:1.

## Component stack

`dash-mantine-components` (dmc 2.8.0) is the design system, alongside `dash-ag-grid`
(35.2.0) for heavy tables and Plotly (6.8.0) for all charts. `dash-bootstrap-components`
(2.0.4) is still used by the navbar and the (not-yet-ported) Occupancy page.

Setup (already wired in `app.py`, needed once for the whole app):

- `_dash_renderer._set_react_version("18.2.0")` is called before `Dash(...)`. Dash 4.3.0
  ships React 18 already; pinning 18.2.0 is dmc's documented target and keeps rendering
  deterministic. dbc and dash-ag-grid also support React 18, so nothing breaks.
- `app.layout` is wrapped in `dmc.MantineProvider(..., theme=DMC_THEME,
  forceColorScheme="light")`. The provider is additive — dbc components render unchanged
  inside it — so pages can be migrated one at a time. dmc bundles its own CSS (no extra
  stylesheets needed).

## Colour + type

Single source of truth is `theme.py`, seeded from `configs/stayery_brand.yaml`.

- Controls (buttons, segmented controls) use `primaryColor: "dark"` → near-black, on
  brand. Brand yellow (`YELLOW`) is an explicit accent only (the KPI headline bar), never
  a control fill, to keep text-on-accent readable.
- Cancellation-rate heatmap uses `theme.CANCEL_SCALE` (green → yellow → red): green = low,
  red = high, the intuitive reading for a "higher is worse" metric.
- Charts run through `theme.brand_figure()` for consistent font, white canvas and the
  brand categorical palette. Reuse it for every Plotly figure on every page.
- Fonts: `FONT_FAMILY` (Neue Haas Grotesk) for body, `HEADING_FONT_FAMILY` (Topol) for
  headings, applied via `DMC_THEME`.

## Reusable components (`components/ui.py`)

- `location_filter(options, id, span_label=)` — the global `dmc.MultiSelect`. Defaults to
  ALL locations selected; an empty selection is treated as "all" by callbacks so charts
  are never accidentally blank.
- `kpi_card(label, value, sub, accent=, tooltip=)` and `kpi_strip(cards)` — metric tiles;
  `accent=True` adds the yellow headline bar; `tooltip=` adds a hover explanation.
- `chart_card(title, graph_id, info=, height=, subtitle=, header_extra=)` — titled surface
  with an ⓘ tooltip and an optional header control (e.g. a `SegmentedControl`), wrapping a
  `dcc.Graph` in a `dcc.Loading(custom_spinner=dmc.Skeleton(...))` shimmer loader.
- `info_icon(text)` — the ⓘ hover explainer used wherever a number could be misread.

## Interaction patterns (keep consistent across pages)

- One global location filter drives every chart on a page via plain callbacks (fast
  server-side aggregation; only aggregated frames reach the client).
- `dmc.SegmentedControl` for compact in-card toggles (aggregate/per-location; 6/12-month
  window).
- Click a chart element (heatmap cell, month point) → a right-side `dmc.Drawer` opens with
  granular detail. The drawer self-closes; callbacks only ever set `opened=True`.
- Skeleton loaders (not bare spinners) for loading states, via `chart_card`.
- Every non-obvious metric carries an ⓘ tooltip / info text.

## Data-layer conventions (`backend/cancellation_history.py`)

- Read-only from the cleaned parquet cache; no live BigQuery, ever.
- Aggregate on the server; return small frames. Filter functions accept `properties`
  (None/empty = all).
- Min-sample guards mask noisy rates to NaN rather than drawing them: 50 bookings per
  monthly point, 30 per heatmap cell, 200 per channel bar, 100 for the KPI "highest-rate
  location". Blanks read as "not enough data", never a fabricated number; missing values
  surface as "unavailable" in the UI.
- Page-specific correctness: on this HISTORICAL page a cancellation is the measured
  quantity, so cancelled bookings are the NUMERATOR and are NOT excluded (the opposite of
  the forward-looking Occupancy page). `status == 1` = cancel-before-arrival.

## Migrating the other pages

- Occupancy — DONE. Ported to the dmc look (dmc header/Stack, `ui.chart_card` heatmap with
  skeleton loader, `ui.kpi_card`/`kpi_strip` tiles, dmc `MultiSelect`/`NumberInput`/`Button`,
  dmc cost + recommendation cards). Pure design change: all callback IDs/logic preserved.
  Two intentional prop-level edits it required: `cost-active-property` became a `dmc.Select`
  so its sync callback now outputs `data` (was dbc `options`); the high-demand toggle stays a
  `dbc.Switch` because dmc `Switch` uses `checked` (not `value`) and the cost callbacks read
  `value`. `dash-ag-grid` and `theme.brand_figure` are kept as-is.
- Home — DONE. dmc hero + nav cards; the Cancellation History card is now enabled,
  Model Performance stays "coming soon" until that page is built.
- XAI / Model Performance: same recipe — wrap in `dmc.Stack`, use `ui.chart_card` /
  `ui.kpi_card` / `ui.location_filter`, run every figure through `theme.brand_figure`.
- The overbooking cost sandbox stays on Occupancy (it needs the forward-looking scored
  data); the Cancellation History page is intentionally descriptive only.

## Occupancy dashboard revamp (fonts, header, heatmap %, composition)

- Custom fonts now actually load via `assets/brand.css` (`@font-face` for Neue Haas
  Grotesk Display Pro + Topol). The header is a white bar with a yellow accent, Topol
  wordmark and an underlined active nav link (`.stayery-*` classes in brand.css).
- Occupancy heatmap shows real occupancy % in every tile (e.g. "79%"). Capacity =
  the performance table's latest `houseCount`, mapped propertyId→property_name via the
  reservations cache's `property_code` (see below). Colour range fixed to 0–100%.
- Composition charts: pies replaced. Business/leisure and channel are horizontal bars;
  rate plan is a compact `dmc.Table`; length-of-stay and lead-time stay as histograms.
- Channel uses the granular OTA `source` field (Booking.com, Expedia, HRS, Airbnb…),
  filled with `channelCode` for Direct/IBE — NOT the coarse "ChannelManager" bucket.

## Still open

- ~~`propertyId` ↔ `property_name` mapping unresolved~~ — RESOLVED. The reservations
  cache carries `property_code` (e.g. `BER_FR`), which is exactly the perf table's
  `propertyId`; `data_access._property_code_to_name()` builds the bridge. This unblocked
  occupancy-% capacity and can also unblock ADR prefill.
- COVID phases are intentionally not shown: the clean cache starts at arrival 2022-08 (the
  COVID regime was deliberately excluded upstream), so the time series is the modelled
  period only.
