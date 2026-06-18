# Stayery Brand Fonts

OTF-Files der Stayery-Brand-Schriften. Werden vom CSS in
`components/brand.py` via `@font-face` geladen.

## Aktuelle Files

```
NeueHaasGroteskDisplay-Regular.otf      ← weight 400 (echtes Display)
NeueHaasGroteskDisplay-Italic.otf       ← weight 400 italic
NeueHaasGroteskDisplay-Medium.otf       ← weight 500
NeueHaasGroteskDisplay-MediumItalic.otf ← weight 500 italic
NeueHaasGroteskDisplay-Bold.otf         ← weight 700
NeueHaasGroteskDisplay-BoldItalic.otf   ← weight 700 italic
Topol-Bold.otf                          ← display font for occasional emphasis
```

Medium und Bold sind technisch **Text-Pro**-Cuts (Linotype liefert die
schwereren Weights nicht im Display-Cut) — visuell ~95% identisch.

## Setup

Static-Serving ist in `.streamlit/config.toml` aktiviert. Files werden
automatisch unter `/app/static/fonts/...` ausgeliefert.

Nach Änderung an den Files: App neu starten + Browser-Hard-Refresh
(`Cmd/Ctrl + Shift + R`).

## Ohne diese Files

Die App fällt auf System-Fonts zurück (Helvetica Neue → Helvetica →
Arial → Browser-Default sans-serif). Keine externen Requests.
