# dash_app/components/__init__.py
# Intentionally minimal. The old sidebar/icons/ui component modules were removed in
# the Phase-1 cleanup; do NOT re-import them here (that caused
# "ModuleNotFoundError: dash_app.components.icons"). Import builders directly, e.g.
#   from dash_app.components import panels
