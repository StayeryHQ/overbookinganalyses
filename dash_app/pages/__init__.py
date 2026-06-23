# dash_app/pages/__init__.py
# Marks `dash_app.pages` as a package. Dash discovers page MODULES via the
# `pages_folder` mechanism (each module calls dash.register_page); this file is
# only needed so `import dash_app.pages.<module>` works for our import test.
# Intentionally empty.
