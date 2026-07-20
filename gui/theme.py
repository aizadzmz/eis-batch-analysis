"""Light/dark theming for the app.

Qt chrome is themed with qdarktheme (the pyqtdarktheme-fork package, imported
as ``qdarktheme``). Matplotlib figures are themed in parallel via rcParams so
plots regenerated after a theme switch match the surrounding UI instead of
staying a bright white (or dark) box.

The core plotting functions read rcParams at figure-creation time and don't
hardcode text/axis colors, so setting rcParams here and then re-rendering is
all that's needed — core/plotting.py stays untouched and the Streamlit app is
unaffected.
"""

import matplotlib
import qdarktheme

THEMES = ("light", "dark")
ACCENT = "#3b6fd4"  # muted blue, applied as qdarktheme's "primary" color

# rcParams applied per mode. Both dicts set the same keys so toggling either
# way fully overrides the other — no leftover colors from the previous theme.
_LIGHT_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#252931",
    "axes.titlecolor": "#252931",
    "text.color": "#252931",
    "xtick.color": "#252931",
    "ytick.color": "#252931",
    "grid.color": "#b0b0b0",
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
}
_DARK_RC = {
    "figure.facecolor": "#1e2228",
    "axes.facecolor": "#252b33",
    "axes.edgecolor": "#8a9099",
    "axes.labelcolor": "#e3e6ea",
    "axes.titlecolor": "#e3e6ea",
    "text.color": "#e3e6ea",
    "xtick.color": "#c7ccd3",
    "ytick.color": "#c7ccd3",
    "grid.color": "#3a424c",
    "legend.facecolor": "#252b33",
    "legend.edgecolor": "#3a424c",
}


def apply_theme(mode: str) -> None:
    """Apply ``mode`` ("light" or "dark") to both Qt and matplotlib.

    qdarktheme.setup_theme finds the running QApplication itself, so no app
    handle is needed here. Call before (re)rendering figures.
    """
    if mode not in THEMES:
        mode = "light"
    qdarktheme.setup_theme(mode, custom_colors={"primary": ACCENT})
    matplotlib.rcParams.update(_DARK_RC if mode == "dark" else _LIGHT_RC)
