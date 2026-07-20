"""PySide6 desktop GUI for EIS batch analysis.

Run from the repository root with:

    python -m gui.app
"""

import matplotlib

# core.plotting imports pyplot, so the backend must be pinned before any
# core import. Agg keeps pyplot from spawning its own windows; figures are
# embedded in Qt via FigureCanvasQTAgg, which works under any pyplot backend.
matplotlib.use("Agg")
