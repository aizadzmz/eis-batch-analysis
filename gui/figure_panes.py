"""Widgets that host matplotlib figures inside Qt layouts.

The core plotting functions create their own Figure objects, so these panes
adopt a finished Figure rather than drawing onto a persistent canvas. Each
swap closes the previous figure to keep pyplot's registry from growing.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget


class FigurePane(QWidget):
    """Hosts a single figure with an optional navigation toolbar."""

    def __init__(self, with_toolbar: bool = True, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._with_toolbar = with_toolbar
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._canvas: Optional[FigureCanvasQTAgg] = None
        self._toolbar: Optional[NavigationToolbar2QT] = None

    def set_figure(self, fig: Figure) -> None:
        self.clear()
        self._canvas = FigureCanvasQTAgg(fig)
        if self._with_toolbar:
            self._toolbar = NavigationToolbar2QT(self._canvas, self)
            self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas)
        self._canvas.draw_idle()

    def clear(self) -> None:
        if self._toolbar is not None:
            self._layout.removeWidget(self._toolbar)
            self._toolbar.deleteLater()
            self._toolbar = None
        if self._canvas is not None:
            fig = self._canvas.figure
            self._layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None
            plt.close(fig)


class FigureListPane(QScrollArea):
    """A scrollable vertical stack of figures (one canvas per figure)."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.addStretch()
        self.setWidget(self._container)
        self._canvases: List[FigureCanvasQTAgg] = []

    def set_figures(self, figs: List[Figure]) -> None:
        self.clear()
        for fig in figs:
            canvas = FigureCanvasQTAgg(fig)
            canvas.setMinimumHeight(340)
            # insert above the trailing stretch
            self._layout.insertWidget(self._layout.count() - 1, canvas)
            self._canvases.append(canvas)
            canvas.draw_idle()

    def clear(self) -> None:
        for canvas in self._canvases:
            fig = canvas.figure
            self._layout.removeWidget(canvas)
            canvas.deleteLater()
            plt.close(fig)
        self._canvases = []
