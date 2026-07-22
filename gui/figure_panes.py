"""Widgets that host matplotlib figures inside Qt layouts.

The core plotting functions create their own Figure objects, so these panes
adopt a finished Figure rather than drawing onto a persistent canvas. Each
swap closes the previous figure to keep pyplot's registry from growing.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QScrollArea, QToolButton, QVBoxLayout, QWidget

from core.plotting import (
    GID_KEPT_POINTS,
    GID_REMOVED_POINTS,
    POINT_META_ATTR,
    equal_aspect_limits,
)

_OVERLAY_STYLE = """
QWidget#figureOverlay {
    background: rgba(128, 128, 128, 40%);
    border: 1px solid rgba(128, 128, 128, 60%);
    border-radius: 6px;
}
QToolButton {
    border: none;
    padding: 3px 8px;
    border-radius: 4px;
}
QToolButton:checked {
    background: rgba(0, 0, 0, 35%);
}
"""


class FigurePane(QWidget):
    """Hosts a single figure with an optional navigation toolbar."""

    replot_requested = Signal()

    def __init__(
        self,
        with_toolbar: bool = True,
        with_overlay_actions: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._with_toolbar = with_toolbar
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._canvas: Optional[FigureCanvasQTAgg] = None
        self._toolbar: Optional[NavigationToolbar2QT] = None

        self._overlay: Optional[QWidget] = None
        self._hide_removed_button: Optional[QToolButton] = None
        if with_overlay_actions:
            self._build_overlay()

        # Remembers the last view (zoom/pan, or the Auto-Scale button) so a
        # replot triggered by unrelated actions (running validation, DRT,
        # etc.) doesn't snap the view back out to the default framing.
        self._locked_xlim: Optional[Tuple[float, float]] = None
        self._locked_ylim: Optional[Tuple[float, float]] = None

        # The framing plot_single/plot_overlay computed for the current
        # figure, captured before any locked view is reapplied — this is
        # what "un-toggling" Auto-Scale restores.
        self._default_xlim: Optional[Tuple[float, float]] = None
        self._default_ylim: Optional[Tuple[float, float]] = None

        # Persists across replots the same way the locked view does, so
        # running validation/DRT etc. doesn't make removed points reappear.
        self._removed_hidden = False

        # The annotation box shown when a plotted point is clicked (see
        # _on_pick); cleared on every click so at most one is ever shown.
        self._tooltip_annotation = None

        # Set by _on_pick and consumed by _on_button_press to tell a
        # click-that-hit-a-point apart from a click-that-missed (see the
        # ordering note on _on_button_press).
        self._picked_this_click = False

    def _build_overlay(self) -> None:
        overlay = QWidget(self)
        overlay.setObjectName("figureOverlay")
        overlay.setStyleSheet(_OVERLAY_STYLE)
        col = QVBoxLayout(overlay)
        col.setContentsMargins(4, 4, 4, 4)
        col.setSpacing(2)

        replot_button = QToolButton(overlay)
        replot_button.setText("Replot")
        replot_button.setToolTip("Replot")
        replot_button.clicked.connect(self.replot_requested)
        col.addWidget(replot_button)

        hide_removed_button = QToolButton(overlay)
        hide_removed_button.setText("Hide Removed Points")
        hide_removed_button.setCheckable(True)
        hide_removed_button.setToolTip("Hide removed points")
        hide_removed_button.toggled.connect(self._on_hide_removed_toggled)
        col.addWidget(hide_removed_button)
        self._hide_removed_button = hide_removed_button

        autoscale_button = QToolButton(overlay)
        autoscale_button.setText("Auto-Scale")
        autoscale_button.setCheckable(True)
        autoscale_button.setToolTip("Autoscale to unmasked data")
        autoscale_button.toggled.connect(self._on_autoscale_toggled)
        col.addWidget(autoscale_button)

        overlay.adjustSize()
        self._overlay = overlay
        self._reposition_overlay()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_overlay()

    def _reposition_overlay(self) -> None:
        if self._overlay is None:
            return
        margin = 10
        self._overlay.adjustSize()
        x = self.width() - self._overlay.width() - margin
        y = self.height() - self._overlay.height() - margin
        self._overlay.move(max(0, x), max(0, y))

    def _current_axes(self):
        if self._canvas is None or not self._canvas.figure.axes:
            return None
        return self._canvas.figure.axes[0]

    def _on_view_changed(self, ax) -> None:
        """Hooked up to the axes' xlim/ylim_changed callbacks, so this fires
        for toolbar zoom/pan and the Auto-Scale button alike."""
        self._locked_xlim = ax.get_xlim()
        self._locked_ylim = ax.get_ylim()

    def _on_button_press(self, event) -> None:
        """Hides the tooltip when a click misses every pickable point.
        The Figure's own internal pick dispatcher is connected to
        button_press_event before FigurePane connects this handler, so for
        a click that *does* land on a point, _on_pick has already fired (and
        set _picked_this_click) by the time this callback runs — in that
        case leave its freshly-drawn annotation alone."""
        if self._picked_this_click:
            self._picked_this_click = False
            return
        self._hide_tooltip()

    def _on_pick(self, event) -> None:
        meta = getattr(event.artist, POINT_META_ATTR, None)
        if meta is None or not len(event.ind):
            return
        ax = self._current_axes()
        if ax is None:
            return

        self._picked_this_click = True
        self._hide_tooltip()

        index = event.ind[0]
        offsets = event.artist.get_offsets() if hasattr(event.artist, "get_offsets") else None
        if offsets is not None and len(offsets):
            x, y = offsets[index]
        else:
            xd, yd = event.artist.get_data()
            x, y = xd[index], yd[index]
        freq = meta["freq"][index]

        text = (
            f"Set: {meta['label']}\n"
            f"Freq: {freq:.4g} Hz\n"
            f"Z': {x:.4g} Ω\n"
            f"Z'': {-y:.4g} Ω"
        )
        self._tooltip_annotation = ax.annotate(
            text,
            xy=(x, y),
            xytext=(15, 15),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="0.5"),
            zorder=10,
        )
        self._canvas.draw_idle()

    def _hide_tooltip(self) -> None:
        if self._tooltip_annotation is not None:
            self._tooltip_annotation.remove()
            self._tooltip_annotation = None
            if self._canvas is not None:
                self._canvas.draw_idle()

    def _on_hide_removed_toggled(self, hidden: bool) -> None:
        self._removed_hidden = hidden
        self._apply_removed_visibility()
        if self._canvas is not None:
            self._canvas.draw_idle()

    def _apply_removed_visibility(self) -> None:
        ax = self._current_axes()
        if ax is None:
            return
        for artist in list(ax.lines) + list(ax.collections):
            if artist.get_gid() == GID_REMOVED_POINTS:
                artist.set_visible(not self._removed_hidden)

    def _on_autoscale_toggled(self, checked: bool) -> None:
        ax = self._current_axes()
        if ax is None:
            return
        if checked:
            self._zoom_to_kept_data(ax)
        elif self._default_xlim is not None and self._default_ylim is not None:
            ax.set_xlim(self._default_xlim)
            ax.set_ylim(self._default_ylim)
        self._canvas.draw_idle()

    def _zoom_to_kept_data(self, ax) -> None:
        xs: List[float] = []
        ys: List[float] = []
        for artist in list(ax.lines) + list(ax.collections):
            if artist.get_gid() != GID_KEPT_POINTS:
                continue
            if hasattr(artist, "get_offsets"):
                offsets = artist.get_offsets()
                if len(offsets):
                    xs.extend(offsets[:, 0])
                    ys.extend(offsets[:, 1])
            else:
                xd, yd = artist.get_data()
                xs.extend(xd)
                ys.extend(yd)
        if not xs:
            return
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # Breathing room around the tight data box, so points aren't drawn
        # right on the axes edge. Falls back to a fixed margin for a
        # degenerate (zero-width/height) bounding box.
        margin = 0.05
        x_pad = (xmax - xmin) * margin or 1e-6
        y_pad = (ymax - ymin) * margin or 1e-6

        xlo, xhi, ylo, yhi = equal_aspect_limits(
            xmin - x_pad, xmax + x_pad, ymin - y_pad, ymax + y_pad, include_origin=False
        )
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

    def set_figure(self, fig: Figure) -> None:
        self.clear()
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.mpl_connect("button_press_event", self._on_button_press)
        self._canvas.mpl_connect("pick_event", self._on_pick)
        if self._with_toolbar:
            self._toolbar = NavigationToolbar2QT(self._canvas, self)
            self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas)

        ax = fig.axes[0] if fig.axes else None
        if ax is not None:
            self._default_xlim = ax.get_xlim()
            self._default_ylim = ax.get_ylim()
            if self._locked_xlim is not None and self._locked_ylim is not None:
                ax.set_xlim(self._locked_xlim)
                ax.set_ylim(self._locked_ylim)
            ax.callbacks.connect("xlim_changed", self._on_view_changed)
            ax.callbacks.connect("ylim_changed", self._on_view_changed)
            self._apply_removed_visibility()

        self._canvas.draw_idle()
        if self._overlay is not None:
            if self._hide_removed_button is not None:
                self._hide_removed_button.blockSignals(True)
                self._hide_removed_button.setChecked(self._removed_hidden)
                self._hide_removed_button.blockSignals(False)
            self._overlay.raise_()
            self._reposition_overlay()

    def clear(self) -> None:
        self._tooltip_annotation = None
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
