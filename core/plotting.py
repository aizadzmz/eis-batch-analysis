from typing import List, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from core.io_utils import EISDataset

def plot_single(
    dataset: EISDataset,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    style: str = "scatter",
) -> tuple[Figure, Axes]:
    """
    Plot a Nyquist plot for a single EISDataset.

    Args:
        dataset   : A single EISDataset to plot.
        title     : Plot title. Defaults to the dataset's full label.
        show      : Whether to call plt.show().
        save_path : If provided, saves the figure to this path.
        style     : "scatter" or "line".

    Returns:
        fig, ax : The Figure and Axes objects.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    Z = dataset.impedances

    _plot_series(ax, Z, dataset.label, style)

    ax.set_title(title or dataset.full_label, fontsize=12)
    fig.canvas.manager.set_window_title(dataset.full_label)
    _format_ax(ax)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig, ax


def plot_overlay(
    datasets: List[EISDataset],
    title: str = "Nyquist Plot",
    show: bool = True,
    save_path: Optional[str] = None,
    style: str = "scatter",
) -> tuple[Figure, Axes]:
    """
    Overlay multiple EISDatasets on a single Nyquist plot.
    Each dataset gets its own color and legend entry.

    Args:
        datasets  : List of EISDataset objects to overlay.
        title     : Plot title.
        show      : Whether to call plt.show().
        save_path : If provided, saves the figure to this path.
        style     : "scatter" or "line".

    Returns:
        fig, ax : The Figure and Axes objects.
    """

    if not datasets:
        raise ValueError("No datasets provided to plot.")

    fig, ax = plt.subplots(figsize=(8, 7))

    for ds in datasets:
        Z = ds.impedances
        _plot_series(ax, Z, ds.label, style)

    window_title = datasets[0].source_file
    ax.set_title(title or window_title, fontsize=12)
    fig.canvas.manager.set_window_title(window_title)
    _format_ax(ax)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig, ax


def _plot_series(ax: Axes, Z, label: str, style: str) -> None:
    """Draw one dataset's impedance on ax, either as a scatter or a connected line."""
    x, y = Z.real, -Z.imag
    if style == "scatter":
        ax.scatter(x, y, s=15, label=label)
    elif style == "line":
        ax.plot(x, y, linewidth=1, label=label)
    else:
        raise ValueError(f"Unknown style '{style}'; expected 'scatter' or 'line'.")


def _format_ax(ax: Axes) -> None:
    """Apply standard Nyquist formatting to an axes."""
    ax.set_xlabel("Z' (Ω)", fontsize=10)
    ax.set_ylabel("-Z'' (Ω)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Ensure the origin falls within each axis's range (not necessarily at an edge).
    x_lo, x_hi = min(xmin, 0), max(xmax, 0)
    y_lo, y_hi = min(ymin, 0), max(ymax, 0)

    # Pad the narrower axis so both spans match, keeping the scale equal.
    span = max(x_hi - x_lo, y_hi - y_lo)
    x_pad = (span - (x_hi - x_lo)) / 2
    y_pad = (span - (y_hi - y_lo)) / 2
    ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
    ax.set_aspect("equal", adjustable="box")

    ax.axhline(0, color="#CC9D33", linewidth=1, zorder=1)
    ax.axvline(0, color="#CC9D33", linewidth=1, zorder=1)

    ax.legend(
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0,
    )