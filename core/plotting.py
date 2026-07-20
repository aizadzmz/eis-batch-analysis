from typing import List, Optional, Tuple

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
    show_removed: bool = False,
) -> tuple[Figure, Axes]:
    """
    Plot a Nyquist plot for a single EISDataset.

    Args:
        dataset      : A single EISDataset to plot.
        title        : Plot title. Defaults to the dataset's full label.
        show         : Whether to call plt.show().
        save_path    : If provided, saves the figure to this path.
        style        : "scatter" or "line".
        show_removed : If True, also plot masked-out points as muted markers.

    Returns:
        fig, ax : The Figure and Axes objects.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    Z = dataset.impedances

    _plot_series(ax, Z, dataset.label, style)

    if show_removed:
        _plot_removed(ax, dataset.data.get_impedances(masked=True), "Removed")

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
    show_removed: bool = False,
) -> tuple[Figure, Axes]:
    """
    Overlay multiple EISDatasets on a single Nyquist plot.
    Each dataset gets its own color and legend entry.

    Args:
        datasets     : List of EISDataset objects to overlay.
        title        : Plot title.
        show         : Whether to call plt.show().
        save_path    : If provided, saves the figure to this path.
        style        : "scatter" or "line".
        show_removed : If True, also plot masked-out points as muted markers
                       (one shared "Removed" legend entry for all datasets).

    Returns:
        fig, ax : The Figure and Axes objects.
    """

    if not datasets:
        raise ValueError("No datasets provided to plot.")

    fig, ax = plt.subplots(figsize=(8, 7))

    for ds in datasets:
        Z = ds.impedances
        _plot_series(ax, Z, ds.label, style)

    if show_removed:
        removed_label = "Removed"
        for ds in datasets:
            _plot_removed(ax, ds.data.get_impedances(masked=True), removed_label)
            removed_label = None  # only the first gets a legend entry

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


def plot_residuals(
    result,
    title: Optional[str] = None,
    threshold: Optional[float] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> tuple[Figure, Axes]:
    """
    Plot the relative residuals (ΔZ'/|Z| and ΔZ''/|Z|, in percent) of a
    validation result (Kramers-Kronig or Z-HIT) against frequency, with a
    box plot summarizing each component: real on the left, imaginary on
    the right. All three panels share the same y-scale.

    Args:
        result    : A KramersKronigResult or ZHITResult (anything with
                    get_residuals_data()).
        title     : Plot title.
        threshold : If provided, draw dashed lines at ±threshold percent.
        show      : Whether to call plt.show().
        save_path : If provided, saves the figure to this path.

    Returns:
        fig, ax : The Figure and the central (residuals vs frequency) Axes.
    """
    freq, res_re, res_im = result.get_residuals_data()

    fig = plt.figure(figsize=(9, 4), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 14, 1], wspace=0.08)
    ax = fig.add_subplot(gs[1])
    ax_re = fig.add_subplot(gs[0], sharey=ax)
    ax_im = fig.add_subplot(gs[2], sharey=ax)

    re_color, im_color = "C0", "C1"

    ax.axhline(0, color="0.6", linewidth=0.8, zorder=1)
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=0.8,
                   label=f"±{threshold}% threshold")
        ax.axhline(-threshold, color="red", linestyle="--", linewidth=0.8)
    ax.semilogx(freq, res_re, "o-", ms=4, color=re_color, label="ΔZ' / |Z|")
    ax.semilogx(freq, res_im, "s-", ms=4, color=im_color, label="ΔZ'' / |Z|")
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(labelleft=False)
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="best")

    for side_ax, values, color in (
        (ax_re, res_re, re_color),
        (ax_im, res_im, im_color),
    ):
        box = side_ax.boxplot(
            values,
            widths=0.5,
            patch_artist=True,
            flierprops=dict(marker="o", markersize=3, markerfacecolor=color,
                            markeredgecolor=color),
        )
        box["boxes"][0].set(facecolor=color, alpha=0.35, edgecolor=color)
        box["medians"][0].set(color=color, linewidth=1.5)
        for part in ("whiskers", "caps"):
            for line in box[part]:
                line.set(color=color)
        side_ax.set_xticks([])
        side_ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    ax_re.set_ylabel("Relative residual (%)", fontsize=10)
    ax_re.set_xlabel("Z'", fontsize=9, color=re_color)
    ax_im.set_xlabel("Z''", fontsize=9, color=im_color)
    ax_im.tick_params(labelleft=False)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig, ax


def plot_drt(
    results: List[Tuple[str, object]],
    title: str = "DRT",
    show: bool = True,
    save_path: Optional[str] = None,
) -> tuple[Figure, Axes]:
    """
    Plot gamma vs tau (distribution of relaxation times) for one or more
    DRT results.

    Args:
        results   : List of (label, result) pairs, where each result exposes
                    get_drt_data() -> (tau, gamma), e.g. core.drt.run_drt's
                    return value. BHTResult (core.drt.run_drt_bht) instead
                    returns (tau, gamma_re, gamma_im), estimated separately
                    from the real and imaginary parts; both are plotted.
        title     : Plot title.
        show      : Whether to call plt.show().
        save_path : If provided, saves the figure to this path.

    Returns:
        fig, ax : The Figure and Axes objects.
    """
    if not results:
        raise ValueError("No DRT results provided to plot.")

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, result in results:
        data = result.get_drt_data()
        if len(data) == 3:
            tau, gamma_re, gamma_im = data
            ax.semilogx(tau, gamma_re, linewidth=1.5, label=f"{label} (Re)")
            ax.semilogx(tau, gamma_im, linewidth=1.5, linestyle="--", label=f"{label} (Im)")
        else:
            tau, gamma = data
            ax.semilogx(tau, gamma, linewidth=1.5, label=label)

    ax.set_xlabel(r"$\tau$ (s)", fontsize=10)
    ax.set_ylabel(r"$\gamma$ ($\Omega$)", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()

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


def _plot_removed(ax: Axes, Z, label: Optional[str]) -> None:
    """Draw masked-out points as muted 'x' markers."""
    if Z.size == 0:
        return
    x, y = Z.real, -Z.imag
    ax.scatter(x, y, s=25, marker="x", color="0.6", linewidths=1, label=label, zorder=2)


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