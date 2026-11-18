"""Shared figure style: validated categorical palette (slots in fixed order),
thin marks, recessive grid, direct labels.  Light-mode static figures."""
import matplotlib as mpl

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelsize": 9.5, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.7, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2.0, "lines.markersize": 5,
        "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9.5,
        "axes.prop_cycle": mpl.cycler(color=SERIES), "figure.dpi": 110, "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def label_end(ax, x, y, text, color, dx=4, dy=0):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                color=TEXT2, fontsize=8.5, va="center",
                bbox=dict(boxstyle="square,pad=0", fc="none", ec="none"))
    ax.plot([], [], color=color)
