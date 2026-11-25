"""Draw the README figures into docs/images/.

Usage:
    python scripts/make_diagrams.py
    python scripts/make_diagrams.py --out-dir docs/images --only reference-numbers

chunked-prediction.png  the chunked method, not yet released (schematic, no data)
reference-numbers.png   DRG-LLaMA's Table 2 (Wang et al., npj Digit Med 2024;7:16)
encoder-baselines.png   the released Bio_ClinicalBERT and ModernBERT-base baselines (schematic)

Figures are drawn 1,760 px wide for display at 880 px. Every text box is checked against the
box it belongs to, and the script stops if a label overflows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
ACCENT = "#eb6834"
ACCENT_TINT = "#fbe6dc"
BLUE = "#2a78d6"
BLUE_TINT = "#e2ecf8"
PANEL = "#f3f2ee"

DPI = 200
WIDTH = 1760
TITLE_PT, SUBTITLE_PT, LABEL_PT, BODY_PT, SMALL_PT = 16, 11, 11, 10, 9.5

# Hiragino Sans or a similar gothic where one is installed, with DejaVu Sans (which ships with
# matplotlib) as the fallback. Only installed fonts are listed, so matplotlib does not warn.
FONTS = ("Hiragino Sans", "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "DejaVu Sans")
_INSTALLED = {f.name for f in font_manager.fontManager.ttflist}

plt.rcParams.update(
    {
        "font.family": [name for name in FONTS if name in _INSTALLED],
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
    }
)


class Canvas:
    """A figure in pixel coordinates (origin top left) that checks its labels fit."""

    def __init__(self, height: int, width: int = WIDTH):
        self.width, self.height = width, height
        self.fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI, facecolor=SURFACE)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        self.ax.axis("off")
        self.checks: list[tuple] = []

    def box(self, x0, y0, x1, y1, fill=SURFACE, edge=BASELINE, lw=2.0, radius=14, z=1):
        patch = FancyBboxPatch(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw * 72 / DPI,
            zorder=z,
        )
        self.ax.add_patch(patch)
        return (x0, y0, x1, y1)

    def rect(self, x0, y0, x1, y1, color, z=2):
        self.ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                boxstyle=f"round,pad=0,rounding_size={min(4, (y1 - y0) / 2)}",
                facecolor=color,
                edgecolor="none",
                zorder=z,
            )
        )

    def text(
        self,
        x,
        y,
        s,
        size=BODY_PT,
        weight="normal",
        color=INK,
        ha="left",
        va="center",
        inside=None,
        z=5,
    ):
        handle = self.ax.text(
            x,
            y,
            s,
            fontsize=size,
            fontweight=weight,
            color=color,
            ha=ha,
            va=va,
            zorder=z,
            linespacing=1.35,
        )
        self.checks.append((handle, inside))
        return handle

    def arrow(self, x0, y0, x1, y1, color=INK_2, lw=2.4):
        self.ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=lw * 72 / DPI,
                color=color,
                shrinkA=0,
                shrinkB=0,
                zorder=3,
            )
        )

    def title(self, title: str, subtitle: str, x: int = 60):
        self.text(x, 58, title, size=TITLE_PT, weight="bold")
        self.text(x, 112, subtitle, size=SUBTITLE_PT, color=INK_2)

    def save(self, path: Path, pad: int = 8):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        problems = []
        for handle, inside in self.checks:
            bb = handle.get_window_extent(renderer)
            x0, x1 = bb.x0, bb.x1
            y0, y1 = self.height - bb.y1, self.height - bb.y0
            limit = (pad, pad, self.width - pad, self.height - pad)
            if inside is not None:
                limit = (inside[0] + pad, inside[1] + pad / 2, inside[2] - pad, inside[3] - pad / 2)
            if x0 < limit[0] or y0 < limit[1] or x1 > limit[2] or y1 > limit[3]:
                problems.append(f"{handle.get_text()!r} at {x0:.0f},{y0:.0f}-{x1:.0f},{y1:.0f}")
        if problems:
            raise SystemExit(f"{path.name}: text overflows its box:\n  " + "\n  ".join(problems))
        self.fig.savefig(path, dpi=DPI, facecolor=SURFACE, metadata={"Software": None})
        plt.close(self.fig)
        print(f"wrote {path}")


def score_strip(c: Canvas, x0, y_base, heights, winner, width=150, bar_gap=8):
    """A small bar chart of scores over DRGs; the winning DRG's bar is in the accent colour."""
    n = len(heights)
    bar = (width - bar_gap * (n - 1)) / n
    c.ax.plot([x0 - 4, x0 + width + 4], [y_base, y_base], color=BASELINE, lw=1.2, zorder=2)
    for i, h in enumerate(heights):
        left = x0 + i * (bar + bar_gap)
        c.rect(left, y_base - h, left + bar, y_base - 1, ACCENT if i == winner else BASELINE)


def chunked_prediction(out: Path) -> None:
    c = Canvas(height=930)
    c.title(
        "Read the whole note in chunks, then combine the chunk scores",
        "Schematic of the chunked method, not data.",
    )

    steps = [
        (60, "1", "Split into overlapping chunks"),
        (640, "2", "Score each chunk"),
        (1180, "3", "Combine into one DRG"),
    ]
    for x, number, label in steps:
        c.text(x, 182, number, size=LABEL_PT, weight="bold")
        c.text(x + 30, 182, label, size=LABEL_PT, weight="bold", color=INK_2)

    # The discharge summary, its sections drawn as heading plus grey text lines.
    card = c.box(60, 222, 500, 900, fill="#ffffff", edge=BASELINE)
    c.text(84, 258, "Discharge summary", size=LABEL_PT, weight="bold", inside=card)
    c.ax.plot([84, 476], [290, 290], color=GRID, lw=1.5, zorder=2)

    sections = [
        ("Chief complaint", 1, False),
        ("Present illness", 2, False),
        ("Exam and test results", 2, False),
        ("Brief hospital course", 2, True),
        ("Discharge diagnosis", 1, False),
    ]
    y = 312
    body_top = y
    for heading, n_lines, is_bhc in sections:
        top = y
        if is_bhc:
            height = 44 + 34 + 30 * n_lines + 8
            band = (72, top - 10, 488, top + height - 6)
            c.box(*band, fill=BLUE_TINT, edge="none", radius=10, z=1.5)
            c.rect(72, top - 10, 80, top + height - 6, BLUE, z=2)
        c.text(96, y + 14, heading, size=BODY_PT, weight="bold", inside=card)
        y += 44
        if is_bhc:
            c.text(96, y + 8, "DRG-LLaMA's only input", size=SMALL_PT, color=INK_2, inside=card)
            y += 34
        for i in range(n_lines):
            right = 460 if i < n_lines - 1 else 380
            c.rect(96, y, right, y + 10, BASELINE)
            y += 30
        y += 18
    body_bottom = y - 18

    # Three overlapping chunks along the right edge of the note.
    span = body_bottom - body_top
    length = span * 0.42
    starts = [body_top, body_top + (span - length) / 2, body_bottom - length]
    mids = []
    for i, start in enumerate(starts):
        x = 516 + i * 22
        c.rect(x, start, x + 12, start + length, ACCENT, z=3)
        mids.append(start + length / 2)

    # The same encoder and DRG head scores every chunk.
    heights = [[30, 62, 22, 38, 16, 26], [26, 44, 18, 58, 20, 30], [22, 66, 30, 34, 14, 40]]
    for i, mid in enumerate(mids):
        x_from = 516 + i * 22 + 12
        c.arrow(x_from + 6, mid, 640, mid)
        enc = c.box(640, mid - 58, 870, mid + 58, fill=PANEL, edge=BASELINE)
        c.text(755, mid - 16, "Encoder", size=BODY_PT, weight="bold", ha="center", inside=enc)
        c.text(755, mid + 18, "+ DRG head", size=BODY_PT, ha="center", inside=enc)
        c.arrow(870, mid, 906, mid)
        score_strip(c, 918, mid + 34, heights[i], winner=1, width=150)
    c.text(755, mids[-1] + 86, "same weights", size=SMALL_PT, color=INK_2, ha="center")

    # Combine the normalised chunk logits into one prediction.
    comb = c.box(1180, mids[0] - 62, 1430, mids[-1] + 62, fill=PANEL, edge=INK_2)
    for mid in mids:
        c.arrow(1080, mid, 1180, mid)
    centre = (mids[0] + mids[-1]) / 2
    c.text(1305, centre - 78, "Combine", size=LABEL_PT, weight="bold", ha="center", inside=comb)
    c.text(
        1305,
        centre + 20,
        "normalise each\nchunk's logits,\nthen aggregate",
        size=BODY_PT,
        color=INK_2,
        ha="center",
        inside=comb,
    )

    c.arrow(1430, centre, 1476, centre)
    pred = c.box(1476, centre - 112, 1700, centre + 112, fill=ACCENT_TINT, edge=ACCENT, lw=3)
    score_strip(c, 1513, centre - 14, [20, 70, 18, 40, 12, 28], winner=1, width=150)
    c.text(1588, centre + 34, "One DRG", size=LABEL_PT, weight="bold", ha="center", inside=pred)
    c.text(1588, centre + 72, "per admission", size=BODY_PT, ha="center", inside=pred)
    c.save(out)


# DRG-LLaMA, Table 2: MIMIC-IV, MS-DRG v34, brief hospital course as input.
TOKENS = [340, 512, 1024]
TABLE_2 = {
    "7B": {"top1": [49.3, 52.0, 53.9], "macro_f1": [0.303, 0.327, 0.346]},
    "13B": {"top1": [49.9, 52.4, 54.6], "macro_f1": [0.312, 0.334, 0.361]},
}


def reference_numbers(out: Path) -> None:
    c = Canvas(height=860)
    c.title(
        "DRG-LLaMA was still improving at its longest input, 1,024 tokens",
        "MIMIC-IV, 738 MS-DRGs, brief hospital course only. Wang et al., npj Digit Med 2024, "
        "Table 2.",
    )
    colours = {"13B": BLUE, "7B": INK_2}
    panels = [
        ("top1", "Top-1 accuracy", (48, 57), [48, 50, 52, 54, 56], "{:.0f}%", "{:.1f}%"),
        ("macro_f1", "Macro-F1", (0.28, 0.37), [0.28, 0.30, 0.32, 0.34, 0.36], "{:.2f}", "{:.3f}"),
    ]
    lefts = [145, 1000]
    axes = []
    for (key, label, ylim, ticks, tick_fmt, value_fmt), left in zip(panels, lefts):
        ax = c.fig.add_axes((left / WIDTH, (860 - 720) / 860, 560 / WIDTH, 510 / 860))
        axes.append((ax, label))
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log", base=2)
        ax.set_xlim(220, 1160)
        ax.set_ylim(*ylim)
        ax.set_xticks(TOKENS, [f"{t:,}" for t in TOKENS])
        ax.minorticks_off()
        ax.set_yticks(ticks, [tick_fmt.format(t) for t in ticks])
        ax.tick_params(labelsize=BODY_PT, length=0, pad=8)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.set_xlabel("Maximum input tokens (log scale)", fontsize=BODY_PT, labelpad=10)
        for model in ("7B", "13B"):
            values = TABLE_2[model][key]
            ax.plot(
                TOKENS,
                values,
                color=colours[model],
                lw=2.6,
                marker="o",
                ms=7,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                zorder=3,
                clip_on=False,
            )
            end = ax.annotate(
                f"{model}  {value_fmt.format(values[-1])}",
                (TOKENS[-1], values[-1]),
                xytext=(10, 0),
                textcoords="offset points",
                va="center",
                fontsize=BODY_PT,
                fontweight="bold",
                color=INK if model == "13B" else INK_2,
            )
            # Start values sit left of their points, clear of the rising lines; the
            # background box keeps gridlines from running through the digits.
            first = ax.annotate(
                value_fmt.format(values[0]),
                (TOKENS[0], values[0]),
                xytext=(-10, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=SMALL_PT,
                color=INK_2,
                bbox={"boxstyle": "square,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"},
            )
            c.checks += [(end, None), (first, None)]
    # Panel titles line up with the left edge of their tick labels.
    c.fig.canvas.draw()
    renderer = c.fig.canvas.get_renderer()
    for ax, label in axes:
        x = min(t.get_window_extent(renderer).x0 for t in ax.get_yticklabels())
        c.text(x, 178, label, size=LABEL_PT, weight="bold")
    c.save(out)


def encoder_baselines(out: Path) -> None:
    c = Canvas(height=720)
    c.title(
        "The long-context baseline can read 16 times as many tokens",
        "Same pipeline and training settings; the ModernBERT-base run adds bf16 and gradient "
        "checkpointing.",
    )
    stages = [
        (60, 300, "Note CSV", "text and\nDRG code"),
        (340, 610, "Truncate", "first 512 or\n8,192 tokens"),
        (650, 980, "Encoder + head", "Bio_ClinicalBERT or\nModernBERT-base"),
        (1020, 1360, "Best checkpoint", "by validation\nmacro-F1"),
        (1400, 1700, "Metrics", "top-1, top-5,\ntop-10, macro-F1"),
    ]
    top, bottom = 180, 380
    for i, (x0, x1, head, body) in enumerate(stages):
        accent = head == "Truncate"
        b = c.box(
            x0,
            top,
            x1,
            bottom,
            fill=ACCENT_TINT if accent else PANEL,
            edge=ACCENT if accent else BASELINE,
            lw=3 if accent else 2,
        )
        mid = (x0 + x1) / 2
        c.text(mid, top + 52, head, size=LABEL_PT, weight="bold", ha="center", inside=b)
        c.text(mid, top + 128, body, size=BODY_PT, color=INK_2, ha="center", inside=b)
        if i < len(stages) - 1:
            c.arrow(x1, (top + bottom) / 2, stages[i + 1][0], (top + bottom) / 2)

    # The token limits, drawn to scale.
    c.text(60, 462, "Most tokens kept per note, to scale", size=BODY_PT, weight="bold")
    x0, full = 360, 1120
    rows = [("Bio_ClinicalBERT", 512, INK_2, 524), ("ModernBERT-base", 8192, ACCENT, 596)]
    for name, tokens, colour, y in rows:
        x1 = x0 + full * tokens / 8192
        c.text(60, y, name, size=BODY_PT)
        c.rect(x0, y - 20, x1, y + 20, colour)
        c.text(x1 + 14, y, f"{tokens:,} tokens", size=BODY_PT, weight="bold")
    c.text(
        60,
        672,
        "Notes over the limit are cut, keeping the start by default or the end on request.",
        size=BODY_PT,
        color=INK_2,
    )
    c.save(out)


FIGURES = {
    "chunked-prediction": chunked_prediction,
    "reference-numbers": reference_numbers,
    "encoder-baselines": encoder_baselines,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the README figures.")
    parser.add_argument("--out-dir", default="docs/images")
    parser.add_argument("--only", choices=sorted(FIGURES), action="append")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.only or FIGURES:
        FIGURES[name](out_dir / f"{name}.png")


if __name__ == "__main__":
    main()
