"""Docs charts: hero to docs/img/, two simple charts to findings/img/.

Reads only per-page numeric columns from out/; never prints or draws field values.

python scripts/plots.py
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DOCS_IMG = ROOT / "docs" / "img"
FINDINGS_IMG = ROOT / "findings" / "img"

# Hero colours (dark panel)
BG = "#181b1f"
PANEL = "#22252b"
GRID = "#2c3235"
TEXT = "#d8d9da"
MUTED = "#8e9297"
GREEN = "#73bf69"
ORANGE = "#ff9830"
BLUE = "#5794f2"

CLIP_THRESHOLD = 0.15
MIN_COLUMN_FILL = 0.92

FIELDS = [
    "serial number",
    "epic id",
    "name",
    "relation type",
    "relation name",
    "house number",
    "age",
    "gender",
]


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.25, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------- hero: synthetic page walkthrough ----------

PAGE_W, PAGE_H = 1983, 2806
HEADER_H, ROW_H = 230, 250
BOX_W, BOX_H = 622, 236

HERO_STYLE = {
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": TEXT,
    "font.family": "DejaVu Sans",
}


def synthetic_boxes():
    # 3 columns x 10 rows in the roll's layout, (x, y, w, h) with y down
    col_w = PAGE_W / 3
    return [
        (c * col_w + (col_w - BOX_W) / 2, HEADER_H + r * ROW_H, BOX_W, BOX_H)
        for r in range(10)
        for c in range(3)
    ]


def draw_page(ax, boxes):
    # A fake roll page: header bars, then each box as grey text bars and a photo slot
    ax.add_patch(Rectangle((0, 0), PAGE_W, PAGE_H, color="#f4f1ea"))
    ax.add_patch(Rectangle((80, 60), PAGE_W - 160, 40, color="#b9b4aa"))
    ax.add_patch(Rectangle((300, 130), PAGE_W - 600, 30, color="#cfcac0"))
    for x, y, w, h in boxes:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#55524c", lw=0.8))
        ax.add_patch(Rectangle((x + w * 0.73, y + h * 0.22), w * 0.22, h * 0.66, color="#dedad2"))
        for i, frac in enumerate([0.55, 0.62, 0.35, 0.45]):
            ax.add_patch(
                Rectangle((x + 18, y + h * (0.3 + 0.16 * i)), w * frac, 12, color="#9a958c")
            )
        ax.add_patch(Rectangle((x + 10, y + 12), 70, 30, color="#bdb8ae"))
        ax.add_patch(Rectangle((x + w * 0.5, y + 14), w * 0.3, 24, color="#bdb8ae"))
    ax.set_xlim(-20, PAGE_W + 20)
    ax.set_ylim(PAGE_H + 20, -20)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_crop(ax):
    # One box at reading size, with clearly made-up values
    ax.add_patch(Rectangle((0, 0), 100, 40, color="#f4f1ea"))
    ax.add_patch(Rectangle((1, 1), 98, 38, fill=False, ec="#55524c", lw=1.5))
    ax.add_patch(Rectangle((3, 2.5), 8, 5, fill=False, ec="#55524c", lw=1))
    ink = "#2b2a28"
    ax.text(7, 5, "7", ha="center", va="center", color=ink, fontsize=12)
    ax.text(95, 5, "ABC1234567", ha="right", va="center", color=ink, fontsize=12, weight="bold")
    lines = [
        "Name : SAMPLE NAME",
        "Father's Name : SAMPLE PARENT",
        "House Number : 12",
        "Age : 34   Gender : Male",
    ]
    for i, line in enumerate(lines):
        ax.text(4, 13 + i * 6.8, line, va="center", color=ink, fontsize=11.5)
    ax.add_patch(Rectangle((76, 10), 20, 27, color="#dedad2"))
    ax.text(86, 23.5, "photo", ha="center", va="center", color="#8f8a80", fontsize=10)
    ax.set_xlim(-1, 101)
    ax.set_ylim(41, -1)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_json(ax):
    # The record the crop becomes, with JSON-ish colouring
    values = ["7", "ABC1234567", "SAMPLE NAME", "Father", "SAMPLE PARENT", "12", "34", "Male"]
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.06),
            0.96,
            0.88,
            boxstyle="round,pad=0,rounding_size=0.03",
            fc=PANEL,
            ec=GRID,
            lw=1.5,
            transform=ax.transAxes,
        )
    )
    y = 0.86
    ax.text(0.08, y, "{", family="monospace", fontsize=14, color=TEXT)
    for key, value in zip(FIELDS, values, strict=False):
        y -= 0.083
        ax.text(0.11, y, f'"{key}":', family="monospace", fontsize=11, color=BLUE)
        ax.text(0.55, y, f'"{value}"', family="monospace", fontsize=11, color=GREEN)
    ax.text(0.08, y - 0.083, "}", family="monospace", fontsize=14, color=TEXT)


def plot_hero():
    boxes = synthetic_boxes()
    pick = boxes[6]  # row 3, column 1
    fig = plt.figure(figsize=(17, 6.6))
    lefts, widths = [0.0, 0.245, 0.49, 0.735], [0.21, 0.21, 0.21, 0.265]
    axes = [fig.add_axes([x, 0.04, w, 0.80]) for x, w in zip(lefts, widths, strict=False)]
    titles = ["1  Scanned page", "2  Boxes detected", "3  One box crop", "4  JSON record"]
    notes = [
        "one embedded image per page",
        "OpenCV lines + contours, 3 x 10 grid",
        "sent alone to Qwen3-VL-8B on vLLM",
        "schema-constrained, 8 fields",
    ]

    draw_page(axes[0], boxes)
    draw_page(axes[1], boxes)
    for x, y, w, h in boxes:
        colour = ORANGE if (x, y, w, h) == pick else GREEN
        axes[1].add_patch(
            Rectangle((x, y), w, h, fill=False, ec=colour, lw=3 if colour == ORANGE else 1.8)
        )
    axes[1].add_patch(Rectangle((0, 0), PAGE_W, HEADER_H - 30, color=BLUE, alpha=0.18))
    draw_crop(axes[2])
    draw_json(axes[3])

    for ax, title, note in zip(axes, titles, notes, strict=False):
        fig.text(ax.get_position().x0, 0.93, title, fontsize=17, weight="bold", color=TEXT)
        fig.text(ax.get_position().x0, 0.885, note, fontsize=12.5, color=MUTED)
    for i in range(3):
        a, b = axes[i].get_position(), axes[i + 1].get_position()
        fig.patches.append(
            FancyArrowPatch(
                (a.x1 + 0.006, 0.44),
                (b.x0 - 0.006, 0.44),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=26,
                color=MUTED,
                lw=2,
            )
        )
    fig.text(
        0.0,
        -0.02,
        "Synthetic page in the roll layout; all names and ids are made up.",
        fontsize=11.5,
        color=MUTED,
    )
    save(fig, DOCS_IMG / "hero.png")


# ---------- simple findings charts (default light style) ----------


def plot_concurrency():
    # Numbers from the three benchmark runs (findings/concurrency.md)
    conc = [8, 16, 32]
    s_rec = [0.373, 0.256, 0.252]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(conc, s_rec, marker="o")
    for c, s in zip(conc, s_rec, strict=False):
        ax.annotate(f"{s:.3f}", (c, s), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_xticks(conc)
    ax.set_xlim(5, 35)
    ax.set_ylim(0, 0.42)
    ax.set_xlabel("Concurrent requests to vLLM")
    ax.set_ylabel("Seconds per record")
    ax.set_title("Seconds per record vs concurrency (RTX 4090)")
    ax.grid(alpha=0.3)
    save(fig, FINDINGS_IMG / "concurrency.png")


def plot_clip_vs_fill():
    # One point per ground-truth page; pages with no detected box drawn at x = 0.70
    truth = {
        (r["pdf"], r["page"]): r["true_label"] for r in read_csv(OUT / "truth-check" / "pages.csv")
    }
    rows = read_csv(OUT / "eval-bench-32" / "pages.csv")
    no_box = 0.70
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, marker, name in [("voter", "o", "voter page"), ("non_voter", "x", "non-voter page")]:
        sel = [r for r in rows if truth[(r["pdf"], r["page"])] == label]
        xs = [float(r["column_fill"]) if r["column_fill"] else no_box for r in sel]
        ys = [float(r["p_voter"]) for r in sel]
        ax.scatter(xs, ys, marker=marker, label=f"{name} ({len(sel)})")
    ax.axvline(MIN_COLUMN_FILL, color="grey", ls="--", lw=1)
    ax.axhline((1 + CLIP_THRESHOLD) / 2, color="grey", ls=":", lw=1)
    ax.text(MIN_COLUMN_FILL - 0.003, 0.42, "fill 0.92", ha="right")
    ax.set_ylim(0.4, 1.02)
    ax.text(0.71, (1 + CLIP_THRESHOLD) / 2 + 0.02, "margin 0.15")
    ax.set_xlim(0.68, 0.97)
    ax.set_xticks([no_box, 0.75, 0.80, 0.85, 0.90, 0.95])
    ax.set_xticklabels(["no boxes", "0.75", "0.80", "0.85", "0.90", "0.95"])
    ax.set_xlabel("Column fill (median box width / one third of page width)")
    ax.set_ylabel("CLIP p(voter)")
    ax.set_title("CLIP p(voter) vs column fill, 184 pages")
    ax.legend(loc="center left")
    ax.grid(alpha=0.3)
    save(fig, FINDINGS_IMG / "clip_vs_fill.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="draw one chart, e.g. hero")
    args = parser.parse_args()
    charts = {
        "hero": plot_hero,
        "concurrency": plot_concurrency,
        "clip_vs_fill": plot_clip_vs_fill,
    }
    for name, fn in charts.items():
        if args.only in (None, name):
            if name == "hero":
                with plt.rc_context(HERO_STYLE):
                    fn()
            else:
                fn()


if __name__ == "__main__":
    main()
