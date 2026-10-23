"""Docs charts from out/ and docs/benchmarks.md -> docs/img/ and findings/img/.

Reads only counts and per-page numeric columns; never prints or draws field values.

python scripts/plots.py
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DOCS_IMG = ROOT / "docs" / "img"
FINDINGS_IMG = ROOT / "findings" / "img"
BENCH_MD = ROOT / "docs" / "benchmarks.md"

# Grafana-like dark theme
BG = "#181b1f"
PANEL = "#22252b"
GRID = "#2c3235"
TEXT = "#d8d9da"
MUTED = "#8e9297"
GREEN = "#73bf69"
ORANGE = "#ff9830"
RED = "#f2495c"
BLUE = "#5794f2"
YELLOW = "#fade2a"
PURPLE = "#b877d9"

HI_RES, LO_RES = "1983", "949"  # page widths of the 240 DPI and 115 DPI scans
CLIP_THRESHOLD = 0.15
MIN_COLUMN_FILL = 0.92
RUNS = [8, 16, 32]

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
FIELD_COLOURS = {
    "name": ORANGE,
    "relation name": RED,
    "house number": YELLOW,
    "relation type": BLUE,
    "epic id": PURPLE,
    "gender": GREEN,
}


def theme():
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    )


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.25, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def bench_numbers():
    # Headline numbers as written by the scorer
    md = BENCH_MD.read_text(encoding="utf-8")

    def grab(pattern):
        return re.search(pattern, md).group(1)

    return {
        "field_acc": grab(r"Field accuracy, overall \| ([\d.]+)%"),
        "field_lo": grab(r"Field accuracy, overall \| [\d.]+% \(95% Wilson ([\d.]+)%"),
        "field_hi": grab(r"Field accuracy, overall \| [\d.]+% \(95% Wilson [\d.]+% to ([\d.]+)%"),
        "record_acc": grab(r"\| overall \| \d+ \| [\d.]+% \| ([\d.]+)%"),
        "records": int(grab(r"\| overall \| (\d+) \|")),
        "s_per_record": grab(r"Seconds per record \| ([\d.]+)"),
        "usd_per_1k": grab(r"\$ per 1000 records \| ([\d.]+)"),
        "dropped": int(grab(r"Voter pages dropped: (\d+)")),
        "voter_pages": int(grab(r"\| voter \| (\d+) \|")),
        "hourly_usd": float(grab(r"`HOURLY_USD` (\d+\.\d+)")),
        "hi_records": int(grab(r"\| 1983x2806 \| (\d+) \|")),
        "lo_records": int(grab(r"\| 949x1343 \| (\d+) \|")),
    }


def mistakes(run):
    return read_csv(OUT / run / "mistakes.csv")


def pages_with_truth():
    # Eval page decisions joined with the true page label on (pdf, page)
    truth = {
        (r["pdf"], r["page"]): r["true_label"] for r in read_csv(OUT / "truth-check" / "pages.csv")
    }
    rows = read_csv(OUT / "eval-bench-32" / "pages.csv")
    for r in rows:
        r["true_label"] = truth[(r["pdf"], r["page"])]
    return rows


# ---------- hero: synthetic page walkthrough ----------

PAGE_W, PAGE_H = 1983, 2806
HEADER_H, ROW_H = 230, 250
BOX_W, BOX_H = 622, 236


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


# ---------- KPI stat cards ----------


def plot_kpis():
    b = bench_numbers()
    cards = [
        (
            "Field accuracy",
            f"{b['field_acc']}%",
            f"Wilson {b['field_lo']} to {b['field_hi']}%",
            GREEN,
        ),
        ("Record accuracy", f"{b['record_acc']}%", "all 8 fields right", GREEN),
        (
            "Voter pages dropped",
            f"{b['dropped']}/{b['voter_pages']}",
            "184 pages classified",
            GREEN,
        ),
        ("Seconds per record", b["s_per_record"], "RTX 4090, 32 in flight", BLUE),
        ("$ per 1k records", f"${b['usd_per_1k']}", f"GPU at ${b['hourly_usd']:.2f}/h", BLUE),
    ]
    fig = plt.figure(figsize=(16, 3.2))
    for i, (title, value, sub, colour) in enumerate(cards):
        ax = fig.add_axes([i * 0.2 + 0.005, 0.0, 0.19, 1.0])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.0, 0.0),
                1.0,
                1.0,
                boxstyle="round,pad=0,rounding_size=0.04",
                fc=PANEL,
                ec=GRID,
                lw=1.5,
            )
        )
        ax.text(0.5, 0.84, title, ha="center", fontsize=14, color=TEXT)
        ax.text(
            0.5, 0.45, value, ha="center", va="center", fontsize=38, weight="bold", color=colour
        )
        ax.text(0.5, 0.12, sub, ha="center", fontsize=12.5, color=MUTED)
    fig.text(
        0.005,
        -0.12,
        f"{b['records']:,} hand-checked records from 6 PDFs.",
        fontsize=12,
        color=MUTED,
    )
    save(fig, DOCS_IMG / "kpis.png")


# ---------- pipeline Sankey ----------


def band(ax, x0, top0, bot0, x1, top1, bot1, colour, alpha=0.45):
    # A smooth flow between two vertical spans (y grows downwards)
    x0, x1 = x0 + 0.012, x1 - 0.012  # start and end at the node edges
    t = [i / 40 for i in range(41)]
    s = [3 * v**2 - 2 * v**3 for v in t]
    xs = [x0 + (x1 - x0) * v for v in t]
    upper = [top0 + (top1 - top0) * v for v in s]
    lower = [bot0 + (bot1 - bot0) * v for v in s]
    pts = list(zip(xs, upper, strict=False)) + list(zip(xs[::-1], lower[::-1], strict=False))
    ax.add_patch(Polygon(pts, closed=True, fc=colour, ec="none", alpha=alpha))


def node(ax, x, top, bot, colour, label, side="right"):
    ax.add_patch(Rectangle((x - 0.012, top), 0.024, bot - top, fc=colour, ec="none"))
    tx = x + 0.025 if side == "right" else x - 0.025
    ax.text(
        tx,
        (top + bot) / 2,
        label,
        ha="left" if side == "right" else "right",
        va="center",
        fontsize=13.5,
        color=TEXT,
    )


def plot_pipeline_flow():
    pages = read_csv(OUT / "eval-bench-32" / "pages.csv")
    n = len(pages)
    outcome = Counter(r["outcome"] for r in pages)
    clip_v = outcome["clip_voter"]
    fallback = outcome["qwen_fallback"]
    fb_voter = sum(
        1 for r in pages if r["outcome"] == "qwen_fallback" and r["final_label"] == "voter"
    )
    fb_non = fallback - fb_voter
    voter_pages = clip_v + fb_voter
    records = bench_numbers()["records"]
    bad = len({(m["pdf"], m["page"], m["box"]) for m in mistakes("eval-bench-32")})
    good = records - bad

    fig, ax = plt.subplots(figsize=(15, 6.4))
    ax.set_facecolor(BG)
    ax.axis("off")
    gap = 0.06

    def h(k):
        return k / n  # page heights; records share the voter-page span

    xs = [0.04, 0.30, 0.56, 0.80]

    # column 0: all pages
    node(ax, xs[0], 0, 1, BLUE, f"{n} pages", side="left")
    # column 1: CLIP trusted vs fallback
    c1 = (0, h(clip_v)), (h(clip_v) + gap, 1 + gap)
    band(ax, xs[0], 0, h(clip_v), xs[1], *c1[0], GREEN)
    band(ax, xs[0], h(clip_v), 1, xs[1], *c1[1], ORANGE)
    node(ax, xs[1], *c1[0], GREEN, f"{clip_v} CLIP voter\n(trusted)")
    node(ax, xs[1], *c1[1], ORANGE, f"{fallback} Qwen fallback")
    # column 2: final page labels
    c2v = (0, h(voter_pages))
    c2n = (h(voter_pages) + 2 * gap, 1 + 2 * gap)
    band(ax, xs[1], *c1[0], xs[2], 0, h(clip_v), GREEN)
    if fb_voter:
        band(ax, xs[1], c1[1][0], c1[1][0] + h(fb_voter), xs[2], h(clip_v), h(voter_pages), ORANGE)
    band(ax, xs[1], c1[1][0] + h(fb_voter), c1[1][1], xs[2], *c2n, MUTED, alpha=0.35)
    node(ax, xs[2], *c2v, GREEN, f"{voter_pages} voter pages")
    node(ax, xs[2], *c2n, MUTED, f"{fb_non} non-voter\n(skipped)")
    # column 3: records, correct vs with an error
    span = h(voter_pages)
    cut = span * good / records
    band(ax, xs[2], 0, cut, xs[3], 0, cut, GREEN)
    band(ax, xs[2], cut, span, xs[3], cut + gap, span + gap, RED, alpha=0.6)
    node(ax, xs[3], 0, cut, GREEN, f"{good:,} records\nall 8 fields right")
    node(ax, xs[3], cut + gap, span + gap, RED, f"{bad} records with\n1+ field error")
    ax.text(
        xs[2] + (xs[3] - xs[2]) / 2,
        -0.05,
        f"{records:,} records extracted",
        ha="center",
        fontsize=13.5,
        color=TEXT,
        weight="bold",
    )
    ax.text(xs[2] - 0.03, -0.05, "0 voter pages dropped", ha="right", fontsize=13.5, color=GREEN)

    ax.set_xlim(-0.08, 1.0)
    ax.set_ylim(1 + 2 * gap + 0.03, -0.1)
    save(fig, DOCS_IMG / "pipeline_flow.png")


# ---------- page classification scatter ----------


def plot_page_classification():
    rows = pages_with_truth()
    no_box_x = 0.66  # pages with no detected rectangle
    lo = (1 - CLIP_THRESHOLD) / 2  # p_voter where the margin is 0.15
    hi = (1 + CLIP_THRESHOLD) / 2

    fig, ax = plt.subplots(figsize=(12, 7))
    x0, x1 = 0.63, 1.0
    ax.add_patch(
        Rectangle((MIN_COLUMN_FILL, hi), x1 - MIN_COLUMN_FILL, 1.05 - hi, fc=GREEN, alpha=0.13)
    )
    ax.add_patch(Rectangle((x0, -0.05), MIN_COLUMN_FILL - x0, lo + 0.05, fc=BLUE, alpha=0.13))
    ax.add_patch(Rectangle((x0, -0.05), x1 - x0, 1.1, fc=ORANGE, alpha=0.06, zorder=0))
    ax.axvline(MIN_COLUMN_FILL, color=TEXT, ls="--", lw=1.4)
    for y in (lo, hi):
        ax.axhline(y, color=TEXT, ls=":", lw=1.2)
    ax.axvline(0.69, color=GRID, lw=1)

    for label, colour, marker, name in [
        ("voter", GREEN, "o", "true voter page"),
        ("non_voter", ORANGE, "D", "true non-voter page"),
    ]:
        sel = [r for r in rows if r["true_label"] == label]
        # small fixed offsets so stacked no-box pages stay visible
        xs = [
            float(r["column_fill"]) if r["column_fill"] else no_box_x + 0.006 * ((i % 5) - 2)
            for i, r in enumerate(sel)
        ]
        ys = [float(r["p_voter"]) for r in sel]
        ax.scatter(
            xs,
            ys,
            s=70,
            c=colour,
            marker=marker,
            edgecolors=BG,
            linewidths=0.8,
            label=f"{name} ({len(sel)})",
            zorder=3,
        )

    ax.text(
        0.975,
        0.80,
        "CLIP\ntrusted:\nvoter",
        color=GREEN,
        fontsize=12,
        ha="center",
        va="center",
        weight="bold",
    )
    ax.text(
        0.80, 0.18, "CLIP trusted: non-voter", color=BLUE, fontsize=13, ha="center", weight="bold"
    )
    ax.text(
        0.80,
        0.92,
        "Qwen fallback\n(CLIP says voter, geometry says no)",
        color=ORANGE,
        fontsize=13,
        ha="center",
        weight="bold",
    )
    ax.text(MIN_COLUMN_FILL + 0.002, -0.03, "fill 0.92", color=TEXT, fontsize=12)
    ax.text(0.695, hi + 0.012, "margin 0.15", color=TEXT, fontsize=12)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([no_box_x, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    ax.set_xticklabels(["no\nboxes", "0.70", "0.75", "0.80", "0.85", "0.90", "0.95"])
    ax.set_xlabel("Column fill (median box width / one third of page width)")
    ax.set_ylabel("CLIP p(voter)")
    ax.set_title("CLIP alone would trust 6 non-voter pages; geometry vetoes them")
    ax.legend(loc="lower left")
    save(fig, FINDINGS_IMG / "page_classification.png")


# ---------- concurrency ----------


def plot_concurrency():
    hourly = bench_numbers()["hourly_usd"]
    s_rec, errors = [], []
    for c in RUNS:
        t = json.loads((OUT / f"eval-bench-{c}" / "timing.json").read_text(encoding="utf-8"))
        s_rec.append(t["seconds"] / t["records"])
        errors.append(len(mistakes(f"eval-bench-{c}")))

    def usd(s):
        return s * 1000 / 3600 * hourly

    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.axvspan(16, 40, color=ORANGE, alpha=0.08)
    ax.text(23, 0.39, "GPU saturated", color=ORANGE, fontsize=14, ha="center", weight="bold")
    ax.plot(RUNS, s_rec, color=BLUE, lw=3, marker="o", ms=11, zorder=3)
    for c, s, e in zip(RUNS, s_rec, errors, strict=False):
        ax.annotate(
            f"{s:.3f} s  ·  ${usd(s):.3f}/1k\n{e} field errors",
            (c, s),
            textcoords="offset points",
            xytext=(0, 18),
            ha="center",
            fontsize=13,
            color=TEXT,
        )
    drop1 = 1 - s_rec[1] / s_rec[0]
    drop2 = 1 - s_rec[2] / s_rec[1]
    ax.text(11.3, 0.30, f"-{drop1:.0%}", color=GREEN, fontsize=16, weight="bold", ha="center")
    ax.text(22.6, 0.235, f"-{drop2:.1%}", color=MUTED, fontsize=16, weight="bold", ha="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(RUNS)
    ax.set_xticklabels([str(c) for c in RUNS])
    ax.set_xlim(6.5, 40)
    ax.set_ylim(0.2, 0.43)
    ax.grid(axis="y")
    ax.set_xlabel("Concurrent requests to vLLM (MAX_CONCURRENT_REQUESTS)")
    ax.set_ylabel("Seconds per record")
    right = ax.secondary_yaxis("right", functions=(usd, lambda d: d * 3600 / 1000 / hourly))
    right.set_ylabel(f"\\$ per 1,000 records (at \\${hourly:.2f}/h)")
    right.tick_params(colors=MUTED)
    ax.set_title("Throughput stops improving past 16 in flight; accuracy is flat")
    save(fig, FINDINGS_IMG / "concurrency.png")


# ---------- field error waffle ----------


def plot_field_errors():
    counts = Counter(m["field"] for m in mistakes("eval-bench-32"))
    order = [f for f, _ in counts.most_common()]
    squares = [f for f in order for _ in range(counts[f])]
    cols = 13
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor(BG)
    for i, f in enumerate(squares):
        r, c = divmod(i, cols)
        ax.add_patch(
            FancyBboxPatch(
                (c + 0.08, r + 0.08),
                0.84,
                0.84,
                boxstyle="round,pad=0,rounding_size=0.12",
                fc=FIELD_COLOURS[f],
                ec="none",
            )
        )
    rows = (len(squares) + cols - 1) // cols
    ax.set_xlim(-0.2, cols + 6.5)
    ax.set_ylim(rows + 0.3, -1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    for j, f in enumerate(order):
        y = 0.6 + j * 1.25
        ax.add_patch(
            FancyBboxPatch(
                (cols + 0.6, y - 0.4),
                0.8,
                0.8,
                boxstyle="round,pad=0,rounding_size=0.12",
                fc=FIELD_COLOURS[f],
                ec="none",
            )
        )
        ax.text(cols + 1.7, y, f"{f}  {counts[f]}", va="center", fontsize=14, color=TEXT)
    zero = [f for f in FIELDS if counts[f] == 0]
    ax.text(
        cols + 0.6,
        0.6 + len(order) * 1.25,
        f"{' and '.join(zero)}  0",
        va="center",
        fontsize=14,
        color=MUTED,
    )
    names = counts["name"] + counts["relation name"]
    total = len(squares)
    ax.text(
        0,
        -0.6,
        f"{total} field errors in 4,829 records: names and relation names are "
        f"{names} ({names / total:.0%})",
        fontsize=15,
        weight="bold",
        color=TEXT,
    )
    save(fig, FINDINGS_IMG / "field_errors.png")


# ---------- scan resolution dumbbell ----------


def plot_scan_resolution():
    b = bench_numbers()
    width = {
        (r["pdf"], r["page"]): r["width"] for r in read_csv(OUT / "eval-bench-32" / "pages.csv")
    }
    counts = Counter((width[(m["pdf"], m["page"])], m["field"]) for m in mistakes("eval-bench-32"))
    n = {HI_RES: b["hi_records"], LO_RES: b["lo_records"]}

    def rate(w, f):
        return 100 * counts[(w, f)] / n[w]

    fields = sorted(FIELDS, key=lambda f: rate(LO_RES, f) + rate(HI_RES, f))
    fig, ax = plt.subplots(figsize=(12, 6.8))
    for y, f in enumerate(fields):
        hi, lo = rate(HI_RES, f), rate(LO_RES, f)
        ax.plot([hi, lo], [y, y], color=MUTED, lw=3, zorder=1)
        ax.scatter([hi], [y], s=150, c=BLUE, zorder=3, edgecolors=BG)
        ax.scatter([lo], [y], s=150, c=ORANGE, zorder=3, edgecolors=BG)
        ax.text(
            max(hi, lo) + 0.08,
            y,
            f"{counts[(HI_RES, f)]} vs {counts[(LO_RES, f)]}",
            va="center",
            fontsize=12.5,
            color=MUTED,
        )
    # all fields together, as % of fields
    tot = {w: sum(counts[(w, f)] for f in FIELDS) for w in n}
    y = len(fields) + 0.6
    all_hi, all_lo = 100 * tot[HI_RES] / (8 * n[HI_RES]), 100 * tot[LO_RES] / (8 * n[LO_RES])
    ax.plot([all_hi, all_lo], [y, y], color=TEXT, lw=3)
    ax.scatter(
        [all_hi],
        [y],
        s=190,
        c=BLUE,
        zorder=3,
        edgecolors=BG,
        label=f"240 DPI ({n[HI_RES]:,} records)",
    )
    ax.scatter(
        [all_lo],
        [y],
        s=190,
        c=ORANGE,
        zorder=3,
        edgecolors=BG,
        label=f"115 DPI ({n[LO_RES]:,} records)",
    )
    ax.text(
        all_lo + 0.08,
        y,
        f"{all_hi:.2f}% vs {all_lo:.2f}% ({all_lo / all_hi:.1f}x)",
        va="center",
        fontsize=13,
        color=TEXT,
        weight="bold",
    )
    ax.set_yticks(list(range(len(fields))) + [y])
    ax.set_yticklabels(fields + ["all fields"])
    ax.get_yticklabels()[-1].set_weight("bold")
    ax.set_xlim(-0.1, 3.9)
    ax.grid(axis="x")
    ax.set_xlabel("Error rate, % of records (all fields: % of fields)")
    ax.set_title("Low-resolution scans mostly add name and house-number errors")
    ax.legend(loc="lower right")
    save(fig, FINDINGS_IMG / "scan_resolution.png")


# ---------- CLIP margin swarm ----------


def swarm(xs, x_px, d_px):
    # Simple beeswarm: stack points vertically so none overlap in pixel space
    placed, offsets = [], []
    for x in xs:
        px = x * x_px
        k = 0
        while True:
            off = (k + 1) // 2 * d_px * (1 if k % 2 else -1)
            if all((px - qx) ** 2 + (off - qy) ** 2 >= d_px**2 for qx, qy in placed):
                break
            k += 1
        placed.append((px, off))
        offsets.append(off)
    return offsets


def plot_clip_margin():
    rows = pages_with_truth()
    groups = []
    for r in rows:
        clip_voter = float(r["p_voter"]) > float(r["p_non_voter"])
        if r["outcome"] != "qwen_fallback":
            g = 2
        elif clip_voter != (r["geometry_label"] == "voter"):
            g = 1  # CLIP and geometry disagree
        else:
            g = 0  # they agree, margin below the threshold
        groups.append(g)
    names = [
        "Fallback: agrees with\ngeometry, margin < 0.15",
        "Fallback: CLIP says voter,\ngeometry disagrees",
        "CLIP trusted",
    ]
    centres = [0.0, 1.6, 4.4]

    fig, ax = plt.subplots(figsize=(13, 7.4))
    x_px, d_px = 1500, 9  # approximate axis width and marker spacing in pixels
    scale = 0.042  # data units per pixel on the y axis
    for g in range(3):
        sel = sorted(
            [r for r, gg in zip(rows, groups, strict=False) if gg == g],
            key=lambda r: float(r["margin"]),
        )
        xs = [float(r["margin"]) for r in sel]
        offs = swarm(xs, x_px, d_px)
        colours = [GREEN if r["true_label"] == "voter" else ORANGE for r in sel]
        ax.scatter(
            xs,
            [centres[g] + o * scale / d_px * 0.38 for o in offs],
            s=48,
            c=colours,
            edgecolors=BG,
            linewidths=0.6,
            zorder=3,
        )
        ax.text(
            -0.02,
            centres[g],
            f"{names[g]}  ({len(sel)})",
            ha="right",
            va="center",
            fontsize=13,
            color=TEXT,
        )
    ax.axvline(CLIP_THRESHOLD, color=TEXT, ls="--", lw=1.4)
    ax.text(CLIP_THRESHOLD + 0.01, 5.95, "threshold 0.15", fontsize=12.5, color=TEXT)
    ax.scatter([], [], s=60, c=GREEN, label="true voter page")
    ax.scatter([], [], s=60, c=ORANGE, label="true non-voter page")
    ax.legend(loc="upper left", bbox_to_anchor=(0.32, 1.0))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.8, 6.3)
    ax.grid(axis="x")
    ax.set_xlabel("CLIP margin |p(voter) - p(non-voter)|")
    ax.set_title(
        "Every trusted page has margin >= 0.33; CLIP is confidently wrong on 6 non-voter pages",
        loc="right",
    )
    save(fig, FINDINGS_IMG / "clip_margin.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="draw one chart, e.g. hero")
    args = parser.parse_args()
    theme()
    charts = {
        "hero": plot_hero,
        "kpis": plot_kpis,
        "pipeline_flow": plot_pipeline_flow,
        "page_classification": plot_page_classification,
        "concurrency": plot_concurrency,
        "field_errors": plot_field_errors,
        "scan_resolution": plot_scan_resolution,
        "clip_margin": plot_clip_margin,
    }
    for name, fn in charts.items():
        if args.only in (None, name):
            fn()


if __name__ == "__main__":
    main()
