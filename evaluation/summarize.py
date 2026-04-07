"""Summarise sweep results from outputs/results.csv.

Prints:
  1. Top-5 configurations by val AUC.
  2. Marginal effect of each design axis (pooling, remove_padding, background,
     slice_width): for each axis, groups all runs by that axis's value and
     reports mean ± std val AUC, averaged over all other axes.

Usage:
    python evaluation/summarize.py
    python evaluation/summarize.py --results outputs/results.csv
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Axes whose marginal effect we report.  stride is intentionally excluded
# because it is coupled to slice_width by construction.
# ---------------------------------------------------------------------------
MARGINAL_AXES = ["pooling", "remove_padding", "background", "slice_width"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No data in {path}")
    for row in rows:
        row["best_val_auc"] = float(row["best_val_auc"])
    return rows


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def std(values: list[float]) -> float:
    return statistics.pstdev(values)   # population std (no Bessel correction)


def hline(char: str = "─", width: int = 72) -> str:
    return char * width


def fmt_auc(v: float) -> str:
    return f"{v:.4f}"


def fmt_mean_std(m: float, s: float) -> str:
    return f"{m:.4f} ± {s:.4f}"


# ---------------------------------------------------------------------------
# Top-N table
# ---------------------------------------------------------------------------

def top_n(rows: list[dict], n: int = 5) -> None:
    sorted_rows = sorted(rows, key=lambda r: r["best_val_auc"], reverse=True)
    top = sorted_rows[:n]

    print(hline("═"))
    print(f"  TOP {n} CONFIGURATIONS BY VAL AUC  ({len(rows)} total runs)")
    print(hline("═"))

    col_widths = {
        "rank":      4,
        "auc":       8,
        "pooling":   11,
        "padding":   8,
        "bg":        6,
        "sw":        3,
        "st":        3,
    }

    header = (
        f"  {'#':>{col_widths['rank']}}  "
        f"{'AUC':>{col_widths['auc']}}  "
        f"{'pooling':<{col_widths['pooling']}}  "
        f"{'pad':>{col_widths['padding']}}  "
        f"{'bg':<{col_widths['bg']}}  "
        f"{'sw':>{col_widths['sw']}}  "
        f"{'st':>{col_widths['st']}}"
    )
    print(header)
    print(hline())

    for rank, row in enumerate(top, 1):
        pad_str = "yes" if str(row["remove_padding"]).lower() in ("true", "1") else "no"
        print(
            f"  {rank:>{col_widths['rank']}}  "
            f"{fmt_auc(row['best_val_auc']):>{col_widths['auc']}}  "
            f"{row['pooling']:<{col_widths['pooling']}}  "
            f"{pad_str:>{col_widths['padding']}}  "
            f"{row['background']:<{col_widths['bg']}}  "
            f"{row['slice_width']:>{col_widths['sw']}}  "
            f"{row['stride']:>{col_widths['st']}}"
        )

    print(hline())


# ---------------------------------------------------------------------------
# Marginal effects
# ---------------------------------------------------------------------------

def marginal_effects(rows: list[dict]) -> None:
    print()
    print(hline("═"))
    print("  MARGINAL EFFECT OF EACH DESIGN AXIS  (mean ± std val AUC)")
    print(hline("═"))

    for axis in MARGINAL_AXES:
        # Group AUC values by this axis's value
        groups: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            key = row[axis]
            groups[key].append(row["best_val_auc"])

        print(f"\n  {axis}")
        print(f"  {'value':<20}  {'n':>4}  {'mean ± std':>16}  {'min':>8}  {'max':>8}")
        print("  " + hline("─", 64))

        # Sort values sensibly: numeric if possible, else lexicographic
        def sort_key(k):
            try:
                return (0, float(k))
            except (ValueError, TypeError):
                return (1, str(k).lower())

        for value in sorted(groups.keys(), key=sort_key):
            aucs = groups[value]
            print(
                f"  {str(value):<20}  "
                f"{len(aucs):>4}  "
                f"{fmt_mean_std(mean(aucs), std(aucs)):>16}  "
                f"{fmt_auc(min(aucs)):>8}  "
                f"{fmt_auc(max(aucs)):>8}"
            )

    print()
    print(hline("═"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="outputs/results.csv",
        help="Path to the results CSV produced by strip_design_sweep.py",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Number of top configurations to show (default: 5)",
    )
    args = parser.parse_args()

    path = Path(args.results)
    if not path.exists():
        raise SystemExit(f"Results file not found: {path}")

    rows = load(path)
    top_n(rows, n=args.top)
    marginal_effects(rows)


if __name__ == "__main__":
    main()
