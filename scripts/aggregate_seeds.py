"""Mean and spread over random seeds.

Combines evaluation results.json files (seed 0 run + extra-seed run) into one table per
method: mean, standard deviation, min and max over seeds, plus curve plots with the spread
shaded. Methods are matched by name after removing a "_s<seed>" suffix, e.g. main, main_s1,
main_s2 -> "main"; main_onestep, main_s1_onestep, ... -> "main_onestep".

  python scripts/aggregate_seeds.py runs/eval_A/results.json runs/eval_B/results.json
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
METHODS = ["main", "main_onestep", "mlp", "action"]
ROWS = [("pos_err_moved_cm", [10, 50, 200], "cm"), ("pen_hand", [50, 200], "%"),
        ("pen_block", [50, 200], "%")]


def base_name(n):
    return re.sub(r"_s\d+", "", n)


def main():
    files = [Path(p) for p in sys.argv[1:]]
    curves = {}   # split -> method -> metric -> list of curves (one per seed)
    sources = []
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        sources.append(str(f))
        for split, v in r["splits"].items():
            for name, ms in v["curves"].items():
                b = base_name(name)
                if b not in METHODS:
                    continue
                for m, c in ms.items():
                    curves.setdefault(split, {}).setdefault(b, {}).setdefault(m, []).append(np.array(c, float))
    out = ROOT / "runs" / f"seeds_summary_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    summary, lines = {"sources": sources, "splits": {}}, []
    for split, meths in curves.items():
        lines += [f"\n## {split}\n", "| method | seeds | " + " | ".join(
            f"{m} @{k}" for m, ks, _ in ROWS for k in ks) + " |",
                  "|---|---|" + "---|" * sum(len(ks) for _, ks, _ in ROWS)]
        summary["splits"][split] = {}
        for b in METHODS:
            if b not in meths:
                continue
            cells, rec = [], {}
            n_seeds = len(meths[b]["pos_err_moved_cm"])
            for m, ks, unit in ROWS:
                arr = np.stack(meths[b][m])               # (seeds, steps)
                for k in ks:
                    v = arr[:, k - 1] * (100 if unit == "%" else 1)
                    rec[f"{m}@{k}"] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                           min=float(v.min()), max=float(v.max()), values=v.round(4).tolist())
                    cells.append(f"{v.mean():.2f} ± {rec[f'{m}@{k}']['std']:.2f}{'%' if unit == '%' else ''}")
            summary["splits"][split][b] = {"n_seeds": n_seeds, **rec}
            lines.append(f"| {b} | {n_seeds} | " + " | ".join(cells) + " |")
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "summary.md").write_text("mean ± std over seeds (std with n-1)\n" + "\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    plot(curves, out)
    print(f"\n-> {out}")


def plot(curves, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for split, meths in curves.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for b in METHODS:
            if b not in meths:
                continue
            for ax, m, scale in zip(axes, ["pos_err_moved_cm", "pen_hand", "pen_block"], [1, 100, 100]):
                arr = np.stack(meths[b][m]) * scale
                k = np.arange(1, arr.shape[1] + 1)
                mu, sd = arr.mean(0), arr.std(0, ddof=1) if len(arr) > 1 else np.zeros(arr.shape[1])
                ax.plot(k, mu, label=f"{b} (n={len(arr)})")
                ax.fill_between(k, mu - sd, mu + sd, alpha=0.2)
        for ax, t in zip(axes, ["position error, blocks that moved (cm)", "hand-block penetration (%)",
                                "block-block overlap (%)"]):
            ax.set_title(t)
            ax.set_xscale("log")
            ax.set_xlabel("rollout step (0.05 s each)")
            ax.grid(alpha=0.3)
        axes[0].legend()
        fig.suptitle(f"{split}: mean ± 1 std over seeds")
        fig.tight_layout()
        fig.savefig(out / f"seed_curves_{split}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
