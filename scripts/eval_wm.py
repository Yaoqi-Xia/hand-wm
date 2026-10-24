"""Grade predictors against MuJoCo ground truth on held-out clips.

  python scripts/eval_wm.py                                   # baselines on the test sets
  python scripts/eval_wm.py --predictors static,const_velocity,model:runs/xxx/model.pt

For each test clip and start step, every predictor rolls out --horizon steps from the true
state using the true hand sequence; the grader scores every step. Results (per-step curves,
a summary table at 1/10/50/200 steps and plots) go to runs/eval_<time>/.
Test sets: "test" (same block counts as training) and "test_ood" (more blocks, e.g. 6-8).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from handwm.dataset import list_shards
from handwm.predictors import HandStep, TruthPredictor, make_predictor, rollout
from handwm.wm_data import excluded_clips, load_states, split_indices
from handwm.wm_metrics import METRICS, Grader

ROOT = Path(__file__).resolve().parent.parent
REPORT_STEPS = [1, 10, 50, 200]


def load_qpos(data_dir, clip_indices, per_shard):
    """Full MuJoCo states of the requested clips (needed for the true hand geometry)."""
    by_shard = {}
    for ci in clip_indices:
        by_shard.setdefault(int(ci) // per_shard, []).append(int(ci))
    out = {}
    for s, cis in by_shard.items():
        with np.load(Path(data_dir) / f"shard_{s:05d}.npz") as z:
            q = z["qpos"]
        for ci in cis:
            out[ci] = q[ci % per_shard]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "hand_blocks_v2"))
    ap.add_argument("--predictors", default="static,const_velocity,auto",
                    help="comma list: static, const_velocity, model:<path>, latest:<tag>[:onestep], "
                         "auto = newest trained model (skipped if none)")
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--starts", default="0,56", help="rollout start steps within each clip")
    ap.add_argument("--max-clips", type=int, default=150, help="per test set")
    ap.add_argument("--max-train-blocks", type=int, default=5)
    ap.add_argument("--pen-mm", type=float, default=5.0, help="penetration threshold (mm)")
    ap.add_argument("--teleport-cm", type=float, default=5.0, help="teleport threshold (cm/step)")
    ap.add_argument("--keep-flagged-clips", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if sys.platform == "win32":  # keep the PC awake while this runs (released on exit)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    info = json.loads((Path(args.data) / "dataset_info.json").read_text(encoding="utf-8"))
    print("loading states ...")
    d = load_states(args.data, verbose=False)
    excl = [] if args.keep_flagged_clips else excluded_clips(args.data)
    splits = split_indices(d, args.max_train_blocks, exclude=excl)
    print(f"excluded {len(excl)} flagged clips: {excl}")
    from handwm.hand_scene import build_hand_xml
    grader = Grader(build_hand_xml(), pen_mm=args.pen_mm, teleport_cm=args.teleport_cm)
    from handwm.predictors import latest_model
    specs = [p for p in args.predictors.split(",") if not (p == "auto" and latest_model() is None)]
    preds = [make_predictor(p) for p in specs]
    truth = TruthPredictor()
    starts = [int(s) for s in args.starts.split(",")]
    H = args.horizon
    T = d["blocks"].shape[1] - 1
    starts = [s for s in starts if s + H <= T]
    rng = np.random.default_rng(0)

    results = {"horizon": H, "starts": starts, "data": str(args.data), "pen_mm": args.pen_mm,
               "teleport_cm": args.teleport_cm, "excluded_clips": excl, "splits": {}}
    for split in ("test", "test_ood"):
        idx = splits[split]
        if len(idx) == 0:
            print(f"[{split}] no clips")
            continue
        idx = np.sort(rng.permutation(idx)[:args.max_clips])
        qpos = load_qpos(args.data, d["clip_index"][idx], info["per_shard"])
        acc = {p.name: {m: [] for m in METRICS} for p in preds + [truth]}
        t0 = time.time()
        for j, c in enumerate(idx):
            n = int(d["n_blocks"][c])
            mask = d["mask"][c]
            q = qpos[int(d["clip_index"][c])]
            for s in starts:
                true = d["blocks"][c, s:s + H + 1].astype(np.float64)
                kp, hp, act = d["keypoints"][c], d["hand_pose"][c], d["action"][c]
                hands = [HandStep(kp[t], kp[t + 1], hp[t], hp[t + 1], act[t]) for t in range(s, s + H)]
                truth.set_sequence(true)
                for p in preds + [truth]:
                    traj = rollout(p, true[0].copy(), mask, hands)
                    g = grader.grade(traj, true, q[s:s + H + 1], n)
                    for m in METRICS:
                        acc[p.name][m].append(g[m])
            if (j + 1) % 25 == 0:
                print(f"  [{split}] {j + 1}/{len(idx)} clips  ({time.time() - t0:.0f} s)", flush=True)
        curves = {name: {m: np.nanmean(np.stack(v), 0).tolist() for m, v in ms.items()}
                  for name, ms in acc.items()}
        results["splits"][split] = {"n_clips": int(len(idx)),
                                    "block_counts": sorted(set(d["n_blocks"][idx].astype(int).tolist())),
                                    "curves": curves}

    out = Path(args.out) if args.out else ROOT / "runs" / f"eval_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results), encoding="utf-8")
    print_table(results)
    plot(results, out)
    print(f"\nresults -> {out}")


def print_table(results):
    for split, r in results["splits"].items():
        print(f"\n=== {split}: {r['n_clips']} clips, blocks {r['block_counts']} ===")
        steps = [k for k in REPORT_STEPS if k <= results["horizon"]]
        head = f"{'predictor':<24}{'metric':<18}" + "".join(f"{'step ' + str(k):>10}" for k in steps)
        print(head)
        for name, ms in r["curves"].items():
            for m in ["pos_err_cm", "pos_err_moved_cm", "pen_hand", "pen_block", "teleport"]:
                vals = [ms[m][k - 1] for k in steps]
                fmt = (lambda v: f"{v:10.2f}") if m.startswith("pos") else (lambda v: f"{v * 100:9.2f}%")
                print(f"{name:<24}{m:<18}" + "".join(fmt(v) for v in vals))


def plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for split, r in results["splits"].items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for name, ms in r["curves"].items():
            k = np.arange(1, len(ms["pos_err_cm"]) + 1)
            style = "--" if name == "mujoco_truth" else "-"
            axes[0].plot(k, ms["pos_err_moved_cm"], style, label=name)
            axes[1].plot(k, np.array(ms["pen_hand"]) * 100, style, label=name)
            axes[2].plot(k, np.array(ms["any_violation"]) * 100, style, label=name)
        for ax, t in zip(axes, ["position error, blocks that moved (cm)",
                                "hand-block penetration (% of blocks)",
                                "any violation (% of blocks)"]):
            ax.set_title(t)
            ax.set_xlabel("rollout step (0.05 s each)")
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
        axes[0].legend()
        fig.suptitle(f"{split}: {r['n_clips']} clips, blocks {r['block_counts']}")
        fig.tight_layout()
        fig.savefig(out / f"curves_{split}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
