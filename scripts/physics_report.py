"""How much do REAL MuJoCo contacts overlap in the dataset? (to check the grader thresholds)

Scans every step of every clip (or every --stride-th step) and reports, for hand-block,
block-block and block-table/wall contacts, the 99th / 99.9th / 99.99th percentile and maximum
overlap depth, plus per-step block displacement and the number of unstable clips.

  python scripts/physics_report.py [--data data/hand_blocks_v2] [--stride 1]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_g = None


def _init():
    global _g
    from handwm.hand_scene import build_hand_xml
    from handwm.wm_metrics import Grader
    _g = Grader(build_hand_xml())


def _scan(args):
    import mujoco
    path, stride = args
    g, m, d = _g, _g.m, _g.d
    with np.load(path) as z:
        qpos, bp, nb, unst = z["qpos"], z["block_pos"], z["n_blocks"], z["unstable"]
    depth = {"hand_block": [], "block_block": [], "block_table_wall": []}
    steps = []
    for i in range(len(nb)):
        n = int(nb[i])
        steps.append(np.linalg.norm(np.diff(bp[i, :, :n], axis=0), axis=-1).ravel())
        for q in qpos[i][::stride]:
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_collision(m, d)
            for c in d.contact[:d.ncon]:
                g1, g2 = int(c.geom1), int(c.geom2)
                b1, b2 = g.geom_to_block.get(g1), g.geom_to_block.get(g2)
                if b1 is None and b2 is None:
                    continue
                if b1 is not None and b2 is not None:
                    depth["block_block"].append(-c.dist)
                elif (g1 in g.hand_geom) or (g2 in g.hand_geom):
                    depth["hand_block"].append(-c.dist)
                else:
                    depth["block_table_wall"].append(-c.dist)
    return ({k: np.array(v, np.float32) for k, v in depth.items()},
            np.concatenate(steps).astype(np.float32), int(unst.sum()), len(nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "hand_blocks_v2"))
    ap.add_argument("--stride", type=int, default=1, help="check every n-th step")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    args = ap.parse_args()
    from handwm.dataset import list_shards
    shards = list_shards(args.data)
    import multiprocessing as mp
    depth = {"hand_block": [], "block_block": [], "block_table_wall": []}
    steps, unstable, clips = [], 0, 0
    with mp.Pool(args.workers, initializer=_init) as pool:
        for k, (dd, st, u, n) in enumerate(pool.imap_unordered(_scan, [(str(p), args.stride) for p in shards]), 1):
            for key in depth:
                depth[key].append(dd[key])
            steps.append(st)
            unstable += u
            clips += n
            if k % 50 == 0:
                print(f"  {k}/{len(shards)} shards", flush=True)
    pc = [99, 99.9, 99.99, 100]
    report = {"clips": clips, "unstable_clips": unstable, "stride": args.stride}
    print(f"\n{clips} clips, unstable: {unstable}, every {args.stride}. step checked")
    print(f"{'contact overlap (mm)':<24}" + "".join(f"{'p' + str(p):>10}" for p in pc[:-1]) + f"{'max':>10}{'count':>12}")
    for key, v in depth.items():
        a = np.concatenate(v) * 1000 if v else np.zeros(1)
        vals = np.percentile(a, pc)
        report[key + "_mm"] = dict(zip(["p99", "p99.9", "p99.99", "max"], vals.round(3).tolist()))
        print(f"{key:<24}" + "".join(f"{x:10.2f}" for x in vals) + f"{len(a):12d}")
    s = np.concatenate(steps) * 100
    vals = np.percentile(s, pc)
    report["block_step_cm"] = dict(zip(["p99", "p99.9", "p99.99", "max"], vals.round(3).tolist()))
    print(f"{'block step (cm)':<24}" + "".join(f"{x:10.2f}" for x in vals))
    out = Path(args.data) / "physics_report.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
