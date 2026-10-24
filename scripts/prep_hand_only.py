"""Compute block-free hand keypoints / poses for every clip (see handwm/hand_only.py).

Writes handonly_00000.npz next to each shard_00000.npz (keys: keypoints, hand_pose).
Resumable: finished files are skipped. About 10-15 min for 30000 clips with 16 processes.

  python scripts/prep_hand_only.py [--data data/hand_blocks_v2]
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_env = None


def _init():
    global _env
    from handwm.hand_env import HandEnv
    _env = HandEnv(source_name="hand_only_replay")


def _job(path):
    from handwm.hand_only import replay_hand_only
    path = Path(path)
    out = path.with_name(path.name.replace("shard_", "handonly_"))
    if out.exists():
        return path.name, 0.0
    t0 = time.time()
    with np.load(path) as z:
        init, act = z["init_qpos"], z["action"]
    kp, pose = zip(*(replay_hand_only(_env, init[i], act[i]) for i in range(len(init))))
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, keypoints=np.stack(kp), hand_pose=np.stack(pose))
    os.replace(tmp, out)
    return path.name, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "hand_blocks_v2"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    args = ap.parse_args()
    from handwm.dataset import list_shards
    shards = list_shards(args.data)
    todo = [p for p in shards if not p.with_name(p.name.replace("shard_", "handonly_")).exists()]
    print(f"{len(shards)} shards, {len(todo)} to process with {args.workers} processes")
    if not todo:
        print("hand-only inputs complete")
        return
    import multiprocessing as mp
    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init) as pool:
        for k, (name, _) in enumerate(pool.imap_unordered(_job, [str(p) for p in todo]), 1):
            el = time.time() - t0
            print(f"[{k}/{len(todo)}] {name} | elapsed {el / 60:.1f} min | "
                  f"remaining ~{el / k * (len(todo) - k) / 60:.0f} min", flush=True)
    print("hand-only inputs complete")


if __name__ == "__main__":
    main()
