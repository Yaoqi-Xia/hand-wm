"""Look at the generated dataset: plays random clips (the 64x64 training frames, enlarged)
next to a full-resolution MuJoCo view of the same moment. Prints dataset statistics.

  python scripts/view_dataset.py                 # default dataset folder
  python scripts/view_dataset.py --stats-only
Keys: SPACE next clip, ESC quit.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mujoco
import numpy as np

from handwm.dataset import clip, list_shards, load_shard
from handwm.hand_scene import build_hand_xml
from handwm.scripted_hand import LABELS

ROOT = Path(__file__).resolve().parent.parent


def stats(shards):
    n_clips = n_bad = 0
    blocks, shapes = np.zeros(9, int), np.zeros(len(LABELS), int)
    size = sum(p.stat().st_size for p in shards)
    for p in shards[:20]:  # sample for speed
        s = load_shard(p, ["n_blocks", "shape_id", "unstable", "action"])
        n_clips += len(s["n_blocks"])
        n_bad += int(s["unstable"].sum())
        blocks += np.bincount(s["n_blocks"], minlength=9)
        shapes += np.bincount(s["shape_id"].ravel(), minlength=len(LABELS))
        T = s["action"].shape[1]
    per = n_clips / min(len(shards), 20)
    print(f"shards: {len(shards)}  clips: ~{int(per * len(shards))}  steps per clip: {T} "
          f"({T * 0.05:.1f} s)  size on disk: {size / 2**30:.2f} GB")
    print(f"(first {min(len(shards), 20)} shards) unstable clips: {n_bad}/{n_clips}; "
          f"blocks per clip 1..8: {blocks[1:].tolist()}; "
          f"hand shape share: " + ", ".join(f"{n} {c / shapes.sum():.0%}" for n, c in zip(LABELS, shapes)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "hand_blocks_v2"))
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()
    shards = list_shards(args.data)
    if not shards:
        print(f"no shards in {args.data} yet - run the generator first")
        return
    stats(shards)
    if args.stats_only:
        return

    model = mujoco.MjModel.from_xml_string(build_hand_xml())
    data = mujoco.MjData(model)
    ren = mujoco.Renderer(model, 384, 512)
    rng = np.random.default_rng()
    win = "dataset clips (left: 64x64 training frame, right: MuJoCo)"
    while True:
        p = shards[rng.integers(len(shards))]
        sh = load_shard(p)
        i = int(rng.integers(len(sh["seed"])))
        c = clip(sh, i)
        for k in range(len(c["frames"])):
            t0 = time.perf_counter()
            small = cv2.resize(c["frames"][k], (384, 384), interpolation=cv2.INTER_NEAREST)
            data.qpos[:] = c["qpos"][k]
            mujoco.mj_forward(model, data)
            ren.update_scene(data, camera="persp")
            img = cv2.cvtColor(np.hstack([small, ren.render()]), cv2.COLOR_RGB2BGR)
            shape = LABELS[c["shape_id"][min(k, len(c["shape_id"]) - 1)]]
            cv2.putText(img, f"{p.name} clip {i}  step {k}  blocks {c['n_blocks']}  hand {shape}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(win, img)
            key = cv2.waitKey(max(1, int(50 - (time.perf_counter() - t0) * 1000)))
            if key == 27 or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                return
            if key == 32:
                break


if __name__ == "__main__":
    main()
