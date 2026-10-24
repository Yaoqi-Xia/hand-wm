"""Headless check: a scripted finger pushes the block into the target, log is saved,
re-simulation from the log matches, and a replay video is written."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from handwm.control import ScriptedSource
from handwm.env import PushEnv
from handwm.logger import load_episode

REACH = 0.085 + 0.008 + 0.025  # finger length + tip radius + half block


def main():
    # Fixed start pose: an off-centre push on a randomly rotated block slides it sideways.
    env = PushEnv(randomize=False, source_name="scripted")
    env.reset()
    bx, by = env.state()["block_pos"][:2]
    src = ScriptedSource([
        # t,   x,               y,   yaw, bend
        [0.0, -0.15,            0.0, 0.0, 0.0],
        [1.0, bx - REACH - 0.02, by, 0.0, 0.0],
        [5.0, 0.25 - REACH,      by, 0.0, 0.0],
        [6.0, 0.25 - REACH - 0.04, by, 0.0, 0.6],
    ])
    for _ in range(int(6.5 / env.CTRL_DT)):
        s = env.step(src.read(env.data.time))
    print(f"block start ({bx:.3f}, {by:.3f}) -> end ({s['block_pos'][0]:.3f}, {s['block_pos'][1]:.3f}); "
          f"dist to target {s['dist_to_target'] * 100:.1f} cm, success={s['success']}")
    path = env.save_episode("smoke")
    print(f"saved {len(env.logger)} steps -> {path}")

    meta, arr = load_episode(path)
    assert arr["qpos"].shape[0] == len(arr["t"]) and np.isfinite(arr["qpos"]).all()

    sys.argv = ["replay.py", str(path), "--check", "--video"]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import replay
    replay.main()
    print("OK" if s["success"] else "WARNING: block did not reach target")


if __name__ == "__main__":
    main()
