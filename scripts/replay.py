"""Replay a logged episode.

  python scripts/replay.py                 # latest episode in the 3D viewer
  python scripts/replay.py logs/xxx.npz    # a specific episode
  python scripts/replay.py --video         # render to an .mp4 next to the log (no window)
  python scripts/replay.py --check         # re-simulate from logged inputs, compare block trajectory
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mujoco
import numpy as np

from handwm.control import ControlInput
from handwm.env import LOG_DIR, PushEnv
from handwm.logger import latest_episode, load_episode


def set_state(model, data, arr, i):
    data.qpos[:] = arr["qpos"][i]
    data.qvel[:] = arr["qvel"][i]
    data.ctrl[:] = arr["ctrl"][i]
    data.time = arr["t"][i]
    mujoco.mj_forward(model, data)


def row_dt(meta):
    """Seconds between logged rows: every physics step (finger) or every control tick (hand)."""
    return meta["ctrl_dt"] if meta.get("env") == "hand" else meta["timestep"]


def replay_viewer(meta, arr, speed, loop):
    import mujoco.viewer
    model = mujoco.MjModel.from_xml_string(meta["xml"])
    data = mujoco.MjData(model)
    n, dt = len(arr["t"]), row_dt(meta)
    stride = max(1, int(round(0.02 / dt)))  # draw at ~50 fps
    with mujoco.viewer.launch_passive(model, data) as v:
        while v.is_running():
            for i in range(0, n, stride):
                if not v.is_running():
                    break
                t0 = time.perf_counter()
                set_state(model, data, arr, i)
                v.sync()
                spare = stride * dt / speed - (time.perf_counter() - t0)
                if spare > 0:
                    time.sleep(spare)
            if not loop:
                time.sleep(1.0)
                break


def replay_video(meta, arr, path, camera):
    model = mujoco.MjModel.from_xml_string(meta["xml"])
    data = mujoco.MjData(model)
    dt = row_dt(meta)
    stride = max(1, int(round(0.02 / dt)))
    out = path.with_suffix(f".{camera}.mp4")
    w, h = 640, 480
    renderer = mujoco.Renderer(model, h, w)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / (stride * dt), (w, h))
    for i in range(0, len(arr["t"]), stride):
        set_state(model, data, arr, i)
        renderer.update_scene(data, camera=camera)
        vw.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    vw.release()
    renderer.close()
    print(f"video -> {out}")


def replay_check_hand(meta, arr):
    from handwm.hand_env import HandEnv
    env = HandEnv(xml_text=meta["xml"], stale_after=meta["stale_after"], source_name="replay_check")
    env.reset(n_blocks=meta["n_blocks"], init_qpos=np.array(meta["init_qpos"]),
              init_qvel=np.array(meta["init_qvel"]))
    sim = []
    for i in range(len(arr["t"])):
        ci = ControlInput(arr["in_timestamp"][i], arr["in_x"][i], arr["in_y"][i], arr["in_yaw"][i],
                          arr["in_bends"][i][1], data_age=arr["in_data_age"][i],
                          valid=bool(arr["in_valid"][i]), bends=tuple(arr["in_bends"][i]),
                          roll=float(arr["in_roll"][i]) if "in_roll" in arr else 0.0)
        sim.append(env.step(ci, log=False)["block_pos"])
    err = np.linalg.norm(np.stack(sim) - arr["block_pos"], axis=-1).max(axis=-1)
    print(f"re-simulated {len(sim)} steps; block position error max {err.max() * 1000:.4f} mm, "
          f"final {err[-1] * 1000:.4f} mm")
    return err.max()


def replay_check(meta, arr):
    """Feed the logged ControlInputs back through a fresh env; physics should match."""
    if meta.get("env") == "hand":
        return replay_check_hand(meta, arr)
    env = PushEnv(xml_text=meta["xml"], randomize=False, stale_after=meta["stale_after"],
                  source_name="replay_check")  # nothing is saved
    env.reset(init_qpos=np.array(meta["init_qpos"]))
    ticks = arr["tick"].astype(int)
    firsts = np.flatnonzero(np.r_[True, np.diff(ticks) != 0])
    sim_block = []
    for i in firsts:
        ci = ControlInput(arr["in_timestamp"][i], arr["in_x"][i], arr["in_y"][i],
                          arr["in_yaw"][i], arr["in_bend"][i],
                          data_age=arr["in_data_age"][i], valid=bool(arr["in_valid"][i]))
        env.step(ci)
    sim_block = np.stack(env.logger.rows["block_pos"])
    n = min(len(sim_block), len(arr["block_pos"]))
    err = np.linalg.norm(sim_block[:n] - arr["block_pos"][:n], axis=1)
    print(f"re-simulated {n} steps; block position error max {err.max() * 1000:.4f} mm, "
          f"final {err[-1] * 1000:.4f} mm")
    return err.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", nargs="?", help="path to .npz (default: latest in logs/)")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--camera", default="persp", choices=["persp", "top"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    path = Path(args.episode) if args.episode else latest_episode(LOG_DIR)
    meta, arr = load_episode(path)
    s = arr["block_pos"].reshape(len(arr["t"]), -1, 3)
    moved = np.linalg.norm(s[-1, :, :2] - s[0, :, :2], axis=-1).max() if s.shape[1] else 0.0
    print(f"{path.name}: {len(arr['t'])} steps, {arr['t'][-1]:.2f} s, source={meta['source']}, "
          f"block moved up to {moved * 100:.1f} cm")
    if args.check:
        replay_check(meta, arr)
    if args.video:
        replay_video(meta, arr, path, args.camera)
    if not (args.check or args.video):
        replay_viewer(meta, arr, args.speed, args.loop)


if __name__ == "__main__":
    main()
