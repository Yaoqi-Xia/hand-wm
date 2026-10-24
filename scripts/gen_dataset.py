"""Unattended dataset generation: scripted random hand pushing 1-8 blocks.

Writes shards of clips to <out>/shard_00000.npz ... Already finished shards are skipped,
so the same command resumes after a crash, a reboot or closing the window.

Dataset format version 2 (adds palm roll + hand keypoints). dataset_info.json in the output
folder records the format; the script refuses to resume into a folder of another format.

Per clip (T steps at 20 Hz, frame k = state at tick k, action k = command between k and k+1):
  frames      uint8  (T+1, 64, 64, 3)  top-down RGB
  block_pos   f32    (T+1, 8, 3)       unused slots are zeros (see n_blocks)
  block_quat  f32    (T+1, 8, 4)       orientation quaternion (w, x, y, z)
  block_vel   f32    (T+1, 8, 6)       linear (m/s) + angular (rad/s) velocity, world frame
  hand_pose   f32    (T+1, 5)          actual palm centre x, y, z, yaw, roll
  keypoints   f32    (T+1, 21, 3)      virtual hand's 21 MediaPipe-style keypoints (world, m)
  action      f64    (T, 9)            commanded x, y, yaw, roll, bends[thumb..little]
  shape_id    int8   (T,)              0 open, 1 fist, 2 poke, 3 pinch, 4 free fingers
  n_blocks    int8, seed int64, init_qpos / init_qvel f64 (exact MuJoCo replay)
  qpos        f32    (T+1, nq)         full state for viewing
  unstable    bool                     simulator warning or runaway block

  python scripts/gen_dataset.py --clips 30000
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FORMAT_VERSION = 2


def make_clip(env, renderer, seed, T):
    from handwm.scripted_hand import RandomHandPolicy
    import mujoco

    env.rng = np.random.default_rng(seed)
    env.reset()
    pol = RandomHandPolicy(env, np.random.default_rng(seed + 10_000_019))
    nq = env.model.nq
    frames = np.empty((T + 1, 64, 64, 3), np.uint8)
    bpos = np.empty((T + 1, 8, 3), np.float32)
    bquat = np.empty((T + 1, 8, 4), np.float32)
    bvel = np.empty((T + 1, 8, 6), np.float32)
    hand = np.empty((T + 1, 5), np.float32)
    kps = np.empty((T + 1, 21, 3), np.float32)
    qpos = np.empty((T + 1, nq), np.float32)
    action = np.empty((T, env.action.size), np.float64)  # full precision: exact MuJoCo replay
    shape = np.empty((T,), np.int8)
    init_qpos, init_qvel = env.data.qpos.copy(), env.data.qvel.copy()
    warn0 = int(env.data.warning.number.sum())
    unstable = False

    def record(k):
        renderer.update_scene(env.data, camera="top")
        frames[k] = renderer.render()
        d = env.data
        bpos[k] = d.qpos[env._bq[:, None] + np.arange(3)]
        bquat[k] = d.qpos[env._bq[:, None] + 3 + np.arange(4)]
        bvel[k] = env.block_velocities()
        bpos[k, env.n_blocks:] = 0
        bquat[k, env.n_blocks:] = 0
        bvel[k, env.n_blocks:] = 0
        hand[k] = d.qpos[env._hq]
        kps[k] = env.keypoints()
        qpos[k] = d.qpos

    record(0)
    for k in range(T):
        ci = pol.read(env.data.time)
        s = env.step(ci, log=False)
        action[k] = env.action
        shape[k] = pol.shape_id
        record(k + 1)
        if env.n_blocks and (np.abs(s["block_vel"]).max() > 3.0 or s["block_pos"][:, 2].max() > 0.15):
            unstable = True
    unstable |= int(env.data.warning.number.sum()) > warn0 or not np.isfinite(env.data.qpos).all()
    return dict(frames=frames, block_pos=bpos, block_quat=bquat, block_vel=bvel,
                hand_pose=hand, keypoints=kps,
                qpos=qpos,
                action=action, shape_id=shape, n_blocks=np.int8(env.n_blocks), seed=np.int64(seed),
                init_qpos=init_qpos, init_qvel=init_qvel, unstable=np.bool_(unstable))


_env = _renderer = None


def _worker_init():
    global _env, _renderer
    import mujoco
    from handwm.hand_env import HandEnv
    _env = HandEnv(source_name="scripted_random_hand")
    _renderer = mujoco.Renderer(_env.model, 64, 64)


def build_shard(args):
    shard, out, per_shard, T, base_seed = args
    path = Path(out) / f"shard_{shard:05d}.npz"
    if path.exists():
        return shard, 0, 0.0
    t0 = time.time()
    clips = [make_clip(_env, _renderer, base_seed + shard * per_shard + i, T)
             for i in range(per_shard)]
    stacked = {k: np.stack([c[k] for c in clips]) for k in clips[0]}
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **stacked)
    os.replace(tmp, path)  # atomic: a shard is either complete or absent
    return shard, int(stacked["unstable"].sum()), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "hand_blocks_v2"))
    ap.add_argument("--clips", type=int, default=30000)
    ap.add_argument("--steps", type=int, default=256, help="control steps per clip (20 Hz)")
    ap.add_argument("--per-shard", type=int, default=50)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from handwm.hand_env import ACTION_NAMES, HAND_POSE_NAMES
    from handwm.scripted_hand import LABELS
    info = {"format_version": FORMAT_VERSION, "steps": args.steps, "per_shard": args.per_shard,
            "seed": args.seed, "ctrl_dt": 0.05, "image": [64, 64, 3],
            "action_names": ACTION_NAMES, "hand_pose_names": HAND_POSE_NAMES,
            "shape_labels": LABELS}
    out = Path(args.out)
    info_path = out / "dataset_info.json"
    if info_path.exists():
        old = json.loads(info_path.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in info} != info:
            print(f"STOP: {out} holds a dataset of a different format/settings:\n  {old}\n"
                  "Use another --out folder (or delete that folder) instead of mixing them.")
            sys.exit(1)
    elif out.exists() and any(out.glob("shard_*.npz")):
        print(f"STOP: {out} already has shards but no dataset_info.json (older format?). "
              "Use another --out folder.")
        sys.exit(1)
    out.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info, indent=1), encoding="utf-8")
    for tmp in out.glob("*.tmp.npz"):  # half-written shard from an interrupted run
        tmp.unlink()
    n_shards = (args.clips + args.per_shard - 1) // args.per_shard
    todo = [s for s in range(n_shards) if not (out / f"shard_{s:05d}.npz").exists()]
    done = n_shards - len(todo)
    print(f"output: {out}\nshards: {n_shards} x {args.per_shard} clips x {args.steps} steps; "
          f"{done} already done, {len(todo)} to go; {args.workers} worker processes")
    if not todo:
        print("nothing to do - dataset complete")
        return

    if sys.platform == "win32":  # keep the PC awake while this window runs (released on exit)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    import multiprocessing as mp
    jobs = [(s, str(out), args.per_shard, args.steps, args.seed * 1_000_000_000) for s in todo]
    t0, n_new, n_bad, shard_mb = time.time(), 0, 0, None
    with mp.Pool(args.workers, initializer=_worker_init) as pool:
        for shard, bad, _ in pool.imap_unordered(build_shard, jobs):
            n_new += 1
            n_bad += bad
            if shard_mb is None:
                shard_mb = (out / f"shard_{shard:05d}.npz").stat().st_size / 2**20
                need_gb = shard_mb * len(todo) / 1024
                free_gb = shutil.disk_usage(out).free / 2**30
                print(f"one shard = {shard_mb:.0f} MB -> about {need_gb:.1f} GB more needed, "
                      f"{free_gb:.1f} GB free")
                if need_gb > free_gb * 0.9:
                    print("NOT ENOUGH DISK SPACE - stopping. Free space or lower --clips.")
                    pool.terminate()
                    sys.exit(1)
            el = time.time() - t0
            eta = el / n_new * (len(todo) - n_new)
            print(f"[{done + n_new}/{n_shards}] shard {shard:05d} done | unstable clips so far "
                  f"{n_bad} | elapsed {el / 60:.1f} min | remaining ~{eta / 60:.0f} min", flush=True)
    print("dataset complete")


if __name__ == "__main__":
    main()
