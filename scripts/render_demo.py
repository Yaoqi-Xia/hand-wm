"""Render the README demo offscreen: learned world vs MuJoCo truth on a held-out test clip.

Solid blocks = world-model rollout (no MuJoCo physics), solid hand = the hand driven by the
clip's commands (block-free replay, the model's input). Translucent blocks and hand = MuJoCo
ground truth. Writes docs/demo.mp4 and docs/demo.gif.

  python scripts/render_demo.py [--clip <global clip index>] [--model models/main_seed0/model.pt]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import mujoco
import numpy as np

from eval_wm import load_qpos
from handwm.hand_env import HandEnv
from handwm.hand_scene import BLOCK_HALF, MAX_BLOCKS, build_hand_xml
from handwm.predictors import HandStep, bundled_model, latest_run_model, rollout
from handwm.wm_data import excluded_clips, load_states, split_indices
from handwm.wm_model import ModelPredictor

ROOT = Path(__file__).resolve().parent.parent


def hand_only_qpos(env, init_qpos, actions):
    q = np.array(init_qpos, float).copy()
    for i in range(MAX_BLOCKS):
        a = env._bq[i]
        q[a:a + 3] = [2 + 0.2 * i, 2, BLOCK_HALF]
        q[a + 3:a + 7] = [1, 0, 0, 0]
    env.reset(n_blocks=0, init_qpos=q, init_qvel=np.zeros(env.model.nv))
    out = [env.data.qpos.copy()]
    for a in actions:
        env.action = np.asarray(a, float)
        env._apply(env.action)
        env._physics_tick()
        out.append(env.data.qpos.copy())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "hand_blocks_v2"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--clip", type=int, default=None, help="global clip index (default: auto-pick)")
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    info = json.loads((Path(args.data) / "dataset_info.json").read_text(encoding="utf-8"))
    d = load_states(args.data, verbose=False)
    sp = split_indices(d, 5, exclude=excluded_clips(args.data))
    test = sp["test"]
    if args.clip is None:     # a 4-5 block test clip where the true blocks move the most
        cand = [c for c in test if d["n_blocks"][c] >= 4]
        moved = [np.linalg.norm(np.diff(d["blocks"][c, :args.steps + 1, :d["n_blocks"][c], :2], axis=0),
                                axis=-1).sum() for c in cand]
        c = cand[int(np.argmax(moved))]
    else:
        c = int(np.flatnonzero(d["clip_index"] == args.clip)[0])
    gci = int(d["clip_index"][c])
    n, mask, H = int(d["n_blocks"][c]), d["mask"][c], args.steps
    model_path = args.model or latest_run_model("main") or bundled_model()
    pred = ModelPredictor(model_path)
    kp, hp, act = d["keypoints"][c], d["hand_pose"][c], d["action"][c]
    hands = [HandStep(kp[t], kp[t + 1], hp[t], hp[t + 1], act[t]) for t in range(H)]
    true = d["blocks"][c, :H + 1].astype(np.float64)
    traj = rollout(pred, true[0].copy(), mask, hands)
    q_true = load_qpos(args.data, [gci], info["per_shard"])[gci]
    with np.load(Path(args.data) / f"shard_{gci // info['per_shard']:05d}.npz") as z:
        init, actions = z["init_qpos"][gci % info["per_shard"]], z["action"][gci % info["per_shard"]]
    q_hand = hand_only_qpos(HandEnv(), init, actions[:H])

    dm = mujoco.MjModel.from_xml_string(build_hand_xml(ghosts=True))
    dd = mujoco.MjData(dm)
    sm = mujoco.MjModel.from_xml_string(build_hand_xml())
    ghost_mocap = [dm.body_mocapid[dm.body(f"ghost{i}").id] for i in range(MAX_BLOCKS)]
    ghost_hand = [(dm.jnt_qposadr[dm.joint("g_" + sm.joint(j).name).id], sm.jnt_qposadr[j])
                  for j in range(sm.njnt) if not sm.joint(j).name.startswith("block")]
    bq = [sm.joint(f"block{i}").qposadr[0] for i in range(MAX_BLOCKS)]
    W, Hh = 560, 420
    r3d, rtop = mujoco.Renderer(dm, Hh, W), mujoco.Renderer(dm, Hh, Hh)
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    vw = cv2.VideoWriter(str(out / "demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (W + Hh, Hh))
    gif = []
    for k in range(H + 1):
        dd.qpos[:sm.nq] = q_hand[k]
        for i in range(n):
            dd.qpos[bq[i]:bq[i] + 3] = traj[k, i, :3]
            dd.qpos[bq[i] + 3:bq[i] + 7] = traj[k, i, 3:7]
        for g, s in ghost_hand:
            dd.qpos[g] = q_true[k][s]
        for i in range(MAX_BLOCKS):
            dd.mocap_pos[ghost_mocap[i]] = true[k, i, :3] if i < n else [2 + 0.2 * i, 3, -1]
            dd.mocap_quat[ghost_mocap[i]] = true[k, i, 3:7] if i < n else [1, 0, 0, 0]
        mujoco.mj_kinematics(dm, dd)
        mujoco.mj_camlight(dm, dd)
        r3d.update_scene(dd, camera="persp")
        rtop.update_scene(dd, camera="top")
        frame = cv2.cvtColor(np.hstack([r3d.render(), rtop.render()]), cv2.COLOR_RGB2BGR)
        err = np.linalg.norm(traj[k, :n, :3] - true[k, :n, :3], axis=1).mean() * 100
        cv2.putText(frame, f"t = {k * 0.05:4.1f} s   mean block error {err:.1f} cm", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "solid: learned world model    translucent: MuJoCo ground truth",
                    (10, Hh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        vw.write(frame)
        if k % 2 == 0:
            gif.append(cv2.cvtColor(cv2.resize(frame, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA),
                                    cv2.COLOR_BGR2RGB))
    vw.release()
    from PIL import Image
    imgs = [Image.fromarray(f).quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in gif]
    imgs[0].save(out / "demo.gif", save_all=True, append_images=imgs[1:], duration=100, loop=0, optimize=True)
    print(f"clip {gci} ({n} blocks), model {Path(model_path).parent.name} -> docs/demo.mp4, docs/demo.gif")


if __name__ == "__main__":
    main()
