"""Play inside the learned world: block motion comes from a predictor (the world model),
MuJoCo only DRAWS the scene - no MuJoCo physics moves the solid blocks.

  python scripts/play_wm.py                         # keyboard, newest trained model (or static baseline)
  python scripts/play_wm.py --camera                # your real hand
  python scripts/play_wm.py --predictor model:runs/xxx/model.pt   (once a model is trained)

What you see:
  solid blocks       = the predictor's world
  translucent blocks = MuJoCo's true physics for the same hand motion (the answer key)
  the hand           = driven by your actions (MuJoCo moves only the hand's motors, with no
                       blocks in that simulation, so the hand never "knows" the true blocks)
Keys: as in teleop_hand (arrows / drag, Q/E, A/D, 1-4, C in camera mode) plus
  G  show / hide the translucent truth      S  snap the predicted blocks to the truth
  R  new random scene                        ESC quit
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import mujoco
import numpy as np

from handwm.control import monotonic
from handwm.hand_env import HandEnv
from handwm.hand_scene import MAX_BLOCKS, ROLL_LIMIT, SHAPE_NAMES, build_hand_xml
from handwm.predictors import HandStep, canon, make_predictor
from handwm.wm_data import finite_diff_velocity
from handwm.wm_metrics import Grader
from teleop_hand import KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP, HandKeyboardSource, top_view_to_world

W, H = 640, 480
WIN = "hand-wm: learned world (solid) vs MuJoCo truth (ghost)"


def block_poses(env: HandEnv):
    q = env.data.qpos
    return q[env._bq[:, None] + np.arange(3)].copy(), canon(q[env._bq[:, None] + 3 + np.arange(4)])


def block_states(env: HandEnv, prev=None) -> np.ndarray:
    """(8, 13) pos, quat, lin vel, ang vel of every block slot. Velocities = change since the
    previous step / 0.05 s (the definition used in training); zero without a previous pose."""
    pos, quat = block_poses(env)
    if prev is None:
        vel = np.zeros((len(pos), 6))
    else:
        lv, av = finite_diff_velocity(pos, quat, prev[0], prev[1])
        vel = np.concatenate([lv, av], 1)
    return np.concatenate([pos, quat, vel], 1)


class World:
    """Truth simulation + hand-only simulation + predicted blocks."""

    def __init__(self, predictor, seed, n_blocks):
        self.pred, self.n_req = predictor, n_blocks
        self.rng = np.random.default_rng(seed)
        self.truth = HandEnv(source_name="play_wm_truth")
        self.hand = HandEnv(source_name="play_wm_hand")   # same model, all blocks parked away
        self.reset()

    def reset(self):
        self.truth.rng = np.random.default_rng(self.rng.integers(1 << 31))
        self.truth.reset(n_blocks=self.n_req)
        self.hand.reset(n_blocks=0)
        self.n = self.truth.n_blocks
        self.mask = np.arange(MAX_BLOCKS) < self.n
        self.blocks = block_states(self.truth)          # blocks start at rest
        self.prev_true = block_poses(self.truth)
        self.true = self.blocks.copy()

    def snap(self):
        self.blocks = self.true.copy()

    def step(self, ci):
        kp0, pose0 = self.hand.keypoints(), self.hand.state()["hand_pose"]
        self.truth.step(ci, log=False)
        self.hand.step(ci, log=False)
        hs = HandStep(kp0, self.hand.keypoints(), pose0, self.hand.state()["hand_pose"],
                      self.hand.action.copy())
        self.blocks = self.pred.step(self.blocks, self.mask, hs)
        true = block_states(self.truth, self.prev_true)
        self.prev_true, self.true = block_poses(self.truth), true
        err = np.linalg.norm(self.blocks[:self.n, :3] - true[:self.n, :3], axis=1) * 100
        return true, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", default="auto",
                    help="auto (newest trained model, else static) | static | const_velocity | model:<checkpoint>")
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera", nargs="?", const=0, type=int, default=None)
    ap.add_argument("--yaw-gain", type=float, default=2.0)
    ap.add_argument("--roll-gain", type=float, default=1.0)
    args = ap.parse_args()

    predictor = make_predictor(args.predictor)
    if args.camera is not None:
        from handwm.camera_source import CameraSource
        src = CameraSource(cam_index=args.camera, yaw_gain=args.yaw_gain, roll_gain=args.roll_gain)
    else:
        src = HandKeyboardSource()
    use_keys = isinstance(src, HandKeyboardSource)

    world = World(predictor, args.seed, args.blocks)
    dm = mujoco.MjModel.from_xml_string(build_hand_xml(ghosts=True))   # display model
    dd = mujoco.MjData(dm)
    ghost_mocap = [dm.body_mocapid[dm.body(f"ghost{i}").id] for i in range(MAX_BLOCKS)]
    # ghost hand = translucent copy of the truth sim's hand: display qpos address of "g_<joint>"
    # for every hand joint of the simulation model (visual only, never simulated)
    sm = world.truth.model
    ghost_hand = []
    for j in range(sm.njnt):
        name = sm.joint(j).name
        if not name.startswith("block"):
            ghost_hand.append((dm.jnt_qposadr[dm.joint("g_" + name).id], sm.jnt_qposadr[j]))
    g_x = dm.jnt_qposadr[dm.joint("g_hand_x").id]
    n_sim = sm.nq
    grader = Grader(build_hand_xml())
    r3d, rtop = mujoco.Renderer(dm, H, W), mujoco.Renderer(dm, H, H)
    show_ghost = True
    mouse = {"down": False}

    def on_mouse(event, u, v, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["down"] = True
        elif event == cv2.EVENT_LBUTTONUP:
            mouse["down"] = False
        if use_keys and mouse["down"] and W <= u < W + H:
            src.pose[:2] = top_view_to_world(u - W, v * 480 // H, world.truth.model)

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    print(__doc__)
    try:
        while True:
            t0 = time.perf_counter()
            key = cv2.waitKeyEx(1)
            if key == 27:
                break
            if key in (ord("r"), ord("R")):
                src.reset() if use_keys else src.calibrate()
                world.reset()
                continue
            if key in (ord("g"), ord("G")):
                show_ghost = not show_ghost
            if key in (ord("s"), ord("S")):
                world.snap()
            if use_keys:
                p = src.pose
                if key in KEY_LEFT:    p[0] -= src.STEP_XY
                elif key in KEY_RIGHT: p[0] += src.STEP_XY
                elif key in KEY_UP:    p[1] += src.STEP_XY
                elif key in KEY_DOWN:  p[1] -= src.STEP_XY
                elif key in (ord("q"), ord("Q")): p[2] += src.STEP_YAW
                elif key in (ord("e"), ord("E")): p[2] -= src.STEP_YAW
                elif key in (ord("a"), ord("A")): src.roll = max(src.roll - src.STEP_ROLL, -ROLL_LIMIT)
                elif key in (ord("d"), ord("D")): src.roll = min(src.roll + src.STEP_ROLL, ROLL_LIMIT)
                elif ord("1") <= key <= ord("4"):
                    src.shape = SHAPE_NAMES[key - ord("1")]
            elif key in (ord("c"), ord("C")):
                src.calibrate()

            true, err = world.step(src.read(monotonic()))

            # display: hand from the hand-only sim, solid blocks from the predictor,
            # ghosts (translucent blocks + translucent hand) = the truth sim
            dd.qpos[:n_sim] = world.hand.data.qpos
            for gadr, sadr in ghost_hand:
                dd.qpos[gadr] = world.truth.data.qpos[sadr]
            if not show_ghost:
                dd.qpos[g_x] = 10.0          # hide the ghost hand far away
            for i in range(world.n):
                a = world.hand._bq[i]
                dd.qpos[a:a + 3] = world.blocks[i, :3]
                dd.qpos[a + 3:a + 7] = world.blocks[i, 3:7]
            for i in range(MAX_BLOCKS):
                on = show_ghost and i < world.n
                dd.mocap_pos[ghost_mocap[i]] = true[i, :3] if on else [2 + 0.2 * i, 3, -1]
                dd.mocap_quat[ghost_mocap[i]] = true[i, 3:7] if on else [1, 0, 0, 0]
            mujoco.mj_kinematics(dm, dd)
            mujoco.mj_camlight(dm, dd)   # cameras (no physics is run)
            r3d.update_scene(dd, camera="persp")
            rtop.update_scene(dd, camera="top")
            frame = cv2.cvtColor(np.hstack([r3d.render(), rtop.render()]), cv2.COLOR_RGB2BGR)
            ph, pb = grader.violations(world.truth.data.qpos,   # judged against the TRUE sim hand (as in eval)
                                        world.blocks[:, :3], world.blocks[:, 3:7], world.n)
            if not use_keys:
                cam = src.preview()
                if cam is not None:
                    frame = np.hstack([frame, cv2.resize(cam, (cam.shape[1] * H // cam.shape[0], H))])
            info = (f"world: {predictor.name}   blocks {world.n}   mean error vs truth "
                    f"{err.mean():.1f} cm (max {err.max():.1f})   hand-through-block {int(ph.sum())}"
                    f"   block-overlap {int(pb.sum())}")
            cv2.putText(frame, info, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "solid = learned world   ghost = MuJoCo truth   G ghost  S snap  R reset  ESC quit",
                        (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow(WIN, frame)
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            spare = world.truth.CTRL_DT - (time.perf_counter() - t0)
            if spare > 0:
                time.sleep(spare)
    finally:
        r3d.close()
        rtop.close()
        if not use_keys:
            src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
