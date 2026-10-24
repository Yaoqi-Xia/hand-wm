"""Five-finger hand teleop: push 1-8 coloured blocks. Every episode is logged to logs/.

Keyboard / mouse (default):
  arrows   move hand            Q / E   rotate hand (yaw)
  A / D    roll palm: A toward palm-down, D toward palm-up
  1 2 3 4  hand shape: open palm / fist / single-finger poke / two-finger pinch
  left-drag in the LEFT (top-down) view to place the hand
  R  save episode + new random scene      ESC  save + quit
Camera (--camera): your real hand drives position, rotation and all five finger bends.
  Calibrate with an OPEN hand (C). R / ESC as above.
  --yaw-gain 2  (default) the virtual hand turns 2x your real wrist turn.
  --roll-gain 1 (default) palm roll; use -1 if the virtual palm rolls the wrong way.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mujoco
import numpy as np

from handwm.control import ControlInput, monotonic
from handwm.hand_env import HOME, HandEnv
from handwm.hand_scene import HAND_SHAPES, ROLL_LIMIT, SHAPE_NAMES

VIEW = 480
WIN = "hand-wm five-finger teleop"
KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = (2424832, 65361), (2490368, 65362), (2555904, 65363), (2621440, 65364)


class HandKeyboardSource:
    name = "keyboard_mouse_hand"
    STEP_XY, STEP_YAW, STEP_ROLL = 0.01, 0.12, 0.12

    def __init__(self):
        self.reset()

    def reset(self):
        self.pose = HOME[:3].copy()
        self.roll = 0.0
        self.shape = "open"

    def read(self, now):
        x, y, yaw = self.pose
        bends = HAND_SHAPES[self.shape]
        return ControlInput(now, x, y, yaw, bends[1], bends=tuple(bends), roll=self.roll)


def top_view_to_world(u, v, model):
    cam = model.camera("top")
    extent = model.cam_fovy[cam.id]
    cx, cy = model.cam_pos[cam.id][:2]
    s = extent / VIEW
    return cx + (u - VIEW / 2 + 0.5) * s, cy - (v - VIEW / 2 + 0.5) * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=None, help="number of blocks (default random 1-8)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera", nargs="?", const=0, type=int, default=None)
    ap.add_argument("--yaw-gain", type=float, default=2.0,
                    help="camera mode: virtual hand turns this many times your real wrist turn")
    ap.add_argument("--roll-gain", type=float, default=1.0,
                    help="camera mode: palm roll multiplier (negative flips direction)")
    args = ap.parse_args()

    if args.camera is not None:
        from handwm.camera_source import CameraSource
        src = CameraSource(cam_index=args.camera, yaw_gain=args.yaw_gain,
                           roll_gain=args.roll_gain)
    else:
        src = HandKeyboardSource()
    use_keys = isinstance(src, HandKeyboardSource)

    env = HandEnv(seed=args.seed, source_name=src.name)
    env.reset(n_blocks=args.blocks)
    renderer = mujoco.Renderer(env.model, VIEW, VIEW)
    mouse = {"down": False}

    def on_mouse(event, u, v, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["down"] = True
        elif event == cv2.EVENT_LBUTTONUP:
            mouse["down"] = False
        if use_keys and mouse["down"] and u < VIEW:
            src.pose[:2] = top_view_to_world(u, v, env.model)

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    episode = 1
    print(__doc__)

    def finish(reason):
        path = env.save_episode("hand")
        if path:
            print(f"[{reason}] saved {len(env.logger)} steps -> {path}")

    try:
        while True:
            t0 = time.perf_counter()
            key = cv2.waitKeyEx(1)
            if key == 27:
                break
            if key in (ord("r"), ord("R")):
                finish("reset")
                src.reset() if use_keys else src.calibrate()
                env.reset(n_blocks=args.blocks)
                episode += 1
                continue
            if use_keys:
                p = src.pose
                if key in KEY_LEFT:    p[0] -= src.STEP_XY
                elif key in KEY_RIGHT: p[0] += src.STEP_XY
                elif key in KEY_UP:    p[1] += src.STEP_XY
                elif key in KEY_DOWN:  p[1] -= src.STEP_XY
                elif key in (ord("q"), ord("Q")): p[2] += src.STEP_YAW
                elif key in (ord("e"), ord("E")): p[2] -= src.STEP_YAW
                elif key in (ord("a"), ord("A")):
                    src.roll = max(src.roll - src.STEP_ROLL, -ROLL_LIMIT)
                elif key in (ord("d"), ord("D")):
                    src.roll = min(src.roll + src.STEP_ROLL, ROLL_LIMIT)
                elif ord("1") <= key <= ord("4"):
                    src.shape = SHAPE_NAMES[key - ord("1")]
            elif key in (ord("c"), ord("C")):
                src.calibrate()

            s = env.step(src.read(monotonic()))

            renderer.update_scene(env.data, camera="top")
            top = renderer.render()
            renderer.update_scene(env.data, camera="persp")
            persp = renderer.render()
            frame = cv2.cvtColor(np.hstack([top, persp]), cv2.COLOR_RGB2BGR)
            if not use_keys:
                cam = src.preview()
                if cam is not None:
                    frame = np.hstack([frame, cv2.resize(cam, (cam.shape[1] * VIEW // cam.shape[0], VIEW))])
            shape = f"{src.shape} roll {np.degrees(src.roll):+.0f}" if use_keys else "camera"
            cv2.putText(frame, f"ep {episode}  t {s['t']:.1f}s  blocks {env.n_blocks}  hand: {shape}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
            help_txt = ("arrows/drag move  Q/E rotate  A/D roll  1-4 shape  R reset  ESC quit" if use_keys
                        else "your hand drives it  C calibrate (open hand)  R reset  ESC quit")
            cv2.putText(frame, help_txt, (10, VIEW - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
            cv2.imshow(WIN, frame)
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            spare = env.CTRL_DT - (time.perf_counter() - t0)
            if spare > 0:
                time.sleep(spare)
    finally:
        finish("quit")
        renderer.close()
        if not use_keys:
            src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
