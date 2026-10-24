"""Interactive teleop: keyboard + mouse drive the finger, every episode is logged.

Keys (click the window first so it has focus):
  arrows   move finger        Q / E   rotate finger
  Z / X    bend / straighten  R       save episode + reset
  ESC      save episode + quit
Mouse: left-drag in the LEFT (top-down) view to place the finger.

With --camera the webcam hand drives the finger instead (keyboard/mouse movement off):
  C  re-calibrate (current hand pose = finger home)   R / ESC as above
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mujoco
import numpy as np

from handwm.control import KeyboardMouseSource, monotonic
from handwm.env import PushEnv

VIEW = 480
WIN = "hand-wm teleop"

# cv2.waitKeyEx codes: Windows, then Linux/GTK
KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = (2424832, 65361), (2490368, 65362), (2555904, 65363), (2621440, 65364)


def top_view_to_world(u, v, model):
    """Pixel in the orthographic top camera -> world (x, y)."""
    cam = model.camera("top")
    extent = model.cam_fovy[cam.id]  # orthographic: frustum height in meters
    cx, cy = model.cam_pos[cam.id][:2]  # camera lives on the world body
    scale = extent / VIEW
    return cx + (u - VIEW / 2 + 0.5) * scale, cy - (v - VIEW / 2 + 0.5) * scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-random", action="store_true", help="block always starts at the same pose")
    ap.add_argument("--camera", nargs="?", const=0, type=int, default=None,
                    help="drive the finger with your hand via webcam (optional camera index)")
    args = ap.parse_args()

    if args.camera is not None:
        from handwm.camera_source import CameraSource
        src = CameraSource(cam_index=args.camera)
    else:
        src = KeyboardMouseSource()
    use_keys = isinstance(src, KeyboardMouseSource)
    env = PushEnv(seed=args.seed, randomize=not args.no_random, source_name=src.name)
    env.reset()
    renderer = mujoco.Renderer(env.model, VIEW, VIEW)

    mouse = {"down": False}

    def on_mouse(event, u, v, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["down"] = True
        elif event == cv2.EVENT_LBUTTONUP:
            mouse["down"] = False
        if use_keys and mouse["down"] and u < VIEW:
            src.set_xy(*top_view_to_world(u, v, env.model))

    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    episode = 1
    print(__doc__)

    def finish(reason):
        path = env.save_episode()
        if path:
            print(f"[{reason}] saved {len(env.logger)} steps -> {path}")

    try:
        while True:
            t0 = time.perf_counter()
            key = cv2.waitKeyEx(1)
            if key == 27:  # ESC
                break
            if not use_keys:
                if key in (ord("c"), ord("C")):
                    src.calibrate()
                if key not in (ord("r"), ord("R")):
                    key = -1  # movement keys are ignored in camera mode
            if key in KEY_LEFT:  src.nudge(dx=-1)
            elif key in KEY_RIGHT: src.nudge(dx=+1)
            elif key in KEY_UP:   src.nudge(dy=+1)
            elif key in KEY_DOWN: src.nudge(dy=-1)
            elif key in (ord("q"), ord("Q")): src.nudge(dyaw=+1)
            elif key in (ord("e"), ord("E")): src.nudge(dyaw=-1)
            elif key in (ord("z"), ord("Z")): src.nudge(dbend=+1)
            elif key in (ord("x"), ord("X")): src.nudge(dbend=-1)
            elif key in (ord("r"), ord("R")):
                finish("reset")
                src.calibrate() if not use_keys else src.reset()
                env.reset()
                episode += 1
                continue

            s = env.step(src.read(monotonic()))

            renderer.update_scene(env.data, camera="top")
            top = renderer.render()
            renderer.update_scene(env.data, camera="persp")
            persp = renderer.render()
            frame = cv2.cvtColor(np.hstack([top, persp]), cv2.COLOR_RGB2BGR)
            if not use_keys:
                cam = src.preview()
                if cam is not None:
                    cam = cv2.resize(cam, (cam.shape[1] * VIEW // cam.shape[0], VIEW))
                    frame = np.hstack([frame, cam])
            status = "SUCCESS" if s["success"] else f"dist {s['dist_to_target'] * 100:.1f} cm"
            cv2.putText(frame, f"ep {episode}  t {s['t']:.1f}s  {status}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 0) if s["success"] else (30, 30, 30), 2)
            help_txt = ("arrows/drag move  Q/E yaw  Z/X bend  R reset  ESC quit" if use_keys
                        else "hand moves finger  C calibrate  R reset  ESC quit")
            cv2.putText(frame, help_txt,
                        (10, VIEW - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
            cv2.imshow(WIN, frame)
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:  # window closed
                break

            # keep roughly real time
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
