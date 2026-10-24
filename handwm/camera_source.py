"""Webcam hand tracking as a ControlSource.

Relative mapping: on calibration the current hand pose becomes the finger's home pose;
afterwards hand motion moves the finger from there. Calibration happens automatically
on the first detected hand and again whenever calibrate() is called (C key).

Camera image is mirrored (like a selfie): move your hand right -> finger moves +x,
hand up -> +y, rotate hand -> finger yaw, curl index finger -> finger bend.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from .control import ControlInput
from .hand_tracker import CONNECTIONS, HandFeatures, HandTracker


# curl (rad, summed joint bends) of a full fist, per finger: thumb, index, middle, ring, little.
# Calibrate with an open hand; a fist then maps to bends of about 1.
FULL_CURL = np.array([1.0, 3.0, 3.0, 3.0, 3.0])


def _palm_roll(feat: HandFeatures, ref: HandFeatures) -> float:
    """Rotation of the palm normal about the hand's long axis since calibration (rad)."""
    f = feat.forward
    a = ref.normal - np.dot(ref.normal, f) * f
    b = feat.normal - np.dot(feat.normal, f) * f
    return float(np.arctan2(np.dot(np.cross(a, b), f), np.dot(a, b)))


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class CameraSource:
    name = "camera_mediapipe"

    def __init__(self, cam_index: int = 0, home=(-0.15, 0.0, 0.0, 0.0),
                 gain: float = 0.6, curl_gain: float = 0.6, smooth: float = 0.5,
                 width: int = 640, height: int = 480, yaw_gain: float = 1.0,
                 roll_gain: float = 1.0):
        self.home = np.array(home, dtype=np.float64)
        self.gain = gain              # meters of finger travel per full image width
        self.curl_gain = curl_gain    # sim bend per rad of index-finger curl
        self.smooth = smooth          # EMA weight of the new sample (1 = no smoothing)
        self.yaw_gain = yaw_gain      # sim rotation per real hand rotation (wrist turns ~+-90 deg)
        self.roll_gain = roll_gain    # palm roll; negative flips the direction
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {cam_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.tracker = HandTracker()

        self._lock = threading.Lock()
        self._latest = ControlInput(time.monotonic(), *self.home, valid=False)
        self._frame = None
        self._feat: HandFeatures | None = None
        self._ref: HandFeatures | None = None   # calibration pose
        self._filt: np.ndarray | None = None
        self._recalib = True
        self.fps = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ---------- public ----------
    def calibrate(self):
        self._recalib = True

    def read(self, now: float) -> ControlInput:
        with self._lock:
            ci = self._latest
        return ControlInput(ci.timestamp, ci.x, ci.y, ci.yaw, ci.bend,
                            data_age=now - ci.timestamp, valid=ci.valid, bends=ci.bends,
                            roll=ci.roll)

    def preview(self) -> np.ndarray | None:
        """Latest mirrored BGR frame with landmarks and readouts drawn."""
        with self._lock:
            frame, feat, ci = self._frame, self._feat, self._latest
        if frame is None:
            return None
        img = frame.copy()
        h, w = img.shape[:2]
        if feat is not None:
            pts = (feat.landmarks[:, :2] * [w, h]).astype(int)
            for a, b in CONNECTIONS:
                cv2.line(img, tuple(pts[a]), tuple(pts[b]), (0, 220, 0), 2)
            for i, p in enumerate(pts):
                cv2.circle(img, tuple(p), 6 if i in (5, 6, 7, 8) else 4,
                           (0, 0, 255) if i in (5, 6, 7, 8) else (255, 255, 255), -1)
            txt = (f"x {ci.x:+.3f} y {ci.y:+.3f} yaw {np.degrees(ci.yaw):+5.0f} "
                   f"bend {np.degrees(ci.bend):4.0f}")
            if ci.bends is not None:
                cv2.putText(img, f"roll {np.degrees(ci.roll):+4.0f}  bends T I M R L  "
                            + " ".join(f"{b:.1f}" for b in ci.bends),
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        else:
            txt = "no hand"
        cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, f"{self.fps:4.1f} fps   C = calibrate", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return img

    def close(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()
        self.tracker.close()

    # ---------- worker thread ----------
    def _loop(self):
        last = time.monotonic()
        while self._running:
            ok, frame = self.cap.read()
            t = time.monotonic()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            feat = self.tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), t)
            ci = self._to_input(feat, t)
            with self._lock:
                self._frame, self._feat, self._latest = frame, feat, ci
            now = time.monotonic()
            self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-3)
            last = now

    def _to_input(self, feat: HandFeatures | None, t: float) -> ControlInput:
        if feat is None:
            prev = self._latest
            return ControlInput(t, prev.x, prev.y, prev.yaw, prev.bend, valid=False,
                                bends=prev.bends, roll=prev.roll)
        if self._recalib or self._ref is None:
            self._ref, self._filt, self._recalib = feat, None, False
        aspect = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / max(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), 1)
        d = feat.palm - self._ref.palm
        raw = np.array([
            self.home[0] + self.gain * d[0],
            self.home[1] - self.gain * aspect * d[1],  # image y points down
            self.home[2] + self.yaw_gain * _wrap(feat.heading - self._ref.heading),
            self.home[3] + self.curl_gain * max(feat.curl - self._ref.curl, 0.0),
            # five-finger hand: per-finger bend in [0,1] relative to the calibration pose
            *np.clip((feat.curls - self._ref.curls) / FULL_CURL, 0.0, 1.0),
            self.roll_gain * _palm_roll(feat, self._ref),
        ])
        if self._filt is None:
            self._filt = raw
        else:
            for i in (2, 9):  # angles: no jump at +-pi
                raw[i] = self._filt[i] + _wrap(raw[i] - self._filt[i])
            self._filt = self.smooth * raw + (1 - self.smooth) * self._filt
        f = self._filt
        return ControlInput(feat.timestamp, *f[:4], valid=True, bends=tuple(f[4:9]),
                            roll=float(f[9]))
