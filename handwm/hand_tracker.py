"""MediaPipe hand landmarks -> hand features (palm position, heading, index-finger curl).

Landmark ids: 0 wrist, 5/9/13/17 finger roots (MCP), index finger 5-6-7-8.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL = Path(__file__).resolve().parent.parent / "assets" / "hand_landmarker.task"
PALM_IDS = [0, 5, 9, 13, 17]
# joint triples whose bend angles add up to one finger's curl
FINGER_CHAINS = [
    [(1, 2, 3), (2, 3, 4)],                 # thumb
    [(0, 5, 6), (5, 6, 7), (6, 7, 8)],      # index
    [(0, 9, 10), (9, 10, 11), (10, 11, 12)],
    [(0, 13, 14), (13, 14, 15), (14, 15, 16)],
    [(0, 17, 18), (17, 18, 19), (18, 19, 20)],
]
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
               (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


@dataclass
class HandFeatures:
    timestamp: float       # capture time (time.monotonic)
    palm: np.ndarray       # palm centre in image, normalized [0,1], x right / y down
    heading: float         # angle of wrist -> middle-finger root, rad, image "up" = +pi/2
    curl: float            # index finger bend at PIP + DIP joints, rad, ~0 straight
    curls: np.ndarray      # (5,) same measure for thumb, index, middle, ring, little
    landmarks: np.ndarray  # (21, 3) normalized image landmarks
    handedness: str
    normal: np.ndarray = None   # unit palm normal (3D world landmarks)
    forward: np.ndarray = None  # unit wrist -> middle-finger-root direction (3D)


def _angle(a, b, c) -> float:
    """Angle at b between b->a and b->c."""
    u, v = a - b, c - b
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


class HandTracker:
    def __init__(self, model_path: str | Path = MODEL, min_conf: float = 0.5):
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_conf,
            min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._last_ms = -1

    def process(self, rgb: np.ndarray, timestamp: float | None = None) -> HandFeatures | None:
        """rgb: HxWx3 uint8. Returns None when no hand is visible."""
        timestamp = time.monotonic() if timestamp is None else timestamp
        ms = max(int(timestamp * 1000), self._last_ms + 1)  # must strictly increase
        self._last_ms = ms
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        if not res.hand_landmarks:
            return None
        h, w = rgb.shape[:2]
        lm = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]])
        world = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]])  # meters, 3D

        palm = lm[PALM_IDS, :2].mean(axis=0)
        d = (lm[9, :2] - lm[0, :2]) * [w, h]           # pixel vector, isotropic
        heading = float(np.arctan2(-d[1], d[0]))      # flip y so image up is positive
        curls = np.array([
            sum(np.pi - _angle(world[a], world[b], world[c]) for a, b, c in chain)
            for chain in FINGER_CHAINS
        ])
        curl = (np.pi - _angle(world[5], world[6], world[7])) + \
               (np.pi - _angle(world[6], world[7], world[8]))  # single-finger env mapping
        fwd = world[9] - world[0]
        nrm = np.cross(world[5] - world[0], world[17] - world[0])
        return HandFeatures(timestamp, palm, heading, curl, curls, lm,
                            res.handedness[0][0].category_name,
                            normal=nrm / (np.linalg.norm(nrm) + 1e-9),
                            forward=fwd / (np.linalg.norm(fwd) + 1e-9))

    def close(self):
        self._lm.close()
