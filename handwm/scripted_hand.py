"""Scripted random hand: wanders, changes shape, and pushes blocks from random directions.

Produces one ControlInput per control tick (env.CTRL_DT). Plans short segments:
  push   - go behind a random block, then drive through it (70%)
  wander - go to a random point with a random heading (20%)
  pause  - stay put and change hand shape (10%)
Each segment picks a hand shape and a speed. Shapes: 70% one of the four labelled shapes
(open / fist / poke / pinch, shape_id 0-3); 30% "free" fingers (shape_id 4): each finger
bends independently at random and is re-drawn every 0.3-1.5 s, also while pushing.
Palm roll is random in both modes: each segment picks a roll target (25% near the vertical
chop, 75% anywhere from palm down to palm up), and it may change again mid-segment.
"""
from __future__ import annotations

import numpy as np

from .control import ControlInput
from .hand_env import ACT_HIGH, ACT_LOW, ROLL_LIMIT
from .hand_scene import HAND_SHAPES, SHAPE_NAMES

FREE_ID = len(SHAPE_NAMES)   # shape_id 4 = free fingers
LABELS = SHAPE_NAMES + ["free"]


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class RandomHandPolicy:
    name = "scripted_random_hand"

    def __init__(self, env, rng: np.random.Generator, shapes=SHAPE_NAMES, p_free: float = 0.3):
        self.env, self.rng = env, rng
        self.shapes = list(shapes)          # allowed shapes (for leave-one-shape-out)
        self.pose = np.r_[env.action[:3]]   # commanded x, y, yaw
        self.roll = float(env.action[3])
        self.roll_target = self.roll
        self.bends = np.array(env.action[4:9], dtype=float)
        self.waypoints: list[np.ndarray] = []
        self.shape_id = 0
        self.speed = 0.2
        self.hold = 0
        self.seg_ticks = 0
        self.p_free = p_free
        self.free_target = np.zeros(5)
        self.free_ticks = 0

    # ---------- planning ----------
    def _new_segment(self):
        r = self.rng
        if r.random() < self.p_free:
            self.shape_id, self.free_ticks = FREE_ID, 0
        else:
            self.shape_id = SHAPE_NAMES.index(r.choice(self.shapes))
        self.speed = r.uniform(0.08, 0.35)
        self._new_roll()
        kind = r.choice(["push", "wander", "pause"], p=[0.7, 0.2, 0.1])
        if kind == "push" and self.env.n_blocks > 0:
            b = self.env.state()["block_pos"][r.integers(self.env.n_blocks), :2]
            th = r.uniform(-np.pi, np.pi)
            u = np.array([np.cos(th), np.sin(th)])
            lead = r.choice([0.0, np.pi / 2, -np.pi / 2])   # fingers first / palm or back first
            yaw = th + lead + r.normal(0, 0.25)
            back = 0.15 if lead == 0.0 else 0.09
            start = b - back * u + r.normal(0, 0.01, 2)
            end = b + r.uniform(0.04, 0.14) * u
            # targets outside the reachable area would never be reached -> clip them
            start, end = (np.clip(p, ACT_LOW[:2], ACT_HIGH[:2]) for p in (start, end))
            self.waypoints = [np.r_[start, yaw], np.r_[end, yaw]]
        elif kind == "wander":
            p = r.uniform(ACT_LOW[:2], ACT_HIGH[:2])
            self.waypoints = [np.r_[p, r.uniform(-np.pi, np.pi)]]
        else:
            self.waypoints = []
            self.hold = int(r.integers(10, 30))

    def _new_roll(self):
        r = self.rng
        self.roll_target = (r.normal(0, 0.2) if r.random() < 0.25
                            else r.uniform(-ROLL_LIMIT, ROLL_LIMIT))
        self.roll_target = float(np.clip(self.roll_target, -ROLL_LIMIT, ROLL_LIMIT))

    # ---------- per tick ----------
    def read(self, now: float) -> ControlInput:
        dt = self.env.CTRL_DT
        self.seg_ticks += 1
        if (not self.waypoints and self.hold <= 0) or self.seg_ticks > 160:  # 8 s timeout
            self._new_segment()
            self.seg_ticks = 0
        if self.waypoints:
            tgt = self.waypoints[0]
            d = tgt[:2] - self.pose[:2]
            dist = np.linalg.norm(d)
            step = self.speed * dt
            self.pose[:2] += d if dist <= step else d / dist * step
            dyaw = _wrap(tgt[2] - self.pose[2])
            self.pose[2] += np.clip(dyaw, -2.5 * dt, 2.5 * dt)
            if dist <= step and abs(dyaw) < 0.05:
                self.waypoints.pop(0)
        else:
            self.hold -= 1
        self.pose[:2] = np.clip(self.pose[:2], ACT_LOW[:2], ACT_HIGH[:2])
        self.pose[2] = _wrap(self.pose[2])
        if self.rng.random() < 0.01:   # occasional mid-segment roll change (~every 5 s)
            self._new_roll()
        self.roll += float(np.clip(self.roll_target - self.roll, -2.0 * dt, 2.0 * dt))

        if self.shape_id == FREE_ID:
            self.free_ticks -= 1
            if self.free_ticks <= 0:  # new independent random bend per finger
                self.free_target = self.rng.uniform(0, 1, 5)
                self.free_ticks = int(self.rng.integers(6, 31))
            target = self.free_target
        else:
            target = np.array(HAND_SHAPES[SHAPE_NAMES[self.shape_id]])
        target = np.clip(target + self.rng.normal(0, 0.05, 5), 0, 1)
        self.bends += 0.25 * (target - self.bends)
        x, y, yaw = self.pose
        return ControlInput(now, x, y, yaw, float(self.bends[1]), data_age=0.0, valid=True,
                            bends=tuple(self.bends), roll=self.roll)
