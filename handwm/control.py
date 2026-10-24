"""Control interface shared by every input device.

Keyboard/mouse today; camera (MediaPipe) and glove sensors (serial) later.
Each device only has to produce a ControlInput; the simulator never changes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Protocol

import numpy as np


@dataclass
class ControlInput:
    timestamp: float      # when the measurement was taken (time.monotonic seconds)
    x: float              # finger base position on the table plane (m)
    y: float
    yaw: float            # finger base heading (rad), 0 = pointing +x
    bend: float           # finger bend (rad), 0 = straight
    data_age: float = 0.0  # seconds between measurement and use (latency)
    valid: bool = True     # False = device lost tracking; sim holds the last command
    bends: tuple | None = None  # five-finger hand: per-finger bend in [0,1]
                                # (thumb, index, middle, ring, little); None = use `bend`
    roll: float = 0.0       # five-finger hand: palm roll (rad), 0 vertical, -pi/2 palm down

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw, self.bend], dtype=np.float64)

    def to_dict(self) -> dict:
        return asdict(self)


class ControlSource(Protocol):
    name: str

    def read(self, now: float) -> ControlInput: ...


class KeyboardMouseSource:
    """Holds a target pose, updated by key presses and mouse drags."""

    name = "keyboard_mouse"
    STEP_XY = 0.005    # m per key press
    STEP_ANGLE = 0.08  # rad per key press

    def __init__(self, home=(-0.15, 0.0, 0.0, 0.0)):
        self.home = np.array(home, dtype=np.float64)
        self.target = self.home.copy()

    def reset(self):
        self.target = self.home.copy()

    def nudge(self, dx=0.0, dy=0.0, dyaw=0.0, dbend=0.0):
        self.target += np.array([dx * self.STEP_XY, dy * self.STEP_XY,
                                 dyaw * self.STEP_ANGLE, dbend * self.STEP_ANGLE])

    def set_xy(self, x: float, y: float):
        self.target[0], self.target[1] = x, y

    def read(self, now: float) -> ControlInput:
        # A held target is always "fresh": age 0, always valid.
        x, y, yaw, bend = self.target
        return ControlInput(now, x, y, yaw, bend, data_age=0.0, valid=True)


class ScriptedSource:
    """Linear interpolation through (t, x, y, yaw, bend) waypoints, in sim time."""

    name = "scripted"

    def __init__(self, waypoints):
        self.wp = np.asarray(waypoints, dtype=np.float64)

    def read(self, now: float) -> ControlInput:
        t = self.wp[:, 0]
        vals = [float(np.interp(now, t, self.wp[:, i])) for i in range(1, 5)]
        return ControlInput(now, *vals, data_age=0.0, valid=True)


def monotonic() -> float:
    return time.monotonic()
