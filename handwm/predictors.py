"""Predictor interface: anything that maps (block states now, hand) -> block states next step.

The world model (architecture still to be decided), the baselines and a MuJoCo "oracle"
all implement this, so the grader, the evaluation script and the live display work with any
of them unchanged.

Block state per slot (13): position (3), quaternion w,x,y,z (4), linear vel (3), angular vel (3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

DT = 0.05


@dataclass
class HandStep:
    """Everything known about the hand for one step t -> t+1."""
    keypoints_t: np.ndarray    # (21, 3) hand keypoints now
    keypoints_t1: np.ndarray   # (21, 3) hand keypoints at the next step (hand follows the action)
    hand_pose_t: np.ndarray    # (5,) x, y, z, yaw, roll now
    hand_pose_t1: np.ndarray   # (5,) next step
    action_t: np.ndarray       # (9,) command x, y, yaw, roll, 5 bends


class Predictor(Protocol):
    name: str

    def step(self, blocks: np.ndarray, mask: np.ndarray, hand: HandStep) -> np.ndarray:
        """blocks (8, 13), mask (8,) bool -> next blocks (8, 13). Inactive slots unchanged."""
        ...


# ---------------- quaternion helpers (numpy, w x y z) ----------------
def quat_mul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


def rotvec_to_quat(r):
    ang = np.linalg.norm(r, axis=-1, keepdims=True)
    k = np.where(ang > 1e-8, np.sin(ang / 2) / np.maximum(ang, 1e-8), 0.5)
    return np.concatenate([np.cos(ang / 2), r * k], -1)


def canon(q):
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    return np.where(q[..., :1] < 0, -q, q)


# ---------------- baselines ----------------
class StaticPredictor:
    """Blocks never move."""
    name = "static"

    def step(self, blocks, mask, hand):
        out = blocks.copy()
        out[mask, 7:13] = 0
        return out


class ConstantVelocityPredictor:
    """Every block keeps its current linear and angular velocity."""
    name = "const_velocity"

    def step(self, blocks, mask, hand):
        out = blocks.copy()
        b = out[mask]
        b[:, :3] += b[:, 7:10] * DT
        b[:, 3:7] = canon(quat_mul(rotvec_to_quat(b[:, 10:13] * DT), b[:, 3:7]))
        out[mask] = b
        return out


class TruthPredictor:
    """Replays the true next states (sanity check for the grader: should score ~0 errors)."""
    name = "mujoco_truth"

    def __init__(self):
        self.seq, self.t = None, 0

    def set_sequence(self, seq):
        self.seq, self.t = seq, 0

    def step(self, blocks, mask, hand):
        self.t += 1
        return self.seq[self.t].copy()


def latest_model(runs_dir=None):
    """Newest runs/*/model.pt, or None."""
    from pathlib import Path
    runs = Path(runs_dir) if runs_dir else Path(__file__).resolve().parent.parent / "runs"
    ms = sorted((p for p in runs.glob("*/model.pt") if "_smoke" not in p.parent.name),
                key=lambda p: p.stat().st_mtime)
    return ms[-1] if ms else None


def bundled_model():
    """models/main_seed0/model.pt shipped with the repository (used when runs/ is empty)."""
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "models" / "main_seed0" / "model.pt"
    return p if p.exists() else None


def latest_run_model(tag: str, onestep: bool = False):
    """Newest runs/train_*_<tag>/model.pt (or model_onestep.pt)."""
    from pathlib import Path
    runs = Path(__file__).resolve().parent.parent / "runs"
    name = "model_onestep.pt" if onestep else "model.pt"
    ms = sorted(runs.glob(f"train_*_{tag}/{name}"), key=lambda p: p.stat().st_mtime)
    return ms[-1] if ms else None


def make_predictor(spec: str):
    """'static' | 'const_velocity' | 'model:<path to checkpoint>' |
    'latest:<tag>' / 'latest:<tag>:onestep' (newest training run with that tag) |
    'auto' (newest trained model if there is one, else static)"""
    if spec.startswith("latest:"):
        parts = spec.split(":")
        m = latest_run_model(parts[1], onestep=len(parts) > 2 and parts[2] == "onestep")
        if m is None:
            raise SystemExit(f"no trained run with tag {parts[1]!r} in runs/")
        p = make_predictor(f"model:{m}")
        p.name = parts[1] + ("_onestep" if len(parts) > 2 else "")
        return p
    if spec == "auto":
        # prefer the newest main model (tag exactly "main"); comparison / extra-seed runs
        # (mlp, action, main_s1, ...) are only used when asked for explicitly
        m = latest_run_model("main") or latest_model() or bundled_model()
        if m is None:
            print("no trained world model yet -> using the 'static' baseline")
            return StaticPredictor()
        spec = f"model:{m}"
    if spec in ("static", "const_velocity"):
        return StaticPredictor() if spec == "static" else ConstantVelocityPredictor()
    if spec.startswith("model:"):
        try:
            from .wm_model import load_predictor  # provided once the architecture is chosen
        except ImportError as e:
            raise SystemExit("No trained world model yet (architecture not decided). "
                             "Use --predictor static or const_velocity for now.") from e
        return load_predictor(spec[len("model:"):])
    raise SystemExit(f"unknown predictor {spec!r}")


def rollout(pred: Predictor, s0: np.ndarray, mask: np.ndarray, hands: list[HandStep]) -> np.ndarray:
    out = [s0]
    s = s0
    for h in hands:
        s = pred.step(s, mask, h)
        out.append(s)
    return np.stack(out)
