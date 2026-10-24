"""Grading predicted block trajectories against MuJoCo ground truth.

For every predicted step we put the predicted block poses into MuJoCo together with the
true hand configuration and let MuJoCo's collision detection measure overlaps (geometry only,
no physics is run). Metrics per rollout step k:
  pos_err_cm        mean distance predicted vs true block centre (active blocks)
  pos_err_moved_cm  same, only blocks that really moved > 1 cm in the window
  pen_hand          fraction of blocks overlapping the hand deeper than pen_mm (default 5)
  pen_block         fraction of blocks overlapping another block deeper than pen_mm
  teleport          fraction of blocks that move more than teleport_cm (default 5) in one step
  any_violation     fraction of blocks with any of the three violations
"""
from __future__ import annotations

import mujoco
import numpy as np

PEN_MM = 5.0          # default: deeper overlap = penetration (Grader(pen_mm=...) to change).
                      # On 600 true clips real MuJoCo contacts reach at most 3.9 mm
                      # (hand-block) / 1.3 mm (block-block)
TELEPORT_CM = 5.0     # one step is 0.05 s. True blocks exceed 4.3 cm/step only 0.01% of the
                      # time (rare flicks), so > 5 cm/step = 1 m/s counts as a teleport
METRICS = ["pos_err_cm", "pos_err_moved_cm", "pen_hand", "pen_block", "teleport", "any_violation"]


class Grader:
    def __init__(self, xml_text: str, pen_mm: float = PEN_MM, teleport_cm: float = TELEPORT_CM):
        self.pen_m, self.teleport_cm = pen_mm / 1000, teleport_cm
        self.m = mujoco.MjModel.from_xml_string(xml_text)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.bq = np.array([m.joint(f"block{i}").qposadr[0] for i in range(8)])
        self.block_geom = np.array([m.geom(f"block{i}").id for i in range(8)])
        self.geom_to_block = {g: i for i, g in enumerate(self.block_geom)}
        self.hand_geom = set(np.flatnonzero(m.geom_contype == 2).tolist())

    def violations(self, qpos_hand: np.ndarray, pos: np.ndarray, quat: np.ndarray, n: int):
        """qpos_hand: full qpos whose hand part is used; pos/quat: (8,3)/(8,4) block poses.
        Returns (pen_hand[8], pen_block[8]) booleans."""
        d = self.d
        d.qpos[:] = qpos_hand
        for i in range(n):
            a = self.bq[i]
            d.qpos[a:a + 3] = pos[i]
            q = quat[i]
            d.qpos[a + 3:a + 7] = q / max(np.linalg.norm(q), 1e-8)
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_collision(self.m, d)
        ph, pb = np.zeros(8, bool), np.zeros(8, bool)
        for c in d.contact[:d.ncon]:
            if c.dist > -self.pen_m:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = self.geom_to_block.get(g1), self.geom_to_block.get(g2)
            if b1 is not None and b2 is not None:
                pb[b1] = pb[b2] = True
            elif b1 is not None and g2 in self.hand_geom:
                ph[b1] = True
            elif b2 is not None and g1 in self.hand_geom:
                ph[b2] = True
        return ph[:n], pb[:n]

    def grade(self, pred: np.ndarray, true: np.ndarray, qpos: np.ndarray, n: int) -> dict:
        """pred/true: (K+1, 8, 13) block states over a rollout (step 0 = shared start),
        qpos: (K+1, nq) true MuJoCo states (for the hand). Returns per-step metric arrays (K,)."""
        K = len(pred) - 1
        pp, tp = pred[:, :n, :3], true[:, :n, :3]
        err = np.linalg.norm(pp - tp, axis=-1)[1:] * 100                  # (K, n) cm
        moved = np.linalg.norm(tp[1:] - tp[:1], axis=-1).max(0) > 0.01      # (n,)
        jump = np.linalg.norm(pp[1:] - pp[:-1], axis=-1) * 100 > self.teleport_cm
        ph = np.zeros((K, n), bool)
        pb = np.zeros((K, n), bool)
        for k in range(1, K + 1):
            ph[k - 1], pb[k - 1] = self.violations(qpos[k], pred[k, :, :3], pred[k, :, 3:7], n)
        return {
            "pos_err_cm": err.mean(1),
            "pos_err_moved_cm": err[:, moved].mean(1) if moved.any() else np.full(K, np.nan),
            "pen_hand": ph.mean(1), "pen_block": pb.mean(1), "teleport": jump.mean(1),
            "any_violation": (ph | pb | jump).mean(1),
        }

# Baselines (static, constant velocity) live in predictors.py.
