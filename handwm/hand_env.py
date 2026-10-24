"""Five-finger hand + blocks environment.

Action (9 numbers, ACTION_NAMES): palm centre x, y (m), yaw (rad, full circle),
roll (rad: 0 vertical chop, -pi/2 palm down, +pi/2 palm up), bends[5] in [0, 1]
(thumb, index, middle, ring, little). Same ControlInput interface as the finger env.

Height is not an action: every tick the env lifts the hand so that its lowest point -
for both the commanded pose and the current pose - stays CLEARANCE above the table.
"""
from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np

from .control import ControlInput
from .hand_scene import (ARENA_HALF, BLOCK_HALF, KEYPOINTS, MAX_BLOCKS, ROLL_LIMIT,
                         bend_targets, build_hand_xml)
from .logger import EpisodeLogger

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"

ACTION_NAMES = ["x", "y", "yaw", "roll", "b_thumb", "b_index", "b_middle", "b_ring", "b_little"]
ACT_DIM = len(ACTION_NAMES)
ACT_LOW = np.array([-0.22, -0.22, -np.pi, -ROLL_LIMIT, 0, 0, 0, 0, 0])
ACT_HIGH = np.array([0.22, 0.22, np.pi, ROLL_LIMIT, 1, 1, 1, 1, 1])
HOME = np.array([-0.15, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0])
SPAWN_HALF = ARENA_HALF - 0.05
CLEARANCE = 0.006   # lowest point of the hand above the table (m)
# The hand may not come closer than WALL_BAND to a wall (a block is 5 cm): it can push blocks
# to and along the walls, but never pin one against a wall, where the hand would get stuck.
WALL_BAND = 0.055
HAND_POSE_NAMES = ["x", "y", "z", "yaw", "roll"]


class HandEnv:
    CTRL_DT = 0.05  # 20 Hz control / logging; physics at 0.002 s

    def __init__(self, xml_text: str | None = None, seed: int = 0, stale_after: float = 0.2,
                 log_dir: str | Path = LOG_DIR, source_name: str = "unknown"):
        self.xml_text = xml_text if xml_text is not None else build_hand_xml()
        self.model = mujoco.MjModel.from_xml_string(self.xml_text)
        self.data = mujoco.MjData(self.model)
        self._probe = mujoco.MjData(self.model)   # for the commanded pose's lowest point
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.stale_after = stale_after
        self.log_dir = Path(log_dir)
        self.source_name = source_name
        self.n_sub = int(round(self.CTRL_DT / self.model.opt.timestep))

        m = self.model
        self._hq = [m.joint(n).qposadr[0] for n in ("hand_x", "hand_y", "hand_z", "hand_yaw", "hand_roll")]
        self._bq = np.array([m.joint(f"block{i}").qposadr[0] for i in range(MAX_BLOCKS)])
        self._bv = np.array([m.joint(f"block{i}").dofadr[0] for i in range(MAX_BLOCKS)])
        self._base_act = [m.actuator(n).id for n in ("A_hand_x", "A_hand_y", "A_hand_yaw", "A_hand_roll")]
        self._z_act = m.actuator("A_hand_z").id
        names = [m.actuator(i).name for i in range(m.nu)]
        self._finger_names = [n for n in names if n.startswith("rh_A_")]
        self._finger_ids = np.array([names.index(n) for n in self._finger_names])
        # hand collision geoms (contype 2) for the lowest-point computation
        self._hand_geoms = np.flatnonzero(m.geom_contype == 2)
        self._aabb = m.geom_aabb[self._hand_geoms]          # local centre(3), half size(3)
        # actuator -> joint(s) map for posing the probe at the commanded finger angles
        self._probe_map = []
        for n in self._finger_names:
            a = m.actuator(n)
            if a.trntype[0] == mujoco.mjtTrn.mjTRN_JOINT:
                self._probe_map.append((n, [m.jnt_qposadr[a.trnid[0]]]))
            else:  # tendon-coupled middle+distal joints: split evenly
                tid = a.trnid[0]
                adr, num = m.tendon_adr[tid], m.tendon_num[tid]
                jids = m.wrap_objid[adr:adr + num]
                self._probe_map.append((n, [m.jnt_qposadr[j] for j in jids]))
        self._kp = [(kind, (m.body(n).id if kind == "body" else m.site(n).id)) for kind, n in KEYPOINTS]
        self.n_blocks = 0
        self.action = HOME.copy()
        self.tick = 0
        self.logger: EpisodeLogger | None = None

    # ---------- episode ----------
    def reset(self, n_blocks: int | None = None, init_qpos=None, init_qvel=None,
              hand_pose=None) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if init_qpos is not None:
            d.qpos[:] = init_qpos
            if init_qvel is not None:
                d.qvel[:] = init_qvel
            self.n_blocks = int(n_blocks)
            q = d.qpos[self._hq]
            self.action = np.r_[q[[0, 1, 3, 4]], np.zeros(5)]
            mujoco.mj_forward(m, d)
        else:
            self.n_blocks = int(self.rng.integers(1, MAX_BLOCKS + 1)) if n_blocks is None else n_blocks
            pose = HOME.copy() if hand_pose is None else np.asarray(hand_pose, float)
            pose = self._keep_off_walls(pose.copy())
            d.qpos[self._hq] = [pose[0], pose[1], 0.0, pose[2], pose[3]]
            self._place_blocks(pose[:2])
            self.action = pose.copy()
            d.qpos[self._hq[2]] = self._commanded_height(self.action)
            mujoco.mj_forward(m, d)
        self._apply(self.action)
        self.tick = 0
        self.logger = EpisodeLogger(self.log_dir, {
            "env": "hand", "xml": self.xml_text, "timestep": m.opt.timestep,
            "ctrl_dt": self.CTRL_DT, "seed": self.seed, "stale_after": self.stale_after,
            "source": self.source_name, "n_blocks": self.n_blocks, "action_names": ACTION_NAMES,
            "init_qpos": d.qpos.tolist(), "init_qvel": d.qvel.tolist(),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, csv_columns=None)

    def _place_blocks(self, hand_xy):
        """Random non-overlapping positions; unused slots parked far outside the arena."""
        d = self.data
        for _ in range(1000):  # whole-layout rejection sampling: every block must fit
            placed = []
            for i in range(self.n_blocks):
                for _ in range(100):
                    p = self.rng.uniform(-SPAWN_HALF, SPAWN_HALF, 2)
                    if all(np.linalg.norm(p - q) > 0.075 for q in placed) and \
                       np.linalg.norm(p - hand_xy) > 0.21:  # hand reaches ~0.14 m from palm centre
                        placed.append(p)
                        break
            if len(placed) == self.n_blocks:
                break
        while len(placed) < self.n_blocks:  # crowded fallback (hand far from home): space only
            p = self.rng.uniform(-SPAWN_HALF, SPAWN_HALF, 2)
            if all(np.linalg.norm(p - q) > 0.075 for q in placed):
                placed.append(p)
        for i in range(MAX_BLOCKS):
            a = self._bq[i]
            if i >= self.n_blocks:
                d.qpos[a:a + 3] = [2 + 0.2 * i, 2, BLOCK_HALF]
                d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
                continue
            p = placed[i]
            yaw = self.rng.uniform(-np.pi, np.pi)
            d.qpos[a:a + 3] = [p[0], p[1], BLOCK_HALF]
            d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    # ---------- height ----------
    def _lowest(self, data) -> float:
        """Lowest z of the hand's collision geoms (axis-aligned bounds, conservative)."""
        R = data.geom_xmat[self._hand_geoms, 6:9]           # third row of each rotation
        c, h = self._aabb[:, :3], self._aabb[:, 3:]
        z = data.geom_xpos[self._hand_geoms, 2] + (R * c).sum(1) - (np.abs(R) * h).sum(1)
        return float(z.min())

    def _pose_probe(self, action):
        """Pose the probe at the commanded yaw, roll and finger targets, palm centre at origin."""
        p = self._probe
        p.qpos[:] = self.model.qpos0
        p.qpos[self._hq[3]] = action[2]
        p.qpos[self._hq[4]] = action[3]
        t = bend_targets(action[4:9])
        for name, adrs in self._probe_map:
            for adr in adrs:
                p.qpos[adr] = t[name] / len(adrs)
        mujoco.mj_kinematics(self.model, p)
        return p

    def _commanded_height(self, action) -> float:
        """Palm height that puts the commanded pose (roll + finger targets) at CLEARANCE."""
        return CLEARANCE - self._lowest(self._pose_probe(action))

    def _keep_off_walls(self, action):
        """Shift the commanded palm position so no part of the hand enters the wall band."""
        p = self._pose_probe(action)
        g = self._hand_geoms
        c, h = self._aabb[:, :3], self._aabb[:, 3:]
        lim = ARENA_HALF - WALL_BAND
        for ax in (0, 1):
            R = p.geom_xmat[g, 3 * ax:3 * ax + 3]
            mid = p.geom_xpos[g, ax] + (R * c).sum(1)
            ext = (np.abs(R) * h).sum(1)
            lo, hi = -lim - (mid - ext).min(), lim - (mid + ext).max()
            action[ax] = (lo + hi) / 2 if lo > hi else np.clip(action[ax], lo, hi)
        return action

    # ---------- stepping ----------
    def _apply(self, action):
        d = self.data
        d.ctrl[self._base_act] = action[:4]
        t = bend_targets(action[4:9])
        d.ctrl[self._finger_ids] = [t[n] for n in self._finger_names]
        self._z_cmd = self._commanded_height(action)
        self._update_z()

    def _update_z(self):
        """Height target = max(height the commanded pose needs, height the current pose needs).
        Called every physics step, so fast finger curls are caught within 2 ms."""
        d = self.data
        z_actual = d.qpos[self._hq[2]] + CLEARANCE - self._lowest(d)
        d.ctrl[self._z_act] = max(self._z_cmd, z_actual)

    def _physics_tick(self):
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
            self._update_z()

    def apply_control(self, ci: ControlInput) -> bool:
        accepted = bool(ci.valid) and ci.data_age <= self.stale_after
        if accepted:
            bends = ci.bends if ci.bends is not None else [ci.bend] * 5
            a = np.r_[ci.x, ci.y, ci.yaw, ci.roll, bends]
            # keep yaw continuous so the position servo never spins the long way round
            a[2] = self.action[2] + (a[2] - self.action[2] + np.pi) % (2 * np.pi) - np.pi
            a[:2] = np.clip(a[:2], ACT_LOW[:2], ACT_HIGH[:2])
            a[3:] = np.clip(a[3:], ACT_LOW[3:], ACT_HIGH[3:])
            self.action = self._keep_off_walls(a)
        self._apply(self.action)
        return accepted

    def step(self, ci: ControlInput, log: bool = True) -> dict:
        """One control tick (n_sub physics steps); logs one row per tick."""
        accepted = self.apply_control(ci)
        self._physics_tick()
        self.tick += 1
        s = self.state()
        if log:
            self.logger.append(
                t=self.data.time, tick=self.tick, wall=time.monotonic(),
                in_x=ci.x, in_y=ci.y, in_yaw=ci.yaw, in_roll=ci.roll,
                in_bends=ci.bends if ci.bends is not None else [ci.bend] * 5,
                in_timestamp=ci.timestamp, in_data_age=ci.data_age,
                in_valid=float(ci.valid), accepted=float(accepted),
                action=self.action, hand_pose=s["hand_pose"], keypoints=s["keypoints"],
                block_pos=s["block_pos"], block_quat=s["block_quat"],
                qpos=self.data.qpos, qvel=self.data.qvel, ctrl=self.data.ctrl,
            )
        return s

    def save_episode(self, tag: str = "hand") -> Path | None:
        return self.logger.save(tag) if self.logger else None

    # ---------- state ----------
    def keypoints(self) -> np.ndarray:
        """21 MediaPipe-style hand keypoints in world coordinates (m), shape (21, 3)."""
        d = self.data
        return np.array([d.xpos[i] if kind == "body" else d.site_xpos[i] for kind, i in self._kp])

    def state(self) -> dict:
        d, n = self.data, self.n_blocks
        idx = self._bq[:n, None]
        return {
            "t": d.time,
            "hand_pose": d.qpos[self._hq].copy(),               # x, y, z, yaw, roll (actual)
            "keypoints": self.keypoints(),
            "block_pos": d.qpos[idx + np.arange(3)].copy(),     # (n, 3)
            "block_quat": d.qpos[idx + 3 + np.arange(4)].copy(),  # (n, 4)
            "block_vel": d.qvel[self._bv[:n, None] + np.arange(3)].copy(),
        }

    def block_velocities(self) -> np.ndarray:
        """(8, 6) linear + angular velocity of every block slot, world frame."""
        out, v6 = np.zeros((MAX_BLOCKS, 6)), np.zeros(6)
        for i in range(MAX_BLOCKS):
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                     self.model.body(f"block{i}").id, v6, 0)
            out[i] = np.r_[v6[3:], v6[:3]]  # MuJoCo gives (angular, linear)
        return out

    def all_block_pos(self) -> np.ndarray:
        return self.data.qpos[self._bq[:, None] + np.arange(3)].copy()
