"""MuJoCo finger-push environment. Input: ControlInput. Output: state + per-step log."""
from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np

from .control import ControlInput
from .logger import EpisodeLogger

ROOT = Path(__file__).resolve().parent.parent
SCENE = ROOT / "assets" / "push_scene.xml"
LOG_DIR = ROOT / "logs"

# Workspace limits for [x, y, yaw, bend]
CMD_LOW = np.array([-0.25, -0.30, -np.pi, -1.4])
CMD_HIGH = np.array([0.45, 0.30, np.pi, 1.4])
HOME = np.array([-0.15, 0.0, 0.0, 0.0])

TARGET_XY = np.array([0.25, 0.0])
SUCCESS_RADIUS = 0.03


class PushEnv:
    CTRL_DT = 0.02  # 50 Hz control; physics runs at model timestep (0.002 s)

    def __init__(self, xml_text: str | None = None, seed: int = 0,
                 randomize: bool = True, stale_after: float = 0.2,
                 log_dir: str | Path = LOG_DIR, source_name: str = "unknown"):
        self.xml_text = xml_text if xml_text is not None else SCENE.read_text(encoding="utf-8")
        self.model = mujoco.MjModel.from_xml_string(self.xml_text)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.randomize = randomize
        self.stale_after = stale_after
        self.log_dir = Path(log_dir)
        self.source_name = source_name
        self.n_sub = int(round(self.CTRL_DT / self.model.opt.timestep))

        m = self.model
        self._fq = [m.jnt_qposadr[m.joint(n).id] for n in ("fx", "fy", "fyaw", "fbend")]
        self._bq = m.jnt_qposadr[m.joint("block").id]   # 7: pos(3) + quat(4)
        self._bv = m.jnt_dofadr[m.joint("block").id]    # 6: linvel(3, world) + angvel(3, body)
        self._tip = m.site("tip").id
        self.cmd = HOME.copy()
        self.tick = 0
        self.logger: EpisodeLogger | None = None

    # ---------- episode control ----------
    def reset(self, init_qpos: np.ndarray | None = None) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        if init_qpos is not None:
            self.data.qpos[:] = init_qpos
        else:
            self.data.qpos[self._fq] = HOME
            if self.randomize:
                b = self._bq
                self.data.qpos[b:b + 2] = [0.05, 0.0] + self.rng.uniform(-0.02, 0.02, 2)
                yaw = self.rng.uniform(-0.3, 0.3)
                self.data.qpos[b + 3:b + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.cmd = self.data.qpos[self._fq].copy()
        self.data.ctrl[:] = self.cmd
        mujoco.mj_forward(self.model, self.data)
        self.tick = 0
        self.logger = EpisodeLogger(self.log_dir, {
            "xml": self.xml_text,
            "timestep": self.model.opt.timestep,
            "ctrl_dt": self.CTRL_DT,
            "seed": self.seed,
            "randomize": self.randomize,
            "stale_after": self.stale_after,
            "source": self.source_name,
            "init_qpos": self.data.qpos.tolist(),
            "target_xy": TARGET_XY.tolist(),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        return self.data.qpos.copy()

    def save_episode(self, tag: str = "ep") -> Path | None:
        return self.logger.save(tag) if self.logger else None

    # ---------- stepping ----------
    def apply_control(self, ci: ControlInput) -> bool:
        """Accept a fresh, valid input; otherwise hold the previous command."""
        accepted = bool(ci.valid) and ci.data_age <= self.stale_after
        if accepted:
            self.cmd = np.clip(ci.as_array(), CMD_LOW, CMD_HIGH)
        self.data.ctrl[:] = self.cmd
        return accepted

    def step(self, ci: ControlInput) -> dict:
        """One control tick = n_sub physics steps, every physics step is logged."""
        accepted = self.apply_control(ci)
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
            self._log(ci, accepted)
        self.tick += 1
        return self.state()

    def _log(self, ci: ControlInput, accepted: bool):
        s = self.state()
        self.logger.append(
            t=self.data.time, tick=self.tick, wall=time.monotonic(),
            in_x=ci.x, in_y=ci.y, in_yaw=ci.yaw, in_bend=ci.bend,
            in_timestamp=ci.timestamp, in_data_age=ci.data_age, in_valid=float(ci.valid),
            accepted=float(accepted), cmd=self.cmd,
            finger_q=s["finger_q"], tip_pos=s["tip_pos"],
            block_pos=s["block_pos"], block_quat=s["block_quat"],
            block_linvel=s["block_linvel"], block_angvel=s["block_angvel"],
            qpos=self.data.qpos, qvel=self.data.qvel, ctrl=self.data.ctrl,
        )

    # ---------- state ----------
    def state(self) -> dict:
        d, b, v = self.data, self._bq, self._bv
        quat = d.qpos[b + 3:b + 7].copy()
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        pos = d.qpos[b:b + 3].copy()
        return {
            "t": d.time,
            "finger_q": d.qpos[self._fq].copy(),
            "tip_pos": d.site_xpos[self._tip].copy(),
            "block_pos": pos,
            "block_quat": quat,
            "block_linvel": d.qvel[v:v + 3].copy(),
            "block_angvel": rot.reshape(3, 3) @ d.qvel[v + 3:v + 6],  # world frame
            "dist_to_target": float(np.linalg.norm(pos[:2] - TARGET_XY)),
            "success": bool(np.linalg.norm(pos[:2] - TARGET_XY) < SUCCESS_RADIUS),
        }
