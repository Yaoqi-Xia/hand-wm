"""Coordinate (state-based) world model - design approved by Xia on 2026-10-18.

  * Object-level Transformer encoder (4 layers, d=128, 4 heads). No order/position encoding
    across blocks, so any number of blocks works.
  * Block token: position, orientation quaternion, linear + angular velocity, + type flag.
  * Hand tokens ("keypoints", main model): the 21 keypoints of the block-free hand replay,
    each = position now + displacement over the step, + type flag (+ keypoint-index embedding,
    pending approval).
    Comparison "action": the hand as ONE token holding the 9 numbers.
  * Output per block: change over 0.05 s (position delta, rotation delta). Hand tokens
    produce no output. Velocities of the next state = change / 0.05 s.
  * Only the current step is used as input.
  * Comparison model ObjectMLP: all block slots + hand concatenated into one MLP.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

DT = 0.05
# fixed normalization scales (metres, m/s, rad/s); all pending approval
S_POS, S_VEL, S_ANG, S_KPMOVE = 0.25, 0.3, 5.0, 0.02
O_DPOS, O_ROT = 0.01, 0.05      # output scales: position delta (m), rotation delta (rad)


# ---------------- quaternion helpers (torch, w x y z) ----------------
def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


def rotvec_to_quat(r):
    ang = r.norm(dim=-1, keepdim=True)
    k = torch.where(ang > 1e-6, torch.sin(ang / 2) / ang.clamp_min(1e-6), 0.5 - ang ** 2 / 48)
    return torch.cat([torch.cos(ang / 2), r * k], -1)


def canon(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.where(q[..., :1] < 0, -q, q)


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


# ---------------- input features ----------------
def block_features(s):
    """s (B, N, 13): pos, quat, lin vel, ang vel -> normalized (B, N, 13)."""
    return torch.cat([s[..., :3] / S_POS, s[..., 3:7], s[..., 7:10] / S_VEL, s[..., 10:13] / S_ANG], -1)


def keypoint_features(k0, k1):
    """keypoints now / next (B, 21, 3) -> (B, 21, 6): position now, displacement over the step."""
    return torch.cat([k0 / S_POS, (k1 - k0) / S_KPMOVE], -1)


def action_features(act):
    """9 numbers (x, y, yaw, roll, 5 bends) -> (B, 1, 11); angles as sin/cos."""
    return torch.cat([act[:, :2] / S_POS, torch.sin(act[:, 2:4]), torch.cos(act[:, 2:4]),
                      act[:, 4:9]], -1)[:, None]


def apply_change(s, o, mask):
    """o (B, N, 6) normalized change -> next state; velocities = change / DT."""
    dpos, rot = o[..., 0:3] * O_DPOS, o[..., 3:6] * O_ROT
    quat = canon(quat_mul(rotvec_to_quat(rot), s[..., 3:7]))
    nxt = torch.cat([s[..., :3] + dpos, quat, dpos / DT, rot / DT], -1)
    return torch.where(mask[..., None], nxt, s)


class WorldModelBase(nn.Module):
    """hand = dict(keypoints_t, keypoints_t1: (B, 21, 3), action_t: (B, 9))"""

    def __init__(self, hand_input):
        super().__init__()
        assert hand_input in ("keypoints", "action")
        self.hand_input = hand_input

    def hand_features(self, hand):
        if self.hand_input == "keypoints":
            return keypoint_features(hand["keypoints_t"], hand["keypoints_t1"])
        return action_features(hand["action_t"])


class ObjectTransformer(WorldModelBase):
    def __init__(self, hand_input="keypoints", d_model=128, layers=4, heads=4, kp_index=True):
        super().__init__(hand_input)
        self.config = dict(arch="transformer", hand_input=hand_input, d_model=d_model,
                           layers=layers, heads=heads, kp_index=kp_index)
        self.block_in = nn.Linear(13, d_model)
        self.hand_in = nn.Linear(6 if hand_input == "keypoints" else 11, d_model)
        self.type_emb = nn.Embedding(2, d_model)               # type flag: 0 block, 1 hand
        self.kp_id = nn.Embedding(21, d_model) if (kp_index and hand_input == "keypoints") else None
        enc = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=0.0,
                                         activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 6))

    def forward(self, s, mask, hand):
        B, N = s.shape[:2]
        tb = self.block_in(block_features(s)) + self.type_emb.weight[0]
        th = self.hand_in(self.hand_features(hand)) + self.type_emb.weight[1]
        if self.kp_id is not None:
            th = th + self.kp_id.weight[None]
        pad = torch.cat([~mask, torch.zeros(B, th.shape[1], dtype=torch.bool, device=s.device)], 1)
        h = self.encoder(torch.cat([tb, th], 1), src_key_padding_mask=pad)
        return apply_change(s, self.out(h[:, :N]), mask)            # block tokens only


class ObjectMLP(WorldModelBase):
    """Comparison: all 8 block slots (+ an 'active' flag each) and the hand, concatenated."""

    def __init__(self, hand_input="keypoints", hidden=512, layers=4, n_slots=8):
        super().__init__(hand_input)
        self.config = dict(arch="mlp", hand_input=hand_input, hidden=hidden, layers=layers,
                           n_slots=n_slots)
        hand_dim = 21 * 6 if hand_input == "keypoints" else 11
        self.net = mlp(n_slots * 14 + hand_dim, hidden, n_slots * 6, layers)

    def forward(self, s, mask, hand):
        B, N = s.shape[:2]
        fb = torch.cat([block_features(s), mask[..., None].float()], -1) * mask[..., None]
        x = torch.cat([fb.reshape(B, -1), self.hand_features(hand).reshape(B, -1)], -1)
        return apply_change(s, self.net(x).reshape(B, N, 6), mask)


def build_model(config: dict) -> WorldModelBase:
    c = dict(config)
    arch = c.pop("arch")
    return ObjectTransformer(**c) if arch == "transformer" else ObjectMLP(**c)


# ---------------- loss ----------------
def change_loss(pred, true, mask):
    """Squared error of position (in units of 5 mm) + rotation (in units of 0.05 rad), averaged
    over active blocks. Used for both stages (stage 2 averages it over the rollout steps)."""
    m = mask.float()
    lp = ((pred[..., :3] - true[..., :3]) / 0.005).pow(2).sum(-1)
    d = (pred[..., 3:7] * true[..., 3:7]).sum(-1)
    lr = 4 * (1 - d * d).clamp_min(0) / 0.05 ** 2          # ~ (angle / 0.05)^2, smooth at 0
    return ((lp + lr) * m).sum() / m.sum().clamp_min(1)


# ---------------- saving / loading / predictor wrapper ----------------
def save_model(model, path, extra=None):
    torch.save({"config": model.config, "state_dict": model.state_dict(), **(extra or {})}, path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build_model(ck["config"]).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


class ModelPredictor:
    """A trained model in the predictors.Predictor interface (one step at a time).
    Note: block velocities fed in are recomputed as (change over the last step) / DT, the same
    definition used in training, so the first step of a rollout must also use that form."""

    def __init__(self, path, device="cpu"):
        # one scene per step: the CPU is ~10x faster than launching tiny GPU kernels on this laptop
        self.device = device
        if device == "cpu":
            torch.set_num_threads(4)
        self.model, ck = load_model(path, self.device)
        self.name = f"wm_{ck['config']['arch']}_{ck['config']['hand_input']}"

    @torch.no_grad()
    def step(self, blocks, mask, hand):
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)[None]
        h = {"keypoints_t": t(hand.keypoints_t), "keypoints_t1": t(hand.keypoints_t1),
             "action_t": t(hand.action_t)}
        out = self.model(t(blocks), torch.as_tensor(np.asarray(mask), device=self.device)[None], h)
        return out[0].double().cpu().numpy()


def load_predictor(path):
    return ModelPredictor(path)
