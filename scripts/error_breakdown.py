"""Position error split by what is happening to each block (analysis only; definitions agreed
with Xia on 2026-10-24). Same test clips, starts and 200-step rollouts as eval_wm.py.

(a) ONE-STEP error from the TRUE state (what the training loss sees), category per block per
    step t -> t+1 from MuJoCo truth, in priority order:
      contacted - the block touches the true sim hand (gap < 2 mm) at t+1, or touches another
                  block (gap < 2 mm) at t+1 while either of the two moves > 0.5 mm in that step
      moving    - not contacted, but moves > 0.5 mm in that step
      static    - everything else
(b) ROLLOUT error at steps 10 / 50 / 200, each block grouped by its TRUE history up to that step:
      never_moved             - never moved > 0.5 mm in any step so far
      moved_not_hand_touched  - moved, but the true hand never touched it (gap < 2 mm) so far
      hand_touched            - the true hand touched it at least once so far

  python scripts/error_breakdown.py
Output: runs/error_breakdown_<time>/ report.md, results.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from analyze_overlap import TOUCH, Geo, select_clips
from eval_wm import load_qpos
from handwm.predictors import (ConstantVelocityPredictor, HandStep, StaticPredictor,
                               latest_run_model, make_predictor, rollout)
from handwm.wm_data import load_states
from handwm.wm_model import load_model

ROOT = Path(__file__).resolve().parent.parent
MOVE = 0.0005                   # 0.5 mm per step
CATS = ["static", "moving", "contacted"]
HIST = ["never_moved", "moved_not_hand_touched", "hand_touched"]
METHODS = {"main": "latest:main", "main_onestep": "latest:main:onestep", "action": "latest:action",
           "mlp": "latest:mlp", "static": "static", "const_velocity": "const_velocity"}
STEPS = [10, 50, 200]


def contact_info(geo, qpos, n):
    gaps, hand = geo.truth_context(qpos, n)
    touching = [(a, b) for (a, b), g in gaps.items() if g < TOUCH]
    return touching, hand


def one_step_categories(disp, touching, hand, n):
    """disp: (n,) true displacement over the step; contact info at t+1."""
    moving = disp > MOVE
    cat = np.where(moving, 1, 0)
    for i in hand:
        if i < n:
            cat[i] = 2
    for a, b in touching:
        if moving[a] or moving[b]:
            cat[a] = cat[b] = 2
    return cat


@torch.no_grad()
def one_step_preds(name, spec, d, c):
    """Next-step predictions from the true state for every step of clip c: (T, 8, 3)."""
    T = d["action"].shape[1]
    s = d["blocks"][c, :T].astype(np.float64)
    if name == "static":
        return s[:, :, :3]
    if name == "const_velocity":
        return s[:, :, :3] + s[:, :, 7:10] * 0.05
    path = latest_run_model(spec.split(":")[1], onestep=spec.endswith(":onestep"))
    if (name, path) not in one_step_preds.cache:
        one_step_preds.cache[(name, path)] = load_model(path, "cpu")[0]
    m = one_step_preds.cache[(name, path)]
    h = {"keypoints_t": torch.as_tensor(d["keypoints"][c, :T]),
         "keypoints_t1": torch.as_tensor(d["keypoints"][c, 1:T + 1]),
         "action_t": torch.as_tensor(d["action"][c, :T])}
    out = m(torch.as_tensor(d["blocks"][c, :T]), torch.as_tensor(d["mask"][c]).expand(T, -1), h)
    return out[:, :, :3].double().numpy()


one_step_preds.cache = {}


def main():
    class A:
        data = str(ROOT / "data" / "hand_blocks_v2")
        max_clips = 150
    t0 = time.time()
    torch.set_num_threads(4)
    info = json.loads((Path(A.data) / "dataset_info.json").read_text(encoding="utf-8"))
    print("loading data ...", flush=True)
    d = load_states(A.data, verbose=False)
    clips = select_clips(d, A)
    geo = Geo()
    preds = {name: make_predictor(spec) for name, spec in METHODS.items()}
    H, starts = 200, [0, 56]
    res = {"definitions": __doc__, "splits": {}}
    for split, idx in clips.items():
        q_all = load_qpos(A.data, d["clip_index"][idx], info["per_shard"])
        one = {name: {c: [] for c in CATS} for name in METHODS}
        cat_count = np.zeros(3)
        roll = {name: {k: {h: [] for h in HIST} for k in STEPS} for name in METHODS}
        hist_count = {k: np.zeros(3) for k in STEPS}
        for j, c in enumerate(idx):
            n, mask = int(d["n_blocks"][c]), d["mask"][c]
            q = q_all[int(d["clip_index"][c])]
            T = d["action"].shape[1]
            truep = d["blocks"][c, :, :n, :3].astype(np.float64)
            info_t = [contact_info(geo, q[t], n) for t in range(T + 1)]        # truth contacts per step
            # (a) one step from the true state
            disp = np.linalg.norm(truep[1:] - truep[:-1], axis=-1)            # (T, n)
            cats = np.stack([one_step_categories(disp[t], *info_t[t + 1], n) for t in range(T)])
            cat_count += np.bincount(cats.ravel(), minlength=3)
            for name, spec in METHODS.items():
                p1 = one_step_preds(name, spec, d, c)[:, :n]
                e = np.linalg.norm(p1 - truep[1:], axis=-1) * 1000            # mm
                for ci, cn in enumerate(CATS):
                    one[name][cn] += e[cats == ci].tolist()
            # (b) rollouts, grouped by true history
            kp, hp, act = d["keypoints"][c], d["hand_pose"][c], d["action"][c]
            for s in starts:
                true = d["blocks"][c, s:s + H + 1].astype(np.float64)
                hands = [HandStep(kp[t], kp[t + 1], hp[t], hp[t + 1], act[t]) for t in range(s, s + H)]
                moved = np.zeros(n, bool)
                touched = np.zeros(n, bool)
                hist_at = {}
                for k in range(1, H + 1):
                    moved |= disp[s + k - 1] > MOVE
                    for i in info_t[s + k][1]:
                        if i < n:
                            touched[i] = True
                    if k in STEPS:
                        hist_at[k] = np.where(touched, 2, np.where(moved, 1, 0))
                        hist_count[k] += np.bincount(hist_at[k], minlength=3)
                for name in METHODS:
                    traj = rollout(preds[name], true[0].copy(), mask, hands)
                    for k in STEPS:
                        e = np.linalg.norm(traj[k, :n, :3] - true[k, :n, :3], axis=-1) * 100   # cm
                        for hi, hn in enumerate(HIST):
                            roll[name][k][hn] += e[hist_at[k] == hi].tolist()
            if (j + 1) % 25 == 0:
                print(f"  [{split}] {j + 1}/{len(idx)} clips ({(time.time() - t0) / 60:.1f} min)", flush=True)
        mean = lambda v: float(np.mean(v)) if v else None
        res["splits"][split] = {
            "one_step_share_pct": dict(zip(CATS, (100 * cat_count / cat_count.sum()).round(2).tolist())),
            "one_step_error_mm": {name: {cn: mean(v) for cn, v in cs.items()} for name, cs in one.items()},
            "rollout_share_pct": {f"@{k}": dict(zip(HIST, (100 * hist_count[k] / hist_count[k].sum()).round(2).tolist()))
                                  for k in STEPS},
            "rollout_error_cm": {name: {f"@{k}": {hn: mean(v) for hn, v in ks[k].items()} for k in STEPS}
                                 for name, ks in roll.items()},
        }
    res["runtime_min"] = (time.time() - t0) / 60
    outd = ROOT / "runs" / f"error_breakdown_{time.strftime('%Y%m%d_%H%M%S')}"
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    f = lambda x: "-" if x is None else f"{x:.2f}"
    L = []
    for split, r in res["splits"].items():
        L += [f"\n## {split}", f"(a) one-step error from true state, mm. share of block-steps: {r['one_step_share_pct']}",
              "| method | static | moving | contacted |", "|---|---|---|---|"]
        L += [f"| {m} | " + " | ".join(f(r["one_step_error_mm"][m][c]) for c in CATS) + " |" for m in METHODS]
        L += [f"(b) rollout error, cm. shares: {r['rollout_share_pct']}",
              "| method | " + " | ".join(f"{h} @{k}" for k in STEPS for h in HIST) + " |", "|---|" + "---|" * 9]
        L += [f"| {m} | " + " | ".join(f(r["rollout_error_cm"][m][f'@{k}'][h]) for k in STEPS for h in HIST) + " |"
              for m in METHODS]
    (outd / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nruntime {res['runtime_min']:.1f} min -> {outd}")


if __name__ == "__main__":
    main()
