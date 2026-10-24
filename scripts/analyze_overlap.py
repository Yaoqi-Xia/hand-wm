"""Diagnosis: WHEN and WHY do predicted blocks overlap each other? (analysis only)

Uses exactly the evaluation's setup (eval_wm.py): same test clips (150 per test set, rng seed 0),
starts 0 and 56, 200-step rollouts, hand input = block-free replay, overlap judged with the TRUE
sim hand and MuJoCo collision on the predicted block poses, threshold 5 mm.

Overlap event = a block pair whose overlap first exceeds 5 mm in a rollout (onset step).
Context at onset from MuJoCo truth (same clip, same step):
  hand_recent  - true hand touched either block (gap < 2 mm) in the previous 10 steps
  moving       - either block's true speed > 2 cm/s
  touching     - the two true blocks are within 2 mm of each other
Categories: push_chain (hand_recent and moving), resting_sinking (touching, both at rest, no hand),
            drift (apart in truth, no hand), other.

  python scripts/analyze_overlap.py [--methods main,action,mlp] [--max-clips 150]
Output: runs/overlap_analysis_<time>/ report.md, results.json, examples/*.png
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import numpy as np
import torch

from eval_wm import load_qpos
from handwm.hand_scene import build_hand_xml
from handwm.predictors import HandStep, latest_run_model, rollout
from handwm.wm_data import excluded_clips, load_states, split_indices
from handwm.wm_metrics import Grader
from handwm.wm_model import ModelPredictor, load_model

ROOT = Path(__file__).resolve().parent.parent
PEN = 0.005          # overlap threshold (as in eval)
TOUCH = 0.002        # "touching" in truth
REST = 0.02          # m/s, below = at rest
HAND_WINDOW = 10


class Geo:
    """MuJoCo geometry helper. `pen` model = the grader's (margin 0, exact eval check);
    `near` model = same scene with a 5 mm margin on blocks, to also see gaps of a few mm."""

    def __init__(self):
        xml = build_hand_xml()
        self.g = Grader(xml)
        self.m, self.d = self.g.m, self.g.d
        self.mn = mujoco.MjModel.from_xml_string(xml)
        self.dn = mujoco.MjData(self.mn)
        bg = self.g.block_geom
        self.mn.geom_margin[bg] = 0.005
        self.bq = self.g.bq

    def _set(self, d, qpos, pos, quat, n):
        d.qpos[:] = qpos
        for i in range(n):
            a = self.bq[i]
            d.qpos[a:a + 3] = pos[i]
            q = quat[i]
            d.qpos[a + 3:a + 7] = q / max(np.linalg.norm(q), 1e-8)

    def block_pairs_pen(self, qpos, pos, quat, n):
        """{(i, j): depth} for block pairs overlapping (exact eval geometry)."""
        self._set(self.d, qpos, pos, quat, n)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        out = {}
        for c in self.d.contact[:self.d.ncon]:
            b1, b2 = self.g.geom_to_block.get(int(c.geom1)), self.g.geom_to_block.get(int(c.geom2))
            if b1 is not None and b2 is not None and c.dist < 0:
                k = (min(b1, b2), max(b1, b2))
                out[k] = max(out.get(k, 0.0), -c.dist)
        return out

    def truth_context(self, qpos, n):
        """True state: block-block gaps {(i,j): gap} (only < 5 mm) and set of blocks touched by hand."""
        self.dn.qpos[:] = qpos
        mujoco.mj_kinematics(self.mn, self.dn)
        mujoco.mj_collision(self.mn, self.dn)
        gaps, hand = {}, set()
        for c in self.dn.contact[:self.dn.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = self.g.geom_to_block.get(g1), self.g.geom_to_block.get(g2)
            if b1 is not None and b2 is not None:
                k = (min(b1, b2), max(b1, b2))
                gaps[k] = min(gaps.get(k, 1.0), c.dist)
            elif c.dist < TOUCH and ((b1 is not None and g2 in self.g.hand_geom) or
                                     (b2 is not None and g1 in self.g.hand_geom)):
                hand.add(b1 if b1 is not None else b2)
        return gaps, hand


def select_clips(d, args):
    excl = excluded_clips(args.data)
    sp = split_indices(d, 5, exclude=excl)
    rng = np.random.default_rng(0)  # identical selection to eval_wm.py
    return {s: np.sort(rng.permutation(sp[s])[:args.max_clips]) for s in ("test", "test_ood")}


def analyse(args):
    info = json.loads((Path(args.data) / "dataset_info.json").read_text(encoding="utf-8"))
    print("loading data ...", flush=True)
    d = load_states(args.data, verbose=False)
    clips = select_clips(d, args)
    geo = Geo()
    methods = args.methods.split(",")
    preds = {}
    for m in methods:
        path = latest_run_model(m)
        preds[m] = ModelPredictor(path)
        print(f"{m}: {path}", flush=True)
    starts, H = [0, 56], 200
    res = {"settings": vars(args), "splits": {}}
    examples = []
    t0 = time.time()
    for split, idx in clips.items():
        qall = load_qpos(args.data, d["clip_index"][idx], info["per_shard"])
        R = {m: {"events": [], "rollouts": 0, "rollouts_with_event": 0, "by_gap": defaultdict(lambda: [0, 0])}
             for m in methods}
        for j, c in enumerate(idx):
            n, mask = int(d["n_blocks"][c]), d["mask"][c]
            q = qall[int(d["clip_index"][c])]
            kp, hp, act = d["keypoints"][c], d["hand_pose"][c], d["action"][c]
            for s in starts:
                true = d["blocks"][c, s:s + H + 1].astype(np.float64)
                hands = [HandStep(kp[t], kp[t + 1], hp[t], hp[t + 1], act[t]) for t in range(s, s + H)]
                # truth context for every step (shared by all methods)
                ctx = [geo.truth_context(q[s + k], n) for k in range(H + 1)]
                p0 = true[0, :n, :2]
                dmin = min((np.linalg.norm(p0[a] - p0[b]) for a in range(n) for b in range(a + 1, n)), default=np.inf)
                gbin = ("<7cm" if dmin < 0.07 else "7-10cm" if dmin < 0.10 else "10-15cm" if dmin < 0.15
                        else ">15cm" if np.isfinite(dmin) else "1 block")
                for m in methods:
                    traj = rollout(preds[m], true[0].copy(), mask, hands)
                    first, maxd = {}, defaultdict(float)
                    for k in range(1, H + 1):
                        for pair, dep in geo.block_pairs_pen(q[s + k], traj[k, :, :3], traj[k, :, 3:7], n).items():
                            maxd[pair] = max(maxd[pair], dep)
                            if dep > PEN and pair not in first:
                                first[pair] = (k, dep)
                    r = R[m]
                    r["rollouts"] += 1
                    r["rollouts_with_event"] += bool(first)
                    r["by_gap"][gbin][0] += 1
                    r["by_gap"][gbin][1] += len(first)
                    for (a, b), (k, dep) in first.items():
                        ev = classify(k, a, b, true, traj, ctx)
                        ev.update(split=split, clip=int(d["clip_index"][c]), start=s, pair=[a, b], onset=k,
                                  depth_onset_mm=dep * 1000, depth_max_mm=maxd[(a, b)] * 1000, n_blocks=n)
                        r["events"].append(ev)
                        if m == "main" and len(examples) < 400:
                            examples.append((ev, q[s + k].copy(), traj[k].copy(), true[k].copy(), n))
            if (j + 1) % 25 == 0:
                print(f"  [{split}] {j + 1}/{len(idx)} clips ({(time.time() - t0) / 60:.1f} min)", flush=True)
        res["splits"][split] = {m: summarise(R[m]) for m in methods}
        res["splits"][split]["_raw"] = {m: R[m]["events"] for m in methods}
    print("one-step check ...", flush=True)
    res["one_step"] = one_step_check(d, clips, geo, args, info)
    return res, examples


def classify(k, a, b, true, traj, ctx):
    gaps, _ = ctx[k]
    hand_recent = any(({a, b} & ctx[t][1]) for t in range(max(0, k - HAND_WINDOW), k + 1))
    va, vb = np.linalg.norm(true[k, a, 7:10]), np.linalg.norm(true[k, b, 7:10])
    moving = max(va, vb) > REST
    gap = gaps.get((a, b), 1.0)
    touching = gap < TOUCH
    if hand_recent and moving:
        cat = "push_chain"
    elif touching and not moving and not hand_recent:
        cat = "resting_sinking"
    elif not touching and not hand_recent:
        cat = "drift"
    else:
        cat = "other"
    ev = dict(category=cat, hand_recent=bool(hand_recent), moving=bool(moving), touching_truth=bool(touching),
              truth_gap_mm=float(min(gap, 0.005) * 1000) if gap < 1 else None,
              speed_a=float(va), speed_b=float(vb))
    if cat == "push_chain":
        # pushed block = the one the true hand touched (if both, the one that moved more in truth)
        touched = set().union(*[ctx[t][1] for t in range(max(0, k - HAND_WINDOW), k + 1)]) & {a, b}
        if len(touched) == 1:
            A = touched.pop()
        else:
            A = a if np.linalg.norm(true[k, a, :2] - true[max(0, k - HAND_WINDOW), a, :2]) >= \
                np.linalg.norm(true[k, b, :2] - true[max(0, k - HAND_WINDOW), b, :2]) else b
        B = b if A == a else a
        # contact start: first step (looking back from onset) of the current truth contact between A and B
        sc = k
        while sc > 0 and ctx[sc - 1][0].get((a, b), 1.0) < TOUCH:
            sc -= 1
        if sc == k and ctx[k][0].get((a, b), 1.0) >= TOUCH:
            sc = max(0, k - HAND_WINDOW)          # never touching in truth: use the hand window
        disp = lambda x, i: float(np.linalg.norm(x[k, i, :2] - x[sc, i, :2]))
        ev.update(pushed=int(A), hit=int(B), contact_start=int(sc),
                  pushed_pred_cm=disp(traj, A) * 100, pushed_true_cm=disp(true, A) * 100,
                  hit_pred_cm=disp(traj, B) * 100, hit_true_cm=disp(true, B) * 100)
    return ev


def summarise(r):
    ev = r["events"]
    cats = Counter(e["category"] for e in ev)
    on = np.array([e["onset"] for e in ev]) if ev else np.zeros(0)
    dep = np.array([e["depth_max_mm"] for e in ev]) if ev else np.zeros(0)
    out = {
        "rollouts": r["rollouts"], "events": len(ev),
        "events_per_rollout": len(ev) / max(r["rollouts"], 1),
        "rollouts_with_event_pct": 100 * r["rollouts_with_event"] / max(r["rollouts"], 1),
        "onset_pct": {lab: float(100 * ((on >= lo) & (on <= hi)).mean()) if len(on) else 0.0
                      for lab, lo, hi in [("1-10", 1, 10), ("11-50", 11, 50), ("51-100", 51, 100), ("101-200", 101, 200)]},
        "onset_median": float(np.median(on)) if len(on) else None,
        "depth_max_pct": {lab: float(100 * ((dep > lo) & (dep <= hi)).mean()) if len(dep) else 0.0
                          for lab, lo, hi in [("5-10mm", 5, 10), ("10-20mm", 10, 20), ("20-40mm", 20, 40), (">40mm", 40, 1e9)]},
        "category_pct": {c: 100 * cats[c] / max(len(ev), 1) for c in ["push_chain", "resting_sinking", "drift", "other"]},
        "by_initial_nn_gap": {g: {"rollouts": v[0], "events_per_rollout": v[1] / max(v[0], 1)}
                              for g, v in sorted(r["by_gap"].items())},
    }
    pc = [e for e in ev if e["category"] == "push_chain"]
    if pc:
        rp = np.array([e["pushed_pred_cm"] - e["pushed_true_cm"] for e in pc])
        rh = np.array([e["hit_pred_cm"] - e["hit_true_cm"] for e in pc])
        out["push_chain"] = {
            "n": len(pc),
            "pushed_pred_minus_true_cm_median": float(np.median(rp)),
            "hit_pred_minus_true_cm_median": float(np.median(rh)),
            "pct_hit_moves_too_little(>0.5cm short)": float(100 * (rh < -0.5).mean()),
            "pct_pushed_moves_too_far(>0.5cm over)": float(100 * (rp > 0.5).mean()),
            "pct_both": float(100 * ((rh < -0.5) & (rp > 0.5)).mean()),
        }
    return out


@torch.no_grad()
def one_step_check(d, clips, geo, args, info):
    """From the TRUE state at every step, predict one step; how often is a block pair > 5 mm?"""
    model, _ = load_model(latest_run_model("main"), "cpu")
    torch.set_num_threads(4)
    out = {}
    for split, idx in clips.items():
        qall = load_qpos(args.data, d["clip_index"][idx], info["per_shard"])
        steps = pen_steps = coll = pen_coll = 0
        for c in idx:
            n, q = int(d["n_blocks"][c]), qall[int(d["clip_index"][c])]
            T = d["action"].shape[1]
            s = torch.as_tensor(d["blocks"][c, :T])
            h = {"keypoints_t": torch.as_tensor(d["keypoints"][c, :T]),
                 "keypoints_t1": torch.as_tensor(d["keypoints"][c, 1:T + 1]),
                 "action_t": torch.as_tensor(d["action"][c, :T])}
            mask = torch.as_tensor(d["mask"][c]).expand(T, -1)
            nxt = model(s, mask, h).double().numpy()
            for t in range(T):
                gaps, _ = geo.truth_context(q[t + 1], n)
                in_coll = any(g < TOUCH for g in gaps.values())
                pen = any(v > PEN for v in geo.block_pairs_pen(q[t + 1], nxt[t, :, :3], nxt[t, :, 3:7], n).values())
                steps += 1
                pen_steps += pen
                coll += in_coll
                pen_coll += pen and in_coll
        out[split] = {"steps": steps, "pct_steps_with_overlap": 100 * pen_steps / steps,
                      "collision_frames": coll, "pct_collision_frames_with_overlap": 100 * pen_coll / max(coll, 1)}
    return out


def render_examples(examples, out):
    """Side by side at the overlap moment: model (left) vs MuJoCo truth (right), top view."""
    import cv2
    exd = out / "examples"
    exd.mkdir(exist_ok=True)
    m = mujoco.MjModel.from_xml_string(build_hand_xml())
    dd = mujoco.MjData(m)
    ren = mujoco.Renderer(m, 400, 400)
    bq = [m.joint(f"block{i}").qposadr[0] for i in range(8)]
    cats = Counter(e[0]["category"] for e in examples)
    if not cats:
        return [], None
    top = cats.most_common(1)[0][0]
    chosen = sorted([e for e in examples if e[0]["category"] == top], key=lambda e: -e[0]["depth_onset_mm"])
    picks, seen = [], set()
    for e in chosen:
        key = (e[0]["clip"], e[0]["start"])
        if key not in seen:
            picks.append(e)
            seen.add(key)
        if len(picks) == 4:
            break
    paths = []
    for i, (ev, q, pred, true, n) in enumerate(picks):
        imgs = []
        for label, st in (("world model", pred), ("MuJoCo truth", true)):
            dd.qpos[:] = q
            for b in range(n):
                dd.qpos[bq[b]:bq[b] + 3] = st[b, :3]
                dd.qpos[bq[b] + 3:bq[b] + 7] = st[b, 3:7]
            mujoco.mj_kinematics(m, dd)
            mujoco.mj_camlight(m, dd)
            ren.update_scene(dd, camera="top")
            img = cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            imgs.append(img)
        frame = np.hstack(imgs)
        cv2.putText(frame, f"{ev['category']}  clip {ev['clip']} start {ev['start']}  step {ev['onset']}  "
                    f"blocks {ev['pair']} overlap {ev['depth_onset_mm']:.1f} mm", (8, 392),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        p = exd / f"example_{i + 1}_{ev['category']}.png"
        cv2.imwrite(str(p), frame)
        paths.append(str(p))
    return paths, top


def report(res, out):
    L = ["# Block-block overlap diagnosis\n"]
    for split, ms in res["splits"].items():
        L.append(f"\n## {split}\n")
        for m, s in ms.items():
            if m.startswith("_"):
                continue
            L.append(f"### {m}: {s['events']} events in {s['rollouts']} rollouts "
                     f"({s['events_per_rollout']:.2f}/rollout, {s['rollouts_with_event_pct']:.1f}% of rollouts)")
            L.append(f"- onset step %: {s['onset_pct']}  (median {s['onset_median']})")
            L.append(f"- max depth %: {s['depth_max_pct']}")
            L.append(f"- category %: {s['category_pct']}")
            if "push_chain" in s:
                L.append(f"- push chain: {s['push_chain']}")
            L.append(f"- by initial nearest-neighbour distance: {s['by_initial_nn_gap']}")
    L.append(f"\n## one-step check (main model, from true states)\n{res['one_step']}")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "hand_blocks_v2"))
    ap.add_argument("--methods", default="main,action,mlp")
    ap.add_argument("--max-clips", type=int, default=150)
    args = ap.parse_args()
    t0 = time.time()
    out = ROOT / "runs" / f"overlap_analysis_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    res, examples = analyse(args)
    paths, top = render_examples(examples, out)
    res["examples"] = {"category": top, "paths": paths}
    res["runtime_min"] = (time.time() - t0) / 60
    (out / "results.json").write_text(json.dumps(res, default=float), encoding="utf-8")
    report(res, out)
    print((out / "report.md").read_text(encoding="utf-8"))
    print(f"\nexamples: {paths}\nruntime {res['runtime_min']:.1f} min -> {out}")


if __name__ == "__main__":
    main()
