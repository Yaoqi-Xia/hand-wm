"""Train the coordinate world model (design approved by Xia; every number below is a default
that is listed for approval - change any of them from the command line).

  Stage 1: one step at a time, with noise on the inputs (state + hand / action), GNS style:
           the target is the TRUE next state, so the model learns to correct small errors.
           Saved as model_onestep.pt  (= the "one-step only" comparison).
  Stage 2: rollout training - the model feeds its own predictions back in for K steps
           (K = 8, then 16, then 32) and is trained on the whole rollout. Saved as model.pt.

  python scripts/train_wm.py                                  # main model
  python scripts/train_wm.py --arch mlp --tag mlp             # comparison: MLP
  python scripts/train_wm.py --hand action --tag action       # comparison: hand = 9 numbers
  python scripts/train_wm.py --smoke                          # 2-minute code check on any data

Output: runs/train_<time>_<tag>/  model.pt, model_onestep.pt, settings.json, log.csv
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from handwm.wm_data import WindowSampler, excluded_clips, load_states, split_indices, subset
from handwm.wm_model import build_model, canon, change_loss, quat_mul, rotvec_to_quat, save_model

ROOT = Path(__file__).resolve().parent.parent


def to_dev(b, device):
    return {k: torch.as_tensor(v, device=device) for k, v in b.items()}


def hand_at(b, t):
    return {"keypoints_t": b["keypoints"][:, t], "keypoints_t1": b["keypoints"][:, t + 1],
            "action_t": b["action"][:, t]}


def add_noise(s, hand, mask, a):
    """Gaussian input noise (stage 1). Unused block slots stay untouched."""
    m = mask[..., None].float()
    s = s.clone()
    s[..., :3] += torch.randn_like(s[..., :3]) * a.noise_pos * m
    rot = torch.randn_like(s[..., :3]) * a.noise_rot * m
    s[..., 3:7] = canon(quat_mul(rotvec_to_quat(rot), s[..., 3:7]))
    s[..., 7:10] += torch.randn_like(s[..., 7:10]) * a.noise_vel * m
    s[..., 10:13] += torch.randn_like(s[..., 10:13]) * a.noise_angvel * m
    h = dict(hand)
    h["keypoints_t"] = h["keypoints_t"] + torch.randn_like(h["keypoints_t"]) * a.noise_kp
    h["keypoints_t1"] = h["keypoints_t1"] + torch.randn_like(h["keypoints_t1"]) * a.noise_kp
    h["action_t"] = h["action_t"] + torch.randn_like(h["action_t"]) * a.act_noise_sd
    return s, h


@torch.no_grad()
def validate(model, val, device, horizon, max_clips, starts=(0, 100)):
    """Mean block position error (cm) after `horizon` rollout steps on validation clips."""
    model.eval()
    n = min(max_clips, len(val["blocks"]))
    errs = []
    for t0 in starts:
        if t0 + horizon >= val["blocks"].shape[1]:
            continue
        b = to_dev({"blocks": val["blocks"][:n, t0:t0 + horizon + 1],
                    "keypoints": val["keypoints"][:n, t0:t0 + horizon + 1],
                    "action": val["action"][:n, t0:t0 + horizon], "mask": val["mask"][:n]}, device)
        s = b["blocks"][:, 0]
        for t in range(horizon):
            s = model(s, b["mask"], hand_at(b, t))
        e = (s[..., :3] - b["blocks"][:, horizon, :, :3]).norm(dim=-1) * 100
        errs.append((e * b["mask"]).sum().item() / b["mask"].sum().item())
    model.train()
    return float(np.mean(errs))


def lr_at(step, total, base, warmup=500):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "hand_blocks_v2"))
    ap.add_argument("--arch", default="transformer", choices=["transformer", "mlp"])
    ap.add_argument("--hand", default="keypoints", choices=["keypoints", "action"])
    ap.add_argument("--no-kp-index", action="store_true", help="no keypoint-index embedding")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--max-train-blocks", type=int, default=5)
    # stage 1
    ap.add_argument("--stage1-steps", type=int, default=30000)
    ap.add_argument("--batch1", type=int, default=256)
    ap.add_argument("--lr1", type=float, default=3e-4)
    ap.add_argument("--noise-pos", type=float, default=0.001, help="m")
    ap.add_argument("--noise-rot", type=float, default=0.01, help="rad")
    ap.add_argument("--noise-vel", type=float, default=0.02, help="m/s")
    ap.add_argument("--noise-angvel", type=float, default=0.2, help="rad/s")
    ap.add_argument("--noise-kp", type=float, default=0.001, help="m, hand keypoints")
    ap.add_argument("--noise-act-xy", type=float, default=0.002, help="m")
    ap.add_argument("--noise-act-angle", type=float, default=0.02, help="rad (yaw, roll)")
    ap.add_argument("--noise-act-bend", type=float, default=0.02, help="bend units (0-1)")
    # stage 2
    ap.add_argument("--stage2-steps", type=int, default=15000, help="0 = one-step only")
    ap.add_argument("--rollout", default="8,16,32", help="rollout lengths, equal share each")
    ap.add_argument("--batch2", type=int, default=64)
    ap.add_argument("--lr2", type=float, default=1e-4)
    # common
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-horizon", type=int, default=50)
    ap.add_argument("--val-clips", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cuda-graph", action="store_true", help="plain (slower) training loop")
    ap.add_argument("--keep-flagged-clips", action="store_true",
                    help="do NOT exclude the clips listed in deep_contacts.json")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the code works")
    a = ap.parse_args()
    if a.smoke:
        a.stage1_steps, a.stage2_steps, a.val_every, a.val_clips = 300, 300, 200, 20
        a.rollout, a.tag = "4,8", a.tag + "_smoke"

    if sys.platform == "win32":  # keep the PC awake while this runs (released on exit)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a.act_noise_sd = torch.tensor([a.noise_act_xy] * 2 + [a.noise_act_angle] * 2 +
                                  [a.noise_act_bend] * 5, device=device)
    out = ROOT / "runs" / f"train_{time.strftime('%Y%m%d_%H%M%S')}_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}   output: {out}")

    print("loading data ...")
    d = load_states(a.data)
    excl = [] if a.keep_flagged_clips else excluded_clips(a.data)
    sp = split_indices(d, a.max_train_blocks, exclude=excl)
    (out / "excluded_clips.json").write_text(json.dumps({
        "excluded_clip_indices": excl, "source": "deep_contacts.json (unstable + >5 mm contact)",
        "note": "global clip index = shard * per_shard + position in shard"}, indent=1), encoding="utf-8")
    print(f"excluded {len(excl)} flagged clips: {excl}")
    train, val = subset(d, sp["train"]), subset(d, sp["val"])
    sizes = {k: int(len(v)) for k, v in sp.items()}
    print(f"clips: {sizes}")

    model = build_model(dict(arch=a.arch, hand_input=a.hand, **(
        {"kp_index": not a.no_kp_index} if a.arch == "transformer" else {}))).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {model.config}  parameters: {n_params:,}")
    settings = {**{k: v for k, v in vars(a).items() if k != "act_noise_sd"}, "device": device,
                "excluded_clips": excl, "split_sizes": sizes, "parameters": n_params,
                "model_config": model.config}
    (out / "settings.json").write_text(json.dumps(settings, indent=1), encoding="utf-8")

    log = open(out / "log.csv", "w", newline="", encoding="utf-8")
    logw = csv.writer(log)
    logw.writerow(["stage", "step", "rollout_K", "lr", "train_loss", f"val_err_cm_at_{a.val_horizon}"])
    best = float("inf")
    t0 = time.time()

    def train_step_fn(stage, K, b, opt):
        """Forward K steps (own predictions fed back), backward, optimizer step. b = batch dict."""
        s = b["blocks"][:, 0]
        loss = 0.0
        for t in range(K):
            h = hand_at(b, t)
            if stage == 1:
                s, h = add_noise(s, h, b["mask"], a)
            s = model(s, b["mask"], h)
            loss = loss + change_loss(s, b["blocks"][:, t + 1], b["mask"])
        loss = loss / K
        opt.zero_grad(set_to_none=False)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        return loss

    def run_stage(stage, steps, batch, base_lr, ks):
        """Training loop. With CUDA graphs (default on GPU) each K's whole training step is
        recorded once and replayed: same maths, ~5x faster on this laptop (launch overhead)."""
        nonlocal best
        if steps <= 0:
            return
        use_graph = device == "cuda" and not a.no_cuda_graph
        lr_t = torch.tensor(base_lr, device=device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr_t if use_graph else base_lr,
                                weight_decay=a.weight_decay, capturable=use_graph)
        graphs = {}

        def get_graph(K):
            if K not in graphs:
                bufs = to_dev(WindowSampler(train, K, rng).sample(batch), device)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):          # warm-up (allocates optimizer state)
                    for _ in range(3):
                        train_step_fn(stage, K, bufs, opt)
                torch.cuda.current_stream().wait_stream(side)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    loss = train_step_fn(stage, K, bufs, opt)
                graphs[K] = (g, bufs, loss)
            return graphs[K]

        per_k = steps // len(ks)
        run = None
        for i in range(steps):
            K = ks[min(i // max(per_k, 1), len(ks) - 1)]
            lr = lr_at(i, steps, base_lr)
            batch_np = WindowSampler(train, K, rng).sample(batch)
            if use_graph:
                g, bufs, loss_t = get_graph(K)
                lr_t.fill_(lr)
                for k, v in batch_np.items():
                    bufs[k].copy_(torch.as_tensor(v), non_blocking=False)
                g.replay()
            else:
                for gr in opt.param_groups:
                    gr["lr"] = lr
                loss_t = train_step_fn(stage, K, to_dev(batch_np, device), opt)
            last = i == steps - 1
            if (i + 1) % 50 == 0 or last or run is None:
                lv = loss_t.item()
                run = lv if run is None else 0.9 * run + 0.1 * lv
            if (i + 1) % a.val_every == 0 or last:
                v = validate(model, val, device, a.val_horizon, a.val_clips)
                logw.writerow([stage, i + 1, K, f"{lr:.2e}", f"{run:.4f}", f"{v:.3f}"])
                log.flush()
                print(f"[stage {stage}] step {i + 1}/{steps}  K={K}  loss {run:.3f}  "
                      f"val error after {a.val_horizon} steps {v:.2f} cm  ({(time.time() - t0) / 60:.1f} min)",
                      flush=True)
                if stage == 2 and v < best:
                    best = v
                    save_model(model, out / "model.pt", {"val_err_cm": v, "settings": settings})
    print("\n=== stage 1: one-step training with input noise ===")
    run_stage(1, a.stage1_steps, a.batch1, a.lr1, [1])
    save_model(model, out / "model_onestep.pt", {"settings": settings})
    if a.stage2_steps <= 0:
        save_model(model, out / "model.pt", {"settings": settings})
    else:
        print("\n=== stage 2: rollout training ===")
        run_stage(2, a.stage2_steps, a.batch2, a.lr2, [int(k) for k in a.rollout.split(",")])
    log.close()
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
