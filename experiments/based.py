"""Based-style per-layer hybrid: linear global + sliding-window local, shared QKV.

Each layer mixes globally with linear attention (fixed state) and locally with
exact softmax over a small window (w=16/64). No global softmax -> no O(S) KV.

Fair params: shared Wq/Wk/Wv/Wo, only the mixer differs. Same Block/Model as mqar.

Run:
    python -m experiments.based --seeds 0 --steps 500 --layers 2 (smoke)
    python -m experiments.based --seeds 0 1 2 3 4 --steps 2000 --layers 4 (canonical)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.mqar import (HEAD_DIM, N_HEADS, D_MODEL, VOCAB, Block, Model,  # noqa: E402
                              StdAttention, make_batch)
from src.common import device_or_default, env_manifest, results_path, seed_all, write_run  # noqa: E402


class BasedAttention(nn.Module):
    """Shared QKV: global linear (elu+1) + local sliding-window softmax, summed."""

    def __init__(self, window: int = 64):
        super().__init__()
        self.window = window
        self.Wq = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wo = nn.Linear(D_MODEL, D_MODEL, bias=False)

    @staticmethod
    def phi(x):
        return F.elu(x) + 1.0

    def forward(self, x):
        B, S, _ = x.shape
        q0 = self.Wq(x).view(B, S, N_HEADS, HEAD_DIM)
        k0 = self.Wk(x).view(B, S, N_HEADS, HEAD_DIM)
        v0 = self.Wv(x).view(B, S, N_HEADS, HEAD_DIM)
        qf, kf = self.phi(q0), self.phi(k0)

        # global linear, causal prefix-sum (S<=~200 in MQAR, loop is fine)
        Sst = x.new_zeros(B, N_HEADS, HEAD_DIM, HEAD_DIM)
        z = x.new_zeros(B, N_HEADS, HEAD_DIM)
        lin_outs = []
        for t in range(S):
            kt, vt, qt = kf[:, t], v0[:, t], qf[:, t]
            Sst = Sst + torch.einsum("bhm,bhd->bhmd", kt, vt)
            z = z + kt
            num = torch.einsum("bhm,bhmd->bhd", qt, Sst)
            den = torch.einsum("bhm,bhm->bh", qt, z).unsqueeze(-1) + 1e-6
            lin_outs.append(num / den)
        out_lin = torch.stack(lin_outs, 1)

        # local sliding-window softmax, shared QKV
        q = q0.transpose(1, 2)
        k = k0.transpose(1, 2)
        v = v0.transpose(1, 2)
        sc = (q @ k.transpose(-2, -1)) / HEAD_DIM ** 0.5
        S_idx = torch.arange(S, device=x.device)
        causal = S_idx.unsqueeze(1) >= S_idx.unsqueeze(0)
        window = (S_idx.unsqueeze(1) - S_idx.unsqueeze(0)) < self.window
        sc = sc.masked_fill(~(causal & window), float("-inf"))
        out_local = (sc.softmax(-1) @ v).transpose(1, 2)

        return self.Wo((out_lin + out_local).reshape(B, S, D_MODEL))


class SWAOnly(nn.Module):
    """Sliding-window softmax only, shared QKV sizes. Isolates window contribution."""

    def __init__(self, window: int = 64):
        super().__init__()
        self.window = window
        self.Wq = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wo = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.Wq(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.Wk(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.Wv(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        sc = (q @ k.transpose(-2, -1)) / HEAD_DIM ** 0.5
        S_idx = torch.arange(S, device=x.device)
        causal = S_idx.unsqueeze(1) >= S_idx.unsqueeze(0)
        window = (S_idx.unsqueeze(1) - S_idx.unsqueeze(0)) < self.window
        sc = sc.masked_fill(~(causal & window), float("-inf"))
        return self.Wo((sc.softmax(-1) @ v).transpose(1, 2).reshape(B, S, D_MODEL))


VARIANTS = {
    "std": lambda: StdAttention(),
    "swa_w16": lambda: SWAOnly(16),
    "swa_w64": lambda: SWAOnly(64),
    "based_w16": lambda: BasedAttention(16),
    "based_w64": lambda: BasedAttention(64),
}


@torch.no_grad()
def accuracy(model, n_pairs, gen, batches=8, device="cpu"):
    model.eval()
    cor = tot = 0
    for _ in range(batches):
        toks, tgt = make_batch(n_pairs, 64, gen)
        toks, tgt = toks.to(device), tgt.to(device)
        lg = model(toks)
        mask = tgt != -100
        cor += (lg[mask].argmax(-1) == tgt[mask]).sum().item()
        tot += mask.sum().item()
    return cor / tot


def train_one(variant, seed, n_pairs, steps, batch, lr, device, n_layers):
    seed_all(seed)
    gen = torch.Generator().manual_seed(seed + 100)
    model = Model(VARIANTS[variant], n_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    loss = torch.tensor(float("nan"))
    t0 = time.perf_counter()
    for _ in range(steps):
        toks, tgt = make_batch(n_pairs, batch, gen)
        toks, tgt = toks.to(device), tgt.to(device)
        loss = F.cross_entropy(model(toks).reshape(-1, VOCAB), tgt.reshape(-1),
                               ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
    return model, float(loss.item()), time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--eval-pairs", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--out", default="based.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import pandas as pd

    device = device_or_default(args.device)
    print(f"device={device} seeds={args.seeds} variants={args.variants} "
          f"steps={args.steps} layers={args.layers}", flush=True)

    rows = []
    for variant in args.variants:
        for seed in args.seeds:
            model, loss, dt = train_one(variant, seed, args.n_pairs, args.steps,
                                        args.batch, args.lr, device, args.layers)
            g = torch.Generator().manual_seed(999)
            row = {"variant": variant, "seed": seed, "final_loss": loss, "train_s": dt}
            for n in args.eval_pairs:
                row[f"acc_N{n}"] = accuracy(model, n, g, device=device)
            rows.append(row)
            print(f"  {variant:<12} seed={seed} loss={loss:5.3f} " +
                  " ".join(f"N{n}={row[f'acc_N{n}']:.3f}" for n in args.eval_pairs),
                  flush=True)

    df = pd.DataFrame(rows)
    agg = []
    for variant in args.variants:
        sub = df[df.variant == variant]
        r = {"variant": variant, "n_seeds": len(sub)}
        for col in ["final_loss"] + [f"acc_N{n}" for n in args.eval_pairs]:
            r[f"{col}_mean"] = sub[col].mean()
            r[f"{col}_std"] = sub[col].std(ddof=0)
        agg.append(r)
    agg = pd.DataFrame(agg)
    agg.to_csv(results_path(args.out), index=False)
    df.to_csv(results_path(args.out.replace(".csv", "_per_seed.csv")), index=False)
    run = write_run("based",
                    {"seeds": args.seeds, "variants": args.variants, "steps": args.steps,
                     "batch": args.batch, "lr": args.lr, "n_pairs": args.n_pairs,
                     "layers": args.layers, "eval_pairs": args.eval_pairs, "device": device},
                    df, env_manifest())
    print(f"\nwrote {results_path(args.out)} and {run}", flush=True)


if __name__ == "__main__":
    main()
