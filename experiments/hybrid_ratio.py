"""Hybrid ratio sweep: how many layers can be linear before recall breaks?

Reuses the MQAR task and blocks from :mod:`experiments.mqar`.  Each configuration is a
per-layer mask: 1 = full softmax attention, 0 = fixed-state linear attention.

    python -m experiments.hybrid_ratio --seeds 0 1 2 3 4 --layers 4

Writes ``results/hybrid_ratio.csv`` (mean +- std over seeds).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.mqar import (HEAD_DIM, VOCAB, LinearAttention,  # noqa: E402
                              StdAttention, make_batch)
from src.common import device_or_default, env_manifest, results_path, seed_all, write_run  # noqa: E402

CONFIGS = {
    "attn_0of4": [0, 0, 0, 0],
    "attn_1of4_first": [1, 0, 0, 0],
    "attn_1of4_last": [0, 0, 0, 1],
    "attn_2of4": [1, 0, 1, 0],
    "attn_3of4": [1, 1, 1, 0],
    "attn_4of4": [1, 1, 1, 1],
}


def build_model(mask):
    """Model whose layer i is attention iff mask[i] == 1."""
    from torch import nn

    from experiments.mqar import Block

    class MaskedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(VOCAB, 128)
            self.blocks = nn.ModuleList([
                Block(StdAttention() if m else LinearAttention("elu", HEAD_DIM))
                for m in mask])
            self.lnf = nn.LayerNorm(128)
            self.head = nn.Linear(128, VOCAB)

        def forward(self, idx):
            x = self.emb(idx)
            for b in self.blocks:
                x = b(x)
            return self.head(self.lnf(x))

    return MaskedModel()


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


def train_one(mask, seed, steps, batch, lr, device, n_pairs=8):
    seed_all(seed)
    gen = torch.Generator().manual_seed(seed + 100)
    model = build_model(mask).to(device)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
    return model, float(loss.item()), time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-pairs", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = device_or_default(args.device)
    print(f"device={device}  seeds={args.seeds}  configs={args.configs}  steps={args.steps}")

    rows = []
    for name in args.configs:
        mask = CONFIGS[name]
        for seed in args.seeds:
            model, loss, dt = train_one(mask, seed, args.steps, args.batch, args.lr,
                                        device, args.eval_pairs[0])
            g = torch.Generator().manual_seed(999)
            row = {"config": name, "n_attn": sum(mask), "layers": len(mask),
                   "seed": seed, "final_loss": loss, "train_s": dt}
            for n in args.eval_pairs:
                row[f"acc_N{n}"] = accuracy(model, n, g, device=device)
            rows.append(row)
            print(f"  {name:<18} seed={seed}  loss={loss:5.3f}  "
                  + "  ".join(f"N{n}={row[f'acc_N{n}']:.3f}" for n in args.eval_pairs),
                  flush=True)

    df = pd.DataFrame(rows)
    agg = []
    for name in args.configs:
        sub = df[df.config == name]
        r = {"config": name, "n_attn": int(sub.n_attn.iloc[0]),
             "layers": int(sub.layers.iloc[0]), "n_seeds": len(sub),
             "attn_frac": int(sub.n_attn.iloc[0]) / int(sub.layers.iloc[0])}
        for col in ["final_loss"] + [f"acc_N{n}" for n in args.eval_pairs]:
            r[f"{col}_mean"] = sub[col].mean()
            r[f"{col}_std"] = sub[col].std(ddof=0)
        agg.append(r)
    agg = pd.DataFrame(agg)
    agg.to_csv(results_path("hybrid_ratio.csv"), index=False)
    df.to_csv(results_path("hybrid_ratio_per_seed.csv"), index=False)
    run = write_run("hybrid_ratio",
                    {"seeds": args.seeds, "configs": args.configs, "steps": args.steps,
                     "batch": args.batch, "lr": args.lr, "device": device},
                    df, env_manifest())

    print("\n=== mean +- std over seeds ===")
    print(f"{'config':<18}{'attn':>6}{'loss':>16}" + "".join(f"{f'N{n}':>16}" for n in args.eval_pairs))
    for r in agg.itertuples():
        line = f"{r.config:<18}{r.n_attn:>3}/{r.layers:<2}{r.final_loss_mean:>8.3f}+-{r.final_loss_std:<7.3f}"
        for n in args.eval_pairs:
            line += f"{getattr(r, f'acc_N{n}_mean'):>8.3f}+-{getattr(r, f'acc_N{n}_std'):<7.3f}"
        print(line)
    print(f"\nwrote {results_path('hybrid_ratio.csv')} and {run}")


if __name__ == "__main__":
    main()
