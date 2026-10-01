"""Multi-query associative recall (MQAR) sweep over sequence-mixer variants.

Task: emit N (key, value) pairs, then N queries; the model must return the value paired
with each queried key.  This is the minimum test for any KV-cache replacement: if it
cannot recall, the memory saving is worthless.

Run::

    python -m experiments.mqar --seeds 0 1 2 3 4
    python -m experiments.mqar --seeds 0 --variants std delta --steps 200

Writes ``results/mqar.csv`` (mean +- std over seeds) plus one versioned run directory.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import device_or_default, env_manifest, results_path, seed_all, write_run  # noqa: E402

K_VOCAB, V_VOCAB = 64, 64
VOCAB = K_VOCAB + V_VOCAB
D_MODEL, N_HEADS, N_LAYERS = 128, 4, 2
HEAD_DIM = D_MODEL // N_HEADS


def make_batch(n_pairs: int, batch: int, gen: torch.Generator):
    keys = torch.stack([torch.randperm(K_VOCAB, generator=gen)[:n_pairs] for _ in range(batch)])
    vals = torch.randint(0, V_VOCAB, (batch, n_pairs), generator=gen)
    toks = torch.empty(batch, 3 * n_pairs, dtype=torch.long)
    tgt = torch.full((batch, 3 * n_pairs), -100, dtype=torch.long)
    for b in range(batch):
        seq = []
        for i in range(n_pairs):
            seq += [keys[b, i].item(), (K_VOCAB + vals[b, i]).item()]
        for j, i in enumerate(torch.randperm(n_pairs, generator=gen)):
            seq.append(keys[b, i].item())
            tgt[b, 2 * n_pairs + j] = K_VOCAB + vals[b, i]
        toks[b] = torch.tensor(seq)
    return toks, tgt


class StdAttention(nn.Module):
    """Causal softmax attention.  Exact recall, quadratic cost, growing KV cache."""

    def __init__(self) -> None:
        super().__init__()
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
        sc = sc.masked_fill(~torch.tril(torch.ones(S, S, dtype=torch.bool, device=x.device)),
                            float("-inf"))
        return self.Wo((sc.softmax(-1) @ v).transpose(1, 2).reshape(B, S, D_MODEL))


class LinearAttention(nn.Module):
    """Additive superposition state ``S = sum phi(k) v^T``, chunked causal.

    ``kind``: ``elu`` / ``relu`` (non-negative features, stable denominator) or
    ``id`` (exact dot product; may produce a near-zero denominator).
    ``state_dim`` > ``HEAD_DIM`` applies a random HD projection first.
    """

    def __init__(self, kind: str = "elu", state_dim: int | None = None, chunk: int = 32):
        super().__init__()
        self.kind = kind
        self.M = state_dim or HEAD_DIM
        self.chunk = chunk
        self.Wq = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wo = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.register_buffer(
            "R",
            torch.randn(N_HEADS, HEAD_DIM, self.M) / HEAD_DIM ** 0.5
            if self.M != HEAD_DIM else None,
        )

    def phi(self, x):
        if self.R is not None:
            x = torch.einsum("bshd,hde->bshe", x, self.R)
        if self.kind == "elu":
            return F.elu(x) + 1.0
        if self.kind == "relu":
            return F.relu(x) + 1e-3
        return x

    def forward(self, x):
        B, S, _ = x.shape
        q = self.phi(self.Wq(x).view(B, S, N_HEADS, HEAD_DIM))
        k = self.phi(self.Wk(x).view(B, S, N_HEADS, HEAD_DIM))
        v = self.Wv(x).view(B, S, N_HEADS, HEAD_DIM)
        M, C = q.shape[-1], self.chunk
        st = x.new_zeros(B, N_HEADS, M, HEAD_DIM)
        z = x.new_zeros(B, N_HEADS, M)
        outs = []
        tril = torch.tril(torch.ones(C, C, dtype=torch.bool, device=x.device))
        for i in range(0, S, C):
            qc, kc, vc = q[:, i:i + C], k[:, i:i + C], v[:, i:i + C]
            c = qc.shape[1]
            inter = torch.einsum("bchm,bhmd->bchd", qc, st)
            zc = torch.einsum("bchm,bhm->bch", qc, z).unsqueeze(-1)
            att = torch.einsum("bihm,bjhm->bhij", qc, kc).masked_fill(~tril[:c, :c], 0.0)
            intra = torch.einsum("bhij,bjhd->bihd", att, vc)
            zi = att.sum(-1).transpose(1, 2).unsqueeze(-1)
            den = (zc + zi).abs() if self.kind == "id" else zc + zi
            outs.append((inter + intra) / (den + 1e-3))
            st = st + torch.einsum("bchm,bchd->bhmd", kc, vc)
            z = z + kc.sum(1)
        return self.Wo(torch.cat(outs, 1).reshape(B, S, D_MODEL))


class DeltaAttention(nn.Module):
    """Mini DeltaNet: an *error-correcting* write instead of additive superposition.

        S_t = S_{t-1} + beta_t * (v_t - S_{t-1}^T k_t) k_t^T

    Same constant-size state ``(N_HEADS, HEAD_DIM, HEAD_DIM)`` as ``LinearAttention``,
    but the update first erases the component already stored at ``k_t``.  This is the
    counter-example to "a fixed-size state cannot do associative recall".
    """

    def __init__(self, normalize: bool = True, **_):
        super().__init__()
        self.normalize = normalize
        self.Wq = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wo = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wb = nn.Linear(D_MODEL, N_HEADS)     # per-head write strength

    def forward(self, x):
        B, S, _ = x.shape
        hd = HEAD_DIM
        q = self.Wq(x).view(B, S, N_HEADS, hd)
        k = self.Wk(x).view(B, S, N_HEADS, hd)
        if self.normalize:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        v = self.Wv(x).view(B, S, N_HEADS, hd)
        beta = torch.sigmoid(self.Wb(x))          # (B,S,H)
        Sst = x.new_zeros(B, N_HEADS, hd, hd)     # S[m,d] = sum k[m] v[d]
        outs = []
        for t in range(S):
            kt, vt, qt, bt = k[:, t], v[:, t], q[:, t], beta[:, t]
            pred = torch.einsum("bhm,bhmd->bhd", kt, Sst)
            err = (vt - pred) * bt.unsqueeze(-1)
            Sst = Sst + torch.einsum("bhm,bhd->bhmd", kt, err)
            outs.append(torch.einsum("bhm,bhmd->bhd", qt, Sst))
        return self.Wo(torch.stack(outs, 1).reshape(B, S, D_MODEL))


class Block(nn.Module):
    """Shared-residual block — see src/model.py for the caveat."""

    def __init__(self, attn):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = attn
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
                                 nn.Linear(4 * D_MODEL, D_MODEL))

    def forward(self, x):
        return x + self.mlp(self.ln2(x + self.attn(self.ln1(x))))


class Model(nn.Module):
    def __init__(self, factory, n_layers: int = N_LAYERS):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList([Block(factory()) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


VARIANTS = {
    "std": lambda: StdAttention(),
    "linear_elu_M32": lambda: LinearAttention("elu", HEAD_DIM),
    "linear_elu_M256": lambda: LinearAttention("elu", 256),
    "linear_relu_M32": lambda: LinearAttention("relu", HEAD_DIM),
    "linear_identity_M32": lambda: LinearAttention("id", HEAD_DIM),
    "delta_rule": lambda: DeltaAttention(normalize=True),
    "delta_rule_nonorm": lambda: DeltaAttention(normalize=False),
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


def train_one(variant, seed, n_pairs, steps, batch, lr, device, n_layers=N_LAYERS):
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
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n-pairs", type=int, default=8, help="pairs seen during training")
    ap.add_argument("--layers", type=int, default=N_LAYERS)
    ap.add_argument("--eval-pairs", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--out", default="mqar.csv", help="aggregated CSV name under results/")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import pandas as pd

    device = device_or_default(args.device)
    print(f"device={device}  seeds={args.seeds}  variants={args.variants}  "
          f"steps={args.steps}  layers={args.layers}")

    per_seed = []
    for variant in args.variants:
        for seed in args.seeds:
            model, loss, dt = train_one(variant, seed, args.n_pairs, args.steps,
                                        args.batch, args.lr, device, args.layers)
            g = torch.Generator().manual_seed(999)
            row = {"variant": variant, "seed": seed, "final_loss": loss, "train_s": dt}
            for n in args.eval_pairs:
                row[f"acc_N{n}"] = accuracy(model, n, g, device=device)
            per_seed.append(row)
            print(f"  {variant:<20} seed={seed}  loss={loss:5.3f}  "
                  + "  ".join(f"N{n}={row[f'acc_N{n}']:.3f}" for n in args.eval_pairs),
                  flush=True)

    df = pd.DataFrame(per_seed)
    agg_rows = []
    for variant in args.variants:
        sub = df[df.variant == variant]
        row = {"variant": variant, "n_seeds": len(sub)}
        for col in ["final_loss"] + [f"acc_N{n}" for n in args.eval_pairs]:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_std"] = sub[col].std(ddof=0)
        row["acc_N8_mean_at_0.9"] = np.nan
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg["recall_at_train_N"] = agg["acc_N8_mean"]
    agg["passes_0.3_at_train_N"] = agg["acc_N8_mean"] >= 0.30

    run = write_run("mqar",
                    {"seeds": args.seeds, "variants": args.variants, "steps": args.steps,
                     "batch": args.batch, "lr": args.lr, "n_pairs": args.n_pairs,
                     "layers": args.layers, "eval_pairs": args.eval_pairs, "device": device},
                    df, env_manifest())
    df.to_csv(results_path(args.out.replace(".csv", "_per_seed.csv")), index=False)
    agg.to_csv(results_path(args.out), index=False)

    print("\n=== mean +- std over seeds ===")
    print(f"{'variant':<20}{'loss':>16}" + "".join(f"{f'N{n}':>16}" for n in args.eval_pairs))
    for r in agg.itertuples():
        line = f"{r.variant:<20}{r.final_loss_mean:>8.3f}+-{r.final_loss_std:<7.3f}"
        for n in args.eval_pairs:
            line += f"{getattr(r, f'acc_N{n}_mean'):>8.3f}+-{getattr(r, f'acc_N{n}_std'):<7.3f}"
        print(line)
    print(f"\nwrote {results_path(args.out)} and {run}")


if __name__ == "__main__":
    main()
