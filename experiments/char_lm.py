"""Char-level LM on tiny shakespeare: quality of attention vs linear vs hybrid.

Uses the model definitions in ``src/model.py``, so the checkpoints written here are the
ones shipped in ``weights/``.  Validation loss and perplexity are computed on a held-out
split of the corpus (not a random training batch), and a repetition metric is reported
for sampled text.

Run::

    python tools/fetch_data.py
    python -m experiments.char_lm --seeds 0 1 2 3 4
    python -m experiments.char_lm --seeds 0 --steps 200 --variants hybrid
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import (DATA, WEIGHTS, device_or_default, env_manifest,  # noqa: E402
                        results_path, seed_all, write_run)
from src.model import CharLM, generate  # noqa: E402

BLOCK = 256
PROMPT = "ROMEO: "
N_NEW = 200

VARIANTS = {
    "all_attn_4of4": [1, 1, 1, 1],
    "hybrid_3of4": [1, 1, 1, 0],
    "all_linear_0of4": [0, 0, 0, 0],
}


def load_corpus(path: Path):
    if not path.exists():
        raise SystemExit(f"missing corpus {path} — run: python tools/fetch_data.py")
    text = path.read_text(encoding="utf-8")[:400_000]
    n_val = len(text) // 10
    train_text, val_text = text[:-n_val], text[-n_val:]
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    train = torch.tensor([stoi[c] for c in train_text], dtype=torch.long)
    val = torch.tensor([stoi[c] for c in val_text], dtype=torch.long)
    return chars, stoi, train, val


def batch_from(data: torch.Tensor, bs: int, gen: torch.Generator):
    ix = torch.randint(len(data) - BLOCK - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK + 1] for i in ix])
    return x, y


def train_one(mask, seed, chars, stoi, train, val, steps, bs, lr, device):
    seed_all(seed)
    gen = torch.Generator().manual_seed(seed + 100)
    vocab = len(chars)
    model = CharLM(mask, vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    t0 = time.perf_counter()
    for _ in range(steps):
        x, y = batch_from(train, bs, gen)
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        optimizer_clip(model)
        opt.step()
        sch.step()

    # held-out validation over several fixed batches
    vgen = torch.Generator().manual_seed(999)
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for _ in range(16):
            x, y = batch_from(val, bs, vgen)
            x, y = x.to(device), y.to(device)
            tot += F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item()
    val_loss = tot / 16

    itos = {i: c for i, c in enumerate(chars)}
    torch.manual_seed(seed)          # make sampling deterministic per seed
    sample = generate(model, PROMPT, chars, itos, n_new=N_NEW, seed=seed)
    grams = [sample[i:i + 4] for i in range(len(sample) - 3)]
    return model, val_loss, len(set(grams)) / max(len(grams), 1), sample, \
        time.perf_counter() - t0


def optimizer_clip(model):
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-seed", type=int, default=0,
                    help="which seed's checkpoint is copied into weights/")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    import pandas as pd

    device = device_or_default(args.device)
    chars, stoi, train, val = load_corpus(DATA / "tiny_shakespeare.txt")
    print(f"device={device}  vocab={len(chars)}  train={len(train):,}  val={len(val):,}")
    print(f"seeds={args.seeds}  variants={args.variants}  steps={args.steps}")

    per_seed, samples = [], {}
    for variant in args.variants:
        mask = VARIANTS[variant]
        for seed in args.seeds:
            model, vl, div, sample, dt = train_one(
                mask, seed, chars, stoi, train, val, args.steps, args.batch, args.lr, device)
            per_seed.append({"variant": variant, "seed": seed, "val_loss": vl,
                             "perplexity": math.exp(vl), "ngram4_diversity": div,
                             "train_s": dt})
            samples[(variant, seed)] = sample
            print(f"  {variant:<18} seed={seed}  val={vl:.4f}  ppl={math.exp(vl):7.2f}  "
                  f"div4={div:.3f}  ({dt:.0f}s)", flush=True)

            if not args.no_save and seed == args.save_seed:
                WEIGHTS.mkdir(parents=True, exist_ok=True)
                torch.save({"mask": [int(m) for m in mask],
                            "state_dict": {k: v.half().cpu()
                                           for k, v in model.state_dict().items()},
                            "chars": chars, "vocab": len(chars), "d_model": 128,
                            "n_heads": 4, "seq_len_trained": BLOCK,
                            "val_loss": vl, "seed": seed},
                           WEIGHTS / f"{variant}.pt")
                print(f"      -> saved weights/{variant}.pt", flush=True)

    df = pd.DataFrame(per_seed)
    agg = []
    for variant in args.variants:
        sub = df[df.variant == variant]
        agg.append({
            "variant": variant, "n_seeds": len(sub),
            "val_loss_mean": sub.val_loss.mean(), "val_loss_std": sub.val_loss.std(ddof=0),
            "perplexity_mean": sub.perplexity.mean(),
            "ngram4_diversity_mean": sub.ngram4_diversity.mean(),
            "train_s_mean": sub.train_s.mean(),
        })
    agg = pd.DataFrame(agg)
    agg.to_csv(results_path("char_lm.csv"), index=False)
    df.to_csv(results_path("char_lm_per_seed.csv"), index=False)
    (results_path("char_lm_samples.txt")).write_text(
        "\n\n".join(f"### {v} seed={s} (val={df[(df.variant == v) & (df.seed == s)].val_loss.iloc[0]:.4f})\n{t}"
                    for (v, s), t in samples.items()), encoding="utf-8")

    run = write_run("char_lm",
                    {"seeds": args.seeds, "variants": args.variants, "steps": args.steps,
                     "batch": args.batch, "lr": args.lr, "block": BLOCK,
                     "device": device, "save_seed": args.save_seed},
                    df, env_manifest())

    print("\n=== valid loss / perplexity, mean +- std over seeds ===")
    print(f"{'variant':<18}{'val_loss':>18}{'ppl':>10}{'div4':>8}")
    for r in agg.itertuples():
        print(f"{r.variant:<18}{r.val_loss_mean:>9.4f}+-{r.val_loss_std:<8.4f}"
              f"{r.perplexity_mean:>10.2f}{r.ngram4_diversity_mean:>8.3f}")
    print(f"\nwrote {results_path('char_lm.csv')} and {run}")


if __name__ == "__main__":
    main()
