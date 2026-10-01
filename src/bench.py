"""Decode-throughput benchmark: tok/s versus context length and batch size.

The cache is filled directly instead of being prefilled token by token, so the
timing isolates the cost of a *single decode step* at a given context length.
That is the regime where a fixed-size state is supposed to pay off.

Examples
--------
    python bench.py --ckpt weights/hybrid_3of4.pt
    python bench.py --ckpt weights/all_linear_0of4.pt --seq-lens 32768 131072 1048576 --batches 1 16 64
    python bench.py --ckpt weights/all_attn_4of4.pt --batches 16 64 256   # OOM is expected
"""

from __future__ import annotations

import argparse
import time

import torch

from model import cache_bytes, load_checkpoint

VOCAB_FALLBACK = 63


def fill_cache(model, seq_len, batch, steps, device):
    """Random KV/state of length ``seq_len``. Values do not affect timing."""
    c = model.new_cache(maxlen=seq_len + steps + 4, batch=batch)
    for bc in c["blocks"]:
        if "k" in bc:
            shape = bc["k"].shape
            bc["k"] = torch.randn(batch, shape[1], seq_len + steps + 4, shape[3],
                                  device=device) * 0.5
            bc["v"] = torch.randn(batch, shape[1], seq_len + steps + 4, shape[3],
                                  device=device) * 0.5
        else:
            sh = bc["S"].shape
            bc["S"] = torch.randn(batch, sh[1], sh[2], sh[3], device=device) * 0.1
            bc["z"] = torch.randn(batch, sh[1], sh[2], device=device) * 0.1
        bc["n"] = seq_len
    c["pos"] = 0  # position embedding index is irrelevant to timing
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512, 2048, 8192, 32768])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--vocab", type=int, default=VOCAB_FALLBACK)
    args = ap.parse_args()

    model, meta = load_checkpoint(args.ckpt, map_location="cpu")
    model = model.to(args.device).eval()
    vocab = meta.get("vocab", args.vocab)
    print(f"{args.ckpt}  mask={meta['mask']}  device={args.device}")
    print(f"{'seq_len':>9} {'batch':>6} {'ms/step':>9} {'tok/s':>10} "
          f"{'kv(MB)':>9} {'state(KB)':>10}")

    for seq_len in args.seq_lens:
        for batch in args.batches:
            try:
                c = fill_cache(model, seq_len, batch, args.steps, args.device)
            except torch.cuda.OutOfMemoryError:
                kv_need = sum(2 * seq_len * b.attn.new_cache(2)["k"].shape[1]
                              * b.attn.new_cache(2)["k"].shape[3] * 4
                              for b in model.blocks if "k" in b.attn.new_cache(2))
                print(f"{seq_len:>9} {batch:>6} {'OOM':>9} "
                      f"(KV would need ~{batch * kv_need / 1024**3:.0f} GB)")
                torch.cuda.empty_cache()
                continue

            cur = torch.randint(0, vocab, (batch, 1), device=args.device)
            with torch.no_grad():
                model.step(cur, c)                 # warmup
                if args.device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.steps):
                    cur = model.step(cur, c)[:, -1, :].argmax(-1, keepdim=True)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0

            kv, st = cache_bytes(c)
            print(f"{seq_len:>9} {batch:>6} {dt / args.steps * 1000:>9.3f} "
                  f"{args.steps * batch / dt:>10,.0f} {kv / 1024**2:>9.1f} "
                  f"{st / 1024:>10.1f}")
            del c, cur
            if args.device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
