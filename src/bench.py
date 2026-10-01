"""**Single-step decode microbenchmark with a simulated KV/state cache.**

This does **not** feed a real sequence through the model.  It fabricates a cache of a
given length and times one decode step against it, which isolates exactly what a
fixed-size state is meant to change:

* per-step cost of attention as the cache grows,
* the memory-bandwidth wall attention hits,
* the flat cost of an all-linear model.

It does **not** show that the model can ingest 1M tokens, that the state stays
numerically stable across 1M updates, that information 1M tokens back is usable, or
that the positional mechanism reaches 1M.  ``CharLM.pe`` is an ``nn.Embedding(8192)``,
so a genuine stream stops at position 8191 — this benchmark sidesteps that by holding
the position index at 0.  Irrelevant to timing, but it means these numbers say nothing
about long-range quality.  See README "Limitations".

    python src/bench.py --ckpt weights/hybrid_3of4.pt
    python src/bench.py --ckpt weights/all_linear_0of4.pt \\
        --seq-lens 32768 131072 1048576 --batches 1 16 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import device_or_default, env_manifest, results_path, write_run  # noqa: E402
from src.model import load_checkpoint  # noqa: E402

VOCAB_FALLBACK = 63


def fill_cache(model, seq_len: int, batch: int, steps: int, device: str) -> dict:
    """Random KV/state of length ``seq_len``.  Values do not affect timing."""
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
    c["pos"] = 0  # position index does not affect timing; see module docstring
    return c


def cache_bytes(c: dict):
    """``(kv_bytes, fixed_state_bytes)`` at the cache's filled length and batch."""
    kv = st = 0
    for bc in c["blocks"]:
        if "k" in bc:
            kv += 2 * bc["n"] * bc["k"].shape[0] * bc["k"].shape[1] * bc["k"].shape[3] * 4
        else:
            st += (bc["S"].numel() + bc["z"].numel()) * 4
    return kv, st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512, 2048, 8192, 32768])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--vocab", type=int, default=VOCAB_FALLBACK)
    ap.add_argument("--label", default=None, help="name recorded in the CSV")
    ap.add_argument("--out", default="decode_bench.csv", help="CSV name under results/")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    import pandas as pd

    device = device_or_default(args.device)
    model, meta = load_checkpoint(args.ckpt, map_location="cpu")
    model = model.to(device).eval()
    vocab = meta.get("vocab", args.vocab)
    label = args.label or Path(args.ckpt).stem

    print("single-step decode against a SIMULATED cache (not a real 1M-token stream)")
    print(f"{args.ckpt}  mask={meta['mask']}  device={device}")
    print(f"{'seq_len':>9} {'batch':>6} {'ms/step':>9} {'tok/s':>10} "
          f"{'kv(MB)':>9} {'state(KB)':>10}")

    rows = []
    for seq_len in args.seq_lens:
        for batch in args.batches:
            try:
                c = fill_cache(model, seq_len, batch, args.steps, device)
            except torch.cuda.OutOfMemoryError:
                print(f"{seq_len:>9} {batch:>6} {'OOM':>9}")
                rows.append({"variant": label, "seq_len": seq_len, "batch": batch,
                             "ms_per_tok": float("nan"), "tok_per_s": float("nan"),
                             "kv_MB": float("nan"), "state_KB": float("nan"),
                             "oom": True})
                torch.cuda.empty_cache()
                continue

            cur = torch.randint(0, vocab, (batch, 1), device=device)
            with torch.no_grad():
                model.step(cur, c)                      # warmup
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.steps):
                    cur = model.step(cur, c)[:, -1, :].argmax(-1, keepdim=True)
                if device == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0

            kv, st = cache_bytes(c)
            rows.append({"variant": label, "seq_len": seq_len, "batch": batch,
                         "ms_per_tok": dt / args.steps * 1000,
                         "tok_per_s": args.steps * batch / dt,
                         "kv_MB": kv / 1024 ** 2, "state_KB": st / 1024, "oom": False})
            print(f"{seq_len:>9} {batch:>6} {dt / args.steps * 1000:>9.3f} "
                  f"{args.steps * batch / dt:>10,.0f} {kv / 1024**2:>9.1f} {st / 1024:>10.1f}")
            del c, cur
            if device == "cuda":
                torch.cuda.empty_cache()

    if not args.no_write:
        df = pd.DataFrame(rows)
        run = write_run("decode_bench",
                        {"ckpt": args.ckpt, "seq_lens": args.seq_lens,
                         "batches": args.batches, "steps": args.steps,
                         "device": device, "note": "simulated cache"},
                        df, env_manifest())
        df.to_csv(results_path(args.out), index=False)
        print(f"\nwrote {results_path(args.out)}\n      {run}")


if __name__ == "__main__":
    main()
