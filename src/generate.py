"""Sample text from a checkpoint.

    python generate.py --ckpt weights/hybrid_3of4.pt
    python generate.py --ckpt weights/all_linear_0of4.pt --n 300 --temperature 0.9

Note: sampled text differs between CPU and CUDA because `torch.multinomial` draws
from a device-specific RNG stream. The model itself is identical.
"""

from __future__ import annotations

import argparse

import torch

from model import generate, load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="ROMEO: ")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, meta = load_checkpoint(args.ckpt, map_location="cpu")
    model = model.to(args.device).eval()
    itos = {i: c for i, c in enumerate(meta["chars"])}
    print(f"# {args.ckpt}  mask={meta['mask']}  device={args.device}\n")
    print(generate(model, args.prompt, meta["chars"], itos,
                   n_new=args.n, temperature=args.temperature, seed=args.seed))


if __name__ == "__main__":
    main()
