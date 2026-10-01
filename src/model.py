"""Standalone model definitions for the char-LM checkpoints in ``weights/``.

These mirror the classes in ``src/notebook.py`` exactly, so ``state_dict`` keys
line up.  Nothing here imports marimo.

The only real content is:

* ``StdAttention``     - causal softmax attention, incremental decode with a KV cache
* ``LinearAttention``  - causal linear attention with a fixed-size matrix state
* ``CharLM``           - 4-layer char LM that mixes the two (``mask`` selects which)

Layer-residual detail worth knowing: the block computes

    z = x + attn(ln1(x))
    out = x + mlp(ln2(z))

i.e. the MLP reads the post-attention value but the residual is the *pre*-attention
``x``.  ``step()`` must mirror that exactly, otherwise incremental decoding silently
diverges from the training-time forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 128
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS
CHUNK = 64


class StdAttention(nn.Module):
    """Causal softmax attention.  Decode cost and KV cache grow linearly in S."""

    def __init__(self) -> None:
        super().__init__()
        self.Wq = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wo = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.Wq(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.Wk(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.Wv(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        sc = (q @ k.transpose(-2, -1)) / HEAD_DIM ** 0.5
        sc = sc.masked_fill(~torch.tril(torch.ones(S, S, dtype=torch.bool, device=x.device)),
                            float("-inf"))
        return self.Wo((sc.softmax(-1) @ v).transpose(1, 2).reshape(B, S, D_MODEL))

    def new_cache(self, maxlen: int, batch: int = 1) -> dict:
        dev = next(self.parameters()).device
        return {"k": torch.zeros(batch, N_HEADS, maxlen, HEAD_DIM, device=dev),
                "v": torch.zeros(batch, N_HEADS, maxlen, HEAD_DIM, device=dev), "n": 0}

    def step(self, xt: torch.Tensor, c: dict) -> torch.Tensor:
        B = xt.shape[0]
        q = self.Wq(xt).view(B, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.Wk(xt).view(B, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.Wv(xt).view(B, 1, N_HEADS, HEAD_DIM).transpose(1, 2)
        n = c["n"]
        c["k"][:, :, n:n + 1] = k
        c["v"][:, :, n:n + 1] = v
        c["n"] = n + 1
        kk, vv = c["k"][:, :, :n + 1], c["v"][:, :, :n + 1]
        sc = (q @ kk.transpose(-2, -1)) / HEAD_DIM ** 0.5
        return self.Wo((sc.softmax(-1) @ vv).transpose(1, 2).reshape(B, 1, D_MODEL))


class LinearAttention(nn.Module):
    """Causal linear attention.  State is ``(N_HEADS, HEAD_DIM, HEAD_DIM)`` - constant in S."""

    def __init__(self) -> None:
        super().__init__()
        self.Wq = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wo = nn.Linear(D_MODEL, D_MODEL, bias=False)

    @staticmethod
    def phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.phi(self.Wq(x).view(B, S, N_HEADS, HEAD_DIM))
        k = self.phi(self.Wk(x).view(B, S, N_HEADS, HEAD_DIM))
        v = self.Wv(x).view(B, S, N_HEADS, HEAD_DIM)
        st = x.new_zeros(B, N_HEADS, HEAD_DIM, HEAD_DIM)
        z = x.new_zeros(B, N_HEADS, HEAD_DIM)
        outs = []
        tril = torch.tril(torch.ones(CHUNK, CHUNK, dtype=torch.bool, device=x.device))
        for i in range(0, S, CHUNK):
            qc, kc, vc = q[:, i:i + CHUNK], k[:, i:i + CHUNK], v[:, i:i + CHUNK]
            c = qc.shape[1]
            inter = torch.einsum("bchm,bhmd->bchd", qc, st)
            zc = torch.einsum("bchm,bhm->bch", qc, z).unsqueeze(-1)
            att = torch.einsum("bihm,bjhm->bhij", qc, kc).masked_fill(~tril[:c, :c], 0.0)
            intra = torch.einsum("bhij,bjhd->bihd", att, vc)
            zi = att.sum(-1).transpose(1, 2).unsqueeze(-1)
            outs.append((inter + intra) / (zc + zi + 1e-3))
            st = st + torch.einsum("bchm,bchd->bhmd", kc, vc)
            z = z + kc.sum(1)
        return self.Wo(torch.cat(outs, 1).reshape(B, S, D_MODEL))

    def new_cache(self, maxlen: int, batch: int = 1) -> dict:
        dev = next(self.parameters()).device
        return {"S": torch.zeros(batch, N_HEADS, HEAD_DIM, HEAD_DIM, device=dev),
                "z": torch.zeros(batch, N_HEADS, HEAD_DIM, device=dev), "n": 0}

    def step(self, xt: torch.Tensor, c: dict) -> torch.Tensor:
        B = xt.shape[0]
        q = self.phi(self.Wq(xt)).view(B, N_HEADS, HEAD_DIM)
        k = self.phi(self.Wk(xt)).view(B, N_HEADS, HEAD_DIM)
        v = self.Wv(xt).view(B, N_HEADS, HEAD_DIM)
        c["S"] = c["S"] + torch.einsum("bhm,bhd->bhmd", k, v)
        c["z"] = c["z"] + k
        c["n"] += 1
        num = torch.einsum("bhm,bhmd->bhd", q, c["S"])
        den = torch.einsum("bhm,bhm->bh", q, c["z"]).unsqueeze(-1) + 1e-3
        return self.Wo((num / den).reshape(B, 1, D_MODEL))


class Block(nn.Module):
    """Pre-norm block with a **shared-residual read** for both sublayers::

        z   = x + attn(ln1(x))
        out = x + mlp(ln2(z))

    This is *not* the standard Transformer block, which would be
    ``out = z + mlp(ln2(z))``.  Every variant in this repo uses this same block, so
    comparisons between variants are internally fair, but it is not a drop-in
    Transformer baseline.  Changing the residual would invalidate the shipped
    checkpoints, so it is documented rather than changed; see README "Limitations".
    """

    def __init__(self, is_attn: bool) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = StdAttention() if is_attn else LinearAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
                                 nn.Linear(4 * D_MODEL, D_MODEL))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.ln2(x + self.attn(self.ln1(x))))

    def step(self, x: torch.Tensor, c: dict) -> torch.Tensor:
        z = x + self.attn.step(self.ln1(x), c)
        return x + self.mlp(self.ln2(z))


class CharLM(nn.Module):
    """``mask`` is a list of 0/1, one per layer; 1 = full attention, 0 = linear."""

    def __init__(self, mask, vocab: int) -> None:
        super().__init__()
        self.mask = [int(m) for m in mask]
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.pe = nn.Embedding(8192, D_MODEL)
        self.blocks = nn.ModuleList([Block(bool(m)) for m in mask])
        self.lnf = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        S = idx.shape[1]
        x = self.emb(idx) + self.pe(torch.arange(S, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))

    def new_cache(self, maxlen: int = 8192, batch: int = 1) -> dict:
        return {"blocks": [b.attn.new_cache(maxlen, batch) for b in self.blocks], "pos": 0}

    def step(self, tok: torch.Tensor, c: dict) -> torch.Tensor:
        x = self.emb(tok) + self.pe(torch.tensor([c["pos"]], device=tok.device))
        for b, bc in zip(self.blocks, c["blocks"], strict=True):
            x = b.step(x, bc)
        c["pos"] += 1
        return self.head(self.lnf(x))


def cache_bytes(c: dict):
    """``(kv_bytes, fixed_state_bytes)`` for a decode cache, at its filled length.

    ``kv_bytes`` grows linearly with context and is zero for an all-linear model;
    ``fixed_state_bytes`` does not grow with context at all.
    """
    kv = st = 0
    for bc in c["blocks"]:
        if "k" in bc:
            kv += 2 * bc["n"] * bc["k"].shape[0] * bc["k"].shape[1] * bc["k"].shape[3] * 4
        else:
            st += (bc["S"].numel() + bc["z"].numel()) * 4
    return kv, st


def load_checkpoint(path, map_location="cpu", dtype=torch.float32):
    """Returns ``(model, meta)``.  ``meta`` holds the char vocabulary and loss.

    Checkpoints are stored in fp16 to halve the download; they are upcast to
    ``dtype`` (fp32 by default) on load, which is the precision they were trained in.
    Pass ``dtype=None`` to keep the stored precision.
    """
    ck = torch.load(path, map_location=map_location, weights_only=True)
    sd = ck["state_dict"]
    if dtype is not None:
        sd = {k: v.to(dtype) for k, v in sd.items()}
    model = CharLM(ck["mask"], ck["vocab"])
    model.load_state_dict(sd)
    model.eval()
    return model, ck


@torch.no_grad()
def generate(model, prompt: str, chars, itos, n_new: int = 200,
             temperature: float = 0.8, seed: int = 0):
    """Greedy-free sampling through the incremental decode path."""
    stoi = {c: i for i, c in enumerate(chars)}
    gen = None
    dev = next(model.parameters()).device
    idx = torch.tensor([[stoi[c] for c in prompt]], device=dev)
    c = model.new_cache(maxlen=idx.shape[1] + n_new + 8)
    for i in range(idx.shape[1] - 1):
        model.step(idx[:, i:i + 1], c)
    out = list(prompt)
    for _ in range(n_new):
        logits = model.step(idx[:, -1:], c)[:, -1, :] / temperature
        if gen is None:
            gen = torch.Generator(device=logits.device).manual_seed(seed)
        nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=gen)
        out.append(itos[int(nxt)])
        idx = torch.cat([idx, nxt], 1)
    return "".join(out)
