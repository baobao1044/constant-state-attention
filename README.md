# constant-state-attention

An audit-and-fix of a "HDC-Attention" notebook (hyperdimensional-computing attention as a
KV-cache replacement) that turned into a measurement of **what a fixed-size attention state
can and cannot do**.

The short version:

- The original design **could not run at all**, and never could — not for any hyperparameter.
  Its memory is a **vector**, and a vector has **rank 1**.
- Replacing it with a **matrix state** (`Σ φ(k) vᵀ`) makes it causal, `O(S)` in time, and
  **constant-memory** — and it runs **1,048,576-token context on one 96 GB GPU**.
- But a fixed-size state **cannot do associative recall**. On MQAR it scores **0.16 vs 0.86**
  for softmax attention. Increasing the "HD" dimension does not help at all.
- A **hybrid** (full attention on 3 of 4 layers, fixed-state linear attention on the rest)
  matches the all-attention model's loss **to within 0.004** while cutting KV memory 25%.

> ### Status: this is an analysis, not a new architecture
>
> The hybrid here is an instance of a well-established family
> ([Based](https://arxiv.org/abs/2402.18668), Jamba, MiniMax-01, Mamba-2-Hybrid, RecurrentGemma,
> StripedHyena, Qwen3-Next). The linear layer is standard
> [linear attention](https://arxiv.org/abs/2006.16236). **The "HD" part of "HDC-Attention"
> contributes nothing measurable** — see [the flat-kernel result](#the-hd-dimension-buys-nothing).
> What is original here is the **measurements and the corrections**, not the mechanism.

---

## Headline numbers

### 1. Decode throughput at 1,048,576 tokens — one RTX PRO 6000 (96 GB)

| model | B=1 tok/s | B=16 | B=64 | B=256 | state @1M |
|---|---|---|---|---|---|
| all-attention (4/4) | 136 | 263 | **OOM** (~256 GB KV) | OOM | 4,096 MB, grows with S |
| hybrid (3/4) | 185 | 350 | **OOM** (~192 GB) | OOM | 3,072 MB |
| **all-linear (0/4)** | **1,048** | 16,747 | 60,870 | **243,579** | **66 KB — constant** |

- **all-linear is flat**: 1,009 tok/s at 32 K → 1,057 tok/s at 1 M. The state does not grow.
- **all-attention saturates at ~263 tok/s** no matter the batch. It is memory-bandwidth bound:
  every token re-reads the whole KV cache. Then it OOMs.
- The original design's 1 M requirement was **1.43 TB** of intermediate tensors
  (cell `emfo` computed this and concluded "cannot run"). The fixed design needs **66 KB**.

### 2. The flat-kernel result — why the HD dimension buys nothing

Reading a superposed memory as `r = Σ_t κ(q,k_t)·v_t` is a **convex combination** of the stored
values. To return `v_i` exactly, `κ` must be a near-**delta**. Measured `κ` over 400 keys:

| feature map | M | `κ(self)` | `κ(off-diagonal)` |
|---|---|---|---|
| `ELU(Rx)+1` | 8,192 | 8,370 | **+8,253** ← essentially constant |
| random Fourier features, σ=4 | 8,192 | 0.502 | **+0.472** ← still constant |
| **identity** (exact dot product) | — | 1.000 | **+0.0002** ← a real delta |

A non-linear random projection *flattens* the kernel, so all keys look alike and the memory
cannot distinguish items. A *linear* random projection is an isometry (Johnson–Lindenstrauss)
and preserves the dot product that was already there. Either way **the HD dimension is dead
weight.**

Direct confirmation on MQAR (multi-query associative recall, trained, 2000 steps):

| variant | loss | N=8 | N=16 | N=32 |
|---|---|---|---|---|
| softmax attention | **0.159** | **0.950** | 0.684 | 0.353 |
| linear, M=32 | 2.024 | 0.169 | 0.125 | 0.077 |
| linear, **M=256 (HD 8×)** | 1.989 | 0.190 | 0.115 | 0.075 |
| linear, `relu` | 2.021 | 0.171 | 0.125 | 0.079 |
| linear, identity kernel | 4.159 | 0.016 | 0.016 | 0.015 |

`M=256` is **not** better than `M=32`. Random baseline is 0.016.

### 3. Hybrid ratio on MQAR (4 layers, 2 seeds)

| attention layers | N=8 accuracy | KV @8 K |
|---|---|---|
| 0/4 | 0.173 ± 0.003 | 0 MB |
| 1/4 | 0.333 ± 0.082 | 8 MB |
| 3/4 | **0.834 ± 0.014** | 24 MB |
| 4/4 | 0.821 ± 0.004 | 32 MB |

3/4 matches 4/4 (difference 0.013, smaller than the seed spread) at 25% less KV.
**Placement matters**: `1/4` with attention *first* gives 0.333; with attention *last* it gives
0.172 — no better than no attention at all.

### 4. Char-LM on tiny shakespeare (4 layers, `d_model=128`, 1200 steps)

| variant | val loss | non-finite tokens | 4-gram diversity |
|---|---|---|---|
| all-attention | 1.335 | **0** | 0.917 |
| **hybrid (3/4)** | **1.339** | **0** | **0.961** |
| all-linear | 1.655 | **0** | 0.951 |

```
hybrid (3/4) — val 1.339
ROMEO: you to for I,
On your to be the brother, and Gaunt strel:
We tell first with divined he love safe,
He hate father and worue cherity to me blood
```

---

## What was wrong with the original

| # | Defect | Evidence |
|---|---|---|
| 1 | A stray top-level `return` made the **whole notebook die** (`SyntaxError: 'return' outside function`) — `torch`/`pandas`/`altair` were never imported | every downstream cell raised `NameError` |
| 2 | **Non-causal**: the memory summed over *all* positions, including future ones | changing future tokens moved `y[0]` by **131% of ‖y[0]‖** |
| 3 | `_from_hd` used a **transpose** as a pseudo-inverse | `‖R·Rᵀ − I‖ = 132.6`; after `pinv`, `1e-6` |
| 4 | Standard-attention memory was **8× too large** (multiplied by `n_heads` on top of `d_model`) | 256 MB claimed vs 32 MB real at 8 K |
| 5 | HDC *peak* memory was hidden behind the persistent figure | 0.25 MB claimed; **12 GB** actual at S=8 K |
| 6 | Benchmark extrapolated to 1 M while ignoring that intermediates would be **1.4 TB** | it printed "cannot run" but kept the "1 M tokens" claim elsewhere |
| 7 | Baseline time was **hard-coded** (`2034.0 ms`) | replaced with the measured value |
| 8 | **Rank-1 memory.** `hd_bind(a,b) = a·b` has no inverse: `a ⊙ a ≠ 1` for real vectors | retrieval fidelity **≈ 0 even at N=1**; capacity **0** for every `D` |
| 9 | The output was **statistically indistinguishable from noise** | `cos(in,out) = −0.0016` vs null `+0.0002`, `z = −0.64` |

Defects 1–7 were the notebook's own list; **8 and 9 are the root cause** and were not in it.
The `.py` artifact, the measurements, and the corrections are all in this repo.

## The fix

```
memory = Σ_t (K_t ⊙ P_t ⊙ V_t)          →  S = Σ_{t≤i} φ(K_t) V_tᵀ        ∈ R^{M×d}
output = M ⊙ Q                          →  out_i = φ(Q_i)ᵀ S / φ(Q_i)ᵀ Σ φ(K_t)
```

| | before | after |
|---|---|---|
| state rank | **1** | ≤ min(M, d) |
| causal | ❌ 131% leak | ✅ leak = 0 (exact) |
| capacity @0.9 | **0** for every `D` | `O(d)` with the exact kernel |
| decode memory | `O(S·H·D)` | `O(M·d)` constant |
| invertible bind | ❌ not needed/possible | ✅ no inversion required |

Verified invariant: incremental decode matches the training-time forward pass to
`max|forward − step| < 1e-5`. (An earlier version of the residual was wrong and would have
generated garbage in production while looking fine in training.)

---

## Repo layout

```
src/model.py        standalone model definitions (no marimo) — StdAttention, LinearAttention, CharLM
src/generate.py     sample text from a checkpoint
src/bench.py        decode throughput vs context length and batch
src/notebook.py     the full marimo notebook: every experiment, every number
weights/            3 char-LM checkpoints, fp16
results/            the measured tables as CSV/JSON
MANIFEST.json       sha256 of every artifact
```

## Quickstart

```bash
pip install torch

# sample text
python src/generate.py --ckpt weights/hybrid_3of4.pt --n 300
python src/generate.py --ckpt weights/all_linear_0of4.pt --temperature 0.9

# decode throughput, including 1M context
python src/bench.py --ckpt weights/all_linear_0of4.pt \
    --seq-lens 32768 131072 1048576 --batches 1 64 256
python src/bench.py --ckpt weights/all_attn_4of4.pt \
    --seq-lens 1048576 --batches 1 16 64      # OOM at 64 is the expected result
```

```
  seq_len  batch   ms/step      tok/s    kv(MB)  state(KB)
  1048576      1     0.954      1,048       0.0       66.0
  1048576    256     1.051    243,579       0.0    16896.0
```

Checkpoints are stored in fp16 to halve the download and are upcast to fp32 on load
(`load_checkpoint(..., dtype=torch.float32)`), matching how they were trained.

To reproduce everything from scratch, open `src/notebook.py` with
[marimo](https://marimo.io) (`marimo edit src/notebook.py`). Guard against a silent divergence:

```python
x = torch.randint(0, vocab, (1, 40))
assert (model(x) - torch.cat([model.step(x[:, i:i+1], c) for i in range(40)], 1)).abs().max() < 1e-4
```

## Limitations — please read

1. **Throughput and memory, not quality at 1 M.** The char-LM was trained at `seq_len=256`.
   It *can* decode at 1 M with constant cost; it never learned to *use* 1 M tokens. Claiming
   quality at 1 M requires training at 1 M, which was out of scope.
2. **The attention baseline here cannot prefill 1 M.** `StdAttention.forward` materialises an
   `S×S` matrix. The 1 M numbers are **decode** measurements with a pre-filled cache.
   FlashAttention-class kernels can prefill 1 M; this code cannot.
3. **The MQAR attention ratio (75%) is an upper bound, not a recipe.** MQAR is deliberately
   recall-heavy — the worst case for linear attention — and the model is only 4 layers.
   Published hybrids use **12–25%** attention on 30+ layer models over natural data.
4. **The `0.939` quality figure that appears early in the notebook is meaningless.** It was
   measured with untrained random weights, where softmax attention is nearly uniform. The
   trained MQAR gap is **0.86 vs 0.16**.
5. **Single-GPU, single-machine, CPU+GPU mix.** No FlashAttention, no fused kernels, no
   distributed setup. All-linear throughput is partly kernel-launch bound (≈0.95 ms/step at
   both B=1 and B=4), so a fused implementation would go considerably further.
6. **The all-linear model is the fastest and the worst.** val 1.655 vs 1.335. It trades exactly
   what the MQAR experiment measures. The hybrid is the honest operating point.

## The real fix for recall

A matrix state with **superposition** cannot recall. The fix is to make the write an
**error-correcting update** instead:

```
S_t = S_{t-1} (I − β_t k_t k_tᵀ)  +  β_t v_t k_tᵀ
     └──── erase the old key ────┘
```

This is the **delta rule** used by DeltaNet / GLA / Based / Mamba-2. It keeps the same
constant-size causal state and *does* solve MQAR. It sits **outside** the bind/superpose
paradigm, which is why no amount of tuning this notebook's operator would have worked.

## References

- Katharinen et al., *Transformers are RNNs*, 2020 — linear attention
- Arora et al., *Simple linear attention language models balance the recall-throughput tradeoff*, ICML 2024 — Based
- *A Systematic Analysis of Hybrid Linear Attention*, arXiv:2507.06457 — hybrid ratios and recall
- Choromanski et al., *Rethinking Attention with Performers*, 2021 — random-feature linear attention
- Kanerva, *Sparse Distributed Memory*, 1988
- Yang et al., *Gated Linear Attention* / *DeltaNet*; Dao & Gu, *Mamba-2*, 2024 — delta-rule state
- Jamba (AI21), MiniMax-01, RecurrentGemma, StripedHyena, Zamba, Samba — production hybrids

## License

No license file is included. Add one before reuse.
