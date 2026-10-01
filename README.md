# constant-state-attention

An audit-and-fix of a "HDC-Attention" notebook (hyperdimensional-computing attention as a
KV-cache replacement), which turned into a measurement of **what a fixed-size attention
state can and cannot do**.

Short version:

- The original design **could not run at all**, and no hyperparameter could have saved it:
  its memory is a **vector**, and a vector has **rank 1**.
- Replacing it with a **matrix state** (`Σ φ(k) vᵀ`) makes it causal, `O(S)` in time, and
  constant-memory in sequence length.
- But a fixed-dimensional **additive superposition** state cannot do exact recall: on MQAR
  it scores **0.168 ± 0.002** against **0.834 ± 0.020** for softmax attention, and
  increasing the "HD" dimension changes nothing.
- A **hybrid** (full attention on 3 of 4 layers) matches all-attention within seed noise
  (**0.858 ± 0.021** vs **0.834 ± 0.020**) at 25 % less KV memory.
- A **single-step decode microbenchmark with a simulated 1 M-token cache** shows the
  all-linear model flat at **1,062 tok/s** (batch 1) and **245,858 tok/s** (batch 256) with
  a **66 KB** state, while all-attention saturates and then OOMs. See the naming caveat below.

> ### Status: an analysis, not a new architecture
>
> The hybrid is an instance of a well-established family ([Based](https://arxiv.org/abs/2402.18668),
> Jamba, MiniMax-01, Mamba-2-Hybrid, RecurrentGemma, StripedHyena, Qwen3-Next) and the
> linear layer is standard [linear attention](https://arxiv.org/abs/2006.16236).
> **The "HD" part contributes nothing measurable** (see [the flat-kernel result](#2-the-hd-dimension-buys-nothing)).
> What is original here is the **measurements and the corrections**, not the mechanism.

Every table below is generated from `results/*.csv` by `tools/render_tables.py`, and CI
fails if the README drifts from the artifacts.

---

## 1. Decode cost at long context — **and what this benchmark does not show**

`src/bench.py` does **not** feed a sequence through the model. It fabricates a cache of a
given length and times one decode step against it. That is a **single-step decode
microbenchmark with a simulated cache**. It isolates the per-step cost, the
memory-bandwidth wall, and the flat cost of a fixed state — and it demonstrates **none** of
the following:

- that the model can ingest / prefill 1 M tokens,
- that the state stays numerically stable across 1 M updates,
- that information 1 M tokens back is usable,
- that the positional mechanism reaches 1 M. It does not: `CharLM.pe` is
  `nn.Embedding(8192)`, so a real stream **stops at position 8191**; the benchmark sidesteps
  this by holding the position index at 0, which is irrelevant to timing but means these
  numbers say nothing about long-range quality.

<!-- TABLE:decode_state -->
| model | KV cache @ 1024K | fixed state @ 1024K |
|---|---|---|
| `all_linear_0of4` | 0 (no attention layer) | **66.0 KB** |
| `hybrid_3of4` | 3,072 MB | **16.5 KB** |
| `all_attn_4of4` | 4,096 MB | **0.0 KB** |

batch = 1.
<!-- /TABLE:decode_state -->

<!-- TABLE:decode -->
**batch = 1** — tok/s

| model | 32K | 128K | 512K | 1024K |
|---|---|---|---|---|
| `all_linear_0of4` | **701** | **1,015** | **994** | **1,062** |
| `hybrid_3of4` | **704** | **1,066** | **352** | **169** |
| `all_attn_4of4` | **537** | **871** | **228** | **117** |

**batch = 4** — tok/s

| model | 32K | 128K | 512K | 1024K |
|---|---|---|---|---|
| `all_linear_0of4` | **3,938** | **4,024** | **4,038** | **4,243** |
| `hybrid_3of4` | **4,167** | **2,412** | **684** | **345** |
| `all_attn_4of4` | **4,081** | **1,888** | **518** | **243** |

**batch = 16** — tok/s

| model | 32K | 128K | 512K | 1024K |
|---|---|---|---|---|
| `all_linear_0of4` | **16,783** | **16,243** | **16,198** | **16,869** |
| `hybrid_3of4` | **10,081** | **2,880** | **705** | **350** |
| `all_attn_4of4` | **7,916** | **2,192** | **530** | **263** |

**batch = 64** — tok/s

| model | 32K | 128K | 512K | 1024K |
|---|---|---|---|---|
| `all_linear_0of4` | **64,796** | **65,057** | **65,257** | **62,465** |
| `hybrid_3of4` | — | — | — | — |
| `all_attn_4of4` | **8,593** | **2,094** | — | — |

**batch = 256** — tok/s

| model | 32K | 128K | 512K | 1024K |
|---|---|---|---|---|
| `all_linear_0of4` | **242,223** | **242,300** | **246,623** | **245,858** |
| `hybrid_3of4` | — | — | — | — |
| `all_attn_4of4` | — | — | — | — |
<!-- /TABLE:decode -->

Reading it: **all-linear is flat** — the same tok/s at 32 K and at 1 M. **All-attention
saturates** (increasing batch barely moves tok/s; it is memory-bandwidth bound because every
token re-reads the whole cache) and then OOMs once the cache no longer fits. The hybrid sits
between them, always ahead of all-attention.

The original design's requirement at 1 M was **1.43 TB** of intermediate tensors — the
notebook computed this itself and concluded "cannot run". The fixed design needs 66 KB.

## 2. The HD dimension buys nothing

Reading a superposed memory as `r = Σ_t κ(q,k_t)·v_t` is a **convex combination** of the
stored values. To return `v_i` exactly, `κ` must be a near-**delta**. Measured over 400 keys:

<!-- TABLE:featmap -->
| feature map | M | κ(self) | κ(off-diagonal) |
|---|---|---|---|
| `elu+1` | 512 | 523.243 | **515.9152** |
| `elu+1` | 8,192 | 8370.041 | **8253.2500** |
| `rff Ïƒ=4` | 512 | 0.486 | **0.4541** |
| `rff Ïƒ=4` | 8,192 | 0.502 | **0.4721** |
| `identity` | 512 | 1.000 | **0.0002** |
| `identity` | 8,192 | 1.000 | **0.0002** |
<!-- /TABLE:featmap -->

`ELU(Rx)+1` and random Fourier features have an off-diagonal similarity **comparable to the
self term** — every key looks like every other key, so the memory cannot discriminate.
Only the exact dot product (`identity`) has `κ(off-diagonal) ≈ 0`. A non-linear random
projection *flattens* the kernel; a linear one is an isometry (Johnson–Lindenstrauss) and
preserves the dot product that was already there. Either way the extra dimension is dead
weight, as the fidelity curves show:

<!-- TABLE:featmap_fidelity -->
| feature map | M | N=1 | N=2 | N=8 | N=32 | N=128 |
|---|---|---|---|---|---|---|
| `elu+1` | 512 | 1.000 | 0.692 | 0.355 | 0.175 | 0.087 |
| `elu+1` | 8,192 | 1.000 | 0.692 | 0.355 | 0.175 | 0.087 |
| `rff Ïƒ=4` | 512 | 1.000 | 0.713 | 0.372 | 0.184 | 0.092 |
| `rff Ïƒ=4` | 8,192 | 1.000 | 0.711 | 0.371 | 0.184 | 0.091 |
| `identity` | 512 | 1.000 | 0.998 | 0.955 | 0.830 | 0.580 |
| `identity` | 8,192 | 1.000 | 0.998 | 0.955 | 0.830 | 0.580 |
<!-- /TABLE:featmap_fidelity -->

Source: `results/notebook/*.csv`, produced by `src/notebook.py`, not by `experiments/`.

### Capacity depends on rank, not size

<!-- TABLE:capacity -->
| state kind | dimension `M` | state elements | capacity @ 0.9 |
|---|---|---|---|
| `hd_linear` | 16 | 16 | **1** |
| `hd_linear` | 64 | 64 | **1** |
| `hd_linear` | 256 | 256 | **1** |
| `hd_linear` | 1,024 | 1,024 | **1** |
| `matrix` | 32 | 1,024 | **8** |
| `matrix` | 64 | 4,096 | **16** |
| `matrix` | 128 | 16,384 | **32** |
| `matrix` | 256 | 65,536 | **32** |
| `matrix` | 512 | 262,144 | **64** |
| `vector` | 1,024 | 1,024 | **0** |
| `vector` | 8,192 | 8,192 | **0** |
| `vector` | 32,768 | 32,768 | **0** |
<!-- /TABLE:capacity -->

A vector state has rank 1, so capacity is **0 at every dimension tested**. A `d×d` matrix
state has rank `d` and capacity grows with `d`.

## 3. Recall: additive superposition fails; the delta rule also failed *here*

MQAR — emit N `(key, value)` pairs, then N queries, and return the paired value. Four
layers, 2000 steps, five seeds:

<!-- TABLE:mqar -->
| variant | final loss | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|---|
| `std` | 0.458 ± 0.061 | **0.834** ± 0.020 | **0.506** ± 0.020 | **0.226** ± 0.011 | **0.107** ± 0.004 |
| `linear_elu_M32` | 2.025 ± 0.006 | **0.169** ± 0.005 | **0.122** ± 0.001 | **0.078** ± 0.001 | **0.051** ± 0.001 |
| `linear_elu_M256` | 2.024 ± 0.011 | **0.171** ± 0.002 | **0.122** ± 0.002 | **0.078** ± 0.003 | **0.052** ± 0.003 |
| `linear_relu_M32` | 2.024 ± 0.006 | **0.169** ± 0.003 | **0.120** ± 0.001 | **0.073** ± 0.003 | **0.048** ± 0.004 |
| `linear_identity_M32` | 4.113 ± 0.038 | **0.028** ± 0.009 | **0.023** ± 0.005 | **0.019** ± 0.003 | **0.018** ± 0.002 |
| `delta_rule` | 2.018 ± 0.010 | **0.168** ± 0.002 | **0.110** ± 0.002 | **0.057** ± 0.003 | **0.034** ± 0.003 |

Mean ± std over **5 seeds**; random baseline = 1/64 = 0.016.
<!-- /TABLE:mqar -->

**Increasing the HD dimension does not help** (`linear_elu_M256` ≈ `linear_elu_M32`), which
directly falsifies the `_hd_dim = 8192` premise. Note also that with five seeds the softmax
baseline is `0.834 ± 0.020` — a single-seed claim of 0.95 would have been luck.

### The delta rule did not rescue it

A delta-rule state (the mechanism behind DeltaNet / GLA / Mamba-2) replaces additive
superposition with an error-correcting write:

```
S_t = S_{t-1} + β_t (v_t − S_{t-1}ᵀ k_t) k_tᵀ
```

<!-- TABLE:delta_stability -->
| variant | final loss | N=8 |
|---|---|---|
| `delta_rule` | 2.015 ± 0.002 | **0.168** ± 0.002 |
| `delta_rule_nonorm` | **NaN** ± **NaN** | **0.000** ± 0.000 |

3 seeds. `delta_rule_nonorm` diverges to NaN without key normalisation.
<!-- /TABLE:delta_stability -->

**This is a negative result and it is reported as one.** At 2 and at 4 layers, with and
without key normalisation, the mini delta rule did not beat plain additive superposition on
this task, and without normalisation it diverges. The likely reason is architectural: at a
value position the layer only sees the *value* token, so a purely recurrent mixer has no
cheap way to bind "this value belongs to the key I saw one step ago". Attention gets that
for free (the value position attends back to the key token). Published delta-rule models
succeed in hybrid configurations — Based pairs linear attention with **local exact
attention** — which is the same conclusion this repo reaches from the other direction.

So the corrected claim is narrow and measured:

> A **fixed-dimensional additive superposition state cannot provide exact, unbounded
> associative recall.** Delta-rule/gated recurrent states are the known direction, but
> whether they suffice depends on the surrounding architecture: in this minimal block they
> did not.

## 4. How much attention can you remove?

Per-layer masks over 4 layers, five seeds:

<!-- TABLE:hybrid_ratio -->
| attention layers | linear layers | final loss | N=8 | N=16 | N=32 |
|---|---|---|---|---|---|
| `attn_0of4` | 0/4 | 4/4 | 2.025 ± 0.006 | **0.169** ± 0.005 | **0.122** ± 0.001 | **0.078** ± 0.001 |
| `attn_1of4_first` | 1/4 | 3/4 | 1.809 ± 0.151 | **0.285** ± 0.085 | **0.156** ± 0.033 | **0.087** ± 0.013 |
| `attn_1of4_last` | 1/4 | 3/4 | 2.027 ± 0.005 | **0.168** ± 0.002 | **0.115** ± 0.002 | **0.061** ± 0.002 |
| `attn_2of4` | 2/4 | 2/4 | 1.936 ± 0.073 | **0.218** ± 0.033 | **0.126** ± 0.012 | **0.069** ± 0.006 |
| `attn_3of4` | 3/4 | 1/4 | 0.417 ± 0.071 | **0.858** ± 0.021 | **0.523** ± 0.024 | **0.249** ± 0.011 |
| `attn_4of4` | 4/4 | 0/4 | 0.458 ± 0.061 | **0.834** ± 0.020 | **0.506** ± 0.020 | **0.226** ± 0.011 |

Mean ± std over **5 seeds**.
<!-- /TABLE:hybrid_ratio -->

- **3/4 matches 4/4** (`0.858 ± 0.021` vs `0.834 ± 0.020`) — one quarter of the attention
  layers can be linearised at no measurable cost, and KV memory drops 25 %.
- **1/4 and 2/4 fail** (`0.285 ± 0.085`, `0.218 ± 0.033`).
- **Placement matters**: attention *first* gives 0.285, attention *last* gives 0.168 —
  no better than no attention at all.

⚠️ This is an **upper bound**, not a recipe. MQAR is deliberately recall-heavy and the model
is 4 layers. Published hybrids use 12–25 % attention on 30+ layer models over natural data.

## 5. Char-LM

tiny shakespeare, `d_model=128`, 4 layers, 1200 steps, 5 seeds, held-out 10 % split:

<!-- TABLE:char_lm -->
| variant | valid loss | perplexity | 4-gram diversity |
|---|---|---|---|
| `all_attn_4of4` | **1.5351** ± 0.0089 | 4.64 | 0.941 |
| `hybrid_3of4` | **1.5403** ± 0.0129 | 4.67 | 0.956 |
| `all_linear_0of4` | **1.8031** ± 0.0225 | 6.07 | 0.927 |

Mean ± std over **5 seeds**. Validation is a held-out 10 % split of tiny shakespeare, not a random training batch.
<!-- /TABLE:char_lm -->

The hybrid is within one standard deviation of all-attention while removing a quarter of the
attention layers. Samples are in `results/char_lm_samples.txt`.

---

## What was wrong with the original

| # | Defect | Evidence |
|---|---|---|
| 1 | A stray top-level `return` made the **whole notebook die** (`SyntaxError: 'return' outside function`) | `torch`/`pandas`/`altair` never imported; every downstream cell raised `NameError` |
| 2 | **Non-causal**: the memory summed over *all* positions, including future ones | changing future tokens moved `y[0]` by **131 % of ‖y[0]‖** |
| 3 | `_from_hd` used a **transpose** as a pseudo-inverse | `‖R·Rᵀ − I‖ = 132.6`; after `pinv`, `1e-6` |
| 4 | Standard-attention memory was **8× too large** (multiplied by `n_heads` on top of `d_model`) | 256 MB claimed vs 32 MB real at 8 K |
| 5 | HDC *peak* memory was hidden behind the persistent figure | 0.25 MB claimed; **12 GB** actual at S=8 K |
| 6 | Extrapolated to 1 M while ignoring that intermediates would be **1.4 TB** | the notebook printed "cannot run" yet kept the 1 M claim elsewhere |
| 7 | Baseline time was **hard-coded** (`2034.0 ms`) | replaced with the measured value |
| 8 | **Rank-1 memory.** `hd_bind(a,b) = a·b` has no inverse: `a ⊙ a ≠ 1` for real vectors | retrieval fidelity **≈ 0 even at N=1**; capacity **0** for every `D` |
| 9 | The output was **statistically indistinguishable from noise** | `cos(in,out) = −0.0016` vs null `+0.0002`, `z = −0.64` |

Defects 1–7 were the notebook's own review list; **8 and 9 are the root cause** and were not
in it.

## The fix

```
memory = Σ_t (K_t ⊙ P_t ⊙ V_t)   →   S = Σ_{t≤i} φ(K_t) V_tᵀ       ∈ R^{M×d}
output = M ⊙ Q                   →   out_i = φ(Q_i)ᵀ S / φ(Q_i)ᵀ Σ φ(K_t)
```

| | before | after |
|---|---|---|
| state rank | **1** | ≤ min(M, d) |
| causal | ❌ 131 % leak | ✅ leak = 0 (exact) |
| capacity @ 0.9 | **0** for every `D` | `O(d)` with the exact kernel |
| decode memory | `O(S·H·D)` | `O(M·d)`, constant in `S` |
| invertible bind | ❌ impossible/not needed | ✅ no inversion required |

Invariant enforced by `tests/test_incremental_equivalence.py`: incremental decode equals the
training-time forward to `< 1e-4`. An earlier version of the residual was wrong and would
have generated garbage in production while looking fine in training.

## Repo layout

```
src/model.py          standalone model definitions (no marimo)
src/generate.py       sample text from a checkpoint
src/bench.py          single-step decode microbenchmark (simulated cache)
src/common.py         seeding, environment manifest, versioned run directories
src/notebook.py       the full marimo notebook — exploratory record, all findings
experiments/mqar.py          MQAR sweep, --seeds, writes results/mqar.csv
experiments/hybrid_ratio.py  attention-fraction sweep
experiments/char_lm.py       char-LM training; writes the shipped checkpoints
tools/render_tables.py  regenerate the README tables from results/*.csv
tools/make_manifest.py  regenerate MANIFEST.json (sha256 of every artifact)
tools/fetch_data.py     download tiny shakespeare into data/
tests/                  causality, incremental equivalence, cache accounting, checkpoints
weights/*.pt            3 char-LM checkpoints, fp16
results/                measured tables + one versioned run dir per experiment
results/notebook/       CSVs produced by src/notebook.py (separate provenance)
MANIFEST.json           sha256 of every artifact
```

## Quickstart

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python src/generate.py --ckpt weights/hybrid_3of4.pt --n 300
python src/bench.py --ckpt weights/all_linear_0of4.pt \
    --seq-lens 32768 131072 1048576 --batches 1 64 256
```

## Reproducing

```bash
python tools/fetch_data.py
python -m experiments.mqar --seeds 0 1 2 3 4 --layers 4     # or --seeds 0 --steps 50 to smoke
python -m experiments.hybrid_ratio --seeds 0 1 2 3 4
python -m experiments.char_lm --seeds 0 1 2 3 4             # also refreshes weights/*.pt
python tools/render_tables.py                               # README tables from results/
python tools/make_manifest.py
python -m pytest -q && ruff check .
```

Every driver writes `results/<name>/<utc-timestamp>/{config,env,metrics}` — the seed, the
config, and the hardware/software manifest are recorded next to the numbers. `experiments/`
is the source of truth for the README; `src/notebook.py` is kept as the exploratory record
and its CSVs live under `results/notebook/` so the two provenances cannot be confused.

CI (`.github/workflows/ci.yml`) runs ruff, the manifest check, the README-sync check, the
test suite on Python 3.10–3.12, and a fast experiment smoke run.

## Limitations — please read

1. **The 1 M numbers are a microbenchmark, not a model run.** No 1 M-token stream is ever
   processed. See §1 for the full list of what is *not* shown.
2. **Positional encoding stops at 8191.** `CharLM.pe` is `nn.Embedding(8192)`. Claiming a
   real 1 M-token context requires replacing it and actually streaming 1 M updates.
3. **The attention baseline cannot prefill long sequences**: `StdAttention.forward`
   materialises an `S×S` matrix. FlashAttention-class kernels can; this code cannot.
4. **The block is not a standard Transformer block.** It uses a shared-residual read
   (`z = x + attn(ln1(x)); out = x + mlp(ln2(z))`), not `out = z + mlp(ln2(z))`. All variants
   share it, so internal comparisons are fair, but it is not a drop-in baseline. Changing it
   would invalidate the shipped checkpoints.
5. **Small-scale, single-GPU, CPU+GPU mix.** 4 layers, `d_model=128`, 1200–2000 steps, tiny
   Shakespeare and a synthetic MQAR. No FlashAttention, no fused kernels, no distributed.
6. **All-linear is the fastest and the worst** (valid loss 1.803 vs 1.535). It trades exactly
   what §3 measures.
7. **The MQAR attention ratio is an upper bound**, not a prescription — see §4.

## References

- Katharinen et al., *Transformers are RNNs*, 2020 — linear attention
- Arora et al., *Simple linear attention language models balance the recall-throughput tradeoff*, ICML 2024 — Based
- *A Systematic Analysis of Hybrid Linear Attention*, arXiv:2507.06457 — hybrid ratios and recall
- Choromanski et al., *Rethinking Attention with Performers*, 2021 — random-feature linear attention
- Kanerva, *Sparse Distributed Memory*, 1988
- Yang et al., *Gated Linear Attention*; *DeltaNet*; Dao & Gu, *Mamba-2*, 2024 — delta-rule state
- Jamba (AI21), MiniMax-01, RecurrentGemma, StripedHyena, Zamba, Samba — production hybrids

## License

MIT — see [LICENSE](LICENSE).
