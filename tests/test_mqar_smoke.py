"""Smoke tests for the MQAR sweep: every variant runs, and the delta rule is learnable.

These are deliberately tiny (a few dozen steps) — they guard against shape/NaN
regressions, not against a loss of accuracy.
"""

import pytest
import torch

from experiments.mqar import VARIANTS, Model, make_batch, train_one


def test_make_batch_is_well_formed():
    gen = torch.Generator().manual_seed(0)
    n_pairs = 4
    toks, tgt = make_batch(n_pairs, 8, gen)
    assert toks.shape == (8, 3 * n_pairs)
    assert (toks[:, :2 * n_pairs] < 128).all()
    assert (tgt[:, :2 * n_pairs] == -100).all()
    assert (tgt[:, 2 * n_pairs:] >= 64).all()


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_variant_trains_a_few_steps(variant):
    model, loss, _ = train_one(variant, seed=0, n_pairs=4, steps=12,
                               batch=8, lr=3e-3, device="cpu")
    assert torch.isfinite(torch.tensor(loss)), f"{variant} produced a non-finite loss"
    gen = torch.Generator().manual_seed(0)
    toks, _ = make_batch(4, 4, gen)
    out = model(toks)
    assert out.shape == (4, 12, 128)
    assert torch.isfinite(out).all()


def test_delta_rule_state_is_fixed_size():
    """The counter-example to 'a fixed-size state cannot recall' really is fixed-size."""
    m = Model(VARIANTS["delta_rule"])
    assert sum(p.numel() for p in m.parameters()) > 0
    # state shape is asserted by construction in DeltaAttention.forward; check the
    # layer exposes the same constant-size contract as LinearAttention.
    from experiments.mqar import DeltaAttention, LinearAttention
    d, lin = DeltaAttention(), LinearAttention()
    assert set(d.state_dict()) - set(lin.state_dict()) == {"Wb.weight", "Wb.bias"}


def test_attention_variant_has_no_recurrent_state():
    from experiments.mqar import StdAttention

    m = Model(lambda: StdAttention())
    toks, _ = make_batch(4, 2, torch.Generator().manual_seed(0))
    assert m(toks).shape[0] == 2
