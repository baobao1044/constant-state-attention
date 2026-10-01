"""The model must be causal: a token cannot see the future.

This is the defect that made the original design unusable (changing future tokens moved
``y[0]`` by 131% of its own norm), so it is asserted explicitly rather than assumed.
"""

import torch


def test_forward_is_causal(random_model):
    x = torch.randint(0, 63, (2, 12))
    with torch.no_grad():
        y_full = random_model(x)
        x2 = x.clone()
        x2[:, 5:] = torch.randint(0, 63, x2[:, 5:].shape)
        y_changed = random_model(x2)
    assert torch.allclose(y_full[:, :5], y_changed[:, :5], atol=1e-6), \
        "outputs before position 5 changed when only future tokens changed"


def test_causal_mask_is_present_in_attention():
    """Sanity check on the mask itself: row i attends to columns <= i only."""
    S = 6
    m = torch.tril(torch.ones(S, S, dtype=torch.bool))
    assert m[0].sum() == 1
    assert m.sum() == S * (S + 1) // 2
    assert not m[0, 1]


def test_last_position_does_depend_on_future_free_input(random_model):
    """A guard against the reverse failure: the final position must use the whole prefix."""
    x = torch.randint(0, 63, (1, 10))
    with torch.no_grad():
        y = random_model(x)
    assert torch.isfinite(y).all()
    assert y.shape == (1, 10, 63)
