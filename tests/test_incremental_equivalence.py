"""Incremental decode must equal the training-time forward pass.

A recurrent decode path that silently diverges from ``forward`` produces a model that
trains fine and generates garbage.  An earlier version of this repo had exactly that bug
(the MLP residual used the post-attention value), so this is pinned.
"""

import torch

TOL = 1e-4


def test_step_matches_forward(random_model):
    x = torch.randint(0, 63, (1, 40))
    with torch.no_grad():
        y_fwd = random_model(x)
        c = random_model.new_cache(maxlen=64)
        y_step = torch.cat([random_model.step(x[:, i:i + 1], c) for i in range(40)], 1)
    assert (y_fwd - y_step).abs().max().item() < TOL


def test_prefix_consistency(random_model):
    """forward(x[:, :k])[-1] must equal forward(x)[:, k-1]."""
    x = torch.randint(0, 63, (1, 24))
    with torch.no_grad():
        full = random_model(x)
        for k in (1, 7, 24):
            assert (random_model(x[:, :k])[:, -1] - full[:, k - 1]).abs().max().item() < TOL


def test_step_matches_forward_with_batch(random_model):
    x = torch.randint(0, 63, (4, 16))
    with torch.no_grad():
        y_fwd = random_model(x)
        c = random_model.new_cache(maxlen=32, batch=4)
        y_step = torch.cat([random_model.step(x[:, i:i + 1], c) for i in range(16)], 1)
    assert (y_fwd - y_step).abs().max().item() < TOL


def test_cache_advances_position(random_model):
    c = random_model.new_cache(maxlen=8)
    x = torch.randint(0, 63, (1, 1))
    with torch.no_grad():
        random_model.step(x, c)
        random_model.step(x, c)
    assert c["pos"] == 2
    for bc in c["blocks"]:
        assert bc["n"] == 2
