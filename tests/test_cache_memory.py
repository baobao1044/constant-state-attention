"""Cache accounting: the fixed state must not grow with sequence length.

Also pins that it *does* grow with batch — "constant" means constant in sequence
length only, which is how the README phrases it.
"""

import torch

D_MODEL, N_HEADS, HEAD_DIM = 128, 4, 32
LAYERS = 4
PER_LAYER_STATE = (N_HEADS * HEAD_DIM * HEAD_DIM + N_HEADS * HEAD_DIM) * 4  # S + z, fp32


def test_all_linear_state_is_constant_in_seq_len():
    from src.model import CharLM

    torch.manual_seed(0)
    m = CharLM([0, 0, 0, 0], 63).eval()
    sizes = []
    for seq_len in (16, 512, 65536):
        c = m.new_cache(maxlen=seq_len, batch=1)
        for bc in c["blocks"]:
            bc["n"] = seq_len
        st = sum((bc["S"].numel() + bc["z"].numel()) * 4 for bc in c["blocks"])
        sizes.append(st)
    assert len(set(sizes)) == 1, f"state grew with seq_len: {sizes}"
    assert sizes[0] == LAYERS * PER_LAYER_STATE == 67584  # 66 KiB


def test_all_linear_state_scales_with_batch():
    from src.model import CharLM

    torch.manual_seed(0)
    m = CharLM([0, 0, 0, 0], 63).eval()
    c1 = m.new_cache(maxlen=8, batch=1)
    c16 = m.new_cache(maxlen=8, batch=16)
    s1 = sum((bc["S"].numel() + bc["z"].numel()) * 4 for bc in c1["blocks"])
    s16 = sum((bc["S"].numel() + bc["z"].numel()) * 4 for bc in c16["blocks"])
    assert s16 == 16 * s1, "'constant' is constant in seq_len, not in batch"


def test_attention_kv_grows_linearly_and_matches_formula():
    from src.model import CharLM

    torch.manual_seed(0)
    m = CharLM([1, 1, 1, 1], 63).eval()
    batch, seq_len = 2, 1000
    c = m.new_cache(maxlen=seq_len, batch=batch)
    for bc in c["blocks"]:
        bc["n"] = seq_len
    kv = sum(2 * bc["n"] * bc["k"].shape[0] * bc["k"].shape[1] * bc["k"].shape[3] * 4
             for bc in c["blocks"])
    expected = LAYERS * 2 * seq_len * batch * N_HEADS * HEAD_DIM * 4
    assert kv == expected


def test_zero_kv_for_all_linear_model():
    from src.model import CharLM

    torch.manual_seed(0)
    m = CharLM([0, 0, 0, 0], 63).eval()
    c = m.new_cache(maxlen=1024)
    assert all("k" not in bc for bc in c["blocks"])
