import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MASKS = {"all_attn": [1, 1, 1, 1], "hybrid": [1, 1, 1, 0], "all_linear": [0, 0, 0, 0]}
VOCAB = 63


@pytest.fixture(params=sorted(MASKS))
def mask(request):
    return MASKS[request.param]


@pytest.fixture
def random_model(mask):
    import torch

    from src.model import CharLM

    torch.manual_seed(0)
    return CharLM(mask, VOCAB).eval()
