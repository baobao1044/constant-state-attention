"""Checkpoints load safely and match the manifest hashes."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = sorted((ROOT / "weights").glob("*.pt"))


def test_manifest_hashes_match_files():
    man = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    assert man["artifacts"], "manifest is empty"
    for art in man["artifacts"]:
        p = ROOT / art["path"]
        assert p.exists(), f"missing artifact {art['path']}"
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest == art["sha256"], f"hash mismatch for {art['path']}"


@pytest.mark.skipif(not WEIGHTS, reason="no checkpoints in weights/")
@pytest.mark.parametrize("path", WEIGHTS, ids=lambda p: p.name)
def test_checkpoint_loads_with_weights_only(path):
    """weights_only=True is required: an unchecked pickle can execute code."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert set(ck) >= {"mask", "state_dict", "chars", "vocab"}
    assert all(m in (0, 1) for m in ck["mask"])
    assert len(ck["chars"]) == ck["vocab"]


@pytest.mark.skipif(not WEIGHTS, reason="no checkpoints in weights/")
@pytest.mark.parametrize("path", WEIGHTS, ids=lambda p: p.name)
def test_checkpoint_reconstructs_and_generates(path):
    from src.model import generate, load_checkpoint

    model, meta = load_checkpoint(path)
    itos = {i: c for i, c in enumerate(meta["chars"])}
    text = generate(model, "ROMEO: ", meta["chars"], itos, n_new=32, seed=0)
    assert len(text) == len("ROMEO: ") + 32
    assert all(c in meta["chars"] for c in text)


@pytest.mark.skipif(not WEIGHTS, reason="no checkpoints in weights/")
def test_weights_are_upcast_to_float32_by_default():
    from src.model import load_checkpoint

    model, _ = load_checkpoint(WEIGHTS[0])
    assert next(model.parameters()).dtype == torch.float32
