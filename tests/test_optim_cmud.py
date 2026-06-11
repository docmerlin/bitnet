"""Regression tests for the C-MUD optimizer and its 8-bit C-Lion fallback."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optim import (
    CMUD,
    build_cmud,
    cautious_mask,
    mud_decorrelate,
    split_parameters_for_cmud,
    _dequantize_blockwise,
    _quantize_blockwise,
)


def test_mud_decorrelate_row_orthonormalizes() -> bool:
    torch.manual_seed(0)
    # Tall matrix: smaller dimension k = 16 (columns). MUD targets Q Qᵀ ≈ I_k along
    # the smaller dimension, so check the gram of the transposed (k x d) form.
    grad = torch.randn(32, 16)
    whitened = mud_decorrelate(grad, passes=3)
    assert whitened.shape == grad.shape
    gram = whitened.t() @ whitened  # 16 x 16, the smaller dimension
    identity = torch.eye(gram.size(0))
    # Diagonal should be near 1 (rows normalized) and off-diagonals decorrelated.
    assert torch.allclose(gram.diag(), torch.ones(gram.size(0)), atol=0.15), gram.diag()
    off_diag = gram - torch.diag(gram.diag())
    assert off_diag.abs().max() < 0.3, off_diag.abs().max()
    # Whitening should reduce correlation versus the raw input.
    raw_gram = grad.t() @ grad
    raw_off = raw_gram - torch.diag(raw_gram.diag())
    norm = lambda g: g.norm() / (g.diag().mean() + 1e-8)
    assert norm(off_diag) < norm(raw_off)
    print("MUD triangular-whitening tests passed")
    return True


def test_cautious_mask_zeroes_disagreeing_coords() -> bool:
    update = torch.tensor([1.0, -1.0, 1.0, -1.0])
    grad = torch.tensor([1.0, 1.0, -1.0, -1.0])  # agrees on coords 0 and 3
    masked = cautious_mask(update, grad)
    assert masked[1] == 0.0 and masked[2] == 0.0
    # Survivors are rescaled by numel / kept = 4 / 2 = 2.
    assert torch.allclose(masked[0], torch.tensor(2.0))
    assert torch.allclose(masked[3], torch.tensor(-2.0))
    print("cautious mask tests passed")
    return True


def test_blockwise_quant_roundtrip_is_close() -> bool:
    torch.manual_seed(1)
    tensor = torch.randn(5000)
    quantized, scale = _quantize_blockwise(tensor)
    assert quantized.dtype == torch.int8
    restored = _dequantize_blockwise(quantized, scale, tensor.shape)
    assert restored.shape == tensor.shape
    # Symmetric int8 with per-block scale keeps relative error small.
    rel_err = (restored - tensor).abs().mean() / tensor.abs().mean()
    assert rel_err < 0.02, rel_err
    print("block-wise int8 roundtrip tests passed")
    return True


def test_split_routes_embeddings_to_fallback() -> bool:
    model = nn.Module()
    model.embed = nn.Embedding(50, 8)
    model.proj = nn.Linear(8, 8, bias=True)
    model.norm = nn.LayerNorm(8)
    model.add_module("scalar", nn.Module())
    model.scalar.gate = nn.Parameter(torch.zeros(1))

    mud_params, fallback_params = split_parameters_for_cmud(model)
    mud_ids = {id(p) for p in mud_params}
    assert id(model.proj.weight) in mud_ids  # 2D linear weight -> MUD
    assert id(model.embed.weight) not in mud_ids  # embedding -> fallback
    fallback_ids = {id(p) for p in fallback_params}
    assert id(model.embed.weight) in fallback_ids
    assert id(model.proj.bias) in fallback_ids  # 1D bias -> fallback
    assert id(model.norm.weight) in fallback_ids
    print("parameter routing tests passed")
    return True


def test_split_dedupes_tied_weights() -> bool:
    model = nn.Module()
    model.embed = nn.Embedding(20, 8)
    model.head = nn.Linear(8, 20, bias=False)
    model.head.weight = model.embed.weight  # tie

    mud_params, fallback_params = split_parameters_for_cmud(model)
    total = mud_params + fallback_params
    ids = [id(p) for p in total]
    assert len(ids) == len(set(ids))  # no duplicates
    assert id(model.embed.weight) in {id(p) for p in fallback_params}
    print("tied-weight dedupe tests passed")
    return True


def _quadratic_descends(optimizer_factory) -> float:
    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(16, 16, bias=True), nn.Linear(16, 16, bias=False))
    optimizer = optimizer_factory(model)
    target = torch.randn(8, 16)
    inputs = torch.randn(8, 16)

    first_loss = None
    for _ in range(50):
        optimizer.zero_grad()
        out = model(inputs)
        loss = (out - target).pow(2).mean()
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = float(loss.item())
    return first_loss - float(loss.item())


def test_cmud_step_reduces_loss() -> bool:
    def factory(model: nn.Module) -> CMUD:
        return build_cmud(model, lr=0.05, fallback_lr=0.02, weight_decay=0.0)

    improvement = _quadratic_descends(factory)
    assert improvement > 0.0, improvement
    print("C-MUD optimization step tests passed")
    return True


def test_cmud_8bit_state_serializes_and_resumes() -> bool:
    torch.manual_seed(3)
    model = nn.Sequential(nn.Embedding(4096, 4), nn.Linear(4, 4, bias=False))
    optimizer = build_cmud(model, lr=0.05, fallback_lr=0.02, weight_decay=0.0, eight_bit=True)

    inputs = torch.randint(0, 4096, (8,))
    for _ in range(3):
        optimizer.zero_grad()
        loss = model[1](model[0](inputs)).pow(2).mean()
        loss.backward()
        optimizer.step()

    # The embedding (>= block size) should hold quantized int8 momentum state.
    emb_state = optimizer.state[model[0].weight]
    assert "exp_avg_q" in emb_state and emb_state["exp_avg_q"].dtype == torch.int8

    payload = optimizer.state_dict()
    restored = build_cmud(model, lr=0.05, fallback_lr=0.02, weight_decay=0.0, eight_bit=True)
    restored.load_state_dict(payload)
    restored_state = restored.state[model[0].weight]
    assert torch.equal(restored_state["exp_avg_q"], emb_state["exp_avg_q"])
    print("C-MUD 8-bit state serialization tests passed")
    return True


if __name__ == "__main__":
    test_mud_decorrelate_row_orthonormalizes()
    test_cautious_mask_zeroes_disagreeing_coords()
    test_blockwise_quant_roundtrip_is_close()
    test_split_routes_embeddings_to_fallback()
    test_split_dedupes_tied_weights()
    test_cmud_step_reduces_loss()
    test_cmud_8bit_state_serializes_and_resumes()
