"""Regression tests for shared Hadamard caching in ``HBitLinear``."""


import torch

from config import TernaryConfig
from layers.h_bitlinear import HBitLinear, get_hadamard_tensor


def test_hadamard_tensor_is_shared() -> bool:
    torch.manual_seed(0)
    cfg = TernaryConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=4,
        intermediate_size=16,
        use_hadamard=True,
        use_4bit_activations=False,
    )
    layer_a = HBitLinear(8, 8, config=cfg)
    layer_b = HBitLinear(8, 8, config=cfg)
    x = torch.randn(2, 8)

    out_a = layer_a(x)
    out_b = layer_b(x)
    shared_a = get_hadamard_tensor(8, x.device, x.dtype)
    shared_b = get_hadamard_tensor(8, x.device, x.dtype)

    assert out_a.shape == (2, 8)
    assert out_b.shape == (2, 8)
    assert shared_a.data_ptr() == shared_b.data_ptr(), "Identical Hadamard requests should reuse one cached tensor"
    assert len(tuple(layer_a.buffers())) == 0, "HBitLinear should not register a per-module Hadamard buffer"
    assert len(tuple(layer_b.buffers())) == 0, "HBitLinear should not register a per-module Hadamard buffer"

    print("HBitLinear shared Hadamard cache tests passed")
    return True


if __name__ == "__main__":
    test_hadamard_tensor_is_shared()
