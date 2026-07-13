"""Checks for DeepSeek Engram-style conditional memory."""

import torch

from config import TernaryConfig
from layers.engram import Engram
from model import BitNetDeep


def _config(**overrides) -> TernaryConfig:
    values = dict(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=8,
        intermediate_size=32,
        block_size=1,
        path_window_size=8,
        infini_memory_dim=8,
        use_hadamard=False,
        use_4bit_activations=False,
        engram_layer_ids=(0,),
        engram_vocab_size=31,
        engram_num_heads=2,
        engram_head_dim=4,
        engram_kernel_size=3,
    )
    values.update(overrides)
    return TernaryConfig(**values)


def test_engram_respects_packed_document_boundaries() -> None:
    torch.manual_seed(0)
    engram = Engram(_config(), layer_id=0).eval()
    hidden = torch.randn(1, 5, 16)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    changed = input_ids.clone()
    changed[0, 1] = 9
    segments = torch.tensor([[0, 0, 1, 1, 1]])

    first = engram(hidden, input_ids, segment_ids=segments)
    second = engram(hidden, changed, segment_ids=segments)
    assert torch.allclose(first[:, 2:], second[:, 2:]), (
        "Engram lookup and short convolution must not cross packed documents"
    )

    unsegmented = engram(hidden, input_ids)
    one_segment = engram(hidden, input_ids, segment_ids=torch.zeros_like(input_ids))
    assert torch.allclose(unsegmented, one_segment, atol=1e-6)


def test_bitnet_engram_forward_and_gradient() -> None:
    torch.manual_seed(0)
    model = BitNetDeep(_config())
    input_ids = torch.randint(0, 32, (2, 8))
    logits = model(input_ids)
    assert logits.shape == (2, 8, 32)
    logits.mean().backward()
    engram = model.layers[0].engram
    assert engram is not None
    assert engram.embedding.weight.grad is not None
