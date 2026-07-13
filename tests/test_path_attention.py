"""PaTH-FoX equation and bounded-window checks."""

import torch

from config import TernaryConfig
from layers.infini_attention import InfiniAttention


def _attention() -> InfiniAttention:
    return InfiniAttention(TernaryConfig(
        vocab_size=64,
        hidden_size=12,
        num_hidden_layers=1,
        num_attention_heads=3,
        head_dim=4,
        intermediate_size=24,
        block_size=1,
        path_window_size=4,
        infini_memory_dim=4,
    ))


def _sequential_path_fox(q, k, v, w, beta, log_forget):
    effective_keys = []
    gate_prefixes = []
    gate_total = torch.zeros_like(log_forget[:, 0])
    outputs = []
    scale = q.size(-1) ** -0.5

    for index in range(q.size(2)):
        direction = w[:, index]
        strength = beta[:, index]
        effective_keys = [
            key - strength.unsqueeze(-1) * (key * direction).sum(-1, keepdim=True) * direction
            for key in effective_keys
        ]
        effective_keys.append(k[:, :, index])
        gate_total = gate_total + log_forget[:, index]
        gate_prefixes.append(gate_total)

        keys = torch.stack(effective_keys, dim=2)
        logits = torch.einsum("bhd,bhsd->bhs", q[:, :, index], keys) * scale
        prefixes = torch.stack(gate_prefixes, dim=-1)
        logits = logits + gate_total.unsqueeze(-1) - prefixes
        probabilities = logits.softmax(dim=-1)
        outputs.append(torch.einsum("bhs,bhsd->bhd", probabilities, v[:, :, : index + 1]))
    return torch.stack(outputs, dim=2)


def test_path_fox_ut_matches_sequential_equations() -> None:
    torch.manual_seed(7)
    attention = _attention()
    q = torch.randn(2, 3, 4, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    w = torch.nn.functional.normalize(torch.randn(2, 4, 3, 4), dim=-1)
    beta = 2 * torch.sigmoid(torch.randn(2, 4, 3))
    log_forget = torch.nn.functional.logsigmoid(torch.randn(2, 4, 3))

    expected = _sequential_path_fox(q, k, v, w, beta, log_forget)
    actual = attention._path_chunk(q, k, v, w, beta, log_forget, None)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_path_windows_bound_attention_and_keep_document_conv_isolated() -> None:
    torch.manual_seed(8)
    attention = _attention()
    x = torch.randn(1, 12, 12)
    segment_ids = torch.tensor([[0] * 6 + [1] * 6])
    changed = x.clone()
    changed[:, :6] += 10

    first = attention._path_vectors(x, segment_ids)
    second = attention._path_vectors(changed, segment_ids)
    torch.testing.assert_close(first.norm(dim=-1), torch.ones_like(first[..., 0]))
    torch.testing.assert_close(first[:, 6:], second[:, 6:])
    assert list(attention._chunk_ranges(12)) == [(0, 4), (4, 8), (8, 12)]


def test_path_parameters_receive_gradients() -> None:
    torch.manual_seed(9)
    attention = _attention()
    attention(torch.randn(1, 8, 12), update_memory=False).square().mean().backward()
    assert attention.path_beta.weight.grad is not None
    assert attention.path_forget.weight.grad is not None
    assert attention.path_conv_weight.grad is not None
    assert attention.path_w_down.weight.grad is not None


def test_infini_memory_bridges_path_windows_without_cross_document_leak() -> None:
    torch.manual_seed(10)
    attention = _attention()
    with torch.no_grad():
        attention.gate.fill_(10)
    baseline = torch.randn(1, 8, 12)
    perturbed = baseline.clone()
    perturbed[:, 0] += 5

    attention.reset_memory()
    first = attention(baseline)
    attention.reset_memory()
    second = attention(perturbed)
    assert not torch.allclose(first[:, 4:], second[:, 4:])

    attention.reset_memory()
    segment_ids = torch.tensor([[0] * 4 + [1] * 4])
    attention(baseline, segment_ids=segment_ids)
    assert not bool(attention.memory_initialized.any())
