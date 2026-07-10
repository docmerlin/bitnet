"""Tests for Hyperloop-style loop-boundary hyper-connections."""

from __future__ import annotations

import torch

from config import TernaryConfig
from layers.loop_mhc import NUM_STREAMS, LoopHyperConnection
from model import BitNetDeep


def _tiny(**kwargs) -> TernaryConfig:
    base = dict(
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=4,
        head_dim=8,
        intermediate_size=64,
        use_hadamard=False,
        use_4bit_activations=False,
        num_prelude_layers=0,
        num_recurrent_layers=2,
        num_coda_layers=0,
        num_loops=2,
    )
    base.update(kwargs)
    return TernaryConfig(**base)


def test_hardcoded_four_streams() -> bool:
    hc = LoopHyperConnection(hidden_size=16)
    assert hc.num_streams == 4 == NUM_STREAMS
    x = torch.randn(2, 5, 16)
    y = hc.expand(x)
    assert y.shape == (2, 5, 4, 16)
    assert torch.allclose(y[:, :, 0, :], x)
    assert torch.allclose(y[:, :, 3, :], x)
    print("Hardcoded 4-stream expand tests passed")
    return True


def test_diagonal_h_res_not_dense() -> bool:
    """H_res is per-stream diagonal scale, not a dense Sinkhorn matrix."""
    torch.manual_seed(0)
    hc = LoopHyperConnection(hidden_size=16)
    y = hc.expand(torch.randn(1, 3, 16))
    x_in, h_pre, h_post, h_res = hc.project_in(y)
    assert h_pre.shape == (1, 3, 4)
    assert h_post.shape == (1, 3, 4)
    assert h_res.shape == (1, 3, 4)  # n diagonal entries, not n×n
    assert x_in.shape == (1, 3, 16)

    u = torch.randn_like(x_in)
    y2 = hc.write_back(y, u, h_post, h_res)
    # Manual: y2[i] = h_res[i]*y[i] + h_post[i]*u
    expected = h_res.unsqueeze(-1) * y + h_post.unsqueeze(-1) * u.unsqueeze(2)
    assert torch.allclose(y2, expected, atol=1e-5)
    print("Diagonal H_res write-back tests passed")
    return True


def test_identity_friendly_init_near_pass_through() -> bool:
    """At init: H_res≈1, H_post≈0 → streams barely change if u is ignored."""
    hc = LoopHyperConnection(hidden_size=16)
    x = torch.randn(2, 4, 16)
    y = hc.expand(x)
    _x_in, _pre, h_post, h_res = hc.project_in(y)
    assert torch.all(h_res > 0.9), f"H_res should be near 1, got min {h_res.min()}"
    assert torch.all(h_post < 0.2), f"H_post should be near 0, got max {h_post.max()}"
    print("Identity-friendly init tests passed")
    return True


def test_loop_hc_runs_every_loop_including_r1() -> bool:
    torch.manual_seed(1)
    model = BitNetDeep(_tiny(num_loops=1))
    calls = {"n": 0}
    orig = model.loop_hc.project_in

    def counting(y):
        calls["n"] += 1
        return orig(y)

    model.loop_hc.project_in = counting  # type: ignore[method-assign]
    model.eval()
    with torch.no_grad():
        model(torch.randint(0, 64, (1, 6)))
    assert calls["n"] == 1, f"R=1 should still apply one loop HC, got {calls['n']}"

    calls["n"] = 0
    model2 = BitNetDeep(_tiny(num_loops=3))
    orig2 = model2.loop_hc.project_in

    def counting2(y):
        calls["n"] += 1
        return orig2(y)

    model2.loop_hc.project_in = counting2  # type: ignore[method-assign]
    model2.eval()
    with torch.no_grad():
        model2(torch.randint(0, 64, (1, 6)))
    assert calls["n"] == 3
    print("Loop HC per-iteration tests passed")
    return True


def test_loop_hc_forward_and_grad() -> bool:
    torch.manual_seed(2)
    model = BitNetDeep(_tiny(num_prelude_layers=1, num_coda_layers=1, num_loops=3))
    model.train()
    logits = model(torch.randint(0, 64, (2, 8)))
    assert logits.shape == (2, 8, 64)
    loss = logits.float().mean()
    loss.backward()
    assert model.loop_hc.w_res.weight.grad is not None
    assert float(model.loop_hc.w_res.weight.grad.abs().sum()) > 0.0
    assert model.loop_hc.loop_embed.weight.grad is not None
    print("Loop HC forward/grad tests passed")
    return True


def test_loop_embedding_changes_output() -> bool:
    torch.manual_seed(3)
    model = BitNetDeep(_tiny(num_loops=2))
    # Break post≈0 init so writes are visible, then tweak loop emb.
    with torch.no_grad():
        model.loop_hc.w_post.bias.fill_(0.0)  # H_post ≈ 2*0.5 = 1
    ids = torch.randint(0, 64, (1, 6))
    model.eval()
    with torch.no_grad():
        y0 = model(ids)
        model.loop_hc.loop_embed.weight[0].fill_(1.0)
        y1 = model(ids)
    assert not torch.allclose(y0, y1, atol=1e-5), "loop embedding should affect logits"
    print("Loop embedding effect tests passed")
    return True


def test_no_config_knobs_for_hc() -> bool:
    cfg = TernaryConfig(
        vocab_size=32,
        hidden_size=32,
        num_attention_heads=4,
        head_dim=8,
        intermediate_size=64,
        num_hidden_layers=2,
        use_hadamard=False,
    )
    fields = cfg.__dataclass_fields__
    assert "use_loop_mhc" not in fields
    assert "loop_mhc_streams" not in fields
    assert "loop_mhc_sinkhorn_iters" not in fields
    model = BitNetDeep(cfg)
    assert isinstance(model.loop_hc, LoopHyperConnection)
    assert model.loop_hc.num_streams == 4
    print("No HC config knobs tests passed")
    return True


if __name__ == "__main__":
    test_hardcoded_four_streams()
    test_diagonal_h_res_not_dense()
    test_identity_friendly_init_near_pass_through()
    test_loop_hc_runs_every_loop_including_r1()
    test_loop_hc_forward_and_grad()
    test_loop_embedding_changes_output()
    test_no_config_knobs_for_hc()
