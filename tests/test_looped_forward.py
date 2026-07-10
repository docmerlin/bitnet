"""Tests for prelude / recurrent / coda looped forward."""

from __future__ import annotations

import torch

from config import TernaryConfig
from model import BitNetDeep


def _tiny_looped(
    prelude: int = 1,
    recurrent: int = 2,
    coda: int = 1,
    num_loops: int = 2,
) -> TernaryConfig:
    return TernaryConfig(
        vocab_size=128,
        hidden_size=64,
        num_attention_heads=4,
        head_dim=16,
        intermediate_size=128,
        block_size=4,
        infini_memory_dim=8,
        use_hadamard=False,
        use_4bit_activations=False,
        num_prelude_layers=prelude,
        num_recurrent_layers=recurrent,
        num_coda_layers=coda,
        num_loops=num_loops,
    )


def test_default_config_is_looped_structure() -> bool:
    cfg = TernaryConfig()
    assert cfg.num_prelude_layers == 8
    assert cfg.num_recurrent_layers == 48
    assert cfg.num_coda_layers == 8
    assert cfg.num_loops == 4
    assert cfg.num_hidden_layers == 64
    assert cfg.effective_depth == 8 + 48 * 4 + 8
    print("Default looped TernaryConfig structure tests passed")
    return True


def test_flat_num_hidden_layers_compat() -> bool:
    cfg = TernaryConfig(
        vocab_size=32,
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        head_dim=16,
        intermediate_size=128,
        use_hadamard=False,
    )
    assert cfg.num_prelude_layers == 0
    assert cfg.num_recurrent_layers == 3
    assert cfg.num_coda_layers == 0
    assert cfg.num_loops == 1
    assert cfg.num_hidden_layers == 3
    assert cfg.effective_depth == 3
    print("Flat num_hidden_layers compat tests passed")
    return True


def test_looped_forward_shapes_and_override() -> bool:
    torch.manual_seed(0)
    config = _tiny_looped(num_loops=2)
    model = BitNetDeep(config)
    # Amplify sublayer outs so multi-loop dynamics are visible (init is near-identity).
    with torch.no_grad():
        for layer in model.layers:
            layer.infini_attn.o_proj.weight.mul_(50.0)
            layer.ffn_down.weight.mul_(50.0)
            layer.attn_res.scale.fill_(1.0)
            layer.mlp_res.scale.fill_(1.0)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        logits_r2 = model(input_ids)
        logits_r1 = model(input_ids, num_loops=1)
        logits_r3 = model(input_ids, num_loops=3)

    assert logits_r2.shape == (2, 8, config.vocab_size)
    assert logits_r1.shape == logits_r2.shape == logits_r3.shape
    # Different unroll depths should change outputs when blocks are non-identity.
    assert not torch.allclose(logits_r1, logits_r2, atol=1e-5), (
        "Expected R=1 and R=2 logits to differ"
    )
    assert not torch.allclose(logits_r2, logits_r3, atol=1e-5), (
        "Expected R=2 and R=3 logits to differ"
    )
    print("Looped forward shape / override tests passed")
    return True


def test_param_count_independent_of_loops() -> bool:
    cfg_r1 = _tiny_looped(num_loops=1)
    cfg_r4 = _tiny_looped(num_loops=4)
    m1 = BitNetDeep(cfg_r1)
    m4 = BitNetDeep(cfg_r4)
    n1 = sum(p.numel() for p in m1.parameters())
    n4 = sum(p.numel() for p in m4.parameters())
    assert n1 == n4, "Parameter count must not depend on num_loops"
    assert len(m1.layers) == cfg_r1.num_hidden_layers == 4
    print("Param count independent of loops tests passed")
    return True


def test_layer_application_count() -> bool:
    """Spy HybridTransformerBlock.forward call count: P + R*loops + C."""
    config = _tiny_looped(prelude=1, recurrent=2, coda=1, num_loops=3)
    model = BitNetDeep(config)
    model.eval()

    calls = {"n": 0}
    for layer in model.layers:
        orig = layer.forward

        def make_counting(orig_fn):
            def counting(*args, **kwargs):
                calls["n"] += 1
                return orig_fn(*args, **kwargs)

            return counting

        layer.forward = make_counting(orig)  # type: ignore[method-assign]

    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    with torch.no_grad():
        model(input_ids)

    expected = 1 + 2 * 3 + 1
    assert calls["n"] == expected, f"expected {expected} block apps, got {calls['n']}"

    calls["n"] = 0
    with torch.no_grad():
        model(input_ids, num_loops=1)
    assert calls["n"] == 1 + 2 * 1 + 1
    print("Layer application count tests passed")
    return True


def test_infini_memory_write_only_last_loop() -> bool:
    """Policy B: early loops read only; compressive memory writes on last loop only."""
    torch.manual_seed(1)
    config = _tiny_looped(prelude=0, recurrent=1, coda=0, num_loops=3)
    model = BitNetDeep(config)
    model.train()
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    attn = model.layers[0].infini_attn

    # Spy: count real buffer updates (training path).
    writes = {"n": 0}
    orig = attn._update_memory

    def counting_update(k, v):
        writes["n"] += 1
        return orig(k, v)

    attn._update_memory = counting_update  # type: ignore[method-assign]

    with torch.no_grad():
        model(input_ids, reset_memory=True, num_loops=3)

    # One middle layer × only last of 3 loops → exactly one write.
    assert writes["n"] == 1, f"expected 1 memory write (last loop only), got {writes['n']}"
    assert not torch.allclose(attn.memory_k, torch.zeros_like(attn.memory_k)), (
        "last loop should leave non-zero Infini memory"
    )

    writes["n"] = 0
    with torch.no_grad():
        model(input_ids, reset_memory=True, num_loops=1)
    assert writes["n"] == 1, "R=1 still writes once (that loop is the last)"

    print("Infini memory write-only-last-loop tests passed")
    return True


if __name__ == "__main__":
    test_default_config_is_looped_structure()
    test_flat_num_hidden_layers_compat()
    test_looped_forward_shapes_and_override()
    test_param_count_independent_of_loops()
    test_layer_application_count()
    test_infini_memory_write_only_last_loop()
