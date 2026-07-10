"""Regression tests for Infini-Attention memory buffer updates."""

import contextlib

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import TernaryConfig
from layers.hybrid_block import HybridTransformerBlock
from model import BitNetDeep



def build_block() -> HybridTransformerBlock:
    config = TernaryConfig(
        vocab_size=1024,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        head_dim=32,
        intermediate_size=512,
        block_size=4,
        infini_memory_dim=8,
        attn_res_init_scale=0.1,
    )
    return HybridTransformerBlock(config)


def build_model() -> BitNetDeep:
    config = TernaryConfig(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=32,
        intermediate_size=256,
        block_size=4,
        infini_memory_dim=8,
        attn_res_init_scale=0.1,
    )
    return BitNetDeep(config)


def test_infini_attention_memory_updates() -> bool:
    torch.manual_seed(0)
    block = build_block()
    x = torch.randn(2, 8, block.hidden_size)

    block.train()
    block.infini_attn.reset_memory()
    initial_state = block.infini_attn.get_memory_state()
    _ = block(x)
    updated_state = block.infini_attn.get_memory_state()
    assert not torch.allclose(initial_state["memory_k"], updated_state["memory_k"]), "Training forward should update memory"
    assert not torch.allclose(initial_state["memory_v"], updated_state["memory_v"]), "Training forward should update memory"

    block.eval()
    eval_state = block.infini_attn.get_memory_state()
    _ = block(x)
    post_eval_state = block.infini_attn.get_memory_state()
    assert torch.allclose(eval_state["memory_k"], post_eval_state["memory_k"]), "Eval forward should not mutate memory_k"
    assert torch.allclose(eval_state["memory_v"], post_eval_state["memory_v"]), "Eval forward should not mutate memory_v"

    print("InfiniAttention memory update gating tests passed")
    return True


def test_checkpoint_recompute_does_not_double_update_memory() -> bool:
    torch.manual_seed(1)
    reference_block = build_block()
    checkpoint_block = build_block()
    checkpoint_block.load_state_dict(reference_block.state_dict())

    x = torch.randn(2, 8, reference_block.hidden_size, requires_grad=True)
    x_checkpoint = x.detach().clone().requires_grad_(True)

    reference_block.train()
    reference_block.infini_attn.reset_memory()
    reference_output = reference_block(x)
    reference_output.mean().backward()
    reference_state = reference_block.infini_attn.get_memory_state()

    checkpoint_block.train()
    checkpoint_block.infini_attn.reset_memory()
    checkpoint_initial_state = checkpoint_block.infini_attn.get_memory_state()
    checkpoint_output = checkpoint(
        lambda hidden_states: checkpoint_block(hidden_states),
        x_checkpoint,
        use_reentrant=False,
        context_fn=lambda: (
            contextlib.nullcontext(),
            checkpoint_block.infini_attn.use_memory_state(checkpoint_initial_state, update_memory_buffers=False),
        ),
    )
    checkpoint_output.mean().backward()
    checkpoint_state = checkpoint_block.infini_attn.get_memory_state()

    assert torch.allclose(reference_state["memory_k"], checkpoint_state["memory_k"], atol=1e-6, rtol=1e-5), (
        "Checkpoint recomputation should not apply a second memory_k update"
    )
    assert torch.allclose(reference_state["memory_v"], checkpoint_state["memory_v"], atol=1e-6, rtol=1e-5), (
        "Checkpoint recomputation should not apply a second memory_v update"
    )

    print("Checkpoint memory update regression tests passed")
    return True


def test_model_forward_resets_memory_between_calls() -> bool:
    torch.manual_seed(2)
    model = build_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))

    model.train()
    first = model(input_ids)
    first_state = model.layers[0].infini_attn.get_memory_state()
    second = model(input_ids)

    assert torch.allclose(first, second, atol=1e-6, rtol=1e-5), (
        "Top-level model forward should reset InfiniAttention memory between unrelated calls"
    )
    assert not torch.allclose(first_state["memory_k"], torch.zeros_like(first_state["memory_k"])), (
        "Training forward should still populate transient memory during the call"
    )

    print("Top-level model memory reset tests passed")
    return True


def test_infini_attention_memory_is_not_serialized() -> bool:
    torch.manual_seed(3)
    block = build_block()
    block.train()
    _ = block(torch.randn(2, 8, block.hidden_size))

    state_dict = block.state_dict()
    assert "infini_attn.memory_k" not in state_dict, "Transient memory_k should not be serialized"
    assert "infini_attn.memory_v" not in state_dict, "Transient memory_v should not be serialized"

    # Legacy checkpoints may still contain these keys; loading should ignore them.
    state_dict["infini_attn.memory_k"] = torch.ones_like(block.infini_attn.memory_k)
    state_dict["infini_attn.memory_v"] = torch.ones_like(block.infini_attn.memory_v)

    restored = build_block()
    restored.load_state_dict(state_dict)
    restored_state = restored.infini_attn.get_memory_state()
    assert torch.count_nonzero(restored_state["memory_k"]) == 0, "Legacy memory_k should be discarded on load"
    assert torch.count_nonzero(restored_state["memory_v"]) == 0, "Legacy memory_v should be discarded on load"

    print("Transient InfiniAttention checkpoint-state tests passed")
    return True


def _assert_checkpoint_matches_reference(granularity: str) -> None:
    torch.manual_seed(4)
    config = TernaryConfig(
        vocab_size=128,
        hidden_size=64,
        num_prelude_layers=1,
        num_recurrent_layers=2,
        num_coda_layers=1,
        num_loops=2,
        num_attention_heads=4,
        head_dim=16,
        intermediate_size=128,
        block_size=4,
        infini_memory_dim=8,
        attn_res_init_scale=0.1,
        use_hadamard=False,
    )
    reference_model = BitNetDeep(config)
    checkpoint_model = BitNetDeep(config)
    checkpoint_model.load_state_dict(reference_model.state_dict())

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = torch.randint(0, config.vocab_size, (2, 8))

    reference_model.gradient_checkpointing = False
    checkpoint_model.gradient_checkpointing = True
    checkpoint_model.checkpoint_granularity = granularity
    reference_model.train()
    checkpoint_model.train()

    reference_logits = reference_model(input_ids)
    checkpoint_logits = checkpoint_model(input_ids)
    reference_loss = F.cross_entropy(
        reference_logits.reshape(-1, reference_logits.size(-1)), labels.reshape(-1)
    )
    checkpoint_loss = F.cross_entropy(
        checkpoint_logits.reshape(-1, checkpoint_logits.size(-1)), labels.reshape(-1)
    )
    reference_loss.backward()
    checkpoint_loss.backward()

    assert torch.allclose(reference_loss, checkpoint_loss, atol=1e-5, rtol=1e-4), (
        f"gradient checkpointing ({granularity}) should preserve the training loss"
    )
    for (reference_name, reference_param), (checkpoint_name, checkpoint_param) in zip(
        reference_model.named_parameters(),
        checkpoint_model.named_parameters(),
    ):
        assert reference_name == checkpoint_name
        assert reference_param.grad is not None, f"Expected gradient for {reference_name}"
        assert checkpoint_param.grad is not None, f"Expected checkpoint gradient for {checkpoint_name}"
        assert torch.allclose(
            reference_param.grad, checkpoint_param.grad, atol=1e-5, rtol=1e-4
        ), f"Checkpointed training should match reference grad for {reference_name} ({granularity})"


def test_training_wrapper_checkpointing_matches_reference_gradients() -> bool:
    _assert_checkpoint_matches_reference("layer")
    print("BitNetDeep layer-granularity checkpoint gradient regression tests passed")
    return True


def test_loop_granularity_checkpointing_matches_reference_gradients() -> bool:
    _assert_checkpoint_matches_reference("loop")
    print("BitNetDeep loop-granularity checkpoint gradient regression tests passed")
    return True


if __name__ == "__main__":
    test_infini_attention_memory_updates()
    test_checkpoint_recompute_does_not_double_update_memory()
    test_model_forward_resets_memory_between_calls()
    test_training_wrapper_checkpointing_matches_reference_gradients()
    test_loop_granularity_checkpointing_matches_reference_gradients()
    test_infini_attention_memory_is_not_serialized()
    test_training_wrapper_checkpointing_matches_reference_gradients()
