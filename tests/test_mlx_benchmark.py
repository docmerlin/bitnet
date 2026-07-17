"""Minimal checks for the representative MLX benchmark."""

from argparse import Namespace
from dataclasses import replace

import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from mlx_benchmark import run_mlx, validate_args
from mlx_model import MLXBitNet, MLXBitNetConfig, MLXPaTHAttention, MLXRFMoE
from mlx_optim import CMUD
from mlx_path_kernel import path_triangular_solve, reference_triangular_solve
from mlx_rfmoe_kernel import masked_grouped_linear
from mlx_train import (
    _gradient_compile_safe,
    build_parser,
    create_apply_step,
    create_gradient_step,
    create_train_step,
    load_checkpoint,
    save_checkpoint,
    validate_args as validate_training_args,
)


def test_path_metal_kernel_matches_forward_and_gradients() -> None:
    mx.random.seed(7)
    lower = mx.tril(mx.random.normal((2, 3, 8, 8)) * 0.02, k=-1)
    matrix = mx.eye(8) + lower
    rhs = mx.random.normal(matrix.shape)

    expected = reference_triangular_solve(matrix, rhs)
    actual = path_triangular_solve(matrix, rhs)
    mx.eval(expected, actual)
    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()

    def loss(solve, a, b):
        return mx.square(solve(a, b)).sum()

    reference_grads = mx.grad(lambda a, b: loss(reference_triangular_solve, a, b), argnums=(0, 1))(
        matrix, rhs
    )
    metal_grads = mx.grad(lambda a, b: loss(path_triangular_solve, a, b), argnums=(0, 1))(matrix, rhs)
    mx.eval(reference_grads, metal_grads)
    assert mx.allclose(metal_grads[0], reference_grads[0], rtol=1e-4, atol=1e-4).item()
    assert mx.allclose(metal_grads[1], reference_grads[1], rtol=1e-4, atol=1e-4).item()


def test_rfmoe_masked_grouped_metal_matches_forward_and_gradients() -> None:
    x = mx.random.normal((3, 5, 8))
    weight = mx.random.normal((3, 4, 8))
    active = mx.array(
        [
            [True, False, True, False, True],
            [False, True, True, False, False],
            [True, True, False, False, True],
        ]
    )

    def reference(values, weights):
        return (values @ weights.swapaxes(-1, -2)) * active[..., None]

    expected = reference(x, weight)
    actual = masked_grouped_linear(x, weight, active)
    expected_gradients = mx.grad(lambda values, weights: mx.square(reference(values, weights)).sum(), argnums=(0, 1))(
        x, weight
    )
    actual_gradients = mx.grad(
        lambda values, weights: mx.square(masked_grouped_linear(values, weights, active)).sum(),
        argnums=(0, 1),
    )(x, weight)
    mx.eval(expected, actual, expected_gradients, actual_gradients)

    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()
    assert mx.allclose(actual_gradients[0], expected_gradients[0], rtol=1e-4, atol=1e-4).item()
    assert mx.allclose(actual_gradients[1], expected_gradients[1], rtol=1e-4, atol=1e-4).item()


def test_mlx_training_step_is_finite() -> None:
    args = Namespace(
        backend="mlx",
        steps=1,
        warmup_steps=0,
        batch_size=1,
        sequence_length=4,
        vocab_size=32,
        hidden_size=8,
        num_heads=2,
        intermediate_size=16,
        num_layers=1,
        path_window_size=4,
        learning_rate=1e-3,
        mlx_dtype="float32",
        mlx_path_kernel=True,
    )
    validate_args(args)
    metrics = run_mlx(args)
    assert metrics["loss"] > 0
    assert metrics["tokens_per_second"] > 0


def test_mlx_training_defaults_use_fast_local_batch() -> None:
    args = build_parser().parse_args([])
    assert args.micro_batch_size == 4
    assert args.grad_accumulation_steps == 4
    assert not args.gradient_checkpointing
    assert args.validation_batches == 5
    assert args.mud_block_size == 256


def test_mlx_gradient_compile_avoids_irregular_blocks_and_hybrid_rfmoe() -> None:
    dense = MLXBitNetConfig()
    hybrid = replace(dense, use_rfmoe=True, rfmoe_backend="hybrid")
    assert _gradient_compile_safe(dense, True, 512, 8)
    assert _gradient_compile_safe(dense, True, 512, 16)
    assert not _gradient_compile_safe(dense, True, 512, 9)
    assert not _gradient_compile_safe(hybrid, True, 512, 8)

    args = build_parser().parse_args(["--final-blocks", "0"])
    with pytest.raises(ValueError, match="blocks must be positive"):
        validate_training_args(args)


def test_mlx_model_preserves_packed_document_boundaries_and_loops() -> None:
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=0,
        num_recurrent_layers=1,
        num_coda_layers=0,
        num_loops=2,
        path_window_size=4,
    )
    model = MLXBitNet(config)
    segments = mx.array([[0, 0, 1, 1]])
    first = model(mx.array([[1, 2, 3, 4]]), segments, num_loops=1)
    changed_previous_document = model(mx.array([[5, 6, 3, 4]]), segments, num_loops=1)
    recurrent = model(mx.array([[1, 2, 3, 4]]), segments, num_loops=2)
    mx.eval(first, changed_previous_document, recurrent)

    assert mx.allclose(first[:, 2:], changed_previous_document[:, 2:], rtol=1e-5, atol=1e-5).item()
    assert not mx.allclose(first, recurrent).item()


def test_mlx_packed_train_step_is_finite() -> None:
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=0,
        num_recurrent_layers=1,
        num_coda_layers=0,
        num_loops=1,
        path_window_size=4,
    )
    model = MLXBitNet(config)
    optimizer = CMUD(
        mud_learning_rate=1e-3,
        fallback_learning_rate=3e-4,
        weight_decay=0.0,
        eight_bit=False,
    )
    train_step, state = create_train_step(model, optimizer, compile_step=True)
    inputs = mx.array([[1, 2, 3, 4]])
    labels = mx.array([[2, 3, 4, 5]])
    segments = mx.array([[0, 0, 1, 1]])
    label_segments = mx.array([[0, 1, 1, 1]])
    loss = train_step(inputs, labels, segments, label_segments)
    mx.eval(loss, state)
    assert mx.isfinite(loss).item()


def test_mlx_compiled_apply_step_updates_model() -> None:
    config = MLXBitNetConfig(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=1,
        num_recurrent_layers=0,
        num_coda_layers=0,
        use_engram=False,
    )
    model = MLXBitNet(config)
    optimizer = CMUD(
        mud_learning_rate=1e-3,
        fallback_learning_rate=3e-4,
        weight_decay=0.0,
        eight_bit=False,
    )
    optimizer.init(model.trainable_parameters())
    apply_step, state = create_apply_step(model, optimizer, grad_clip=1.0, compile_step=True)
    before = mx.array(model.embedding.weight)
    gradients = tree_map(mx.ones_like, model.trainable_parameters())

    grad_norm = apply_step(gradients, mx.array(0.5))
    mx.eval(grad_norm, state)
    after_update = mx.array(model.embedding.weight)
    apply_step(gradients, mx.array(0.0))
    mx.eval(state)

    assert mx.isfinite(grad_norm).item()
    assert not mx.array_equal(after_update, before).item()
    assert mx.array_equal(model.embedding.weight, after_update).item()


@pytest.mark.parametrize(("length", "memory_dim"), [(4, 2), (2, 4), (3, 2)])
def test_mlx_infini_pooling_fast_paths_match_reference(length, memory_dim) -> None:
    config = MLXBitNetConfig(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=1,
        num_recurrent_layers=0,
        num_coda_layers=0,
        infini_memory_dim=memory_dim,
        use_engram=False,
    )
    attention = MLXPaTHAttention(config)
    keys = mx.arange(2 * length * 4, dtype=mx.float32).reshape(1, 2, length, 4)
    values = keys + 1
    expected_keys = []
    expected_values = []
    for index in range(memory_dim):
        start = index * length // memory_dim
        end = max(start + 1, ((index + 1) * length + memory_dim - 1) // memory_dim)
        expected_keys.append(mx.mean(keys[:, :, start:end], axis=2))
        expected_values.append(mx.mean(values[:, :, start:end], axis=2))
    expected_keys = 0.01 * mx.stack(expected_keys, axis=2)
    expected_values = 0.01 * mx.stack(expected_values, axis=2)
    shape = (1, 2, memory_dim, 4)

    actual_keys, actual_values, initialized = attention._next_memory(
        keys,
        values,
        mx.array([True]),
        mx.zeros(shape),
        mx.zeros(shape),
        mx.array([False]),
    )
    mx.eval(actual_keys, actual_values, initialized)

    assert mx.array_equal(actual_keys, expected_keys).item()
    assert mx.array_equal(actual_values, expected_values).item()
    assert initialized.item()


def test_mlx_compiled_irregular_infini_pooling_fits_metal_argument_buffer() -> None:
    config = MLXBitNetConfig(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=2,
        num_recurrent_layers=4,
        num_coda_layers=2,
        num_loops=2,
        block_size=9,
        path_window_size=64,
        infini_memory_dim=64,
        use_engram=False,
    )
    model = MLXBitNet(config)
    gradient_step = create_gradient_step(
        model,
        compile_step=True,
        num_loops=2,
        gradient_checkpointing=True,
    )
    tokens = mx.zeros((1, 512), dtype=mx.int32)
    segments = mx.zeros_like(tokens)

    loss, gradients = gradient_step(
        tokens,
        tokens,
        segments,
        segments,
        mx.array(0.0),
        mx.array(1.0),
        mx.array(0.1),
    )
    mx.eval(loss, gradients, model.state)

    assert mx.isfinite(loss).item()


def test_mlx_full_feature_model_states_and_heads() -> None:
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=0,
        num_recurrent_layers=1,
        num_coda_layers=0,
        num_loops=2,
        block_size=1,
        path_window_size=4,
        infini_memory_dim=4,
        use_engram=True,
        engram_layer_ids=(0,),
        engram_vocab_size=17,
        engram_num_heads=2,
        engram_head_dim=2,
        use_rfmoe=True,
        rfmoe_num_experts=2,
        rfmoe_expert_dim=4,
        rfmoe_rank=2,
        mtp_depth=2,
    )
    model = MLXBitNet(config)
    tokens = mx.array([[1, 2, 3, 4]])
    segments = mx.array([[0, 0, 1, 1]])
    logits, mtp = model(tokens, segments, return_mtp=True)
    density, locality, diversity, hard_density = model.rfmoe_aux_losses(1.0, 0.1)
    mx.eval(logits, mtp, density, locality, diversity, hard_density)
    assert logits.shape == (1, 4, 32)
    assert [head.shape for head in mtp] == [(1, 4, 32), (1, 4, 32)]
    assert 0.0 <= hard_density.item() <= 1.0
    assert mx.isfinite(locality).item() and mx.isfinite(diversity).item()

    attention = MLXPaTHAttention(config)
    attention.reset_memory(1)
    attention(mx.random.normal((1, 4, 8)), segment_ids=mx.zeros((1, 4), dtype=mx.int32), update_memory=True)
    mx.eval(attention.memory_initialized, attention.memory_k)
    assert attention.memory_initialized.item()
    assert mx.count_nonzero(attention.memory_k).item() > 0

    attention.reset_memory(2)
    mixed_segments = mx.array([[0, 0, 0, 0], [0, 0, 1, 1]])
    attention(mx.random.normal((2, 4, 8)), segment_ids=mixed_segments, update_memory=True)
    mx.eval(attention.memory_initialized)
    assert attention.memory_initialized.tolist() == [True, False]


def test_mlx_rfmoe_dispatches_only_fired_rows(monkeypatch) -> None:
    config = MLXBitNetConfig(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=1,
        num_recurrent_layers=0,
        num_coda_layers=0,
        use_engram=False,
        use_hadamard=False,
        use_rfmoe=True,
        rfmoe_num_experts=4,
        rfmoe_expert_dim=4,
        rfmoe_rank=2,
        rfmoe_backend="host",
    )
    moe = MLXRFMoE(config)
    moe.experts[0].bias = mx.array([100.0])
    for expert in moe.experts[1:]:
        expert.bias = mx.array([-1.0])
    dispatched = []
    gather_mm = mx.gather_mm

    def record_dispatch(*args, **kwargs):
        dispatched.append(kwargs["rhs_indices"].size)
        return gather_mm(*args, **kwargs)

    monkeypatch.setattr(mx, "gather_mm", record_dispatch)
    inputs = mx.random.normal((2, 3, 8))
    loss, gradients = nn.value_and_grad(
        moe,
        lambda values: mx.sum(moe(values)),
    )(inputs)
    mx.eval(loss, gradients)

    assert dispatched == [18, 18, 18, 18]
    assert mx.isfinite(loss).item()
    assert moe.last_density.item() == 0.75


def test_mlx_rfmoe_auto_backend_prefers_hybrid_with_metal() -> None:
    config = MLXBitNetConfig(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=1,
        num_recurrent_layers=0,
        num_coda_layers=0,
        use_engram=False,
        use_rfmoe=True,
        rfmoe_num_experts=2,
        rfmoe_expert_dim=4,
        rfmoe_rank=2,
    )
    assert MLXRFMoE(config).backend == ("hybrid" if mx.metal.is_available() else "host")


def test_mlx_rfmoe_hybrid_handles_empty_dispatch() -> None:
    config = MLXBitNetConfig(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=1,
        num_recurrent_layers=0,
        num_coda_layers=0,
        use_engram=False,
        use_hadamard=False,
        use_rfmoe=True,
        rfmoe_num_experts=2,
        rfmoe_expert_dim=4,
        rfmoe_rank=2,
        rfmoe_backend="hybrid",
    )
    moe = MLXRFMoE(config)
    for expert in moe.experts:
        expert.bias = mx.array([100.0])
    inputs = mx.random.normal((2, 3, 8))

    loss, gradients = nn.value_and_grad(moe, lambda values: mx.square(moe(values)).sum())(inputs)
    mx.eval(loss, gradients)

    assert loss.item() == 0.0
    assert all(mx.all(mx.isfinite(value)).item() for _, value in tree_flatten(gradients))


@pytest.mark.parametrize("backend", ["metal", "hybrid"])
def test_mlx_rfmoe_accelerated_backends_match_host_compaction(backend) -> None:
    config = MLXBitNetConfig(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=1,
        num_recurrent_layers=0,
        num_coda_layers=0,
        use_engram=False,
        use_hadamard=False,
        use_rfmoe=True,
        rfmoe_num_experts=4,
        rfmoe_expert_dim=4,
        rfmoe_rank=2,
    )
    accelerated = MLXRFMoE(replace(config, rfmoe_backend=backend))
    host = MLXRFMoE(replace(config, rfmoe_backend="host"))
    host.load_weights(list(tree_flatten(accelerated.parameters())))
    for index, (accelerated_expert, host_expert) in enumerate(zip(accelerated.experts, host.experts)):
        bias = mx.array([100.0 if index == 0 else -1.0])
        accelerated_expert.bias = bias
        host_expert.bias = bias
    inputs = mx.random.normal((2, 3, 8))

    accelerated_loss, accelerated_gradients = nn.value_and_grad(
        accelerated,
        lambda values: mx.square(accelerated(values)).sum(),
    )(inputs)
    host_loss, host_gradients = nn.value_and_grad(
        host,
        lambda values: mx.square(host(values)).sum(),
    )(inputs)
    mx.eval(accelerated_loss, accelerated_gradients, host_loss, host_gradients)

    assert mx.allclose(accelerated_loss, host_loss, rtol=1e-4, atol=1e-5).item()
    expected = dict(tree_flatten(host_gradients))
    actual = dict(tree_flatten(accelerated_gradients))
    assert actual.keys() == expected.keys()
    assert all(mx.allclose(actual[key], value, rtol=1e-4, atol=1e-5).item() for key, value in expected.items())
    assert accelerated.last_density.item() == host.last_density.item() == 0.75


def test_mlx_activation_checkpointing_preserves_gradients_and_state() -> None:
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=1,
        num_recurrent_layers=0,
        num_coda_layers=0,
        block_size=1,
        path_window_size=4,
        infini_memory_dim=4,
        use_engram=True,
        engram_layer_ids=(0,),
        engram_vocab_size=17,
        engram_num_heads=2,
        engram_head_dim=2,
        use_rfmoe=True,
        rfmoe_num_experts=2,
        rfmoe_expert_dim=4,
        rfmoe_rank=2,
    )
    reference = MLXBitNet(config)
    checkpointed = MLXBitNet(config)
    checkpointed.load_weights(list(tree_flatten(reference.parameters())))
    tokens = mx.array([[1, 2, 3, 4]])
    segments = mx.zeros_like(tokens)

    reference_loss, reference_gradients = nn.value_and_grad(
        reference,
        lambda ids, segment_ids: mx.mean(mx.square(reference(ids, segment_ids))),
    )(tokens, segments)
    checkpointed_loss, checkpointed_gradients = nn.value_and_grad(
        checkpointed,
        lambda ids, segment_ids: mx.mean(
            mx.square(checkpointed(ids, segment_ids, checkpoint_activations=True))
        ),
    )(tokens, segments)
    mx.eval(reference_loss, reference_gradients, checkpointed_loss, checkpointed_gradients)

    assert mx.allclose(checkpointed_loss, reference_loss, rtol=1e-5, atol=1e-5).item()
    expected = dict(tree_flatten(reference_gradients))
    actual = dict(tree_flatten(checkpointed_gradients))
    assert actual.keys() == expected.keys()
    assert all(mx.allclose(actual[key], value, rtol=1e-4, atol=1e-5).item() for key, value in expected.items())
    assert mx.allclose(
        checkpointed.blocks[0].attn.memory_k,
        reference.blocks[0].attn.memory_k,
        rtol=1e-5,
        atol=1e-6,
    ).item()
    assert mx.allclose(
        checkpointed.blocks[0].moe.usage_ema,
        reference.blocks[0].moe.usage_ema,
        rtol=1e-5,
        atol=1e-6,
    ).item()


def test_mlx_checkpoint_restores_parameters_optimizer_and_state(tmp_path) -> None:
    config = MLXBitNetConfig(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=1,
        num_recurrent_layers=0,
        num_coda_layers=0,
        num_loops=1,
        block_size=1,
        path_window_size=4,
        use_engram=False,
    )
    model = MLXBitNet(config)
    optimizer = CMUD(
        mud_learning_rate=1e-3,
        fallback_learning_rate=3e-4,
        weight_decay=0.0,
        eight_bit=False,
    )
    train_step, state = create_train_step(model, optimizer, compile_step=False)
    batch = mx.array([[1, 2, 3, 4]])
    segments = mx.zeros_like(batch)
    loss = train_step(batch, batch, segments, segments)
    mx.eval(loss, state)
    expected = dict(model.parameters())["embedding"]["weight"]
    checkpoint = save_checkpoint(
        tmp_path,
        model,
        optimizer,
        config,
        {"step": 3, "tokens_processed": 12},
        "test",
    )
    assert not any(key.endswith((".memory_k", ".memory_v", ".memory_initialized")) for key in mx.load(str(checkpoint)))
    expected_optimizer = dict(tree_flatten(optimizer.state))
    expected_random = mx.random.uniform(shape=(4,))
    mx.eval(expected_random)

    restored = MLXBitNet(config)
    restored_optimizer = CMUD(
        mud_learning_rate=1e-3,
        fallback_learning_rate=3e-4,
        weight_decay=0.0,
        eight_bit=False,
    )
    restored_optimizer.init(restored.trainable_parameters())
    trainer_state = load_checkpoint(checkpoint, restored, restored_optimizer)
    actual_random = mx.random.uniform(shape=(4,))
    actual = dict(restored.parameters())["embedding"]["weight"]
    actual_optimizer = dict(tree_flatten(restored_optimizer.state))
    mx.eval(expected, actual, restored_optimizer.state)
    assert mx.allclose(actual, expected).item()
    assert actual_optimizer.keys() == expected_optimizer.keys()
    assert all(mx.array_equal(actual_optimizer[key], value).item() for key, value in expected_optimizer.items())
    assert trainer_state["step"] == 3
    assert mx.array_equal(actual_random, expected_random).item()

    checkpoint.with_name(f"{checkpoint.stem}.optimizer.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="Missing optimizer checkpoint"):
        load_checkpoint(checkpoint, restored, restored_optimizer)
