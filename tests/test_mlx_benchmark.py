"""Minimal checks for the representative MLX benchmark."""

from argparse import Namespace
from dataclasses import replace

import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from mlx_benchmark import run_mlx, validate_args
from mlx_model import MLXBitNet, MLXBitNetConfig, MLXHBitLinear, MLXPaTHAttention, MLXRFMoE
from mlx_optim import CMUD
from mlx_path_kernel import path_triangular_solve, reference_triangular_solve
from mlx_rfmoe_kernel import masked_grouped_linear
from mlx_ternary_kernel import pack_ternary_weight, ternary_quantized_linear
from mlx_train import (
    _gradient_compile_safe,
    build_validation_batches,
    build_parser,
    create_apply_step,
    create_gradient_step,
    create_train_step,
    evaluate,
    load_checkpoint,
    mtp_head_index,
    prepare_mtp_batch,
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
        optimizer="adamw",
        steps=1,
        warmup_steps=0,
        batch_size=1,
        sequence_length=4,
        vocab_size=32,
        hidden_size=8,
        num_heads=2,
        intermediate_size=16,
        num_layers=1,
        num_prelude_layers=0,
        num_coda_layers=0,
        num_loops=1,
        active_loops=None,
        path_window_size=4,
        learning_rate=1e-3,
        mlx_dtype="float32",
        mlx_path_kernel=True,
        reuse_recurrent_weights=False,
        recurrent_quantized_matmul=False,
        cmud_momentum_8bit=False,
    )
    validate_args(args)
    metrics = run_mlx(args)
    assert metrics["loss"] > 0
    assert metrics["tokens_per_second"] > 0
    args.num_loops = 0
    with pytest.raises(ValueError, match="num-loops must be positive"):
        validate_args(args)


def test_mlx_training_defaults_use_fast_local_batch() -> None:
    args = build_parser().parse_args([])
    assert args.micro_batch_size == 4
    assert args.grad_accumulation_steps == 4
    assert not args.gradient_checkpointing
    assert args.validation_batches == 5
    assert args.mud_block_size == 256
    assert args.mtp_depth == 4
    assert args.recurrent_quantized_matmul
    assert args.cmud_momentum_8bit
    assert args.loop_curriculum_start_ratio == 0.0
    assert args.loop_curriculum_ratio == 0.2
    args.loop_curriculum_start_ratio = 0.3
    with pytest.raises(ValueError, match="0 <= start <= end <= 1"):
        validate_training_args(args)


def test_mlx_validation_batches_are_materialized_once(monkeypatch) -> None:
    calls = 0

    def fake_stream(*args, **kwargs):
        nonlocal calls
        calls += 1
        return iter((0, 1))

    monkeypatch.setattr("mlx_train.build_batch_stream", fake_stream)
    monkeypatch.setattr("mlx_train.convert_batch", lambda batch: batch)
    args = build_parser().parse_args(["--validation-batches", "2"])

    assert build_validation_batches(None, args) == [0, 1]
    assert calls == 1


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


def test_mlx_recurrent_loops_reuse_effective_weights(monkeypatch) -> None:
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=0,
        num_recurrent_layers=1,
        num_coda_layers=0,
        num_loops=3,
        path_window_size=4,
        use_engram=False,
    )
    reference = MLXBitNet(config, reuse_recurrent_weights=False)
    model = MLXBitNet(config, reuse_recurrent_weights=True)
    model.load_weights(list(tree_flatten(reference.parameters())))
    tokens = mx.array([[1, 2, 3, 4]])
    segments = mx.zeros((1, 4), dtype=mx.int32)
    reference_loss, reference_gradients = nn.value_and_grad(
        reference,
        lambda ids, segment_ids: mx.mean(mx.square(reference(ids, segment_ids))),
    )(tokens, segments)
    cached_loss, cached_gradients = nn.value_and_grad(
        model,
        lambda ids, segment_ids: mx.mean(mx.square(model(ids, segment_ids))),
    )(tokens, segments)
    mx.eval(reference_loss, reference_gradients, cached_loss, cached_gradients)
    assert mx.array_equal(cached_loss, reference_loss).item()
    for (_, expected), (_, actual) in zip(tree_flatten(reference_gradients), tree_flatten(cached_gradients)):
        assert mx.array_equal(actual, expected).item()

    original = MLXHBitLinear.effective_weight
    seen = {}
    reuses = 0

    def counted(self, dtype, weight=None, cache_key=None):
        nonlocal reuses
        result = original(self, dtype, weight, cache_key)
        key = (id(self) if weight is None else cache_key, str(dtype))
        if key in seen and result is seen[key]:
            reuses += 1
        seen[key] = result
        return result

    monkeypatch.setattr(MLXHBitLinear, "effective_weight", counted)
    output = model(tokens, segments)
    mx.eval(output)

    assert len(seen) == 7
    assert reuses == 14


def test_mlx_packed_ternary_linear_matches_forward_and_vjp() -> None:
    x = mx.random.normal((2, 3, 64)).astype(mx.float32)
    weight = mx.random.normal((32, 64)).astype(mx.float32)
    packed, scales, _ = pack_ternary_weight(weight)
    scale = mx.maximum(mx.mean(mx.abs(weight), axis=-1, keepdims=True), 1e-5)
    normalized = weight / scale
    effective = mx.where(normalized > 0.5, scale, mx.where(normalized < -0.5, -scale, 0.0))
    expected = x @ effective.T
    actual = ternary_quantized_linear(x, weight, packed, scales)
    cotangent = mx.random.normal(actual.shape)
    gradients = mx.vjp(
        lambda values, weights: ternary_quantized_linear(
            values,
            weights,
            *pack_ternary_weight(weights)[:2],
        ),
        [x, weight],
        [cotangent],
    )[1]
    expected_x = cotangent @ effective
    expected_weight = cotangent.reshape(-1, weight.shape[0]).T @ x.reshape(-1, weight.shape[1])
    mx.eval(expected, actual, gradients, expected_x, expected_weight)

    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-5).item()
    assert mx.allclose(gradients[0], expected_x, rtol=1e-5, atol=1e-5).item()
    assert mx.array_equal(gradients[1], expected_weight).item()


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

    hidden = model.hidden_states(tokens, segments)
    drafts = model.draft_logits(hidden)
    mx.eval(drafts)
    assert drafts.shape == (1, 2, 32)
    assert mx.allclose(drafts[:, 0], mtp[0][:, -1], rtol=1e-5, atol=1e-5).item()
    assert mx.allclose(drafts[:, 1], mtp[1][:, -1], rtol=1e-5, atol=1e-5).item()

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


def test_mlx_sampled_mtp_matches_exact_depth_mean() -> None:
    config = MLXBitNetConfig(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=0,
        num_recurrent_layers=0,
        num_coda_layers=0,
        use_engram=False,
        mtp_depth=2,
    )
    model = MLXBitNet(config)
    gradient_step = create_gradient_step(model, compile_step=False, num_loops=1, mtp_loss_coef=0.3)
    inputs = mx.array([[1, 2, 3, 4]])
    targets = mx.array([[2, 3, 4, 5]])
    segments = mx.zeros_like(inputs)
    args = (inputs, targets, segments, segments, mx.array(0.0), mx.array(1.0), mx.array(0.1))

    exact_loss, exact_gradients = gradient_step(*args)
    sampled = []
    sampled_gradients = []
    for index in range(config.mtp_depth):
        mtp_batch = prepare_mtp_batch(targets, segments, segments, index, config.mtp_depth)
        loss, gradients = gradient_step(*args, *mtp_batch)
        sampled.append(loss)
        sampled_gradients.append(gradients)
    mean_loss = mx.mean(mx.stack(sampled))
    mean_gradients = tree_map(lambda *values: mx.mean(mx.stack(values), axis=0), *sampled_gradients)
    mx.eval(exact_loss, exact_gradients, mean_loss, mean_gradients)

    assert mx.allclose(mean_loss, exact_loss, rtol=1e-5, atol=1e-5).item()
    for (_, exact), (_, sampled_mean) in zip(tree_flatten(exact_gradients), tree_flatten(mean_gradients)):
        assert mx.allclose(sampled_mean, exact, rtol=1e-4, atol=1e-5).item()


def test_mlx_mtp_head_schedule_is_resume_stable() -> None:
    assert [mtp_head_index(1, index, 4, 3) for index in range(4)] == [0, 1, 2, 0]
    assert [mtp_head_index(2, index, 4, 3) for index in range(4)] == [1, 2, 0, 1]
    assert [mtp_head_index(8, index, 4, 4) for index in range(4)] == [0, 1, 2, 3]


def test_mlx_evaluation_reports_mtp_quality() -> None:
    config = MLXBitNetConfig(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=0,
        num_recurrent_layers=0,
        num_coda_layers=0,
        use_engram=False,
        mtp_depth=1,
    )
    model = MLXBitNet(config)
    inputs = mx.array([[1, 2, 3, 4]])
    targets = mx.array([[2, 3, 4, 5]])
    segments = mx.zeros_like(inputs)

    metrics = evaluate(model, [(inputs, targets, segments, segments)])

    assert metrics["val_perplexity"] > 0
    assert metrics["mtp_loss_depth_2"] > 0
    assert 0 <= metrics["mtp_accuracy_depth_2"] <= 1
    assert 0 <= metrics["mtp_agreement_depth_2"] <= 1


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
