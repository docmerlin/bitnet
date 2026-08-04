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
    accumulate_gradients,
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
        optimizer="cmud",
        steps=1,
        warmup_steps=0,
        batch_size=1,
        sequence_length=4,
        grad_accumulation_steps=1,
        vocab_size=32,
        hidden_size=8,
        num_heads=2,
        intermediate_size=16,
        num_layers=1,
        num_prelude_layers=0,
        num_coda_layers=0,
        num_loops=1,
        active_loops=None,
        mtp_depth=0,
        path_window_size=4,
        learning_rate=1e-3,
        mud_block_size=4,
        mlx_dtype="float32",
        mlx_path_kernel=True,
        reuse_recurrent_weights=False,
        recurrent_quantized_matmul=False,
        cmud_momentum_8bit=False,
        cmud_master_dtype="float32",
        gradient_checkpoint_scope="none",
        profile_phases=True,
    )
    validate_args(args)
    metrics = run_mlx(args)
    assert metrics["loss"] > 0
    assert metrics["peak_memory_gib"] > 0
    assert metrics["profile_forward_backward_seconds"] > 0
    assert metrics["profile_mud_seconds"] > 0
    assert metrics["profile_sync_wait_seconds"] > 0
    assert metrics["profile_validation_seconds"] > 0
    assert metrics["tokens_per_second"] > 0
    args.profile_phases = False
    args.grad_accumulation_steps = 2
    args.mtp_depth = 1
    accumulated_metrics = run_mlx(args)
    assert accumulated_metrics["loss"] > 0
    assert accumulated_metrics["tokens_per_second"] > 0
    args.mtp_depth = 0
    args.backend = "torch"
    args.optimizer = "adamw"
    with pytest.raises(ValueError, match="Gradient accumulation benchmark is only available with MLX"):
        validate_args(args)
    args.backend = "mlx"
    args.optimizer = "cmud"
    args.grad_accumulation_steps = 1
    args.backend = "torch"
    args.mtp_depth = 1
    with pytest.raises(ValueError, match="MTP benchmark is only available with MLX"):
        validate_args(args)
    args.backend = "mlx"
    args.mtp_depth = 0
    args.grad_accumulation_steps = 0
    with pytest.raises(ValueError, match="grad-accumulation-steps must be positive"):
        validate_args(args)
    args.grad_accumulation_steps = 1
    args.num_loops = 0
    with pytest.raises(ValueError, match="num-loops must be positive"):
        validate_args(args)


def test_mlx_training_defaults_use_fast_local_batch() -> None:
    args = build_parser().parse_args([])
    assert args.micro_batch_size == 4
    assert args.grad_accumulation_steps == 4
    assert not args.gradient_checkpointing
    assert args.gradient_checkpoint_scope == "recurrent"
    assert not args.profile_phases
    assert build_parser().parse_args(["--profile-phases"]).profile_phases
    assert args.validation_batches == 5
    assert args.mud_block_size == 64
    assert args.mtp_depth == 4
    assert args.recurrent_quantized_matmul
    assert args.cmud_momentum_8bit
    assert args.cmud_master_dtype == "bfloat16"
    assert args.loop_curriculum_start_ratio == 0.0
    assert args.loop_curriculum_ratio == 0.2
    args.loop_curriculum_start_ratio = 0.3
    with pytest.raises(ValueError, match="0 <= start <= end <= 1"):
        validate_training_args(args)


def test_mlx_materialized_gradient_accumulation_matches_deferred_mean() -> None:
    gradients = [
        {"weight": mx.array([[1.0, 2.0], [3.0, 4.0]])},
        {"weight": mx.array([[5.0, 6.0], [7.0, 8.0]])},
        {"weight": mx.array([[9.0, 10.0], [11.0, 12.0]])},
    ]
    deferred = tree_map(lambda *values: mx.mean(mx.stack(values), axis=0), *gradients)
    accumulated = None
    for current in gradients:
        accumulated = accumulate_gradients(accumulated, current)
        mx.eval(accumulated)
    materialized = tree_map(lambda value: value / len(gradients), accumulated)
    mx.eval(deferred, materialized)

    for (_, expected), (_, actual) in zip(tree_flatten(deferred), tree_flatten(materialized)):
        assert mx.array_equal(actual, expected).item()


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


@pytest.mark.parametrize(
    ("shape", "different_settings", "expected_calls"),
    [
        ((2, 4, 64), False, {"qkv": 2, "path": 0}),
        ((2, 4, 64), True, {"qkv": 2, "path": 2}),
        ((1, 1, 64), False, {"qkv": 0, "path": 0}),
    ],
)
def test_mlx_qkv_and_path_share_input_preparation(
    monkeypatch,
    shape,
    different_settings,
    expected_calls,
) -> None:
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=128,
        num_prelude_layers=1,
        num_recurrent_layers=0,
        num_coda_layers=0,
        num_loops=1,
        path_window_size=4,
        use_engram=False,
    )
    attention = MLXPaTHAttention(config)
    for layer in (attention.qkv, attention.path_down, attention.path_up):
        layer.set_quantization_state(0.75, 0.5, 8)
    if different_settings:
        attention.path_down.set_quantization_state(0.75, 0.25, 8)
    batch, length, _ = shape
    if length == 1:
        from mlx_model import _recurrent_quantized_matmul

        token = _recurrent_quantized_matmul.set(True)
        try:
            for layer in (attention.qkv, attention.path_down):
                layer.set_quantization_state(1.0, 1.0, 4)
                layer.pin_inference_weight(mx.float32, prefer_packed=True)
        finally:
            _recurrent_quantized_matmul.reset(token)
    x = mx.random.normal(shape)
    segments = mx.zeros((batch, length), dtype=mx.int32)

    def independent(values):
        qkv = attention.qkv(values).reshape(batch, length, 3, config.num_attention_heads, 16)
        q = attention.q_norm(qkv[:, :, 0].transpose(0, 2, 1, 3))
        k = attention.k_norm(qkv[:, :, 1].transpose(0, 2, 1, 3))
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)
        w, projected = attention._path_vectors(values, segments)
        beta = 2.0 * mx.sigmoid(attention.path_beta(values).astype(mx.float32))
        forget_logits = attention.path_forget(values).astype(mx.float32)
        log_forget = -mx.logaddexp(mx.zeros_like(forget_logits), -forget_logits)
        return q, k, v, w, beta, log_forget, projected

    expected = independent(x)
    expected_grad = (
        mx.grad(lambda values: sum(mx.sum(item) for item in independent(values)))(x)
        if length > 1
        else None
    )

    calls = {"qkv": 0, "path": 0}
    original = MLXHBitLinear.prepare_input

    def counted(layer, values):
        if layer is attention.qkv:
            calls["qkv"] += 1
        elif layer is attention.path_down:
            calls["path"] += 1
        return original(layer, values)

    monkeypatch.setattr(MLXHBitLinear, "prepare_input", counted)
    actual = attention._project(x, segments)
    actual_grad = (
        mx.grad(lambda values: sum(mx.sum(item) for item in attention._project(values, segments)))(x)
        if length > 1
        else None
    )
    mx.eval(expected, actual, *(() if expected_grad is None else (expected_grad, actual_grad)))

    assert calls == expected_calls
    for expected_item, actual_item in zip(expected, actual):
        assert mx.array_equal(actual_item, expected_item).item()
    if expected_grad is not None:
        assert mx.allclose(actual_grad, expected_grad, rtol=1e-6, atol=1e-6).item()


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


@pytest.mark.parametrize("use_delta", [True, False])
def test_mlx_paper_infini_memory_update_matches_formula(use_delta: bool) -> None:
    """Associative M,z update matches Munkhdalai et al. Linear / Linear+Delta."""
    from mlx_model import _MEMORY_EPS, _infini_sigma

    config = MLXBitNetConfig(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=1,
        num_recurrent_layers=0,
        num_coda_layers=0,
        path_window_size=4,
        infini_delta_rule=use_delta,
        use_engram=False,
    )
    attention = MLXPaTHAttention(config)
    length, d, h = 3, 4, 2
    keys = mx.arange(h * length * d, dtype=mx.float32).reshape(1, h, length, d)
    values = keys + 1
    memory_m = mx.zeros((1, h, d, d), dtype=mx.float32)
    memory_z = mx.zeros((1, h, d), dtype=mx.float32)
    sk = _infini_sigma(keys)
    if use_delta:
        den = mx.maximum(mx.sum(sk * memory_z[:, :, None, :], axis=-1, keepdims=True), _MEMORY_EPS)
        retrieved = (sk @ memory_m) / den
        expected_m = memory_m + sk.swapaxes(-1, -2) @ (values.astype(mx.float32) - retrieved)
    else:
        expected_m = memory_m + sk.swapaxes(-1, -2) @ values.astype(mx.float32)
    expected_z = memory_z + mx.sum(sk, axis=2)

    actual_m, actual_z, initialized = attention._next_memory(
        keys,
        values,
        mx.array([True]),
        memory_m,
        memory_z,
        mx.array([False]),
    )
    mx.eval(actual_m, actual_z, initialized)
    assert mx.allclose(actual_m, expected_m, rtol=1e-5, atol=1e-6).item()
    assert mx.allclose(actual_z, expected_z, rtol=1e-5, atol=1e-6).item()
    assert initialized.tolist() == [True]


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
    attention(
        mx.random.normal((1, 4, 8)),
        segment_ids=mx.zeros((1, 4), dtype=mx.int32),
        update_memory=True,
        persist_memory=True,
    )
    mx.eval(attention.memory_initialized, attention.memory_m)
    assert attention.memory_initialized.item()
    # mx.count_nonzero does not exist in this MLX version; the point is only
    # that the memory bank actually got written.
    assert int(mx.sum((attention.memory_m != 0).astype(mx.int32))) > 0

    attention.reset_memory(2)
    mixed_segments = mx.array([[0, 0, 0, 0], [0, 0, 1, 1]])
    attention(
        mx.random.normal((2, 4, 8)),
        segment_ids=mixed_segments,
        update_memory=True,
        persist_memory=True,
    )
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


@pytest.mark.parametrize("index", [0, 1])
def test_mlx_selected_mtp_matches_direct_head_gradients(index: int) -> None:
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
    hidden = mx.random.normal((1, 4, config.hidden_size))

    selected_step = nn.value_and_grad(
        model,
        lambda values: mx.mean(
            model.selected_mtp_logits(values, mx.array(index, dtype=mx.int32)).astype(mx.float32) ** 2
        ),
    )
    direct_step = nn.value_and_grad(
        model,
        lambda values: mx.mean(
            model.logits_from(model.mtp_transforms[index](values)).astype(mx.float32) ** 2
        ),
    )
    selected_loss, selected_gradients = selected_step(hidden)
    direct_loss, direct_gradients = direct_step(hidden)
    mx.eval(selected_loss, selected_gradients, direct_loss, direct_gradients)

    assert mx.allclose(selected_loss, direct_loss, rtol=1e-5, atol=1e-6).item()
    for (_, expected), (_, actual) in zip(
        tree_flatten(direct_gradients),
        tree_flatten(selected_gradients),
    ):
        assert mx.allclose(actual, expected, rtol=1e-4, atol=1e-5).item()

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


@pytest.mark.parametrize(
    ("num_prelude_layers", "num_recurrent_layers", "checkpoint_scope"),
    ((1, 0, True), (0, 1, "recurrent")),
)
def test_mlx_activation_checkpointing_preserves_gradients_and_state(
    num_prelude_layers,
    num_recurrent_layers,
    checkpoint_scope,
) -> None:
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=num_prelude_layers,
        num_recurrent_layers=num_recurrent_layers,
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
            mx.square(checkpointed(ids, segment_ids, checkpoint_activations=checkpoint_scope))
        ),
    )(tokens, segments)
    mx.eval(reference_loss, reference_gradients, checkpointed_loss, checkpointed_gradients)

    assert mx.allclose(checkpointed_loss, reference_loss, rtol=1e-5, atol=1e-5).item()
    expected = dict(tree_flatten(reference_gradients))
    actual = dict(tree_flatten(checkpointed_gradients))
    assert actual.keys() == expected.keys()
    assert all(mx.allclose(actual[key], value, rtol=1e-4, atol=1e-5).item() for key, value in expected.items())
    assert mx.allclose(
        checkpointed.blocks[0].attn.memory_m,
        reference.blocks[0].attn.memory_m,
        rtol=1e-5,
        atol=1e-6,
    ).item()
    assert mx.allclose(
        checkpointed.blocks[0].moe.usage_ema,
        reference.blocks[0].moe.usage_ema,
        rtol=1e-5,
        atol=1e-6,
    ).item()


def test_mlx_activation_checkpointing_selects_recurrent_blocks(monkeypatch) -> None:
    config = MLXBitNetConfig(
        vocab_size=16,
        hidden_size=8,
        num_attention_heads=2,
        intermediate_size=16,
        num_prelude_layers=1,
        num_recurrent_layers=1,
        num_coda_layers=1,
        num_loops=2,
        block_size=1,
        path_window_size=4,
        use_engram=False,
    )
    model = MLXBitNet(config)
    checkpointed_modules = []
    original = nn.utils.checkpoint

    def counted(module, *args, **kwargs):
        checkpointed_modules.append(module)
        return original(module, *args, **kwargs)

    monkeypatch.setattr("mlx_model.activation_checkpoint", counted)
    tokens = mx.array([[1, 2, 3, 4]])
    output = model(tokens, checkpoint_activations="recurrent")
    mx.eval(output)

    assert len(checkpointed_modules) == 4
    assert all(
        module in (model.blocks[1], model.blocks[1].attn)
        for module in checkpointed_modules
    )


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
    assert not any(
        key.endswith((".memory_m", ".memory_z", ".memory_initialized", ".memory_k", ".memory_v"))
        for key in mx.load(str(checkpoint))
    )
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


def test_ternary_fused_linear_m1_matches_dense_effective() -> None:
    from mlx_ternary_kernel import (
        pack_ternary_weight,
        ternary_effective_weight,
        ternary_fused_linear_m1,
    )

    mx.random.seed(0)
    weight = mx.random.normal((96, 128)).astype(mx.float32)
    x = mx.random.normal((1, 1, 128)).astype(mx.float32)
    packed, scales, group_size = pack_ternary_weight(weight)
    expected = x @ ternary_effective_weight(weight).T
    actual = ternary_fused_linear_m1(
        x,
        packed,
        scales,
        in_dim=128,
        out_dim=96,
        group_size=group_size,
        quantize_acts=False,
    )
    # with act quant
    levels = 7.0
    amax = mx.maximum(mx.max(mx.abs(x), axis=-1, keepdims=True), 1e-5)
    scale = amax / levels
    xq = mx.clip(mx.round(x / scale), -(levels + 1), levels) * scale
    expected_q = xq @ ternary_effective_weight(weight).T
    actual_q = ternary_fused_linear_m1(
        x,
        packed,
        scales,
        in_dim=128,
        out_dim=96,
        group_size=group_size,
        quantize_acts=True,
        act_levels=levels,
    )
    mx.eval(expected, actual, expected_q, actual_q)
    assert mx.allclose(actual, expected, rtol=1e-4, atol=1e-4).item()
    assert mx.allclose(actual_q, expected_q, rtol=1e-4, atol=1e-4).item()


def test_ternary_fused_ffn_m1_matches_dense_reference() -> None:
    from mlx_ternary_kernel import pack_ternary_weight, ternary_effective_weight, ternary_fused_ffn_m1

    mx.random.seed(1)
    hidden, inter = 64, 128
    x = mx.random.normal((1, 1, hidden)).astype(mx.float32)
    w_up = mx.random.normal((inter * 2, hidden)).astype(mx.float32)
    w_mid = mx.random.normal((inter, inter)).astype(mx.float32)
    w_down = mx.random.normal((hidden, inter)).astype(mx.float32)

    def quant(t, levels=7.0):
        amax = mx.maximum(mx.max(mx.abs(t), axis=-1, keepdims=True), 1e-5)
        scale = amax / levels
        return mx.clip(mx.round(t / scale), -(levels + 1), levels) * scale

    u = quant(x) @ ternary_effective_weight(w_up).T
    gate, value = mx.split(u, 2, axis=-1)
    h = nn.silu(gate) * value
    h = quant(h) @ ternary_effective_weight(w_mid).T
    h = nn.silu(h)
    expected = quant(h) @ ternary_effective_weight(w_down).T
    up_p, mid_p, down_p = pack_ternary_weight(w_up), pack_ternary_weight(w_mid), pack_ternary_weight(w_down)
    actual = ternary_fused_ffn_m1(
        x,
        up_p[0],
        up_p[1],
        mid_p[0],
        mid_p[1],
        down_p[0],
        down_p[1],
        hidden=hidden,
        intermediate=inter,
        quantize_acts=True,
        act_levels=7.0,
    )
    mx.eval(expected, actual)
    assert mx.allclose(actual, expected, rtol=1e-3, atol=1e-3).item()


def test_hbitlinear_m1_uses_fused_ternary_path() -> None:
    """M=1 decode path should match dense ternary effective weights."""
    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=128,
        num_prelude_layers=1,
        num_recurrent_layers=0,
        num_coda_layers=0,
        use_engram=False,
        use_hadamard=False,
        use_4bit_activations=True,
    )
    layer = MLXHBitLinear(64, 128, config)
    layer.set_quantization_state(1.0, 1.0, 4)
    from mlx_model import _recurrent_quantized_matmul

    token = _recurrent_quantized_matmul.set(True)
    try:
        layer.pin_inference_weight(mx.float32, prefer_packed=True)
    finally:
        _recurrent_quantized_matmul.reset(token)
    assert layer._pinned_packed is not None
    x = mx.random.normal((1, 1, 64)).astype(mx.float32)
    # dense reference with same prepare semantics (full act quant)
    xq = layer.prepare_input(x)
    from mlx_ternary_kernel import ternary_effective_weight

    expected = xq @ ternary_effective_weight(layer.weight).astype(mx.float32).T
    actual = layer(x)
    mx.eval(expected, actual)
    assert mx.allclose(actual, expected, rtol=1e-3, atol=1e-3).item()


def test_fused_ffn_m1_is_skipped_when_hadamard_is_enabled(monkeypatch) -> None:
    import mlx_model

    config = MLXBitNetConfig(
        vocab_size=32,
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=128,
        num_prelude_layers=1,
        num_recurrent_layers=0,
        num_coda_layers=0,
        use_engram=False,
        use_hadamard=True,
        use_4bit_activations=True,
    )
    block = mlx_model.MLXHybridBlock(config, 0)
    token = mlx_model._recurrent_quantized_matmul.set(True)
    try:
        for layer in (block.up, block.mid, block.down):
            layer.set_quantization_state(1.0, 1.0, 4)
            layer.pin_inference_weight(mx.float32, prefer_packed=True)
    finally:
        mlx_model._recurrent_quantized_matmul.reset(token)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("fused FFN omits required Hadamard transforms")

    monkeypatch.setattr(mlx_model, "ternary_fused_ffn_m1", fail_if_called)
    x = mx.random.normal((1, 1, 64)).astype(mx.float32)
    output = block._dense_mlp(x)

    from mlx_ternary_kernel import ternary_effective_weight

    def project(value, layer):
        transformed = mx.hadamard_transform(value)
        levels = 7.0
        scale = mx.maximum(mx.max(mx.abs(transformed), axis=-1, keepdims=True), 1e-5) / levels
        quantized = mx.clip(mx.round(transformed / scale), -(levels + 1), levels) * scale
        return quantized @ ternary_effective_weight(layer.weight).T

    gate, value = mx.split(project(x, block.up), 2, axis=-1)
    expected = project(nn.silu(project(nn.silu(gate) * value, block.mid)), block.down)
    mx.eval(output, expected)
    assert mx.allclose(output, expected, rtol=1e-3, atol=1e-3).item()
