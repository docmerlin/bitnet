"""Minimal checks for the representative MLX benchmark."""

from argparse import Namespace

import pytest

mx = pytest.importorskip("mlx.core")
from mlx.utils import tree_flatten

from mlx_benchmark import run_mlx, validate_args
from mlx_model import MLXBitNet, MLXBitNetConfig, MLXPaTHAttention
from mlx_optim import CMUD
from mlx_path_kernel import path_triangular_solve, reference_triangular_solve
from mlx_train import create_train_step, load_checkpoint, save_checkpoint


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

    restored = MLXBitNet(config)
    restored_optimizer = CMUD(
        mud_learning_rate=1e-3,
        fallback_learning_rate=3e-4,
        weight_decay=0.0,
        eight_bit=False,
    )
    restored_optimizer.init(restored.trainable_parameters())
    trainer_state = load_checkpoint(checkpoint, restored, restored_optimizer)
    actual = dict(restored.parameters())["embedding"]["weight"]
    actual_optimizer = dict(tree_flatten(restored_optimizer.state))
    mx.eval(expected, actual, restored_optimizer.state)
    assert mx.allclose(actual, expected).item()
    assert actual_optimizer.keys() == expected_optimizer.keys()
    assert all(mx.array_equal(actual_optimizer[key], value).item() for key, value in expected_optimizer.items())
    assert trainer_state["step"] == 3

    checkpoint.with_name(f"{checkpoint.stem}.optimizer.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="Missing optimizer checkpoint"):
        load_checkpoint(checkpoint, restored, restored_optimizer)
