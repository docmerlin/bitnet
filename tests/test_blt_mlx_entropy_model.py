"""The MLX entropy model must agree with the torch one, including its patching.

The patcher is the one component both stacks use: MLX trains the student, torch
runs ``blt.generate``. If the two disagree about where patches begin, a model
trained in one and served in the other decodes against latents it was never
trained with -- and nothing else in the suite would notice.
"""

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx.utils import tree_flatten, tree_unflatten

from blt.config import TernaryBLTConfig
from blt.mlx_entropy_model import MLXByteEntropyModel
from blt.mlx_entropy_model import boundaries_from_entropy as mlx_boundaries
from blt.mlx_entropy_model import calibrate_threshold as mlx_calibrate
from blt.mlx_entropy_model import cap_patch_lengths as mlx_cap
from blt.mlx_entropy_model import next_byte_entropy as mlx_entropy
from blt.mlx_entropy_model import patch_lengths_from_starts as mlx_lengths
from blt.patching.entropy_model import (
    ByteEntropyModel,
    boundaries_from_entropy,
    calibrate_threshold,
    cap_patch_lengths,
    next_byte_entropy,
    patch_lengths_from_starts,
)


def _config() -> TernaryBLTConfig:
    return TernaryBLTConfig(local_dim=32, global_dim=32, decoder_dim=32, n_heads_cross=4)


def _pair(**kwargs):
    """Torch and MLX entropy models holding identical weights."""
    settings = dict(dim=32, num_layers=2, num_heads=4, max_seq_len=64)
    settings.update(kwargs)
    config = _config()
    torch.manual_seed(0)
    torch_model = ByteEntropyModel(config, **settings).eval()
    mlx_model = MLXByteEntropyModel(config, **settings)
    mlx_model.update(
        tree_unflatten(
            [(name, mx.array(t.detach().numpy())) for name, t in torch_model.state_dict().items()]
        )
    )
    mx.eval(mlx_model.parameters())
    return torch_model, mlx_model


def _tokens(batch=2, seq=24, seed=1):
    return np.random.default_rng(seed).integers(4, 260, size=(batch, seq)).astype(np.int64)


def test_parameter_names_match():
    torch_model, mlx_model = _pair()
    torch_names = set(torch_model.state_dict())
    mlx_names = {name for name, _ in tree_flatten(mlx_model.parameters())}
    # _threshold is a torch buffer and a frozen MLX array, so it is not a
    # trainable parameter on either side; everything else has to line up.
    assert torch_names - {"_threshold"} == mlx_names


def test_logits_match_torch():
    torch_model, mlx_model = _pair()
    tokens = _tokens()
    with torch.no_grad():
        expected = torch_model(torch.from_numpy(tokens)).numpy()
    actual = np.asarray(mlx_model(mx.array(tokens)))
    assert np.abs(actual - expected).max() < 2e-4, np.abs(actual - expected).max()


def test_entropy_matches_torch():
    torch_model, mlx_model = _pair()
    tokens = _tokens()
    expected = torch_model.entropy(torch.from_numpy(tokens)).numpy()
    actual = np.asarray(mlx_model.entropy(mx.array(tokens)))
    assert np.abs(actual - expected).max() < 2e-4


def test_loss_matches_torch():
    torch_model, mlx_model = _pair()
    tokens = _tokens()
    expected = float(torch_model.loss(torch.from_numpy(tokens)))
    actual = float(mlx_model.loss(mx.array(tokens)))
    assert actual == pytest.approx(expected, rel=1e-4)


def test_next_byte_entropy_matches_on_shared_logits():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((2, 6, 260)).astype(np.float32) * 3
    expected = next_byte_entropy(torch.from_numpy(logits)).numpy()
    actual = np.asarray(mlx_entropy(mx.array(logits)))
    assert np.abs(actual - expected).max() < 1e-5


@pytest.mark.parametrize(
    "threshold,relative",
    [(1.0, None), (None, 0.5), (0.5, 0.25)],
)
def test_boundary_rules_match(threshold, relative):
    rng = np.random.default_rng(3)
    entropy = rng.uniform(0.0, 5.0, size=(3, 20)).astype(np.float32)
    expected = boundaries_from_entropy(
        torch.from_numpy(entropy), threshold=threshold, relative_threshold=relative
    ).numpy()
    actual = np.asarray(
        mlx_boundaries(mx.array(entropy), threshold=threshold, relative_threshold=relative)
    )
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("max_patch_length", [1, 2, 3, 5, 16])
def test_cap_and_lengths_match(max_patch_length):
    rng = np.random.default_rng(4)
    starts = rng.random((3, 32)) < 0.2
    starts[:, 0] = True

    expected_capped = cap_patch_lengths(torch.from_numpy(starts), max_patch_length)
    actual_capped = mlx_cap(mx.array(starts), max_patch_length)
    assert np.array_equal(np.asarray(actual_capped), expected_capped.numpy())

    expected = patch_lengths_from_starts(expected_capped).numpy()
    actual = np.asarray(mlx_lengths(actual_capped))
    assert np.array_equal(actual, expected.astype(np.int32))
    assert np.all(actual.sum(axis=1) == 32)


def test_predicted_patch_lengths_match_torch():
    torch_model, mlx_model = _pair()
    tokens = _tokens(seq=48)
    torch_model.set_threshold(4.0)
    mlx_model.set_threshold(4.0)
    expected = torch_model.predict_patch_lengths(torch.from_numpy(tokens)).numpy()
    actual = np.asarray(mlx_model.predict_patch_lengths(mx.array(tokens)))
    assert np.array_equal(actual, expected.astype(np.int32))


def test_opens_new_patch_matches_torch():
    torch_model, mlx_model = _pair()
    tokens = _tokens(seq=32)
    torch_model.set_threshold(4.0)
    mlx_model.set_threshold(4.0)
    for cut in range(4, 30, 5):
        prefix = tokens[:, :cut]
        expected = torch_model.opens_new_patch(torch.from_numpy(prefix)).numpy()
        actual = np.asarray(mlx_model.opens_new_patch(mx.array(prefix)))
        assert np.array_equal(actual, expected)


@pytest.mark.parametrize("cap", [0, 1, 3, 8])
def test_next_patch_decision_shares_capped_segmentation(monkeypatch, cap):
    _, model = _pair()
    model.max_patch_length = cap
    entropy = mx.array([[0.0] * 17, [0.0, 9.0] + [0.0] * 15])
    monkeypatch.setattr(model, "entropy", lambda ids: entropy[:, :ids.shape[1]])
    tokens = mx.ones((2, 17), dtype=mx.int32)
    for cut in range(1, 17):
        lengths = model.predict_patch_lengths(tokens[:, :cut + 1], threshold=1.0)
        starts = mx.any(mx.cumsum(lengths, axis=1)[:, :-1] == cut, axis=1)
        assert np.array_equal(
            np.asarray(model.opens_new_patch(tokens[:, :cut], threshold=1.0)), np.asarray(starts)
        )
        if cap:
            assert int(mx.max(lengths)) <= cap


def test_calibration_agrees_across_stacks():
    torch_model, mlx_model = _pair()
    tokens = _tokens(batch=4, seq=64, seed=5)
    for target in (2.0, 4.0, 8.0):
        expected = calibrate_threshold(torch_model, torch.from_numpy(tokens), target_patch_size=target)
        actual = mlx_calibrate(mlx_model, mx.array(tokens), target_patch_size=target)
        # torch interpolates between order statistics, MLX takes the element;
        # on a 252-sample calibration set that gap is small but not zero.
        assert actual == pytest.approx(expected, abs=0.05)


def test_mlx_model_trains_down():
    config = _config()
    mx.random.seed(0)
    model = MLXByteEntropyModel(config, dim=32, num_layers=1, num_heads=4, max_seq_len=64)
    mx.eval(model.parameters())
    import mlx.optimizers as optim

    optimizer = optim.AdamW(learning_rate=3e-3)
    tokens = mx.array(_tokens(batch=4, seq=24, seed=6))
    loss_and_grad = mx.value_and_grad(lambda m: m.loss(tokens))

    first = float(model.loss(tokens))
    for _ in range(40):
        loss, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
    assert float(loss) < first


def test_threshold_is_not_a_trainable_parameter():
    # It is calibration state, not something gradient descent should move.
    mlx_model = MLXByteEntropyModel(_config(), dim=32, num_layers=1, num_heads=4)
    names = {name for name, _ in tree_flatten(mlx_model.trainable_parameters())}
    assert not any("_threshold" in name for name in names)


def test_oversized_sequence_is_refused():
    model = MLXByteEntropyModel(_config(), dim=32, num_layers=1, num_heads=4, max_seq_len=16)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(mx.zeros((1, 17), dtype=mx.int32))
