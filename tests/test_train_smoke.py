"""Smoke tests for BitNet training internals without Hugging Face data."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TernaryConfig
from model import BitNetDeep
from train import (
    TrainerState,
    TrainingWrapper,
    create_optimizer,
    language_modeling_loss,
    load_checkpoint,
    multi_token_loss,
    save_checkpoint,
)
from utils import load_checkpoint_payload


def _tiny_config() -> TernaryConfig:
    return TernaryConfig(
        vocab_size=512,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=8,
        head_dim=8,
        intermediate_size=128,
        use_hadamard=False,
        use_4bit_activations=False,
    )


def _tiny_args(optimizer: str = "cmud") -> argparse.Namespace:
    return argparse.Namespace(
        optimizer=optimizer,
        learning_rate=1e-4,
        mud_learning_rate=1e-3,
        mud_momentum=0.95,
        mud_passes=1,
        no_optimizer_8bit=False,
        weight_decay=0.0,
        lion_beta1=0.9,
        lion_beta2=0.99,
        output_dir=".",
    )


def test_single_training_step_updates_parameters() -> bool:
    config = _tiny_config()
    model = BitNetDeep(config)
    runner = TrainingWrapper(model, gradient_checkpointing=False)
    optimizer = create_optimizer(model, _tiny_args())

    before = model.layers[0].gate.item()
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    labels = input_ids.clone()

    runner.train()
    optimizer.zero_grad(set_to_none=True)
    logits = runner(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    loss.backward()
    optimizer.step()
    after = model.layers[0].gate.item()

    assert after != before, "Expected at least one parameter to change after one optimizer step"
    print("BitNet single-step training smoke tests passed")
    return True


def test_checkpoint_save_load_roundtrip() -> bool:
    from torch.optim.lr_scheduler import LambdaLR

    config = _tiny_config()
    model = BitNetDeep(config)
    args = _tiny_args()
    optimizer = create_optimizer(model, args)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    state = TrainerState(step=7, tokens_processed=128, samples_processed=4)

    with TemporaryDirectory() as tempdir:
        output_dir = Path(tempdir)
        checkpoint_path = save_checkpoint(
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            state=state,
            model_config=config,
            args=args,
            checkpoint_name="smoke.pt",
        )
        payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
        assert payload["trainer_state"]["step"] == 7
        assert payload["model_config"]["hidden_size"] == 64

        restored_model = BitNetDeep(config)
        restored_optimizer = create_optimizer(restored_model, args)
        restored_scheduler = LambdaLR(restored_optimizer, lr_lambda=lambda _: 1.0)
        restored_state = load_checkpoint(
            checkpoint_path,
            restored_model,
            restored_optimizer,
            restored_scheduler,
            None,
        )
        assert restored_state.step == 7
        assert restored_state.tokens_processed == 128

    print("BitNet checkpoint round-trip smoke tests passed")
    return True


def test_z_loss_matches_cross_entropy_when_disabled() -> bool:
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 32)
    labels = torch.randint(0, 32, (2, 5))

    plain = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    disabled = language_modeling_loss(logits, labels, z_loss_coef=0.0)
    assert torch.allclose(plain, disabled), (plain, disabled)

    # A positive coefficient adds a strictly positive log-partition penalty.
    regularized = language_modeling_loss(logits, labels, z_loss_coef=1e-3)
    assert regularized > plain, (regularized, plain)
    print("language-modeling z-loss tests passed")
    return True


def test_multi_token_loss_handles_short_sequences() -> bool:
    torch.manual_seed(0)
    seq_len, vocab = 3, 16
    labels = torch.randint(0, vocab, (2, seq_len))

    # Depth 0 (shift 1) has targets; depths whose shift >= seq_len are empty and
    # must be skipped, not fed to cross_entropy (which returns NaN on 0 rows).
    depth_logits = [torch.randn(2, seq_len, vocab) for _ in range(seq_len + 2)]
    loss = multi_token_loss(depth_logits, labels)
    assert torch.isfinite(loss), loss

    # All depths empty (seq_len 1) -> zero, still finite.
    tiny = multi_token_loss([torch.randn(2, 1, vocab)], torch.randint(0, vocab, (2, 1)))
    assert torch.isfinite(tiny) and float(tiny) == 0.0, tiny
    print("multi-token-loss short-sequence tests passed")
    return True


if __name__ == "__main__":
    test_single_training_step_updates_parameters()
    test_checkpoint_save_load_roundtrip()
    test_z_loss_matches_cross_entropy_when_disabled()
    test_multi_token_loss_handles_short_sequences()