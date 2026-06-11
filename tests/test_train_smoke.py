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
    load_checkpoint,
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


def _tiny_args() -> argparse.Namespace:
    return argparse.Namespace(
        learning_rate=1e-4,
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


if __name__ == "__main__":
    test_single_training_step_updates_parameters()
    test_checkpoint_save_load_roundtrip()