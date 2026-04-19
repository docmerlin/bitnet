"""Smoke tests for BLT distillation training."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blt.config import TernaryBLTConfig
from blt.model import TernaryBLTModel
from blt.train_distill import BLTDistillationBatch, BLTDistillationTrainer


class ToyTeacher:
    def __init__(self, model: TernaryBLTModel) -> None:
        self.model = model.eval()

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, *, attention_mask: torch.Tensor | None = None, patch_lengths: torch.Tensor | None = None):
        return self.model(input_ids, attention_mask=attention_mask, patch_lengths=patch_lengths)


def build_config() -> TernaryBLTConfig:
    return TernaryBLTConfig(
        local_dim=32,
        global_dim=64,
        decoder_dim=32,
        n_layers_local_encoder=2,
        n_layers_global=2,
        n_layers_local_decoder=2,
        n_heads_local_encoder=4,
        n_heads_global=4,
        n_heads_local_decoder=4,
        n_heads_cross=4,
        patch_size=3,
        max_patch_length=8,
        use_hadamard=False,
        use_4bit_activations=False,
    )


def build_teacher_config_with_wider_decoder() -> TernaryBLTConfig:
    return TernaryBLTConfig(
        local_dim=32,
        global_dim=64,
        decoder_dim=48,
        n_layers_local_encoder=2,
        n_layers_global=2,
        n_layers_local_decoder=2,
        n_heads_local_encoder=4,
        n_heads_global=4,
        n_heads_local_decoder=4,
        n_heads_cross=4,
        patch_size=3,
        max_patch_length=8,
        use_hadamard=False,
        use_4bit_activations=False,
    )


def test_blt_distillation_train_step() -> bool:
    torch.manual_seed(1)
    config = build_config()
    student = TernaryBLTModel(config)
    teacher = ToyTeacher(TernaryBLTModel(config))
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    trainer = BLTDistillationTrainer(student, optimizer=optimizer, config=config, teacher=teacher)

    batch = BLTDistillationBatch(
        input_ids=torch.tensor(
            [
                [config.bos_id, config.byte_to_token_id(ord("a")), config.byte_to_token_id(ord("b")), config.byte_to_token_id(ord("c")), config.eos_id],
                [config.bos_id, config.byte_to_token_id(ord("x")), config.byte_to_token_id(ord("y")), config.byte_to_token_id(ord("z")), config.eos_id],
            ],
            dtype=torch.long,
        ),
        labels=torch.tensor(
            [
                [config.byte_to_token_id(ord("a")), config.byte_to_token_id(ord("b")), config.byte_to_token_id(ord("c")), config.eos_id, config.eos_id],
                [config.byte_to_token_id(ord("x")), config.byte_to_token_id(ord("y")), config.byte_to_token_id(ord("z")), config.eos_id, config.eos_id],
            ],
            dtype=torch.long,
        ),
        attention_mask=torch.ones(2, 5, dtype=torch.long),
    )

    _, _, metrics = trainer.train_step(batch)
    assert torch.isfinite(torch.tensor(metrics["loss"])), metrics
    total_grad = 0.0
    for parameter in student.parameters():
        if parameter.grad is not None:
            total_grad += float(parameter.grad.abs().sum().item())
    assert total_grad > 0.0, "Expected non-zero gradients after a distillation train step"
    print("BLT distillation smoke tests passed")
    return True


def test_blt_distillation_skips_mismatched_decoder_hidden_loss() -> bool:
    torch.manual_seed(2)
    student_config = build_config()
    teacher_config = build_teacher_config_with_wider_decoder()
    student = TernaryBLTModel(student_config)
    teacher = ToyTeacher(TernaryBLTModel(teacher_config))
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    trainer = BLTDistillationTrainer(student, optimizer=optimizer, config=student_config, teacher=teacher)

    batch = BLTDistillationBatch(
        input_ids=torch.tensor(
            [
                [student_config.bos_id, student_config.byte_to_token_id(ord("a")), student_config.byte_to_token_id(ord("b")), student_config.byte_to_token_id(ord("c")), student_config.eos_id],
                [student_config.bos_id, student_config.byte_to_token_id(ord("x")), student_config.byte_to_token_id(ord("y")), student_config.byte_to_token_id(ord("z")), student_config.eos_id],
            ],
            dtype=torch.long,
        ),
        labels=torch.tensor(
            [
                [student_config.byte_to_token_id(ord("a")), student_config.byte_to_token_id(ord("b")), student_config.byte_to_token_id(ord("c")), student_config.eos_id, student_config.eos_id],
                [student_config.byte_to_token_id(ord("x")), student_config.byte_to_token_id(ord("y")), student_config.byte_to_token_id(ord("z")), student_config.eos_id, student_config.eos_id],
            ],
            dtype=torch.long,
        ),
        attention_mask=torch.ones(2, 5, dtype=torch.long),
    )

    _, _, metrics = trainer.train_step(batch)
    assert torch.isfinite(torch.tensor(metrics["loss"])), metrics
    assert metrics.get("decoder_hidden_mse_skipped") == 1.0, metrics
    assert "decoder_hidden_mse" not in metrics, metrics
    print("BLT mismatched decoder distillation tests passed")
    return True


if __name__ == "__main__":
    test_blt_distillation_train_step()
    test_blt_distillation_skips_mismatched_decoder_hidden_loss()
