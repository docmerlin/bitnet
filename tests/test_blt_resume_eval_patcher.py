"""Regression tests for BLT resume/eval/patcher workflow."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn

from blt.config import TernaryBLTConfig
from blt.model import TernaryBLTModel
from blt.train_distill import BLTDistillationBatch, BLTDistillationTrainer, parse_args, run_distillation


class ToyTeacher:
    def __init__(self, model: TernaryBLTModel) -> None:
        self.model = model.eval()

    @torch.no_grad()
    def forward(self, input_ids, *, attention_mask=None, patch_lengths=None):
        return self.model(input_ids, attention_mask=attention_mask, patch_lengths=patch_lengths)


class RecordingTeacher(ToyTeacher):
    def __init__(self, model: TernaryBLTModel) -> None:
        super().__init__(model)
        self.calls: list[torch.Tensor | None] = []

    @torch.no_grad()
    def forward(self, input_ids, *, attention_mask=None, patch_lengths=None):
        self.calls.append(None if patch_lengths is None else patch_lengths.detach().clone())
        return super().forward(input_ids, attention_mask=attention_mask, patch_lengths=patch_lengths)


class FixedStudentPatcher(nn.Module):
    def __init__(self, predicted_patch_lengths: torch.Tensor) -> None:
        super().__init__()
        self.predicted_patch_lengths = predicted_patch_lengths
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.new_zeros(input_ids.shape, dtype=torch.float32) + self.scale

    def predict_patch_lengths_from_logits(self, boundary_logits: torch.Tensor, *, threshold: float = 0.0) -> torch.Tensor:
        return self.predicted_patch_lengths.to(boundary_logits.device)


def build_teacher_config() -> TernaryBLTConfig:
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


def test_blt_resume_eval_and_student_patcher() -> bool:
    teacher = ToyTeacher(TernaryBLTModel(build_teacher_config()))
    with TemporaryDirectory() as tempdir:
        checkpoint_path = Path(tempdir) / "blt-resume.pt"

        first_args = parse_args(
            [
                "--text",
                "Byte latent distillation",
                "--text",
                "Second tiny document",
                "--eval-text",
                "Held out eval text",
                "--steps",
                "1",
                "--batch-size",
                "2",
                "--eval-batch-size",
                "1",
                "--sequence-length",
                "8",
                "--max-document-bytes",
                "32",
                "--device",
                "cpu",
                "--teacher-device",
                "cpu",
                "--teacher-upstream-repo",
                str(Path(__file__).resolve().parents[1]),
                "--local-dim",
                "32",
                "--global-dim",
                "64",
                "--decoder-dim",
                "32",
                "--n-layers-local-encoder",
                "2",
                "--n-layers-global",
                "2",
                "--n-layers-local-decoder",
                "2",
                "--n-heads-local-encoder",
                "4",
                "--n-heads-global",
                "4",
                "--n-heads-local-decoder",
                "4",
                "--n-heads-cross",
                "4",
                "--patch-size",
                "3",
                "--max-patch-length",
                "8",
                "--disable-hadamard",
                "--disable-4bit-activations",
                "--eval-every",
                "1",
                "--eval-steps",
                "1",
                "--save-path",
                str(checkpoint_path),
                "--save-every",
                "1",
                "--student-patcher-mode",
                "teacher_then_student",
                "--student-patcher-dim",
                "96",
                "--student-patcher-layers",
                "1",
                "--student-patcher-heads",
                "2",
                "--student-patcher-warmup-steps",
                "5",
            ]
        )
        first_result = run_distillation(first_args, teacher_override=teacher)
        assert first_result["final_step"] == 1
        assert "eval_metrics" in first_result
        assert first_result["student_patcher_enabled"] is True
        assert checkpoint_path.exists(), f"Expected checkpoint at {checkpoint_path}"
        step_checkpoint = checkpoint_path.with_name("blt-resume-step1.pt")
        assert step_checkpoint.exists(), f"Expected periodic checkpoint at {step_checkpoint}"

        from utils import load_checkpoint_payload

        saved_payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
        assert saved_payload["student_patcher"] is not None, "Expected student patcher state in checkpoint"
        assert saved_payload["step"] == 1

        second_args = parse_args(
            [
                "--text",
                "Byte latent distillation",
                "--text",
                "Second tiny document",
                "--eval-text",
                "Held out eval text",
                "--steps",
                "1",
                "--batch-size",
                "2",
                "--eval-batch-size",
                "1",
                "--sequence-length",
                "8",
                "--max-document-bytes",
                "32",
                "--device",
                "cpu",
                "--teacher-device",
                "cpu",
                "--teacher-upstream-repo",
                str(Path(__file__).resolve().parents[1]),
                "--local-dim",
                "32",
                "--global-dim",
                "64",
                "--decoder-dim",
                "32",
                "--n-layers-local-encoder",
                "2",
                "--n-layers-global",
                "2",
                "--n-layers-local-decoder",
                "2",
                "--n-heads-local-encoder",
                "4",
                "--n-heads-global",
                "4",
                "--n-heads-local-decoder",
                "4",
                "--n-heads-cross",
                "4",
                "--patch-size",
                "3",
                "--max-patch-length",
                "8",
                "--disable-hadamard",
                "--disable-4bit-activations",
                "--eval-every",
                "1",
                "--eval-steps",
                "1",
                "--save-path",
                str(checkpoint_path),
                "--resume-from",
                str(checkpoint_path),
                "--student-patcher-mode",
                "off",
                "--student-patcher-dim",
                "64",
                "--student-patcher-layers",
                "2",
                "--student-patcher-heads",
                "4",
                "--student-patcher-warmup-steps",
                "0",
            ]
        )
        second_result = run_distillation(second_args, teacher_override=teacher)
        assert second_result["start_step"] == 1
        assert second_result["final_step"] == 2
        assert second_result["resumed_from"] == str(checkpoint_path.resolve())
        assert "eval_metrics" in second_result
        assert second_result["student_patcher_enabled"] is True
        assert second_result["metrics"]["student_patcher_active"] == 0.0

        resumed_payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
        resumed_args = resumed_payload["training_args"]
        assert resumed_args["student_patcher_mode"] == "teacher_then_student"
        assert resumed_args["student_patcher_dim"] == 96
        assert resumed_args["student_patcher_layers"] == 1
        assert resumed_args["student_patcher_heads"] == 2
        assert resumed_args["student_patcher_warmup_steps"] == 5

    print("BLT resume/eval/patcher tests passed")
    return True


def test_student_patcher_reruns_teacher_on_selected_patch_lengths() -> bool:
    config = build_teacher_config()
    student = TernaryBLTModel(config)
    teacher = RecordingTeacher(TernaryBLTModel(config))
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    patcher = FixedStudentPatcher(torch.tensor([[2, 2]], dtype=torch.long))
    patcher_optimizer = torch.optim.AdamW(patcher.parameters(), lr=1e-3)
    trainer = BLTDistillationTrainer(
        student,
        optimizer=optimizer,
        config=config,
        teacher=teacher,
        student_patcher=patcher,
        patcher_optimizer=patcher_optimizer,
        patcher_mode="student",
        device=torch.device("cpu"),
    )

    batch = BLTDistillationBatch(
        input_ids=torch.tensor([[config.bos_id, config.byte_to_token_id(ord("a")), config.byte_to_token_id(ord("b")), config.eos_id]], dtype=torch.long),
        labels=torch.tensor([[config.byte_to_token_id(ord("a")), config.byte_to_token_id(ord("b")), config.eos_id, config.eos_id]], dtype=torch.long),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        patch_lengths=torch.tensor([[3, 1]], dtype=torch.long),
    )

    trainer.train_step(batch)
    assert len(teacher.calls) == 2, teacher.calls
    assert teacher.calls[0].tolist() == [[3, 1]], teacher.calls[0].tolist()
    assert teacher.calls[1].tolist() == [[2, 2]], teacher.calls[1].tolist()
    print("BLT teacher rerun on student patcher takeover tests passed")
    return True


if __name__ == "__main__":
    test_blt_resume_eval_and_student_patcher()
    test_student_patcher_reruns_teacher_on_selected_patch_lengths()
