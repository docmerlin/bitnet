"""CLI/runtime smoke tests for BLT distillation."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blt.config import TernaryBLTConfig
from blt.model import TernaryBLTModel
from blt.train_distill import parse_args, run_distillation


class ToyTeacher:
    def __init__(self, model: TernaryBLTModel) -> None:
        self.model = model.eval()

    @torch.no_grad()
    def forward(self, input_ids, *, attention_mask=None, patch_lengths=None):
        return self.model(input_ids, attention_mask=attention_mask, patch_lengths=patch_lengths)


class AssertingTeacher(ToyTeacher):
    def __init__(self, model: TernaryBLTModel) -> None:
        super().__init__(model)
        self.seen_patch_lengths = None

    @torch.no_grad()
    def forward(self, input_ids, *, attention_mask=None, patch_lengths=None):
        assert patch_lengths is not None, "Expected a fallback patch length source when teacher patching is disabled"
        self.seen_patch_lengths = patch_lengths.detach().clone()
        return super().forward(input_ids, attention_mask=attention_mask, patch_lengths=patch_lengths)


def test_blt_cli_runtime_smoke() -> bool:
    with TemporaryDirectory() as tempdir:
        checkpoint_path = Path(tempdir) / "blt-smoke.pt"
        args = parse_args(
            [
                "--text",
                "Byte latent distillation",
                "--text",
                "Second tiny document",
                "--steps",
                "1",
                "--batch-size",
                "2",
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
                "--save-path",
                str(checkpoint_path),
            ]
        )
        teacher_config = TernaryBLTConfig(
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
        result = run_distillation(args, teacher_override=ToyTeacher(TernaryBLTModel(teacher_config)))
        assert result["start_step"] == 0
        assert result["final_step"] == 1
        assert result["teacher_enabled"] is True
        assert checkpoint_path.exists(), f"Expected checkpoint at {checkpoint_path}"
        assert float(result["metrics"]["loss"]) > 0.0
    print("BLT distillation CLI smoke tests passed")
    return True


def test_disable_teacher_patcher_uses_static_patch_lengths() -> bool:
    args = parse_args(
        [
            "--text",
            "Byte latent distillation",
            "--text",
            "Second tiny document",
            "--steps",
            "1",
            "--batch-size",
            "2",
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
            "--disable-teacher-patcher",
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
        ]
    )
    teacher = AssertingTeacher(TernaryBLTModel(TernaryBLTConfig(
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
    )))
    result = run_distillation(args, teacher_override=teacher)
    assert result["teacher_enabled"] is True
    assert teacher.seen_patch_lengths is not None
    assert teacher.seen_patch_lengths.sum(dim=1).tolist() == [8, 8], teacher.seen_patch_lengths.tolist()
    print("BLT disable-teacher-patcher fallback tests passed")
    return True


def test_run_distillation_is_seeded_directly() -> bool:
    base_args = [
        "--text",
        "Byte latent distillation",
        "--text",
        "Second tiny document",
        "--steps",
        "1",
        "--batch-size",
        "2",
        "--sequence-length",
        "8",
        "--max-document-bytes",
        "32",
        "--device",
        "cpu",
        "--teacher-device",
        "cpu",
        "--no-teacher",
        "--seed",
        "123",
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
    ]
    first = run_distillation(parse_args(base_args))
    second = run_distillation(parse_args(base_args))
    assert first["metrics"] == second["metrics"], "Direct run_distillation() calls should honor --seed"
    print("BLT direct run_distillation seeding tests passed")
    return True


def test_resume_defaults_optimizer_to_adamw_for_legacy_checkpoints() -> bool:
    # Checkpoints written before C-MUD became the default have no "optimizer" key;
    # resume must fall back to AdamW rather than loading AdamW state into C-MUD.
    import argparse

    from blt.train_distill import apply_checkpoint_training_args

    args = argparse.Namespace(optimizer="cmud")
    apply_checkpoint_training_args(args, {"training_args": {"seed": 0}})  # no "optimizer" key
    assert args.optimizer == "adamw", args.optimizer

    args = argparse.Namespace(optimizer="cmud")
    apply_checkpoint_training_args(args, {"training_args": {"optimizer": "cmud"}})
    assert args.optimizer == "cmud", args.optimizer
    print("BLT legacy-checkpoint optimizer default tests passed")
    return True


if __name__ == "__main__":
    test_blt_cli_runtime_smoke()
    test_disable_teacher_patcher_uses_static_patch_lengths()
    test_run_distillation_is_seeded_directly()
    test_resume_defaults_optimizer_to_adamw_for_legacy_checkpoints()
