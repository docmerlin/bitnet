"""Smoke tests for BitNet training internals without Hugging Face data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
import torch.nn.functional as F

from config import TernaryConfig
from model import BitNetDeep
from training.checkpoint import TrainerState, load_checkpoint, save_checkpoint
from training.losses import language_modeling_loss, multi_token_loss
from training.runtime import create_optimizer
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
    )


def _tiny_args(optimizer: str = "cmud") -> argparse.Namespace:
    return argparse.Namespace(
        optimizer=optimizer,
        learning_rate=1e-4,
        mud_learning_rate=1e-3,
        mud_momentum=0.95,
        mud_passes=1,
        mud_block_size=64,
        no_optimizer_8bit=False,
        weight_decay=0.0,
        lion_beta1=0.9,
        lion_beta2=0.99,
        output_dir=".",
    )


def test_single_training_step_updates_parameters() -> bool:
    config = _tiny_config()
    model = BitNetDeep(config)
    optimizer = create_optimizer(model, _tiny_args())

    before = model.layers[0].ffn_up.weight.detach().clone()
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    labels = input_ids.clone()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    loss.backward()
    optimizer.step()
    after = model.layers[0].ffn_up.weight.detach()

    assert not torch.equal(after, before), "Expected at least one parameter to change after one optimizer step"
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


@pytest.mark.parametrize("checkpoint_step", [2, 7, 9])
def test_cli_resume_preserves_nondefault_lr_schedule_and_vocab(monkeypatch, tmp_path, checkpoint_step):
    import train
    from torch.optim.lr_scheduler import LambdaLR
    from training.schedules import lr_schedule_multiplier

    args = train.build_arg_parser().parse_args([
        "--total-tokens", "160", "--micro-batch-size", "1", "--sequence-length", "8",
        "--grad-accumulation-steps", "2", "--path-window-size", "4",
        "--warmup-steps", "3", "--cooldown-steps", "2", "--min-lr-ratio", "0.37",
        "--learning-rate", "0.007", "--mud-learning-rate", "0.013",
        "--vocab-size", "32768", "--tokenizer-max-patch-size", "6",
    ])
    config = _tiny_config()
    config.path_window_size = 4
    model = BitNetDeep(config)

    def optimizer_for(model, args):
        parameters = list(model.parameters())
        return torch.optim.SGD([
            {"params": parameters[:1], "lr": args.learning_rate},
            {"params": parameters[1:], "lr": args.mud_learning_rate},
        ])

    optimizer = optimizer_for(model, args)
    scheduler = LambdaLR(optimizer, lambda step: lr_schedule_multiplier(step, 10, 3, 2, 0.37))
    for _ in range(checkpoint_step):
        optimizer.step()
        scheduler.step()
    checkpoint = save_checkpoint(tmp_path, model, optimizer, scheduler, None,
                                 TrainerState(step=checkpoint_step), config, args, "resume.pt")
    expected = [scheduler.get_last_lr()]
    for _ in range(checkpoint_step, 11):
        optimizer.step()
        scheduler.step()
        expected.append(scheduler.get_last_lr())

    class Tokenizer:
        def __init__(self, **kwargs):
            assert kwargs == {"max_patch_size": 6, "vocab_size_target": 32768}

        def __len__(self):
            return 260

    class ResumeVerified(Exception):
        pass

    def check_resume(path, restored_model, restored_optimizer, restored_scheduler, scaler):
        load_checkpoint(path, restored_model, restored_optimizer, restored_scheduler, scaler)
        assert restored_model.config.vocab_size == 512  # Not the new tokenizer length.
        actual = [restored_scheduler.get_last_lr()]
        for _ in range(checkpoint_step, 11):
            restored_optimizer.step()
            restored_scheduler.step()
            actual.append(restored_scheduler.get_last_lr())
        assert actual == expected
        raise ResumeVerified

    monkeypatch.setattr(train, "HierarchicalTokenizer", Tokenizer)
    monkeypatch.setattr(train, "create_optimizer", optimizer_for)
    monkeypatch.setattr(train, "build_batch_stream", lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(train, "load_checkpoint", check_resume)
    monkeypatch.setattr("sys.argv", ["train", "--resume-from", str(checkpoint),
                                    "--output-dir", str(tmp_path), "--device", "cpu"])
    with pytest.raises(ResumeVerified):
        train.main()


def test_resume_rejects_missing_schedule_configuration():
    from training.checkpoint import restore_resume_args

    with pytest.raises(ValueError, match="Cannot resume LR schedule"):
        restore_resume_args(argparse.Namespace(), {})


def test_soft_resume_pre_mid_checkpoint_identity_inits_mid() -> bool:
    """Pre-mid checkpoints load non-strict; mid mats get identity, not Kaiming."""
    from torch.optim.lr_scheduler import LambdaLR

    config = _tiny_config()
    model = BitNetDeep(config)
    args = _tiny_args()
    optimizer = create_optimizer(model, args)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    state = TrainerState(step=11, tokens_processed=256, samples_processed=8)

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
            checkpoint_name="pre_mid.pt",
        )
        payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
        # Simulate a 2-stage FFN checkpoint: drop all mid weights.
        stripped = {
            k: v for k, v in payload["model"].items()
            if "ffn_mid" not in k and "w_mid" not in k and "mid_proj" not in k
        }
        assert any("ffn_up" in k for k in stripped)
        assert not any("ffn_mid" in k for k in stripped)
        payload["model"] = stripped
        # Corrupt optimizer param-group sizes so a naive load would fail if attempted.
        payload["optimizer"] = {"param_groups": [], "state": {}}
        torch.save(payload, checkpoint_path)

        restored = BitNetDeep(config)
        # Scramble mid so we can detect identity upgrade.
        with torch.no_grad():
            for name, param in restored.named_parameters():
                if "ffn_mid" in name and name.endswith("weight"):
                    param.normal_()
        restored_opt = create_optimizer(restored, args)
        restored_sched = LambdaLR(restored_opt, lr_lambda=lambda _: 1.0)
        restored_state = load_checkpoint(
            checkpoint_path, restored, restored_opt, restored_sched, None,
        )
        assert restored_state.step == 11
        assert restored_state.tokens_processed == 256

        mid_mats = [
            p for n, p in restored.named_parameters()
            if "ffn_mid" in n and n.endswith("weight")
        ]
        assert mid_mats, "expected dense ffn_mid weights"
        for mat in mid_mats:
            # eye(N)*N, because these are ternary weights: the per-output-channel
            # scale is mean(|row|), which is 1/N for a plain identity row, so a
            # raw eye(N) would quantise to eye(N)/N -- an attenuator rather than
            # the pass-through the upgrade exists to install.
            size = mat.size(0)
            eye = torch.eye(size, dtype=mat.dtype) * size
            assert torch.allclose(mat.detach().cpu(), eye, atol=1e-5), mat

        # Non-mid weights should still match the source model (up/down restored).
        src_up = model.layers[0].ffn_up.weight.detach()
        dst_up = restored.layers[0].ffn_up.weight.detach()
        assert torch.allclose(src_up, dst_up), "ffn_up should restore from stripped ckpt"

    print("BitNet pre-mid soft-resume identity-init tests passed")
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
    test_soft_resume_pre_mid_checkpoint_identity_inits_mid()
    test_z_loss_matches_cross_entropy_when_disabled()
    test_multi_token_loss_handles_short_sequences()
