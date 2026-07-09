"""Regression tests for BLT padding and mask handling."""


import torch

from blt.config import TernaryBLTConfig
from blt.data import ByteVocabulary
from blt.losses import DistillationLossWeights, compute_blt_distillation_loss
from blt.model import TernaryBLTModel, TernaryBLTOutput


def build_config(*, pad_id: int = -1) -> TernaryBLTConfig:
    return TernaryBLTConfig(
        pad_id=pad_id,
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


def make_student_output(logits: torch.Tensor) -> TernaryBLTOutput:
    batch_size, seq_len, hidden_dim = logits.shape
    patch_lengths = torch.full((batch_size, 1), seq_len, dtype=torch.long)
    patch_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    hidden = torch.zeros(batch_size, seq_len, hidden_dim)
    patches = torch.zeros(batch_size, 1, hidden_dim)
    return TernaryBLTOutput(
        logits=logits,
        patch_lengths=patch_lengths,
        patch_ids=patch_ids,
        encoder_hidden=hidden,
        encoder_patches=patches,
        global_hidden=patches,
        decoder_hidden=hidden,
    )


def test_masked_hard_ce_ignores_masked_labels() -> bool:
    logits = torch.tensor(
        [[[5.0, 0.0], [0.0, 5.0], [3.0, 1.0], [1.0, 3.0]]],
        dtype=torch.float32,
    )
    student = make_student_output(logits)
    attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    weights = DistillationLossWeights(hard_ce=1.0, logits_kl=0.0, encoder_patch_mse=0.0, global_patch_mse=0.0, decoder_hidden_mse=0.0)

    first_loss, first_metrics = compute_blt_distillation_loss(
        student,
        None,
        labels=torch.tensor([[0, 1, 0, 1]], dtype=torch.long),
        attention_mask=attention_mask,
        weights=weights,
        temperature=1.0,
    )
    second_loss, second_metrics = compute_blt_distillation_loss(
        student,
        None,
        labels=torch.tensor([[0, 1, 1, 0]], dtype=torch.long),
        attention_mask=attention_mask,
        weights=weights,
        temperature=1.0,
    )

    assert torch.allclose(first_loss, second_loss), "Masked labels should not affect hard CE"
    assert first_metrics["hard_ce"] == second_metrics["hard_ce"], "Masked labels should not affect hard CE metrics"
    print("BLT masked hard-CE tests passed")
    return True


def test_local_encoder_ignores_padded_tokens_in_patch_states() -> bool:
    torch.manual_seed(0)
    config = build_config()
    model = TernaryBLTModel(config).eval()
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
    patch_lengths = torch.tensor([[3, 3]], dtype=torch.long)
    valid_prefix = [config.bos_id, config.byte_to_token_id(ord("b")), config.byte_to_token_id(ord("l")), config.eos_id]
    first = torch.tensor([valid_prefix + [config.byte_to_token_id(ord("x")), config.byte_to_token_id(ord("y"))]], dtype=torch.long)
    second = torch.tensor([valid_prefix + [config.byte_to_token_id(ord("q")), config.byte_to_token_id(ord("z"))]], dtype=torch.long)

    with torch.no_grad():
        first_output = model(first, attention_mask=attention_mask, patch_lengths=patch_lengths)
        second_output = model(second, attention_mask=attention_mask, patch_lengths=patch_lengths)

    assert first_output.patch_lengths.tolist() == [[3, 1]], first_output.patch_lengths.tolist()
    assert second_output.patch_lengths.tolist() == [[3, 1]], second_output.patch_lengths.tolist()
    assert first_output.patch_ids.tolist() == [[0, 0, 0, 1, -1, -1]], first_output.patch_ids.tolist()
    assert torch.allclose(
        first_output.encoder_patches,
        second_output.encoder_patches,
        atol=1e-6,
        rtol=1e-5,
    ), "Padded token ids should not affect encoder patch states"
    assert torch.allclose(
        first_output.logits[:, :4, :],
        second_output.logits[:, :4, :],
        atol=1e-6,
        rtol=1e-5,
    ), "Padded token ids should not affect valid-token logits"
    assert torch.count_nonzero(first_output.encoder_hidden[:, 4:, :]) == 0, "Masked encoder rows should stay zeroed"
    assert torch.count_nonzero(first_output.decoder_hidden[:, 4:, :]) == 0, "Masked decoder rows should stay zeroed"
    assert torch.count_nonzero(first_output.logits[:, 4:, :]) == 0, "Masked decoder logits should stay zeroed"
    print("BLT padded encoder mask tests passed")
    return True


def test_model_infers_attention_mask_from_pad_id() -> bool:
    torch.manual_seed(1)
    config = build_config(pad_id=260)
    model = TernaryBLTModel(config).eval()
    input_ids = torch.tensor(
        [[config.bos_id, config.byte_to_token_id(ord("b")), config.byte_to_token_id(ord("l")), config.eos_id, config.pad_id, config.pad_id]],
        dtype=torch.long,
    )

    with torch.no_grad():
        output = model(input_ids)

    assert output.patch_lengths.tolist() == [[3, 1]], output.patch_lengths.tolist()
    assert output.patch_ids.tolist() == [[0, 0, 0, 1, -1, -1]], output.patch_ids.tolist()
    assert torch.count_nonzero(output.decoder_hidden[:, 4:, :]) == 0, "Implicit pad masking should zero decoder rows"
    assert torch.count_nonzero(output.logits[:, 4:, :]) == 0, "Implicit pad masking should zero decoder logits"
    print("BLT implicit pad-id masking tests passed")
    return True


def test_model_rejects_non_suffix_attention_mask() -> bool:
    config = build_config()
    model = TernaryBLTModel(config).eval()
    input_ids = torch.tensor(
        [[
            config.bos_id,
            config.byte_to_token_id(ord("b")),
            config.byte_to_token_id(ord("l")),
            config.byte_to_token_id(ord("t")),
            config.eos_id,
        ]],
        dtype=torch.long,
    )
    attention_mask = torch.tensor([[1, 0, 1, 1, 1]], dtype=torch.long)

    try:
        model(input_ids, attention_mask=attention_mask, patch_lengths=torch.tensor([[2, 2]], dtype=torch.long))
    except ValueError as exc:
        assert "suffix-padded" in str(exc)
    else:
        raise AssertionError("Expected non-suffix BLT attention masks to be rejected")

    print("BLT non-suffix mask rejection tests passed")
    return True


def test_byte_vocabulary_decode_skips_pad_id() -> bool:
    config = build_config(pad_id=260)
    vocabulary = ByteVocabulary(config)
    decoded = vocabulary.decode(
        [config.bos_id, config.byte_to_token_id(ord("h")), config.byte_to_token_id(ord("i")), config.pad_id, config.pad_id, config.eos_id]
    )
    assert decoded == "hi", decoded
    print("BLT pad-id decode tests passed")
    return True


if __name__ == "__main__":
    test_masked_hard_ce_ignores_masked_labels()
    test_local_encoder_ignores_padded_tokens_in_patch_states()
    test_model_infers_attention_mask_from_pad_id()
    test_byte_vocabulary_decode_skips_pad_id()
    test_model_rejects_non_suffix_attention_mask()
