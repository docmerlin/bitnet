"""Shape tests for the separate ternary BLT student."""


import torch

from blt.config import TernaryBLTConfig
from blt.model import TernaryBLTModel
from blt.patching.teacher_patcher import build_uniform_patch_lengths


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


def test_blt_forward_shapes() -> bool:
    torch.manual_seed(0)
    config = build_config()
    model = TernaryBLTModel(config)
    input_ids = torch.tensor(
        [
            [config.bos_id, config.byte_to_token_id(ord("h")), config.byte_to_token_id(ord("i")), config.eos_id, config.byte_to_token_id(ord("!")), config.eos_id],
            [config.bos_id, config.byte_to_token_id(ord("b")), config.byte_to_token_id(ord("l")), config.byte_to_token_id(ord("t")), config.byte_to_token_id(ord("?")), config.eos_id],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    patch_lengths = build_uniform_patch_lengths(input_ids.size(0), input_ids.size(1), config.patch_size, device=input_ids.device)
    output = model(input_ids, attention_mask=attention_mask, patch_lengths=patch_lengths)

    assert output.logits.shape == (2, 6, config.vocab_size)
    assert output.encoder_hidden.shape == (2, 6, config.local_dim)
    assert output.encoder_patches.shape == (2, 2, config.global_dim)
    assert output.global_hidden.shape == (2, 2, config.global_dim)
    assert output.decoder_hidden.shape == (2, 6, config.decoder_dim)
    assert output.patch_ids.shape == (2, 6)
    assert output.patch_lengths.sum(dim=1).tolist() == [6, 6]
    print("Ternary BLT shape tests passed")
    return True


if __name__ == "__main__":
    test_blt_forward_shapes()
