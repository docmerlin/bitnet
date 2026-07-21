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

    # Every stack uses TernaryMLP with mid_proj (SwiGLU expand → mid → down).
    local_hidden = max(int(config.local_dim * config.ffn_multiplier_local), config.local_dim)
    global_hidden = max(int(config.global_dim * config.ffn_multiplier_global), config.global_dim)
    decoder_hidden = max(int(config.decoder_dim * config.ffn_multiplier_decoder), config.decoder_dim)
    enc_mlp = model.local_encoder.blocks[0].mlp
    glob_mlp = model.global_transformer.blocks[0].mlp
    dec_mlp = model.local_decoder.blocks[0].mlp
    for mlp, h in ((enc_mlp, local_hidden), (glob_mlp, global_hidden), (dec_mlp, decoder_hidden)):
        assert hasattr(mlp, "mid_proj"), "TernaryMLP missing mid_proj"
        assert mlp.mid_proj.weight.shape == (h, h), mlp.mid_proj.weight.shape
        eye = torch.eye(h, dtype=mlp.mid_proj.weight.dtype)
        assert torch.allclose(mlp.mid_proj.weight.detach().cpu(), eye, atol=1e-5), (
            "mid_proj must cold-start as identity"
        )

    # Mid participates in backward (not dead/unused).
    model.train()
    loss = output.logits.float().mean()
    loss.backward()
    for mlp in (enc_mlp, glob_mlp, dec_mlp):
        assert mlp.mid_proj.weight.grad is not None, "mid_proj must get gradients"
        assert float(mlp.mid_proj.weight.grad.abs().sum()) > 0.0

    print("Ternary BLT shape tests passed")
    return True


def test_blt_logits_do_not_see_future_bytes_in_same_patch() -> None:
    torch.manual_seed(0)
    config = build_config()
    model = TernaryBLTModel(config).eval()
    first = torch.tensor([[config.bos_id, 10, 11, 12, 13, config.eos_id]])
    changed = first.clone()
    changed[0, 2] = 99
    mask = torch.ones_like(first)
    patch_lengths = torch.tensor([[3, 3]])

    first_logits = model(first, attention_mask=mask, patch_lengths=patch_lengths).logits
    changed_logits = model(changed, attention_mask=mask, patch_lengths=patch_lengths).logits

    assert torch.equal(first_logits[:, :2], changed_logits[:, :2])


if __name__ == "__main__":
    test_blt_forward_shapes()
