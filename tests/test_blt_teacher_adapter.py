"""Tests for optional Meta BLT teacher adapter behavior."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blt.teacher.facebook_blt import FacebookBLTTeacher, import_upstream_blt


class FakeLocalEncoder:
    def __call__(self, *, tokens, embeds, patch_embeds, cross_mask, num_patches, patch_ids):
        hidden = tokens.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 1, 2)
        return (hidden, None), None


class FakeGlobalTransformer:
    def __call__(self, *, embeds, tokens):
        return embeds, None


class FakeDecoderModule:
    def __init__(self) -> None:
        self.patch_embedding_projection = None
        self.cross_attn_all_layers_decoder = False
        self.cross_attn_layers = []
        self.layers = []
        self.norm = lambda x: x
        self.output = lambda x: x


class FakeTeacherModel:
    def __init__(self) -> None:
        self.cross_attn_encoder = False
        self.cross_attn_decoder = False
        self.cross_attn_k = None
        self.cross_attn_window_encoder = None
        self.cross_attn_window_decoder = None
        self.cross_attn_use_flex_attention = False
        self.encoder_hash_tok_embedding = None
        self.encoder_hash_byte_group_nb_functions = 0
        self.encoder_hash_byte_group_size = None
        self.encoder_hash_byte_group_vocab = 0
        self.downsampling_by_pooling = "mean"
        self.patch_size = 5
        self.boe_id = 0
        self.eos_id = 2
        self.local_encoder = FakeLocalEncoder()
        self.global_transformer = FakeGlobalTransformer()
        self.local_decoder = FakeDecoderModule()


def fake_downsample(hidden, num_patches, patch_lengths, patch_ids, downsampling_by_pooling, patch_size):
    pooled = hidden.new_zeros(hidden.size(0), patch_lengths.size(1), hidden.size(-1))
    for batch_index in range(hidden.size(0)):
        for patch_index in range(patch_lengths.size(1)):
            members = patch_ids[batch_index] == patch_index
            if torch.any(members):
                pooled[batch_index, patch_index] = hidden[batch_index, members].mean(dim=0)
    return pooled


def test_teacher_adapter_requires_valid_upstream_repo() -> bool:
    invalid_path = Path(__file__).resolve().parents[1] / "does-not-exist-blt-upstream"
    try:
        import_upstream_blt(invalid_path)
    except ImportError as exc:
        assert "facebookresearch/blt" in str(exc) or "bytelatent" in str(exc)
    else:
        raise AssertionError("Expected an ImportError for a missing upstream BLT checkout")

    print("BLT teacher adapter import guard tests passed")
    return True


def test_teacher_adapter_respects_attention_mask_padding() -> bool:
    teacher = FacebookBLTTeacher(
        model=FakeTeacherModel(),
        upstream={
            "patch_ids_from_lengths": lambda patch_lengths, seq_len: (torch.arange(seq_len).view(1, 1, seq_len) >= patch_lengths.cumsum(dim=-1).unsqueeze(-1)).sum(dim=1),
            "compute_hash_embeddings": lambda **_: None,
            "downsample": fake_downsample,
            "cross_attn_mask": lambda *args, **kwargs: None,
        },
        patcher=None,
        device=torch.device("cpu"),
    )
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.long)
    patch_lengths = torch.tensor([[5], [5]], dtype=torch.long)
    first = torch.tensor([[1, 10, 11, 99, 98], [1, 10, 11, 77, 66]], dtype=torch.long)
    output = teacher.forward(first, attention_mask=attention_mask, patch_lengths=patch_lengths)

    assert torch.allclose(
        output.logits[0, :3],
        output.logits[1, :3],
        atol=1e-6,
        rtol=1e-5,
    ), "Teacher valid-token outputs should ignore padded suffix token ids"
    assert torch.count_nonzero(output.logits[:, 3:, :]) == 0, "Teacher padded rows should be zeroed after trimming"
    assert output.patch_lengths.tolist() == [[3], [3]], output.patch_lengths.tolist()
    print("BLT teacher adapter masking tests passed")
    return True


if __name__ == "__main__":
    test_teacher_adapter_requires_valid_upstream_repo()
    test_teacher_adapter_respects_attention_mask_padding()
