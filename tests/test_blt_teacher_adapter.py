"""Tests for optional Meta BLT teacher adapter behavior."""

from pathlib import Path

import pytest
import torch
from torch import nn

from blt.patching.teacher_patcher import patch_ids_from_lengths
from blt.teacher.facebook_blt import FacebookBLTTeacher, import_upstream_blt


class FakeLocalEncoder(nn.Module):
    def forward(self, *, tokens, **kwargs):
        hidden = tokens.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 1, 2)
        return (hidden, None), None


class FakeGlobalTransformer(nn.Module):
    def forward(self, *, embeds, tokens):
        return embeds, None


class FakeDecoderLayer(nn.Module):
    def forward(self, hidden, *, mask, **kwargs):
        if mask is None:
            mask = hidden.new_zeros((hidden.size(0), hidden.size(1), hidden.size(1)))
        return hidden + torch.softmax(mask, dim=-1) @ hidden


class FakeDecoderModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding_projection = None
        self.cross_attn_all_layers_decoder = False
        self.cross_attn_layers = []
        self.layers = nn.ModuleList([FakeDecoderLayer()])
        self.norm = nn.Identity()
        self.output = nn.Identity()

    def forward(self, *, tokens, embeds, patch_embeds, cross_mask):
        positions = torch.arange(tokens.size(1), device=tokens.device)
        docs = (tokens == 2).long().cumsum(dim=1) - (tokens == 2).long()
        allowed = (positions[:, None] >= positions[None, :]) & (
            docs[:, :, None] == docs[:, None, :]
        )
        self.last_mask = torch.where(allowed, 0.0, float("-inf"))
        if cross_mask is not None:
            patch_embeds = torch.softmax(cross_mask, dim=-1) @ patch_embeds
        hidden = embeds + patch_embeds
        for layer in self.layers:
            hidden = layer(hidden, mask=self.last_mask)
        return self.output(self.norm(hidden)), None


class FakeTeacherModel(nn.Module):
    def __init__(self, cross_attn_decoder=False) -> None:
        super().__init__()
        self.cross_attn_encoder = False
        self.cross_attn_decoder = cross_attn_decoder
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
        self.patching_mode = "entropy"
        self.boe_id = 0
        self.eos_id = 2
        self.local_encoder = FakeLocalEncoder()
        self.global_transformer = FakeGlobalTransformer()
        self.local_decoder = FakeDecoderModule()
        self.calls = 0

    def forward(self, *, tokens, patch_lengths):
        # Mirror upstream's native alignment and module calls, not the adapter.
        self.calls += 1
        assert torch.all(patch_lengths[:, 0] == 1)
        nb_boe = self.patch_size - 1 if self.patching_mode == "" else 0
        encoder_tokens = tokens
        if nb_boe:
            encoder_tokens = torch.cat([tokens.new_full((tokens.size(0), nb_boe), self.boe_id), tokens], dim=1)
            patch_lengths[:, 0] += nb_boe
        ids = patch_ids_from_lengths(patch_lengths, encoder_tokens.size(1))
        (hidden, _), _ = self.local_encoder(tokens=encoder_tokens)
        pooled = fake_downsample(hidden, patch_lengths.size(1), patch_lengths, ids, "mean", 5)
        patches, _ = self.global_transformer(embeds=pooled, tokens=tokens)
        decoder_ids = patch_ids_from_lengths(patch_lengths[:, 1:], tokens.size(1))
        cross_mask = None
        if self.cross_attn_decoder:
            allowed = decoder_ids[:, :, None] == torch.arange(patches.size(1))[None, None]
            cross_mask = torch.where(allowed, 0.0, float("-inf"))
        else:
            patches = patches.gather(1, decoder_ids[:, :, None].expand(-1, -1, patches.size(-1)))
        output, _ = self.local_decoder(
            tokens=tokens, embeds=hidden[:, nb_boe:], patch_embeds=patches, cross_mask=cross_mask
        )
        return output


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
        upstream={"patch_ids_from_lengths": patch_ids_from_lengths},
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
    assert output.patch_lengths.tolist() == [[1, 2], [1, 2]], output.patch_lengths.tolist()
    print("BLT teacher adapter masking tests passed")
    return True


@pytest.mark.parametrize("cross_attn_decoder", [False, True])
def test_teacher_native_forward_preserves_causality_and_document_masks(cross_attn_decoder):
    model = FakeTeacherModel(cross_attn_decoder)
    teacher = FacebookBLTTeacher(model=model, upstream={"patch_ids_from_lengths": patch_ids_from_lengths})
    tokens = torch.tensor([[1, 10, 11, 12, 2, 20, 21, 22]])
    lengths = torch.tensor([[1, 3, 1, 3]])
    output = teacher.forward(tokens, patch_lengths=lengths)
    assert model.calls == 1, "Adapter must call native forward"
    torch.testing.assert_close(output.logits, model(tokens=tokens, patch_lengths=lengths))
    torch.testing.assert_close(output.decoder_hidden, output.logits)
    assert torch.isneginf(model.local_decoder.last_mask[0, 5:, :5]).all()

    changed = tokens.clone()
    changed[:, 3] = 90  # A future byte inside the same multi-byte patch.
    after = teacher.forward(changed, patch_lengths=lengths)
    torch.testing.assert_close(output.logits[:, :3], after.logits[:, :3])
    torch.testing.assert_close(output.decoder_hidden[:, :3], after.decoder_hidden[:, :3])
    assert not torch.equal(output.logits[:, 3:], after.logits[:, 3:])
    assert not model.local_encoder._forward_hooks
    assert not model.global_transformer._forward_hooks
    assert not model.local_decoder.norm._forward_pre_hooks


def test_teacher_native_hooks_removed_on_failure(monkeypatch):
    model = FakeTeacherModel()
    teacher = FacebookBLTTeacher(model=model, upstream={"patch_ids_from_lengths": patch_ids_from_lengths})

    def fail(**kwargs):
        raise RuntimeError("native forward failed")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="native forward failed"):
        teacher.forward(torch.tensor([[1, 10]]), patch_lengths=torch.tensor([[1, 1]]))
    assert not model.local_encoder._forward_hooks
    assert not model.global_transformer._forward_hooks
    assert not model.local_decoder.norm._forward_pre_hooks


def test_teacher_handles_single_byte_and_unequal_patch_counts():
    teacher = FacebookBLTTeacher(
        model=FakeTeacherModel(), upstream={"patch_ids_from_lengths": patch_ids_from_lengths}
    )
    single = teacher.forward(torch.tensor([[1]]), patch_lengths=torch.tensor([[1]]))
    assert single.patch_lengths.tolist() == [[1]]
    assert single.global_hidden.shape == (1, 1, 2)
    tokens = torch.tensor([[1, 10, 11, 12], [1, 20, 21, 22]])
    lengths = torch.tensor([[1, 3, 0], [1, 1, 2]])
    batch = teacher.forward(tokens, patch_lengths=lengths)
    for row in range(2):
        alone = teacher.forward(tokens[row:row + 1], patch_lengths=lengths[row:row + 1])
        torch.testing.assert_close(batch.logits[row:row + 1], alone.logits)


def test_teacher_static_boe_padding_does_not_mutate_returned_lengths():
    model = FakeTeacherModel()
    model.patching_mode = ""
    teacher = FacebookBLTTeacher(model=model, upstream={"patch_ids_from_lengths": patch_ids_from_lengths})
    tokens = torch.tensor([[1, 10, 11, 12]])
    lengths = torch.tensor([[1, 3]])
    output = teacher.forward(tokens, patch_lengths=lengths)
    assert lengths.tolist() == output.patch_lengths.tolist() == [[1, 3]]
    assert output.encoder_hidden.shape == (1, 4, 2)
    torch.testing.assert_close(output.logits, model(tokens=tokens, patch_lengths=lengths.clone()))
    torch.testing.assert_close(output.encoder_hidden[..., 0], tokens.float())


if __name__ == "__main__":
    test_teacher_adapter_requires_valid_upstream_repo()
    test_teacher_adapter_respects_attention_mask_padding()
