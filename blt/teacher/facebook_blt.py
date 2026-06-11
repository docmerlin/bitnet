"""Optional adapter for distilling from Meta's public BLT checkpoints."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

from blt.model import TernaryBLTOutput
from blt.patching.teacher_patcher import normalize_patch_lengths
from utils import validate_suffix_padded_mask


def import_upstream_blt(upstream_repo_path: str | Path | None = None) -> Dict[str, Any]:
    """Import upstream BLT modules from an installed package or repo checkout."""

    if upstream_repo_path is not None:
        repo_path = Path(upstream_repo_path).expanduser().resolve()
        if not (repo_path / "bytelatent").exists():
            raise ImportError(
                f"{repo_path} does not look like the facebookresearch/blt repository. "
                "Expected a sibling 'bytelatent' package."
            )
        repo_path_text = str(repo_path)
        if repo_path_text not in sys.path:
            sys.path.insert(0, repo_path_text)

    try:
        blt_module = importlib.import_module("bytelatent.model.blt")
        patcher_module = importlib.import_module("bytelatent.data.patcher")
    except ImportError as exc:
        raise ImportError(
            "Unable to import upstream BLT code. Install or checkout facebookresearch/blt "
            "and pass its path as upstream_repo_path. The upstream runtime also expects "
            "its own dependencies such as xformers."
        ) from exc

    return {
        "ByteLatentTransformer": getattr(blt_module, "ByteLatentTransformer"),
        "compute_hash_embeddings": getattr(blt_module, "compute_hash_embeddings"),
        "cross_attn_mask": getattr(blt_module, "cross_attn_mask"),
        "patch_ids_from_lengths": getattr(blt_module, "patch_ids_from_lengths"),
        "Patcher": getattr(patcher_module, "Patcher"),
        "PatcherArgs": getattr(patcher_module, "PatcherArgs"),
        "downsample": getattr(importlib.import_module("bytelatent.model.utils"), "downsample"),
    }


@dataclass(slots=True)
class FacebookBLTTeacher:
    """Thin wrapper over Meta BLT that exposes distillation-friendly tensors.

    The adapter is intentionally optional. It requires:

    - access to the gated Hugging Face checkpoints
    - a local checkout of `facebookresearch/blt` or an equivalent installed package
    - the upstream runtime dependencies such as `xformers`
    """

    model: Any
    upstream: Dict[str, Any]
    patcher: Any | None = None
    device: torch.device = torch.device("cpu")

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = "facebook/blt-1b",
        entropy_model_id: str | None = "facebook/blt-entropy",
        upstream_repo_path: str | Path | None = None,
        device: str = "cpu",
    ) -> "FacebookBLTTeacher":
        upstream = import_upstream_blt(upstream_repo_path)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required to download the gated BLT teacher checkpoints"
            ) from exc

        teacher_model = upstream["ByteLatentTransformer"].from_pretrained(model_id)
        teacher_model = teacher_model.to(device)
        teacher_model.eval()

        patcher = None
        if entropy_model_id is not None:
            entropy_dir = snapshot_download(repo_id=entropy_model_id)
            patcher_args = upstream["PatcherArgs"](
                patching_mode="entropy",
                entropy_model_checkpoint_dir=entropy_dir,
                realtime_patching=True,
                threshold=float(getattr(teacher_model, "patching_threshold", 1.335442066192627)),
                threshold_add=None,
                max_patch_length=int(getattr(teacher_model, "max_patch_length", 0) or 0) or None,
                patch_size=float(getattr(teacher_model, "patch_size", 4.5)),
                patching_batch_size=1,
                device=device,
                patching_device=device,
                monotonicity=False,
            )
            patcher = upstream["Patcher"](patcher_args)

        return cls(model=teacher_model, upstream=upstream, patcher=patcher, device=torch.device(device))

    def _resolve_patch_lengths(
        self,
        tokens: torch.Tensor,
        patch_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        if patch_lengths is None:
            if self.patcher is None:
                raise ValueError("No patch_lengths were provided and no teacher patcher is configured")
            patch_lengths, _ = self.patcher.patch(tokens, include_next_token=False)
        return normalize_patch_lengths(patch_lengths.to(self.device), tokens.size(1))

    def _forward_trimmed(
        self,
        tokens: torch.Tensor,
        patch_lengths: torch.Tensor,
    ) -> TernaryBLTOutput:
        patch_ids = self.upstream["patch_ids_from_lengths"](patch_lengths, tokens.size(1))
        cross_attn_mask_enc = None
        if self.model.cross_attn_encoder:
            cross_attn_mask_enc = self.upstream["cross_attn_mask"](
                patch_ids,
                patch_lengths,
                tokens.size(1),
                patches_as_queries=True,
                cross_attn_k=self.model.cross_attn_k,
                window=self.model.cross_attn_window_encoder,
                block_mask=self.model.cross_attn_use_flex_attention,
            )

        local_encoder_embeds = self.upstream["compute_hash_embeddings"](
            local_encoder_tokens=tokens,
            local_encoder=self.model.local_encoder,
            encoder_hash_tok_embedding=self.model.encoder_hash_tok_embedding,
            encoder_hash_byte_group_nb_functions=self.model.encoder_hash_byte_group_nb_functions,
            encoder_hash_byte_group_size=self.model.encoder_hash_byte_group_size,
            encoder_hash_byte_group_vocab=self.model.encoder_hash_byte_group_vocab,
        )

        (h_encoder, h_cross), _ = self.model.local_encoder(
            tokens=tokens,
            embeds=local_encoder_embeds,
            patch_embeds=None,
            cross_mask=cross_attn_mask_enc,
            num_patches=patch_lengths.shape[1],
            patch_ids=patch_ids,
        )

        if not self.model.cross_attn_encoder:
            encoder_patches = self.upstream["downsample"](
                h_encoder,
                patch_lengths.shape[1],
                patch_lengths,
                patch_ids,
                downsampling_by_pooling=self.model.downsampling_by_pooling,
                patch_size=self.model.patch_size,
            )
        else:
            encoder_patches = h_cross.view(tokens.size(0), patch_lengths.shape[1], -1)

        global_tokens = tokens.new_full((encoder_patches.size(0), encoder_patches.size(1)), int(self.model.boe_id))
        rows, cols = torch.where(tokens == self.model.eos_id)
        eos_patch_ids = patch_ids[rows, cols]
        global_tokens[rows, eos_patch_ids] = self.model.eos_id
        global_hidden, _ = self.model.global_transformer(embeds=encoder_patches, tokens=global_tokens)

        decoder_patch_ids = patch_ids
        patch_for_decoder = global_hidden
        cross_attn_mask_dec = None
        if self.model.cross_attn_decoder:
            cross_attn_mask_dec = self.upstream["cross_attn_mask"](
                decoder_patch_ids,
                patch_lengths,
                tokens.size(1),
                patches_as_queries=False,
                cross_attn_k=self.model.cross_attn_k,
                window=self.model.cross_attn_window_decoder,
                block_mask=self.model.cross_attn_use_flex_attention,
            )
        else:
            patch_for_decoder = torch.gather(
                global_hidden,
                1,
                decoder_patch_ids.unsqueeze(-1).expand(-1, -1, global_hidden.size(-1)),
            )

        decoder_hidden = h_encoder
        decoder_module = self.model.local_decoder
        if getattr(decoder_module, "patch_embedding_projection", None) is not None:
            patch_for_decoder = decoder_module.patch_embedding_projection(patch_for_decoder)
            if self.model.cross_attn_k is not None:
                patch_for_decoder = patch_for_decoder.reshape(
                    tokens.size(0),
                    patch_for_decoder.shape[1] * self.model.cross_attn_k,
                    decoder_module.dim,
                )

        if patch_for_decoder is not None and not self.model.cross_attn_decoder:
            decoder_hidden = decoder_hidden + patch_for_decoder

        for layer_index, layer in enumerate(decoder_module.layers):
            if self.model.cross_attn_decoder and (
                layer_index == 0 or decoder_module.cross_attn_all_layers_decoder
            ):
                decoder_hidden = decoder_hidden + decoder_module.cross_attn_layers[layer_index](
                    x=decoder_hidden,
                    kv=patch_for_decoder,
                    mask=cross_attn_mask_dec,
                )
            decoder_hidden = layer(
                decoder_hidden,
                mask=None,
                freq_cis=decoder_module.rope(seqlen=tokens.size(1)) if decoder_module.use_rope else None,
                attn_impl=decoder_module.attn_impl,
            )

        logits = decoder_module.output(decoder_module.norm(decoder_hidden)).float()
        return TernaryBLTOutput(
            logits=logits,
            patch_lengths=patch_lengths,
            patch_ids=patch_ids,
            encoder_hidden=h_encoder,
            encoder_patches=encoder_patches,
            global_hidden=global_hidden,
            decoder_hidden=decoder_hidden,
        )

    @staticmethod
    def _slice_output(output: TernaryBLTOutput, batch_index: int) -> TernaryBLTOutput:
        return TernaryBLTOutput(
            logits=output.logits[batch_index : batch_index + 1],
            patch_lengths=output.patch_lengths[batch_index : batch_index + 1],
            patch_ids=output.patch_ids[batch_index : batch_index + 1],
            encoder_hidden=output.encoder_hidden[batch_index : batch_index + 1],
            encoder_patches=output.encoder_patches[batch_index : batch_index + 1],
            global_hidden=output.global_hidden[batch_index : batch_index + 1],
            decoder_hidden=output.decoder_hidden[batch_index : batch_index + 1],
        )

    def _forward_suffix_padded_batch(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_patch_lengths: torch.Tensor | None,
    ) -> TernaryBLTOutput:
        validate_suffix_padded_mask(attention_mask)
        batch_size, seq_len = tokens.shape
        valid_lengths = attention_mask.sum(dim=1)
        if torch.any(valid_lengths <= 0):
            raise ValueError("attention_mask must contain at least one valid token per sequence")

        outputs_by_index: dict[int, tuple[int, TernaryBLTOutput]] = {}
        max_num_patches = 0
        for valid_length in valid_lengths.unique(sorted=True).tolist():
            valid_length = int(valid_length)
            batch_indices = torch.nonzero(valid_lengths == valid_length, as_tuple=False).flatten()
            group_tokens = tokens[batch_indices, :valid_length]
            group_patch_lengths = None
            if batch_patch_lengths is not None:
                group_patch_lengths = batch_patch_lengths[batch_indices]
            resolved_patch_lengths = self._resolve_patch_lengths(group_tokens, group_patch_lengths)
            group_output = self._forward_trimmed(group_tokens, resolved_patch_lengths)
            for local_index, batch_index in enumerate(batch_indices.tolist()):
                row_output = self._slice_output(group_output, local_index)
                outputs_by_index[batch_index] = (valid_length, row_output)
                max_num_patches = max(max_num_patches, row_output.patch_lengths.size(1))

        first_output = outputs_by_index[min(outputs_by_index)][1]
        hidden_dim = first_output.encoder_hidden.size(-1)
        patch_dim = first_output.encoder_patches.size(-1)
        decoder_dim = first_output.decoder_hidden.size(-1)
        vocab_size = first_output.logits.size(-1)
        batch_logits = tokens.new_zeros((batch_size, seq_len, vocab_size), dtype=first_output.logits.dtype)
        batch_patch_lengths_out = tokens.new_zeros((batch_size, max_num_patches), dtype=torch.long)
        batch_patch_ids = tokens.new_zeros((batch_size, seq_len), dtype=torch.long)
        batch_encoder_hidden = tokens.new_zeros(
            (batch_size, seq_len, hidden_dim),
            dtype=first_output.encoder_hidden.dtype,
        )
        batch_encoder_patches = tokens.new_zeros(
            (batch_size, max_num_patches, patch_dim),
            dtype=first_output.encoder_patches.dtype,
        )
        batch_global_hidden = tokens.new_zeros(
            (batch_size, max_num_patches, patch_dim),
            dtype=first_output.global_hidden.dtype,
        )
        batch_decoder_hidden = tokens.new_zeros(
            (batch_size, seq_len, decoder_dim),
            dtype=first_output.decoder_hidden.dtype,
        )

        for batch_index, (valid_length, output) in outputs_by_index.items():
            num_patches = output.patch_lengths.size(1)
            batch_logits[batch_index, :valid_length] = output.logits[0]
            batch_patch_lengths_out[batch_index, :num_patches] = output.patch_lengths[0]
            batch_patch_ids[batch_index, :valid_length] = output.patch_ids[0]
            batch_encoder_hidden[batch_index, :valid_length] = output.encoder_hidden[0]
            batch_encoder_patches[batch_index, :num_patches] = output.encoder_patches[0]
            batch_global_hidden[batch_index, :num_patches] = output.global_hidden[0]
            batch_decoder_hidden[batch_index, :valid_length] = output.decoder_hidden[0]

        return TernaryBLTOutput(
            logits=batch_logits,
            patch_lengths=batch_patch_lengths_out,
            patch_ids=batch_patch_ids,
            encoder_hidden=batch_encoder_hidden,
            encoder_patches=batch_encoder_patches,
            global_hidden=batch_global_hidden,
            decoder_hidden=batch_decoder_hidden,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        patch_lengths: torch.Tensor | None = None,
    ) -> TernaryBLTOutput:
        tokens = input_ids.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(tokens, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        batch_patch_lengths = None
        if patch_lengths is not None:
            batch_patch_lengths = patch_lengths.to(self.device)

        if bool(torch.all(attention_mask)):
            resolved_patch_lengths = self._resolve_patch_lengths(tokens, batch_patch_lengths)
            return self._forward_trimmed(tokens, resolved_patch_lengths)

        return self._forward_suffix_padded_batch(tokens, attention_mask, batch_patch_lengths)
