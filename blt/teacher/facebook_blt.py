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
        "patch_ids_from_lengths": getattr(blt_module, "patch_ids_from_lengths"),
        "Patcher": getattr(patcher_module, "Patcher"),
        "PatcherArgs": getattr(patcher_module, "PatcherArgs"),
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
        patch_lengths = normalize_patch_lengths(patch_lengths.to(self.device), tokens.size(1))
        # Native BLT requires one real byte in the first encoder patch (also
        # before its BOE padding in static mode). Preserve all other boundaries.
        rows = []
        for row in patch_lengths:
            lengths = row[row > 0]
            rows.append(torch.cat([lengths.new_ones(1), lengths[:1] - 1, lengths[1:]]))
        return torch.nn.utils.rnn.pad_sequence(
            [row[row > 0] for row in rows], batch_first=True
        )

    def _forward_trimmed(
        self,
        tokens: torch.Tensor,
        patch_lengths: torch.Tensor,
    ) -> TernaryBLTOutput:
        """Capture features without replacing native alignment or attention masks."""
        captured = {}

        def capture_encoder(module, args, output):
            captured["encoder_hidden"] = output[0][0]

        def capture_global(module, args, kwargs, output):
            captured["encoder_patches"] = kwargs["embeds"]
            captured["global_hidden"] = output[0]

        def capture_decoder(module, args):
            captured["decoder_hidden"] = args[0]

        handles = [
            self.model.local_encoder.register_forward_hook(capture_encoder),
            self.model.global_transformer.register_forward_hook(capture_global, with_kwargs=True),
            self.model.local_decoder.norm.register_forward_pre_hook(capture_decoder),
        ]
        try:
            # Upstream owns decoder_patch_ids_from_lengths, cross-attention,
            # and local/global causal and EOS document masks. Never replay layers.
            # Static-mode upstream forward adds BOE lengths in place.
            logits = self.model(tokens=tokens, patch_lengths=patch_lengths.clone())
        finally:
            for handle in handles:
                handle.remove()

        nb_boe = int(self.model.patch_size - 1) if self.model.patching_mode == "" else 0
        num_patches = patch_lengths.size(1)
        return TernaryBLTOutput(
            logits=logits,
            patch_lengths=patch_lengths,
            patch_ids=self.upstream["patch_ids_from_lengths"](patch_lengths, tokens.size(1)),
            encoder_hidden=captured["encoder_hidden"][:, nb_boe : nb_boe + tokens.size(1)],
            encoder_patches=captured["encoder_patches"][:, :num_patches],
            global_hidden=captured["global_hidden"][:, :num_patches],
            decoder_hidden=captured["decoder_hidden"],
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

    def _forward_grouped_batch(
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
            # Upstream's shifted decoder IDs count trailing zero lengths at the
            # final byte. Group by patch count as well, so native sees no padding.
            patch_counts = (resolved_patch_lengths > 0).sum(dim=1)
            for count in patch_counts.unique(sorted=True).tolist():
                members = torch.nonzero(patch_counts == count, as_tuple=False).flatten()
                group_output = self._forward_trimmed(
                    group_tokens[members], resolved_patch_lengths[members, :count]
                )
                for local_index, batch_index in enumerate(batch_indices[members].tolist()):
                    row_output = self._slice_output(group_output, local_index)
                    outputs_by_index[batch_index] = (valid_length, row_output)
                    max_num_patches = max(max_num_patches, count)

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
            (batch_size, max_num_patches, first_output.global_hidden.size(-1)),
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

        return self._forward_grouped_batch(tokens, attention_mask, batch_patch_lengths)
