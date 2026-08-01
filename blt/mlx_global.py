"""The BitNet stack, wired in as BLT's global transformer.

BLT's global model is just "a transformer over patch latents", which is exactly
what :class:`mlx_model.MLXBitNet` is once you stop feeding it token ids. This
adapter presents it under the interface
:class:`blt.mlx_model.MLXTernaryBLTModel` expects -- ``(patch_states,
attention_mask) -> latents`` -- so the byte-level front end gains PaTH
attention, Infini memory and RFMoE without either stack
learning about the other.

Two things the wiring has to get right.

*Engram is off.* It hashes token n-grams, which needs a discrete vocabulary;
patches have no ids. That is not a limitation of the adapter so much as a
statement about where n-gram memory belongs -- Meta's BLT puts hash n-gram
embeddings in the *local encoder*, over bytes, which is the level at which
n-grams exist. ``MLXBitNet.hidden_states`` raises rather than silently
producing garbage if Engram is left on.

*Padded patches are isolated.* Bucketing the patch axis for ``mx.compile`` and
normalising ragged rows both leave trailing zero-length patches. Causal
attention alone would tolerate them -- they sit after every real patch and
nothing reads their output -- but Infini memory *writes*, so an unmasked pad
would fold junk into the memory that later real patches read back. They get
their own segment id, which keeps them out of both attention and the memory.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from blt.config import TernaryBLTConfig
from mlx_model import MLXBitNet, MLXBitNetConfig


def global_config_for(
    config: TernaryBLTConfig,
    *,
    num_layers: int | None = None,
    num_heads: int | None = None,
    **overrides,
) -> MLXBitNetConfig:
    """An :class:`MLXBitNetConfig` sized to slot into ``config`` as the global model.

    ``vocab_size`` is set but unused: the adapter never embeds or unembeds, and
    BLT's own output head owns the vocabulary. Engram defaults off for the
    reason in the module docstring.
    """
    settings = dict(
        hidden_size=config.global_dim,
        num_attention_heads=num_heads if num_heads is not None else config.n_heads_global,
        vocab_size=config.vocab_size,
        num_prelude_layers=1,
        num_recurrent_layers=max((num_layers or config.n_layers_global) - 2, 1),
        num_coda_layers=1,
        num_loops=1,
        use_engram=False,
        mtp_depth=0,
    )
    settings.update(overrides)
    return MLXBitNetConfig(**settings)


class MLXBitNetGlobalTransformer(nn.Module):
    """Drop-in replacement for :class:`blt.mlx_model.MLXGlobalTransformer`.

    Unlike that one it cannot take zero-length padding on the patch axis; see
    :attr:`accepts_padding`.
    """

    #: False: padding perturbs this backbone (module docstring). Callers check
    #: this rather than isinstance, so an alternative backbone can say otherwise.
    accepts_padding = False

    def __init__(self, config: TernaryBLTConfig, global_config: MLXBitNetConfig | None = None) -> None:
        super().__init__()
        self.global_config = global_config or global_config_for(config)
        if self.global_config.hidden_size != config.global_dim:
            raise ValueError(
                f"global model width {self.global_config.hidden_size} != BLT global_dim {config.global_dim}"
            )
        if self.global_config.use_engram:
            raise ValueError(
                "Engram needs token ids and patches have none; build the global "
                "config with use_engram=False"
            )
        self.backbone = MLXBitNet(self.global_config)
        # Reading the mask to detect padding forces a GPU sync, which makes the
        # step uncompilable. On by default so a stray padded batch is caught; a
        # caller that guarantees unpadded patches by construction (a fixed patch
        # count) turns it off. See accepts_padding.
        self.validate_inputs = True
        self.output_norm = nn.RMSNorm(config.global_dim, eps=1.1920928955078125e-07)

    def __call__(
        self,
        patch_states: mx.array,
        *,
        attention_mask: mx.array | None = None,
        num_loops: int | None = None,
    ) -> mx.array:
        segment_ids = None
        if attention_mask is not None:
            mask = attention_mask.astype(mx.bool_)
            if self.validate_inputs and not bool(mx.all(mask)):
                # Segment isolation is not enough, and neither is fixing the
                # PaTH block width. Measured: a single block is exact, but drift
                # appears at two and amplifies with depth -- 0.0 / 1.4e-3 /
                # 2.2e-1 at 1 / 2 / 8 layers, the last being a relative error of
                # 1.0, i.e. a completely different output. It does not grow with
                # the *amount* of padding, so this is a small perturbation being
                # amplified through depth by the 4-bit activation quantisation,
                # which is a step function: a tiny shift flips a bucket, which
                # shifts more, and so on.
                #
                # BLT's own global transformer tolerates padding exactly, so
                # bucketing is safe there and only there. For a fixed patch
                # count with this backbone, patch by count rather than by
                # threshold -- see MLXByteEntropyModel.predict_patch_lengths's
                # ``num_patches``, which gives stable shapes with no padding.
                raise ValueError(
                    "the BitNet global backbone cannot take padded patches: the "
                    "perturbation amplifies through depth to a completely "
                    "different output. Use patch_bucket=0 with a fixed patch "
                    "count (patches_per_sequence), or MLXGlobalTransformer."
                )
            segment_ids = mx.zeros(mask.shape, dtype=mx.int32)
        hidden = self.backbone.hidden_states(
            inputs_embeds=patch_states,
            segment_ids=segment_ids,
            num_loops=num_loops,
        )
        return self.output_norm(hidden)
