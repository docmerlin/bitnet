"""Configuration for the separate ternary BLT stack."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TernaryBLTConfig:
    """Configuration for a ternary Byte Latent Transformer student.

    The vocabulary layout follows Meta BLT conventions:

    - ``0``: BOE
    - ``1``: BOS
    - ``2``: EOS
    - ``3``: BPE delimiter
    - ``4..259``: raw byte values offset by 4
    """

    byte_vocab_size: int = 256
    offset: int = 4
    boe_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    bpe_id: int = 3
    pad_id: int = -1

    local_dim: int = 256
    global_dim: int = 512
    decoder_dim: int = 256

    # Meta's BLT runs 1 encoder layer at 400M/1B and 3 at 2B-8B, on the grounds
    # that hash n-gram embeddings supply the multi-byte identity the encoder
    # would otherwise need depth to rebuild -- the paper's word is that the
    # encoder is "extremely light-weight" *when using* them. Those are in now
    # (use_ngram_embeddings), and the encoder is 30% of the forward, so this is
    # the cheap end of that trade: 4 -> 1 layer measured 3.4x on the encoder.
    #
    # Justified on speed and on Meta's architecture, NOT on measured quality: at
    # the scale this repo can train, seed-to-seed spread in held-out loss (0.138)
    # is larger than the gap between 1 and 4 encoder layers (0.052). See todo.md.
    n_layers_local_encoder: int = 1
    n_layers_global: int = 8
    # Meta's budget is encoder-light and decoder-heavy: 1/9 at 400M-1B, 3/7 at
    # 2B-8B. The decoder does the harder job -- turning a patch latent back into
    # individual bytes -- and unlike the encoder it has no n-gram embeddings to
    # lean on. Measured cost against 4 layers: +1.77M parameters and 193 vs 146
    # ms/step. Held-out loss does not separate them (see todo.md); this follows
    # the published architecture rather than a local measurement.
    n_layers_local_decoder: int = 7

    n_heads_local_encoder: int = 4
    n_heads_global: int = 8
    n_heads_local_decoder: int = 4
    n_heads_cross: int = 4

    # Key/value slots each patch presents to the decoder's cross-attention.
    #
    # A byte belongs to exactly one patch, so at k=1 the cross-attention mask is
    # one-hot, softmax over its single permitted key is a constant 1, and the
    # query/key projections cannot affect the output -- they take exactly zero
    # gradient. Meta's BLT defaults to 2 for this reason: the patch latent is
    # projected to k vectors, the byte attends across them, and the choice
    # becomes real. k=1 is still supported and takes a cheaper gather path that
    # skips the inert projections entirely.
    cross_attn_k: int = 2

    ffn_multiplier_local: float = 4.0
    ffn_multiplier_global: float = 4.0
    ffn_multiplier_decoder: float = 4.0

    local_window: int | None = 256
    dropout: float = 0.0
    rope_theta: float = 10000.0

    # Hashed byte n-gram embeddings summed into the byte embedding, as in Meta's
    # BLT (arXiv:2412.09871 eq. 3). They are what lets that model run a
    # one-layer local encoder: the paper calls the encoder "extremely
    # light-weight" *when paired with* hash n-gram embeddings, because the
    # n-grams supply the multi-byte identity the encoder would otherwise need
    # depth to rebuild.
    use_ngram_embeddings: bool = True
    ngram_sizes: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
    # Hashes per n-gram size. Meta uses ~500K total at 8B; that would be 128M
    # parameters at local_dim 256, i.e. larger than this whole model, so the
    # tables are narrow instead (below) and the count is swept separately.
    ngram_vocab_size: int = 16384
    # Table width. None -> local_dim // 4, then one shared projection back up.
    # Meta embeds at full width; at this scale the narrow-plus-project shape
    # borrowed from Engram is what keeps the tables from dwarfing the body.
    ngram_dim: int | None = None

    # Meta's teacher averages ~4.5 bytes per patch and the BLT paper's entropy
    # patcher runs finer still; 4 keeps the uniform fallback in that range rather
    # than handing the global model a 6x-compressed sequence it was not tuned for.
    patch_size: int = 4
    max_patch_length: int = 32

    use_hadamard: bool = True
    # Whether activations are fake-quantised at all. The name is historical --
    # the width is ``activation_bits``, not fixed at 4.
    use_4bit_activations: bool = True
    # Activation quantisation width. 8 rather than 4 because 4 buys nothing and
    # costs stability: quantisation here is fake (``x + stop_gradient(q - x)``),
    # the tensor stays float and the matmul is float x ternary either way, so
    # fewer bits is measurably *slower* (skipping it entirely is 1.29x on the
    # forward) and only changes the rounding grid. 4-bit also diverges from a
    # cold start -- 7 levels is not enough for the per-token max scale -- while
    # 8 bits gives 127. Set 4 only to match a deployment that really does
    # quantise activations to 4 bits in its kernels.
    activation_bits: int = 8

    # Auxiliary byte-level future heads. These operate after the causal local
    # decoder; patch-level MTP is undefined because patches have variable widths.
    mtp_depth: int = 0

    distill_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.byte_vocab_size <= 0:
            raise ValueError("byte_vocab_size must be positive")
        if self.offset <= 0:
            raise ValueError("offset must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.max_patch_length <= 0:
            raise ValueError("max_patch_length must be positive")
        if self.distill_temperature <= 0:
            raise ValueError("distill_temperature must be positive")
        if self.cross_attn_k <= 0:
            raise ValueError("cross_attn_k must be positive")
        if self.activation_bits < 2:
            raise ValueError("activation_bits must be at least 2")
        if self.mtp_depth < 0:
            raise ValueError("mtp_depth must be non-negative")
        if self.use_ngram_embeddings:
            sizes = tuple(int(size) for size in self.ngram_sizes)
            if not sizes or min(sizes) < 1:
                raise ValueError("ngram_sizes must be non-empty and positive")
            if len(set(sizes)) != len(sizes):
                raise ValueError("ngram_sizes must not repeat a size")
            object.__setattr__(self, "ngram_sizes", tuple(sorted(sizes)))
            if self.ngram_vocab_size < 1:
                raise ValueError("ngram_vocab_size must be positive")
            if self.ngram_dim is None:
                object.__setattr__(self, "ngram_dim", max(self.local_dim // 4, 1))
            elif self.ngram_dim < 1:
                raise ValueError("ngram_dim must be positive")
        if self.pad_id < -1:
            raise ValueError("pad_id must be -1 or a non-negative token id")
        if 0 <= self.pad_id < self.offset + self.byte_vocab_size:
            raise ValueError("pad_id must be -1 or a dedicated token id outside the base BLT vocabulary")

        self._validate_dim(self.local_dim, self.n_heads_local_encoder, "local")
        self._validate_dim(self.global_dim, self.n_heads_global, "global")
        self._validate_dim(self.decoder_dim, self.n_heads_local_decoder, "decoder")

        if self.global_dim % self.n_heads_cross != 0:
            raise ValueError("global_dim must be divisible by n_heads_cross")
        if (self.global_dim // self.n_heads_cross) % 2 != 0:
            raise ValueError("cross-attention head_dim must be even for rotary compatibility")
        if self.decoder_dim % self.n_heads_cross != 0:
            raise ValueError("decoder_dim must be divisible by n_heads_cross")
        if (self.decoder_dim // self.n_heads_cross) % 2 != 0:
            raise ValueError("decoder cross-attention head_dim must be even")

    @staticmethod
    def _validate_dim(dim: int, num_heads: int, name: str) -> None:
        if dim <= 0:
            raise ValueError(f"{name}_dim must be positive")
        if num_heads <= 0:
            raise ValueError(f"n_heads_{name} must be positive")
        if dim % num_heads != 0:
            raise ValueError(f"{name}_dim must be divisible by its attention head count")
        if (dim // num_heads) % 2 != 0:
            raise ValueError(f"{name} head_dim must be even for rotary embeddings")

    @property
    def vocab_size(self) -> int:
        base_vocab_size = self.offset + self.byte_vocab_size
        if self.pad_id >= 0:
            return max(base_vocab_size, self.pad_id + 1)
        return base_vocab_size

    def byte_to_token_id(self, value: int) -> int:
        if value < 0 or value >= self.byte_vocab_size:
            raise ValueError(f"byte value out of range: {value}")
        return value + self.offset
