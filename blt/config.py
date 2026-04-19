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

    n_layers_local_encoder: int = 4
    n_layers_global: int = 8
    n_layers_local_decoder: int = 4

    n_heads_local_encoder: int = 4
    n_heads_global: int = 8
    n_heads_local_decoder: int = 4
    n_heads_cross: int = 4

    ffn_multiplier_local: float = 4.0
    ffn_multiplier_global: float = 4.0
    ffn_multiplier_decoder: float = 4.0

    local_window: int | None = 256
    dropout: float = 0.0
    rope_theta: float = 10000.0

    patch_size: int = 6
    max_patch_length: int = 32

    use_hadamard: bool = True
    use_4bit_activations: bool = True

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
