"""Configuration for the deep ternary (1.58-bit) LLM based on BitNet b1.58."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TernaryConfig:
    """Configuration for BitNetDeep model.

    Key design choices for deep ternary stability and efficiency on weak hardware:
    - Hidden size 1024 (power of 2, friendly for Hadamard transform)
    - 64 layers (scalable to 128-192)
    - RMSNorm + SubLN (extra sub-layer norm for ternary weight stability)
    - RoPE with YaRN/NTK scaling for long context
    - EVERY layer uses BOTH Infini-Attention and Attention Residuals (AttnRes)
    - Hierarchical tokenizer targets a 128k-token byte-and-merge vocabulary
    """
    vocab_size: int = 131072  # ~128k target (first-stage ~100k + hierarchical merges)
    hidden_size: int = 1024
    num_hidden_layers: int = 64
    num_attention_heads: int = 32
    head_dim: int = 32  # hidden_size // num_attention_heads
    intermediate_size: int = 2048  # 2x hidden for FFN (standard for BitNet)
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: dict = None  # YaRN/NTK params set in __post_init__
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02

    # Hybrid block parameters (every layer now uses both Infini-Attention + AttnRes)
    block_size: int = 8          # Number of local blocks for attention residual (supports progressive growth)
    infini_memory_dim: int = 64  # Compressive memory dimension per head for Infini-Attention
    attn_res_init_scale: float = 0.1  # Initial scale for Attention Residual connections

    # Ternary training / inference
    use_hadamard: bool = True
    use_4bit_activations: bool = True
    ternary_weight_bits: int = 2  # 1.58-bit effective (-1,0,1)

    def __post_init__(self):
        if self.rope_scaling is None:
            self.rope_scaling = {
                "type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 4096,
            }
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")

config = TernaryConfig()
