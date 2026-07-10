"""Configuration for the deep ternary (1.58-bit) LLM based on BitNet b1.58."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TernaryConfig:
    """Configuration for BitNetDeep model.

    Key design choices for deep ternary stability and efficiency on weak hardware:
    - Hidden size 1024 (power of 2, friendly for Hadamard transform)
    - Looped depth: unique prelude + recurrent core × R + coda (default 8+48×R+8)
    - RMSNorm + SubLN (extra sub-layer norm for ternary weight stability)
    - RoPE with YaRN/NTK scaling for long context
    - EVERY layer uses BOTH Infini-Attention and sandwich RMSNorm AttnRes residuals
    - Hierarchical tokenizer targets a 128k-token byte-and-merge vocabulary
    - Always: tied embeddings, full-precision lm_head, per-head QK norm

    Layer structure (resolved in ``__post_init__``):
    - Production default: prelude=8, recurrent=48, coda=8, num_loops=4
    - Flat / test: pass only ``num_hidden_layers=N`` → (0, N, 0) with num_loops=1
    - Explicit structure: pass any of prelude/recurrent/coda; sum becomes unique count
    """
    vocab_size: int = 131072  # ~128k target (first-stage ~100k + hierarchical merges)
    hidden_size: int = 1024
    # Unique block count after post_init (prelude + recurrent + coda). None = derive.
    num_hidden_layers: Optional[int] = None
    num_attention_heads: int = 32
    head_dim: int = 32  # hidden_size // num_attention_heads
    # 2x hidden. Dense FFN is SwiGLU expand + square mid + down (3 stages).
    # Mid alone adds ~intermediate_size^2 params (and FLOPs) per block — large jump
    # vs the classic 2-mat FFN; power-of-two intermediate also enables Hadamard on mid.
    intermediate_size: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: dict = None  # YaRN/NTK params set in __post_init__
    initializer_range: float = 0.02

    # Hybrid block parameters (every layer now uses both Infini-Attention + AttnRes)
    block_size: int = 8          # Number of local blocks for attention residual (supports progressive growth)
    infini_memory_dim: int = 64  # Compressive memory dimension per head for Infini-Attention
    attn_res_init_scale: float = 0.1  # Initial scale for Attention Residual connections

    # Ternary training / inference
    use_hadamard: bool = True
    use_4bit_activations: bool = True

    # Routing-free MoE FFN (off by default -> dense GLU FFN)
    use_rfmoe: bool = False
    rfmoe_num_experts: int = 8
    rfmoe_expert_dim: Optional[int] = None  # None -> intermediate_size // 4
    rfmoe_rank: Optional[int] = None        # None -> hidden_size // 16
    rfmoe_theta: float = 0.01               # fire threshold / compute knob

    # Multi-token prediction (data-efficiency). 0 = off (plain next-token).
    mtp_depth: int = 0

    # Looped / recurrent-depth structure. None = resolve in __post_init__.
    # Effective depth = prelude + recurrent * num_loops + coda.
    # Loop-boundary hyper-connections (Hyperloop-style, 4 streams, diagonal H_res)
    # are always on when the recurrent core runs — not config knobs.
    num_prelude_layers: Optional[int] = None
    num_recurrent_layers: Optional[int] = None
    num_coda_layers: Optional[int] = None
    num_loops: Optional[int] = None

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

        self._resolve_layer_structure()

    def _resolve_layer_structure(self) -> None:
        """Resolve prelude / recurrent / coda / loops and unique layer count.

        Rules:
        1. Any structure field set → structure mode (missing pieces default to 0).
           Default loops=4 when unset.
        2. Only ``num_hidden_layers`` set → flat stack (0, N, 0), default loops=1
           (compat for tests and old checkpoints).
        3. Nothing set → production 8 / 48 / 8 with loops=4.
        """
        structure_given = (
            self.num_prelude_layers is not None
            or self.num_recurrent_layers is not None
            or self.num_coda_layers is not None
        )

        if structure_given:
            p = 0 if self.num_prelude_layers is None else int(self.num_prelude_layers)
            r = 0 if self.num_recurrent_layers is None else int(self.num_recurrent_layers)
            c = 0 if self.num_coda_layers is None else int(self.num_coda_layers)
            loops = 4 if self.num_loops is None else int(self.num_loops)
        elif self.num_hidden_layers is not None:
            # Flat unique stack: all layers are the "recurrent" region; R=1 by default.
            p, r, c = 0, int(self.num_hidden_layers), 0
            loops = 1 if self.num_loops is None else int(self.num_loops)
        else:
            p, r, c = 8, 48, 8
            loops = 4 if self.num_loops is None else int(self.num_loops)

        if p < 0 or r < 0 or c < 0:
            raise ValueError("layer counts must be non-negative")
        if p + r + c < 1:
            raise ValueError("need at least one unique layer")
        if loops < 1:
            raise ValueError("num_loops must be >= 1")
        if loops > 1 and r == 0:
            raise ValueError("num_loops > 1 requires num_recurrent_layers >= 1")

        if self.num_hidden_layers is not None and structure_given:
            expected = p + r + c
            if int(self.num_hidden_layers) != expected:
                raise ValueError(
                    f"num_hidden_layers ({self.num_hidden_layers}) must equal "
                    f"prelude+recurrent+coda ({expected})"
                )

        self.num_prelude_layers = p
        self.num_recurrent_layers = r
        self.num_coda_layers = c
        self.num_loops = loops
        self.num_hidden_layers = p + r + c

    @property
    def effective_depth(self) -> int:
        """Layer applications per forward: prelude + recurrent * R + coda."""
        return (
            int(self.num_prelude_layers)
            + int(self.num_recurrent_layers) * int(self.num_loops)
            + int(self.num_coda_layers)
        )
