"""Configuration for the deep ternary (1.58-bit) LLM based on BitNet b1.58."""
from dataclasses import dataclass
from typing import Optional, Tuple


def effective_path_window(
    *, path_window_size: int, block_size: int, sequence_length: int
) -> int:
    """Chunk width PaTH local attention actually uses.

    ``block_size`` partitions the sequence first and ``path_window_size`` only
    caps each block, so a window wider than ``sequence_length / block_size``
    never binds. At seq 1024 with 16 blocks every window >= 64 is identical.
    """
    block_width = (sequence_length + block_size - 1) // block_size
    return min(int(path_window_size), block_width)


def _nearest_odd_table_size(target: int) -> int:
    """Pick an odd table size near ``target`` (hash-friendly; avoid tiny tables)."""
    n = max(17, int(target) | 1)  # odd, at least 17
    return n


def estimate_engram_params_per_layer(
    *,
    hidden_size: int,
    table_size: int,
    max_ngram_size: int,
    num_heads: int,
    head_dim: int,
    kernel_size: int,
) -> int:
    """Params for one Engram module (matches ``layers/engram.py``)."""
    num_tables = max(0, int(max_ngram_size) - 1) * int(num_heads)
    emb = num_tables * int(table_size) * int(head_dim)
    mem = num_tables * int(head_dim)
    projs = 2 * mem * int(hidden_size)  # key + value
    conv = int(hidden_size) * int(kernel_size)
    norms = 3 * int(hidden_size)
    return emb + projs + conv + norms


@dataclass
class TernaryConfig:
    """Configuration for BitNetDeep model.

    Key design choices for deep ternary stability and efficiency on weak hardware:
    - Hidden size 1024 (power of 2, friendly for Hadamard transform)
    - Looped depth: unique prelude + recurrent core × R + coda (default 8+32×R+8)
    - RMSNorm + SubLN (extra sub-layer norm for ternary weight stability)
    - PaTH-FoX data-dependent positions inside bounded local windows
    - EVERY layer uses BOTH Infini-Attention and residual path (Kimi Block AttnRes default)
    - Selected layers add DeepSeek Engram-style conditional N-gram memory
    - Hierarchical tokenizer targets a 128k-token byte-and-merge vocabulary
    - Always: tied embeddings, full-precision lm_head, per-head QK norm

    Layer structure (resolved in ``__post_init__``):
    - Production default: prelude=8, recurrent=32, coda=8, num_loops=4
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
    initializer_range: float = 0.02

    # Hybrid block parameters (every layer: Infini/PaTH + residual path)
    block_size: int = 8          # Infini local sequence blocks (progressive growth); NOT AttnRes group size
    path_window_size: int = 1024  # PaTH-FoX local window; 1024 won quality A/B at free wall
    # Paper Infini (arXiv:2404.07143): associative M is head_dim×head_dim per head.
    # infini_memory_dim is legacy/unused (kept for CLI + old configs).
    infini_memory_dim: int = 64
    # Key-feature width of the Infini memory. M holds infini_memory_expand * head_dim
    # numbers per head, so this is the only knob that changes how much the memory can
    # actually hold; 0 keeps the paper's head_dim. Useful ceiling is the number of
    # tokens between memory resets (one sequence in training) — past that you are
    # paying for capacity the horizon never fills.
    infini_memory_expand: int = 0
    # Feature map phi() behind the associative memory. "elu" is the paper's ELU+1;
    # its near-constant baseline makes phi(q).phi(k) nearly uniform, so every query
    # reads back roughly the same vector no matter how large the memory is. "favor"
    # uses the exp kernel (Performer FAVOR+), which approximates softmax attention
    # and makes reads content-dependent. Needs infini_memory_expand > 0 to have
    # random features to project onto.
    infini_feature_map: str = "elu"
    infini_delta_rule: bool = True  # Linear+Delta memory update; False = plain Linear
    # Optional third attention branch: block-granular top-k retrieval over past
    # tokens, alongside the local PaTH window and the Infini memory. Training only
    # (decode caches keep no full history). Off by default; A/B with --topk-blocks.
    use_topk_blocks: bool = False
    topk_blocks: int = 4        # past blocks retrieved per query chunk
    topk_block_size: int = 64   # tokens per retrievable block
    # Hybrid depth: every ``mamba_layer_period``-th unique layer (starting at 0) uses a
    # Mamba-3-style selective SSM mixer instead of PaTH+Infini. period=3 → ~1/3 layers.
    # Off by default: measured on the BLT + BitNet stack, the SSM layers cost 1.36-1.51x
    # end-to-end throughput against the PaTH attention layers that replace them, and no
    # quality comparison has been run in either direction. Turn on to A/B.
    use_mamba3_layers: bool = False
    mamba_layer_period: int = 3
    mamba_d_state: int = 64
    mamba_expand: int = 2
    mamba_headdim: int = 32
    mamba_d_conv: int = 4
    mamba_dt_min: float = 0.001
    mamba_dt_max: float = 0.1
    mamba_a_floor: float = 1e-4
    # Residual path: "kimi" = Block AttnRes (arXiv:2603.15031); "sandwich" = legacy scalar residual
    attn_res_mode: str = "kimi"
    # Transformer layers per AttnRes depth-block (None → max(1, unique_layers // 8)).
    attn_res_group_size: Optional[int] = None
    attn_res_init_scale: float = 0.1  # Sandwich mode residual scale init only

    # DeepSeek Engram conditional N-gram memory.
    # Default table size is derived so Engram is ~engram_param_fraction of the body
    # (unique blocks + embeddings), not a fixed 4k toy table.
    use_engram: bool = True
    engram_layer_ids: Tuple[int, ...] = (1, 15)
    engram_vocab_size: Optional[int] = None  # None → size from engram_param_fraction
    engram_param_fraction: float = 0.05      # target Engram / body params; 0 = require explicit vocab
    engram_max_ngram_size: int = 3      # Bigram + trigram tables
    engram_num_heads: int = 4
    engram_head_dim: int = 16
    engram_kernel_size: int = 4
    engram_pad_id: int = 257
    engram_seed: int = 0

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

    # Share the output projection with the input embedding. False (untied) is the
    # modded-nanogpt configuration: an untied head plus a much higher embedding
    # learning rate was their largest win after Muon. Costs vocab*hidden extra
    # parameters (16.7M at vocab 32768 / hidden 512).
    tie_word_embeddings: bool = False

    # Looped / recurrent-depth structure. None = resolve in __post_init__.
    # Effective depth = prelude + recurrent * num_loops + coda.
    # Loop-boundary hyper-connections (Hyperloop-style, 4 streams, diagonal H_res)
    # are always on when the recurrent core runs — not config knobs.
    num_prelude_layers: Optional[int] = None
    num_recurrent_layers: Optional[int] = None
    num_coda_layers: Optional[int] = None
    num_loops: Optional[int] = None

    def __post_init__(self):
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.path_window_size < 1:
            raise ValueError("path_window_size must be positive")
        if int(self.infini_memory_expand) < 0:
            raise ValueError("infini_memory_expand must be non-negative (0 = head_dim)")
        if self.infini_feature_map not in ("elu", "favor"):
            raise ValueError("infini_feature_map must be 'elu' or 'favor'")
        self.use_topk_blocks = bool(self.use_topk_blocks)
        if min(int(self.topk_blocks), int(self.topk_block_size)) < 1:
            raise ValueError("topk_blocks and topk_block_size must be positive")
        if int(self.mamba_layer_period) < 1:
            raise ValueError("mamba_layer_period must be >= 1")
        self.mamba_layer_period = int(self.mamba_layer_period)
        self.use_mamba3_layers = bool(self.use_mamba3_layers)
        if int(self.mamba_d_state) < 1 or int(self.mamba_expand) < 1:
            raise ValueError("mamba_d_state and mamba_expand must be positive")
        if int(self.mamba_headdim) < 1 or int(self.mamba_d_conv) < 1:
            raise ValueError("mamba_headdim and mamba_d_conv must be positive")
        if self.engram_max_ngram_size < 2:
            raise ValueError("engram_max_ngram_size must be >= 2")
        if self.engram_num_heads < 1 or self.engram_head_dim < 1:
            raise ValueError("Engram head count and dimension must be positive")
        if self.engram_kernel_size < 1:
            raise ValueError("engram_kernel_size must be positive")
        if not 0.0 <= float(self.engram_param_fraction) <= 1.0:
            raise ValueError("engram_param_fraction must be in [0, 1]")
        self.engram_layer_ids = tuple(int(layer_id) for layer_id in self.engram_layer_ids)
        if len(set(self.engram_layer_ids)) != len(self.engram_layer_ids) or any(
            layer_id < 0 for layer_id in self.engram_layer_ids
        ):
            raise ValueError("engram_layer_ids must contain unique non-negative IDs")
        mode = str(self.attn_res_mode).lower()
        if mode not in {"kimi", "sandwich"}:
            raise ValueError("attn_res_mode must be 'kimi' or 'sandwich'")
        self.attn_res_mode = mode

        self._resolve_layer_structure()
        if self.attn_res_group_size is None:
            self.attn_res_group_size = max(1, int(self.num_hidden_layers) // 8)
        elif int(self.attn_res_group_size) < 1:
            raise ValueError("attn_res_group_size must be >= 1")
        else:
            self.attn_res_group_size = int(self.attn_res_group_size)

        # Drop Engram injects past the unique stack (common when shrinking L for tests).
        L = int(self.num_hidden_layers)
        valid_ids = tuple(i for i in self.engram_layer_ids if i < L)
        if self.use_engram and not valid_ids:
            # Prefer early + mid of the actual stack.
            if L == 1:
                valid_ids = (0,)
            else:
                valid_ids = (min(1, L - 1), L // 2)
                valid_ids = tuple(dict.fromkeys(valid_ids))  # unique, order-preserving
        self.engram_layer_ids = valid_ids

        self._resolve_engram_vocab_size()

    def estimate_body_params(self) -> int:
        """Rough unique-parameter count excluding Engram (for table auto-size)."""
        H = int(self.hidden_size)
        I = int(self.intermediate_size)
        L = int(self.num_hidden_layers)
        # Tied embed + lm_head share weights.
        emb = int(self.vocab_size) * H
        # Per hybrid block: attn ≈ 4H² + path/infini extras ≈ 6H²; dense 3-mat FFN.
        ffn = H * (2 * I) + I * I + I * H
        attn = 6 * H * H
        norms = 10 * H
        # Kimi AttnRes: two Linear(d→1) + two RMSNorms per layer ≈ 2H + 2H.
        attn_res = 4 * H if self.attn_res_mode == "kimi" else 2 * H
        per_layer = attn + ffn + norms + attn_res
        # Loop HC is small (O(n*H) with n=4).
        loop_hc = 4 * 4 * H + 4 * H
        return emb + L * per_layer + loop_hc

    def estimate_engram_params(self, table_size: Optional[int] = None) -> int:
        """Total Engram params for configured inject layers and table size."""
        if not self.use_engram or not self.engram_layer_ids:
            return 0
        v = int(self.engram_vocab_size if table_size is None else table_size)
        per = estimate_engram_params_per_layer(
            hidden_size=int(self.hidden_size),
            table_size=v,
            max_ngram_size=int(self.engram_max_ngram_size),
            num_heads=int(self.engram_num_heads),
            head_dim=int(self.engram_head_dim),
            kernel_size=int(self.engram_kernel_size),
        )
        return per * len(self.engram_layer_ids)

    def _resolve_engram_vocab_size(self) -> None:
        """Set ``engram_vocab_size`` from fraction of body params when unset."""
        if self.engram_vocab_size is not None:
            if int(self.engram_vocab_size) < 1:
                raise ValueError("engram_vocab_size must be positive")
            self.engram_vocab_size = int(self.engram_vocab_size)
            return
        if not self.use_engram or not self.engram_layer_ids:
            self.engram_vocab_size = 4093  # unused placeholder
            return
        frac = float(self.engram_param_fraction)
        if frac <= 0.0:
            raise ValueError(
                "engram_vocab_size is None; set it explicitly or set engram_param_fraction > 0"
            )
        body = max(1, self.estimate_body_params())
        target_engram = frac * body
        n_layers = len(self.engram_layer_ids)
        num_tables = max(1, (int(self.engram_max_ngram_size) - 1) * int(self.engram_num_heads))
        hd = int(self.engram_head_dim)
        H = int(self.hidden_size)
        # per layer ≈ num_tables*V*hd + 2*num_tables*hd*H + small
        fixed_per = 2 * num_tables * hd * H + H * int(self.engram_kernel_size) + 3 * H
        # target_engram = n_layers * (num_tables*hd*V + fixed_per)
        budget_per = max(0.0, target_engram / n_layers - fixed_per)
        denom = num_tables * hd
        raw_v = int(budget_per / denom) if denom > 0 else 17
        self.engram_vocab_size = _nearest_odd_table_size(raw_v)

    def _resolve_layer_structure(self) -> None:
        """Resolve prelude / recurrent / coda / loops and unique layer count.

        Rules:
        1. Any structure field set → structure mode (missing pieces default to 0).
           Default loops=4 when unset.
        2. Only ``num_hidden_layers`` set → flat stack (0, N, 0), default loops=1
           (compat for tests and old checkpoints).
        3. Nothing set → production 8 / 32 / 8 with loops=4.
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
            p, r, c = 8, 32, 8
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
