"""Hyperloop-style hyper-connections at recurrent loop boundaries.

Follows Hyperloop Transformers (Zeitoun et al., arXiv:2604.21254):

- Expand residual into ``n=4`` parallel streams (hardcoded)
- Input-dependent ``H_pre`` / ``H_post`` (sigmoid; post scaled by 2)
- **Diagonal** ``H_res = diag(σ(·))`` — not Sinkhorn / dense doubly-stochastic mHC
- Loop position embedding ``e_l`` added after the middle-block output
- Applied only between / after recurrent loop iterations, not every layer

Update per loop::

    z = RMSNorm(flatten(Y))
    H_pre  = σ(α_pre  · (W_pre  z) + b_pre)     # (n,)
    H_post = 2 · σ(α_post · (W_post z) + b_post) # (n,)
    H_res  = diag(σ(α_res · (W_res z) + b_res))  # diagonal only
    x      = H_pre · Y                           # collapse streams → C
    u      = F(x) + e_l                          # middle stack + loop embed
    Y      = H_res · Y + H_post ⊗ u              # mix + write back

After all loops, average streams for the coda.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Hardcoded Hyperloop defaults (not config knobs).
NUM_STREAMS = 4
MAX_LOOP_EMBEDS = 64


class LoopHyperConnection(nn.Module):
    """Loop-boundary hyper-connections (Hyperloop / simplified mHC)."""

    def __init__(self, hidden_size: int, rms_norm_eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_streams = NUM_STREAMS
        n = NUM_STREAMS
        flat = n * hidden_size

        self.norm = nn.RMSNorm(flat, eps=rms_norm_eps)

        # Input-dependent projections on flattened multi-stream residual.
        self.w_pre = nn.Linear(flat, n, bias=True)
        self.w_post = nn.Linear(flat, n, bias=True)
        self.w_res = nn.Linear(flat, n, bias=True)
        self.alpha_pre = nn.Parameter(torch.ones(1))
        self.alpha_post = nn.Parameter(torch.ones(1))
        self.alpha_res = nn.Parameter(torch.ones(1))

        # Loop position embedding e_l ∈ R^C (depth-wise RNN "input").
        self.loop_embed = nn.Embedding(MAX_LOOP_EMBEDS, hidden_size)

        self._init_identity_friendly()

    def _init_identity_friendly(self) -> None:
        """Bias toward average-read / unit residual / small write at step 0."""
        n = self.num_streams
        with torch.no_grad():
            # H_pre ≈ 1/n via equal logits after sigmoid → soft average of streams.
            self.w_pre.weight.zero_()
            self.w_pre.bias.fill_(0.0)
            # H_post = 2·σ(·): bias so σ≈0 → post≈0 (no write at init).
            self.w_post.weight.zero_()
            self.w_post.bias.fill_(-3.0)
            # H_res diagonal ≈ 1: σ large → identity residual on each stream.
            self.w_res.weight.zero_()
            self.w_res.bias.fill_(3.0)
            self.loop_embed.weight.zero_()

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, C) → (B, T, n, C)`` by copying the residual into all streams."""
        if x.ndim != 3 or x.size(-1) != self.hidden_size:
            raise ValueError(
                f"expected x shape (B, T, {self.hidden_size}), got {tuple(x.shape)}"
            )
        return x.unsqueeze(2).expand(-1, -1, self.num_streams, -1).contiguous()

    def fold(self, y: torch.Tensor) -> torch.Tensor:
        """Average parallel streams → ``(B, T, C)`` for the coda / non-loop path."""
        return y.mean(dim=2)

    def project_in(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute input-dependent gates and collapse streams for the middle block.

        Returns
        -------
        x_in : (B, T, C)
        h_pre, h_post, h_res : (B, T, n)
        """
        bsz, seq_len, n, width = y.shape
        if n != self.num_streams or width != self.hidden_size:
            raise ValueError(f"bad stream tensor shape {tuple(y.shape)}")

        z = self.norm(y.reshape(bsz, seq_len, n * width))
        h_pre = torch.sigmoid(self.alpha_pre * self.w_pre(z))
        h_post = 2.0 * torch.sigmoid(self.alpha_post * self.w_post(z))
        h_res = torch.sigmoid(self.alpha_res * self.w_res(z))  # diagonal entries

        # H_pre · Y : weighted sum over streams.
        x_in = (h_pre.unsqueeze(-1) * y).sum(dim=2)
        return x_in, h_pre, h_post, h_res

    def write_back(
        self,
        y: torch.Tensor,
        u: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
    ) -> torch.Tensor:
        """``Y ← diag(H_res) Y + H_post ⊗ u``."""
        # h_res, h_post: (B, T, n); y: (B, T, n, C); u: (B, T, C)
        return h_res.unsqueeze(-1) * y + h_post.unsqueeze(-1) * u.unsqueeze(2)

    def loop_embedding(self, loop_idx: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        idx = min(max(loop_idx, 0), MAX_LOOP_EMBEDS - 1)
        emb = self.loop_embed.weight[idx]
        return emb.to(device=device, dtype=dtype)
