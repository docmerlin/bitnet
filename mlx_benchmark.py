"""Compare representative BitNet training steps on MLX and PyTorch MPS."""

from __future__ import annotations

import argparse
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mlx", "torch"), required=True)
    parser.add_argument("--optimizer", choices=("adamw", "cmud"), default="adamw")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-prelude-layers", type=int, default=0)
    parser.add_argument("--num-coda-layers", type=int, default=0)
    parser.add_argument("--num-loops", type=int, default=1)
    parser.add_argument("--active-loops", type=int, default=None)
    parser.add_argument("--path-window-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlx-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--mlx-path-kernel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-recurrent-weights", action="store_true")
    parser.add_argument("--recurrent-quantized-matmul", action="store_true")
    parser.add_argument("--cmud-momentum-8bit", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1 or args.warmup_steps < 0:
        raise ValueError("steps must be positive and warmup-steps non-negative")
    if args.hidden_size % args.num_heads:
        raise ValueError("hidden-size must be divisible by num-heads")
    if args.num_loops < 1:
        raise ValueError("num-loops must be positive")
    if args.active_loops is not None and not 1 <= args.active_loops <= args.num_loops:
        raise ValueError("active-loops must be between one and num-loops")
    if min(args.num_prelude_layers, args.num_coda_layers) < 0:
        raise ValueError("prelude and coda layer counts must be non-negative")
    if args.path_window_size < 1 or args.sequence_length % args.path_window_size:
        raise ValueError("sequence-length must be divisible by path-window-size")
    if args.backend != "mlx" and args.optimizer != "adamw":
        raise ValueError("CMUD benchmark is only available with MLX")


def run_mlx(args: argparse.Namespace) -> dict[str, float]:
    from functools import partial

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from mlx_model import MLXBitNet, MLXBitNetConfig
    from mlx_optim import CMUD

    dtype = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}[args.mlx_dtype]
    mx.random.seed(1337)
    config = MLXBitNetConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        num_prelude_layers=args.num_layers if args.num_loops == 1 else args.num_prelude_layers,
        num_recurrent_layers=args.num_layers if args.num_loops > 1 else 0,
        num_coda_layers=args.num_coda_layers,
        num_loops=args.num_loops,
        block_size=1,
        path_window_size=args.path_window_size,
        use_path_kernel=args.mlx_path_kernel,
        use_engram=False,
        use_rfmoe=False,
    )
    model = MLXBitNet(
        config,
        reuse_recurrent_weights=args.reuse_recurrent_weights,
        recurrent_quantized_matmul=args.recurrent_quantized_matmul,
    )
    model.set_dtype(dtype)
    optimizer = (
        CMUD(
            mud_learning_rate=args.learning_rate,
            fallback_learning_rate=args.learning_rate,
            weight_decay=0.01,
            block_size=256,
            eight_bit=True,
            mud_eight_bit=args.cmud_momentum_8bit,
        )
        if args.optimizer == "cmud"
        else optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    )

    def loss_fn(inputs: mx.array, targets: mx.array, segment_ids: mx.array) -> mx.array:
        logits = model(inputs, segment_ids, num_loops=args.active_loops).astype(mx.float32)
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer.init(model.trainable_parameters())
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(inputs: mx.array, targets: mx.array, segment_ids: mx.array) -> mx.array:
        loss, gradients = loss_and_grad(inputs, targets, segment_ids)
        optimizer.update(model, gradients)
        return loss

    tokens = mx.random.randint(0, args.vocab_size, (args.batch_size, args.sequence_length + 1))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    segment_ids = mx.zeros(inputs.shape, dtype=mx.int32)
    for _ in range(args.warmup_steps):
        mx.eval(train_step(inputs, targets, segment_ids), state)
    start = time.perf_counter()
    loss = None
    for _ in range(args.steps):
        loss = train_step(inputs, targets, segment_ids)
        mx.eval(loss, state)
    elapsed = time.perf_counter() - start
    parameters = sum(value.size for _, value in tree_flatten(model.parameters()))
    return {
        "elapsed_sec": elapsed,
        "loss": float(loss.item()),
        "parameters": float(parameters),
        "steps_per_second": args.steps / elapsed,
        "tokens_per_second": args.steps * args.batch_size * args.sequence_length / elapsed,
    }


def run_torch(args: argparse.Namespace) -> dict[str, float]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from config import TernaryConfig
    from layers.hybrid_block import HybridTransformerBlock

    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is unavailable")
    device = torch.device("mps")
    config = TernaryConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        head_dim=args.hidden_size // args.num_heads,
        intermediate_size=args.intermediate_size,
        block_size=1,
        path_window_size=args.path_window_size,
        use_engram=False,
        use_rfmoe=False,
        num_loops=1,
    )

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(args.vocab_size, args.hidden_size)
            self.blocks = nn.ModuleList(HybridTransformerBlock(config, index) for index in range(args.num_layers))
            self.norm = nn.RMSNorm(args.hidden_size)

        def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            x = self.embedding(tokens)
            for _ in range(args.active_loops or args.num_loops):
                for block in self.blocks:
                    x = block(x, mask, input_ids=tokens, update_memory=False)
            return F.linear(self.norm(x), self.embedding.weight)

    torch.manual_seed(1337)
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    tokens = torch.randint(
        args.vocab_size,
        (args.batch_size, args.sequence_length + 1),
        device=device,
    )
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    mask = torch.ones(
        args.batch_size,
        args.sequence_length,
        args.sequence_length,
        dtype=torch.bool,
        device=device,
    )

    def train_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, mask)
        loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        return loss

    for _ in range(args.warmup_steps):
        train_step()
    torch.mps.synchronize()
    start = time.perf_counter()
    loss = None
    for _ in range(args.steps):
        loss = train_step()
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "elapsed_sec": elapsed,
        "loss": float(loss.detach()),
        "parameters": float(parameters),
        "steps_per_second": args.steps / elapsed,
        "tokens_per_second": args.steps * args.batch_size * args.sequence_length / elapsed,
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    metrics = run_mlx(args) if args.backend == "mlx" else run_torch(args)
    values = " | ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    print(f"backend={args.backend} | {values}")


if __name__ == "__main__":
    main()
