"""Training skeleton for the deep ternary LLM.

Includes:
- Progressive block growth option (increase block_size during training)
- Basic HF datasets loader
- Low-memory optimizer settings suitable for Mac/Raspberry Pi
- Placeholder for ternary-specific training tricks (STE, bit flipping)
"""

from __future__ import annotations

import torch
from datasets import load_dataset
import torch.optim as optim

from model import BitNetDeep
from tokenizer.hierarchical_tokenizer import HierarchicalTokenizer
from config import config


def main() -> None:
    print("Initializing deep ternary LLM training...")

    tokenizer = HierarchicalTokenizer()
    model = BitNetDeep(config)

    # Use CPU-friendly settings (or MPS on Apple Silicon)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")

    # Small Hugging Face subset for quick smoke tests.
    dataset = load_dataset("roneneldan/TinyStories", split="train[:256]")
    print(f"Loaded dataset with {len(dataset)} examples")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Progressive block growth example (increase block_size every N steps)
    print("\nTraining ready. Progressive Block Growth and ternary STE can be added here.")
    print("Run with: python3 train.py")
    print("Current model has", sum(p.numel() for p in model.parameters()) // 1_000_000, "M parameters")

    sample_text = dataset[0]["text"]
    sample_ids = tokenizer.encode(sample_text, max_length=64)
    input_ids = torch.tensor([sample_ids], device=device, dtype=torch.long)
    logits = model(input_ids)
    print(f"Training skeleton test passed. Logits shape: {logits.shape}")


if __name__ == "__main__":
    main()
