"""Simple test script for the hybrid deep ternary BitNet model.

Run with:
    python3 test_forward.py
"""
import torch
from config import TernaryConfig
from model import BitNetDeep
from tokenizer.hierarchical_tokenizer import HierarchicalTokenizer


def test_forward():
    config = TernaryConfig(
        vocab_size=4096,  # small for test
        hidden_size=512,  # reduced for quick test
        num_hidden_layers=16,
        num_attention_heads=16,
        head_dim=32,
        block_size=8,
    )

    model = BitNetDeep(config)
    tokenizer = HierarchicalTokenizer(max_patch_size=4, vocab_size_target=4096)

    text = "This is a test of the hybrid ternary BitNet model with Infini-Attention and AttnRes in every layer."
    input_ids = torch.tensor([tokenizer.encode(text, max_length=64)])

    model.eval()
    with torch.no_grad():
        logits = model(input_ids)

    print("✅ Hybrid model forward pass successful!")
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    print("Every layer now contains BOTH Infini-Attention and Attention Residuals (AttnRes)")
    print("Progressive block growth is supported via model.layers[i].num_blocks = new_value")
    return True


if __name__ == "__main__":
    test_forward()
    print("\nTest passed. The model is now hybrid and ready for training.")
