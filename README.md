# bitnet

Experimental PyTorch implementation of a deep ternary BitNet-style language model optimized for weak hardware.

WARNING: THIS IS VERY EARLY STAGE ALPHA SOFTWARE.

## Current architecture

- Hidden size: `1024`
- Layers: `64`
- Attention heads: `32`
- Residual stack: `64` layers of block attention residuals
- Linear layers: `HBitLinear` with ternary weight quantization, optional Hadamard preprocessing, and 4-bit activation quantization
- Tokenizer: two-stage hierarchical byte-and-merge tokenizer inspired by dynamic grouping with hierarchical BPE
- Positional encoding: rotary embeddings with simple YaRN-style scaling support

## Repository layout

- `config.py`: model and training configuration
- `model.py`: main `BitNetDeep` model
- `layers/`: ternary linear, block attention residual, and Infini-Attention modules
- `tokenizer/`: hierarchical tokenizer implementation
- `train.py`: basic training skeleton using Hugging Face datasets
- `utils.py`: rotary and ternary helper functions

## Quick start

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a quick model smoke test:

```bash
python3 model.py
```

Run the training skeleton:

```bash
python3 train.py
```

## Notes

- The training loop is still a scaffold and not a full pretraining pipeline.
- The model is designed to be hackable and CPU-friendly first, with room for more optimized kernels later.
