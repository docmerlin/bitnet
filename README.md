# bitnet

Experimental PyTorch implementation of a deep ternary BitNet-style language model optimized for weak hardware.

WARNING: THIS IS VERY EARLY STAGE ALPHA SOFTWARE.

## Current architecture (after latest improvements)

- Hidden size: `1024`
- Layers: `64`
- Heads: `32` (head dim = 32)
- These are the default full-training settings used by `run_train.sh`; `run_local_train.sh` uses a smaller local profile for weaker hardware.
- **Unified hybrid block in every layer**: Every Transformer block contains BOTH Infini-Attention (local + compressive memory with per-head gating) AND Attention Residuals (AttnRes) around both attention and MLP sublayers. No layer splitting.
- All linear projections use `H-BitLinear` (Hadamard transform on input + ternary weights + 4-bit activations with STE)
- Tokenizer: two-stage hierarchical tokenizer with end-of-patch markers and learned second-stage merges over first-stage token bytes
- RMSNorm + extra SubLN for ternary stability
- RoPE with YaRN scaling
- Two-stage quantization schedule and progressive block growth during training

## Repository layout

- `config.py`: model and training configuration
- `model.py`: main `BitNetDeep` model
- `layers/hybrid_block.py`: main transformer block combining Infini-Attention and AttnRes in every layer
- `layers/`: ternary linear, block attention residual, Infini-Attention, and supporting modules
- `tokenizer/`: hierarchical tokenizer implementation
- `train.py`: streaming training pipeline using Hugging Face datasets
- `utils.py`: rotary and ternary helper functions

## Quick start

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependency notes:

- `test_forward.py` requires `tiktoken`
- `train.py` requires both `tiktoken` and `datasets`

Run a quick model smoke test:

```bash
python3 test_forward.py
```

This exercises the hybrid model stack, hierarchical tokenizer, and H-BitLinear path.

Run the training entrypoint:

```bash
# Local test (smaller model, fast on Mac Mini)
./run_local_train.sh

# Full model (64 layers, requires GPU with good VRAM)
./run_train.sh
```

## Programming data presets

`train.py` now includes built-in streaming presets for programming data via
CodeSearchNet language shards:

- `code_search_net_all`
- `code_search_net_python`
- `code_search_net_go`
- `code_search_net_javascript`
- `code_search_net_java`
- `code_search_net_php`
- `code_search_net_ruby`

Recommended default for broad coding ability: `code_search_net_all`.

It expands to all available CodeSearchNet languages with a weighted split:

- Python: 30%
- JavaScript: 22%
- Java: 18%
- Go: 15%
- PHP: 8%
- Ruby: 7%

Example code-heavy mixture:

```bash
python3 train.py \
  --train-mixture fineweb_edu=0.55,dclm=0.25,code_search_net_all=0.20 \
  --val-mixture fineweb_edu=0.5,code_search_net_all=0.5
```

If you want to use a gated programming corpus such as StarCoderData, `train.py`
also supports custom entries in the form:

```bash
path|config|split|text_field=weight
```

For example:

```bash
bigcode/starcoderdata|python|train|content=0.2
```

That requires an authenticated Hugging Face token in your environment.

## Math data presets

`train.py` now includes built-in streaming presets for math-heavy pretraining
data:

- `finemath_3plus`
- `open_web_math`

Recommended default: `finemath_3plus` for cleaner math-focused web text, with
`open_web_math` available when you want broader math coverage.

Example math-heavy mixture:

```bash
python3 train.py \
  --train-mixture fineweb_edu=0.45,dclm=0.20,code_search_net_python=0.10,finemath_3plus=0.25 \
  --val-mixture fineweb_edu=0.30,code_search_net_python=0.20,finemath_3plus=0.50
```

The default full-hardware launcher `run_train.sh` now includes modest Python
code and math ratios:

```bash
./run_train.sh
```

## Early/Late data curriculum

`train.py` can now switch training mixtures partway through a run. Use:

- `--early-train-mixture`
- `--late-train-mixture`
- `--mixture-switch-ratio`

If these are omitted, the trainer falls back to the single `--train-mixture`
for the whole run.

Example curriculum:

```bash
python3 train.py \
  --early-train-mixture fineweb_edu=0.60,dclm=0.25,code_search_net_all=0.10,finemath_3plus=0.05 \
  --late-train-mixture fineweb_edu=0.35,dclm=0.15,code_search_net_all=0.20,finemath_3plus=0.30 \
  --mixture-switch-ratio 0.70 \
  --val-mixture fineweb_edu=0.35,dclm=0.15,code_search_net_all=0.20,finemath_3plus=0.30
```

## Notes

- The model is designed to be hackable and CPU-friendly first, with room for more optimized kernels later.
