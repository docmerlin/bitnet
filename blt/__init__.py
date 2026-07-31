"""Separate Byte Latent Transformer package.

This package intentionally lives alongside the existing BitNet stack rather than
sharing its model and training entrypoints. It supports both training from
scratch on raw bytes and teacher-forced distillation from Meta BLT.

Deliberately thin. Re-exporting a symbol whose name matches a submodule shadows
that submodule -- exporting ``generate`` made ``blt.generate`` resolve to the
function, so ``import blt.generate as module`` silently handed back a callable.
Import from the submodules instead::

    from blt.generate import generate            # torch
    from blt.mlx_generate import generate        # MLX
    from blt.mlx_train import MLXBLTTrainer
"""

from blt.config import TernaryBLTConfig
from blt.model import TernaryBLTModel, TernaryBLTOutput

__all__ = [
    "TernaryBLTConfig",
    "TernaryBLTModel",
    "TernaryBLTOutput",
]
