"""Separate Byte Latent Transformer package.

This package intentionally lives alongside the existing BitNet stack rather than
sharing its model and training entrypoints. The first implementation focuses on
teacher-forced distillation from Meta BLT into a ternary student.
"""

from blt.config import TernaryBLTConfig
from blt.model import TernaryBLTModel, TernaryBLTOutput
from blt.train_distill import BLTDistillationBatch, BLTDistillationTrainer

__all__ = [
    "BLTDistillationBatch",
    "BLTDistillationTrainer",
    "TernaryBLTConfig",
    "TernaryBLTModel",
    "TernaryBLTOutput",
]
