"""Infini-Attention compressive-memory helpers."""

from __future__ import annotations

from typing import Dict, Iterator, List, Sequence

import torch
import torch.nn as nn

from layers.infini_attention import InfiniAttention


def iter_infini_attention_modules(module: nn.Module) -> Iterator[InfiniAttention]:
    for submodule in module.modules():
        if isinstance(submodule, InfiniAttention):
            yield submodule


def capture_infini_memory_state(module: nn.Module) -> List[Dict[str, torch.Tensor]]:
    return [submodule.get_memory_state() for submodule in iter_infini_attention_modules(module)]


def restore_infini_memory_state(
    module: nn.Module,
    state: Sequence[Dict[str, torch.Tensor]],
) -> None:
    infini_modules = list(iter_infini_attention_modules(module))
    if len(infini_modules) != len(state):
        raise ValueError("InfiniAttention state does not match the current model layout")
    for submodule, memory_state in zip(infini_modules, state):
        submodule.load_memory_state(memory_state)


def reset_infini_memory(module: nn.Module) -> None:
    for submodule in iter_infini_attention_modules(module):
        submodule.reset_memory()
