"""Regression tests for document-boundary masking in packed training windows."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import (
    causal_block_attention_bias,
    combine_attention_bias,
    document_attention_keep_mask,
)
from train import PackedSequenceStream


def test_document_keep_mask_is_block_diagonal() -> bool:
    segment_ids = torch.tensor([[0, 0, 1, 1]])
    keep = document_attention_keep_mask(segment_ids)
    expected = torch.tensor(
        [[[True, True, False, False],
          [True, True, False, False],
          [False, False, True, True],
          [False, False, True, True]]]
    )
    assert torch.equal(keep, expected), keep
    print("document keep-mask block-diagonal tests passed")
    return True


def test_keep_mask_folds_into_causal_bias() -> bool:
    # Two packed documents [0,0,1,1] under a pure causal base bias. A token may
    # attend only to same-document, non-future positions.
    segment_ids = torch.tensor([[0, 0, 1, 1]])
    keep = document_attention_keep_mask(segment_ids)
    base = causal_block_attention_bias(4, 1, dtype=torch.float32, device=torch.device("cpu"))
    attn_bias, query_valid = combine_attention_bias(
        keep,
        base_bias=base,
        batch_size=1,
        q_len=4,
        k_len=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    floor = torch.finfo(torch.float32).min
    # Position 2 (doc 1) is blocked from doc-0 keys, allowed on itself, blocked on the future.
    assert attn_bias[0, 0, 2, 0] == floor, "cross-document attention leaked"
    assert attn_bias[0, 0, 2, 1] == floor, "cross-document attention leaked"
    assert attn_bias[0, 0, 2, 2] == 0.0, "self-attention was masked"
    assert attn_bias[0, 0, 2, 3] == floor, "future token was visible"
    # Position 1 (doc 0) still attends its own past within the document.
    assert attn_bias[0, 0, 1, 0] == 0.0
    # Every query keeps at least its own position, so no rows are fully masked.
    assert bool(query_valid.all()), query_valid
    print("keep-mask / causal-bias fold tests passed")
    return True


class _CharTokenizer:
    """Minimal stand-in: one token per character, ignoring special tokens."""

    def encode(self, text, max_length=None, add_special_tokens=True):
        ids = [ord(c) for c in text]
        return ids if max_length is None else ids[:max_length]


def test_packed_stream_partitions_and_rebases_documents() -> bool:
    texts = iter(["aaaa", "bbbb", "cccc", "dddd", "eeee"])
    stream = PackedSequenceStream(
        texts,
        _CharTokenizer(),
        sequence_length=6,
        max_document_tokens=8,
    )

    first = next(stream)
    # 4 'a' tokens + 2 'b' tokens fill the window; ids start at 0.
    assert first["segment_ids"].tolist() == [0, 0, 0, 0, 1, 1], first["segment_ids"].tolist()
    assert first["input_ids"].shape == first["segment_ids"].shape

    second = next(stream)
    # Next window leads with leftover 'b' tokens, then 'c'/'d'; ids are re-based to 0.
    assert second["segment_ids"].tolist() == [0, 0, 1, 1, 1, 1], second["segment_ids"].tolist()
    print("packed-stream document partition tests passed")
    return True


if __name__ == "__main__":
    test_document_keep_mask_is_block_diagonal()
    test_keep_mask_folds_into_causal_bias()
    test_packed_stream_partitions_and_rebases_documents()
