"""Two-stage hierarchical tokenizer with dynamic grouping.

The tokenizer follows the structure from "From Characters to Tokens: Dynamic
Grouping with Hierarchical BPE":

1. First stage: standard BPE tokenization with ``tiktoken``.
2. Second stage: each first-stage token is expanded to raw bytes, an explicit
   end-of-patch marker is appended, and a learned BPE merge table compresses the
   patch until it is at most ``max_patch_size`` tokens long.

This implementation learns the second-stage merge rules from the first-stage
vocabulary itself, which keeps initialization self-contained and makes the
hierarchical grouping usable immediately.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple

try:
    import tiktoken
except ImportError as exc:  # pragma: no cover - depends on local environment.
    raise ImportError(
        "HierarchicalTokenizer requires the `tiktoken` package. Install it with "
        "`python3 -m pip install -r requirements.txt` or inside a virtual environment."
    ) from exc


Pair = Tuple[int, int]


class HierarchicalTokenizer:
    """Two-stage dynamic grouping tokenizer.

    The emitted ids live in a byte-and-merge vocabulary:
    - ``0..255`` are raw bytes
    - ``256`` is the end-of-patch marker
    - ``257`` is padding for optional fixed-size patch tensors
    - ``258`` and ``259`` are BOS/EOS
    - ``260+`` are learned second-stage merge tokens
    """

    def __init__(self, max_patch_size: int = 8, vocab_size_target: int = 131072) -> None:
        self.max_patch_size = max_patch_size
        self.vocab_size_target = vocab_size_target
        self.first_stage = tiktoken.get_encoding("cl100k_base")

        self.eop_id = 256
        self.pad_id = 257
        self.bos_id = 258
        self.eos_id = 259
        self.next_token_id = 260

        self.merges: List[Tuple[Pair, int]] = []
        self.merge_lookup: Dict[Pair, int] = {}
        self.reverse_merges: Dict[int, Pair] = {}

        self._build_second_stage_vocab()

    def _iter_first_stage_patches(self) -> Iterable[List[int]]:
        """Yield byte patches for each first-stage BPE token.

        Each patch is a byte sequence followed by the explicit end-of-patch
        marker. These patches are the training data for the second-stage BPE.
        """
        for token_id in range(self.first_stage.n_vocab):
            try:
                token_bytes = self.first_stage.decode_single_token_bytes(token_id)
            except KeyError:
                continue

            patch = list(token_bytes) + [self.eop_id]
            if patch:
                yield patch

    def _merge_pair(self, sequence: List[int], pair: Pair, new_id: int) -> List[int]:
        """Replace non-overlapping occurrences of ``pair`` with ``new_id``."""
        merged: List[int] = []
        index = 0
        while index < len(sequence):
            if index < len(sequence) - 1 and (sequence[index], sequence[index + 1]) == pair:
                merged.append(new_id)
                index += 2
            else:
                merged.append(sequence[index])
                index += 1
        return merged

    def _build_second_stage_vocab(self) -> None:
        """Learn hierarchical BPE merges over first-stage vocabulary patches."""
        active_sequences = [patch for patch in self._iter_first_stage_patches() if len(patch) > self.max_patch_size]

        while active_sequences and self.next_token_id < self.vocab_size_target:
            pair_counts: Counter[Pair] = Counter()
            for patch in active_sequences:
                pair_counts.update(zip(patch, patch[1:]))

            if not pair_counts:
                break

            pair, frequency = pair_counts.most_common(1)[0]
            if frequency < 2:
                break

            new_id = self.next_token_id
            self.next_token_id += 1
            self.merges.append((pair, new_id))
            self.merge_lookup[pair] = new_id
            self.reverse_merges[new_id] = pair

            updated_sequences: List[List[int]] = []
            for patch in active_sequences:
                merged_patch = self._merge_pair(patch, pair, new_id)
                if len(merged_patch) > self.max_patch_size:
                    updated_sequences.append(merged_patch)
            active_sequences = updated_sequences

        self.vocab_size = max(self.vocab_size_target, self.next_token_id)

    def _bytes_to_patch(self, token_id: int) -> List[int]:
        token_bytes = self.first_stage.decode_single_token_bytes(token_id)
        return list(token_bytes) + [self.eop_id]

    def _apply_merges(self, patch: List[int]) -> List[int]:
        """Apply learned second-stage merges in training order."""
        if len(patch) <= self.max_patch_size:
            return patch

        for pair, new_id in self.merges:
            patch = self._merge_pair(patch, pair, new_id)
            if len(patch) <= self.max_patch_size:
                break

        return patch

    def encode_patches(self, text: str, add_special_tokens: bool = False) -> List[List[int]]:
        """Encode text to a list of hierarchical patches.

        Each patch corresponds to one first-stage BPE token compressed by the
        learned second-stage BPE rules.
        """
        first_stage_ids = self.first_stage.encode(text)
        patches = [self._apply_merges(self._bytes_to_patch(token_id)) for token_id in first_stage_ids]

        if add_special_tokens:
            patches = [[self.bos_id], *patches, [self.eos_id]]

        return patches

    def encode(
        self,
        text: str,
        max_length: int = 8192,
        add_special_tokens: bool = False,
    ) -> List[int]:
        """Encode text to a flat token stream for the main transformer."""
        flat_ids = [token for patch in self.encode_patches(text, add_special_tokens=add_special_tokens) for token in patch]
        return flat_ids[:max_length]

    def encode_fixed_patches(self, text: str, add_special_tokens: bool = False) -> List[List[int]]:
        """Encode text to patches padded to ``max_patch_size`` tokens.

        This is useful when a separate local encoder is used for hierarchical
        processing.
        """
        padded_patches: List[List[int]] = []
        for patch in self.encode_patches(text, add_special_tokens=add_special_tokens):
            if len(patch) > self.max_patch_size:
                patch = patch[: self.max_patch_size]
            padded_patches.append(patch + [self.pad_id] * (self.max_patch_size - len(patch)))
        return padded_patches

    def _expand_token(self, token_id: int) -> List[int]:
        if token_id in self.reverse_merges:
            left, right = self.reverse_merges[token_id]
            return [*self._expand_token(left), *self._expand_token(right)]
        if token_id in {self.eop_id, self.pad_id, self.bos_id, self.eos_id}:
            return []
        return [token_id]

    def decode(self, ids: List[int]) -> str:
        """Decode a flat hierarchical token stream back to text."""
        byte_values: List[int] = []
        for token_id in ids:
            byte_values.extend(self._expand_token(token_id))
        return bytes(byte_values).decode("utf-8", errors="replace")

    def __len__(self) -> int:
        return self.vocab_size


if __name__ == "__main__":
    tokenizer = HierarchicalTokenizer()
    sample = "Hello, this is a hierarchical tokenizer test."
    tokens = tokenizer.encode(sample)
    print(f"Encoded {len(tokens)} ids: {tokens[:20]}")
