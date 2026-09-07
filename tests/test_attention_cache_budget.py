import torch

from utils import _tensor_byte_cache


def test_attention_cache_evicts_by_storage_and_skips_oversized_values():
    @_tensor_byte_cache(max_bytes=128)
    def mask(length):
        return torch.zeros(length, length).view(1, 1, length, length)

    small = mask(4)
    assert mask(4) is small
    mask(5)  # 100 bytes evicts the older 64-byte table.
    assert mask.cache_info().currsize == 1
    assert mask.cache_info().retained_bytes == 100
    assert mask(4) is not small
    mask(8)  # Too large to retain, without evicting the useful small table.
    assert mask.cache_info().retained_bytes == 64
    assert mask(8) is not mask(8)
    mask.cache_clear()
    assert mask.cache_info().retained_bytes == 0
