"""Named Hugging Face dataset presets and mixture parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


COMMON_TEXT_FIELDS = ("text", "content", "raw_content", "document", "body")


@dataclass(frozen=True)
class DatasetSource:
    alias: str
    path: str
    config_name: Optional[str]
    split: str
    text_field: str


def _src(alias: str, path: str, config_name: Optional[str], text_field: str = "text") -> DatasetSource:
    return DatasetSource(alias=alias, path=path, config_name=config_name, split="train", text_field=text_field)


_CODE_SEARCH_LANGS = ("python", "go", "javascript", "java", "php", "ruby")

DATASET_PRESETS: Dict[str, DatasetSource] = {
    "fineweb_edu": _src("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT"),
    "dclm": _src("dclm", "mlfoundations/dclm-baseline-1.0", None),
    "c4": _src("c4", "allenai/c4", "en"),
    "finemath_3plus": _src("finemath_3plus", "HuggingFaceTB/finemath", "finemath-3plus"),
    "open_web_math": _src("open_web_math", "open-web-math/open-web-math", None),
    **{
        f"code_search_net_{lang}": _src(
            f"code_search_net_{lang}",
            "code-search-net/code_search_net",
            lang,
            "whole_func_string",
        )
        for lang in _CODE_SEARCH_LANGS
    },
}


MIXTURE_GROUP_PRESETS: Dict[str, Tuple[Tuple[str, float], ...]] = {
    # Weighted toward the most broadly useful languages while still covering
    # every CodeSearchNet shard the trainer can stream without gated access.
    "code_search_net_all": (
        ("code_search_net_python", 0.30),
        ("code_search_net_javascript", 0.22),
        ("code_search_net_java", 0.18),
        ("code_search_net_go", 0.15),
        ("code_search_net_php", 0.08),
        ("code_search_net_ruby", 0.07),
    ),
}


def parse_mixture_entry(entry: str) -> List[Tuple[DatasetSource, float]]:
    entry = entry.strip()
    if not entry:
        raise ValueError("Mixture entries must not be empty")
    if "=" not in entry:
        raise ValueError(f"Mixture entry '{entry}' is missing '=weight'")

    source_name, weight_text = entry.rsplit("=", 1)
    weight = float(weight_text)
    if weight <= 0:
        raise ValueError(f"Mixture weight must be positive: {entry}")

    source_name = source_name.strip()
    if source_name in DATASET_PRESETS:
        return [(DATASET_PRESETS[source_name], weight)]

    if source_name in MIXTURE_GROUP_PRESETS:
        return [
            (DATASET_PRESETS[member_name], weight * member_weight)
            for member_name, member_weight in MIXTURE_GROUP_PRESETS[source_name]
        ]

    parts = source_name.split("|")
    if len(parts) != 4:
        raise ValueError(
            "Custom mixture entries must use 'path|config|split|text_field=weight'"
        )

    path, config_name, split, text_field = parts
    return [(
        DatasetSource(
            alias=path,
            path=path,
            config_name=config_name or None,
            split=split,
            text_field=text_field,
        ),
        weight,
    )]


def parse_mixture(spec: str) -> List[Tuple[DatasetSource, float]]:
    expanded: List[Tuple[DatasetSource, float]] = []
    for entry in spec.split(","):
        if entry.strip():
            expanded.extend(parse_mixture_entry(entry))
    return expanded
