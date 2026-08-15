"""Deterministic, duplicate-group-aware prompt splitting for model editing."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass


def _normalize_prompt(text: str) -> str:
    """Normalize only enough to keep obvious duplicates in the same group."""

    return re.sub(r"\s+", " ", text).strip().casefold()


def _pair_key(harmful: str, harmless: str) -> str:
    return f"{_normalize_prompt(harmful)}\0{_normalize_prompt(harmless)}"


@dataclass(frozen=True)
class PromptSplit:
    discovery_harmful: tuple[str, ...]
    discovery_harmless: tuple[str, ...]
    holdout_harmful: tuple[str, ...]
    holdout_harmless: tuple[str, ...]
    fingerprint: str
    disjoint: bool
    explicit_evaluation_set: bool

    def to_metadata(self) -> dict[str, str | int | bool]:
        return {
            "discovery_pairs": len(self.discovery_harmful),
            "holdout_pairs": len(self.holdout_harmful),
            "fingerprint_sha256": self.fingerprint,
            "duplicate_group_disjoint": self.disjoint,
            "explicit_evaluation_set": self.explicit_evaluation_set,
        }


def split_prompt_pairs(
    harmful: Sequence[str],
    harmless: Sequence[str],
    *,
    holdout_fraction: float = 0.15,
    seed: int = 42,
    min_holdout: int = 32,
    min_discovery: int | None = None,
    evaluation_harmful: Sequence[str] | None = None,
    evaluation_harmless: Sequence[str] | None = None,
) -> PromptSplit:
    """Split paired prompts without putting duplicate pairs on both sides.

    Pair groups are ranked by SHA-256 rather than input order, making the split
    stable and resistant to corpus reordering.  If an explicit evaluation set
    is supplied, discovery retains the full training corpus and overlap is
    rejected.

    ``min_discovery`` can be declared independently when a search needs a
    larger selection/confirmation holdout. Small corpora may still yield fewer
    than ``min_holdout`` items. The caller must decide whether that evidence is
    sufficient before editing; the default damage gate rejects it, while
    construction and exploratory dry-runs remain possible.
    """

    harmful_items = list(harmful)
    harmless_items = list(harmless)
    if len(harmful_items) != len(harmless_items):
        raise ValueError("harmful and harmless prompt lists must have equal length")
    if not harmful_items:
        raise ValueError("at least one prompt pair is required")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not isinstance(min_holdout, int) or isinstance(min_holdout, bool) or min_holdout < 0:
        raise ValueError("min_holdout must be a non-negative integer")
    if min_discovery is None:
        min_discovery = min_holdout
    if (
        not isinstance(min_discovery, int)
        or isinstance(min_discovery, bool)
        or min_discovery < 1
    ):
        raise ValueError("min_discovery must be a positive integer")

    explicit = evaluation_harmful is not None or evaluation_harmless is not None
    if explicit:
        if evaluation_harmful is None or evaluation_harmless is None:
            raise ValueError("both evaluation_harmful and evaluation_harmless are required")
        eval_harmful = list(evaluation_harmful)
        eval_harmless = list(evaluation_harmless)
        if not eval_harmful or len(eval_harmful) != len(eval_harmless):
            raise ValueError("explicit evaluation prompt lists must be non-empty and paired")

        # Repeated rows are not independent evidence.  In particular, a caller
        # must not be able to satisfy a 32-prompt acceptance minimum by passing
        # the same prompt 32 times (or by changing only its paired control).
        normalized_eval_harmful = [_normalize_prompt(item) for item in eval_harmful]
        normalized_eval_harmless = [_normalize_prompt(item) for item in eval_harmless]
        if (
            len(set(normalized_eval_harmful)) != len(normalized_eval_harmful)
            or len(set(normalized_eval_harmless)) != len(normalized_eval_harmless)
        ):
            raise ValueError(
                "explicit evaluation prompts must contain distinct normalized "
                "harmful and harmless rows"
            )

        discovery_harmful_texts = {_normalize_prompt(item) for item in harmful_items}
        discovery_harmless_texts = {_normalize_prompt(item) for item in harmless_items}
        holdout_harmful_texts = {_normalize_prompt(item) for item in eval_harmful}
        holdout_harmless_texts = {_normalize_prompt(item) for item in eval_harmless}
        if (
            discovery_harmful_texts & holdout_harmful_texts
            or discovery_harmless_texts & holdout_harmless_texts
        ):
            raise ValueError(
                "explicit evaluation prompts overlap the direction-discovery corpus"
            )
        holdout_keys = {
            _pair_key(h, b) for h, b in zip(eval_harmful, eval_harmless, strict=True)
        }
        fingerprint = _fingerprint(holdout_keys, seed)
        return PromptSplit(
            discovery_harmful=tuple(harmful_items),
            discovery_harmless=tuple(harmless_items),
            holdout_harmful=tuple(eval_harmful),
            holdout_harmless=tuple(eval_harmless),
            fingerprint=fingerprint,
            disjoint=True,
            explicit_evaluation_set=True,
        )

    # Build connected duplicate groups. Two pairs belong to the same group if
    # either side repeats after normalization, including transitive repeats.
    # Grouping only by the combined pair would leak a repeated harmful prompt
    # whenever it happened to be paired with a different benign control.
    parent = list(range(len(harmful_items)))

    def _find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def _union(left: int, right: int) -> None:
        left_root = _find(left)
        right_root = _find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_harmful: dict[str, int] = {}
    first_harmless: dict[str, int] = {}
    for index, (harm, benign) in enumerate(
        zip(harmful_items, harmless_items, strict=True)
    ):
        normalized_harm = _normalize_prompt(harm)
        normalized_benign = _normalize_prompt(benign)
        if normalized_harm in first_harmful:
            _union(index, first_harmful[normalized_harm])
        else:
            first_harmful[normalized_harm] = index
        if normalized_benign in first_harmless:
            _union(index, first_harmless[normalized_benign])
        else:
            first_harmless[normalized_benign] = index

    grouped_indices: dict[int, list[int]] = {}
    for index in range(len(harmful_items)):
        grouped_indices.setdefault(_find(index), []).append(index)

    groups: dict[str, list[int]] = {}
    for indices in grouped_indices.values():
        stable_group_key = "\n".join(
            sorted(
                _pair_key(harmful_items[index], harmless_items[index])
                for index in indices
            )
        )
        groups[stable_group_key] = indices

    # A one-group corpus cannot produce a group-disjoint split.  Keep it all
    # for discovery; the fail-closed evidence check will reject before edit.
    if len(groups) < 2:
        return PromptSplit(
            discovery_harmful=tuple(harmful_items),
            discovery_harmless=tuple(harmless_items),
            holdout_harmful=(),
            holdout_harmless=(),
            fingerprint=_fingerprint((), seed),
            disjoint=False,
            explicit_evaluation_set=False,
        )

    # Acceptance samples duplicate *groups*, not rows.  Every selected group
    # contributes one deterministic representative to evaluation while all of
    # that group's other rows are excluded from discovery.  This preserves the
    # no-leak boundary without pretending duplicated prompts add sample size.
    group_count = len(groups)
    target = max(min_holdout, math.ceil(group_count * holdout_fraction))
    # Keep a useful discovery side too.  For a small corpus this deliberately
    # yields an undersized holdout, which the fail-closed evidence gate reports
    # before editing instead of silently training on one prompt.
    discovery_floor = min(min_discovery, group_count - 1)
    target = min(target, group_count - discovery_floor)
    target = max(1, target)

    ranked_keys = sorted(
        groups,
        key=lambda key: hashlib.sha256(f"{seed}\0{key}".encode()).digest(),
    )
    selected_group_keys = ranked_keys[:target]
    selected_member_indices = {
        index
        for key in selected_group_keys
        for index in groups[key]
    }
    discovery_indices = [
        index
        for index in range(len(harmful_items))
        if index not in selected_member_indices
    ]

    # Pick one stable representative per connected duplicate group.  Raw text
    # is used only as a deterministic tie-break after normalized pair content.
    holdout_order = sorted(
        min(
            groups[key],
            key=lambda index: (
                _pair_key(harmful_items[index], harmless_items[index]),
                harmful_items[index],
                harmless_items[index],
            ),
        )
        for key in selected_group_keys
    )
    discovery_keys = {
        _pair_key(harmful_items[i], harmless_items[i]) for i in discovery_indices
    }
    holdout_keys = {
        _pair_key(harmful_items[i], harmless_items[i]) for i in holdout_order
    }
    discovery_harmful_keys = {
        _normalize_prompt(harmful_items[index]) for index in discovery_indices
    }
    discovery_harmless_keys = {
        _normalize_prompt(harmless_items[index]) for index in discovery_indices
    }
    holdout_harmful_keys = {
        _normalize_prompt(harmful_items[index]) for index in holdout_order
    }
    holdout_harmless_keys = {
        _normalize_prompt(harmless_items[index]) for index in holdout_order
    }
    disjoint = bool(holdout_order) and (
        discovery_keys.isdisjoint(holdout_keys)
        and discovery_harmful_keys.isdisjoint(holdout_harmful_keys)
        and discovery_harmless_keys.isdisjoint(holdout_harmless_keys)
    )

    return PromptSplit(
        discovery_harmful=tuple(harmful_items[i] for i in discovery_indices),
        discovery_harmless=tuple(harmless_items[i] for i in discovery_indices),
        holdout_harmful=tuple(harmful_items[i] for i in holdout_order),
        holdout_harmless=tuple(harmless_items[i] for i in holdout_order),
        fingerprint=_fingerprint(holdout_keys, seed),
        disjoint=disjoint,
        explicit_evaluation_set=False,
    )


def _fingerprint(keys: Sequence[str] | set[str], seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"seed={seed}\n".encode())
    for key in sorted(keys):
        digest.update(key.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


__all__ = ["PromptSplit", "split_prompt_pairs"]
