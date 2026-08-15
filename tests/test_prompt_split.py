"""Tests for held-out prompt construction."""

from __future__ import annotations

import pytest

from obliteratus.evaluation.prompt_split import split_prompt_pairs


def _corpus(n: int = 100):
    return [f"harm {i}" for i in range(n)], [f"benign {i}" for i in range(n)]


def test_split_is_deterministic_and_order_independent():
    harmful, harmless = _corpus()
    first = split_prompt_pairs(harmful, harmless, seed=9)

    reordered = list(zip(harmful, harmless, strict=True))[::-1]
    second = split_prompt_pairs(
        [pair[0] for pair in reordered],
        [pair[1] for pair in reordered],
        seed=9,
    )

    assert set(zip(first.holdout_harmful, first.holdout_harmless, strict=True)) == set(
        zip(second.holdout_harmful, second.holdout_harmless, strict=True)
    )
    assert first.fingerprint == second.fingerprint


def test_duplicate_groups_never_cross_the_boundary():
    harmful, harmless = _corpus(80)
    harmful.extend([" HARM 3 ", "harm 3"])
    harmless.extend(["BENIGN   3", "benign 3"])

    split = split_prompt_pairs(harmful, harmless, min_holdout=20)
    discovery = {
        (h.strip().casefold(), " ".join(b.split()).casefold())
        for h, b in zip(split.discovery_harmful, split.discovery_harmless, strict=True)
    }
    holdout = {
        (h.strip().casefold(), " ".join(b.split()).casefold())
        for h, b in zip(split.holdout_harmful, split.holdout_harmless, strict=True)
    }

    assert split.disjoint is True
    assert discovery.isdisjoint(holdout)
    assert len(split.holdout_harmful) == len(
        {" ".join(item.split()).casefold() for item in split.holdout_harmful}
    )
    assert len(split.holdout_harmless) == len(
        {" ".join(item.split()).casefold() for item in split.holdout_harmless}
    )


def test_repeating_only_one_side_still_keeps_pairs_together():
    harmful, harmless = _corpus(80)
    harmful.extend(["harm 3", "new harmful"])
    harmless.extend(["new benign", "benign 7"])

    split = split_prompt_pairs(harmful, harmless, min_holdout=20)
    discovery_harmful = {item.strip().casefold() for item in split.discovery_harmful}
    holdout_harmful = {item.strip().casefold() for item in split.holdout_harmful}
    discovery_harmless = {item.strip().casefold() for item in split.discovery_harmless}
    holdout_harmless = {item.strip().casefold() for item in split.holdout_harmless}

    assert discovery_harmful.isdisjoint(holdout_harmful)
    assert discovery_harmless.isdisjoint(holdout_harmless)


def test_explicit_evaluation_set_is_kept_separate():
    harmful, harmless = _corpus(10)
    split = split_prompt_pairs(
        harmful,
        harmless,
        evaluation_harmful=["held-out harm"],
        evaluation_harmless=["held-out benign"],
    )

    assert split.discovery_harmful == tuple(harmful)
    assert split.holdout_harmful == ("held-out harm",)
    assert split.explicit_evaluation_set is True


def test_explicit_evaluation_overlap_is_rejected():
    harmful, harmless = _corpus(10)
    with pytest.raises(ValueError, match="overlap"):
        split_prompt_pairs(
            harmful,
            harmless,
            evaluation_harmful=[harmful[0]],
            evaluation_harmless=[harmless[0]],
        )


def test_explicit_evaluation_rejects_overlap_on_either_side():
    harmful, harmless = _corpus(10)
    with pytest.raises(ValueError, match="overlap"):
        split_prompt_pairs(
            harmful,
            harmless,
            evaluation_harmful=["new harmful"],
            evaluation_harmless=[harmless[0]],
        )


def test_explicit_evaluation_rejects_duplicate_rows_as_fake_sample_size():
    harmful, harmless = _corpus(10)
    with pytest.raises(ValueError, match="distinct normalized"):
        split_prompt_pairs(
            harmful,
            harmless,
            evaluation_harmful=["held out"] * 32,
            evaluation_harmless=[f"benign held out {index}" for index in range(32)],
        )


def test_automatic_holdout_counts_duplicate_groups_once():
    harmful, harmless = _corpus(70)
    harmful.extend(["harm 3", " HARM   3 ", "new harmful"])
    harmless.extend(["new benign one", "new benign two", "benign 7"])

    split = split_prompt_pairs(harmful, harmless, min_holdout=20)

    normalized_harmful = [" ".join(item.split()).casefold() for item in split.holdout_harmful]
    normalized_harmless = [" ".join(item.split()).casefold() for item in split.holdout_harmless]
    assert len(normalized_harmful) == len(set(normalized_harmful))
    assert len(normalized_harmless) == len(set(normalized_harmless))
    assert len(split.holdout_harmful) >= 20


def test_large_confirmation_holdout_can_use_a_separate_discovery_minimum():
    harmful, harmless = _corpus(99)

    split = split_prompt_pairs(
        harmful,
        harmless,
        min_holdout=64,
        min_discovery=32,
    )

    assert len(split.holdout_harmful) == 64
    assert len(split.discovery_harmful) == 35
    assert split.disjoint is True


def test_single_unique_group_cannot_claim_disjoint_evidence():
    split = split_prompt_pairs(["same"] * 10, ["pair"] * 10)

    assert split.holdout_harmful == ()
    assert split.disjoint is False
