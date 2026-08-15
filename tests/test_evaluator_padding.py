"""Causal perplexity must exclude padded target tokens."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from obliteratus.evaluation.evaluator import Evaluator


class _Batch(dict):
    def to(self, device):
        return _Batch({key: value.to(device) for key, value in self.items()})


class _Tokenizer:
    def __call__(self, _texts, **_kwargs):
        return _Batch(
            input_ids=torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]]),
            attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
        )


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.seen_labels = None

    def forward(self, *, input_ids, attention_mask, labels):
        self.seen_labels = labels.detach().cpu()
        assert input_ids.shape == attention_mask.shape == labels.shape
        return SimpleNamespace(loss=torch.tensor(math.log(2.0), device=input_ids.device))


class _Dataset:
    def __len__(self):
        return 2

    def __getitem__(self, item):
        if isinstance(item, slice):
            return {"text": ["short", "long"][item]}
        return {"text": ["short", "long"][item]}


def test_causal_lm_perplexity_masks_padding_labels():
    model = _Model()
    handle = SimpleNamespace(model=model, tokenizer=_Tokenizer(), task="causal_lm")

    result = Evaluator(handle, _Dataset(), batch_size=2).evaluate()

    assert result["perplexity"] == pytest.approx(2.0)
    assert model.seen_labels.tolist() == [[1, 2, 3, -100], [4, 5, 6, 7]]
