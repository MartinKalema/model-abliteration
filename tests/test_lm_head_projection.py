from __future__ import annotations

import torch
from torch import nn

from obliteratus.abliterate import AbliterationPipeline


class _NestedLanguageModel(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)


class _CompositeModel(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.language_model = _NestedLanguageModel(hidden_size, vocab_size)

    def get_output_embeddings(self):
        return self.language_model.lm_head


def test_nested_output_embedding_is_resolved_and_projected():
    hidden_size = 8
    model = _CompositeModel(hidden_size, vocab_size=16)
    original = model.language_model.lm_head.weight.detach().clone()
    direction = torch.randn(hidden_size, 1)
    direction /= direction.norm()

    parent, attribute, head = AbliterationPipeline._resolve_lm_head_projection(model)

    assert parent is model.language_model
    assert attribute == "lm_head"
    assert head is model.language_model.lm_head
    count = AbliterationPipeline._project_out_advanced(
        parent,
        direction,
        [attribute],
        orientation="input",
        regularization=0.0,
    )

    assert count == 1
    assert not torch.allclose(model.language_model.lm_head.weight, original)
    residual = model.language_model.lm_head.weight @ direction
    assert residual.norm().item() < 1e-5


class _RootFallbackModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_out = nn.Linear(4, 8, bias=False)

    def get_output_embeddings(self):
        raise NotImplementedError


def test_root_head_name_remains_a_compatibility_fallback():
    model = _RootFallbackModel()

    parent, attribute, head = AbliterationPipeline._resolve_lm_head_projection(model)

    assert parent is model
    assert attribute == "embed_out"
    assert head is model.embed_out
