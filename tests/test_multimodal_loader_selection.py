from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from obliteratus.models import loader
from obliteratus.architecture_manifest import build_projection_manifest
from obliteratus.models.loader import ModelHandle


@pytest.mark.parametrize(
    "model_type",
    sorted(loader._IMAGE_TEXT_MODEL_TYPES),
)
def test_composite_model_types_use_image_text_auto_class(model_type):
    config = SimpleNamespace(model_type=model_type, architectures=[])

    if loader.AutoModelForImageTextToText is None:
        with pytest.raises(RuntimeError, match="AutoModelForImageTextToText"):
            loader._select_model_class("causal_lm", config)
    else:
        assert (
            loader._select_model_class("causal_lm", config)
            is loader.AutoModelForImageTextToText
        )


def test_composite_model_requires_image_text_auto_support(monkeypatch):
    monkeypatch.setattr(loader, "AutoModelForImageTextToText", None)
    config = SimpleNamespace(model_type="llama4", architectures=[])

    with pytest.raises(RuntimeError, match="AutoModelForImageTextToText"):
        loader._select_model_class("causal_lm", config)


def test_conditional_generation_architecture_selects_image_text_auto_class():
    config = SimpleNamespace(
        model_type="custom_wrapper",
        architectures="Qwen3VLForConditionalGeneration",
    )

    if loader.AutoModelForImageTextToText is None:
        with pytest.raises(RuntimeError, match="AutoModelForImageTextToText"):
            loader._select_model_class("causal_lm", config)
    else:
        assert (
            loader._select_model_class("causal_lm", config)
            is loader.AutoModelForImageTextToText
        )


@pytest.mark.parametrize(
    ("model_type", "architecture"),
    [
        ("gemma3_text", "Gemma3ForCausalLM"),
        ("llama4_text", "Llama4ForCausalLM"),
        ("mllama_text_model", "MllamaForCausalLM"),
        ("qwen3_5_text", "Qwen3_5ForCausalLM"),
        ("qwen3_5_moe_text", "Qwen3_5MoeForCausalLM"),
        ("llama", "LlamaForCausalLM"),
    ],
)
def test_text_only_siblings_stay_on_causal_lm_auto_class(model_type, architecture):
    config = SimpleNamespace(model_type=model_type, architectures=[architecture])

    assert loader._select_model_class("causal_lm", config) is loader.AutoModelForCausalLM


def test_classification_task_does_not_switch_to_image_text_auto_class():
    config = SimpleNamespace(
        model_type="llama4",
        architectures=["Llama4ForConditionalGeneration"],
    )

    assert (
        loader._select_model_class("classification", config)
        is loader.AutoModelForSequenceClassification
    )


class _CompositeWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.Linear(2, 2, bias=False)
        self.language_model = nn.Linear(2, 2, bias=False)


class _FakeImageTextAutoModel:
    loaded_model = None
    load_kwargs = None

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.load_kwargs = kwargs
        cls.loaded_model = _CompositeWrapper()
        return cls.loaded_model


class _FakeAutoConfig:
    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return SimpleNamespace(
            model_type="qwen3_vl",
            architectures=["Qwen3VLForConditionalGeneration"],
            text_config=SimpleNamespace(
                num_hidden_layers=1,
                num_attention_heads=1,
                hidden_size=2,
                intermediate_size=4,
                vocab_size=8,
            ),
        )


class _FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"


class _FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        return _FakeTokenizer()


def test_load_model_preserves_the_composite_wrapper(monkeypatch):
    monkeypatch.setattr(loader, "AutoConfig", _FakeAutoConfig)
    monkeypatch.setattr(loader, "AutoTokenizer", _FakeAutoTokenizer)
    monkeypatch.setattr(loader, "AutoModelForImageTextToText", _FakeImageTextAutoModel)
    monkeypatch.setattr(loader, "_apply_deferred_shims", lambda: None)
    monkeypatch.setattr(loader, "_available_gpu_memory_gb", lambda: 0.0)
    monkeypatch.setattr(loader.dev, "empty_cache", lambda: None)

    handle = loader.load_model(
        "offline/qwen3-vl-fixture",
        device="cpu",
        skip_snapshot=True,
    )

    assert handle.model is _FakeImageTextAutoModel.loaded_model
    assert isinstance(handle.model.vision_model, nn.Linear)
    assert isinstance(handle.model.language_model, nn.Linear)
    assert _FakeImageTextAutoModel.load_kwargs["config"] is handle.config
    assert handle.model.training is False


_TEXT_HIDDEN_SIZE = 4
_TEXT_INTERMEDIATE_SIZE = 6
_TEXT_LAYER_COUNT = 2


class _SyntheticAttention(nn.Module):
    def __init__(self, *, output_name: str = "o_proj"):
        super().__init__()
        self.q_proj = nn.Linear(_TEXT_HIDDEN_SIZE, _TEXT_HIDDEN_SIZE, bias=False)
        self.k_proj = nn.Linear(_TEXT_HIDDEN_SIZE, _TEXT_HIDDEN_SIZE, bias=False)
        self.v_proj = nn.Linear(_TEXT_HIDDEN_SIZE, _TEXT_HIDDEN_SIZE, bias=False)
        setattr(
            self,
            output_name,
            nn.Linear(_TEXT_HIDDEN_SIZE, _TEXT_HIDDEN_SIZE, bias=False),
        )


class _SyntheticDenseMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(
            _TEXT_HIDDEN_SIZE, _TEXT_INTERMEDIATE_SIZE, bias=False
        )
        self.up_proj = nn.Linear(
            _TEXT_HIDDEN_SIZE, _TEXT_INTERMEDIATE_SIZE, bias=False
        )
        self.down_proj = nn.Linear(
            _TEXT_INTERMEDIATE_SIZE, _TEXT_HIDDEN_SIZE, bias=False
        )


class _SyntheticDenseTextLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _SyntheticAttention()
        self.mlp = _SyntheticDenseMlp()


class _SyntheticPackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.empty(2, 2 * _TEXT_INTERMEDIATE_SIZE, _TEXT_HIDDEN_SIZE)
        )
        self.down_proj = nn.Parameter(
            torch.empty(2, _TEXT_HIDDEN_SIZE, _TEXT_INTERMEDIATE_SIZE)
        )


class _SyntheticMoe(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(_TEXT_HIDDEN_SIZE, 2, bias=False)
        self.experts = _SyntheticPackedExperts()


class _SyntheticMoeTextLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _SyntheticAttention()
        self.mlp = _SyntheticMoe()


class _SyntheticShortConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(_TEXT_HIDDEN_SIZE, 3 * _TEXT_HIDDEN_SIZE, bias=False)
        self.out_proj = nn.Linear(_TEXT_HIDDEN_SIZE, _TEXT_HIDDEN_SIZE, bias=False)
        # A stateful convolution kernel is deliberately outside the manifest.
        self.conv = nn.Conv1d(
            _TEXT_HIDDEN_SIZE,
            _TEXT_HIDDEN_SIZE,
            kernel_size=2,
            groups=_TEXT_HIDDEN_SIZE,
        )


class _SyntheticLfm2TextLayer(nn.Module):
    def __init__(self, *, attention: bool):
        super().__init__()
        if attention:
            self.self_attn = _SyntheticAttention(output_name="out_proj")
        else:
            self.conv = _SyntheticShortConv()
        self.feed_forward = nn.Module()
        self.feed_forward.w1 = nn.Linear(
            _TEXT_HIDDEN_SIZE, _TEXT_INTERMEDIATE_SIZE, bias=False
        )
        self.feed_forward.w3 = nn.Linear(
            _TEXT_HIDDEN_SIZE, _TEXT_INTERMEDIATE_SIZE, bias=False
        )
        self.feed_forward.w2 = nn.Linear(
            _TEXT_INTERMEDIATE_SIZE, _TEXT_HIDDEN_SIZE, bias=False
        )


class _SyntheticVisionLayer(nn.Module):
    """A valid-looking decoder layout that must never enter the text manifest."""

    def __init__(self):
        super().__init__()
        self.self_attn = _SyntheticAttention()
        self.mlp = _SyntheticDenseMlp()


class _SyntheticMultimodalWrapper(nn.Module):
    def __init__(self, model_type: str, text_kind: str):
        super().__init__()
        self.model = nn.Module()

        vision_layers = nn.ModuleList(
            [_SyntheticVisionLayer() for _ in range(_TEXT_LAYER_COUNT)]
        )
        if model_type == "lfm2_vl":
            self.model.vision_tower = nn.Module()
            self.model.vision_tower.layers = vision_layers
        else:
            self.model.visual = nn.Module()
            self.model.visual.blocks = vision_layers

        self.model.multi_modal_projector = nn.Linear(
            _TEXT_HIDDEN_SIZE, _TEXT_HIDDEN_SIZE, bias=False
        )
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(17, _TEXT_HIDDEN_SIZE)
        if text_kind == "dense":
            text_layers = [_SyntheticDenseTextLayer() for _ in range(_TEXT_LAYER_COUNT)]
        elif text_kind == "moe":
            text_layers = [_SyntheticMoeTextLayer() for _ in range(_TEXT_LAYER_COUNT)]
        else:
            text_layers = [
                _SyntheticLfm2TextLayer(attention=False),
                _SyntheticLfm2TextLayer(attention=True),
            ]
        self.model.language_model.layers = nn.ModuleList(text_layers)
        self.model.language_model.norm = nn.LayerNorm(_TEXT_HIDDEN_SIZE)
        self.lm_head = nn.Linear(_TEXT_HIDDEN_SIZE, 17, bias=False)


def _synthetic_multimodal_handle(model_type: str, text_kind: str) -> ModelHandle:
    return ModelHandle(
        model=_SyntheticMultimodalWrapper(model_type, text_kind),
        tokenizer=SimpleNamespace(padding_side="right", pad_token_id=None),
        config=SimpleNamespace(
            model_type=model_type,
            text_config=SimpleNamespace(
                num_hidden_layers=_TEXT_LAYER_COUNT,
                num_attention_heads=2,
                hidden_size=_TEXT_HIDDEN_SIZE,
                intermediate_size=_TEXT_INTERMEDIATE_SIZE,
            ),
        ),
        model_name=f"offline/{model_type}",
        task="causal_lm",
    )


def _dense_manifest_names() -> set[str]:
    names: set[str] = set()
    for layer_index in range(_TEXT_LAYER_COUNT):
        prefix = f"model.language_model.layers.{layer_index}"
        names.update(
            f"{prefix}.self_attn.{name}.weight"
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        names.update(
            f"{prefix}.mlp.{name}.weight"
            for name in ("gate_proj", "up_proj", "down_proj")
        )
    return names


def _moe_manifest_names() -> set[str]:
    names: set[str] = set()
    for layer_index in range(_TEXT_LAYER_COUNT):
        prefix = f"model.language_model.layers.{layer_index}"
        names.update(
            f"{prefix}.self_attn.{name}.weight"
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        names.update(
            {
                f"{prefix}.mlp.gate.weight",
                f"{prefix}.mlp.experts.gate_up_proj",
                f"{prefix}.mlp.experts.down_proj",
            }
        )
    return names


def _lfm2_manifest_names() -> set[str]:
    names = {
        "model.language_model.layers.0.conv.in_proj.weight",
        "model.language_model.layers.0.conv.out_proj.weight",
    }
    names.update(
        f"model.language_model.layers.1.self_attn.{name}.weight"
        for name in ("q_proj", "k_proj", "v_proj", "out_proj")
    )
    for layer_index in range(_TEXT_LAYER_COUNT):
        names.update(
            f"model.language_model.layers.{layer_index}.feed_forward.{name}.weight"
            for name in ("w1", "w2", "w3")
        )
    return names


@pytest.mark.parametrize(
    ("model_type", "text_kind", "expected_names", "expected_branch_paths"),
    [
        (
            "qwen2_vl",
            "dense",
            _dense_manifest_names(),
            {("attention", "self_attn"), ("ffn", "mlp")},
        ),
        (
            "qwen2_5_vl",
            "dense",
            _dense_manifest_names(),
            {("attention", "self_attn"), ("ffn", "mlp")},
        ),
        (
            "qwen3_vl",
            "dense",
            _dense_manifest_names(),
            {("attention", "self_attn"), ("ffn", "mlp")},
        ),
        (
            "qwen3_vl_moe",
            "moe",
            _moe_manifest_names(),
            {("attention", "self_attn"), ("ffn", "mlp")},
        ),
        (
            "lfm2_vl",
            "hybrid",
            _lfm2_manifest_names(),
            {
                ("attention", "conv"),
                ("attention", "self_attn"),
                ("ffn", "feed_forward"),
            },
        ),
    ],
)
def test_multimodal_alias_builds_text_only_projection_manifest(
    model_type, text_kind, expected_names, expected_branch_paths
):
    handle = _synthetic_multimodal_handle(model_type, text_kind)

    manifest = build_projection_manifest(handle, "all")
    manifest_names = {entry.qualified_name for entry in manifest.entries}

    assert manifest.architecture == model_type
    assert manifest.layer_path == "model.language_model.layers"
    assert manifest_names == expected_names
    assert all(
        name.startswith("model.language_model.layers.") for name in manifest_names
    )
    assert {
        (coverage["kind"], coverage["path"])
        for coverage in manifest.branch_coverage
    } == expected_branch_paths

    distractor_names = {
        name
        for name, _ in handle.model.named_parameters()
        if ".visual." in name
        or ".vision_tower." in name
        or ".multi_modal_projector." in name
    }
    assert distractor_names
    assert manifest_names.isdisjoint(distractor_names)
