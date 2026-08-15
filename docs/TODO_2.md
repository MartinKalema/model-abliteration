Implemented the multimodal loader selection in [`loader.py`](/Users/martin/Desktop/OBLITERATUS/obliteratus/models/loader.py:282) and added [`test_multimodal_loader_selection.py`](/Users/martin/Desktop/OBLITERATUS/tests/test_multimodal_loader_selection.py).

Key behavior:

- Composite Gemma, Llama4/Mllama, Mistral3, LFM2-VL, Qwen-VL, and Qwen3.5 configs use `AutoModelForImageTextToText`.
- Detection uses explicit normalized `model_type` and architecture names.
- Text-only sibling configs remain on `AutoModelForCausalLM`.
- Classification behavior is unchanged.
- Older Transformers versions produce a clear upgrade error.
- The offline load test confirms the complete wrapper, including its vision tower, is retained.

Validation: 42 focused tests passed.

For [`abliterate.py`](/Users/martin/Desktop/OBLITERATUS/obliteratus/abliterate.py:4826), I recommend replacing the root-only `lm_head` scan with:

```python
lm_head_obj = None
try:
    lm_head_obj = model.get_output_embeddings()
except (AttributeError, NotImplementedError, TypeError):
    pass

lm_head_parent = None
lm_head_attr = None
if isinstance(getattr(lm_head_obj, "weight", None), torch.Tensor):
    for qualified_name, module in model.named_modules(remove_duplicate=False):
        if module is lm_head_obj and qualified_name:
            parent_name, _, lm_head_attr = qualified_name.rpartition(".")
            lm_head_parent = (
                model.get_submodule(parent_name) if parent_name else model
            )
            break

# Compatibility fallback for custom models with no embedding API.
if lm_head_parent is None:
    for head_name in ("lm_head", "embed_out", "output"):
        head = getattr(model, head_name, None)
        if isinstance(getattr(head, "weight", None), torch.Tensor):
            lm_head_obj = head
            lm_head_parent = model
            lm_head_attr = head_name
            break
```

Then the projection call around line 4884 should use:

```python
self._project_out_advanced(
    lm_head_parent,
    d,
    [lm_head_attr],
    orientation="input",
    ...
)
```

That preserves existing projection and quantization behavior while supporting nested heads such as `language_model.lm_head`. I did not edit `abliterate.py`.