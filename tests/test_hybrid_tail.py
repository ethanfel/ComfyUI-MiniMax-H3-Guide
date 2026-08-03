from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _load_tail_module(monkeypatch):
    folder_paths = ModuleType("folder_paths")
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    comfy = ModuleType("comfy")
    comfy.__path__ = []
    text_encoders = ModuleType("comfy.text_encoders")
    text_encoders.__path__ = []
    llama = ModuleType("comfy.text_encoders.llama")

    @dataclass
    class Qwen3VL32BConfig:
        pass

    llama.Qwen3VL_32BConfig = Qwen3VL32BConfig
    llama.Llama2_ = object
    text_encoders.llama = llama
    comfy.text_encoders = text_encoders

    for name in ("hooks", "model_management", "model_patcher", "ops", "utils"):
        submodule = ModuleType(f"comfy.{name}")
        setattr(comfy, name, submodule)
        monkeypatch.setitem(sys.modules, f"comfy.{name}", submodule)
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.text_encoders", text_encoders)
    monkeypatch.setitem(sys.modules, "comfy.text_encoders.llama", llama)

    module_name = "_minimax_h3_hybrid_tail_test"
    path = Path(__file__).parents[1] / "hybrid_tail.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _weight(qdata, scale, *, layout="TensorWiseINT8Layout", **params):
    return SimpleNamespace(
        _qdata=qdata,
        _layout_cls=layout,
        _params=SimpleNamespace(scale=scale, **params),
    )


def test_chunked_int8_head_accepts_tensorwise_scalar_scale(monkeypatch):
    tail_module = _load_tail_module(monkeypatch)
    qdata = torch.tensor([[1, -2], [3, 4], [-5, 6]], dtype=torch.int8)
    scale = torch.tensor(0.25)
    weight = _weight(qdata, scale)

    class Head:
        def __init__(self):
            self.weight = weight

        def __call__(self, _hidden):
            raise AssertionError("quantized tail must use the chunked path")

    tail = object.__new__(tail_module.Qwen3VL32BGenerationTail)
    torch.nn.Module.__init__(tail)
    tail.model = SimpleNamespace(lm_head=Head())
    hidden = torch.tensor([[[9.0, 9.0], [2.0, -1.0]]])

    actual = tail.logits(hidden)
    expected = torch.nn.functional.linear(
        hidden[:, -1:].float(), qdata.float() * scale
    )
    assert torch.equal(actual, expected)


def test_chunked_int8_head_reshapes_per_row_scale(monkeypatch):
    tail_module = _load_tail_module(monkeypatch)
    scale = torch.tensor([0.1, 0.2, 0.3, 0.4])
    chunk = tail_module._int8_scale_chunk(scale, 1, 3, 4)
    assert chunk.shape == (2, 1)
    assert torch.allclose(chunk[:, 0], torch.tensor([0.2, 0.3]))


def test_chunked_head_rejects_unsupported_quantized_layout(monkeypatch):
    tail_module = _load_tail_module(monkeypatch)
    weight = _weight(
        torch.zeros(3, 4, dtype=torch.int8),
        torch.ones(3, 1),
        layout="TensorCoreNVFP4Layout",
    )
    with pytest.raises(ValueError, match="must use ComfyUI int8_tensorwise"):
        tail_module._validate_int8_head(
            weight, weight._qdata, weight._params.scale
        )


def test_chunked_head_rejects_invalid_convrot_layout(monkeypatch):
    tail_module = _load_tail_module(monkeypatch)
    weight = _weight(
        torch.zeros(3, 12, dtype=torch.int8),
        torch.tensor(0.1),
        convrot=True,
        convrot_groupsize=8,
    )
    with pytest.raises(ValueError, match="power of four"):
        tail_module._validate_int8_head(
            weight, weight._qdata, weight._params.scale
        )
