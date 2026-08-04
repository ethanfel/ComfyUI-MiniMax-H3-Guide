from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import weakref

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


def test_tail_temporaries_finish_before_managed_unload(monkeypatch):
    tail_module = _load_tail_module(monkeypatch)
    events = []
    wrapper = SimpleNamespace(execution_device="cpu")
    base = object()
    tail = object()
    base_patcher = SimpleNamespace(load_device="cuda:0", offload_device="cpu")
    tail_patcher = object()
    clip = SimpleNamespace(patcher=base_patcher)

    monkeypatch.setattr(tail_module, "_get_minimax_base", lambda _clip: (wrapper, base))
    monkeypatch.setattr(
        tail_module,
        "_build_tail",
        lambda _name, _load_device, _offload_device: (tail, tail_patcher),
    )

    def load_models(patchers, **kwargs):
        events.append(("load", patchers, kwargs))

    def generate(*args):
        assert wrapper.execution_device == "cuda:0"
        events.append(("generation_frame_returned", args))
        return [7, 8]

    def unload(patcher, **kwargs):
        assert events[-1][0] == "generation_frame_returned"
        events.append(("unload", patcher, kwargs))

    tail_module.comfy.model_management.load_models_gpu = load_models
    tail_module.comfy.model_management.unload_model_and_clones = unload
    tail_module.comfy.model_management.soft_empty_cache = lambda **_kwargs: None
    monkeypatch.setattr(tail_module, "_generate_tail_tokens", generate)

    result = tail_module.generate_with_tail(clip, "tail.safetensors", {}, {})

    assert result == [7, 8]
    assert wrapper.execution_device == "cpu"
    assert events[0] == (
        "load",
        [base_patcher, tail_patcher],
        {"memory_required": tail_module.RUNTIME_HEADROOM},
    )
    assert events[-1] == (
        "unload",
        tail_patcher,
        {"unload_additional_models": False, "all_devices": True},
    )


def test_tail_managed_unload_runs_after_generation_error(monkeypatch):
    tail_module = _load_tail_module(monkeypatch)
    wrapper = SimpleNamespace(execution_device="cpu")
    base_patcher = SimpleNamespace(load_device="cuda:0", offload_device="cpu")
    tail_patcher = object()
    clip = SimpleNamespace(patcher=base_patcher)
    unloaded = []

    monkeypatch.setattr(
        tail_module, "_get_minimax_base", lambda _clip: (wrapper, object())
    )
    monkeypatch.setattr(
        tail_module,
        "_build_tail",
        lambda _name, _load_device, _offload_device: (object(), tail_patcher),
    )
    tail_module.comfy.model_management.load_models_gpu = lambda *_args, **_kwargs: None
    tail_module.comfy.model_management.unload_model_and_clones = (
        lambda patcher, **kwargs: unloaded.append((patcher, kwargs))
    )
    tail_module.comfy.model_management.soft_empty_cache = lambda **_kwargs: None

    def fail(*_args):
        raise RuntimeError("tail generation failed")

    monkeypatch.setattr(tail_module, "_generate_tail_tokens", fail)
    with pytest.raises(RuntimeError, match="tail generation failed"):
        tail_module.generate_with_tail(clip, "tail.safetensors", {}, {})

    assert wrapper.execution_device == "cpu"
    assert unloaded == [
        (
            tail_patcher,
            {"unload_additional_models": False, "all_devices": True},
        )
    ]


def test_tail_interrupt_traceback_releases_tensor_before_unload(monkeypatch):
    tail_module = _load_tail_module(monkeypatch)
    wrapper = SimpleNamespace(execution_device="cpu")
    base_patcher = SimpleNamespace(load_device="cuda:0", offload_device="cpu")
    tail_patcher = object()
    clip = SimpleNamespace(patcher=base_patcher)
    tensor_ref = None

    class TailInterrupt(BaseException):
        pass

    monkeypatch.setattr(
        tail_module, "_get_minimax_base", lambda _clip: (wrapper, object())
    )
    monkeypatch.setattr(
        tail_module,
        "_build_tail",
        lambda _name, _load_device, _offload_device: (object(), tail_patcher),
    )
    tail_module.comfy.model_management.load_models_gpu = lambda *_args, **_kwargs: None
    tail_module.comfy.model_management.soft_empty_cache = lambda **_kwargs: None

    def interrupt(*_args):
        nonlocal tensor_ref
        temporary = torch.ones(1)
        tensor_ref = weakref.ref(temporary)
        raise TailInterrupt("stop")

    def unload(_patcher, **_kwargs):
        assert tensor_ref is not None
        assert tensor_ref() is None

    monkeypatch.setattr(tail_module, "_generate_tail_tokens", interrupt)
    tail_module.comfy.model_management.unload_model_and_clones = unload

    with pytest.raises(TailInterrupt, match="stop"):
        tail_module.generate_with_tail(clip, "tail.safetensors", {}, {})


def test_tail_cleanup_failure_does_not_mask_generation_error(monkeypatch):
    tail_module = _load_tail_module(monkeypatch)
    wrapper = SimpleNamespace(execution_device="cpu")
    base_patcher = SimpleNamespace(load_device="cuda:0", offload_device="cpu")
    tail_patcher = object()
    clip = SimpleNamespace(patcher=base_patcher)
    cache_flushes = []

    monkeypatch.setattr(
        tail_module, "_get_minimax_base", lambda _clip: (wrapper, object())
    )
    monkeypatch.setattr(
        tail_module,
        "_build_tail",
        lambda _name, _load_device, _offload_device: (object(), tail_patcher),
    )
    tail_module.comfy.model_management.load_models_gpu = lambda *_args, **_kwargs: None
    tail_module.comfy.model_management.unload_model_and_clones = (
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unload failed"))
    )
    tail_module.comfy.model_management.soft_empty_cache = (
        lambda **kwargs: cache_flushes.append(kwargs)
    )

    def fail(*_args):
        raise ValueError("primary generation failure")

    monkeypatch.setattr(tail_module, "_generate_tail_tokens", fail)
    with pytest.raises(ValueError, match="primary generation failure"):
        tail_module.generate_with_tail(clip, "tail.safetensors", {}, {})

    assert cache_flushes == [{"force": True}]
