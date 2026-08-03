from contextlib import nullcontext
import sys
from types import ModuleType, SimpleNamespace

from enhancer import (
    DEFAULT_SYSTEM_PROMPT,
    NO_TAIL,
    MiniMaxH3PromptEnhancer,
    _generate_with_clip,
    build_llm_user_prompt,
    clip_generation_issue,
    clean_generated_prompt,
    format_chat_prompt,
    generation_collapse_reason,
)


class GenericTokenizer:
    pass


class MiniMaxH3Tokenizer:
    pass


class FakeClip:
    def __init__(self, decoded, minimax=False):
        self.decoded = decoded
        self.tokenizer = MiniMaxH3Tokenizer() if minimax else GenericTokenizer()
        self.tokenize_calls = []
        self.generate_call = None

    def tokenize(self, text, **kwargs):
        tokenized = {"text": text, "kwargs": kwargs}
        self.tokenize_calls.append(tokenized)
        return tokenized

    def generate(self, tokens, **kwargs):
        self.generate_call = {"tokens": tokens, **kwargs}
        return [10, 20, 30]

    def decode(self, token_ids):
        assert token_ids == [10, 20, 30]
        return self.decoded


def enhancer_kwargs(**overrides):
    values = {
        "manual_prompt": "integrated_multimodal_description: [Shot 1] A fox jumps.",
        "mode_report": "Recommended mode: T2VA",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "max_new_tokens": 1400,
        "sampling": "sample",
        "temperature": 0.7,
        "top_k": 64,
        "top_p": 0.95,
        "min_p": 0.05,
        "repetition_penalty": 1.05,
        "presence_penalty": 0.0,
        "seed": 12,
        "thinking": False,
        "image": None,
    }
    values.update(overrides)
    return values


def test_user_prompt_reports_optional_image_state():
    assert "No image is attached" in build_llm_user_prompt("draft", has_image=False)
    assert "One image is attached" in build_llm_user_prompt("draft", has_image=True)


def test_generic_qwen_chat_places_visual_token_in_user_turn():
    chat = format_chat_prompt("system", "user", has_image=True, minimax_clip=False)
    assert chat.startswith("<|im_start|>system\nsystem")
    assert "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" in chat


def test_minimax_chat_does_not_duplicate_visual_token():
    chat = format_chat_prompt("system", "user", has_image=True, minimax_clip=True)
    assert "<|vision_start|>" not in chat


def test_clean_generated_prompt_removes_thinking_and_fences():
    text = "<think>private reasoning</think>\n```text\nsummary:\nDone.\n```"
    assert clean_generated_prompt(text, "fallback") == "summary:\nDone."


def test_enhancer_generates_and_returns_clean_text_artifacts():
    clip = FakeClip("```text\nintegrated_multimodal_description: [Shot 1] Enhanced fox.\n```")
    enhanced, system, llm_prompt, report = MiniMaxH3PromptEnhancer().enhance(
        clip=clip, **enhancer_kwargs()
    )
    assert enhanced == "integrated_multimodal_description: [Shot 1] Enhanced fox."
    assert system == DEFAULT_SYSTEM_PROMPT
    assert "Recommended mode: T2VA" in llm_prompt
    assert clip.generate_call["max_length"] == 1400
    assert clip.generate_call["do_sample"] is True
    assert report == "Enhancement completed successfully with the connected complete CLIP."


def test_enhancer_passes_optional_image_to_qwen_tokenizer():
    clip = FakeClip("enhanced", minimax=True)
    image = ["frame zero", "frame one"]
    MiniMaxH3PromptEnhancer().enhance(clip=clip, **enhancer_kwargs(image=image))
    generation_call = clip.tokenize_calls[0]
    assert generation_call["kwargs"]["image"] == ["frame zero"]
    assert generation_call["kwargs"]["images"] == [["frame zero"]]
    assert "<|vision_start|>" not in generation_call["text"]


def test_blank_system_prompt_restores_editable_default():
    clip = FakeClip("enhanced")
    _, system, _, _ = MiniMaxH3PromptEnhancer().enhance(
        clip=clip, **enhancer_kwargs(system_prompt="")
    )
    assert system == DEFAULT_SYSTEM_PROMPT


def test_minimax_image_generation_forwards_qwen_visual_metadata(monkeypatch):
    model_management = ModuleType("comfy.model_management")
    model_management.cuda_device_context = lambda _device: nullcontext()
    comfy = ModuleType("comfy")
    comfy.__path__ = []
    comfy.model_management = model_management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)

    class Transformer:
        def build_image_inputs(self, embeds, embeds_info):
            assert embeds == "embeds"
            assert embeds_info == "embed metadata"
            return "positions", "mask", "deepstack"

        def generate(self, embeds, **kwargs):
            assert embeds == "embeds"
            assert kwargs["position_ids"] == "positions"
            assert kwargs["visual_pos_masks"] == "mask"
            assert kwargs["deepstack_embeds"] == "deepstack"
            return [99]

    class InnerClip:
        transformer = Transformer()

        def process_tokens(self, tokens, device):
            assert tokens == [[151, {"type": "image"}]]
            assert device == "gpu"
            return "embeds", None, None, "embed metadata"

    class Stage:
        clip = "qwen"
        qwen = InnerClip()

        def reset_clip_options(self):
            pass

        def set_clip_options(self, _options):
            pass

    clip = SimpleNamespace(
        cond_stage_model=Stage(),
        patcher=SimpleNamespace(load_device="gpu"),
        load_model=lambda _tokens: None,
    )
    tokens = {"qwen": [[(151, 1.0), ({"type": "image"}, 1.0)]]}
    result = _generate_with_clip(
        clip,
        tokens,
        {"do_sample": False, "max_length": 10},
        use_minimax_image_path=True,
    )
    assert result == [99]


def test_enhancer_tooltips_explain_outputs_and_optional_image_scope():
    schema = MiniMaxH3PromptEnhancer.INPUT_TYPES()
    assert schema["required"]["clip_tail"][0][0] == NO_TAIL
    assert "not automatically an H3 first frame" in schema["optional"]["image"][1]["tooltip"]
    assert "1000-1400" in schema["required"]["max_new_tokens"][1]["tooltip"]
    assert MiniMaxH3PromptEnhancer.RETURN_NAMES == (
        "enhanced_prompt",
        "system_prompt",
        "llm_prompt",
        "enhancer_report",
    )


def test_comma_collapse_returns_manual_prompt_with_report():
    clip = FakeClip("," * 500)
    manual = "integrated_multimodal_description: [Shot 1] Keep this draft."
    enhanced, _, _, report = MiniMaxH3PromptEnhancer().enhance(
        clip=clip, **enhancer_kwargs(manual_prompt=manual)
    )
    assert enhanced == manual
    assert "collapsed into punctuation" in report


def test_generation_collapse_detection_keeps_normal_text():
    assert generation_collapse_reason("," * 100) == "the output collapsed into punctuation"
    assert generation_collapse_reason("A coherent prompt with varied words and useful punctuation.") is None


def test_truncated_h3_conditioning_clip_is_rejected_before_generation():
    class Config:
        num_hidden_layers = 50
        lm_head = False
        final_norm = False

    transformer = SimpleNamespace(model=SimpleNamespace(config=Config()))
    inner = SimpleNamespace(transformer=transformer)
    stage = SimpleNamespace(clip="qwen", qwen=inner)
    clip = FakeClip("should never decode", minimax=True)
    clip.cond_stage_model = stage

    issue = clip_generation_issue(clip)
    assert "conditioning-only" in issue
    enhanced, _, _, report = MiniMaxH3PromptEnhancer().enhance(
        clip=clip, **enhancer_kwargs()
    )
    assert enhanced == enhancer_kwargs()["manual_prompt"]
    assert clip.generate_call is None
    assert report == issue


def test_truncated_h3_clip_uses_selected_tail(monkeypatch):
    class Config:
        num_hidden_layers = 50
        lm_head = False
        final_norm = False

    transformer = SimpleNamespace(model=SimpleNamespace(config=Config()))
    inner = SimpleNamespace(transformer=transformer)
    stage = SimpleNamespace(clip="qwen", qwen=inner)
    clip = FakeClip("integrated_multimodal_description: [Shot 1] Tail result.", minimax=True)
    clip.cond_stage_model = stage
    calls = []

    def fake_tail(connected_clip, tail_name, tokens, generation_options):
        calls.append((connected_clip, tail_name, tokens, generation_options))
        return [10, 20, 30]

    monkeypatch.setattr("enhancer._generate_with_tail", fake_tail)
    enhanced, _, _, report = MiniMaxH3PromptEnhancer().enhance(
        clip=clip,
        **enhancer_kwargs(clip_tail="MiniMax-H3/generation_tail_50_63_int8.safetensors"),
    )
    assert enhanced.endswith("Tail result.")
    assert calls[0][0] is clip
    assert calls[0][1].endswith("generation_tail_50_63_int8.safetensors")
    assert clip.generate_call is None
    assert "temporary MiniMax generation tail" in report
