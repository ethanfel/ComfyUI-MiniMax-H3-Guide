from contextlib import nullcontext
import hashlib
import sys
from types import ModuleType, SimpleNamespace
import weakref

import pytest
import torch

from enhancer import (
    DEFAULT_SYSTEM_PROMPT,
    HISTORICAL_SYSTEM_PROMPT_SHA256,
    NO_TAIL,
    TAIL_TYPE,
    MiniMaxH3GenerationTailLoader,
    MiniMaxH3PromptEnhancer,
    _generate_with_clip,
    _offload_connected_clip,
    build_llm_user_prompt,
    clip_generation_issue,
    clean_generated_prompt,
    enhancer_reference_inventory,
    expected_h3_mode,
    format_chat_prompt,
    generation_collapse_reason,
    h3_structure_warnings,
    reference_context_mode_hint,
    resolve_system_prompt,
    validate_clip_compatibility,
    validate_reference_bindings,
)
from media_context import (
    AUTO_RELATION,
    FIRST_FRAME_ROLE,
    FULL_RELATION,
    IDENTITY_ROLE,
    ITEM_ROLE,
    LAST_FRAME_ROLE,
    MOTION_ROLE,
    PICTURE_MEDIA,
    VIDEO_MEDIA,
    MiniMaxH3EnhancerVisualReference,
    MiniMaxH3VisualReferenceRole,
    reference_entries,
)


VALID_BASE_PROMPT = """integrated_multimodal_description:
[Shot 1] A fox jumps over a fallen branch.

overall_soundscape:
Wind moves through the trees.

non_diegetic_music:
N/A"""


VALID_REFERENCE_PROMPT = """subject_definitions:
<Subject 1> is the red-coated fox derived from <Picture 1>.

summary:
[reference generation] The target video shows <Subject 1> jumping over a branch.

retention_analysis:
<Subject 1>: fully_preserved - retain the fox's visible identity and red coat.

detailed_description:
The target uses a cinematic, naturalistic live-action style.
[Shot 1] <Subject 1>, derived from <Picture 1>, jumps over a branch in a forest clearing.

overall_soundscape:
Wind moves through the trees and paws land on leaves.

non_diegetic_music:
N/A"""


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
    assert "No visual media is attached" in build_llm_user_prompt(
        "draft", has_image=False
    )
    assert "One legacy image is attached" in build_llm_user_prompt(
        "draft", has_image=True
    )
    chained = build_llm_user_prompt(
        "draft", has_image=True, reference_context="<Picture 1>: identity"
    )
    assert "Chained pictures" in chained
    assert "<Picture 1>: identity" in chained


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
    assistant_text = f"<|im_start|>assistant\n{VALID_BASE_PROMPT}"
    assert clean_generated_prompt(assistant_text, "fallback") == VALID_BASE_PROMPT
    assistant_thinking = (
        f"<|im_start|>assistant\n<think>private reasoning</think>\n{VALID_BASE_PROMPT}"
    )
    assert clean_generated_prompt(assistant_thinking, "fallback") == VALID_BASE_PROMPT


def test_enhancer_generates_and_returns_clean_text_artifacts():
    clip = FakeClip(f"```text\n{VALID_BASE_PROMPT}\n```")
    enhanced, system, llm_prompt, report = MiniMaxH3PromptEnhancer().enhance(
        clip=clip, **enhancer_kwargs()
    )
    assert enhanced == VALID_BASE_PROMPT
    assert system == DEFAULT_SYSTEM_PROMPT
    assert "Recommended mode: T2VA" in llm_prompt
    assert clip.generate_call["max_length"] == 1400
    assert clip.generate_call["do_sample"] is True
    assert report == (
        "Resolved H3 output family: T2VA. Enhancement completed successfully with the "
        "connected complete CLIP."
    )


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


def test_historical_system_prompt_hashes_are_exact_and_custom_text_is_preserved():
    assert HISTORICAL_SYSTEM_PROMPT_SHA256 == {
        "2c60735ca7a5ffa8b8195b690f08bad9300ab63d4c4bf73ce3254e840480509a",
        "107b26269e6b23db5c71ecca4593834950dff9ad5b763f5ecdd160e2be02b230",
        "8c1214e96dca59f94a7d18ab87cbc2255f371327d4e68eff660f075eff6357c1",
    }
    current_hash = hashlib.sha256(DEFAULT_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert current_hash not in HISTORICAL_SYSTEM_PROMPT_SHA256
    assert resolve_system_prompt("My custom system rule.") == (
        "My custom system rule.",
        False,
    )


def test_exact_historical_prompt_is_upgraded_and_reported(monkeypatch):
    historical = "A former unedited built-in prompt."
    historical_hash = hashlib.sha256(historical.encode("utf-8")).hexdigest()
    monkeypatch.setattr(
        "enhancer.HISTORICAL_SYSTEM_PROMPT_SHA256",
        frozenset({historical_hash}),
    )
    resolved, migrated = resolve_system_prompt(f"\n{historical}\n")
    assert resolved == DEFAULT_SYSTEM_PROMPT
    assert migrated is True

    _, system, _, report = MiniMaxH3PromptEnhancer().enhance(
        clip=FakeClip(VALID_BASE_PROMPT),
        **enhancer_kwargs(system_prompt=historical),
    )
    assert system == DEFAULT_SYSTEM_PROMPT
    assert "historical built-in system prompt was upgraded" in report


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
    assert "clip_tail" not in schema["required"]
    assert schema["optional"]["clip_tail"][0] == TAIL_TYPE
    assert "Legacy compatibility" in schema["optional"]["image"][1]["tooltip"]
    assert "reference_context" in schema["optional"]
    assert schema["optional"]["offload_after_generation"][1]["default"] is True
    assert (
        "explicitly moves"
        in schema["optional"]["offload_after_generation"][1]["tooltip"]
    )
    assert (
        "Image to Video endpoint inputs"
        in schema["optional"]["reference_context"][1]["tooltip"]
    )
    assert "1000-1400" in schema["required"]["max_new_tokens"][1]["tooltip"]
    assert "upgraded automatically" in schema["required"]["system_prompt"][1]["tooltip"]
    assert "Cleaned candidate" in MiniMaxH3PromptEnhancer.OUTPUT_TOOLTIPS[0]
    assert "Pixel tensors remain external" in MiniMaxH3PromptEnhancer.OUTPUT_TOOLTIPS[2]
    assert "endpoint or Ref2VA pictures" in MiniMaxH3PromptEnhancer.OUTPUT_TOOLTIPS[2]
    assert "structure warnings" in MiniMaxH3PromptEnhancer.OUTPUT_TOOLTIPS[3]
    assert MiniMaxH3PromptEnhancer.RETURN_NAMES == (
        "enhanced_prompt",
        "system_prompt",
        "llm_prompt",
        "enhancer_report",
    )


def test_enhancer_explicitly_offloads_managed_clip_after_decode(monkeypatch):
    calls = []
    model_management = ModuleType("comfy.model_management")

    def unload(patcher, **kwargs):
        calls.append((patcher, kwargs))

    model_management.unload_model_and_clones = unload
    comfy = ModuleType("comfy")
    comfy.__path__ = []
    comfy.model_management = model_management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)

    patcher = SimpleNamespace(load_device="cuda:0", offload_device="cpu")
    clip = FakeClip(VALID_BASE_PROMPT)
    clip.patcher = patcher
    _, _, _, report = MiniMaxH3PromptEnhancer().enhance(clip=clip, **enhancer_kwargs())

    assert calls == [
        (
            patcher,
            {"unload_additional_models": False, "all_devices": True},
        )
    ]
    assert "explicitly moved to its configured offload device" in report


def test_enhancer_offloads_managed_clip_even_when_generation_fails(monkeypatch):
    calls = []
    tensor_ref = None
    model_management = ModuleType("comfy.model_management")

    def unload(patcher, **kwargs):
        assert tensor_ref is not None
        assert tensor_ref() is None
        calls.append((patcher, kwargs))

    model_management.unload_model_and_clones = unload
    comfy = ModuleType("comfy")
    comfy.__path__ = []
    comfy.model_management = model_management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)

    class GenerationInterrupt(BaseException):
        pass

    class FailingClip(FakeClip):
        def generate(self, _tokens, **_kwargs):
            nonlocal tensor_ref
            temporary = torch.ones(1)
            tensor_ref = weakref.ref(temporary)
            raise GenerationInterrupt("generation interrupted")

    patcher = SimpleNamespace(load_device="cuda:0", offload_device="cpu")
    clip = FailingClip("unused")
    clip.patcher = patcher
    with pytest.raises(GenerationInterrupt, match="generation interrupted"):
        MiniMaxH3PromptEnhancer().enhance(clip=clip, **enhancer_kwargs())

    assert calls == [
        (
            patcher,
            {"unload_additional_models": False, "all_devices": True},
        )
    ]


def test_clip_offload_reports_identical_load_and_offload_devices(monkeypatch):
    model_management = ModuleType("comfy.model_management")
    model_management.unload_model_and_clones = lambda *_args, **_kwargs: pytest.fail(
        "same-device residency cannot be offloaded"
    )
    comfy = ModuleType("comfy")
    comfy.__path__ = []
    comfy.model_management = model_management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)

    clip = SimpleNamespace(
        patcher=SimpleNamespace(load_device="cuda:0", offload_device="cuda:0")
    )
    assert _offload_connected_clip(clip) == "same_device"


def test_clip_offload_does_not_swallow_fresh_interrupt(monkeypatch):
    model_management = ModuleType("comfy.model_management")

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt("stop")

    model_management.unload_model_and_clones = interrupt
    comfy = ModuleType("comfy")
    comfy.__path__ = []
    comfy.model_management = model_management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)

    clip = SimpleNamespace(
        patcher=SimpleNamespace(load_device="cuda:0", offload_device="cpu")
    )
    with pytest.raises(KeyboardInterrupt, match="stop"):
        _offload_connected_clip(clip)


def test_default_system_prompt_covers_official_high_risk_rules():
    assert "actual final [Shot N]" in DEFAULT_SYSTEM_PROMPT
    assert "style in one or two English sentences" in DEFAULT_SYSTEM_PROMPT
    assert "says in an off-screen voiceover" in DEFAULT_SYSTEM_PROMPT
    assert "<scenetrans>" in DEFAULT_SYSTEM_PROMPT
    assert "<cutoff>" in DEFAULT_SYSTEM_PROMPT
    assert (
        "Concrete target-frame anchors use keyframe completion" in DEFAULT_SYSTEM_PROMPT
    )
    assert (
        "video editing only when a source video is directly modified"
        in DEFAULT_SYSTEM_PROMPT
    )
    assert "complete final audio track" in DEFAULT_SYSTEM_PROMPT


def test_generation_tail_loader_returns_lightweight_typed_descriptor():
    loader = MiniMaxH3GenerationTailLoader()
    tail_name = "MiniMax-H3/generation_tail_50_63_int8.safetensors"
    (descriptor,) = loader.select_tail(tail_name)
    assert descriptor == {"tail_name": tail_name}
    assert loader.RETURN_TYPES == (TAIL_TYPE,)
    with pytest.raises(RuntimeError, match="No compatible"):
        loader.select_tail(NO_TAIL)


def test_enhancer_analyzes_chained_pictures_and_video_with_generic_qwen():
    context_node = MiniMaxH3EnhancerVisualReference()
    picture, _, _ = context_node.add_reference(
        torch.zeros(1, 64, 64, 3),
        PICTURE_MEDIA,
        IDENTITY_ROLE,
        "Preserve the coat",
        24.0,
        1.0,
        16,
        768,
    )
    context, _, _ = context_node.add_reference(
        torch.zeros(48, 64, 96, 3),
        VIDEO_MEDIA,
        MOTION_ROLE,
        "Copy only the running motion",
        24.0,
        1.0,
        16,
        768,
        picture,
    )
    clip = FakeClip("enhanced prompt")
    _, system_prompt, llm_prompt, report = MiniMaxH3PromptEnhancer().enhance(
        clip=clip, **enhancer_kwargs(reference_context=context)
    )
    tokenized = clip.tokenize_calls[0]
    assert len(tokenized["kwargs"]["images"]) == 3
    assert llm_prompt.count("<|image_pad|>") == 3
    assert (
        "Binding 1: role=Identity or appearance; retention=fully_preserved"
        in llm_prompt
    )
    assert "H3 label policy=derive the requested reusable visible content" in llm_prompt
    assert "Visual inputs 2-3 are chronological samples from <Video 1>" in llm_prompt
    assert "not a requirement to invent a <Subject N>" in system_prompt
    assert "1 chained picture(s) and 1 timestamped reference video(s)" in report


def test_enhancer_inventory_renders_every_binding_for_one_asset():
    role_node = MiniMaxH3VisualReferenceRole()
    identity_roles, _ = role_node.append_role(
        reference_role=IDENTITY_ROLE,
        retention=FULL_RELATION,
        content_group="hero",
        transfer_target="",
        shot_scope="all shots",
        notes="Preserve the face and coat.",
    )
    all_roles, _ = role_node.append_role(
        reference_role=ITEM_ROLE,
        retention=FULL_RELATION,
        content_group="watch",
        transfer_target="",
        shot_scope="Shot 2",
        notes="Preserve the silver watch.",
        previous_roles=identity_roles,
    )
    context, _, _ = MiniMaxH3EnhancerVisualReference().add_reference(
        media=torch.zeros(1, 64, 64, 3),
        media_type=PICTURE_MEDIA,
        reference_role=IDENTITY_ROLE,
        notes="",
        source_fps=24.0,
        analysis_fps=1.0,
        max_analysis_frames=16,
        analysis_long_edge=768,
        role_bindings=all_roles,
    )

    entries = reference_entries(context)
    inventory = enhancer_reference_inventory(entries)
    assert (
        "Binding 1: role=Identity or appearance; retention=fully_preserved" in inventory
    )
    assert "content_group=hero" in inventory
    assert "notes=Preserve the face and coat." in inventory
    assert (
        "Binding 2: role=Object, prop, clothing, interface, or effect; "
        "retention=fully_preserved"
    ) in inventory
    assert "content_group=watch" in inventory
    assert "shot_scope=Shot 2" in inventory

    aliased_entries = [dict(entries[0])]
    aliased_entries[0]["roles"] = aliased_entries[0].pop("bindings")
    aliased_inventory = enhancer_reference_inventory(aliased_entries)
    assert "Binding 1:" in aliased_inventory
    assert "Binding 2:" in aliased_inventory

    clip = FakeClip(VALID_REFERENCE_PROMPT)
    MiniMaxH3PromptEnhancer().enhance(
        clip=clip, **enhancer_kwargs(reference_context=context)
    )
    assert "Binding 1:" in clip.tokenize_calls[0]["text"]
    assert "Binding 2:" in clip.tokenize_calls[0]["text"]


def test_unresolved_attribute_transfer_is_rejected_before_generation():
    transfer = {
        "label": "<Picture 1>",
        "bindings": [
            {
                "role": ITEM_ROLE,
                "retention": "attribute_transfer",
                "content_group": "jacket",
                "transfer_target": "hero",
            }
        ],
    }
    with pytest.raises(ValueError, match="no chained role binding defines that group"):
        validate_reference_bindings([transfer])

    target = {
        "label": "<Picture 2>",
        "roles": [
            {
                "role": IDENTITY_ROLE,
                "retention": FULL_RELATION,
                "content_group": "hero",
            }
        ],
    }
    validate_reference_bindings([transfer, target])


def test_enhancer_uses_native_minimax_reference_payload_for_video_context():
    context, _, _ = MiniMaxH3EnhancerVisualReference().add_reference(
        torch.zeros(48, 64, 96, 3),
        VIDEO_MEDIA,
        MOTION_ROLE,
        "Copy motion",
        24.0,
        1.0,
        16,
        768,
    )
    clip = FakeClip("enhanced prompt", minimax=True)
    _, _, llm_prompt, _ = MiniMaxH3PromptEnhancer().enhance(
        clip=clip, **enhancer_kwargs(reference_context=context)
    )
    tokenized = clip.tokenize_calls[0]
    items = tokenized["kwargs"]["minimax_ref_items"]
    assert len(items) == 1
    assert items[0]["type"] == "video"
    assert items[0]["data"].shape[0] == 4
    assert items[0]["timestamps"] == [0.0, 0.5, 1.0, 1.5]
    assert "<|image_pad|>" not in llm_prompt


def test_minimax_endpoint_context_uses_base_images_not_ref2va_payload():
    role_chain, _ = MiniMaxH3VisualReferenceRole().append_role(
        reference_role=FIRST_FRAME_ROLE,
        retention=AUTO_RELATION,
        content_group="",
        transfer_target="",
        shot_scope="Shot 1",
        notes="Use this exact opening frame.",
    )
    context, _, _ = MiniMaxH3EnhancerVisualReference().add_reference(
        media=torch.zeros(1, 64, 64, 3),
        media_type=PICTURE_MEDIA,
        reference_role=IDENTITY_ROLE,
        notes="",
        source_fps=24.0,
        analysis_fps=1.0,
        max_analysis_frames=16,
        analysis_long_edge=768,
        role_bindings=role_chain,
    )
    output = """For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description:
[Shot 1] Begin from <Picture 1> and show the subject taking one step forward.

overall_soundscape:
One synchronized footstep and quiet room tone.

non_diegetic_music:
N/A"""
    clip = FakeClip(output, minimax=True)
    _, _, llm_prompt, report = MiniMaxH3PromptEnhancer().enhance(
        clip=clip,
        **enhancer_kwargs(
            reference_context=context,
            mode_report="Recommended mode: Ref2VA",
        ),
    )

    entries = reference_entries(context)
    hint = reference_context_mode_hint(context, entries)
    assert hint == "I2VA"
    assert expected_h3_mode("", "Recommended mode: Ref2VA", True, hint) == "I2VA"
    tokenized = clip.tokenize_calls[0]
    assert len(tokenized["kwargs"]["images"]) == 1
    assert "minimax_ref_items" not in tokenized["kwargs"]
    assert "Context route=endpoint; mode hint=I2VA." in llm_prompt
    assert "<|image_pad|>" not in llm_prompt
    assert "Resolved H3 output family: I2VA." in report
    assert "H3 structure warning" not in report


def test_reversed_endpoint_chain_is_tokenized_in_first_then_last_frame_order():
    role_node = MiniMaxH3VisualReferenceRole()
    last_role, _ = role_node.append_role(
        reference_role=LAST_FRAME_ROLE,
        retention=AUTO_RELATION,
        content_group="",
        transfer_target="",
        shot_scope="Shot 2",
        notes="Exact ending.",
    )
    last_context, _, _ = MiniMaxH3EnhancerVisualReference().add_reference(
        media=torch.zeros(1, 32, 32, 3),
        media_type=PICTURE_MEDIA,
        reference_role=IDENTITY_ROLE,
        notes="",
        source_fps=24.0,
        analysis_fps=1.0,
        max_analysis_frames=16,
        analysis_long_edge=768,
        role_bindings=last_role,
    )
    first_role, _ = role_node.append_role(
        reference_role=FIRST_FRAME_ROLE,
        retention=AUTO_RELATION,
        content_group="",
        transfer_target="",
        shot_scope="Shot 1",
        notes="Exact opening.",
    )
    context, _, _ = MiniMaxH3EnhancerVisualReference().add_reference(
        media=torch.ones(1, 32, 32, 3),
        media_type=PICTURE_MEDIA,
        reference_role=IDENTITY_ROLE,
        notes="",
        source_fps=24.0,
        analysis_fps=1.0,
        max_analysis_frames=16,
        analysis_long_edge=768,
        previous_context=last_context,
        role_bindings=first_role,
    )
    output = """How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 2) aligns with the 6.00-second mark of the target video.

integrated_multimodal_description:
[Shot 1] Start exactly from <Picture 1>.
[Shot 2] At 00:03.000 End exactly on <Picture 2>.

overall_soundscape:
Quiet room tone.

non_diegetic_music:
N/A"""
    clip = FakeClip(output, minimax=True)
    MiniMaxH3PromptEnhancer().enhance(
        clip=clip,
        **enhancer_kwargs(reference_context=context),
    )

    images = clip.tokenize_calls[0]["kwargs"]["images"]
    assert len(images) == 2
    assert torch.all(images[0] == 1)
    assert torch.all(images[1] == 0)


def test_enhancer_rejects_legacy_image_and_reference_chain_together():
    context, _, _ = MiniMaxH3EnhancerVisualReference().add_reference(
        torch.zeros(1, 32, 32, 3),
        PICTURE_MEDIA,
        IDENTITY_ROLE,
        "",
        24.0,
        1.0,
        16,
        768,
    )
    with pytest.raises(
        ValueError, match="either reference_context or the legacy image"
    ):
        MiniMaxH3PromptEnhancer().enhance(
            clip=FakeClip("unused"),
            **enhancer_kwargs(
                reference_context=context, image=torch.zeros(1, 32, 32, 3)
            ),
        )


def test_comma_collapse_returns_manual_prompt_with_report():
    clip = FakeClip("," * 500)
    manual = "integrated_multimodal_description: [Shot 1] Keep this draft."
    enhanced, _, _, report = MiniMaxH3PromptEnhancer().enhance(
        clip=clip, **enhancer_kwargs(manual_prompt=manual)
    )
    assert enhanced == manual
    assert "collapsed into punctuation" in report


@pytest.mark.parametrize(
    ("decoded", "expected_reason"),
    [
        ("", "returned no text"),
        (",", "collapsed into punctuation"),
        ("<think>unfinished reasoning", "unfinished reasoning block"),
        ("<think>finished reasoning</think>", "returned no text"),
    ],
)
def test_raw_generation_collapse_returns_manual_prompt(decoded, expected_reason):
    manual = "integrated_multimodal_description: [Shot 1] Preserve this manual draft."
    enhanced, _, _, report = MiniMaxH3PromptEnhancer().enhance(
        clip=FakeClip(decoded),
        **enhancer_kwargs(manual_prompt=manual),
    )
    assert enhanced == manual
    assert expected_reason in report


def test_generation_collapse_detection_keeps_normal_text():
    assert (
        generation_collapse_reason("," * 100) == "the output collapsed into punctuation"
    )
    assert generation_collapse_reason(",") == "the output collapsed into punctuation"
    assert generation_collapse_reason("<think>unfinished") == (
        "the model ended inside an unfinished reasoning block"
    )
    assert (
        generation_collapse_reason(
            "A coherent prompt with varied words and useful punctuation."
        )
        is None
    )


def test_structural_validator_accepts_complete_base_and_ref2va_prompts():
    assert h3_structure_warnings(VALID_BASE_PROMPT, "T2VA") == []
    assert (
        h3_structure_warnings(
            VALID_REFERENCE_PROMPT,
            "Ref2VA",
            ["<Picture 1>"],
        )
        == []
    )


def test_structural_validator_warns_about_sections_from_the_wrong_mode():
    wrong_mode = f"summary:\n[reference generation] Extra.\n\n{VALID_BASE_PROMPT}"
    assert "unexpected section(s) for this mode: summary" in h3_structure_warnings(
        wrong_mode,
        "T2VA",
    )


def test_structural_validator_reports_present_but_empty_required_sections():
    empty_base = """integrated_multimodal_description:

overall_soundscape:

non_diegetic_music:
"""
    warnings = h3_structure_warnings(empty_base, "T2VA")
    assert (
        "empty required section(s): integrated_multimodal_description, overall_soundscape, non_diegetic_music"
        in warnings
    )


def test_structural_validator_accepts_official_bare_fl2va_picture_labels_only():
    fl2va = """How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 7.29-second mark of the target video.

integrated_multimodal_description:
[Shot 1] Begin at Picture 1 and land exactly on Picture 2.

overall_soundscape:
Rain and distant footsteps.

non_diegetic_music:
N/A"""
    assert (
        h3_structure_warnings(
            fl2va,
            "FL2VA",
            ["<Picture 1>", "<Picture 2>"],
        )
        == []
    )

    i2va_with_bare_label = fl2va.replace(
        "How the reference pictures align with the target video —",
        "For the target video, at 0.00 seconds into the target video,",
        1,
    )
    assert (
        "supplied reference label(s) are absent: <Picture 1>"
        in h3_structure_warnings(
            i2va_with_bare_label,
            "I2VA",
            ["<Picture 1>"],
        )
    )


def _retention_marker_prompt(visual_marker: str, audio_marker: str) -> str:
    return f"""subject_definitions:
<Subject 1> is the visible performer.
<Audio 1> is the supplied synchronized audio.

summary:
[reference generation + audio reference] Generate the target from the references.

retention_analysis:
<Subject 1> (appears in [Shot 1]): {visual_marker} - visual relationship.
<Audio 1>: {audio_marker} - audio relationship.

detailed_description:
The target uses a cinematic live-action style.
[Shot 1] <Subject 1> performs while <Audio 1> is heard.

overall_soundscape:
Use the defined synchronized sound.

non_diegetic_music:
N/A"""


def test_structural_validator_enforces_visual_and_audio_retention_markers():
    for visual_marker in (
        "fully_preserved",
        "partially_preserved",
        "attribute_transfer",
        "weak_reference",
    ):
        warnings = h3_structure_warnings(
            _retention_marker_prompt(visual_marker, "reference"),
            "Ref2VA",
        )
        assert not any("retention marker" in warning for warning in warnings)

    for audio_marker in (
        "fully_copy",
        "partially_copy",
        "reference",
        "weak_reference",
    ):
        warnings = h3_structure_warnings(
            _retention_marker_prompt("fully_preserved", audio_marker),
            "Ref2VA",
        )
        assert not any("retention marker" in warning for warning in warnings)

    invalid = h3_structure_warnings(
        _retention_marker_prompt("fully_copy", "fully_preserved"),
        "Ref2VA",
    )
    assert (
        "invalid retention marker(s): <Subject 1>=fully_copy, <Audio 1>=fully_preserved"
        in invalid
    )

    missing = _retention_marker_prompt("fully_preserved", "reference").replace(
        "<Subject 1> (appears in [Shot 1]): fully_preserved",
        "<Subject 1> (appears in [Shot 1]) - fully_preserved",
    )
    assert "retention row(s) missing a recognizable marker: <Subject 1>" in (
        h3_structure_warnings(missing, "Ref2VA")
    )


def test_structural_validator_enforces_fully_copy_audio_section_exclusivity():
    valid = _retention_marker_prompt("fully_preserved", "fully_copy").replace(
        "Use the defined synchronized sound.",
        "<Audio 1> is reused 1:1 as the complete final audio track with no added layers.",
    )
    assert not any(
        "fully_copy audio" in warning
        for warning in h3_structure_warnings(valid, "Ref2VA")
    )

    invalid = _retention_marker_prompt("fully_preserved", "fully_copy")
    warnings = h3_structure_warnings(invalid, "Ref2VA")
    assert (
        "fully_copy audio is not cited in every applicable audio section: <Audio 1>"
        in warnings
    )
    assert (
        "fully_copy audio sections do not state that the copied track remains exclusive"
        in warnings
    )


def test_structural_validator_reports_malformed_ref2va_without_replacing_it():
    malformed = """summary:
[animation] Make a fox video.

subject_definitions:
<Picture 1> is a concrete reference frame.
<Subject 1> is a fox derived from <Picture 1>.

retention_analysis:
<Subject 1>: fully_preserved - keep the fox.
<Subject 1>: fully_preserved - duplicate row.

detailed_description:
[Shot 1] <Subject 1> runs past the unsupplied <Picture 2>.
[Shot 3] At 00:00.700 The fox stops.

overall_soundscape:
Footsteps.

non_diegetic_music:
N/A"""
    warnings = h3_structure_warnings(malformed, "Ref2VA", ["<Picture 1>"])
    report = "; ".join(warnings)
    assert "required sections are out of order" in report
    assert "unsupported task type(s): animation" in report
    assert "missing a retention row: <Picture 1>" in report
    assert "duplicate retention row(s): <Subject 1>" in report
    assert "style opening before [Shot 1]" in report
    assert "shot numbers are not contiguous" in report
    assert "unsupplied visual asset label(s): <Picture 2>" in report

    enhanced, _, _, enhancement_report = MiniMaxH3PromptEnhancer().enhance(
        clip=FakeClip(malformed),
        **enhancer_kwargs(mode_report="Recommended mode: Ref2VA"),
    )
    assert enhanced == malformed
    assert "H3 structure warning" in enhancement_report


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("I2VA", "first-frame alignment instruction"),
        ("FL2VA", "endpoint alignment instruction"),
        ("L2VA", "endpoint alignment instruction"),
    ],
)
def test_structural_validator_warns_when_endpoint_alignment_is_missing(mode, expected):
    assert expected in "; ".join(h3_structure_warnings(VALID_BASE_PROMPT, mode))


def test_clip_compatibility_rejects_known_non_qwen_comfy_tokenizer_early():
    class CLIPTokenizer:
        pass

    CLIPTokenizer.__module__ = "comfy.text_encoders.clip_l"
    clip = FakeClip(VALID_BASE_PROMPT)
    clip.tokenizer = CLIPTokenizer()
    with pytest.raises(RuntimeError, match="formats Qwen chat"):
        MiniMaxH3PromptEnhancer().enhance(
            clip=clip,
            **enhancer_kwargs(reference_context={"not": "a valid context"}),
        )
    assert clip.tokenize_calls == []


def test_clip_compatibility_accepts_qwen_and_tail_backed_wrappers():
    class Qwen3VLTokenizer:
        pass

    Qwen3VLTokenizer.__module__ = "comfy.text_encoders.qwen3_vl"
    clip = FakeClip(VALID_BASE_PROMPT)
    clip.tokenizer = Qwen3VLTokenizer()
    validate_clip_compatibility(clip)

    tail_backed = SimpleNamespace(
        tokenizer=Qwen3VLTokenizer(),
        tokenize=lambda _text, **_kwargs: {},
        decode=lambda _tokens: VALID_BASE_PROMPT,
    )
    validate_clip_compatibility(tail_backed, "generation_tail_50_63.safetensors")
    with pytest.raises(RuntimeError, match="missing generate"):
        validate_clip_compatibility(tail_backed)


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("comfy.text_encoders.qwen35", "Qwen35ImageTokenizer_"),
        ("comfy.text_encoders.qwen3_5", "Qwen3_5Tokenizer"),
    ],
)
def test_clip_compatibility_accepts_comfy_qwen35_tokenizers(module_name, class_name):
    tokenizer_type = type(class_name, (), {})
    tokenizer_type.__module__ = module_name
    clip = FakeClip(VALID_BASE_PROMPT)
    clip.tokenizer = tokenizer_type()

    validate_clip_compatibility(clip)
    enhanced, _system, llm_prompt, report = MiniMaxH3PromptEnhancer().enhance(
        clip=clip,
        **enhancer_kwargs(),
    )

    assert enhanced == VALID_BASE_PROMPT
    assert llm_prompt.startswith("<|im_start|>system")
    assert clip.generate_call is not None
    assert "connected complete CLIP" in report


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
    assert report == f"Resolved H3 output family: T2VA. {issue}"


def test_truncated_h3_clip_uses_selected_tail(monkeypatch):
    class Config:
        num_hidden_layers = 50
        lm_head = False
        final_norm = False

    transformer = SimpleNamespace(model=SimpleNamespace(config=Config()))
    inner = SimpleNamespace(transformer=transformer)
    stage = SimpleNamespace(clip="qwen", qwen=inner)
    clip = FakeClip(
        "integrated_multimodal_description: [Shot 1] Tail result.", minimax=True
    )
    clip.cond_stage_model = stage
    calls = []

    def fake_tail(connected_clip, tail_name, tokens, generation_options):
        calls.append((connected_clip, tail_name, tokens, generation_options))
        return [10, 20, 30]

    monkeypatch.setattr("enhancer._generate_with_tail", fake_tail)
    enhanced, _, _, report = MiniMaxH3PromptEnhancer().enhance(
        clip=clip,
        **enhancer_kwargs(
            clip_tail={"tail_name": "MiniMax-H3/generation_tail_50_63_int8.safetensors"}
        ),
    )
    assert enhanced.endswith("Tail result.")
    assert calls[0][0] is clip
    assert calls[0][1].endswith("generation_tail_50_63_int8.safetensors")
    assert clip.generate_call is None
    assert "temporary MiniMax generation tail" in report
