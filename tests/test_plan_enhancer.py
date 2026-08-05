import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from plan_enhancer import (
    PLAN_ENHANCER_SYSTEM_PROMPT,
    VISUAL_ANALYSIS_DISABLED,
    VISUAL_ANALYSIS_ENABLED,
    MiniMaxH3PlanV2ApplyProse,
    MiniMaxH3PlanV2PromptEnhancer,
    NODE_CLASS_MAPPINGS,
    _analysis_entries,
    _plan_inventory,
    apply_editable_prose,
    editable_prose_json,
    parse_editable_prose,
)
from plan_v2 import (
    AUDIO_VOICE,
    CONTENT_IDENTITY,
    IMAGE_DEFINE_VISIBLE,
    RETENTION_AUTO,
    UNASSIGNED_CONTENT_TYPE,
    VIDEO_STRUCTURE,
    MiniMaxH3PlanV2AudioReference,
    MiniMaxH3PlanV2DialogueEvent,
    MiniMaxH3PlanV2ImageReference,
    MiniMaxH3PlanV2ProjectSetup,
    MiniMaxH3PlanV2PromptMerge,
    MiniMaxH3PlanV2Shot,
    MiniMaxH3PlanV2VideoReference,
)


class GenericTokenizer:
    pass


class FakeClip:
    def __init__(self, decoded):
        self.decoded = decoded
        self.tokenizer = GenericTokenizer()
        self.tokenize_calls = []
        self.generate_call = None

    def tokenize(self, text, **kwargs):
        value = {"text": text, "kwargs": kwargs}
        self.tokenize_calls.append(value)
        return value

    def generate(self, tokens, **kwargs):
        self.generate_call = {"tokens": tokens, **kwargs}
        return [11, 22]

    def decode(self, token_ids):
        assert token_ids == [11, 22]
        return self.decoded


def audio_value(seconds=3.0, sample_rate=8_000):
    return {
        "waveform": torch.zeros(1, 1, round(seconds * sample_rate)),
        "sample_rate": sample_rate,
    }


def compiled_scene(*, include_video=False):
    plan = MiniMaxH3PlanV2ProjectSetup().start(
        "A complete scene unfolds inside a moving truck.",
        6.0,
        "cinematic, grounded live-action",
        "Truck engine, road vibration, and synchronized movement.",
        "N/A",
    )[0]
    plan, _handle, _image, _preview = MiniMaxH3PlanV2ImageReference().add_image(
        plan,
        torch.zeros(1, 48, 64, 3),
        IMAGE_DEFINE_VISIBLE,
        "woman portrait",
        "A dark-haired woman in a denim jacket.",
        CONTENT_IDENTITY,
        "woman",
        RETENTION_AUTO,
        "1-2",
        "",
    )
    if include_video:
        plan, _handle, _video, _preview = MiniMaxH3PlanV2VideoReference().add_video(
            plan,
            torch.zeros(48, 40, 64, 3),
            VIDEO_STRUCTURE,
            "truck camera rhythm",
            "Use only the source camera rhythm and temporal structure.",
            24.0,
            UNASSIGNED_CONTENT_TYPE,
            "",
            "",
            RETENTION_AUTO,
            "",
        )
    plan, _audio, _preview = MiniMaxH3PlanV2AudioReference().add_audio(
        plan,
        audio_value(),
        AUDIO_VOICE,
        "woman voice",
        "woman",
        "",
        "",
        "",
        "Warm, natural delivery.",
        "2",
    )
    plan = MiniMaxH3PlanV2Shot().add_shot(
        plan,
        0.0,
        "<Subject 1> opens the passenger door and enters the truck.",
        "A restrained push-in follows her movement.",
        "Direct cut",
    )[0]
    plan = MiniMaxH3PlanV2Shot().add_shot(
        plan,
        0.4,
        "<Subject 1> settles into the passenger seat and looks toward the driver.",
        "A steady two-shot holds both seats.",
        "Direct cut",
    )[0]
    plan = MiniMaxH3PlanV2DialogueEvent().add_dialogue(
        plan,
        "woman",
        "English",
        "Thanks for the ride.",
        "warmly",
        "On-screen speech",
    )[0]
    return MiniMaxH3PlanV2PromptMerge().merge(plan)[2]


def enhancer_kwargs(**overrides):
    values = {
        "system_prompt": PLAN_ENHANCER_SYSTEM_PROMPT,
        "visual_analysis": VISUAL_ANALYSIS_DISABLED,
        "analysis_long_edge": 512,
        "video_analysis_fps": 1.0,
        "max_analysis_frames": 12,
        "max_new_tokens": 1800,
        "sampling": "sample",
        "temperature": 0.65,
        "top_k": 64,
        "top_p": 0.95,
        "min_p": 0.05,
        "repetition_penalty": 1.02,
        "presence_penalty": 0.0,
        "seed": 9,
        "thinking": False,
        "offload_after_generation": False,
    }
    values.update(overrides)
    return values


def enhanced_json(plan_context):
    payload = json.loads(editable_prose_json(plan_context))
    payload["visual_style"] = "cinematic naturalism with warm midday light"
    payload["shots"][0]["description"] = (
        "<Subject 1> pulls open the passenger door and climbs into the moving truck "
        "with a clear, physically continuous motion."
    )
    payload["shots"][1]["description"] = (
        "<Subject 1> settles into the passenger seat, turns toward the driver, "
        "and holds an attentive expression."
    )
    payload["overall_soundscape"] = (
        "A steady truck engine, subtle road vibration, door movement, and seat creaks."
    )
    return json.dumps(payload)


def test_editable_prose_is_the_complete_scene_without_structural_fields():
    plan = compiled_scene()
    payload = parse_editable_prose(editable_prose_json(plan), plan)

    assert [shot["shot_number"] for shot in payload["shots"]] == [1, 2]
    assert "<Subject 1>" in payload["shots"][0]["description"]
    assert "cut_at" not in payload["shots"][1]
    assert "transition" not in payload["shots"][1]
    assert "Thanks for the ride." not in json.dumps(payload)


def test_apply_prose_reconstructs_and_preserves_every_locked_fact():
    plan = compiled_scene()
    prompt, enhanced_plan, report, length, canonical = apply_editable_prose(
        plan, enhanced_json(plan)
    )

    assert "physically continuous motion" in prompt
    assert "[Shot 2] At 00:00.400" in prompt
    assert "<d>[English] Thanks for the ride.</d>" in prompt
    assert "(S1)" in prompt
    assert enhanced_plan["compiled"] == plan["compiled"]
    assert enhanced_plan["dialogue_events"] == plan["dialogue_events"]
    assert [shot["cut_at"] for shot in enhanced_plan["shots"]] == [0.0, 0.4]
    assert length == 158
    assert "Locked labels, roles, retention, speakers, dialogue, cut times" in report
    assert "Prose delta: 4/8 editable fields changed" in report
    assert json.loads(canonical)["schema_version"] == 1


def test_apply_prose_reports_a_valid_no_op_instead_of_hiding_it():
    plan = compiled_scene()

    _prompt, _enhanced_plan, report, _length, _canonical = apply_editable_prose(
        plan,
        editable_prose_json(plan),
    )

    assert "Prose delta: 0/8 editable fields changed" in report
    assert "valid but unchanged prose patch" in report


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload["shots"][0].update(
                {"description": "[Shot 1] An invalid structural rewrite."}
            ),
            r"\[Shot N\] markers",
        ),
        (
            lambda payload: payload["shots"][1].update({"shot_number": 3}),
            "must keep shot_number 2",
        ),
        (
            lambda payload: payload.update({"unexpected": "field"}),
            "unexpected unexpected",
        ),
    ],
)
def test_structural_json_changes_are_rejected(mutator, message):
    plan = compiled_scene()
    payload = json.loads(editable_prose_json(plan))
    mutator(payload)

    with pytest.raises(ValueError, match=message):
        parse_editable_prose(json.dumps(payload), plan)


def test_removing_a_scoped_subject_label_is_rejected_by_recompiler():
    plan = compiled_scene()
    payload = json.loads(editable_prose_json(plan))
    payload["shots"][0]["description"] = "She opens the passenger door."

    with pytest.raises(ValueError, match="removed.*reference label"):
        apply_editable_prose(plan, json.dumps(payload))


def test_replacing_one_valid_label_with_another_is_still_rejected():
    plan = compiled_scene()
    payload = json.loads(editable_prose_json(plan))
    payload["shots"][0]["description"] = payload["shots"][0]["description"].replace(
        "<Subject 1>", "<Picture 1>"
    )

    with pytest.raises(ValueError, match="renamed.*reference label"):
        apply_editable_prose(plan, json.dumps(payload))


def test_qwen_receives_full_scene_context_with_locked_dialogue_masked():
    plan = compiled_scene()
    clip = FakeClip(enhanced_json(plan))

    (
        prompt,
        prose,
        enhanced_plan,
        base_system,
        effective_system,
        llm_prompt,
        report,
    ) = MiniMaxH3PlanV2PromptEnhancer().enhance_plan(
        clip,
        plan,
        **enhancer_kwargs(),
    )

    assert "VALID COMPILED H3 CONTEXT" in llm_prompt
    assert "[Shot 1]" in llm_prompt
    assert "[Shot 2] At 00:00.400" in llm_prompt
    assert "A compiler-managed dialogue event occurs here" in llm_prompt
    assert "Thanks for the ride." not in llm_prompt
    assert "<d>" not in llm_prompt
    assert "<Audio 1>" in llm_prompt
    assert "EDITABLE PROSE JSON" in llm_prompt
    assert "PRODUCTION DETAIL TARGET" in llm_prompt
    assert "Aim for 80-140 useful words per Shot" in llm_prompt
    assert "physically continuous motion" in prompt
    assert json.loads(prose)["shots"][1]["shot_number"] == 2
    assert enhanced_plan["compiled"] == plan["compiled"]
    assert base_system == PLAN_ENHANCER_SYSTEM_PROMPT
    assert "material production-detail rewrite" in base_system
    assert "When it is blank, you may supply" in base_system
    assert "LOCKED OUTPUT CONTRACT" in effective_system
    assert "text metadata only" in report
    assert "images" not in clip.tokenize_calls[0]["kwargs"]


def test_structured_enhancer_repairs_copied_compiler_dialogue_once():
    plan = compiled_scene()
    payload = json.loads(enhanced_json(plan))
    payload["shots"][1]["description"] += (
        " <Subject 1> (S1) speaks using the voice referenced from <Audio 1>: "
        "<d>[English] Thanks for the ride.</d>. Delivery: warmly."
    )
    clip = FakeClip(json.dumps(payload))

    prompt, prose, _enhanced_plan, *_rest, report = (
        MiniMaxH3PlanV2PromptEnhancer().enhance_plan(
            clip,
            plan,
            **enhancer_kwargs(),
        )
    )

    assert "Structured enhancement fallback" not in report
    assert "copied 1 compiler-owned dialogue segment" in report
    assert prompt.count("<d>[English] Thanks for the ride.</d>") == 1
    assert "physically continuous motion" in prompt
    assert "<d>" not in json.dumps(json.loads(prose))


def test_structured_enhancer_accepts_comfy_qwen35_clip():
    tokenizer_type = type("Qwen35ImageTokenizer_", (), {})
    tokenizer_type.__module__ = "comfy.text_encoders.qwen35"
    plan = compiled_scene()
    clip = FakeClip(enhanced_json(plan))
    clip.tokenizer = tokenizer_type()

    prompt, _prose, _enhanced_plan, *_rest, report = (
        MiniMaxH3PlanV2PromptEnhancer().enhance_plan(
            clip,
            plan,
            **enhancer_kwargs(),
        )
    )

    assert "physically continuous motion" in prompt
    assert clip.generate_call is not None
    assert "connected complete CLIP" in report


def test_qwen_inventory_keeps_binding_notes_retention_scope_and_transfer_fields():
    context = compiled_scene()

    inventory = _plan_inventory(context)

    assert "retention=fully_preserved" in inventory
    assert "shot_scope=1-2" in inventory
    assert "transfer_target=none" in inventory
    assert "notes=A dark-haired woman in a denim jacket." in inventory


def test_invalid_qwen_json_falls_back_to_valid_compiler_outputs():
    plan = compiled_scene()
    clip = FakeClip(
        "This is not JSON, but it is long enough to avoid punctuation collapse."
    )

    prompt, prose, enhanced_plan, *_rest, report = (
        MiniMaxH3PlanV2PromptEnhancer().enhance_plan(
            clip,
            plan,
            **enhancer_kwargs(),
        )
    )

    assert prompt.startswith("subject_definitions:")
    assert json.loads(prose)["schema_version"] == 1
    assert enhanced_plan["compiled"] == plan["compiled"]
    assert "Structured enhancement fallback" in report
    assert "not valid JSON" in report


def test_visual_analysis_attaches_only_images_and_sampled_video_not_audio():
    plan = compiled_scene(include_video=True)
    entries = _analysis_entries(plan, 256, 1.0, 4)

    assert [entry["kind"] for entry in entries] == ["image", "video"]
    assert entries[0]["analysis_media"].shape[0] == 1
    assert entries[1]["analysis_media"].shape[0] <= 4
    assert len(entries[1]["timestamps"]) == entries[1]["analysis_media"].shape[0]

    clip = FakeClip(enhanced_json(plan))
    *_, llm_prompt, report = MiniMaxH3PlanV2PromptEnhancer().enhance_plan(
        clip,
        plan,
        **enhancer_kwargs(
            visual_analysis=VISUAL_ANALYSIS_ENABLED,
            analysis_long_edge=256,
            max_analysis_frames=4,
        ),
    )
    images = clip.tokenize_calls[0]["kwargs"]["images"]
    assert len(images) == 1 + entries[1]["analysis_media"].shape[0]
    assert "audio metadata only" in llm_prompt
    assert "waveform" not in llm_prompt
    assert "Qwen analyzed 1 image(s) and 1 sampled video(s)" in report


def test_phase_two_node_contract():
    assert set(NODE_CLASS_MAPPINGS) == {
        "MiniMaxH3PlanV2ApplyProse",
        "MiniMaxH3PlanV2PromptEnhancer",
    }
    assert MiniMaxH3PlanV2ApplyProse.RETURN_NAMES[0] == "h3_prompt"
    assert MiniMaxH3PlanV2PromptEnhancer.RETURN_NAMES == (
        "enhanced_prompt",
        "editable_prose",
        "enhanced_plan_context",
        "base_system_prompt",
        "effective_system_prompt",
        "llm_prompt",
        "enhancer_report",
    )
    offload = MiniMaxH3PlanV2PromptEnhancer.INPUT_TYPES()["optional"][
        "offload_after_generation"
    ][1]
    assert offload["default"] is False
    assert "never explicitly unloads" in offload["tooltip"]
    required = MiniMaxH3PlanV2PromptEnhancer.INPUT_TYPES()["required"]
    assert required["max_analysis_frames"][1]["default"] == 8
    assert required["max_new_tokens"][1]["default"] == 1200


def test_structured_complete_qwen_never_forces_legacy_unload(monkeypatch):
    model_management = ModuleType("comfy.model_management")
    model_management.unload_model_and_clones = lambda *_args, **_kwargs: pytest.fail(
        "the structured enhancer must not force-unload a connected complete Qwen model"
    )
    comfy = ModuleType("comfy")
    comfy.__path__ = []
    comfy.model_management = model_management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)

    plan = compiled_scene()
    clip = FakeClip(enhanced_json(plan))
    clip.patcher = SimpleNamespace(load_device="cuda:0", offload_device="cpu")
    *_, report = MiniMaxH3PlanV2PromptEnhancer().enhance_plan(
        clip,
        plan,
        **enhancer_kwargs(offload_after_generation=True),
    )

    assert "unload request was suppressed for safety" in report
    assert "ComfyUI manages" in report
