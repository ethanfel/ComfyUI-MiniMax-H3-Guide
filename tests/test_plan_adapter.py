import types

import pytest
import torch

import plan_adapter
from plan_adapter import (
    ENDPOINT_NODE_ID,
    NATIVE_H3_CONTRACT,
    REFERENCE_NODE_ID,
    MiniMaxH3PlanV2ApplyReferencePlan,
    native_h3_compatibility_report,
    prepare_native_h3_call,
)
from plan_v2 import (
    AUDIO_COPY_PARTIAL,
    AUDIO_VOICE,
    CONTENT_IDENTITY,
    IMAGE_DEFINE_VISIBLE,
    IMAGE_FIRST_FRAME,
    IMAGE_LAST_FRAME,
    RETENTION_AUTO,
    UNASSIGNED_CONTENT_TYPE,
    VIDEO_EDIT,
    MiniMaxH3PlanV2AudioReference,
    MiniMaxH3PlanV2DialogueEvent,
    MiniMaxH3PlanV2FoleyTarget,
    MiniMaxH3PlanV2ImageReference,
    MiniMaxH3PlanV2ProjectSetup,
    MiniMaxH3PlanV2Shot,
    MiniMaxH3PlanV2VideoReference,
    compile_h3_plan,
)


def project(prompt="A grounded cinematic scene unfolds."):
    return MiniMaxH3PlanV2ProjectSetup().start(
        prompt,
        6.0,
        "cinematic, live-action",
        "Natural synchronized ambience.",
        "N/A",
    )[0]


def image_reference(plan, use, *, name="reference", subject=""):
    defines_subject = use == IMAGE_DEFINE_VISIBLE
    return MiniMaxH3PlanV2ImageReference().add_image(
        plan,
        torch.zeros(1, 32, 48, 3),
        use,
        name,
        "Stable visible reference facts.",
        CONTENT_IDENTITY if defines_subject else UNASSIGNED_CONTENT_TYPE,
        subject if defines_subject else "",
        RETENTION_AUTO,
        "",
        "",
    )


def video_reference(plan):
    return MiniMaxH3PlanV2VideoReference().add_video(
        plan,
        torch.zeros(48, 32, 48, 3),
        VIDEO_EDIT,
        "source video",
        "The source video whose visible timeline is edited.",
        24.0,
        UNASSIGNED_CONTENT_TYPE,
        "",
        "",
        RETENTION_AUTO,
        "",
    )


def audio_value(seconds=3.0):
    return {
        "waveform": torch.zeros(1, 1, round(seconds * 32_000)),
        "sample_rate": 32_000,
    }


def compile_plan(plan):
    prompt, _rewrite, _report, compiled, length = compile_h3_plan(plan)
    return prompt, compiled, length


def compiled_foley():
    plan = project("Generate realistic Foley synchronized to the locked picture track.")
    plan = MiniMaxH3PlanV2FoleyTarget().set_foley_target(
        plan,
        torch.zeros(144, 32, 32, 3),
        24.0,
    )[0]
    plan = MiniMaxH3PlanV2Shot().add_shot(
        plan,
        0.0,
        "A person walks across the room; every footfall is synchronized to contact.",
        "",
        "Direct cut",
    )[0]
    return compile_plan(plan)


class FakeNested:
    is_nested = True

    def __init__(self, streams):
        self.streams = tuple(streams)

    def unbind(self):
        return self.streams


def mixed_reference_package():
    plan = project("Edit the source video while preserving the referenced woman.")
    plan, _image_handle, _image, _preview = image_reference(
        plan,
        IMAGE_DEFINE_VISIBLE,
        name="woman portrait",
        subject="woman",
    )
    plan, video_handle, _video, _preview = video_reference(plan)
    plan, _audio, _preview = MiniMaxH3PlanV2AudioReference().add_audio(
        plan,
        audio_value(),
        AUDIO_COPY_PARTIAL,
        "source music",
        "",
        "",
        "",
        "",
        "Copy only the background music from 00:00-00:02.",
        "",
        video_handle,
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
        "Warm, conversational delivery.",
        "",
        None,
    )
    plan = MiniMaxH3PlanV2Shot().add_shot(
        plan,
        0.0,
        "<Subject 1> sits inside the moving source scene.",
        "A steady medium shot.",
        "Direct cut",
    )[0]
    plan = MiniMaxH3PlanV2DialogueEvent().add_dialogue(
        plan,
        "woman",
        "English",
        "Where are we going?",
        "Warm and curious.",
        "On-screen speech",
    )[0]
    return compile_plan(plan)


def test_text_only_plan_prepares_native_endpoint_call():
    prompt, compiled, length = compile_plan(project("A fox crosses fresh snow."))

    package = prepare_native_h3_call(compiled, prompt, 1344, 768)

    assert package["node_id"] == ENDPOINT_NODE_ID
    assert package["mode"] == "T2VA"
    assert package["checkpoint"] == "H3-Base-FL2VA"
    assert package["h3_length"] == length == 158
    assert package["kwargs"]["first_frame"] is None
    assert package["kwargs"]["last_frame"] is None
    assert package["needs_audio_vae"] is False
    assert "Applied routes:\n- none" in package["report"]


def test_foley_package_keeps_source_out_of_reference_routes():
    prompt, compiled, length = compiled_foley()

    package = prepare_native_h3_call(compiled, prompt, 32, 32)

    assert package["node_id"] == ENDPOINT_NODE_ID
    assert package["mode"] == "T2VA"
    assert package["h3_length"] == length == 158
    assert package["target_task"] == plan_adapter.TARGET_FOLEY
    assert package["target"]["media"].shape[0] == length
    assert package["kwargs"]["first_frame"] is None
    assert package["kwargs"]["last_frame"] is None
    assert "source video VAE latent with mask 0" in package["report"]
    assert "Applied routes:\n- none" in package["report"]


def test_foley_latent_preserves_video_and_generates_audio_masks():
    encoded_video = torch.randn(1, 24, 47, 2, 2)
    empty_audio = torch.zeros(1, 32, 2, 264)
    native = {
        "samples": FakeNested((torch.zeros_like(encoded_video), empty_audio)),
        "batch_index": [0],
    }

    result = plan_adapter._masked_foley_latent(native, encoded_video)
    video, audio = result["samples"].unbind()
    video_mask, audio_mask = result["noise_mask"].unbind()

    assert video is encoded_video
    assert audio is empty_audio
    assert torch.count_nonzero(video_mask) == 0
    assert torch.all(audio_mask == 1)
    assert result["batch_index"] == [0]


def test_first_and_last_frames_are_routed_without_reference_dicts():
    plan = project("Move between the supplied endpoint compositions.")
    plan, _handle, first, _preview = image_reference(
        plan, IMAGE_FIRST_FRAME, name="opening"
    )
    plan, _handle, last, _preview = image_reference(
        plan, IMAGE_LAST_FRAME, name="ending"
    )
    prompt, compiled, _length = compile_plan(plan)

    package = prepare_native_h3_call(compiled, prompt, 768, 1344)

    assert package["mode"] == "FL2VA"
    assert package["kwargs"]["first_frame"] is first
    assert package["kwargs"]["last_frame"] is last
    assert "<Picture 1> -> first_frame" in package["report"]
    assert "<Picture 2> -> last_frame" in package["report"]


def test_ref2va_routes_media_in_the_native_named_collections():
    prompt, compiled, _length = mixed_reference_package()

    package = prepare_native_h3_call(compiled, prompt, 1344, 768, "max")

    assert package["node_id"] == REFERENCE_NODE_ID
    assert package["mode"] == "Ref2VA"
    assert package["checkpoint"] == "H3-Base-Ref2VA"
    assert package["needs_audio_vae"] is True
    assert list(package["kwargs"]["ref_images"]) == ["ref_image_0"]
    assert list(package["kwargs"]["ref_video_audios"]) == ["ref_video_audio_0"]
    assert list(package["kwargs"]["ref_videos"]) == ["ref_video_0"]
    assert list(package["kwargs"]["ref_audios"]) == ["ref_audio_0"]
    assert package["kwargs"]["ref_image_size"] == "max"
    assert "<Audio 1> -> ref_video_audio_0" in package["report"]
    assert "<Audio 2> -> ref_audio_0" in package["report"]


def test_prompt_and_context_must_be_from_the_same_compiler_result():
    prompt, compiled, _length = compile_plan(project())

    with pytest.raises(ValueError, match="does not match the connected plan_context"):
        prepare_native_h3_call(compiled, prompt + "\nmanual mutation", 1344, 768)


@pytest.mark.parametrize(
    ("width", "height"),
    [(0, 768), (1343, 768), (1344, 767), ("not-a-size", 768)],
)
def test_invalid_native_target_sizes_are_rejected(width, height):
    prompt, compiled, _length = compile_plan(project())

    with pytest.raises(ValueError, match="width and height"):
        prepare_native_h3_call(compiled, prompt, width, height)


def test_adapter_delegates_to_native_class_and_returns_its_two_outputs(monkeypatch):
    prompt, compiled, length = compile_plan(project())
    captured = {}

    class FakeNative:
        @classmethod
        def execute(cls, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(result=("conditioning", {"samples": "latent"}))

    monkeypatch.setattr(plan_adapter, "_native_h3_class", lambda _node_id: FakeNative)

    result = MiniMaxH3PlanV2ApplyReferencePlan().apply_reference_plan(
        "clip",
        "video-vae",
        prompt,
        compiled,
        1344,
        768,
        "match",
    )

    assert result[:2] == ("conditioning", {"samples": "latent"})
    assert result[2] == prompt
    assert result[3] == length
    assert captured["clip"] == "clip"
    assert captured["vae"] == "video-vae"
    assert captured["length"] == length


def test_adapter_replaces_native_empty_latent_for_foley(monkeypatch):
    prompt, compiled, length = compiled_foley()
    native_latent = {"samples": "empty-av"}
    captured = {}

    class FakeNative:
        @classmethod
        def execute(cls, **kwargs):
            return types.SimpleNamespace(result=("conditioning", native_latent))

    def fake_foley(target, latent, vae, width, height):
        captured.update(
            target=target,
            latent=latent,
            vae=vae,
            width=width,
            height=height,
        )
        return {"samples": "masked-foley"}

    monkeypatch.setattr(plan_adapter, "_native_h3_class", lambda _node_id: FakeNative)
    monkeypatch.setattr(plan_adapter, "_foley_target_latent", fake_foley)

    result = MiniMaxH3PlanV2ApplyReferencePlan().apply_reference_plan(
        "clip",
        "video-vae",
        prompt,
        compiled,
        32,
        32,
        "match",
    )

    assert result[0] == "conditioning"
    assert result[1] == {"samples": "masked-foley"}
    assert result[3] == length
    assert captured["latent"] is native_latent
    assert captured["vae"] == "video-vae"
    assert (captured["width"], captured["height"]) == (32, 32)


def test_reference_audio_requires_audio_vae_before_native_execution(monkeypatch):
    prompt, compiled, _length = mixed_reference_package()
    monkeypatch.setattr(
        plan_adapter,
        "_native_h3_class",
        lambda _node_id: pytest.fail("native class should not be loaded"),
    )

    with pytest.raises(ValueError, match="Connect the MiniMax H3 audio VAE"):
        MiniMaxH3PlanV2ApplyReferencePlan().apply_reference_plan(
            "clip",
            "video-vae",
            prompt,
            compiled,
            1344,
            768,
            "match",
        )


def test_current_native_signature_contract_is_checked_without_loading_weights(
    monkeypatch,
):
    class FakeEndpoint:
        @classmethod
        def execute(
            cls,
            clip,
            vae,
            prompt,
            width,
            height,
            length,
            first_frame=None,
            last_frame=None,
        ):
            return None

    class FakeReference:
        @classmethod
        def execute(
            cls,
            clip,
            vae,
            audio_vae,
            prompt,
            width,
            height,
            length,
            ref_image_size="match",
            ref_images=None,
            ref_videos=None,
            ref_video_audios=None,
            ref_audios=None,
        ):
            return None

    fake_module = types.SimpleNamespace(
        MiniMaxH3ImageToVideo=FakeEndpoint,
        MiniMaxH3ReferenceToVideo=FakeReference,
    )
    monkeypatch.setattr(
        plan_adapter.importlib, "import_module", lambda _name: fake_module
    )

    report = native_h3_compatibility_report()

    assert NATIVE_H3_CONTRACT in report
    assert ENDPOINT_NODE_ID in report
    assert REFERENCE_NODE_ID in report


def test_changed_native_signature_has_an_actionable_compatibility_error(monkeypatch):
    class ChangedEndpoint:
        @classmethod
        def execute(cls, clip, vae, prompt):
            return None

    fake_module = types.SimpleNamespace(MiniMaxH3ImageToVideo=ChangedEndpoint)
    monkeypatch.setattr(
        plan_adapter.importlib, "import_module", lambda _name: fake_module
    )

    with pytest.raises(RuntimeError, match="incompatible API"):
        plan_adapter._native_h3_class(ENDPOINT_NODE_ID)


def test_adapter_node_contract_keeps_audio_vae_optional():
    schema = MiniMaxH3PlanV2ApplyReferencePlan.INPUT_TYPES()

    assert list(schema["optional"]) == ["audio_vae"]
    assert schema["required"]["h3_prompt"][1]["forceInput"] is True
    assert MiniMaxH3PlanV2ApplyReferencePlan.RETURN_NAMES[:2] == (
        "positive",
        "latent",
    )
