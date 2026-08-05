import pytest
import torch

from plan_v2 import (
    AUDIO_BEAT,
    AUDIO_BROAD,
    AUDIO_CONTENT,
    AUDIO_CONTINUITY,
    AUDIO_COPY_COMPLETE,
    AUDIO_COPY_PARTIAL,
    AUDIO_MUSIC,
    AUDIO_SFX,
    AUDIO_VOICE,
    CONTENT_ACTION,
    CONTENT_IDENTITY,
    CONTENT_OBJECT,
    IMAGE_DEFINE_VISIBLE,
    IMAGE_FIRST_FRAME,
    IMAGE_LAST_FRAME,
    IMAGE_STORYBOARD,
    PLAN_TYPE,
    RETENTION_AUTO,
    RETENTION_TRANSFER,
    UNASSIGNED_CONTENT_TYPE,
    VIDEO_DEFINE_VISIBLE,
    VIDEO_EDIT,
    VIDEO_MOTION,
    VIDEO_STRUCTURE,
    MiniMaxH3PlanV2AudioReference,
    MiniMaxH3PlanV2DialogueEvent,
    MiniMaxH3PlanV2ImageReference,
    MiniMaxH3PlanV2ProjectSetup,
    MiniMaxH3PlanV2PromptMerge,
    MiniMaxH3PlanV2Shot,
    MiniMaxH3PlanV2SubjectBinding,
    MiniMaxH3PlanV2VideoReference,
    NODE_CLASS_MAPPINGS,
    compile_h3_plan,
)


def project(
    prompt="A grounded scene unfolds inside a moving truck.",
    *,
    duration=6.0,
    soundscape="Truck engine and road vibration.",
    music="N/A",
):
    return MiniMaxH3PlanV2ProjectSetup().start(
        prompt,
        duration,
        "cinematic, live-action",
        soundscape,
        music,
    )[0]


def image_reference(
    plan,
    *,
    use=IMAGE_DEFINE_VISIBLE,
    name="woman portrait",
    description="A woman with dark hair and a denim jacket.",
    content_type=CONTENT_IDENTITY,
    subject="woman",
    retention=RETENTION_AUTO,
    scope="",
    transfer_target="",
    value=None,
):
    image = torch.zeros(1, 32, 48, 3) if value is None else value
    return MiniMaxH3PlanV2ImageReference().add_image(
        plan,
        image,
        use,
        name,
        description,
        content_type,
        subject,
        retention,
        scope,
        transfer_target,
    )


def video_reference(
    plan,
    *,
    use=VIDEO_EDIT,
    name="source truck video",
    description="The supplied moving-truck source video.",
    frames=None,
):
    video = torch.zeros(48, 32, 48, 3) if frames is None else frames
    return MiniMaxH3PlanV2VideoReference().add_video(
        plan,
        video,
        use,
        name,
        description,
        24.0,
        UNASSIGNED_CONTENT_TYPE,
        "",
        "",
        RETENTION_AUTO,
        "",
    )


def audio_value(seconds=3.0, sample_rate=32_000):
    return {
        "waveform": torch.zeros(1, 1, round(seconds * sample_rate)),
        "sample_rate": sample_rate,
    }


def audio_reference(
    plan,
    *,
    use,
    name,
    speaker="",
    language="",
    transcript="",
    layer="",
    instructions="",
    scope="",
    paired_video=None,
    seconds=3.0,
):
    return MiniMaxH3PlanV2AudioReference().add_audio(
        plan,
        audio_value(seconds),
        use,
        name,
        speaker,
        language,
        transcript,
        layer,
        instructions,
        scope,
        paired_video,
    )


def shot(plan, cut, description, transition="Direct cut"):
    return MiniMaxH3PlanV2Shot().add_shot(
        plan,
        cut,
        description,
        "",
        transition,
    )[0]


def dialogue(
    plan,
    speaker,
    words,
    *,
    language="English",
    delivery="natural",
    voice_mode="On-screen speech",
):
    return MiniMaxH3PlanV2DialogueEvent().add_dialogue(
        plan,
        speaker,
        language,
        words,
        delivery,
        voice_mode,
    )[0]


def test_project_and_text_only_compile_to_native_t2va():
    plan, h3_length, preview = MiniMaxH3PlanV2ProjectSetup().start(
        "A fox walks across fresh snow.",
        6.0,
        "cinematic",
        "Soft wind and footsteps.",
        "N/A",
    )
    prompt, rewrite, report, compiled, merged_length = compile_h3_plan(plan)

    assert h3_length == merged_length == 158
    assert "native 158 frames" in preview
    assert prompt.startswith("integrated_multimodal_description: [Shot 1]")
    assert "overall_soundscape: Soft wind and footsteps." in prompt
    assert "subject_definitions:" not in prompt
    assert "Mode: T2VA" in report
    assert "Checkpoint: H3-Base-FL2VA" in report
    assert compiled["phase"] == "compiled"
    assert compiled["compiled"]["mode"] == "T2VA"
    assert compiled["compiled"]["checkpoint"] == "H3-Base-FL2VA"
    assert "Do not add, remove, rename, or renumber" in rewrite


def test_endpoint_image_does_not_create_a_subject():
    plan = project("Animate the supplied opening composition.")
    plan, _handle, _image, _preview = image_reference(
        plan,
        use=IMAGE_FIRST_FRAME,
        content_type=UNASSIGNED_CONTENT_TYPE,
        subject="",
    )
    prompt, _rewrite, report, compiled, _length = compile_h3_plan(plan)

    assert prompt.startswith(
        "For the target video, at 0.00 seconds into the target video, "
        "<Picture 1> (from [Shot 1]) is fully referenced."
    )
    assert "<Subject 1>" not in prompt
    assert "Mode: I2VA" in report
    assert "Checkpoint: H3-Base-FL2VA" in report
    assert compiled["compiled"]["routes"][0]["route"] == "first_frame"


def test_first_and_last_endpoints_compile_to_fl2va_without_subjects():
    plan = project("Connect the supplied opening and ending compositions.")
    plan, _handle, _image, _preview = image_reference(
        plan,
        use=IMAGE_FIRST_FRAME,
        name="opening",
        content_type=UNASSIGNED_CONTENT_TYPE,
        subject="",
    )
    plan, _handle, _image, _preview = image_reference(
        plan,
        use=IMAGE_LAST_FRAME,
        name="ending",
        content_type=UNASSIGNED_CONTENT_TYPE,
        subject="",
    )
    plan = shot(plan, 0.0, "The opening composition begins to move.")
    plan = shot(plan, 3.0, "The action resolves into the ending composition.")

    prompt, _rewrite, report, compiled, _length = compile_h3_plan(plan)

    assert prompt.startswith("How the reference pictures align with the target video")
    assert "<Picture 1>" in prompt and "<Picture 2>" in prompt
    assert "6.58-second mark" in prompt
    assert "<Subject" not in prompt
    assert "Mode: FL2VA" in report
    assert [route["route"] for route in compiled["compiled"]["routes"]] == [
        "first_frame",
        "last_frame",
    ]


def test_endpoint_roles_cannot_be_silently_downgraded_to_ref2va_images():
    plan = project("Start from the opening frame, then preserve the referenced woman.")
    plan, _handle, _image, _preview = image_reference(
        plan,
        use=IMAGE_FIRST_FRAME,
        name="opening",
        content_type=UNASSIGNED_CONTENT_TYPE,
        subject="",
    )
    plan, _handle, _image, _preview = image_reference(
        plan,
        name="woman portrait",
        subject="woman",
    )

    with pytest.raises(ValueError, match="cannot be mixed.*Ref2VA role"):
        compile_h3_plan(plan)


def test_exact_endpoint_handle_rejects_subject_binding_before_merge():
    plan = project("Animate the supplied opening frame.")
    plan, handle, _image, _preview = image_reference(
        plan,
        use=IMAGE_FIRST_FRAME,
        name="opening",
        content_type=UNASSIGNED_CONTENT_TYPE,
        subject="",
    )

    with pytest.raises(ValueError, match="cannot receive Subject Bindings"):
        MiniMaxH3PlanV2SubjectBinding().bind_subject(
            plan,
            handle,
            "woman",
            CONTENT_IDENTITY,
            RETENTION_AUTO,
            "",
            "Identity from the opening frame.",
            "",
        )


def test_direct_picture_role_cannot_claim_an_unbound_attribute_transfer():
    with pytest.raises(ValueError, match="attribute_transfer requires Define reusable"):
        image_reference(
            project(),
            use=IMAGE_STORYBOARD,
            content_type=UNASSIGNED_CONTENT_TYPE,
            subject="",
            retention=RETENTION_TRANSFER,
        )


def test_voice_reference_is_exactly_bound_to_subject_and_dialogue_order():
    plan = project()
    plan, _handle, _image, _preview = image_reference(plan, scope="1-2")
    plan, _audio, _preview = audio_reference(
        plan,
        use=AUDIO_VOICE,
        name="woman voice",
        speaker="woman",
        instructions="Calm but slightly breathless delivery.",
        scope="1",
    )
    plan = shot(
        plan,
        0.0,
        "Inside the truck, <Subject 1> looks toward the driver.",
    )
    plan = dialogue(plan, "woman", "Thanks for stopping for me.")
    plan = shot(
        plan,
        2.8,
        "A close shot holds on <Subject 1> as the truck continues.",
    )

    prompt, _rewrite, report, compiled, _length = compile_h3_plan(plan)

    assert (
        "<Audio 1> is the voice-timbre and delivery reference for "
        "<Subject 1> (S1); do not reuse its source words."
    ) in prompt
    assert (
        "<Subject 1> (S1) speaks using the voice timbre and delivery "
        "referenced from <Audio 1>: "
        "<d>[English] Thanks for stopping for me.</d>"
    ) in prompt
    assert "voice, music, beat, or sound" not in prompt
    assert "<Subject 1> (appears in [Shot 1] and [Shot 2])" in prompt
    assert compiled["compiled"]["speaker_ids"] == {"woman": "S1"}
    assert "ref_audio_0" in report


def test_mixed_paired_music_and_standalone_voice_follow_native_label_order():
    plan = project()
    plan, _picture_handle, _image, _preview = image_reference(plan)
    plan, video_handle, h3_video, _preview = video_reference(plan)
    assert h3_video.shape[0] == 39

    plan, _paired_audio, _preview = audio_reference(
        plan,
        use=AUDIO_COPY_PARTIAL,
        name="source music layer",
        instructions="Copy only the background-music layer from 00:00-00:02.",
        paired_video=video_handle,
    )
    plan, _voice_audio, _preview = audio_reference(
        plan,
        use=AUDIO_VOICE,
        name="woman voice",
        speaker="woman",
        instructions="Warm conversational delivery.",
    )
    plan = shot(
        plan,
        0.0,
        "<Subject 1> sits inside the moving truck while the source edit continues.",
    )
    plan = dialogue(plan, "woman", "Where are we going?")

    prompt, _rewrite, report, compiled, _length = compile_h3_plan(plan)
    routes = compiled["compiled"]["routes"]

    assert [entry["label"] for entry in routes] == [
        "<Picture 1>",
        "<Audio 1>",
        "<Video 1>",
        "<Audio 2>",
    ]
    assert [entry["route"] for entry in routes] == [
        "ref_image_0",
        "ref_video_audio_0",
        "ref_video_0",
        "ref_audio_0",
    ]
    assert "<Audio 1> provides only the selected copied range or layers" in prompt
    assert (
        "<Audio 2> is the voice-timbre and delivery reference for <Subject 1> (S1)"
    ) in prompt
    assert (
        "[reference generation + video editing + audio reuse + audio reference]"
        in prompt
    )
    assert "ref_video_audio_0" in report


@pytest.mark.parametrize(
    ("audio_use", "expected_definition", "expected_retention"),
    [
        (AUDIO_MUSIC, "is the background-music style reference", "reference"),
        (AUDIO_BEAT, "is the beat-and-rhythm reference", "reference"),
        (AUDIO_SFX, "is the sound-effect texture reference", "reference"),
        (AUDIO_CONTINUITY, "is the audio-continuity reference", "reference"),
        (AUDIO_BROAD, "is a weak broad audio-inspiration reference", "weak_reference"),
    ],
)
def test_each_nonverbal_audio_role_gets_its_own_definition(
    audio_use,
    expected_definition,
    expected_retention,
):
    plan = project()
    plan, _handle, _image, _preview = image_reference(plan)
    plan, _audio, _preview = audio_reference(
        plan,
        use=audio_use,
        name="sonic source",
        layer="the target scene's declared sound layer",
        instructions="Use only the selected characteristic.",
    )
    plan = shot(plan, 0.0, "<Subject 1> sits inside the moving truck.")

    prompt, _rewrite, _report, _compiled, _length = compile_h3_plan(plan)

    assert f"<Audio 1> {expected_definition}" in prompt
    assert f"<Audio 1>: {expected_retention}" in prompt


def test_irrelevant_audio_metadata_is_removed_when_the_exact_role_changes():
    plan = project()
    plan, _handle, _image, _preview = image_reference(plan)
    plan, _audio, _preview = MiniMaxH3PlanV2AudioReference().add_audio(
        plan,
        audio_value(),
        AUDIO_MUSIC,
        "music style",
        "stale speaker",
        "stale language",
        "stale transcript",
        "the non-diegetic score",
        "Use restrained instrumentation.",
        "",
    )

    relationship = plan["audio_relationships"][0]
    assert relationship["target_speaker"] == ""
    assert relationship["language"] == ""
    assert relationship["transcript"] == ""
    assert relationship["target_layer_or_event"] == "the non-diegetic score"


def test_dialogue_content_requires_and_matches_structured_vocal_event():
    plan = project()
    plan, _handle, _image, _preview = image_reference(plan)
    plan, _audio, _preview = audio_reference(
        plan,
        use=AUDIO_CONTENT,
        name="spoken content",
        speaker="woman",
        language="French",
        transcript="Merci de vous être arrêté.",
    )
    plan = shot(plan, 0.0, "<Subject 1> turns toward the driver.")
    plan = dialogue(
        plan,
        "woman",
        "Merci de vous être arrêté.",
        language="French",
    )

    prompt, _rewrite, _report, _compiled, _length = compile_h3_plan(plan)

    assert (
        "<Audio 1> provides the referenced spoken or lyric content for <Subject 1> (S1)"
    ) in prompt
    assert "<d>[French] Merci de vous être arrêté.</d>" in prompt


def test_valid_complete_audio_copy_controls_both_audio_sections():
    plan = project(soundscape="", music="N/A")
    plan, _handle, _image, _preview = image_reference(plan)
    plan, _audio, _preview = audio_reference(
        plan,
        use=AUDIO_COPY_COMPLETE,
        name="complete source mix",
    )
    plan = shot(plan, 0.0, "<Subject 1> walks through the source scene.")

    prompt, _rewrite, _report, _compiled, _length = compile_h3_plan(plan)

    assert "<Audio 1> is reused as the complete final audio track" in prompt
    assert "Reuse <Audio 1> as the complete final audio track" in prompt
    assert "Contained entirely in <Audio 1>" in prompt


def test_speaker_ids_come_from_vocal_events_not_subject_numbering():
    plan = project()
    plan, _woman_handle, _image, _preview = image_reference(
        plan,
        name="woman",
        subject="woman",
    )
    plan, _man_handle, _image, _preview = image_reference(
        plan,
        name="driver",
        description="An older male truck driver.",
        subject="driver",
    )
    plan, _audio, _preview = audio_reference(
        plan,
        use=AUDIO_VOICE,
        name="woman voice",
        speaker="woman",
    )
    plan = shot(
        plan,
        0.0,
        "<Subject 1> sits beside <Subject 2> in the moving truck.",
    )
    plan = dialogue(plan, "driver", "Long way from home?")
    plan = dialogue(plan, "woman", "A little.")

    prompt, _rewrite, _report, compiled, _length = compile_h3_plan(plan)

    assert "<Subject 2> (S1) speaks" in prompt
    assert "<Subject 1> (S2) speaks using the voice timbre" in prompt
    assert "voice-timbre and delivery reference for <Subject 1> (S2)" in prompt
    assert compiled["compiled"]["speaker_ids"] == {
        "driver": "S1",
        "woman": "S2",
    }


def test_numeric_scope_needs_label_only_in_the_scoped_shots():
    plan = project("A four-shot product reveal.")
    plan, _handle, _image, _preview = image_reference(
        plan,
        name="watch",
        description="A silver wristwatch.",
        content_type=CONTENT_OBJECT,
        subject="watch",
        scope="3,4",
    )
    plan = shot(plan, 0.0, "A person enters an empty studio.")
    plan = shot(plan, 1.0, "The camera moves toward a table.")
    plan = shot(plan, 2.0, "The person places <Subject 1> on the table.")
    plan = shot(plan, 3.0, "A macro view holds on <Subject 1>.")

    prompt, _rewrite, _report, _compiled, _length = compile_h3_plan(plan)

    assert "<Subject 1> (appears in [Shot 3] and [Shot 4])" in prompt
    assert "Apply <Subject 1>" not in prompt


def test_scope_reports_the_exact_shot_missing_the_subject_label():
    plan = project()
    plan, _handle, _image, _preview = image_reference(
        plan,
        content_type=CONTENT_OBJECT,
        subject="watch",
        scope="2",
    )
    plan = shot(plan, 0.0, "An empty room is established.")
    plan = shot(plan, 2.0, "A close-up shows a watch.")

    with pytest.raises(ValueError, match=r"<Subject 1>.*scoped to.*\[Shot 2\]"):
        compile_h3_plan(plan)


def test_subject_binding_reuses_one_physical_picture_for_two_subjects():
    plan = project()
    plan, handle, _image, _preview = image_reference(
        plan,
        description="A woman wearing a distinctive red jacket.",
    )
    plan, returned_handle, preview = MiniMaxH3PlanV2SubjectBinding().bind_subject(
        plan,
        handle,
        "red jacket",
        CONTENT_OBJECT,
        RETENTION_AUTO,
        "",
        "Preserve the jacket's material and cut.",
        "",
    )
    plan = shot(
        plan,
        0.0,
        "<Subject 1> enters while wearing <Subject 2>.",
    )

    prompt, _rewrite, report, compiled, _length = compile_h3_plan(plan)

    assert returned_handle == handle
    assert "red jacket" in preview
    assert prompt.count("<Picture 1>") >= 2
    assert "<Subject 1>" in prompt and "<Subject 2>" in prompt
    assert "<Picture 2>" not in prompt
    assert [route["route"] for route in compiled["compiled"]["routes"]] == [
        "ref_image_0"
    ]
    assert "1 picture(s)" in report


def test_video_edit_does_not_invent_a_subject():
    plan = project("Remove the logo while preserving the source camera movement.")
    plan, _handle, _video, _preview = video_reference(plan)
    plan = shot(plan, 0.0, "Edit the source composition without adding a new subject.")

    prompt, _rewrite, report, _compiled, _length = compile_h3_plan(plan)

    assert prompt.startswith("subject_definitions:\n<Video 1> is the source video")
    assert "<Subject 1>" not in prompt
    assert "[video editing]" in prompt
    assert "Mode: Ref2VA" in report


def test_motion_video_requires_and_targets_an_upstream_subject():
    plan = project("Transfer the supplied running motion to the woman.")
    plan, _handle, _image, _preview = image_reference(plan)
    frames = torch.zeros(48, 32, 48, 3)
    plan, _video_handle, _video, _preview = MiniMaxH3PlanV2VideoReference().add_video(
        plan,
        frames,
        VIDEO_MOTION,
        "running source",
        "A runner accelerates with a strong forward lean.",
        24.0,
        UNASSIGNED_CONTENT_TYPE,
        "",
        "woman",
        RETENTION_AUTO,
        "",
    )
    plan = shot(plan, 0.0, "<Subject 1> performs the transferred running motion.")

    prompt, _rewrite, _report, _compiled, _length = compile_h3_plan(plan)

    assert (
        "<Video 1> is the motion and action reference transferred to <Subject 1>"
        in prompt
    )
    assert "<Video 1>: attribute_transfer" in prompt


def test_temporal_structure_video_uses_weak_reference_not_attribute_transfer():
    plan = project("Follow the supplied edit rhythm without copying its scene.")
    plan, _video_handle, _video, _preview = video_reference(
        plan,
        use=VIDEO_STRUCTURE,
        name="editing rhythm",
        description="Three measured cuts followed by a long closing hold.",
    )
    plan = shot(plan, 0.0, "A new scene follows the referenced pacing structure.")

    prompt, _rewrite, _report, compiled, _length = compile_h3_plan(plan)

    assert "is the camera, cuts, rhythm, and temporal-structure reference" in prompt
    assert "<Video 1>: weak_reference" in prompt
    assert "<Video 1>: attribute_transfer" not in prompt
    assert compiled["assets"][0]["retention"] == "weak_reference"


def test_reusable_video_action_auto_retention_accepts_and_names_transfer_target():
    plan = project("Transfer the supplied running action to the referenced woman.")
    plan, _handle, _image, _preview = image_reference(plan)
    plan, _video_handle, _video, _preview = MiniMaxH3PlanV2VideoReference().add_video(
        plan,
        torch.zeros(48, 32, 48, 3),
        VIDEO_DEFINE_VISIBLE,
        "running action",
        "A runner accelerates with a strong forward lean.",
        24.0,
        CONTENT_ACTION,
        "running action",
        "woman",
        RETENTION_AUTO,
        "",
    )
    plan = shot(
        plan,
        0.0,
        "<Subject 1> performs the action defined by <Subject 2>.",
    )

    prompt, _rewrite, _report, _compiled, _length = compile_h3_plan(plan)

    assert (
        "Transfer the pose and movement defined by <Video 1> to <Subject 1>"
    ) in prompt
    assert (
        "<Subject 2> (appears wherever cited in the Shot plan): attribute_transfer"
        in prompt
    )
    assert "are transferred to <Subject 1>" in prompt


def test_subject_definitions_and_retention_use_exact_selected_roles():
    plan = project("A woman carries a referenced object through the scene.")
    plan, _woman_handle, _image, _preview = image_reference(
        plan,
        scope="1",
    )
    plan, _object_handle, _image, _preview = image_reference(
        plan,
        name="silver watch",
        description="A square silver wristwatch.",
        content_type=CONTENT_OBJECT,
        subject="watch",
        scope="1",
    )
    plan = shot(
        plan,
        0.0,
        "<Subject 1> enters while carrying <Subject 2>.",
    )

    prompt, _rewrite, _report, _compiled, _length = compile_h3_plan(plan)

    assert (
        "<Subject 1> is woman. The identity and appearance of <Subject 1> "
        "are defined by <Picture 1>."
    ) in prompt
    assert (
        "<Subject 2> is watch. The visible object appearance of <Subject 2> "
        "is defined by <Picture 2>."
    ) in prompt
    assert (
        "<Subject 1> (appears in [Shot 1]): fully_preserved - "
        "the defined identity and appearance are preserved."
    ) in prompt
    assert (
        "<Subject 2> (appears in [Shot 1]): fully_preserved - "
        "the defined visible object appearance is preserved."
    ) in prompt
    assert "reusable visible subject or scene" not in prompt
    assert "object, prop, clothing, interface" not in prompt
    assert "identity, appearance, or composition" not in prompt


def test_references_cannot_be_appended_after_the_first_shot():
    plan = shot(project(), 0.0, "The opening shot.")

    with pytest.raises(ValueError, match="cannot follow.*timeline"):
        image_reference(plan)


def test_voice_requires_dialogue_event_and_content_requires_exact_metadata():
    plan = project()
    plan, _handle, _image, _preview = image_reference(plan)
    plan, _audio, _preview = audio_reference(
        plan,
        use=AUDIO_VOICE,
        name="woman voice",
        speaker="woman",
    )
    plan = shot(plan, 0.0, "<Subject 1> silently looks through the window.")

    with pytest.raises(ValueError, match="has no Dialogue Event"):
        compile_h3_plan(plan)

    with pytest.raises(ValueError, match="requires both language and exact transcript"):
        audio_reference(
            project(),
            use="Dialogue or lyric content",
            name="spoken source",
            speaker="narrator",
        )


def test_complete_audio_copy_rejects_new_soundscape():
    plan = project(soundscape="New rain ambience.")
    plan, _handle, _image, _preview = image_reference(plan)
    plan, _audio, _preview = audio_reference(
        plan,
        use=AUDIO_COPY_COMPLETE,
        name="complete source mix",
    )
    plan = shot(plan, 0.0, "<Subject 1> walks through the rain.")

    with pytest.raises(ValueError, match="clear the new overall_soundscape"):
        compile_h3_plan(plan)


def test_node_contract_exposes_the_complete_phase_one_chain():
    assert set(NODE_CLASS_MAPPINGS) == {
        "MiniMaxH3PlanV2ProjectSetup",
        "MiniMaxH3PlanV2ImageReference",
        "MiniMaxH3PlanV2SubjectBinding",
        "MiniMaxH3PlanV2VideoReference",
        "MiniMaxH3PlanV2AudioReference",
        "MiniMaxH3PlanV2Shot",
        "MiniMaxH3PlanV2DialogueEvent",
        "MiniMaxH3PlanV2PromptMerge",
    }
    assert MiniMaxH3PlanV2ProjectSetup.RETURN_TYPES[0] == PLAN_TYPE
    assert MiniMaxH3PlanV2PromptMerge.RETURN_TYPES[2] == PLAN_TYPE
    assert list(MiniMaxH3PlanV2AudioReference.INPUT_TYPES()["optional"]) == [
        "paired_video"
    ]
    assert MiniMaxH3PlanV2PromptMerge.RETURN_NAMES == (
        "h3_prompt",
        "rewrite_request",
        "plan_context",
        "problems_report",
        "h3_length",
    )
