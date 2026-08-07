from types import MappingProxyType

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
    CHARACTER_REPLACEMENT_APPEARANCE_POLICIES,
    CONTENT_ACTION,
    CONTENT_IDENTITY,
    CONTENT_OBJECT,
    DIALOGUE_CONTINUITY_CUTOFF,
    DIALOGUE_CONTINUITY_FROM_PREVIOUS,
    DIALOGUE_CONTINUITY_TO_NEXT,
    IMAGE_DEFINE_VISIBLE,
    IMAGE_FIRST_FRAME,
    IMAGE_KEYFRAME,
    IMAGE_LAST_FRAME,
    IMAGE_STORYBOARD,
    KEYFRAME_SHOT_ENDING,
    KEYFRAME_SHOT_INTERNAL,
    KEYFRAME_SHOT_OPENING,
    PLAN_TYPE,
    SHOT_HANDLE_TYPE,
    TARGET_FOLEY,
    RETENTION_AUTO,
    RETENTION_TRANSFER,
    UNASSIGNED_CONTENT_TYPE,
    VIDEO_CONTINUE,
    VIDEO_DEFINE_VISIBLE,
    VIDEO_EDIT,
    VIDEO_MOTION,
    VIDEO_STRUCTURE,
    MiniMaxH3PlanV2AudioReference,
    MiniMaxH3PlanV2CharacterReplacement,
    MiniMaxH3PlanV2DialogueEvent,
    MiniMaxH3PlanV2FoleyTarget,
    MiniMaxH3PlanV2ImageReference,
    MiniMaxH3PlanV2ProjectSetup,
    MiniMaxH3PlanV2PromptMerge,
    MiniMaxH3PlanV2Shot,
    MiniMaxH3PlanV2ShotKeyframe,
    MiniMaxH3PlanV2ShotMotionReference,
    MiniMaxH3PlanV2SubjectBinding,
    MiniMaxH3PlanV2VideoReference,
    NODE_CLASS_MAPPINGS,
    compile_h3_plan,
    validated_plan,
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


def foley_target(plan, *, frames=None, source_fps=24.0):
    video = torch.zeros(144, 32, 48, 3) if frames is None else frames
    return MiniMaxH3PlanV2FoleyTarget().set_foley_target(
        plan,
        video,
        source_fps,
    )


def test_foley_target_compiles_audio_only_prompt_without_video_reference_route():
    plan = project(
        "Generate realistic production Foley synchronized to every visible action.",
        duration=6.0,
        soundscape=(
            "Continuous room tone with footsteps, cloth movement, and object contacts "
            "synchronized to the picture."
        ),
    )
    plan, prepared, preview = foley_target(plan)
    plan = shot(
        plan,
        0.0,
        "A person crosses the room and places a glass on a table; each footfall and the "
        "glass contact are heard at the exact visible moment.",
    )

    prompt, _rewrite, report, compiled, length = compile_h3_plan(plan)

    assert prepared.shape[0] == length == 158
    assert plan["target"]["task"] == TARGET_FOLEY
    assert "video=0" in preview and "audio=1" in preview
    assert "picture track remains exactly unchanged" in prompt
    assert "Generate only a new synchronized audio track" in prompt
    assert "<Video 1>" not in prompt
    assert compiled["compiled"]["mode"] == "T2VA"
    assert compiled["compiled"]["target_task"] == TARGET_FOLEY
    assert compiled["compiled"]["latent_strategy"] == "preserve_video_generate_audio"
    assert compiled["compiled"]["routes"] == []
    assert "video mask = 0" in report and "audio mask = 1" in report


def test_foley_target_allows_audio_only_reference_but_rejects_copy_roles():
    plan = project(duration=6.0)
    plan = foley_target(plan)[0]
    plan = audio_reference(
        plan,
        use=AUDIO_SFX,
        name="footstep texture",
        layer="footsteps synchronized to visible ground contact",
        scope="1",
    )[0]
    plan = shot(
        plan,
        0.0,
        "Each visible foot contact produces the texture referenced by <Audio 1>.",
    )
    prompt, _rewrite, _report, compiled, _length = compile_h3_plan(plan)
    assert "[reference generation + audio reference]" in prompt
    assert compiled["compiled"]["mode"] == "Ref2VA"
    assert [route["route"] for route in compiled["compiled"]["routes"]] == [
        "ref_audio_0"
    ]

    copy_plan = project(duration=6.0, soundscape="")
    copy_plan = foley_target(copy_plan)[0]
    copy_plan = audio_reference(
        copy_plan,
        use=AUDIO_COPY_COMPLETE,
        name="source soundtrack",
        seconds=6.0,
    )[0]
    copy_plan = shot(copy_plan, 0.0, "The visible picture track remains unchanged.")
    with pytest.raises(ValueError, match="Foley uses audio mask 1"):
        compile_h3_plan(copy_plan)


def test_video_reference_accepts_only_the_native_15_second_padding_boundary():
    plan = project(duration=15.0)
    _plan, _handle, video, _preview = video_reference(
        plan,
        frames=torch.zeros(362, 8, 8, 3),
    )

    assert video.shape[0] == 362

    with pytest.raises(
        ValueError,
        match=r"Received 363 frames at 24 FPS \(15\.125 seconds\)",
    ):
        video_reference(
            plan,
            frames=torch.zeros(363, 8, 8, 3),
        )


def test_five_shots_compose_keyframes_and_one_motion_reference_without_manual_scopes():
    plan = project(
        "A five-shot sequence follows the referenced woman across one continuous scene.",
        duration=10.0,
    )
    plan, _subject_handle, _subject_image, _preview = image_reference(plan)

    positions = (
        KEYFRAME_SHOT_OPENING,
        KEYFRAME_SHOT_INTERNAL,
        KEYFRAME_SHOT_ENDING,
        KEYFRAME_SHOT_OPENING,
        KEYFRAME_SHOT_ENDING,
    )
    for number, (cut, position) in enumerate(
        zip((0.0, 2.0, 4.0, 6.0, 8.0), positions),
        start=1,
    ):
        plan, _timeline, handle = MiniMaxH3PlanV2Shot().add_shot(
            plan,
            cut,
            f"<Subject 1> performs the visible action planned for shot {number}.",
            f"Camera composition for shot {number}.",
            "Direct cut",
        )
        plan, forwarded, _image, preview = MiniMaxH3PlanV2ShotKeyframe().attach_keyframe(
            plan,
            handle,
            torch.zeros(1, 8, 8, 3),
            f"shot {number} keyframe",
            f"The exact composition and opening state for shot {number}.",
            position,
        )
        assert forwarded == handle
        assert f"[Shot {number}]" in preview
        if number == 3:
            plan, forwarded, video, preview = (
                MiniMaxH3PlanV2ShotMotionReference().attach_motion(
                    plan,
                    handle,
                    torch.zeros(48, 8, 8, 3),
                    "woman",
                    "shot 3 movement",
                    "A controlled turn and two measured steps.",
                    24.0,
                )
            )
            assert forwarded == handle
            assert video.shape[0] == 39
            assert "<Video 1>" in preview
            assert "<Subject 2>" in preview

    prompt, _rewrite, report, compiled, _length = compile_h3_plan(plan)

    keyframes = [
        asset for asset in compiled["assets"] if asset["relationship"] == IMAGE_KEYFRAME
    ]
    assert [asset["shot_scope"] for asset in keyframes] == ["1", "2", "3", "4", "5"]
    assert [asset["keyframe_position"] for asset in keyframes] == list(positions)
    shot_lines = {
        number: next(
            line for line in prompt.splitlines() if line.startswith(f"[Shot {number}]")
        )
        for number in range(1, 6)
    }
    for number in range(1, 6):
        label = f"<Picture {number + 1}>"
        assert label in shot_lines[number]
        assert all(
            label not in line
            for other_number, line in shot_lines.items()
            if other_number != number
        )
    assert "<Subject 2>" in shot_lines[3]
    assert "<Video 1>" not in shot_lines[3]
    assert "performs the pose sequence, action, and motion timing" in shot_lines[3]
    assert all("<Subject 2>" not in shot_lines[number] for number in (1, 2, 4, 5))
    assert "<Subject 2> is the reusable pose, action, and motion from <Video 1>" in prompt
    assert "<Video 1> is the motion and action reference" not in prompt
    assert "<Subject 2> (appears in [Shot 3]): attribute_transfer" in prompt
    assert "<Video 1> ([Shot 3]): attribute_transfer" not in prompt
    assert "<Picture 2> ([Shot 1] opening frame): fully_preserved" in prompt
    assert "<Picture 3> ([Shot 2] internal composition keyframe): fully_preserved" in prompt
    assert "<Picture 4> ([Shot 3] ending frame): fully_preserved" in prompt
    assert "The declared reference roles use" not in prompt
    assert compiled["compiled"]["subject_labels"] == {"woman": "<Subject 1>"}
    assert compiled["compiled"]["motion_subject_labels"] == {
        "video-1": "<Subject 2>"
    }
    assert "the H3 reference guide normally recommends 350-500 English words" in report
    assert [entry["route"] for entry in compiled["compiled"]["routes"]] == [
        "ref_image_0",
        "ref_image_1",
        "ref_image_2",
        "ref_image_3",
        "ref_image_4",
        "ref_image_5",
        "ref_video_0",
    ]


def test_shot_attachments_reject_a_handle_for_another_timeline_position():
    plan = project()
    plan, _preview, handle = MiniMaxH3PlanV2Shot().add_shot(
        plan,
        0.0,
        "The opening composition is established.",
        "",
        "Direct cut",
    )
    invalid = dict(handle, cut_at=1.0)

    with pytest.raises(ValueError, match="does not belong"):
        MiniMaxH3PlanV2ShotKeyframe().attach_keyframe(
            plan,
            invalid,
            torch.zeros(1, 8, 8, 3),
            "opening",
            "Opening composition.",
        )


def audio_value(seconds=3.0, sample_rate=32_000):
    return {
        "waveform": torch.zeros(1, 1, round(seconds * sample_rate)),
        "sample_rate": sample_rate,
    }


def character_replacement(
    plan,
    video_handle,
    *,
    subject="woman",
    source_character="the woman in the red jacket",
    policy=None,
    preserve_performance=True,
    preserve_scene=True,
    scope="all",
    instructions="",
):
    return MiniMaxH3PlanV2CharacterReplacement().add_replacement(
        plan,
        video_handle,
        subject,
        source_character,
        policy or CHARACTER_REPLACEMENT_APPEARANCE_POLICIES[0],
        preserve_performance,
        preserve_scene,
        scope,
        instructions,
    )


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
    continuity_mode="Complete in this Shot",
):
    return MiniMaxH3PlanV2DialogueEvent().add_dialogue(
        plan,
        speaker,
        language,
        words,
        delivery,
        voice_mode,
        continuity_mode,
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


def test_plan_v2_accepts_pre_replacement_payloads_as_empty_mappings():
    plan = project()
    plan.pop("character_replacements")

    normalized = validated_plan(plan)

    assert normalized["character_replacements"] == []


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


def test_last_frame_endpoint_uses_the_guide_l2va_alignment_preamble():
    plan = project("Resolve into the supplied final composition.")
    plan, _handle, _image, _preview = image_reference(
        plan,
        use=IMAGE_LAST_FRAME,
        name="ending",
        content_type=UNASSIGNED_CONTENT_TYPE,
        subject="",
    )
    plan = shot(plan, 0.0, "The action resolves into <Picture 1>.")

    prompt, _rewrite, report, _compiled, _length = compile_h3_plan(plan)

    assert prompt.startswith(
        "How the reference pictures align with the target video — "
        "<Picture 1> (from [Shot 1]) aligns with the 6.58-second mark of the target video."
    )
    assert "Mode: L2VA" in report


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


def test_audio_reference_chain_reports_the_cumulative_duration_limit():
    valid = project()
    valid, _audio, _preview = audio_reference(
        valid,
        use=AUDIO_VOICE,
        name="first voice",
        speaker="woman",
        seconds=7.4,
    )
    valid, _audio, preview = audio_reference(
        valid,
        use=AUDIO_VOICE,
        name="second voice",
        speaker="man",
        seconds=7.4,
    )
    assert "7.400s" in preview

    invalid = project()
    invalid, _audio, _preview = audio_reference(
        invalid,
        use=AUDIO_VOICE,
        name="first voice",
        speaker="woman",
        seconds=8.0,
    )
    with pytest.raises(ValueError) as caught:
        audio_reference(
            invalid,
            use=AUDIO_VOICE,
            name="second voice",
            speaker="man",
            seconds=8.0,
        )
    message = str(caught.value)
    assert "totals 16.000s" in message
    assert "first voice=8.000s" in message
    assert "second voice=8.000s" in message
    assert "15-second cumulative limit" in message


def test_audio_reference_accepts_and_normalizes_lazy_audio_mappings():
    plan = project()
    lazy_audio = MappingProxyType(audio_value(seconds=3.25))

    plan, normalized_audio, preview = MiniMaxH3PlanV2AudioReference().add_audio(
        plan,
        lazy_audio,
        AUDIO_VOICE,
        "lazy loader voice",
        "woman",
        "",
        "",
        "",
        "Natural delivery.",
        "all",
    )

    assert isinstance(normalized_audio, dict)
    assert normalized_audio["waveform"] is lazy_audio["waveform"]
    assert plan["assets"][0]["media"] is normalized_audio
    assert "3.250s" in preview


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


def test_paired_continuation_audio_explicitly_forbids_source_looping():
    plan = project(
        prompt="Continue naturally after the final frame of the supplied source video.",
        soundscape="Continue the established location ambience and synchronized action sounds.",
    )
    plan, video_handle, _h3_video, _preview = video_reference(
        plan,
        use=VIDEO_CONTINUE,
        name="source clip to extend",
        description="The supplied source video whose final audiovisual state starts the extension.",
    )
    plan, _audio, _preview = audio_reference(
        plan,
        use=AUDIO_CONTINUITY,
        name="source synchronized soundtrack",
        layer="the established ambience, action sounds, and soundtrack progression",
        instructions="Preserve a seamless tonal and spatial transition at the boundary.",
        paired_video=video_handle,
        seconds=2.0,
    )
    plan = shot(
        plan,
        0.0,
        "The action resumes immediately after <Video 1>'s final frame and moves forward.",
    )

    prompt, _rewrite, report, compiled, _length = compile_h3_plan(plan)

    assert "[video continuation + audio reference]" in prompt
    assert (
        "<Audio 1> is the synchronized soundtrack of <Video 1> and the "
        "audio-continuity reference"
    ) in prompt
    assert "Generate new audio beginning after the source endpoint" in prompt
    assert "do not copy, restart, replay, repeat, or loop the source signal" in prompt
    assert "guides newly generated audio after <Video 1>'s endpoint" in prompt
    assert "newly generated, forward-developing audio" in prompt
    assert [entry["route"] for entry in compiled["compiled"]["routes"]] == [
        "ref_video_audio_0",
        "ref_video_0",
    ]
    assert "<Audio 1> -> ref_video_audio_0" in report


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


def test_inline_sound_effect_tag_controls_shot_placement_without_global_duplicate():
    plan = project(soundscape="Truck engine and road vibration.")
    plan, _handle, _image, _preview = image_reference(plan)
    plan, _audio, preview = audio_reference(
        plan,
        use=AUDIO_SFX,
        name="suction texture",
        layer="the synchronized suction event",
        instructions="Use the supplied wet suction texture.",
        scope="2",
    )
    plan = shot(plan, 0.0, "<Subject 1> sits inside the moving truck.")
    placement = (
        "Each visible downward movement produces the suction texture referenced by "
        "<Audio 1>, synchronized at the point of contact."
    )
    plan = shot(plan, 3.0, placement)

    prompt, _rewrite, _report, _compiled, _length = compile_h3_plan(plan)

    assert "insert that tag in the intended Shot sound sentence" in preview
    assert f"[Shot 2] At 00:03.000, cut directly to {placement[0].lower() + placement[1:]}" in prompt
    assert "<Audio 1> is the sound-effect texture reference" in prompt
    assert "overall_soundscape:\nTruck engine and road vibration." in prompt
    assert "Apply <Audio 1> only according to its declared sound-effect texture role." not in prompt


def test_untagged_sound_effect_keeps_global_soundscape_fallback():
    plan = project(soundscape="Truck engine and road vibration.")
    plan, _handle, _image, _preview = image_reference(plan)
    plan, _audio, _preview = audio_reference(
        plan,
        use=AUDIO_SFX,
        name="suction texture",
        layer="the synchronized suction event",
    )
    plan = shot(plan, 0.0, "<Subject 1> sits inside the moving truck.")

    prompt, _rewrite, _report, _compiled, _length = compile_h3_plan(plan)

    assert "Apply <Audio 1> only according to its declared sound-effect texture role." in prompt


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


def test_dialogue_across_a_cut_marks_both_adjacent_parts():
    plan = project()
    plan, _handle, _image, _preview = image_reference(plan)
    plan = shot(plan, 0.0, "<Subject 1> begins speaking in the passenger seat.")
    plan = dialogue(
        plan,
        "woman",
        "I was about to",
        continuity_mode=DIALOGUE_CONTINUITY_TO_NEXT,
    )
    plan = shot(plan, 3.0, "A close view holds on <Subject 1> across the cut.")
    plan = dialogue(
        plan,
        "woman",
        "tell you something.",
        continuity_mode=DIALOGUE_CONTINUITY_FROM_PREVIOUS,
    )

    prompt, _rewrite, _report, _compiled, _length = compile_h3_plan(plan)

    assert "<d>[English] I was about to <scenetrans></d>" in prompt
    assert "<d>[English] <scenetrans> tell you something.</d>" in prompt


def test_dialogue_cutoff_and_user_punctuation_are_preserved_inside_tag():
    plan = project()
    plan, _handle, _image, _preview = image_reference(plan)
    plan = shot(plan, 0.0, "<Subject 1> calls out as the video ends.")
    plan = dialogue(
        plan,
        "woman",
        "Wait, I need to",
        continuity_mode=DIALOGUE_CONTINUITY_CUTOFF,
    )

    prompt, _rewrite, _report, _compiled, _length = compile_h3_plan(plan)

    assert "<d>[English] Wait, I need to <cutoff></d>" in prompt
    no_punctuation = dialogue(
        shot(project(), 0.0, "A speaker talks."),
        "speaker",
        "These are the exact words",
        voice_mode="Off-screen speech",
        delivery="",
    )
    no_punctuation_prompt = compile_h3_plan(no_punctuation)[0]
    assert "<d>[English] These are the exact words</d>" in no_punctuation_prompt
    assert "<d>[English] These are the exact words</d>." not in no_punctuation_prompt


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
    assert "any such music already present in <Audio 1> remains unchanged" in prompt


def test_paired_complete_audio_names_video_and_requires_matching_interval():
    plan = project(soundscape="", music="N/A")
    plan, video_handle, _video, _preview = video_reference(plan)
    with pytest.raises(ValueError, match="must cover the same source interval"):
        audio_reference(
            plan,
            use=AUDIO_COPY_COMPLETE,
            name="mismatched source soundtrack",
            paired_video=video_handle,
            seconds=3.0,
        )

    plan, _audio, _preview = audio_reference(
        plan,
        use=AUDIO_COPY_COMPLETE,
        name="source soundtrack",
        paired_video=video_handle,
        seconds=2.0,
    )
    plan = shot(plan, 0.0, "Edit <Video 1> while preserving its source timeline.")
    prompt, _rewrite, report, _compiled, _length = compile_h3_plan(plan)

    assert (
        "<Audio 1> is the synchronized audio track of <Video 1> and is reused "
        "as the target video's complete final audio track."
    ) in prompt
    assert (
        "<Audio 1>: fully_copy - the synchronized audio track of <Video 1> is "
        "reused 1:1 as the target video's complete final audio track."
    ) in prompt
    assert "<Video 1>: 2.000s source" in report
    assert "native-grid preparation omits the final 0.375s" in report
    assert "Guide warning: this source edit is represented by one Shot" in report


def test_blank_generated_soundscape_gets_a_non_silent_default():
    plan = shot(
        project(soundscape="", music="N/A"),
        0.0,
        "A person crosses a quiet room.",
    )

    prompt, _rewrite, _report, _compiled, _length = compile_h3_plan(plan)

    assert (
        "overall_soundscape: Use coherent ambience and synchronized physical sounds."
        in prompt
    )


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


def test_character_replacement_maps_source_performer_and_injects_locked_shot_instructions():
    plan = project("A woman enters a truck and sits beside the driver.")
    plan, _image_handle, _image, _preview = image_reference(plan)
    plan, video_handle, _video, _preview = video_reference(
        plan,
        description=(
            "A woman in a red jacket enters a truck and sits beside the driver."
        ),
    )
    plan, preview = character_replacement(
        plan,
        video_handle,
        source_character="the woman in the red jacket",
        scope="1-2",
        instructions="Maintain clean identity continuity through the cut.",
    )
    plan = shot(plan, 0.0, "The source performer opens the passenger door.")
    plan = shot(plan, 2.0, "She settles into the passenger seat.")

    prompt, _rewrite, report, compiled, _length = compile_h3_plan(plan)

    assert "<Video 1> performer (the woman in the red jacket) -> <Subject 1>" in preview
    assert (
        "In [Shot 1] and [Shot 2], <Subject 1> replaces only the source "
        "performer described as the woman in the red jacket from <Video 1>."
        in prompt
    )
    detailed = prompt.split("detailed_description:\n", 1)[1]
    assert detailed.count(
        "Using <Video 1> as the source timeline, replace only the source performer "
        "described as the woman in the red jacket with <Subject 1>."
    ) == 2
    assert detailed.count(
        "retain the source performer's body proportions and wardrobe from <Video 1>"
    ) == 2
    assert "Preserve that source performer's pose, motion, timing" in detailed
    assert "Keep every other person, the environment, props, lighting" in detailed
    assert "<Subject 1> (appears in [Shot 1] and [Shot 2])" in prompt
    assert "only the declared character replacement is applied" in prompt
    assert len(compiled["character_replacements"]) == 1
    assert "character replacement mapping(s)" in report
    assert "Character replacement controls:" in report
    assert "preserve performance: True" in report
    assert "preserve scene/camera/cuts: True" in report


def test_character_replacement_edits_source_then_continues_same_character():
    plan = project(
        "Continue after the source endpoint with the referenced replacement character."
    )
    plan, _image_handle, _image, _preview = image_reference(
        plan,
        name="replacement character",
        subject="replacement character",
    )
    plan, video_handle, _video, _preview = video_reference(
        plan,
        use=VIDEO_CONTINUE,
        name="source clip to continue",
        description=(
            "A woman in a red jacket runs toward the truck as the camera pans right."
        ),
    )
    plan, _preview = character_replacement(
        plan,
        video_handle,
        subject="replacement character",
        source_character="the woman in the red jacket",
        scope="all",
    )
    plan = shot(
        plan,
        0.0,
        "The action continues after <Video 1>'s endpoint as the performer reaches the truck.",
    )

    prompt, _rewrite, _report, compiled, _length = compile_h3_plan(plan)

    assert "[reference generation + video editing + video continuation]" in prompt
    assert (
        "<Subject 1> replaces only the source performer described as the woman in the "
        "red jacket throughout the target portion derived from <Video 1> and remains "
        "the same character when the target continues beyond <Video 1>'s endpoint"
    ) in prompt
    assert (
        "Using <Video 1> as both the source timeline and continuation anchor, replace "
        "only the source performer described as the woman in the red jacket with "
        "<Subject 1> from the first source-derived frame onward."
    ) in prompt
    assert (
        "<Picture 1> provides identity and appearance only; never use it as a target "
        "frame, opening composition, standalone shot, or animated segment."
    ) in prompt
    assert "evolve the performance forward without restarting or replaying it" in prompt
    assert "then continue them beyond its endpoint without a reset" in prompt
    assert (
        "the source timeline is recreated with only the selected performer replaced "
        "by the declared Subject, then that edited state continues beyond the endpoint"
    ) in prompt
    assert compiled["character_replacements"][0]["source_video_asset_id"] == "video-1"


def test_replacement_report_exposes_disabled_preservation_controls():
    plan = project("Replace one source performer.")
    plan, _image_handle, _image, _preview = image_reference(plan)
    plan, video_handle, _video, _preview = video_reference(plan)
    plan, _preview = character_replacement(
        plan,
        video_handle,
        source_character="the source performer in the foreground",
        preserve_performance=False,
        preserve_scene=False,
        scope="all",
    )
    plan = shot(plan, 0.0, "The source timeline is edited with the replacement.")

    _prompt, _rewrite, report, _compiled, _length = compile_h3_plan(plan)

    assert "preserve performance: False" in report
    assert "preserve scene/camera/cuts: False" in report
    assert "source action timing and expressions may drift" in report
    assert "source shot order and composition may drift" in report


def test_character_replacement_continuation_requires_total_duration_beyond_source():
    plan = project(duration=6.0)
    plan, _image_handle, _image, _preview = image_reference(plan)
    plan, video_handle, _video, _preview = video_reference(
        plan,
        use=VIDEO_CONTINUE,
        frames=torch.zeros(168, 32, 48, 3),
    )

    with pytest.raises(ValueError, match="Project duration.*longer than the source video"):
        character_replacement(plan, video_handle)


def test_character_replacement_scope_is_the_subject_citation_only_for_selected_shots():
    plan = project()
    plan, _image_handle, _image, _preview = image_reference(plan)
    plan, video_handle, _video, _preview = video_reference(plan)
    plan, _preview = character_replacement(plan, video_handle, scope="2")
    plan = shot(plan, 0.0, "The driver is alone in the first source shot.")
    plan = shot(plan, 2.0, "The woman in the red jacket enters the frame.")

    prompt, *_rest = compile_h3_plan(plan)
    detailed = prompt.split("detailed_description:\n", 1)[1]
    shot_one, shot_two = detailed.split("[Shot 2]", 1)

    assert "Using <Video 1> as the source timeline" not in shot_one
    assert "Using <Video 1> as the source timeline" in shot_two


def test_character_replacement_rejects_non_edit_or_continuation_video_and_non_identity_subject():
    plan = project()
    plan, _image_handle, _image, _preview = image_reference(plan)
    plan, video_handle, _video, _preview = video_reference(
        plan, use=VIDEO_STRUCTURE
    )
    with pytest.raises(ValueError, match="Source video to edit or Source video to continue"):
        character_replacement(plan, video_handle)

    plan = project()
    plan, _image_handle, _image, _preview = image_reference(
        plan,
        content_type=CONTENT_OBJECT,
        subject="red jacket",
    )
    plan, video_handle, _video, _preview = video_reference(plan)
    with pytest.raises(ValueError, match="Identity or appearance binding"):
        character_replacement(plan, video_handle, subject="red jacket")


def test_character_replacement_owns_h3_labels_and_requires_an_upstream_subject():
    plan = project()
    plan, _image_handle, _image, _preview = image_reference(plan)
    plan, video_handle, _video, _preview = video_reference(plan)

    with pytest.raises(ValueError, match="not an upstream Subject"):
        character_replacement(plan, video_handle, subject="unknown character")
    with pytest.raises(ValueError, match="plain language"):
        character_replacement(
            plan,
            video_handle,
            source_character="replace <Subject 2> in <Video 1>",
        )


def test_motion_video_requires_and_targets_an_upstream_subject():
    plan = project("Transfer the supplied running motion to the woman.")
    plan, _handle, _image, _preview = image_reference(plan)
    frames = torch.zeros(48, 32, 48, 3)
    plan, _video_handle, _video, preview = MiniMaxH3PlanV2VideoReference().add_video(
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
    assert "<Subject 2> reusable action sourced from provisional <Video 1>" in preview
    plan = shot(plan, 0.0, "<Subject 1> performs the transferred running motion.")

    prompt, _rewrite, _report, _compiled, _length = compile_h3_plan(plan)

    assert "<Subject 2> is the reusable pose, action, and motion from <Video 1>" in prompt
    assert "Transfer <Subject 2>'s visible performance to <Subject 1>" in prompt
    assert "<Subject 2> (appears in [Shot 1]): attribute_transfer" in prompt
    assert "<Video 1>: attribute_transfer" not in prompt
    shot_line = next(
        line for line in prompt.splitlines() if line.startswith("[Shot 1]")
    )
    assert "<Subject 2>" in shot_line
    assert "<Video 1>" not in shot_line


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
        "MiniMaxH3PlanV2FoleyTarget",
        "MiniMaxH3PlanV2ImageReference",
        "MiniMaxH3PlanV2SubjectBinding",
        "MiniMaxH3PlanV2VideoReference",
        "MiniMaxH3PlanV2CharacterReplacement",
        "MiniMaxH3PlanV2AudioReference",
        "MiniMaxH3PlanV2Shot",
        "MiniMaxH3PlanV2ShotKeyframe",
        "MiniMaxH3PlanV2ShotMotionReference",
        "MiniMaxH3PlanV2DialogueEvent",
        "MiniMaxH3PlanV2PromptMerge",
    }
    assert MiniMaxH3PlanV2ProjectSetup.RETURN_TYPES[0] == PLAN_TYPE
    assert MiniMaxH3PlanV2Shot.RETURN_TYPES[-1] == SHOT_HANDLE_TYPE
    assert MiniMaxH3PlanV2ShotKeyframe.RETURN_TYPES[1] == SHOT_HANDLE_TYPE
    assert MiniMaxH3PlanV2ShotMotionReference.RETURN_TYPES[1] == SHOT_HANDLE_TYPE
    keyframe_position = MiniMaxH3PlanV2ShotKeyframe.INPUT_TYPES()["required"][
        "keyframe_position"
    ]
    assert keyframe_position[0] == [
        KEYFRAME_SHOT_OPENING,
        KEYFRAME_SHOT_INTERNAL,
        KEYFRAME_SHOT_ENDING,
    ]
    assert keyframe_position[1]["default"] == KEYFRAME_SHOT_OPENING
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
