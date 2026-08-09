import torch

from composer import (
    DEFAULT_ANIMATION_GOAL,
    DEFAULT_CAMERA_DIRECTION,
    DEFAULT_IDENTITY_DESCRIPTION,
    DEFAULT_KEYFRAME_DESCRIPTION,
    DEFAULT_MOTION_DESCRIPTION,
    DEFAULT_SHOT_DESCRIPTION,
    MiniMaxH3PlanV2KeyframeMotionComposer,
    NODE_CLASS_MAPPINGS,
)
from plan_v2 import (
    AUDIO_COPY_COMPLETE,
    CONTENT_IDENTITY,
    IMAGE_DEFINE_VISIBLE,
    IMAGE_KEYFRAME,
    KEYFRAME_SHOT_OPENING,
    PROMPT_STYLE_COMPACT,
    RETENTION_AUTO,
    VIDEO_MOTION,
    MiniMaxH3PlanV2AudioReference,
    MiniMaxH3PlanV2ImageReference,
    MiniMaxH3PlanV2ProjectSetup,
    MiniMaxH3PlanV2PromptMerge,
    MiniMaxH3PlanV2Shot,
    MiniMaxH3PlanV2ShotKeyframe,
    MiniMaxH3PlanV2ShotMotionReference,
)


FRAME_COUNT = 345
STYLE = "cinematic, photorealistic live-action"
SUBJECT = "character"
IDENTITY_NAME = "character identity"
AUDIO_NAME = "original control-video soundtrack"
KEYFRAME_NAME = "target opening keyframe"
MOTION_NAME = "character motion"


def media_values():
    identity = torch.zeros(1, 8, 8, 3)
    keyframe = torch.zeros(1, 8, 8, 3)
    motion = torch.zeros(FRAME_COUNT, 8, 8, 3)
    audio = {
        "waveform": torch.zeros(1, 1, round(FRAME_COUNT / 24 * 32_000)),
        "sample_rate": 32_000,
    }
    return identity, keyframe, motion, audio


def expanded_plan(identity, keyframe, motion, audio):
    plan = MiniMaxH3PlanV2ProjectSetup().start(
        DEFAULT_ANIMATION_GOAL,
        FRAME_COUNT,
        STYLE,
        "",
        "N/A",
    )[0]
    plan = MiniMaxH3PlanV2ImageReference().add_image(
        plan,
        identity,
        IMAGE_DEFINE_VISIBLE,
        IDENTITY_NAME,
        DEFAULT_IDENTITY_DESCRIPTION,
        CONTENT_IDENTITY,
        SUBJECT,
        RETENTION_AUTO,
        "all",
        "",
    )[0]
    plan = MiniMaxH3PlanV2AudioReference().add_audio(
        plan,
        audio,
        AUDIO_COPY_COMPLETE,
        AUDIO_NAME,
        "",
        "",
        "",
        "",
        "",
        "",
    )[0]
    plan, _preview, shot_handle = MiniMaxH3PlanV2Shot().add_shot(
        plan,
        0.0,
        DEFAULT_SHOT_DESCRIPTION,
        DEFAULT_CAMERA_DIRECTION,
        "Direct cut",
    )
    plan, shot_handle, _image, _preview = (
        MiniMaxH3PlanV2ShotKeyframe().attach_keyframe(
            plan,
            shot_handle,
            keyframe,
            KEYFRAME_NAME,
            DEFAULT_KEYFRAME_DESCRIPTION,
            KEYFRAME_SHOT_OPENING,
        )
    )
    return MiniMaxH3PlanV2ShotMotionReference().attach_motion(
        plan,
        shot_handle,
        motion,
        SUBJECT,
        MOTION_NAME,
        DEFAULT_MOTION_DESCRIPTION,
        24.0,
    )[0]


def composer_plan(identity, keyframe, motion, audio):
    return MiniMaxH3PlanV2KeyframeMotionComposer().compose(
        identity,
        keyframe,
        motion,
        audio,
        FRAME_COUNT,
        DEFAULT_ANIMATION_GOAL,
        STYLE,
        SUBJECT,
        DEFAULT_SHOT_DESCRIPTION,
        DEFAULT_CAMERA_DIRECTION,
        False,
        IDENTITY_NAME,
        DEFAULT_IDENTITY_DESCRIPTION,
        AUDIO_NAME,
        KEYFRAME_NAME,
        DEFAULT_KEYFRAME_DESCRIPTION,
        MOTION_NAME,
        DEFAULT_MOTION_DESCRIPTION,
        24.0,
    )


def test_keyframe_motion_composer_is_registered_with_real_typed_media_inputs():
    assert (
        NODE_CLASS_MAPPINGS["MiniMaxH3PlanV2KeyframeMotionComposer"]
        is MiniMaxH3PlanV2KeyframeMotionComposer
    )
    schema = MiniMaxH3PlanV2KeyframeMotionComposer.INPUT_TYPES()["required"]
    assert schema["identity_image"][0] == "IMAGE"
    assert schema["opening_keyframe"][0] == "IMAGE"
    assert schema["motion_video_frames"][0] == "IMAGE"
    assert schema["original_audio"][0] == "AUDIO"
    assert schema["frame_count"][1]["step"] == 17


def test_keyframe_motion_composer_reproduces_the_expanded_plan_and_prompt():
    identity, keyframe, motion, audio = media_values()
    expected = expanded_plan(identity, keyframe, motion, audio)
    actual, length, preview, prepared_motion, normalized_audio = composer_plan(
        identity,
        keyframe,
        motion,
        audio,
    )

    assert length == FRAME_COUNT
    assert actual["phase"] == "timeline"
    assert [asset["relationship"] for asset in actual["assets"]] == [
        IMAGE_DEFINE_VISIBLE,
        AUDIO_COPY_COMPLETE,
        IMAGE_KEYFRAME,
        VIDEO_MOTION,
    ]
    assert actual["shots"] == expected["shots"]
    assert actual["bindings"] == expected["bindings"]
    assert actual["audio_relationships"] == expected["audio_relationships"]
    assert prepared_motion.shape[0] == FRAME_COUNT
    assert normalized_audio["waveform"] is audio["waveform"]
    assert "2 Pictures" not in preview  # Preview stays semantic rather than inventory-only.
    assert "Identity:" in preview and "Motion:" in preview

    expected_merge = MiniMaxH3PlanV2PromptMerge().merge(
        expected,
        PROMPT_STYLE_COMPACT,
    )
    actual_merge = MiniMaxH3PlanV2PromptMerge().merge(
        actual,
        PROMPT_STYLE_COMPACT,
    )
    assert actual_merge[0] == expected_merge[0]
    assert actual_merge[1] == expected_merge[1]
    assert actual_merge[3] == expected_merge[3]
    assert actual_merge[4] == expected_merge[4] == FRAME_COUNT
    assert actual_merge[2]["compiled"] == expected_merge[2]["compiled"]


def test_keyframe_motion_composer_fixes_complete_copy_audio_policy():
    identity, keyframe, motion, audio = media_values()
    plan = composer_plan(identity, keyframe, motion, audio)[0]

    assert plan["project"]["overall_soundscape"] == ""
    assert plan["project"]["non_diegetic_music"] == "N/A"
    relationship = plan["audio_relationships"][0]
    assert relationship["use"] == AUDIO_COPY_COMPLETE
    assert relationship["instructions"] == ""
    assert relationship["shot_scope"] == ""
