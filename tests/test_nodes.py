from nodes import (
    AUTO_FIDELITY,
    AUTO_GOAL,
    COPY_ALL_AUDIO,
    EDIT_GOAL,
    EDIT_VIDEO,
    FIRST_IMAGE,
    FIRST_LAST_IMAGES,
    LAST_IMAGE,
    MOTION_GOAL,
    MOTION_TARGET_IMAGE,
    MOTION_VIDEO,
    NO_AUDIO,
    NO_IMAGE,
    NO_VIDEO,
    REFERENCE_GOAL,
    MiniMaxH3PromptGuide,
    MiniMaxH3Shot,
    choose_mode,
    parse_assets,
)

import pytest


def build(**overrides):
    values = {
        "what_do_you_want": AUTO_GOAL,
        "target_description": "A woman walks through a rainy market.",
        "how_images_are_used": NO_IMAGE,
        "how_video_is_used": NO_VIDEO,
        "how_audio_is_used": NO_AUDIO,
        "reference_fidelity": AUTO_FIDELITY,
        "reference_assets": "",
        "duration_seconds": 8.0,
        "visual_style": "cinematic, live-action",
        "shot_and_timing_plan": "",
        "camera_direction": "",
        "dialogue_lyrics_and_visible_text": "",
        "overall_soundscape": "Rain and distant footsteps.",
        "non_diegetic_music": "N/A",
    }
    values.update(overrides)
    return MiniMaxH3PromptGuide().build(**values)


def test_auto_text_selects_t2va_and_three_fields():
    prompt, _, report = build()
    assert "Recommended mode: T2VA" in report
    assert prompt.startswith("integrated_multimodal_description: [Shot 1]")
    assert "overall_soundscape:" in prompt
    assert "non_diegetic_music:" in prompt
    assert "subject_definitions:" not in prompt


def test_auto_first_image_selects_i2va():
    prompt, _, report = build(how_images_are_used=FIRST_IMAGE)
    assert "Recommended mode: I2VA" in report
    assert prompt.startswith("For the target video, at 0.00 seconds")


def test_first_and_last_image_uses_exact_duration():
    prompt, _, report = build(how_images_are_used=FIRST_LAST_IMAGES, duration_seconds=7.25)
    assert "Recommended mode: FL2VA" in report
    assert "7.25-second mark" in prompt
    assert "Picture 2" in prompt


def test_last_image_selects_l2va():
    prompt, _, report = build(how_images_are_used=LAST_IMAGE)
    assert "Recommended mode: L2VA" in report
    assert "<Picture 1> (from [Shot 1]) aligns with the 8.00-second mark" in prompt


def test_motion_transfer_builds_ref2va_relationships():
    prompt, rewrite, report = build(
        what_do_you_want=MOTION_GOAL,
        how_images_are_used=MOTION_TARGET_IMAGE,
        how_video_is_used=MOTION_VIDEO,
        reference_assets="Picture 1: a red ceramic robot\nVideo 1: a dancer performing a fast spin",
    )
    assert "Recommended mode: Ref2VA" in report
    assert "Task-type prefix: [reference generation]" in report
    assert prompt.startswith("subject_definitions:")
    assert "<Subject 1> is the target visible subject" in prompt
    assert "<Subject 2> is the action or motion performance" in prompt
    assert "<Subject 2> (appears in [Shot 1]): attribute_transfer" in prompt
    assert "Return exactly these six English sections" in rewrite


def test_video_edit_with_reused_audio_gets_combined_task_prefix():
    prompt, _, report = build(
        how_video_is_used=EDIT_VIDEO,
        how_audio_is_used=COPY_ALL_AUDIO,
        reference_assets="Video 1: original clip\nAudio 1: synchronized source soundtrack",
    )
    assert "Recommended mode: Ref2VA" in report
    assert "[video editing + audio reuse]" in prompt
    assert "The target video is an edited version of <Video 1>." in prompt
    assert "<Audio 1>: fully_copy" in prompt


def test_explicit_edit_goal_infers_the_source_video_role():
    prompt, _, report = build(what_do_you_want=EDIT_GOAL)
    assert "<Video 1> is the source video for the target video edit" in prompt
    assert "<Subject 1>" not in prompt
    assert "The target video is an edited version of <Video 1>." in prompt
    assert "Warnings: none." in report


def test_reference_mode_without_declared_assets_does_not_invent_subject():
    prompt, _, report = build(what_do_you_want=REFERENCE_GOAL)
    assert prompt.startswith("subject_definitions:\n\nsummary:")
    assert "<Subject 1>" not in prompt
    assert "The reference roles use ." not in prompt
    assert "Ref2VA is selected but no image, video, or audio reference is described" in report


def test_audio_only_reference_warns_about_required_visual_input():
    _, _, report = build(how_audio_is_used=COPY_ALL_AUDIO)
    assert "Reference audio cannot be the only Ref2VA media input" in report


def test_asset_parser_normalizes_labels_and_keeps_notes():
    assets, notes = parse_assets(
        "<Picture 1>: hero portrait\nVideo 2 - camera move\nLighting should remain warm"
    )
    assert [asset.label for asset in assets] == ["<Picture 1>", "<Video 2>"]
    assert notes == ["Lighting should remain warm"]


def test_auto_infers_generic_reference_roles_from_inventory():
    prompt, _, report = build(reference_assets="Picture 1: a bronze owl\nVideo 1: a wing movement")
    assert "Recommended mode: Ref2VA" in report
    assert "Resolved image role: Reference appearance, scene, or style" in report
    assert "Resolved video role: Reference its subject, scene, or style" in report
    assert "[reference generation]" in prompt


def test_first_last_reference_adds_a_missing_second_picture():
    prompt, _, _ = build(
        what_do_you_want=REFERENCE_GOAL,
        how_images_are_used=FIRST_LAST_IMAGES,
        reference_assets="Picture 1: opening composition",
    )
    assert "<Picture 1> is the first frame" in prompt
    assert "<Picture 2> is the target video's final frame" in prompt


def test_choose_mode_returns_expected_checkpoint():
    decision = choose_mode(AUTO_GOAL, NO_IMAGE, MOTION_VIDEO, NO_AUDIO)
    assert decision.mode == "Ref2VA"
    assert decision.checkpoint == "H3-Base-Ref2VA"


def test_na_is_not_changed():
    prompt, _, _ = build()
    assert prompt.endswith("non_diegetic_music: N/A")


def test_guide_tooltips_explain_real_input_and_multishot_syntax():
    schema = MiniMaxH3PromptGuide.INPUT_TYPES()
    required = schema["required"]
    assert required["target_description"][1]["default"] == ""
    assert "actual creative request" in required["target_description"][1]["tooltip"]
    assert "[Shot 2] At 00:02.500" in required["shot_and_timing_plan"][1]["tooltip"]
    assert required["shot_and_timing_plan"][1]["advanced"] is True
    assert schema["optional"]["shot_plan"][0] == "MINIMAX_H3_SHOT_PLAN"
    assert "copies the whole signal 1:1" in required["how_audio_is_used"][1]["tooltip"]


def test_empty_target_description_produces_actionable_warning():
    _, _, report = build(target_description="")
    assert "Target description is empty" in report


def test_shot_nodes_chain_float_ranges_and_preview():
    shot_node = MiniMaxH3Shot()
    first, first_preview = shot_node.append_shot(
        start_time=0.0,
        end_time=2.5,
        description="A medium shot establishes the rainy market.",
        camera_direction="Static camera",
        transition="Direct cut",
    )
    chain, preview = shot_node.append_shot(
        start_time=2.5,
        end_time=6.0,
        description="A close-up follows the woman opening an umbrella.",
        camera_direction="Slow push-in",
        transition="Cross-dissolve",
        previous_shots=first,
    )
    assert "00:00.000–00:02.500" in first_preview
    assert "00:02.500–00:06.000" in preview
    assert len(chain["shots"]) == 2


def test_shot_chain_rejects_gaps_and_invalid_first_range():
    shot_node = MiniMaxH3Shot()
    with pytest.raises(ValueError, match="Shot 1 must start"):
        shot_node.append_shot(0.5, 2.0, "Late opening", "", "Direct cut")

    (first, _) = shot_node.append_shot(0.0, 2.0, "Opening", "", "Direct cut")
    with pytest.raises(ValueError, match="exactly when Shot 1 ends"):
        shot_node.append_shot(2.5, 6.0, "Ending", "", "Direct cut", first)
    with pytest.raises(ValueError, match="greater than"):
        shot_node.append_shot(2.0, 2.0, "Ending", "", "Direct cut", first)


def test_connected_shot_chain_writes_real_h3_markers_and_sets_duration():
    shot_node = MiniMaxH3Shot()
    (first, _) = shot_node.append_shot(
        0.0,
        2.5,
        "A medium view establishes the market",
        "Static camera",
        "Direct cut",
    )
    (chain, _) = shot_node.append_shot(
        2.5,
        7.25,
        "A close-up shows the umbrella opening",
        "Slow push-in",
        "Direct cut",
        first,
    )
    prompt, _, report = build(
        how_images_are_used=FIRST_LAST_IMAGES,
        duration_seconds=10.0,
        shot_and_timing_plan="This manual fallback should be ignored",
        shot_plan=chain,
    )
    assert "[Shot 1]" in prompt
    assert "[Shot 2] At 00:02.500" in prompt
    assert "Shot and timing plan:" not in prompt
    assert "7.25-second mark" in prompt
    assert "final end_time overrides duration_seconds" in report
    assert "Manual shot fallback: ignored" in report


def test_connected_shot_chain_must_reach_h3_minimum_duration():
    (short_plan, _) = MiniMaxH3Shot().append_shot(
        0.0, 3.0, "A short test composition", "", "Direct cut"
    )
    with pytest.raises(ValueError, match="4-second minimum"):
        build(shot_plan=short_plan)
