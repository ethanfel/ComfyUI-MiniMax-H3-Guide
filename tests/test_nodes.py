from nodes import (
    APPEARANCE_IMAGE,
    AUTO_FIDELITY,
    AUTO_GOAL,
    AUTO_VISUAL_STYLE,
    COPY_ALL_AUDIO,
    COPY_PART_AUDIO,
    EDIT_GOAL,
    EDIT_VIDEO,
    FIRST_IMAGE,
    FIRST_LAST_IMAGES,
    KEYFRAME_IMAGE,
    LAST_IMAGE,
    MOTION_GOAL,
    MOTION_TARGET_IMAGE,
    MOTION_VIDEO,
    NO_AUDIO,
    NO_IMAGE,
    NO_VIDEO,
    REFERENCE_GOAL,
    REFERENCE_AUDIO,
    STORYBOARD_IMAGE,
    STORYBOARD_REFERENCE_IMAGE,
    TRANSFER_FIDELITY,
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
    prompt, _, report, _ = build()
    assert "Recommended mode: T2VA" in report
    assert prompt.startswith("integrated_multimodal_description: [Shot 1]")
    assert "overall_soundscape:" in prompt
    assert "non_diegetic_music:" in prompt
    assert "subject_definitions:" not in prompt


def test_auto_first_image_selects_i2va():
    prompt, _, report, _ = build(how_images_are_used=FIRST_IMAGE)
    assert "Recommended mode: I2VA" in report
    assert prompt.startswith("For the target video, at 0.00 seconds")


def test_first_and_last_image_uses_native_effective_duration():
    prompt, _, report, h3_length = build(
        how_images_are_used=FIRST_LAST_IMAGES, duration_seconds=7.25
    )
    assert "Recommended mode: FL2VA" in report
    assert "7.29-second mark" in prompt
    assert "Picture 2" in prompt
    assert h3_length == 175
    assert "17k+5 frame grid" in report


def test_last_image_selects_l2va():
    prompt, _, report, h3_length = build(how_images_are_used=LAST_IMAGE)
    assert "Recommended mode: L2VA" in report
    assert "<Picture 1> (from [Shot 1]) aligns with the 8.00-second mark" in prompt
    assert h3_length == 192


def test_motion_transfer_builds_ref2va_relationships():
    prompt, rewrite, report, _ = build(
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
    prompt, _, report, _ = build(
        how_video_is_used=EDIT_VIDEO,
        how_audio_is_used=COPY_ALL_AUDIO,
        reference_assets="Video 1: original clip\nAudio 1: synchronized source soundtrack",
    )
    assert "Recommended mode: Ref2VA" in report
    assert "[video editing + audio reuse]" in prompt
    assert "The target video is an edited version of <Video 1>." in prompt
    assert "<Audio 1>: fully_copy" in prompt


def test_explicit_edit_goal_infers_the_source_video_role():
    prompt, _, report, _ = build(what_do_you_want=EDIT_GOAL)
    assert "<Video 1> is the source video for the target video edit" in prompt
    assert "<Subject 1>" not in prompt
    assert "The target video is an edited version of <Video 1>." in prompt
    assert "Warnings: none." in report


def test_reference_mode_without_declared_assets_does_not_invent_subject():
    prompt, _, report, _ = build(what_do_you_want=REFERENCE_GOAL)
    assert prompt.startswith("subject_definitions:\n\nsummary:")
    assert "<Subject 1>" not in prompt
    assert "The reference roles use ." not in prompt
    assert "Ref2VA is selected but no image, video, or audio reference is described" in report


def test_audio_only_reference_warns_about_required_visual_input():
    _, _, report, _ = build(how_audio_is_used=COPY_ALL_AUDIO)
    assert "Reference audio cannot be the only Ref2VA media input" in report


def test_asset_parser_normalizes_labels_and_keeps_notes():
    assets, notes = parse_assets(
        "<Picture 1>: hero portrait\nVideo 2 - camera move\nLighting should remain warm"
    )
    assert [asset.label for asset in assets] == ["<Picture 1>", "<Video 2>"]
    assert notes == ["Lighting should remain warm"]


def test_auto_does_not_invent_roles_from_bare_inventory():
    prompt, _, report, _ = build(
        reference_assets="Picture 1: a bronze owl\nVideo 1: a wing movement"
    )
    assert "Recommended mode: T2VA" in report
    assert "Resolved image role: No image" in report
    assert "Resolved video role: No reference video" in report
    assert "their role is set to none" in report
    assert "<Subject" not in prompt


def test_first_last_reference_adds_a_missing_second_picture():
    prompt, _, _, _ = build(
        what_do_you_want=REFERENCE_GOAL,
        how_images_are_used=FIRST_LAST_IMAGES,
        reference_assets="Picture 1: opening composition",
    )
    assert "<Picture 1> is the first frame" in prompt
    assert "<Picture 2> is the final frame of [Shot 1]" in prompt


def test_choose_mode_returns_expected_checkpoint():
    decision = choose_mode(AUTO_GOAL, NO_IMAGE, MOTION_VIDEO, NO_AUDIO)
    assert decision.mode == "Ref2VA"
    assert decision.checkpoint == "H3-Base-Ref2VA"


def test_na_is_not_changed():
    prompt, _, _, _ = build()
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
    _, _, report, _ = build(target_description="")
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
    prompt, _, report, h3_length = build(
        how_images_are_used=FIRST_LAST_IMAGES,
        duration_seconds=10.0,
        shot_and_timing_plan="This manual fallback should be ignored",
        shot_plan=chain,
    )
    assert "[Shot 1]" in prompt
    assert "[Shot 2] At 00:02.500" in prompt
    assert "Shot and timing plan:" not in prompt
    assert "Picture 2 (from Shot 2) aligns with the 7.29-second mark" in prompt
    assert h3_length == 175
    assert "final end_time overrides duration_seconds" in report
    assert "Manual shot fallback: ignored" in report


def test_connected_shot_chain_must_reach_h3_minimum_duration():
    (short_plan, _) = MiniMaxH3Shot().append_shot(
        0.0, 3.0, "A short test composition", "", "Direct cut"
    )
    with pytest.raises(ValueError, match="4-second minimum"):
        build(shot_plan=short_plan)


def test_manual_multishot_plan_sets_last_frame_alignment_to_actual_final_shot():
    prompt, _, report, _ = build(
        how_images_are_used=LAST_IMAGE,
        shot_and_timing_plan=(
            "Shot 1: opening view. Shot 2, cut at 00:03.000: closer view. "
            "Shot 3, cut at 00:06.000: final composition."
        ),
    )
    assert "<Picture 1> (from [Shot 3]) aligns with the 8.00-second mark" in prompt
    assert "[Shot 2] At 00:03.000" in prompt
    assert "[Shot 3] At 00:06.000" in prompt
    assert "Shot and timing plan:" not in prompt
    assert "Manual shot plan: parsed and validated 3 shot(s)" in report


@pytest.mark.parametrize(
    ("manual_plan", "message"),
    [
        (
            "Shot 1, 00:00-00:02.000: opening. Shot 2, cut at 00:03.000: ending.",
            "must start at 00:02.000",
        ),
        (
            "Shot 1: opening. Shot 3, cut at 00:03.000: ending.",
            "expected Shot 2",
        ),
        (
            "Shot 1: opening. Shot 2: ending.",
            "needs a cut time",
        ),
        (
            "Shot 1, 00:00-00:02.000: opening. Shot 2, cut at 00:08.000: ending.",
            "must be before its ending",
        ),
        (
            "Shot 1, 00:00-00:16.000: opening.",
            "must stay within the H3 range",
        ),
    ],
)
def test_manual_shot_plan_rejects_gaps_ranges_and_missing_times(manual_plan, message):
    with pytest.raises(ValueError, match=message):
        build(shot_and_timing_plan=manual_plan)


def test_manual_shot_plan_rejects_missing_description():
    with pytest.raises(ValueError, match="Shot 1 needs a visible action"):
        build(
            shot_and_timing_plan=(
                "Shot 1, 00:00-00:02.000:   Shot 2, cut at 00:02.000: ending."
            )
        )


def test_native_grid_extends_the_actual_final_shot():
    shot_node = MiniMaxH3Shot()
    first, _ = shot_node.append_shot(0.0, 2.5, "Opening", "", "Direct cut")
    chain, _ = shot_node.append_shot(2.5, 7.25, "Ending", "", "Direct cut", first)
    prompt, _, report, _ = build(
        how_images_are_used=LAST_IMAGE,
        shot_plan=chain,
    )
    assert "<Picture 1> (from [Shot 2]) aligns with the 7.29-second mark" in prompt
    assert "Final [Shot 2] is extended through 00:07.292" in report


def test_manual_explicit_final_end_overrides_duration_instead_of_leaving_a_gap():
    prompt, _, report, h3_length = build(
        how_images_are_used=LAST_IMAGE,
        duration_seconds=8.0,
        shot_and_timing_plan="Shot 1, 00:00-00:06.000: exact ending composition.",
    )
    assert h3_length == 158
    assert "aligns with the 6.58-second mark" in prompt
    assert "explicit final end overrides duration_seconds" in report
    assert "Final [Shot 1] is extended through 00:06.583" in report


def test_camera_direction_is_written_without_a_literal_field_label():
    plan, _ = MiniMaxH3Shot().append_shot(
        0.0,
        6.0,
        "A close view follows the folded letter",
        "The camera pushes in with small amplitude at slow speed.",
        "Direct cut",
    )
    prompt, _, _, _ = build(shot_plan=plan)
    assert "The camera pushes in with small amplitude at slow speed." in prompt
    assert "Camera direction:" not in prompt


def test_storyboard_and_concrete_keyframe_have_distinct_task_types():
    keyframe_prompt, _, keyframe_report, _ = build(
        what_do_you_want=REFERENCE_GOAL,
        how_images_are_used=KEYFRAME_IMAGE,
        reference_assets="Picture 1: exact internal composition",
    )
    storyboard_prompt, _, storyboard_report, _ = build(
        what_do_you_want=REFERENCE_GOAL,
        how_images_are_used=STORYBOARD_REFERENCE_IMAGE,
        reference_assets="Picture 1: two-panel camera plan",
    )
    assert "Task-type prefix: [keyframe completion]" in keyframe_report
    assert "is a concrete keyframe anchor" in keyframe_prompt
    assert "Task-type prefix: [reference generation]" in storyboard_report
    assert "is a storyboard reference defining viewpoint" in storyboard_prompt


def test_legacy_combined_storyboard_role_is_preserved_but_reported():
    prompt, _, report, _ = build(
        what_do_you_want=REFERENCE_GOAL,
        how_images_are_used=STORYBOARD_IMAGE,
        reference_assets="Picture 1: old workflow anchor",
    )
    assert "Task-type prefix: [keyframe completion]" in report
    assert "is a concrete keyframe anchor" in prompt
    assert "legacy combined storyboard/keyframe role" in report


def test_reference_goal_does_not_add_reference_generation_to_an_endpoint_only_task():
    prompt, _, report, _ = build(
        what_do_you_want=REFERENCE_GOAL,
        how_images_are_used=FIRST_IMAGE,
        reference_assets="Picture 1: exact opening frame",
    )
    assert "Task-type prefix: [keyframe completion]" in report
    assert "[keyframe completion + reference generation]" not in prompt


def test_endpoint_and_edit_roles_cannot_receive_attribute_transfer():
    endpoint_prompt, _, endpoint_report, _ = build(
        what_do_you_want=REFERENCE_GOAL,
        how_images_are_used=FIRST_IMAGE,
        reference_fidelity=TRANSFER_FIDELITY,
        reference_assets="Picture 1: exact opening frame",
    )
    edit_prompt, _, edit_report, _ = build(
        how_video_is_used=EDIT_VIDEO,
        reference_fidelity=TRANSFER_FIDELITY,
        reference_assets="Video 1: source clip",
    )
    assert "<Picture 1> ([Shot 1] first frame): fully_preserved" in endpoint_prompt
    assert "Concrete frame anchors are always fully_preserved" in endpoint_report
    assert "<Video 1> (source video editing): partially_preserved" in edit_prompt
    assert "<Video 1> (source video editing): attribute_transfer" not in edit_prompt
    assert "cannot use attribute_transfer" in edit_report


def test_ref2va_last_frame_retention_uses_the_actual_final_shot():
    shot_node = MiniMaxH3Shot()
    first, _ = shot_node.append_shot(0.0, 3.0, "Opening", "", "Direct cut")
    chain, _ = shot_node.append_shot(3.0, 8.0, "Ending", "", "Direct cut", first)
    prompt, _, _, _ = build(
        what_do_you_want=REFERENCE_GOAL,
        how_images_are_used=LAST_IMAGE,
        reference_assets="Picture 1: exact ending",
        shot_plan=chain,
    )
    assert "<Picture 1> is the final frame of [Shot 2]" in prompt
    assert "<Picture 1> ([Shot 2] last frame): fully_preserved" in prompt


def test_multishot_subject_retention_does_not_invent_shot_one_appearance():
    shot_node = MiniMaxH3Shot()
    first, _ = shot_node.append_shot(0.0, 3.0, "Empty room", "", "Direct cut")
    chain, _ = shot_node.append_shot(3.0, 8.0, "The owl enters", "", "Direct cut", first)
    prompt, _, _, _ = build(
        what_do_you_want=REFERENCE_GOAL,
        how_images_are_used=KEYFRAME_IMAGE,
        reference_assets="Subject 1: bronze owl\nPicture 1: exact second-shot composition",
        shot_plan=chain,
    )
    assert "<Subject 1>: fully_preserved" in prompt
    assert "<Subject 1> (appears in [Shot 1])" not in prompt


def test_asset_labels_are_one_based_and_gaps_are_reported():
    with pytest.raises(ValueError, match="labels are one-based"):
        parse_assets("Picture 0: invalid")
    _, _, report, _ = build(reference_assets="Picture 2: missing first input")
    assert "Picture labels must be contiguous from 1" in report
    with pytest.raises(ValueError, match="Active Picture labels must be contiguous"):
        build(
            how_images_are_used=APPEARANCE_IMAGE,
            reference_assets="Picture 2: active but unbound input",
        )


def test_duplicate_asset_labels_are_actionable_errors():
    with pytest.raises(ValueError, match="Duplicate <Picture 1> definition"):
        parse_assets("Picture 1: front view\nPicture 1: side-view note")


def test_frame_role_cardinality_and_subject_only_reference_are_reported():
    _, _, count_report, _ = build(
        what_do_you_want=REFERENCE_GOAL,
        how_images_are_used=FIRST_IMAGE,
        reference_assets="Picture 1: opening\nPicture 2: extra",
    )
    _, _, subject_report, _ = build(
        what_do_you_want=REFERENCE_GOAL,
        reference_assets="Subject 1: ungrounded robot",
    )
    assert "requires exactly 1 ordered Picture label(s); found 2" in count_report
    assert "no image, video, or audio reference is described" in subject_report


def test_incomplete_motion_transfer_is_reported():
    _, _, report, _ = build(how_images_are_used=MOTION_TARGET_IMAGE)
    assert "requires both a target-picture role and a motion-source video role" in report


def test_global_transfer_fidelity_without_a_pair_never_emits_attribute_transfer():
    prompt, _, report, _ = build(
        what_do_you_want=REFERENCE_GOAL,
        how_images_are_used=APPEARANCE_IMAGE,
        reference_fidelity=TRANSFER_FIDELITY,
        reference_assets="Picture 1: bronze owl appearance",
    )
    assert "attribute_transfer" not in prompt
    assert "attribute_transfer requires the explicit motion target + motion source pair" in report


def test_complete_audio_copy_cites_the_track_and_does_not_add_a_new_mix():
    prompt, _, _, _ = build(
        how_images_are_used=APPEARANCE_IMAGE,
        how_audio_is_used=COPY_ALL_AUDIO,
        reference_assets="Picture 1: woman\nAudio 1: complete synchronized soundtrack",
        overall_soundscape="Rain already present in the copied track.",
        non_diegetic_music="Strings already present in the copied track.",
    )
    assert "<Audio 1> is reused 1:1 as the target video's complete final audio track" in prompt
    assert "with no added, removed, or replaced sound layers" in prompt
    assert "no new score is added" in prompt
    assert "Use coherent ambience" not in prompt


@pytest.mark.parametrize(
    ("audio_role", "relationship"),
    [
        (COPY_PART_AUDIO, "Selected time ranges or layers are copied"),
        (REFERENCE_AUDIO, "guides the target without copying samples"),
    ],
)
def test_partial_and_reference_audio_are_cited_in_the_audio_phase(audio_role, relationship):
    prompt, _, _, _ = build(
        how_images_are_used=APPEARANCE_IMAGE,
        how_audio_is_used=audio_role,
        reference_assets="Picture 1: woman\nAudio 1: voice and ambience source",
    )
    assert "<Audio 1>:" in prompt
    assert relationship in prompt


def test_auto_visual_style_derives_from_keyframes_and_old_explicit_value_still_works():
    schema = MiniMaxH3PromptGuide.INPUT_TYPES()
    assert schema["required"]["visual_style"][1]["default"] == AUTO_VISUAL_STYLE
    auto_prompt, _, _, _ = build(
        how_images_are_used=FIRST_IMAGE,
        visual_style=AUTO_VISUAL_STYLE,
    )
    old_prompt, _, _, _ = build(
        how_images_are_used=FIRST_IMAGE,
        visual_style="cinematic, live-action",
    )
    assert "Preserve <Picture 1>'s existing visual style" in auto_prompt
    assert "cinematic, live-action" not in auto_prompt
    assert "cinematic, live-action" in old_prompt


def test_prompt_output_tooltip_recommends_the_enhancer():
    assert "recommended" in MiniMaxH3PromptGuide.OUTPUT_TOOLTIPS[0]
    assert "reviewing every reference" in MiniMaxH3PromptGuide.OUTPUT_TOOLTIPS[0]


def test_rewrite_request_carries_official_dialogue_edge_case_rules():
    _, rewrite, _, _ = build(
        dialogue_lyrics_and_visible_text="S1 voiceover: Keep moving."
    )
    assert "says in an off-screen voiceover" in rewrite
    assert "lips remain completely closed" in rewrite
    assert "<scenetrans>" in rewrite
    assert "<cutoff>" in rewrite
    assert "only to actual vocal sources" in rewrite


def test_duration_tooltip_explains_native_rounding():
    schema = MiniMaxH3PromptGuide.INPUT_TYPES()
    tooltip = schema["required"]["duration_seconds"][1]["tooltip"]
    assert "17k+5" in tooltip
    assert "24 FPS" in tooltip


def test_rewrite_request_carries_dialogue_continuity_and_voiceover_rules():
    _, rewrite, _, _ = build()
    assert "stable (S1), (S2)" in rewrite
    assert "says in an off-screen voiceover" in rewrite
    assert "<scenetrans>" in rewrite
    assert "<cutoff>" in rewrite
    assert "lips remain completely closed" in rewrite
