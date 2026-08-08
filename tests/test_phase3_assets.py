import json
import tomllib
from pathlib import Path

import torch

import enhancer
import media_context
import nodes
import plan_adapter
import plan_v2
import prompt_review
import reference_sheet


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIRECTORY = ROOT / "example_workflows"


def _workflows():
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(WORKFLOW_DIRECTORY.glob("*.json"))
    }


def test_phase3_workflow_templates_have_consistent_graph_links():
    workflows = _workflows()

    assert set(workflows) == {
        "MiniMax H3 Plan v2 - Animate Keyframe with Motion Reference.json",
        "MiniMax H3 Plan v2 - Character Replacement.json",
        "MiniMax H3 Plan v2 - Five Shot Keyframe Composition.json",
        "MiniMax H3 Plan v2 - First and Last Frames.json",
        "MiniMax H3 Plan v2 - Identity and Voice.json",
        "MiniMax H3 Plan v2 - Prompt Builder App.json",
        "MiniMax H3 Plan v2 - Video to Audio Foley.json",
        "MiniMax H3 Plan v2 - Video to Audio Foley with Sound Reference.json",
        "MiniMax H3 Plan v2 - Video Extension with Audio Continuity.json",
    }
    for workflow in workflows.values():
        assert workflow["version"] == 0.4
        nodes_by_id = {node["id"]: node for node in workflow["nodes"]}
        assert len(nodes_by_id) == len(workflow["nodes"])
        assert workflow["last_node_id"] == max(nodes_by_id)

        links_by_id = {link[0]: link for link in workflow["links"]}
        assert len(links_by_id) == len(workflow["links"])
        assert workflow["last_link_id"] == max(links_by_id)
        for (
            link_id,
            origin_id,
            origin_slot,
            target_id,
            target_slot,
            link_type,
        ) in workflow["links"]:
            origin = nodes_by_id[origin_id]
            target = nodes_by_id[target_id]
            assert link_id in origin["outputs"][origin_slot]["links"]
            assert target["inputs"][target_slot]["link"] == link_id
            assert origin["outputs"][origin_slot]["type"] == link_type
            target_type = target["inputs"][target_slot]["type"]
            assert target_type == "*" or target_type == link_type


def test_templates_use_exact_roles_and_compiled_plan_v2_chain():
    workflows = _workflows()
    identity = workflows["MiniMax H3 Plan v2 - Identity and Voice.json"]
    endpoints = workflows["MiniMax H3 Plan v2 - First and Last Frames.json"]
    replacement = workflows["MiniMax H3 Plan v2 - Character Replacement.json"]
    continuation = workflows[
        "MiniMax H3 Plan v2 - Video Extension with Audio Continuity.json"
    ]
    composition = workflows[
        "MiniMax H3 Plan v2 - Five Shot Keyframe Composition.json"
    ]
    keyframe_motion = workflows[
        "MiniMax H3 Plan v2 - Animate Keyframe with Motion Reference.json"
    ]
    foley = workflows["MiniMax H3 Plan v2 - Video to Audio Foley.json"]
    foley_reference = workflows[
        "MiniMax H3 Plan v2 - Video to Audio Foley with Sound Reference.json"
    ]

    identity_types = [node["type"] for node in identity["nodes"]]
    assert "MiniMaxH3PlanV2ImageReference" in identity_types
    assert "MiniMaxH3PlanV2AudioReference" in identity_types
    assert "MiniMaxH3PlanV2DialogueEvent" in identity_types
    assert "MiniMaxH3PlanV2PromptReview" in identity_types
    assert "MiniMaxH3PlanV2ApplyReferencePlan" in identity_types
    assert "SamplerCustomAdvanced" in identity_types
    assert "SaveVideo" in identity_types
    identity_values = [
        value
        for node in identity["nodes"]
        for value in (node.get("widgets_values") or [])
    ]
    assert "Define reusable visible content" in identity_values
    assert "Voice timbre and delivery" in identity_values
    assert not any(
        "voice, music, beat, or sound" in str(value) for value in identity_values
    )

    endpoint_values = [
        value
        for node in endpoints["nodes"]
        for value in (node.get("widgets_values") or [])
    ]
    assert "Exact first frame" in endpoint_values
    assert "Exact last frame" in endpoint_values
    assert not any(value == "Identity or appearance" for value in endpoint_values)

    replacement_types = [node["type"] for node in replacement["nodes"]]
    assert "VHS_LoadVideo" in replacement_types
    assert "MiniMaxH3PlanV2AudioReference" in replacement_types
    assert "MiniMaxH3PlanV2CharacterReplacement" in replacement_types
    assert "MiniMaxH3PlanV2PromptReview" in replacement_types
    assert "MiniMaxH3PlanV2ApplyReferencePlan" in replacement_types
    assert "SamplerCustomAdvanced" in replacement_types
    assert "SaveVideo" in replacement_types
    replacement_values = [
        value
        for node in replacement["nodes"]
        for value in (
            node.get("widgets_values", {}).values()
            if isinstance(node.get("widgets_values"), dict)
            else node.get("widgets_values", [])
        )
    ]
    assert "Source video to edit" in replacement_values
    assert "Copy complete signal" in replacement_values
    assert "Replace identity and body; keep source wardrobe" in replacement_values
    assert any(
        "ONE SHOT NODE PER REAL SOURCE SHOT" in str(node.get("title", ""))
        for node in replacement["nodes"]
    )
    assert not any(
        "without introducing a new viewpoint or an additional shot" in str(value)
        for value in replacement_values
    )

    continuation_types = [node["type"] for node in continuation["nodes"]]
    continuation_values = [
        value
        for node in continuation["nodes"]
        for value in (
            node.get("widgets_values", {}).values()
            if isinstance(node.get("widgets_values"), dict)
            else node.get("widgets_values", [])
        )
    ]
    assert "VHS_LoadVideo" in continuation_types
    assert "LoadImage" in continuation_types
    assert "MiniMaxH3PlanV2ImageReference" in continuation_types
    assert "MiniMaxH3PlanV2VideoReference" in continuation_types
    assert "MiniMaxH3PlanV2AudioReference" in continuation_types
    assert "MiniMaxH3PlanV2CharacterReplacement" in continuation_types
    assert "MiniMaxH3PlanV2PromptReview" in continuation_types
    assert "Source video to continue" in continuation_values
    assert "Audio continuity" in continuation_values
    assert "Replace identity and body; keep source wardrobe" in continuation_values
    assert any(
        "never display or animate" in str(value).casefold()
        for value in continuation_values
    )
    assert any("source-derived" in str(value) for value in continuation_values)
    assert any(
        "never restart, replay, repeat, or loop" in str(value)
        for value in continuation_values
    )

    composition_types = [node["type"] for node in composition["nodes"]]
    assert composition_types.count("MiniMaxH3PlanV2Shot") == 5
    assert composition_types.count("MiniMaxH3PlanV2ShotKeyframe") == 5
    assert composition_types.count("MiniMaxH3PlanV2ShotMotionReference") == 1
    assert all(
        "shot_scope" not in {
            input_slot["name"]
            for input_slot in node.get("inputs", [])
        }
        for node in composition["nodes"]
        if node["type"]
        in {"MiniMaxH3PlanV2ShotKeyframe", "MiniMaxH3PlanV2ShotMotionReference"}
    )

    keyframe_motion_types = [node["type"] for node in keyframe_motion["nodes"]]
    assert keyframe_motion_types.count("MiniMaxH3PlanV2ImageReference") == 1
    assert keyframe_motion_types.count("MiniMaxH3PlanV2Shot") == 1
    assert keyframe_motion_types.count("MiniMaxH3PlanV2ShotKeyframe") == 1
    assert keyframe_motion_types.count("MiniMaxH3PlanV2ShotMotionReference") == 1
    assert "MiniMaxH3PlanV2PromptReview" in keyframe_motion_types
    assert "MiniMaxH3PlanV2ApplyReferencePlan" in keyframe_motion_types
    assert "SamplerCustomAdvanced" in keyframe_motion_types
    assert "SaveVideo" in keyframe_motion_types
    keyframe_motion_values = [
        value
        for node in keyframe_motion["nodes"]
        for value in (
            node.get("widgets_values", {}).values()
            if isinstance(node.get("widgets_values"), dict)
            else node.get("widgets_values", [])
        )
    ]
    assert "Compact low-token prompt" in keyframe_motion_values
    assert "Shot opening frame" in keyframe_motion_values
    assert not any(value == "Exact first frame" for value in keyframe_motion_values)

    foley_types = [node["type"] for node in foley["nodes"]]
    assert "VHS_LoadVideo" in foley_types
    assert "MiniMaxH3PlanV2FoleyTarget" in foley_types
    assert "MiniMaxH3PlanV2VideoReference" not in foley_types
    assert "MiniMaxH3PlanV2PromptMerge" in foley_types
    assert "MiniMaxH3PlanV2ApplyReferencePlan" in foley_types
    assert any(
        "video-mask-0 + audio-mask-1" in str(node.get("title", ""))
        for node in foley["nodes"]
    )

    foley_reference_types = [node["type"] for node in foley_reference["nodes"]]
    assert "MiniMaxH3PlanV2FoleyTarget" in foley_reference_types
    assert "MiniMaxH3PlanV2AudioReference" in foley_reference_types
    assert "MiniMaxH3PlanV2VideoReference" not in foley_reference_types
    assert foley_reference_types.count("VAELoader") == 2

    for workflow in (
        identity,
        endpoints,
        replacement,
        continuation,
        composition,
        keyframe_motion,
        foley,
        foley_reference,
    ):
        assert any(
            node["type"] == "MiniMaxH3PlanV2PromptMerge" for node in workflow["nodes"]
        )


def test_prompt_builder_app_exposes_only_real_widget_names():
    workflow = _workflows()["MiniMax H3 Plan v2 - Prompt Builder App.json"]
    assert workflow["extra"]["linearMode"] is True
    nodes_by_id = {node["id"]: node for node in workflow["nodes"]}
    class_by_type = plan_v2.NODE_CLASS_MAPPINGS

    for node_id, widget_name in workflow["extra"]["linearData"]["inputs"]:
        node = nodes_by_id[node_id]
        schema = class_by_type[node["type"]].INPUT_TYPES()
        widget_names = set(schema.get("required", {})) | set(schema.get("optional", {}))
        assert widget_name in widget_names
    for output_id in workflow["extra"]["linearData"]["outputs"]:
        assert output_id in nodes_by_id


def test_character_replacement_template_compiles_with_placeholder_media():
    workflow = _workflows()["MiniMax H3 Plan v2 - Character Replacement.json"]
    nodes = {node["id"]: node for node in workflow["nodes"]}

    plan = plan_v2.MiniMaxH3PlanV2ProjectSetup().start(
        *nodes[1]["widgets_values"]
    )[0]
    plan, _image_handle, _image, _preview = (
        plan_v2.MiniMaxH3PlanV2ImageReference().add_image(
            plan,
            torch.zeros(1, 48, 64, 3),
            *nodes[25]["widgets_values"],
        )
    )
    plan, video_handle, _video, _preview = (
        plan_v2.MiniMaxH3PlanV2VideoReference().add_video(
            plan,
            torch.zeros(144, 48, 64, 3),
            *nodes[3]["widgets_values"],
        )
    )
    audio = {
        "waveform": torch.zeros(1, 1, 192_000),
        "sample_rate": 32_000,
    }
    plan, _audio, _preview = plan_v2.MiniMaxH3PlanV2AudioReference().add_audio(
        plan,
        audio,
        *nodes[4]["widgets_values"],
        video_handle,
    )
    plan, _preview = plan_v2.MiniMaxH3PlanV2CharacterReplacement().add_replacement(
        plan,
        video_handle,
        *nodes[26]["widgets_values"],
    )
    plan = plan_v2.MiniMaxH3PlanV2Shot().add_shot(
        plan, *nodes[5]["widgets_values"]
    )[0]

    prompt, _rewrite, context, report, _length = (
        plan_v2.MiniMaxH3PlanV2PromptMerge().merge(plan)
    )

    assert "[reference generation + video editing + audio reuse]" in prompt
    assert "replace only the source performer" in prompt
    assert "<Video 1>" in prompt and "<Subject 1>" in prompt and "<Audio 1>" in prompt
    assert context["character_replacements"][0]["shot_scope"] == "all"
    assert [entry["route"] for entry in context["compiled"]["routes"]] == [
        "ref_image_0",
        "ref_video_audio_0",
        "ref_video_0",
    ]
    assert "Plan ready: 0 errors" in report


def test_foley_template_compiles_locked_video_without_reference_route():
    workflow = _workflows()["MiniMax H3 Plan v2 - Video to Audio Foley.json"]
    nodes = {node["id"]: node for node in workflow["nodes"]}

    plan = plan_v2.MiniMaxH3PlanV2ProjectSetup().start(
        *nodes[1]["widgets_values"]
    )[0]
    plan, prepared, preview = plan_v2.MiniMaxH3PlanV2FoleyTarget().set_foley_target(
        plan,
        torch.zeros(144, 48, 64, 3),
        *nodes[3]["widgets_values"],
    )
    plan = plan_v2.MiniMaxH3PlanV2Shot().add_shot(
        plan,
        *nodes[4]["widgets_values"],
    )[0]
    prompt, _rewrite, context, report, length = (
        plan_v2.MiniMaxH3PlanV2PromptMerge().merge(plan)
    )

    assert prepared.shape[0] == length == 158
    assert "video=0" in preview and "audio=1" in preview
    assert "picture track remains exactly unchanged" in prompt
    assert "<Video 1>" not in prompt
    assert context["compiled"]["target_task"] == plan_v2.TARGET_FOLEY
    assert context["compiled"]["routes"] == []
    assert "video mask = 0" in report and "audio mask = 1" in report


def test_foley_sound_reference_template_compiles_ref2va_audio_only_route():
    workflow = _workflows()[
        "MiniMax H3 Plan v2 - Video to Audio Foley with Sound Reference.json"
    ]
    nodes = {node["id"]: node for node in workflow["nodes"]}

    plan = plan_v2.MiniMaxH3PlanV2ProjectSetup().start(
        *nodes[1]["widgets_values"]
    )[0]
    plan = plan_v2.MiniMaxH3PlanV2FoleyTarget().set_foley_target(
        plan,
        torch.zeros(144, 48, 64, 3),
        *nodes[3]["widgets_values"],
    )[0]
    plan = plan_v2.MiniMaxH3PlanV2AudioReference().add_audio(
        plan,
        {"waveform": torch.zeros(1, 1, 96000), "sample_rate": 48000},
        *nodes[5]["widgets_values"],
    )[0]
    plan = plan_v2.MiniMaxH3PlanV2Shot().add_shot(
        plan,
        *nodes[6]["widgets_values"],
    )[0]
    prompt, _rewrite, context, report, _length = (
        plan_v2.MiniMaxH3PlanV2PromptMerge().merge(plan)
    )

    assert context["compiled"]["mode"] == "Ref2VA"
    assert [entry["route"] for entry in context["compiled"]["routes"]] == [
        "ref_audio_0"
    ]
    assert "<Audio 1>" in prompt and "<Video 1>" not in prompt
    assert "Sound-effect texture" not in prompt
    assert "sound-effect texture reference" in prompt
    assert "video mask = 0" in report and "audio mask = 1" in report


def test_five_shot_composition_template_compiles_automatic_media_scopes():
    workflow = _workflows()[
        "MiniMax H3 Plan v2 - Five Shot Keyframe Composition.json"
    ]
    nodes = {node["id"]: node for node in workflow["nodes"]}
    plan = plan_v2.MiniMaxH3PlanV2ProjectSetup().start(
        *nodes[1]["widgets_values"]
    )[0]
    plan, _handle, _image, _preview = (
        plan_v2.MiniMaxH3PlanV2ImageReference().add_image(
            plan,
            torch.zeros(1, 48, 64, 3),
            *nodes[3]["widgets_values"],
        )
    )

    for shot_id, keyframe_id in ((4, 6), (7, 9), (10, 12), (15, 17), (18, 20)):
        plan, _preview, shot_handle = plan_v2.MiniMaxH3PlanV2Shot().add_shot(
            plan,
            *nodes[shot_id]["widgets_values"],
        )
        plan, shot_handle, _image, _preview = (
            plan_v2.MiniMaxH3PlanV2ShotKeyframe().attach_keyframe(
                plan,
                shot_handle,
                torch.zeros(1, 48, 64, 3),
                *nodes[keyframe_id]["widgets_values"],
            )
        )
        if shot_id == 10:
            plan, _shot_handle, _video, _preview = (
                plan_v2.MiniMaxH3PlanV2ShotMotionReference().attach_motion(
                    plan,
                    shot_handle,
                    torch.zeros(48, 48, 64, 3),
                    *nodes[14]["widgets_values"],
                )
            )

    prompt, _rewrite, context, report, _length = (
        plan_v2.MiniMaxH3PlanV2PromptMerge().merge(plan)
    )

    keyframe_scopes = [
        asset["shot_scope"]
        for asset in context["assets"]
        if asset["relationship"] == plan_v2.IMAGE_KEYFRAME
    ]
    assert keyframe_scopes == ["1", "2", "3", "4", "5"]
    assert all(
        asset["keyframe_position"] == plan_v2.KEYFRAME_SHOT_OPENING
        for asset in context["assets"]
        if asset["relationship"] == plan_v2.IMAGE_KEYFRAME
    )
    shot_three = next(
        line for line in prompt.splitlines() if line.startswith("[Shot 3]")
    )
    assert "<Subject 2>" in shot_three
    assert "<Video 1>" not in shot_three
    assert "<Subject 2> is the reusable pose, action, and motion from <Video 1>" in prompt
    assert "<Video 1> ([Shot 3]): attribute_transfer" not in prompt
    assert "Plan ready: 0 errors" in report


def test_keyframe_motion_template_compiles_distinct_identity_composition_and_motion_roles():
    workflow = _workflows()[
        "MiniMax H3 Plan v2 - Animate Keyframe with Motion Reference.json"
    ]
    nodes = {node["id"]: node for node in workflow["nodes"]}
    plan = plan_v2.MiniMaxH3PlanV2ProjectSetup().start(
        *nodes[1]["widgets_values"]
    )[0]
    plan = plan_v2.MiniMaxH3PlanV2ImageReference().add_image(
        plan,
        torch.zeros(1, 48, 64, 3),
        *nodes[3]["widgets_values"],
    )[0]
    plan, _preview, shot_handle = plan_v2.MiniMaxH3PlanV2Shot().add_shot(
        plan,
        *nodes[4]["widgets_values"],
    )
    plan, shot_handle, _image, _preview = (
        plan_v2.MiniMaxH3PlanV2ShotKeyframe().attach_keyframe(
            plan,
            shot_handle,
            torch.zeros(1, 48, 64, 3),
            *nodes[6]["widgets_values"],
        )
    )
    plan, _shot_handle, _video, _preview = (
        plan_v2.MiniMaxH3PlanV2ShotMotionReference().attach_motion(
            plan,
            shot_handle,
            torch.zeros(144, 48, 64, 3),
            *nodes[8]["widgets_values"],
        )
    )
    prompt, _rewrite, context, report, _length = (
        plan_v2.MiniMaxH3PlanV2PromptMerge().merge(
            plan,
            *nodes[9]["widgets_values"],
        )
    )

    assert context["compiled"]["mode"] == "Ref2VA"
    assert context["prompt_style"] == plan_v2.PROMPT_STYLE_COMPACT
    assert [entry["route"] for entry in context["compiled"]["routes"]] == [
        "ref_image_0",
        "ref_image_1",
        "ref_video_0",
    ]
    assert context["assets"][1]["relationship"] == plan_v2.IMAGE_KEYFRAME
    assert context["assets"][2]["relationship"] == plan_v2.VIDEO_MOTION
    assert "<Subject 2> is the reusable pose, action, and motion from <Video 1>" in prompt
    assert "Plan ready: 0 errors" in report


def test_video_extension_template_compiles_paired_non_looping_audio_routes():
    workflow = _workflows()[
        "MiniMax H3 Plan v2 - Video Extension with Audio Continuity.json"
    ]
    workflow_nodes = {node["id"]: node for node in workflow["nodes"]}

    plan = plan_v2.MiniMaxH3PlanV2ProjectSetup().start(
        *workflow_nodes[1]["widgets_values"]
    )[0]
    plan, _image_handle, _image, _preview = (
        plan_v2.MiniMaxH3PlanV2ImageReference().add_image(
            plan,
            torch.zeros(1, 48, 64, 3),
            *workflow_nodes[25]["widgets_values"],
        )
    )
    plan, video_handle, _video, _preview = (
        plan_v2.MiniMaxH3PlanV2VideoReference().add_video(
            plan,
            torch.zeros(48, 48, 64, 3),
            *workflow_nodes[3]["widgets_values"],
        )
    )
    audio = {
        "waveform": torch.zeros(1, 1, 64_000),
        "sample_rate": 32_000,
    }
    plan, _audio, _preview = plan_v2.MiniMaxH3PlanV2AudioReference().add_audio(
        plan,
        audio,
        *workflow_nodes[4]["widgets_values"],
        video_handle,
    )
    plan, _preview = plan_v2.MiniMaxH3PlanV2CharacterReplacement().add_replacement(
        plan,
        video_handle,
        *workflow_nodes[26]["widgets_values"],
    )
    plan = plan_v2.MiniMaxH3PlanV2Shot().add_shot(
        plan, *workflow_nodes[5]["widgets_values"]
    )[0]
    prompt, _rewrite, context, report, _length = (
        plan_v2.MiniMaxH3PlanV2PromptMerge().merge(plan)
    )

    assert (
        "[reference generation + video editing + video continuation + audio reference]"
        in prompt
    )
    assert "without restarting, replaying, repeating, or looping" in prompt
    assert "from the first source-derived frame onward" in prompt
    assert "never use it as a target frame, opening composition" in prompt
    assert "evolve the performance forward without restarting or replaying it" in prompt
    assert [entry["route"] for entry in context["compiled"]["routes"]] == [
        "ref_image_0",
        "ref_video_audio_0",
        "ref_video_0",
    ]
    assert len(context["character_replacements"]) == 1
    assert "Plan ready: 0 errors" in report


def test_phase3_nodes_and_legacy_labels_remain_registered():
    assert "MiniMaxH3PlanV2ApplyReferencePlan" in plan_adapter.NODE_CLASS_MAPPINGS
    assert "MiniMaxH3PlanV2PromptReview" in prompt_review.NODE_CLASS_MAPPINGS
    assert plan_adapter.native_h3_compatibility_report.__doc__
    assert reference_sheet.MiniMaxH3ReferenceSheet.RETURN_NAMES[-1] == "selected_audio"

    expected_legacy_ids = {
        "MiniMaxH3PromptGuide": nodes.NODE_DISPLAY_NAME_MAPPINGS,
        "MiniMaxH3Shot": nodes.NODE_DISPLAY_NAME_MAPPINGS,
        "MiniMaxH3TargetTiming": nodes.NODE_DISPLAY_NAME_MAPPINGS,
        "MiniMaxH3VisualReferenceRole": media_context.NODE_DISPLAY_NAME_MAPPINGS,
        "MiniMaxH3EnhancerVisualReference": media_context.NODE_DISPLAY_NAME_MAPPINGS,
        "MiniMaxH3PromptEnhancer": enhancer.NODE_DISPLAY_NAME_MAPPINGS,
        "MiniMaxH3ReferenceSheetVisualReference": reference_sheet.NODE_DISPLAY_NAME_MAPPINGS,
        "MiniMaxH3ReferenceSheetAudioReference": reference_sheet.NODE_DISPLAY_NAME_MAPPINGS,
    }
    for node_id, display_names in expected_legacy_ids.items():
        assert node_id in display_names
        assert "Legacy" in display_names[node_id]

    assert (
        "Legacy"
        not in reference_sheet.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3ReferenceSheet"]
    )
    assert (
        "Legacy"
        not in enhancer.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3GenerationTailLoader"]
    )


def test_phase3_release_and_manual_migration_document_are_present():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    migration = (ROOT / "MIGRATION_TO_PLAN_V2.md").read_text(encoding="utf-8")

    assert metadata["project"]["version"] == "0.15.0"
    assert "There is intentionally no automatic conversion" in migration
    assert "Reference Sheet.selected_audio -> Audio Reference.audio" in migration
    assert "Apply Reference Plan" in migration


def test_plan_v2_browser_logic_handles_auto_transfer_and_clears_hidden_role_data():
    source = (ROOT / "web" / "plan_v2.js").read_text(encoding="utf-8")

    assert "retention === RETENTION_AUTO && contentType === CONTENT_ACTION" in source
    assert 'setWidgetValue(node, "transcript", "")' in source
    assert 'setWidgetValue(node, "target_subject", "")' in source
    assert 'const APPLY_REFERENCE = "MiniMaxH3PlanV2ApplyReferencePlan"' in source
    assert 'const REPLACEMENT = "MiniMaxH3PlanV2CharacterReplacement"' in source
    assert 'const SHOT_KEYFRAME = "MiniMaxH3PlanV2ShotKeyframe"' in source
    assert 'const SHOT_MOTION = "MiniMaxH3PlanV2ShotMotionReference"' in source
    assert 'const FOLEY = "MiniMaxH3PlanV2FoleyTarget"' in source
    assert "const motionSubjectLabels = new Map()" in source
    assert "catalog.motionSubjectLabels.get(node.id)" in source
    assert 'node.addOutput?.("shot_handle", "MINIMAX_H3_SHOT_HANDLE_V2")' in source
    assert 'const VIDEO_CONTINUE = "Source video to continue"' in source
    assert "[VIDEO_EDIT, VIDEO_CONTINUE].includes(sourceUse)" in source
    assert '"Replacement Subject"' in source
    assert "const endpointConflict = hasEndpoint && (requiresRef2va || Boolean(foley))" in source
    assert "unassignedReferenceIds" in source
    assert "endpoint and Ref2VA roles need separate plans" in source
