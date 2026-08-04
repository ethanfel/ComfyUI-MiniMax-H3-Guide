import json
import tomllib
from pathlib import Path

import enhancer
import media_context
import nodes
import plan_adapter
import plan_v2
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
        "MiniMax H3 Plan v2 - First and Last Frames.json",
        "MiniMax H3 Plan v2 - Identity and Voice.json",
        "MiniMax H3 Plan v2 - Prompt Builder App.json",
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

    identity_types = [node["type"] for node in identity["nodes"]]
    assert "MiniMaxH3PlanV2ImageReference" in identity_types
    assert "MiniMaxH3PlanV2AudioReference" in identity_types
    assert "MiniMaxH3PlanV2DialogueEvent" in identity_types
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

    for workflow in (identity, endpoints):
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


def test_phase3_nodes_and_legacy_labels_remain_registered():
    assert "MiniMaxH3PlanV2ApplyReferencePlan" in plan_adapter.NODE_CLASS_MAPPINGS
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

    assert metadata["project"]["version"] == "0.13.0"
    assert "There is intentionally no automatic conversion" in migration
    assert "Reference Sheet.selected_audio -> Audio Reference.audio" in migration
    assert "Apply Reference Plan" in migration
