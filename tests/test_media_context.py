import copy

import pytest
import torch

from media_context import (
    FIRST_FRAME_ROLE,
    FULL_RELATION,
    LAST_FRAME_ROLE,
    PARTIAL_RELATION,
    ROLE_CHAIN_TYPE,
    STYLE_ROLE,
    TRANSFER_RELATION,
    UNASSIGNED_ROLE,
    EDIT_ROLE,
    IDENTITY_ROLE,
    ITEM_ROLE,
    MOTION_ROLE,
    PICTURE_MEDIA,
    REFERENCE_CONTEXT_TYPE,
    VIDEO_MEDIA,
    MiniMaxH3EnhancerVisualReference,
    MiniMaxH3VisualReferenceRole,
    reference_entries,
    reference_inventory,
)


def test_new_visual_nodes_require_an_explicit_reference_role():
    role_schema = MiniMaxH3VisualReferenceRole.INPUT_TYPES()["required"]["reference_role"]
    media_schema = MiniMaxH3EnhancerVisualReference.INPUT_TYPES()["required"]["reference_role"]
    assert role_schema[1]["default"] == UNASSIGNED_ROLE
    assert media_schema[1]["default"] == UNASSIGNED_ROLE

    with pytest.raises(ValueError, match="Choose an explicit visual reference role"):
        MiniMaxH3VisualReferenceRole().append_role(
            UNASSIGNED_ROLE,
            "Auto - choose from this role",
            "",
            "",
            "",
            "",
        )
    with pytest.raises(ValueError, match="does not imply identity"):
        MiniMaxH3EnhancerVisualReference().add_reference(
            torch.zeros(1, 32, 32, 3),
            PICTURE_MEDIA,
            UNASSIGNED_ROLE,
            "",
            24.0,
            1.0,
            16,
            768,
        )


def test_picture_reference_preserves_h3_media_and_resizes_only_analysis_copy():
    media = torch.rand(1, 600, 1200, 3)
    context, h3_media, report = MiniMaxH3EnhancerVisualReference().add_reference(
        media=media,
        media_type=PICTURE_MEDIA,
        reference_role=IDENTITY_ROLE,
        notes="Preserve the red jacket",
        source_fps=24.0,
        analysis_fps=1.0,
        max_analysis_frames=16,
        analysis_long_edge=768,
    )
    entry = reference_entries(context)[0]
    assert h3_media.shape == media.shape
    assert torch.equal(h3_media, media)
    assert entry["analysis_media"].shape == (1, 384, 768, 3)
    assert entry["label"] == "<Picture 1>"
    assert entry["h3_input"] == "ref_image_0"
    assert "ref_image_0" in report
    assert MiniMaxH3EnhancerVisualReference.RETURN_TYPES[0] == REFERENCE_CONTEXT_TYPE


def test_object_role_is_available_and_explains_h3_subject_semantics():
    node = MiniMaxH3EnhancerVisualReference()
    assert ITEM_ROLE in node.INPUT_TYPES()["required"]["reference_role"][0]
    context, _, _ = node.add_reference(
        torch.zeros(1, 32, 32, 3),
        PICTURE_MEDIA,
        ITEM_ROLE,
        "Use only the wristwatch",
        24.0,
        1.0,
        16,
        768,
    )
    entry = reference_entries(context)[0]
    assert entry["role"] == ITEM_ROLE


def test_mixed_chain_numbers_each_media_type_for_native_h3_inputs():
    node = MiniMaxH3EnhancerVisualReference()
    picture, _, _ = node.add_reference(
        torch.zeros(1, 64, 64, 3),
        PICTURE_MEDIA,
        IDENTITY_ROLE,
        "",
        24.0,
        1.0,
        16,
        768,
    )
    video, h3_video, report = node.add_reference(
        torch.zeros(48, 64, 96, 3),
        VIDEO_MEDIA,
        MOTION_ROLE,
        "Copy the running rhythm",
        24.0,
        1.0,
        16,
        768,
        picture,
    )
    final, _, report = node.add_reference(
        torch.zeros(1, 80, 80, 3),
        PICTURE_MEDIA,
        IDENTITY_ROLE,
        "Second view",
        24.0,
        1.0,
        16,
        768,
        video,
    )
    entries = reference_entries(final)
    assert [entry["label"] for entry in entries] == [
        "<Picture 1>",
        "<Video 1>",
        "<Picture 2>",
    ]
    assert [entry["h3_input"] for entry in entries] == [
        "ref_image_0",
        "ref_video_0",
        "ref_image_1",
    ]
    assert h3_video.shape[0] == 39
    assert "<Picture 2>" in report


def test_video_analysis_is_timestamped_and_h3_output_is_resampled_to_24_fps():
    frames = torch.arange(24 * 32 * 32 * 3, dtype=torch.float32).reshape(24, 32, 32, 3)
    context, h3_video, _ = MiniMaxH3EnhancerVisualReference().add_reference(
        frames,
        VIDEO_MEDIA,
        MOTION_ROLE,
        "",
        12.0,
        1.0,
        16,
        512,
    )
    entry = reference_entries(context)[0]
    assert h3_video.shape[0] == 39
    assert entry["source_duration_seconds"] == 2.0
    assert entry["duration_seconds"] == 39 / 24
    assert entry["timestamps"] == [0.0, 1.0]
    assert entry["analysis_media"].shape[0] == 2
    assert entry["minimax_timestamps"] == [0.0, 0.5, 1.0, 1.5]
    assert entry["minimax_analysis_media"].shape[0] == 4


def test_video_analysis_caps_frames_across_the_complete_clip():
    frames = torch.zeros(360, 32, 48, 3)
    context, _, _ = MiniMaxH3EnhancerVisualReference().add_reference(
        frames,
        VIDEO_MEDIA,
        MOTION_ROLE,
        "",
        24.0,
        2.0,
        8,
        512,
    )
    timestamps = reference_entries(context)[0]["timestamps"]
    assert len(timestamps) == 8
    assert timestamps[0] == 0.0
    assert timestamps[-1] > 13.0


def test_visual_reference_rejects_ambiguous_picture_batches_and_bad_video_duration():
    node = MiniMaxH3EnhancerVisualReference()
    with pytest.raises(ValueError, match="exactly one image"):
        node.add_reference(
            torch.zeros(2, 32, 32, 3),
            PICTURE_MEDIA,
            IDENTITY_ROLE,
            "",
            24.0,
            1.0,
            16,
            768,
        )
    with pytest.raises(ValueError, match="expects 2-15 seconds"):
        node.add_reference(
            torch.zeros(24, 32, 32, 3),
            VIDEO_MEDIA,
            MOTION_ROLE,
            "",
            24.0,
            1.0,
            16,
            768,
        )
    with pytest.raises(ValueError, match="requires media_type=Video"):
        node.add_reference(
            torch.zeros(1, 32, 32, 3),
            PICTURE_MEDIA,
            EDIT_ROLE,
            "",
            24.0,
            1.0,
            16,
            768,
        )


def test_visual_reference_enforces_total_reference_video_duration():
    node = MiniMaxH3EnhancerVisualReference()
    first, _, _ = node.add_reference(
        torch.zeros(240, 32, 32, 3),
        VIDEO_MEDIA,
        MOTION_ROLE,
        "",
        24.0,
        1.0,
        16,
        512,
    )
    with pytest.raises(ValueError, match="up to 15 seconds.*in total"):
        node.add_reference(
            torch.zeros(144, 32, 32, 3),
            VIDEO_MEDIA,
            MOTION_ROLE,
            "",
            24.0,
            1.0,
            16,
            512,
            first,
        )


def test_role_node_chains_repeatable_bindings_and_preserves_metadata():
    role_node = MiniMaxH3VisualReferenceRole()
    first, preview = role_node.append_role(
        reference_role=IDENTITY_ROLE,
        retention=FULL_RELATION,
        content_group="hero",
        transfer_target="",
        shot_scope="all shots",
        notes="Keep the face",
    )
    roles, preview = role_node.append_role(
        reference_role=ITEM_ROLE,
        retention=PARTIAL_RELATION,
        content_group="hero",
        transfer_target="",
        shot_scope="Shot 2",
        notes="Keep the jacket, change the badge",
        previous_roles=first,
    )
    assert MiniMaxH3VisualReferenceRole.RETURN_TYPES[0] == ROLE_CHAIN_TYPE
    assert len(roles["bindings"]) == 2
    assert roles["bindings"][0]["transfer_target"] == ""
    assert "content group: hero" in preview
    assert "Shot 2" in preview

    context, _, _ = MiniMaxH3EnhancerVisualReference().add_reference(
        torch.zeros(1, 32, 32, 3),
        PICTURE_MEDIA,
        IDENTITY_ROLE,
        "media-wide note",
        24.0,
        1.0,
        16,
        768,
        role_bindings=roles,
    )
    entry = reference_entries(context)[0]
    assert [binding["role"] for binding in entry["bindings"]] == [
        IDENTITY_ROLE,
        ITEM_ROLE,
    ]
    assert entry["notes"] == "media-wide note"


def test_attribute_transfer_requires_distinct_groups_and_visible_final_target():
    role_node = MiniMaxH3VisualReferenceRole()
    with pytest.raises(ValueError, match="source content_group"):
        role_node.append_role(
            reference_role=STYLE_ROLE,
            retention=TRANSFER_RELATION,
            content_group="",
            transfer_target="hero",
            shot_scope="all shots",
            notes="Copy the palette",
        )
    with pytest.raises(ValueError, match="differ"):
        role_node.append_role(
            reference_role=STYLE_ROLE,
            retention=TRANSFER_RELATION,
            content_group="palette",
            transfer_target="palette",
            shot_scope="all shots",
            notes="Copy the palette",
        )

    source_roles, _ = role_node.append_role(
        reference_role=STYLE_ROLE,
        retention=TRANSFER_RELATION,
        content_group="palette",
        transfer_target="hero",
        shot_scope="all shots",
        notes="Copy only the color palette",
    )
    node = MiniMaxH3EnhancerVisualReference()
    source_context, _, _ = node.add_reference(
        torch.zeros(1, 32, 32, 3),
        PICTURE_MEDIA,
        STYLE_ROLE,
        "",
        24.0,
        1.0,
        16,
        768,
        role_bindings=source_roles,
    )
    with pytest.raises(ValueError, match="visible non-transfer target"):
        reference_entries(source_context)

    target_roles, _ = role_node.append_role(
        reference_role=IDENTITY_ROLE,
        retention=FULL_RELATION,
        content_group="hero",
        transfer_target="",
        shot_scope="all shots",
        notes="This is the destination subject",
    )
    final_context, _, _ = node.add_reference(
        torch.ones(1, 32, 32, 3),
        PICTURE_MEDIA,
        IDENTITY_ROLE,
        "",
        24.0,
        1.0,
        16,
        768,
        previous_context=source_context,
        role_bindings=target_roles,
    )
    entries = reference_entries(final_context)
    inventory = reference_inventory(entries)
    assert len(entries) == 2
    assert "transfer_target=hero" in inventory
    assert "do not replace the target's identity" in inventory


def test_endpoint_roles_route_to_image_to_video_and_cannot_mix_with_ref2va():
    node = MiniMaxH3EnhancerVisualReference()
    last_context, _, last_report = node.add_reference(
        torch.zeros(1, 32, 48, 3),
        PICTURE_MEDIA,
        LAST_FRAME_ROLE,
        "",
        24.0,
        1.0,
        16,
        768,
    )
    assert last_context["mode_hint"] == "L2VA"
    assert last_context["entries"][0]["h3_input"] == "last_frame"
    assert "Image to Video.last_frame" in last_report

    both_context, _, report = node.add_reference(
        torch.ones(1, 32, 48, 3),
        PICTURE_MEDIA,
        FIRST_FRAME_ROLE,
        "",
        24.0,
        1.0,
        16,
        768,
        previous_context=last_context,
    )
    entries = reference_entries(both_context)
    by_role = {entry["role"]: entry for entry in entries}
    assert both_context["routing_mode"] == "endpoint"
    assert both_context["mode_hint"] == "FL2VA"
    assert by_role[FIRST_FRAME_ROLE]["label"] == "<Picture 1>"
    assert by_role[FIRST_FRAME_ROLE]["h3_input"] == "first_frame"
    assert by_role[LAST_FRAME_ROLE]["label"] == "<Picture 2>"
    assert report.index("first_frame") < report.index("last_frame")

    with pytest.raises(ValueError, match="cannot share a reference_context"):
        node.add_reference(
            torch.zeros(1, 32, 32, 3),
            PICTURE_MEDIA,
            IDENTITY_ROLE,
            "",
            24.0,
            1.0,
            16,
            768,
            previous_context=both_context,
        )


def test_h3_length_truncates_before_native_floor_and_raw_duration_stays_authoritative():
    node = MiniMaxH3EnhancerVisualReference()
    context, h3_video, report = node.add_reference(
        torch.zeros(240, 32, 48, 3),
        VIDEO_MEDIA,
        MOTION_ROLE,
        "",
        24.0,
        1.0,
        3,
        512,
        h3_length=80,
    )
    entry = reference_entries(context)[0]
    assert h3_video.shape[0] == 90  # 80 snaps up to the next native 17k+5 length.
    assert entry["target_frame_count"] == 90
    assert entry["native_frame_count"] == 90
    assert entry["source_duration_seconds"] == 10.0
    assert entry["duration_seconds"] == 3.75
    assert entry["timestamps"] == [0.0, 2.0, 3.0]
    assert entry["minimax_timestamps"] == [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
    ]
    assert "trimmed from 240" in report

    with pytest.raises(ValueError, match="15 seconds.*in total"):
        node.add_reference(
            torch.zeros(144, 32, 48, 3),
            VIDEO_MEDIA,
            MOTION_ROLE,
            "",
            24.0,
            1.0,
            3,
            512,
            previous_context=context,
            h3_length=39,
        )


def test_v1_context_is_adapted_and_survives_a_v2_chain_append():
    legacy_media = torch.zeros(1, 32, 32, 3)
    legacy = {
        "version": 1,
        "entries": [
            {
                "kind": "image",
                "label": "<Picture 99>",
                "h3_input": "ref_image_98",
                "role": IDENTITY_ROLE,
                "notes": "legacy",
                "analysis_media": legacy_media,
            }
        ],
    }
    adapted = reference_entries(legacy)
    assert adapted[0]["schema_version"] == 2
    assert adapted[0]["legacy_context"] is True
    assert adapted[0]["label"] == "<Picture 1>"
    assert adapted[0]["h3_input"] == "ref_image_0"
    assert adapted[0]["minimax_analysis_media"] is legacy_media

    context, _, _ = MiniMaxH3EnhancerVisualReference().add_reference(
        torch.ones(1, 32, 32, 3),
        PICTURE_MEDIA,
        IDENTITY_ROLE,
        "new",
        24.0,
        1.0,
        16,
        768,
        previous_context=legacy,
    )
    assert len(reference_entries(context)) == 2


def test_context_v2_rejects_stale_routing_and_non_native_video_evidence():
    node = MiniMaxH3EnhancerVisualReference()
    picture_context, _, _ = node.add_reference(
        torch.zeros(1, 32, 32, 3),
        PICTURE_MEDIA,
        IDENTITY_ROLE,
        "",
        24.0,
        1.0,
        16,
        768,
    )
    stale = copy.copy(picture_context)
    stale["entries"] = [copy.copy(picture_context["entries"][0])]
    stale["entries"][0]["h3_input"] = "ref_image_8"
    with pytest.raises(ValueError, match="stale or invalid routing"):
        reference_entries(stale)

    video_context, _, _ = node.add_reference(
        torch.zeros(48, 32, 32, 3),
        VIDEO_MEDIA,
        MOTION_ROLE,
        "",
        24.0,
        1.0,
        16,
        768,
    )
    malformed = copy.copy(video_context)
    malformed["entries"] = [copy.copy(video_context["entries"][0])]
    malformed["entries"][0]["minimax_timestamps"] = [0.0, 0.4, 0.8, 1.2]
    with pytest.raises(ValueError, match="fixed 2 FPS"):
        reference_entries(malformed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_video_batch_indexing_stays_on_the_media_device():
    context, h3_video, _ = MiniMaxH3EnhancerVisualReference().add_reference(
        torch.zeros(48, 16, 16, 3, device="cuda"),
        VIDEO_MEDIA,
        MOTION_ROLE,
        "",
        24.0,
        1.0,
        16,
        768,
    )
    entry = reference_entries(context)[0]
    assert h3_video.device.type == "cuda"
    assert entry["analysis_media"].device.type == "cuda"
    assert entry["minimax_analysis_media"].device.type == "cuda"
