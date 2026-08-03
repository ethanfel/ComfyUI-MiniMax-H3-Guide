import pytest
import torch

from media_context import (
    EDIT_ROLE,
    IDENTITY_ROLE,
    MOTION_ROLE,
    PICTURE_MEDIA,
    REFERENCE_CONTEXT_TYPE,
    VIDEO_MEDIA,
    MiniMaxH3EnhancerVisualReference,
    reference_entries,
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
    assert h3_video.shape[0] == 48
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
    assert h3_video.shape[0] == 48
    assert entry["duration_seconds"] == 2.0
    assert entry["timestamps"] == [0.0, 1.0]
    assert entry["analysis_media"].shape[0] == 2


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
