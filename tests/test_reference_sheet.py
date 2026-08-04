import json
import math
import wave
from pathlib import Path

import numpy
import pytest
from PIL import Image

from media_context import (
    AUTO_RELATION,
    FULL_RELATION,
    IDENTITY_ROLE,
    ITEM_ROLE,
    MiniMaxH3VisualReferenceRole,
    reference_entries,
)
from nodes import (
    AUTO_FIDELITY,
    AUTO_GOAL,
    AUTO_VISUAL_STYLE,
    NO_AUDIO,
    NO_IMAGE,
    NO_VIDEO,
    MiniMaxH3PromptGuide,
    MiniMaxH3Shot,
)
from reference_sheet import (
    AUDIO_REFERENCE_CONTEXT_TYPE,
    AUDIO_REFERENCE,
    CREATE_SHEET,
    LOAD_SHEET,
    NO_SUGGESTED_ROLE,
    UPDATE_SHEET,
    USE_SHEET_ROLE,
    MiniMaxH3ReferenceSheetAudioAsset,
    MiniMaxH3ReferenceSheetAudioReference,
    MiniMaxH3ReferenceSheetImageAsset,
    MiniMaxH3ReferenceSheetLibrary,
    MiniMaxH3ReferenceSheetVisualReference,
    NODE_CLASS_MAPPINGS,
    REFERENCE_SHEET_DRAFT_TYPE,
    REFERENCE_SHEET_TYPE,
    audio_reference_entries,
    reference_sheet_manifest,
)


def _write_image(path: Path, color=(192, 64, 32)):
    pixels = numpy.zeros((24, 32, 3), dtype=numpy.uint8)
    pixels[:, :] = color
    Image.fromarray(pixels).save(path)


def _write_wav(path: Path, seconds=3.0, sample_rate=8000):
    frame_count = round(seconds * sample_rate)
    time = numpy.arange(frame_count, dtype=numpy.float32) / sample_rate
    samples = (numpy.sin(2 * math.pi * 220 * time) * 0.2 * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


@pytest.fixture
def sheet_environment(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    library_dir = tmp_path / "reference_sheets"
    input_dir.mkdir()
    library_dir.mkdir()
    _write_image(input_dir / "primary.png")
    _write_image(input_dir / "detail.png", (32, 96, 208))
    _write_wav(input_dir / "voice.wav")
    monkeypatch.setenv("MINIMAX_H3_INPUT_DIR", str(input_dir))
    monkeypatch.setenv("MINIMAX_H3_REFERENCE_SHEETS_DIR", str(library_dir))
    return input_dir, library_dir


def _build_sheet(sheet_environment, *, audio_seconds=3.0):
    input_dir, _library_dir = sheet_environment
    if audio_seconds != 3.0:
        _write_wav(input_dir / "voice.wav", seconds=audio_seconds)
    image_assets, preview, _ = MiniMaxH3ReferenceSheetImageAsset().add_image(
        "primary.png",
        "primary",
        "Front view of the reusable subject",
        IDENTITY_ROLE,
        "main",
    )
    assert preview.shape == (1, 24, 32, 3)
    assets, audio, _ = MiniMaxH3ReferenceSheetAudioAsset().add_audio(
        "voice.wav",
        "voice",
        "Warm low speaking voice with a French accent",
        AUDIO_REFERENCE,
        "voice",
        previous_assets=image_assets,
    )
    assert audio["waveform"].shape[0:2] == (1, 1)
    sheet, report = MiniMaxH3ReferenceSheetLibrary().manage(
        CREATE_SHEET,
        "(no saved reference sheets)",
        "Studio Subject",
        "Reusable identity and voice references",
        "identity, voice",
        False,
        assets=assets,
    )
    return sheet, report


def _guide_build(**overrides):
    values = {
        "what_do_you_want": AUTO_GOAL,
        "target_description": "The subject enters a studio and speaks.",
        "how_images_are_used": NO_IMAGE,
        "how_video_is_used": NO_VIDEO,
        "how_audio_is_used": NO_AUDIO,
        "reference_fidelity": AUTO_FIDELITY,
        "reference_assets": "",
        "duration_seconds": 8.0,
        "visual_style": AUTO_VISUAL_STYLE,
        "shot_and_timing_plan": "",
        "camera_direction": "",
        "dialogue_lyrics_and_visible_text": "",
        "overall_soundscape": "Quiet studio ambience.",
        "non_diegetic_music": "N/A",
    }
    values.update(overrides)
    return MiniMaxH3PromptGuide().build(**values)


def _four_shots():
    node = MiniMaxH3Shot()
    chain, _ = node.append_shot(0.0, 2.0, "Empty studio", "", "Direct cut")
    chain, _ = node.append_shot(
        2.0,
        4.0,
        "The door opens",
        "",
        "Direct cut",
        previous_shots=chain,
    )
    chain, _ = node.append_shot(
        4.0,
        6.0,
        "The subject enters",
        "",
        "Direct cut",
        previous_shots=chain,
    )
    chain, _ = node.append_shot(
        6.0,
        8.0,
        "The subject speaks",
        "",
        "Direct cut",
        previous_shots=chain,
    )
    return chain


def test_sheet_create_load_and_self_contained_media(sheet_environment):
    input_dir, _library_dir = sheet_environment
    sheet, report = _build_sheet(sheet_environment)
    manifest, root = reference_sheet_manifest(sheet)

    assert manifest["name"] == "Studio Subject"
    assert manifest["tags"] == ["identity", "voice"]
    assert [asset["key"] for asset in manifest["assets"]] == ["primary", "voice"]
    assert all((root / asset["file"]).is_file() for asset in manifest["assets"])
    assert str(root) in report
    assert "primary (image" in report
    assert "voice (audio" in report

    (input_dir / "primary.png").unlink()
    (input_dir / "voice.wav").unlink()
    selection = f"{manifest['name']} [{manifest['id'][:8]}]"
    loaded, loaded_report = MiniMaxH3ReferenceSheetLibrary().manage(
        LOAD_SHEET,
        selection,
        "",
        "",
        "",
        False,
    )
    loaded_manifest, loaded_root = reference_sheet_manifest(loaded)
    assert loaded_manifest == manifest
    assert loaded_root == root
    assert "Studio Subject" in loaded_report


def test_duplicate_asset_keys_are_rejected_case_insensitively(sheet_environment):
    assets, _, _ = MiniMaxH3ReferenceSheetImageAsset().add_image(
        "primary.png",
        "Main",
        "",
        NO_SUGGESTED_ROLE,
        "main",
    )
    with pytest.raises(ValueError, match="duplicated"):
        MiniMaxH3ReferenceSheetAudioAsset().add_audio(
            "voice.wav",
            "main",
            "",
            AUDIO_REFERENCE,
            "voice",
            previous_assets=assets,
        )


def test_sheet_update_requires_confirmation_and_checksum_tampering_is_detected(
    sheet_environment,
):
    sheet, _ = _build_sheet(sheet_environment)
    manifest, root = reference_sheet_manifest(sheet)
    selection = f"{manifest['name']} [{manifest['id'][:8]}]"
    assets, _, _ = MiniMaxH3ReferenceSheetImageAsset().add_image(
        "detail.png",
        "detail",
        "Alternate detail",
        ITEM_ROLE,
        "detail",
    )

    with pytest.raises(ValueError, match="confirm_update=true"):
        MiniMaxH3ReferenceSheetLibrary().manage(
            UPDATE_SHEET,
            selection,
            "",
            "",
            "",
            False,
            assets=assets,
        )

    updated, _ = MiniMaxH3ReferenceSheetLibrary().manage(
        UPDATE_SHEET,
        selection,
        "Studio Subject Revised",
        "Updated description",
        "detail",
        True,
        assets=assets,
    )
    updated_manifest, updated_root = reference_sheet_manifest(updated)
    assert updated_manifest["id"] == manifest["id"]
    assert updated_manifest["name"] == "Studio Subject Revised"
    assert [asset["key"] for asset in updated_manifest["assets"]] == ["detail"]
    assert updated_root == root

    updated_selection = (
        f"{updated_manifest['name']} [{updated_manifest['id'][:8]}]"
    )
    preserved, _ = MiniMaxH3ReferenceSheetLibrary().manage(
        UPDATE_SHEET,
        updated_selection,
        "",
        "",
        "",
        True,
        assets=assets,
    )
    preserved_manifest, _ = reference_sheet_manifest(preserved)
    assert preserved_manifest["description"] == "Updated description"
    assert preserved_manifest["tags"] == ["detail"]

    first_asset = root / preserved_manifest["assets"][0]["file"]
    first_asset.write_bytes(first_asset.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum failed"):
        reference_sheet_manifest(preserved)


def test_saved_visual_and_audio_assets_feed_guide_with_workflow_scopes(
    sheet_environment,
):
    sheet, _ = _build_sheet(sheet_environment)
    visual_context, image, visual_report = (
        MiniMaxH3ReferenceSheetVisualReference().use_image(
            sheet,
            "primary",
            USE_SHEET_ROLE,
            FULL_RELATION,
            "",
            "",
            "3,4",
            "Keep the identity stable",
            768,
        )
    )
    entries = reference_entries(visual_context)
    binding = entries[0]["bindings"][0]
    assert image.shape == (1, 24, 32, 3)
    assert binding["role"] == IDENTITY_ROLE
    assert binding["content_group"].startswith("sheet_")
    assert binding["shot_scope"] == "3,4"
    assert "workflow roles=Identity or appearance" in visual_report

    audio_context, audio, audio_report = (
        MiniMaxH3ReferenceSheetAudioReference().use_audio(
            sheet,
            "voice",
            "Use the sheet asset's suggested audio role",
            "3,4",
            "Use this voice only when the subject speaks",
        )
    )
    audio_entries = audio_reference_entries(audio_context)
    assert audio["waveform"].shape[-1] == 24000
    assert audio_entries[0]["label"] == "<Audio 1>"
    assert audio_entries[0]["h3_input"] == "ref_audio_0"
    assert "Reference Sheet audio references" in audio_report

    prompt, rewrite, report, _ = _guide_build(
        reference_context=visual_context,
        audio_context=audio_context,
        shot_plan=_four_shots(),
    )
    assert "[reference generation + audio reference]" in prompt
    assert "<Subject 1> (Shots 3, 4): fully_preserved" in prompt
    assert "<Audio 1> (Shots 3, 4): reference" in prompt
    shot_1, remainder = prompt.split("[Shot 1]", 1)[1].split("[Shot 2]", 1)
    shot_2, remainder = remainder.split("[Shot 3]", 1)
    shot_3, shot_4 = remainder.split("[Shot 4]", 1)
    assert "<Subject 1>" not in shot_1 + shot_2
    assert "<Audio 1>" not in shot_1 + shot_2
    assert "<Subject 1>" in shot_3 and "<Audio 1>" in shot_3
    assert "<Subject 1>" in shot_4 and "<Audio 1>" in shot_4
    assert "ref_audio_0" in report
    assert "Saved Reference Sheet audio references" in rewrite


def test_audio_context_requires_visual_context_and_valid_shot_numbers(sheet_environment):
    sheet, _ = _build_sheet(sheet_environment)
    audio_context, _, _ = MiniMaxH3ReferenceSheetAudioReference().use_audio(
        sheet,
        "voice",
        AUDIO_REFERENCE,
        "5",
        "",
    )
    with pytest.raises(ValueError, match="also needs a visual reference_context"):
        _guide_build(audio_context=audio_context)

    visual_context, _, _ = MiniMaxH3ReferenceSheetVisualReference().use_image(
        sheet,
        "primary",
        USE_SHEET_ROLE,
        AUTO_RELATION,
        "",
        "",
        "all",
        "",
        768,
    )
    with pytest.raises(ValueError, match="Audio shot_scope refers to a Shot"):
        _guide_build(
            reference_context=visual_context,
            audio_context=audio_context,
            shot_plan=_four_shots(),
        )


def test_sheet_visual_accepts_a_repeatable_role_chain(sheet_environment):
    sheet, _ = _build_sheet(sheet_environment)
    role_node = MiniMaxH3VisualReferenceRole()
    first, _ = role_node.append_role(
        IDENTITY_ROLE,
        FULL_RELATION,
        "main_subject",
        "",
        "3",
        "Preserve identity",
    )
    roles, _ = role_node.append_role(
        ITEM_ROLE,
        FULL_RELATION,
        "main_subject",
        "",
        "4",
        "Preserve clothing",
        previous_roles=first,
    )
    context, _, report = MiniMaxH3ReferenceSheetVisualReference().use_image(
        sheet,
        "primary",
        USE_SHEET_ROLE,
        AUTO_RELATION,
        "",
        "",
        "",
        "",
        768,
        role_bindings=roles,
    )
    bindings = reference_entries(context)[0]["bindings"]
    assert [binding["role"] for binding in bindings] == [IDENTITY_ROLE, ITEM_ROLE]
    assert [binding["shot_scope"] for binding in bindings] == ["3", "4"]
    assert "workflow roles=Identity or appearance, Object" in report


def test_audio_reference_duration_is_checked_when_used(sheet_environment):
    sheet, _ = _build_sheet(sheet_environment, audio_seconds=1.0)
    with pytest.raises(ValueError, match="2 through 15 seconds"):
        MiniMaxH3ReferenceSheetAudioReference().use_audio(
            sheet,
            "voice",
            AUDIO_REFERENCE,
            "",
            "",
        )


def test_reference_sheet_manifest_uses_relative_paths(sheet_environment):
    sheet, _ = _build_sheet(sheet_environment)
    manifest, root = reference_sheet_manifest(sheet)
    serialized = json.dumps(manifest)
    assert str(root) not in serialized
    assert all(not Path(asset["file"]).is_absolute() for asset in manifest["assets"])


def test_reference_sheet_node_contracts_and_guide_socket_order():
    assert len(NODE_CLASS_MAPPINGS) == 5
    assert MiniMaxH3ReferenceSheetImageAsset.RETURN_TYPES[0] == REFERENCE_SHEET_DRAFT_TYPE
    assert MiniMaxH3ReferenceSheetAudioAsset.RETURN_TYPES[0] == REFERENCE_SHEET_DRAFT_TYPE
    assert MiniMaxH3ReferenceSheetLibrary.RETURN_TYPES[0] == REFERENCE_SHEET_TYPE
    assert (
        MiniMaxH3ReferenceSheetAudioReference.RETURN_TYPES[0]
        == AUDIO_REFERENCE_CONTEXT_TYPE
    )
    optional = list(MiniMaxH3PromptGuide.INPUT_TYPES()["optional"])
    assert optional == [
        "shot_plan",
        "timing_context",
        "reference_context",
        "audio_context",
    ]
