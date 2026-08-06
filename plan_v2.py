"""Typed MiniMax H3 planning chain and deterministic prompt compiler.

The v2 nodes keep workflow semantics in a versioned Python dictionary until
Prompt Merge assigns the final H3 labels. Media tensors travel through that
dictionary by reference; workflow JSON stores only the normal ComfyUI nodes and
links.
"""

from __future__ import annotations

import math
import re
from typing import Any


PLAN_VERSION = 2
PLAN_TYPE = "MINIMAX_H3_PLAN_V2"
REFERENCE_HANDLE_TYPE = "MINIMAX_H3_REFERENCE_HANDLE_V2"

H3_FPS = 24
H3_FRAME_MODULUS = 17
H3_FRAME_OFFSET = 5

PHASE_SETUP = "setup"
PHASE_TIMELINE = "timeline"
PHASE_COMPILED = "compiled"

UNASSIGNED_IMAGE_USE = "Choose an image relationship"
IMAGE_DEFINE_VISIBLE = "Define reusable visible content"
IMAGE_FIRST_FRAME = "Exact first frame"
IMAGE_LAST_FRAME = "Exact last frame"
IMAGE_KEYFRAME = "Concrete keyframe / composition anchor"
IMAGE_STORYBOARD = "Storyboard / shot planning"
IMAGE_USES = [
    UNASSIGNED_IMAGE_USE,
    IMAGE_DEFINE_VISIBLE,
    IMAGE_FIRST_FRAME,
    IMAGE_LAST_FRAME,
    IMAGE_KEYFRAME,
    IMAGE_STORYBOARD,
]

UNASSIGNED_VIDEO_USE = "Choose a video relationship"
VIDEO_EDIT = "Source video to edit"
VIDEO_CONTINUE = "Source video to continue"
VIDEO_DEFINE_VISIBLE = "Define reusable visible content"
VIDEO_MOTION = "Motion or action reference"
VIDEO_STRUCTURE = "Camera, cuts, rhythm, or temporal-structure reference"
VIDEO_USES = [
    UNASSIGNED_VIDEO_USE,
    VIDEO_EDIT,
    VIDEO_CONTINUE,
    VIDEO_DEFINE_VISIBLE,
    VIDEO_MOTION,
    VIDEO_STRUCTURE,
]

REPLACEMENT_IDENTITY_KEEP_BODY_WARDROBE = (
    "Replace identity; keep source body and wardrobe"
)
REPLACEMENT_IDENTITY_BODY_KEEP_WARDROBE = (
    "Replace identity and body; keep source wardrobe"
)
REPLACEMENT_COMPLETE_APPEARANCE = (
    "Replace complete referenced appearance including wardrobe"
)
CHARACTER_REPLACEMENT_APPEARANCE_POLICIES = [
    REPLACEMENT_IDENTITY_KEEP_BODY_WARDROBE,
    REPLACEMENT_IDENTITY_BODY_KEEP_WARDROBE,
    REPLACEMENT_COMPLETE_APPEARANCE,
]

UNASSIGNED_CONTENT_TYPE = "Choose visible content type"
CONTENT_IDENTITY = "Identity or appearance"
CONTENT_OBJECT = "Object, prop, clothing, interface, or visual effect"
CONTENT_SCENE = "Scene or environment"
CONTENT_STYLE = "Visual style"
CONTENT_ACTION = "Pose, expression, action, or motion"
CONTENT_TYPES = [
    UNASSIGNED_CONTENT_TYPE,
    CONTENT_IDENTITY,
    CONTENT_OBJECT,
    CONTENT_SCENE,
    CONTENT_STYLE,
    CONTENT_ACTION,
]

UNASSIGNED_AUDIO_USE = "Choose an audio relationship"
AUDIO_VOICE = "Voice timbre and delivery"
AUDIO_MUSIC = "Background-music style"
AUDIO_BEAT = "Beat or rhythm"
AUDIO_SFX = "Sound-effect texture"
AUDIO_CONTENT = "Dialogue or lyric content"
AUDIO_CONTINUITY = "Audio continuity"
AUDIO_COPY_COMPLETE = "Copy complete signal"
AUDIO_COPY_PARTIAL = "Copy selected part or layers"
AUDIO_BROAD = "Broad audio inspiration"
AUDIO_USES = [
    UNASSIGNED_AUDIO_USE,
    AUDIO_VOICE,
    AUDIO_MUSIC,
    AUDIO_BEAT,
    AUDIO_SFX,
    AUDIO_CONTENT,
    AUDIO_CONTINUITY,
    AUDIO_COPY_COMPLETE,
    AUDIO_COPY_PARTIAL,
    AUDIO_BROAD,
]

RETENTION_AUTO = "Auto for this relationship"
RETENTION_FULL = "fully_preserved"
RETENTION_PARTIAL = "partially_preserved"
RETENTION_TRANSFER = "attribute_transfer"
RETENTION_WEAK = "weak_reference"
VISUAL_RETENTIONS = [
    RETENTION_AUTO,
    RETENTION_FULL,
    RETENTION_PARTIAL,
    RETENTION_TRANSFER,
    RETENTION_WEAK,
]

AUDIO_FULL_COPY = "fully_copy"
AUDIO_PARTIAL_COPY = "partially_copy"
AUDIO_REFERENCE = "reference"
AUDIO_WEAK_REFERENCE = "weak_reference"

SHOT_TRANSITIONS = ["Direct cut", "Cross-dissolve", "Fade", "Wipe"]
DIALOGUE_MODES = ["On-screen speech", "Off-screen speech", "Voiceover"]

MODE_T2VA = "T2VA"
MODE_I2VA = "I2VA"
MODE_FL2VA = "FL2VA"
MODE_L2VA = "L2VA"
MODE_REF2VA = "Ref2VA"

_REFERENCE_TOKEN_RE = re.compile(
    r"<\s*(Subject|Picture|Video|Audio)\s+(\d+)\s*>",
    re.IGNORECASE,
)
_SCOPE_PREFIX_RE = re.compile(r"\bshots?\b", re.IGNORECASE)


def _clean_inline(value: Any, fallback: str = "") -> str:
    text = " ".join(str(value or "").strip().split())
    return text or fallback


def _clean_block(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def _sentence(value: Any, fallback: str = "") -> str:
    text = _clean_inline(value, fallback)
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _clause(value: Any, fallback: str = "") -> str:
    return _clean_inline(value, fallback).rstrip(".!?")


def _alias_key(value: Any) -> str:
    return _clean_inline(value).casefold()


def _format_timestamp(seconds: float) -> str:
    total_milliseconds = round(float(seconds) * 1000)
    minutes, remainder = divmod(total_milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def native_frame_count(duration_seconds: float) -> int:
    """Round a duration upward to the native H3 17k+5 frame grid."""

    duration = float(duration_seconds)
    if not math.isfinite(duration) or duration < 4.0 or duration > 15.0:
        raise ValueError("H3 target duration must be from 4 through 15 seconds.")
    requested = max(H3_FRAME_OFFSET, math.ceil(duration * H3_FPS - 1e-9))
    return requested + (H3_FRAME_OFFSET - requested) % H3_FRAME_MODULUS


def _copy_plan(plan: dict) -> dict:
    """Copy plan containers without cloning large media tensors."""

    copied = dict(plan)
    copied["project"] = dict(plan["project"])
    for key in (
        "assets",
        "bindings",
        "character_replacements",
        "audio_relationships",
        "shots",
        "dialogue_events",
    ):
        copied[key] = [dict(entry) for entry in plan.get(key, [])]
    if isinstance(plan.get("compiled"), dict):
        copied["compiled"] = dict(plan["compiled"])
    return copied


def validated_plan(plan: Any, *, allowed_phases: set[str] | None = None) -> dict:
    """Validate the stable outer plan contract and return a shallow safe copy."""

    if not isinstance(plan, dict) or plan.get("version") != PLAN_VERSION:
        raise ValueError("h3_plan must come from MiniMax H3 Project Setup (Plan v2).")
    phase = plan.get("phase")
    if phase not in {PHASE_SETUP, PHASE_TIMELINE, PHASE_COMPILED}:
        raise ValueError("h3_plan contains an invalid workflow phase.")
    if allowed_phases is not None and phase not in allowed_phases:
        expected = " or ".join(sorted(allowed_phases))
        raise ValueError(
            f"This node cannot follow the plan's {phase!r} phase; connect it during {expected}."
        )
    project = plan.get("project")
    if not isinstance(project, dict):
        raise ValueError("h3_plan is missing its Project Setup data.")
    try:
        requested_duration = float(project["duration_seconds"])
        h3_length = int(project["h3_length"])
        effective_duration = float(project["effective_duration"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("h3_plan contains invalid Project duration data.") from error
    expected_length = native_frame_count(requested_duration)
    if h3_length != expected_length or not math.isclose(
        effective_duration, h3_length / H3_FPS, abs_tol=0.0005
    ):
        raise ValueError("h3_plan contains stale native duration data.")

    normalized = dict(plan)
    normalized.setdefault("character_replacements", [])
    for key in (
        "assets",
        "bindings",
        "character_replacements",
        "audio_relationships",
        "shots",
        "dialogue_events",
    ):
        entries = normalized.get(key)
        if not isinstance(entries, (list, tuple)) or not all(
            isinstance(entry, dict) for entry in entries
        ):
            raise ValueError(f"h3_plan contains an invalid {key} collection.")

    asset_ids = [entry.get("asset_id") for entry in plan["assets"]]
    if any(not isinstance(asset_id, str) or not asset_id for asset_id in asset_ids):
        raise ValueError("Every plan asset needs a stable internal asset_id.")
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("h3_plan contains duplicate internal asset IDs.")
    return _copy_plan(normalized)


def _new_plan(
    initial_prompt: str,
    duration_seconds: float,
    visual_style: str,
    overall_soundscape: str,
    non_diegetic_music: str,
) -> dict:
    h3_length = native_frame_count(duration_seconds)
    return {
        "version": PLAN_VERSION,
        "phase": PHASE_SETUP,
        "project": {
            "initial_prompt": _clean_block(initial_prompt),
            "duration_seconds": float(duration_seconds),
            "h3_length": h3_length,
            "effective_duration": h3_length / H3_FPS,
            "visual_style": _clean_inline(visual_style),
            "overall_soundscape": _clean_block(overall_soundscape),
            "non_diegetic_music": _clean_block(non_diegetic_music, "N/A"),
        },
        "assets": [],
        "bindings": [],
        "character_replacements": [],
        "audio_relationships": [],
        "shots": [],
        "dialogue_events": [],
    }


def _next_asset_id(plan: dict, media_kind: str) -> str:
    number = 1 + sum(asset["media_kind"] == media_kind for asset in plan["assets"])
    return f"{media_kind}-{number}"


def _next_binding_id(plan: dict) -> str:
    return f"binding-{len(plan['bindings']) + 1}"


def _next_character_replacement_id(plan: dict) -> str:
    return f"character-replacement-{len(plan['character_replacements']) + 1}"


def _reference_handle(asset: dict) -> dict:
    return {
        "version": PLAN_VERSION,
        "asset_id": asset["asset_id"],
        "media_kind": asset["media_kind"],
    }


def _resolved_handle(plan: dict, handle: Any, *, allowed_kinds: set[str]) -> dict:
    if not isinstance(handle, dict) or handle.get("version") != PLAN_VERSION:
        raise ValueError("reference_handle must come from a Plan v2 reference node.")
    asset_id = handle.get("asset_id")
    asset = next(
        (entry for entry in plan["assets"] if entry["asset_id"] == asset_id), None
    )
    if asset is None:
        raise ValueError("reference_handle does not belong to the connected h3_plan.")
    if asset["media_kind"] not in allowed_kinds:
        choices = " or ".join(sorted(allowed_kinds))
        raise ValueError(f"This binding needs a {choices} reference handle.")
    return asset


def _validate_image(image: Any) -> None:
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) != 4 or int(shape[0]) != 1 or int(shape[-1]) < 3:
        raise ValueError(
            "H3 Image Reference needs exactly one IMAGE in [1, height, width, channels] form."
        )


def _prepare_video_frames(video_frames: Any, source_fps: float, h3_length: int):
    shape = getattr(video_frames, "shape", None)
    if shape is None or len(shape) != 4 or int(shape[0]) < 1 or int(shape[-1]) < 3:
        raise ValueError(
            "H3 Video Reference needs an IMAGE frame batch in [frames, height, width, channels] form."
        )
    fps = float(source_fps)
    if not math.isfinite(fps) or fps <= 0.0 or fps > 240.0:
        raise ValueError(
            "Video source_fps must be a finite value above 0 and at most 240."
        )
    source_count = int(shape[0])
    source_duration = source_count / fps
    if source_duration < 2.0 - 0.0005 or source_duration > 15.0 + 0.0005:
        raise ValueError(
            "Every H3 reference video must last from 2 through 15 seconds."
        )

    resampled_count = max(1, math.ceil(source_duration * H3_FPS - 1e-9))
    indices = [
        min(source_count - 1, int(math.floor(index * fps / H3_FPS)))
        for index in range(resampled_count)
    ]
    try:
        import torch

        index_tensor = torch.tensor(
            indices, dtype=torch.long, device=video_frames.device
        )
        resampled = video_frames.index_select(0, index_tensor)
    except (ImportError, AttributeError):
        resampled = video_frames[indices]

    capped_count = min(int(resampled.shape[0]), int(h3_length))
    native_count = capped_count
    while (
        native_count >= H3_FRAME_OFFSET
        and native_count % H3_FRAME_MODULUS != H3_FRAME_OFFSET
    ):
        native_count -= 1
    if native_count < H3_FRAME_OFFSET:
        raise ValueError(
            "The prepared reference video is too short for H3's native frame grid."
        )
    return (
        resampled[:native_count],
        source_duration,
        resampled_count,
        native_count,
    )


def _audio_duration(audio: Any) -> float:
    if (
        not isinstance(audio, dict)
        or "waveform" not in audio
        or "sample_rate" not in audio
    ):
        raise ValueError("H3 Audio Reference needs a ComfyUI AUDIO value.")
    waveform = audio["waveform"]
    shape = getattr(waveform, "shape", None)
    if shape is None or len(shape) < 1 or int(shape[-1]) < 1:
        raise ValueError("Audio waveform is empty or malformed.")
    try:
        sample_rate = int(audio["sample_rate"])
    except (TypeError, ValueError) as error:
        raise ValueError("Audio sample_rate must be a positive integer.") from error
    if sample_rate <= 0:
        raise ValueError("Audio sample_rate must be a positive integer.")
    duration = int(shape[-1]) / sample_rate
    if duration < 2.0 - 0.0005 or duration > 15.0 + 0.0005:
        raise ValueError(
            "Every H3 reference-audio clip must last from 2 through 15 seconds."
        )
    return duration


def _default_binding_retention(content_type: str) -> str:
    if content_type == CONTENT_ACTION:
        return RETENTION_TRANSFER
    if content_type == CONTENT_STYLE:
        return RETENTION_WEAK
    return RETENTION_FULL


def _resolve_binding_retention(
    retention: str,
    content_type: str,
    transfer_target: str,
) -> str:
    if content_type not in CONTENT_TYPES[1:]:
        raise ValueError(
            "Choose the exact visible content type for this Subject binding."
        )
    resolved = (
        _default_binding_retention(content_type)
        if retention == RETENTION_AUTO
        else retention
    )
    if resolved not in VISUAL_RETENTIONS[1:]:
        raise ValueError("Choose a valid visual retention relationship.")
    if resolved == RETENTION_TRANSFER and not _clean_inline(transfer_target):
        raise ValueError(
            "attribute_transfer needs transfer_target_subject, such as the person receiving the motion."
        )
    if resolved != RETENTION_TRANSFER and _clean_inline(transfer_target):
        raise ValueError(
            "transfer_target_subject is only valid when retention is attribute_transfer."
        )
    return resolved


def _append_binding(
    plan: dict,
    asset: dict,
    subject_name: str,
    content_type: str,
    retention: str,
    shot_scope: str,
    notes: str,
    transfer_target: str,
) -> dict:
    alias = _clean_inline(subject_name)
    if not alias:
        raise ValueError("Subject name is required for reusable visible content.")
    key = _alias_key(alias)
    if any(
        entry["asset_id"] == asset["asset_id"]
        and _alias_key(entry["subject_name"]) == key
        and entry["content_type"] == content_type
        for entry in plan["bindings"]
    ):
        raise ValueError(
            f"{alias!r} already has the {content_type!r} binding on this reference."
        )
    resolved_retention = _resolve_binding_retention(
        retention,
        content_type,
        transfer_target,
    )
    target = _clean_inline(transfer_target)
    if target and _alias_key(target) not in {
        _alias_key(entry["subject_name"]) for entry in plan["bindings"]
    }:
        raise ValueError(
            f"transfer_target_subject {target!r} is not an upstream Subject. "
            "Define the target before adding the transfer binding."
        )
    binding = {
        "binding_id": _next_binding_id(plan),
        "asset_id": asset["asset_id"],
        "subject_name": alias,
        "content_type": content_type,
        "retention": resolved_retention,
        "shot_scope": _clean_inline(shot_scope),
        "notes": _clean_block(notes),
        "transfer_target_subject": target,
    }
    updated = _copy_plan(plan)
    updated["bindings"].append(binding)
    return updated


def _parse_shot_scope(scope: str, shot_count: int) -> list[int]:
    text = _clean_inline(scope)
    if not text:
        return []
    if shot_count < 1:
        raise ValueError(f"shot_scope {text!r} is set, but the plan has no Shot nodes.")
    normalized = _SCOPE_PREFIX_RE.sub("", text).strip().strip("[]()")
    if normalized.casefold() in {"all", "*"}:
        return list(range(1, shot_count + 1))
    numbers: list[int] = []
    for fragment in normalized.split(","):
        part = fragment.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if match:
            start, end = (int(match.group(1)), int(match.group(2)))
            if end < start:
                raise ValueError(f"shot_scope range {part!r} is reversed.")
            numbers.extend(range(start, end + 1))
        elif part.isdigit():
            numbers.append(int(part))
        else:
            raise ValueError(
                f"shot_scope {text!r} is invalid. Use forms such as 3, 3,4, 3-4, or all."
            )
    unique = list(dict.fromkeys(numbers))
    if not unique or min(unique) < 1 or max(unique) > shot_count:
        raise ValueError(
            f"shot_scope {text!r} refers outside the current 1-{shot_count} Shot range."
        )
    return unique


def _scope_text(numbers: list[int]) -> str:
    labels = [f"[Shot {number}]" for number in numbers]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _audio_retention(audio_use: str) -> str:
    if audio_use == AUDIO_COPY_COMPLETE:
        return AUDIO_FULL_COPY
    if audio_use == AUDIO_COPY_PARTIAL:
        return AUDIO_PARTIAL_COPY
    if audio_use == AUDIO_BROAD:
        return AUDIO_WEAK_REFERENCE
    return AUDIO_REFERENCE


def _image_default_retention(image_use: str, content_type: str) -> str:
    if image_use in {IMAGE_FIRST_FRAME, IMAGE_LAST_FRAME, IMAGE_KEYFRAME}:
        return RETENTION_FULL
    if image_use == IMAGE_STORYBOARD:
        return RETENTION_WEAK
    return _default_binding_retention(content_type)


def _validate_image_retention(
    image_use: str,
    content_type: str,
    retention: str,
    transfer_target: str,
) -> str:
    resolved = (
        _image_default_retention(image_use, content_type)
        if retention == RETENTION_AUTO
        else retention
    )
    if (
        image_use in {IMAGE_FIRST_FRAME, IMAGE_LAST_FRAME, IMAGE_KEYFRAME}
        and resolved != RETENTION_FULL
    ):
        raise ValueError(f"{image_use} requires fully_preserved retention.")
    if image_use == IMAGE_DEFINE_VISIBLE:
        return _resolve_binding_retention(resolved, content_type, transfer_target)
    if resolved not in VISUAL_RETENTIONS[1:]:
        raise ValueError("Choose a valid visual retention relationship.")
    if resolved == RETENTION_TRANSFER:
        raise ValueError(
            "attribute_transfer requires Define reusable visible content and an "
            "explicit transfer_target_subject. Direct endpoint, keyframe, and storyboard "
            "Picture roles cannot silently transfer an attribute."
        )
    return resolved


def _validate_reference_counts(plan: dict) -> None:
    pictures = [asset for asset in plan["assets"] if asset["media_kind"] == "image"]
    videos = [asset for asset in plan["assets"] if asset["media_kind"] == "video"]
    audios = [asset for asset in plan["assets"] if asset["media_kind"] == "audio"]
    if len(pictures) > 9:
        raise ValueError("MiniMax H3 accepts at most 9 reference images.")
    if len(videos) > 3:
        raise ValueError("MiniMax H3 accepts at most 3 reference videos.")
    if len(audios) > 3:
        raise ValueError("MiniMax H3 accepts at most 3 reference-audio clips.")
    if len(pictures) + len(videos) + len(audios) > 12:
        raise ValueError("MiniMax H3 accepts at most 12 mixed reference media files.")
    if sum(float(asset["source_duration"]) for asset in videos) > 15.0 + 0.0005:
        raise ValueError(
            "Reference videos total more than MiniMax H3's 15-second limit."
        )
    audio_total = sum(float(asset["duration"]) for asset in audios)
    if audio_total > 15.0 + 0.0005:
        audio_details = ", ".join(
            f"{asset['reference_name']}={float(asset['duration']):.3f}s"
            for asset in audios
        )
        raise ValueError(
            f"Reference-audio chain totals {audio_total:.3f}s, more than MiniMax H3's "
            f"15-second cumulative limit ({audio_details}). Shorten a selected Reference "
            "Sheet audio segment or remove an Audio Reference from this plan."
        )


def _catalog(plan: dict) -> dict:
    pictures = [asset for asset in plan["assets"] if asset["media_kind"] == "image"]
    videos = [asset for asset in plan["assets"] if asset["media_kind"] == "video"]
    audios = [asset for asset in plan["assets"] if asset["media_kind"] == "audio"]

    picture_labels = {
        asset["asset_id"]: f"<Picture {index}>"
        for index, asset in enumerate(pictures, start=1)
    }
    video_labels = {
        asset["asset_id"]: f"<Video {index}>"
        for index, asset in enumerate(videos, start=1)
    }

    audio_labels: dict[str, str] = {}
    presentation: list[dict] = []
    for picture_index, asset in enumerate(pictures):
        presentation.append(
            {
                "asset_id": asset["asset_id"],
                "label": picture_labels[asset["asset_id"]],
                "route": f"ref_image_{picture_index}",
                "media_kind": "image",
            }
        )

    audio_number = 0
    for video_index, video in enumerate(videos):
        paired = [
            audio
            for audio in audios
            if audio.get("paired_video_asset_id") == video["asset_id"]
        ]
        if len(paired) > 1:
            raise ValueError(
                f"{video_labels[video['asset_id']]} has more than one paired soundtrack."
            )
        if paired:
            audio_number += 1
            audio = paired[0]
            audio_labels[audio["asset_id"]] = f"<Audio {audio_number}>"
            presentation.append(
                {
                    "asset_id": audio["asset_id"],
                    "label": audio_labels[audio["asset_id"]],
                    "route": f"ref_video_audio_{video_index}",
                    "media_kind": "audio",
                    "paired_video_asset_id": video["asset_id"],
                }
            )
        presentation.append(
            {
                "asset_id": video["asset_id"],
                "label": video_labels[video["asset_id"]],
                "route": f"ref_video_{video_index}",
                "media_kind": "video",
            }
        )

    standalone_number = 0
    for audio in audios:
        if audio.get("paired_video_asset_id"):
            continue
        audio_number += 1
        audio_labels[audio["asset_id"]] = f"<Audio {audio_number}>"
        presentation.append(
            {
                "asset_id": audio["asset_id"],
                "label": audio_labels[audio["asset_id"]],
                "route": f"ref_audio_{standalone_number}",
                "media_kind": "audio",
            }
        )
        standalone_number += 1

    subject_groups: list[dict] = []
    subjects_by_alias: dict[str, dict] = {}
    for binding in plan["bindings"]:
        key = _alias_key(binding["subject_name"])
        if key not in subjects_by_alias:
            group = {
                "label": f"<Subject {len(subject_groups) + 1}>",
                "subject_name": binding["subject_name"],
                "bindings": [],
            }
            subject_groups.append(group)
            subjects_by_alias[key] = group
        subjects_by_alias[key]["bindings"].append(binding)

    speaker_ids: dict[str, str] = {}
    for event in plan["dialogue_events"]:
        key = _alias_key(event["speaker"])
        if key not in speaker_ids:
            speaker_ids[key] = f"S{len(speaker_ids) + 1}"

    return {
        "pictures": pictures,
        "videos": videos,
        "audios": audios,
        "picture_labels": picture_labels,
        "video_labels": video_labels,
        "audio_labels": audio_labels,
        "presentation": presentation,
        "subject_groups": subject_groups,
        "subjects_by_alias": subjects_by_alias,
        "speaker_ids": speaker_ids,
    }


def _determine_mode(plan: dict, catalog: dict) -> tuple[str, str]:
    pictures = catalog["pictures"]
    has_video_or_audio = bool(catalog["videos"] or catalog["audios"])
    has_subjects = bool(catalog["subject_groups"])
    image_uses = [asset["relationship"] for asset in pictures]
    endpoint_only = all(
        image_use in {IMAGE_FIRST_FRAME, IMAGE_LAST_FRAME} for image_use in image_uses
    )
    requires_ref2va = bool(
        has_video_or_audio or has_subjects or (pictures and not endpoint_only)
    )
    has_endpoint = any(
        image_use in {IMAGE_FIRST_FRAME, IMAGE_LAST_FRAME} for image_use in image_uses
    )
    if has_endpoint and requires_ref2va:
        raise ValueError(
            "Exact first/last-frame relationships use H3 endpoint conditioning and cannot "
            "be mixed with reusable Subjects, keyframes/storyboards, reference video, or "
            "reference audio. Duplicate the image with a Ref2VA role, or keep a separate "
            "endpoint-only workflow."
        )
    if requires_ref2va:
        return MODE_REF2VA, "H3-Base-Ref2VA"
    if not pictures:
        return MODE_T2VA, "H3-Base-FL2VA"
    first_count = image_uses.count(IMAGE_FIRST_FRAME)
    last_count = image_uses.count(IMAGE_LAST_FRAME)
    if first_count > 1 or last_count > 1:
        raise ValueError(
            "Endpoint generation accepts at most one exact first and one exact last frame."
        )
    if first_count and last_count:
        return MODE_FL2VA, "H3-Base-FL2VA"
    if first_count:
        return MODE_I2VA, "H3-Base-FL2VA"
    if last_count:
        return MODE_L2VA, "H3-Base-FL2VA"
    raise ValueError(
        "The plan's image relationships do not resolve to a supported H3 mode."
    )


def _asset_by_id(plan: dict) -> dict[str, dict]:
    return {asset["asset_id"]: asset for asset in plan["assets"]}


def _known_labels(catalog: dict) -> set[str]:
    return {
        *(group["label"] for group in catalog["subject_groups"]),
        *catalog["picture_labels"].values(),
        *catalog["video_labels"].values(),
        *catalog["audio_labels"].values(),
    }


def _canonical_token(match: re.Match[str]) -> str:
    return f"<{match.group(1).title()} {int(match.group(2))}>"


def _validate_typed_references(plan: dict, catalog: dict) -> None:
    known = _known_labels(catalog)
    text_fields = [plan["project"]["initial_prompt"]]
    for shot in plan["shots"]:
        text_fields.extend([shot["description"], shot["camera_direction"]])
    unknown = sorted(
        {
            _canonical_token(match)
            for text in text_fields
            for match in _REFERENCE_TOKEN_RE.finditer(text or "")
            if _canonical_token(match) not in known
        }
    )
    if unknown:
        raise ValueError(
            "Shot or Project text contains unknown H3 labels: "
            + ", ".join(unknown)
            + ". Choose an upstream reference instead of keeping a stale number."
        )


def _shot_mentions_label(
    plan: dict, shot_number: int, label: str, subject_alias: str = ""
) -> bool:
    shot = plan["shots"][shot_number - 1]
    combined = f"{shot['description']} {shot['camera_direction']}"
    if label.casefold() in combined.casefold():
        return True
    if subject_alias:
        return any(
            event["shot_number"] == shot_number
            and _alias_key(event["speaker"]) == _alias_key(subject_alias)
            for event in plan["dialogue_events"]
        )
    return False


def _shot_prose_mentions_label(plan: dict, label: str) -> bool:
    """Return whether the author placed one reference label in visible Shot prose."""

    shots = plan["shots"]
    fields = (
        [
            field
            for shot in shots
            for field in (shot["description"], shot["camera_direction"])
        ]
        if shots
        else [plan["project"]["initial_prompt"]]
    )
    return any(
        _canonical_token(match) == label
        for field in fields
        for match in _REFERENCE_TOKEN_RE.finditer(field or "")
    )


def _replacement_subject_group(replacement: dict, catalog: dict) -> dict:
    group = catalog["subjects_by_alias"].get(
        _alias_key(replacement["replacement_subject"])
    )
    if group is None:
        raise ValueError(
            f"Character replacement Subject {replacement['replacement_subject']!r} "
            "is not an upstream Subject. Define its reference image and identity first."
        )
    return group


def _replacement_video(replacement: dict, plan: dict) -> dict:
    asset = _asset_by_id(plan).get(replacement["source_video_asset_id"])
    if asset is None or asset.get("media_kind") != "video":
        raise ValueError(
            "A Character Replacement refers to a source video that is not present in h3_plan."
        )
    if asset.get("relationship") not in {VIDEO_EDIT, VIDEO_CONTINUE}:
        raise ValueError(
            "Character Replacement requires a Video Reference set to Source video to edit "
            "or Source video to continue."
        )
    return asset


def _replacement_scope(replacement: dict, plan: dict) -> list[int]:
    return _parse_shot_scope(replacement["shot_scope"], len(plan["shots"]))


def _replacement_covers_subject_shot(
    plan: dict, subject_alias: str, shot_number: int
) -> bool:
    return any(
        _alias_key(replacement["replacement_subject"]) == _alias_key(subject_alias)
        and shot_number in _replacement_scope(replacement, plan)
        for replacement in plan["character_replacements"]
    )


def _validate_character_replacements(plan: dict, catalog: dict) -> None:
    replacement_ids: set[str] = set()
    source_targets: set[tuple[str, str]] = set()
    for replacement in plan["character_replacements"]:
        replacement_id = replacement.get("replacement_id")
        if not isinstance(replacement_id, str) or not replacement_id:
            raise ValueError("Every Character Replacement needs a stable replacement_id.")
        if replacement_id in replacement_ids:
            raise ValueError("h3_plan contains duplicate Character Replacement IDs.")
        replacement_ids.add(replacement_id)

        video = _replacement_video(replacement, plan)
        group = _replacement_subject_group(replacement, catalog)
        if not any(
            binding["content_type"] == CONTENT_IDENTITY
            for binding in group["bindings"]
        ):
            raise ValueError(
                f"{group['label']} ({group['subject_name']}) cannot replace a character "
                "until it has an Identity or appearance binding."
            )
        source_character = _clean_inline(
            replacement.get("source_character_description")
        )
        if not source_character:
            raise ValueError(
                "Character Replacement needs a precise source_character_description."
            )
        instructions = _clean_block(replacement.get("instructions"))
        if _REFERENCE_TOKEN_RE.search(source_character) or _REFERENCE_TOKEN_RE.search(
            instructions
        ):
            raise ValueError(
                "Character Replacement description and instructions must use plain language; "
                "the compiler assigns H3 labels automatically."
            )
        if (
            replacement.get("appearance_policy")
            not in CHARACTER_REPLACEMENT_APPEARANCE_POLICIES
        ):
            raise ValueError("Character Replacement contains an invalid appearance policy.")
        if not isinstance(
            replacement.get("preserve_performance"), bool
        ) or not isinstance(
            replacement.get("preserve_scene"), bool
        ):
            raise ValueError("Character Replacement preservation controls must be booleans.")
        numbers = _replacement_scope(replacement, plan)
        if not numbers:
            raise ValueError("Character Replacement requires a non-empty shot_scope.")
        video_scope = _parse_shot_scope(video.get("shot_scope", ""), len(plan["shots"]))
        if video_scope and not set(numbers).issubset(video_scope):
            raise ValueError(
                "Character Replacement shot_scope extends outside its source Video "
                "Reference shot_scope."
            )
        source_key = (video["asset_id"], _alias_key(source_character))
        if source_key in source_targets:
            raise ValueError(
                "The same source performer in one video cannot have two Character "
                "Replacement mappings. Edit the existing mapping instead."
            )
        source_targets.add(source_key)


def _validate_scopes(plan: dict, catalog: dict) -> None:
    shot_count = len(plan["shots"])
    for group in catalog["subject_groups"]:
        for binding in group["bindings"]:
            numbers = _parse_shot_scope(binding["shot_scope"], shot_count)
            missing = [
                number
                for number in numbers
                if not _shot_mentions_label(
                    plan,
                    number,
                    group["label"],
                    group["subject_name"],
                )
                and not _replacement_covers_subject_shot(
                    plan, group["subject_name"], number
                )
            ]
            if missing:
                raise ValueError(
                    f"{group['label']} ({group['subject_name']}) is scoped to "
                    f"{_scope_text(missing)}, but those Shot descriptions do not cite it. "
                    "Insert the Subject label at the intended character or item."
                )

    relationships = {entry["asset_id"]: entry for entry in plan["audio_relationships"]}
    for audio in catalog["audios"]:
        relationship = relationships[audio["asset_id"]]
        numbers = _parse_shot_scope(relationship["shot_scope"], shot_count)
        if not numbers:
            continue
        label = catalog["audio_labels"][audio["asset_id"]]
        target = relationship["target_speaker"]
        missing = []
        for number in numbers:
            mentioned = _shot_mentions_label(plan, number, label)
            if relationship["use"] in {AUDIO_VOICE, AUDIO_CONTENT} and target:
                mentioned = mentioned or any(
                    event["shot_number"] == number
                    and _alias_key(event["speaker"]) == _alias_key(target)
                    for event in plan["dialogue_events"]
                )
            if not mentioned:
                missing.append(number)
        if missing:
            raise ValueError(
                f"{label} is scoped to {_scope_text(missing)}, but those Shots neither cite it "
                "nor contain its bound vocal event. For a sound effect, insert the Audio label "
                "inside the exact Shot sound sentence where it should be heard."
            )


def _validate_speakers(plan: dict, catalog: dict) -> None:
    events_by_speaker: dict[str, list[dict]] = {}
    for event in plan["dialogue_events"]:
        events_by_speaker.setdefault(_alias_key(event["speaker"]), []).append(event)

    relationships = {entry["asset_id"]: entry for entry in plan["audio_relationships"]}
    voice_targets: set[str] = set()
    for audio in catalog["audios"]:
        relationship = relationships[audio["asset_id"]]
        use = relationship["use"]
        target = relationship["target_speaker"]
        if use in {AUDIO_VOICE, AUDIO_CONTENT}:
            key = _alias_key(target)
            if key not in events_by_speaker:
                label = catalog["audio_labels"][audio["asset_id"]]
                raise ValueError(
                    f"{label} is set to {use}, but {target!r} has no Dialogue Event. "
                    "Add the vocal event so its speaker order and Shot are known."
                )
            if use == AUDIO_VOICE:
                if key in voice_targets:
                    raise ValueError(
                        f"Speaker {target!r} has more than one voice-timbre reference."
                    )
                voice_targets.add(key)
            if use == AUDIO_CONTENT:
                matching = [
                    event
                    for event in events_by_speaker[key]
                    if _clean_block(event["exact_text"])
                    == _clean_block(relationship["transcript"])
                    and _alias_key(event["language"])
                    == _alias_key(relationship["language"])
                ]
                if not matching:
                    raise ValueError(
                        f"{catalog['audio_labels'][audio['asset_id']]} provides exact dialogue or lyric "
                        "content, but no Dialogue Event has the same speaker, language, and transcript."
                    )


def _validate_complete_audio_copy(plan: dict, catalog: dict) -> None:
    relationships = {entry["asset_id"]: entry for entry in plan["audio_relationships"]}
    complete = [
        audio
        for audio in catalog["audios"]
        if relationships[audio["asset_id"]]["use"] == AUDIO_COPY_COMPLETE
    ]
    if not complete:
        return
    if len(complete) != 1 or len(catalog["audios"]) != 1:
        raise ValueError(
            "Complete audio copy requires exactly one reference-audio clip."
        )
    project = plan["project"]
    if plan["dialogue_events"]:
        raise ValueError(
            "Complete audio copy cannot be combined with new Dialogue Events."
        )
    if _clean_inline(project["overall_soundscape"]).casefold() not in {
        "",
        "n/a",
        "none",
    }:
        raise ValueError(
            "Complete audio copy is the whole final track; clear the new overall_soundscape."
        )
    if _clean_inline(project["non_diegetic_music"]).casefold() not in {
        "",
        "n/a",
        "none",
    }:
        raise ValueError(
            "Complete audio copy is the whole final track; set non_diegetic_music to N/A."
        )


def _validate_final_plan(plan: dict, catalog: dict, mode: str) -> None:
    _validate_reference_counts(plan)
    asset_ids = {asset["asset_id"] for asset in plan["assets"]}
    if any(binding.get("asset_id") not in asset_ids for binding in plan["bindings"]):
        raise ValueError(
            "A Subject binding refers to media that is not present in h3_plan."
        )
    audio_ids = {
        asset["asset_id"] for asset in plan["assets"] if asset["media_kind"] == "audio"
    }
    relationship_ids = {
        relationship.get("asset_id") for relationship in plan["audio_relationships"]
    }
    if relationship_ids != audio_ids or len(relationship_ids) != len(
        plan["audio_relationships"]
    ):
        raise ValueError(
            "Every audio asset needs exactly one matching audio relationship."
        )
    if mode == MODE_REF2VA and not (catalog["pictures"] or catalog["videos"]):
        raise ValueError(
            "Reference audio cannot be the only Ref2VA input; add at least one image or video."
        )
    if not plan["shots"] and not _clean_block(plan["project"]["initial_prompt"]):
        raise ValueError("Add an initial prompt or at least one H3 Shot description.")
    if plan["shots"]:
        starts = [float(shot["cut_at"]) for shot in plan["shots"]]
        if not math.isclose(starts[0], 0.0, abs_tol=0.0005):
            raise ValueError("Shot 1 must start at 0.000 seconds.")
        if any(current <= previous for previous, current in zip(starts, starts[1:])):
            raise ValueError("Shot cut times must be strictly increasing.")
        duration = float(plan["project"]["duration_seconds"])
        if starts[-1] >= duration:
            raise ValueError("The final Shot must begin before the Project duration.")
    for asset in [*catalog["pictures"], *catalog["videos"]]:
        if asset.get("shot_scope"):
            _parse_shot_scope(asset["shot_scope"], len(plan["shots"]))
    _validate_typed_references(plan, catalog)
    _validate_character_replacements(plan, catalog)
    _validate_scopes(plan, catalog)
    _validate_speakers(plan, catalog)
    _validate_complete_audio_copy(plan, catalog)


def _binding_role_grammar(content_type: str) -> tuple[str, bool]:
    """Return one exact role phrase and whether it is grammatically singular."""

    return {
        CONTENT_IDENTITY: ("identity and appearance", False),
        CONTENT_OBJECT: ("visible object appearance", True),
        CONTENT_SCENE: ("scene and environment", False),
        CONTENT_STYLE: ("visual style", True),
        CONTENT_ACTION: ("pose and movement", False),
    }[content_type]


def _replacement_labels(
    replacement: dict, plan: dict, catalog: dict
) -> tuple[str, str]:
    subject = _replacement_subject_group(replacement, catalog)["label"]
    video = catalog["video_labels"][_replacement_video(replacement, plan)["asset_id"]]
    return subject, video


def _replacement_mapping_sentence(
    replacement: dict, plan: dict, catalog: dict
) -> str:
    subject, video = _replacement_labels(replacement, plan, catalog)
    source_video = _replacement_video(replacement, plan)
    scope = _scope_text(_replacement_scope(replacement, plan))
    source_character = _clause(replacement["source_character_description"])
    if source_video["relationship"] == VIDEO_CONTINUE:
        return (
            f"In {scope}, the source performer described as {source_character} at "
            f"{video}'s endpoint is continued as {subject} in the newly generated frames."
        )
    return (
        f"In {scope}, {subject} replaces only the source performer described as "
        f"{source_character} from {video}."
    )


def _replacement_shot_instruction(
    replacement: dict, plan: dict, catalog: dict
) -> str:
    subject, video = _replacement_labels(replacement, plan, catalog)
    source_video = _replacement_video(replacement, plan)
    continuation = source_video["relationship"] == VIDEO_CONTINUE
    source_character = _clause(replacement["source_character_description"])
    if continuation:
        lines = [
            f"Using {video} as the continuation source, carry the source performer "
            f"described as {source_character} across the endpoint as {subject} in only "
            "the newly generated frames."
        ]
    else:
        lines = [
            f"Using {video} as the source timeline, replace only the source performer "
            f"described as {source_character} with {subject}."
        ]
    policy = replacement["appearance_policy"]
    if policy == REPLACEMENT_IDENTITY_KEEP_BODY_WARDROBE:
        lines.append(
            f"Apply only {subject}'s identity and facial appearance; retain the source "
            f"performer's body proportions and wardrobe from {video}."
        )
    elif policy == REPLACEMENT_IDENTITY_BODY_KEEP_WARDROBE:
        lines.append(
            f"Apply {subject}'s identity, facial appearance, hair, and body proportions; "
            f"retain the source performer's wardrobe from {video}."
        )
    else:
        lines.append(
            f"Apply {subject}'s complete referenced identity and appearance, including wardrobe."
        )
    if replacement["preserve_performance"]:
        if continuation:
            lines.append(
                "Begin from that source performer's final pose, position, motion direction, "
                "gaze, expression, and physical interactions, then evolve the performance "
                "forward without replaying the source timeline."
            )
        else:
            lines.append(
                "Preserve that source performer's pose, motion, timing, gaze, expression, "
                f"and physical interactions from {video}."
            )
    if replacement["preserve_scene"]:
        if continuation:
            lines.append(
                "Continue every other person, the environment, props, lighting, camera "
                f"movement, framing, and spatial state from {video}'s endpoint without a reset."
            )
        else:
            lines.append(
                "Keep every other person, the environment, props, lighting, camera movement, "
                f"framing, and cuts from {video} unchanged."
            )
    instructions = _sentence(replacement["instructions"])
    if instructions:
        lines.append(instructions)
    return " ".join(lines)


def _subject_definition(group: dict, plan: dict, catalog: dict) -> str:
    definition_clauses: list[str] = []
    transfer_clauses: list[str] = []
    for binding in group["bindings"]:
        asset_id = binding["asset_id"]
        label = catalog["picture_labels"].get(asset_id) or catalog["video_labels"].get(
            asset_id
        )
        phrase, singular = _binding_role_grammar(binding["content_type"])
        clause = (
            f"The {phrase} of {group['label']} "
            f"{'is' if singular else 'are'} defined by {label}."
        )
        if clause not in definition_clauses:
            definition_clauses.append(clause)
        target_alias = binding.get("transfer_target_subject", "")
        if binding["retention"] == RETENTION_TRANSFER and target_alias:
            target = catalog["subjects_by_alias"].get(_alias_key(target_alias))
            target_text = target["label"] if target else target_alias
            clause = f"Transfer the {phrase} defined by {label} to {target_text}."
            if clause not in transfer_clauses:
                transfer_clauses.append(clause)
    replacement_clauses = [
        _replacement_mapping_sentence(replacement, plan, catalog)
        for replacement in plan["character_replacements"]
        if _alias_key(replacement["replacement_subject"])
        == _alias_key(group["subject_name"])
    ]
    return " ".join(
        (
            f"{group['label']} is {group['subject_name']}.",
            *definition_clauses,
            *transfer_clauses,
            *replacement_clauses,
        )
    )


def _direct_picture_definition(asset: dict, label: str, plan: dict) -> str | None:
    description = _clause(asset["description"], "the supplied reference image")
    relationship = asset["relationship"]
    final_shot = max(1, len(plan["shots"]))
    if relationship == IMAGE_FIRST_FRAME:
        return f"{label} is the first frame of [Shot 1], showing {description}."
    if relationship == IMAGE_LAST_FRAME:
        return (
            f"{label} is the final frame of [Shot {final_shot}], showing {description}."
        )
    if relationship == IMAGE_KEYFRAME:
        scope = _parse_shot_scope(asset["shot_scope"], len(plan["shots"]))
        scope_text = f" for {_scope_text(scope)}" if scope else ""
        return (
            f"{label} is the concrete keyframe and composition anchor{scope_text}, "
            f"showing {description}."
        )
    if relationship == IMAGE_STORYBOARD:
        scope = _parse_shot_scope(asset["shot_scope"], len(plan["shots"]))
        scope_text = f" for {_scope_text(scope)}" if scope else ""
        return f"{label} is the storyboard and shot-planning reference{scope_text}: {description}."
    return None


def _direct_video_definition(
    asset: dict,
    label: str,
    catalog: dict,
    plan: dict,
) -> str | None:
    description = _clause(asset["description"], "the supplied reference video")
    relationship = asset["relationship"]
    scope = _parse_shot_scope(asset["shot_scope"], len(plan["shots"]))
    scope_text = f" for {_scope_text(scope)}" if scope else ""
    if relationship == VIDEO_EDIT:
        return (
            f"{label} is the source video for the target video edit{scope_text}: "
            f"{description}."
        )
    if relationship == VIDEO_CONTINUE:
        return (
            f"{label} is the source video continued by the target video{scope_text}: "
            f"{description}."
        )
    if relationship == VIDEO_MOTION:
        target = catalog["subjects_by_alias"].get(_alias_key(asset["target_subject"]))
        target_text = target["label"] if target else asset["target_subject"]
        return (
            f"{label} is the motion and action reference transferred to {target_text}"
            f"{scope_text}: "
            f"{description}."
        )
    if relationship == VIDEO_STRUCTURE:
        return (
            f"{label} is the camera, cuts, rhythm, and temporal-structure reference"
            f"{scope_text}: "
            f"{description}."
        )
    return None


def _speaker_reference(target: str, catalog: dict) -> str:
    key = _alias_key(target)
    subject = catalog["subjects_by_alias"].get(key)
    visible = subject["label"] if subject else target
    speaker_id = catalog["speaker_ids"].get(key)
    return f"{visible} ({speaker_id})" if speaker_id else visible


def _audio_definition(
    audio: dict,
    relationship: dict,
    label: str,
    catalog: dict,
) -> str:
    use = relationship["use"]
    target = relationship["target_speaker"]
    layer = relationship["target_layer_or_event"]
    instructions = _sentence(relationship["instructions"])
    suffix = f" {instructions}" if instructions else ""
    if use == AUDIO_VOICE:
        return (
            f"{label} is the voice-timbre and delivery reference for "
            f"{_speaker_reference(target, catalog)}; do not reuse its source words.{suffix}"
        )
    if use == AUDIO_MUSIC:
        return (
            f"{label} is the background-music style reference for {layer}, guiding musical "
            f"character without copying the source signal.{suffix}"
        )
    if use == AUDIO_BEAT:
        return f"{label} is the beat-and-rhythm reference for {layer}.{suffix}"
    if use == AUDIO_SFX:
        return f"{label} is the sound-effect texture reference for {layer}.{suffix}"
    if use == AUDIO_CONTENT:
        return (
            f"{label} provides the referenced spoken or lyric content for "
            f"{_speaker_reference(target, catalog)}: "
            f"<d>[{relationship['language']}] {relationship['transcript']}</d>.{suffix}"
        )
    if use == AUDIO_CONTINUITY:
        paired_video_id = audio.get("paired_video_asset_id")
        paired_video = next(
            (
                video
                for video in catalog["videos"]
                if video["asset_id"] == paired_video_id
            ),
            None,
        )
        if paired_video is not None and paired_video["relationship"] == VIDEO_CONTINUE:
            video_label = catalog["video_labels"][paired_video_id]
            return (
                f"{label} is the synchronized soundtrack of {video_label} and the "
                f"audio-continuity reference for {layer}. Generate new audio beginning after "
                "the source endpoint and developing forward from its final audible state; do "
                f"not copy, restart, replay, repeat, or loop the source signal.{suffix}"
            )
        return f"{label} is the audio-continuity reference for {layer}.{suffix}"
    if use == AUDIO_COPY_COMPLETE:
        return f"{label} is reused as the complete final audio track."
    if use == AUDIO_COPY_PARTIAL:
        return (
            f"{label} provides only the selected copied range or layers: {instructions}"
        )
    if use == AUDIO_BROAD:
        return (
            f"{label} is a weak broad audio-inspiration reference for {layer}.{suffix}"
        )
    raise ValueError(f"Unsupported audio relationship: {use!r}.")


def _subject_definitions(plan: dict, catalog: dict) -> list[str]:
    assets = _asset_by_id(plan)
    lines = [
        _subject_definition(group, plan, catalog)
        for group in catalog["subject_groups"]
    ]
    for asset in catalog["pictures"]:
        definition = _direct_picture_definition(
            asset,
            catalog["picture_labels"][asset["asset_id"]],
            plan,
        )
        if definition:
            lines.append(definition)
    for asset in catalog["videos"]:
        definition = _direct_video_definition(
            asset,
            catalog["video_labels"][asset["asset_id"]],
            catalog,
            plan,
        )
        if definition:
            lines.append(definition)
    relationships = {entry["asset_id"]: entry for entry in plan["audio_relationships"]}
    for asset in catalog["audios"]:
        lines.append(
            _audio_definition(
                assets[asset["asset_id"]],
                relationships[asset["asset_id"]],
                catalog["audio_labels"][asset["asset_id"]],
                catalog,
            )
        )
    return lines


def _task_types(plan: dict, catalog: dict) -> list[str]:
    picture_uses = {asset["relationship"] for asset in catalog["pictures"]}
    video_uses = {asset["relationship"] for asset in catalog["videos"]}
    audio_uses = {entry["use"] for entry in plan["audio_relationships"]}
    tasks: list[str] = []
    if picture_uses & {IMAGE_FIRST_FRAME, IMAGE_LAST_FRAME, IMAGE_KEYFRAME}:
        tasks.append("keyframe completion")
    if (
        catalog["subject_groups"]
        or IMAGE_STORYBOARD in picture_uses
        or video_uses & {VIDEO_MOTION, VIDEO_STRUCTURE}
    ):
        tasks.append("reference generation")
    if VIDEO_EDIT in video_uses:
        tasks.append("video editing")
    if VIDEO_CONTINUE in video_uses:
        tasks.append("video continuation")
    if audio_uses & {AUDIO_COPY_COMPLETE, AUDIO_COPY_PARTIAL}:
        tasks.append("audio reuse")
    if audio_uses - {AUDIO_COPY_COMPLETE, AUDIO_COPY_PARTIAL}:
        tasks.append("audio reference")
    return tasks or ["reference generation"]


def _summary(plan: dict, catalog: dict) -> str:
    tasks = _task_types(plan, catalog)
    sentences: list[str] = [f"[{' + '.join(tasks)}]"]
    for video in catalog["videos"]:
        label = catalog["video_labels"][video["asset_id"]]
        if video["relationship"] == VIDEO_EDIT:
            sentences.append(f"The target video is an edited version of {label}.")
        elif video["relationship"] == VIDEO_CONTINUE:
            sentences.append(f"The target video continues after {label}.")
    sentences.extend(
        _replacement_mapping_sentence(replacement, plan, catalog)
        for replacement in plan["character_replacements"]
    )
    role_labels = [
        *(group["label"] for group in catalog["subject_groups"]),
        *catalog["picture_labels"].values(),
        *catalog["video_labels"].values(),
        *catalog["audio_labels"].values(),
    ]
    if role_labels:
        sentences.append(f"The declared reference roles use {', '.join(role_labels)}.")
    return " ".join(sentences)


def _subject_retention_line(group: dict, plan: dict, catalog: dict) -> str:
    markers = {binding["retention"] for binding in group["bindings"]}
    if len(markers) != 1:
        raise ValueError(
            f"{group['label']} ({group['subject_name']}) combines different retention markers. "
            "Use separate subject names or align their retention values."
        )
    scopes = sorted(
        {
            number
            for binding in group["bindings"]
            for number in _parse_shot_scope(binding["shot_scope"], len(plan["shots"]))
        }
        | {
            number
            for replacement in plan["character_replacements"]
            if _alias_key(replacement["replacement_subject"])
            == _alias_key(group["subject_name"])
            for number in _replacement_scope(replacement, plan)
        }
    )
    context = (
        f" (appears in {_scope_text(scopes)})"
        if scopes
        else " (appears wherever cited in the Shot plan)"
    )
    marker = next(iter(markers))
    role_grammar = list(
        dict.fromkeys(
            _binding_role_grammar(binding["content_type"])
            for binding in group["bindings"]
        )
    )
    if len(role_grammar) == 1:
        phrase, singular = role_grammar[0]
    else:
        phrase, singular = "visual properties", False
    subject = f"the defined {phrase}"
    explanation = {
        RETENTION_FULL: f"{subject} {'is' if singular else 'are'} preserved",
        RETENTION_PARTIAL: (
            f"{subject} {'remains' if singular else 'remain'} recognizable while "
            "requested changes are applied"
        ),
        RETENTION_TRANSFER: f"{subject} {'is' if singular else 'are'} transferred",
        RETENTION_WEAK: (
            f"{subject} {'guides' if singular else 'guide'} the result without exact preservation"
        ),
    }[marker]
    if marker == RETENTION_TRANSFER:
        targets = []
        for binding in group["bindings"]:
            target_alias = binding.get("transfer_target_subject", "")
            if not target_alias:
                continue
            target = catalog["subjects_by_alias"].get(_alias_key(target_alias))
            target_text = target["label"] if target else target_alias
            if target_text not in targets:
                targets.append(target_text)
        if targets:
            explanation += " to " + ", ".join(targets)
    return f"{group['label']}{context}: {marker} - {explanation}."


def _retention_analysis(plan: dict, catalog: dict) -> list[str]:
    lines = [
        _subject_retention_line(group, plan, catalog)
        for group in catalog["subject_groups"]
    ]
    for asset in catalog["pictures"]:
        relationship = asset["relationship"]
        if relationship == IMAGE_DEFINE_VISIBLE:
            continue
        label = catalog["picture_labels"][asset["asset_id"]]
        marker = asset["retention"]
        role = {
            IMAGE_FIRST_FRAME: "[Shot 1] first frame",
            IMAGE_LAST_FRAME: f"[Shot {max(1, len(plan['shots']))}] final frame",
            IMAGE_KEYFRAME: "concrete keyframe/composition anchor",
            IMAGE_STORYBOARD: "storyboard/shot-planning reference",
        }[relationship]
        lines.append(
            f"{label} ({role}): {marker} - its declared reference role is retained."
        )
    for asset in catalog["videos"]:
        relationship = asset["relationship"]
        if relationship == VIDEO_DEFINE_VISIBLE:
            continue
        label = catalog["video_labels"][asset["asset_id"]]
        marker = {
            VIDEO_EDIT: RETENTION_PARTIAL,
            VIDEO_CONTINUE: RETENTION_PARTIAL,
            VIDEO_MOTION: RETENTION_TRANSFER,
            VIDEO_STRUCTURE: RETENTION_WEAK,
        }[relationship]
        scope = _parse_shot_scope(asset["shot_scope"], len(plan["shots"]))
        context = f" ({_scope_text(scope)})" if scope else ""
        replacements = [
            replacement
            for replacement in plan["character_replacements"]
            if replacement["source_video_asset_id"] == asset["asset_id"]
        ]
        if relationship in {VIDEO_EDIT, VIDEO_CONTINUE} and replacements:
            preserved: list[str] = []
            if relationship == VIDEO_CONTINUE:
                if any(entry["preserve_performance"] for entry in replacements):
                    preserved.append(
                        "endpoint pose, motion direction, gaze, expression, and interactions"
                    )
                if any(entry["preserve_scene"] for entry in replacements):
                    preserved.append(
                        "other people, environment, lighting, camera, framing, and spatial state"
                    )
                explanation = (
                    "the source endpoint remains identifiable and the selected performer is "
                    "continued as the declared replacement Subject only in newly generated frames"
                )
            else:
                if any(entry["preserve_performance"] for entry in replacements):
                    preserved.append("source performance and timing")
                if any(entry["preserve_scene"] for entry in replacements):
                    preserved.append(
                        "scene, other people, lighting, camera, framing, and cuts"
                    )
                explanation = (
                    "the source timeline remains identifiable; only the declared character "
                    "replacement is applied"
                )
            if preserved:
                explanation += ". Its " + " plus ".join(preserved) + " are preserved"
            lines.append(f"{label}{context}: {marker} - {explanation}.")
        else:
            lines.append(
                f"{label}{context}: {marker} - its declared whole-video relationship is applied."
            )
    relationships = {entry["asset_id"]: entry for entry in plan["audio_relationships"]}
    for audio in catalog["audios"]:
        relationship = relationships[audio["asset_id"]]
        label = catalog["audio_labels"][audio["asset_id"]]
        marker = relationship["retention"]
        scope = _parse_shot_scope(relationship["shot_scope"], len(plan["shots"]))
        context = f" ({_scope_text(scope)})" if scope else ""
        paired_video_id = audio.get("paired_video_asset_id")
        paired_video = next(
            (
                video
                for video in catalog["videos"]
                if video["asset_id"] == paired_video_id
            ),
            None,
        )
        if (
            relationship["use"] == AUDIO_CONTINUITY
            and paired_video is not None
            and paired_video["relationship"] == VIDEO_CONTINUE
        ):
            video_label = catalog["video_labels"][paired_video_id]
            lines.append(
                f"{label}{context}: {marker} - it guides newly generated audio after "
                f"{video_label}'s endpoint; the source signal is not copied, restarted, "
                "replayed, repeated, or looped."
            )
            continue
        lines.append(
            f"{label}{context}: {marker} - its exact declared audio relationship is used."
        )
    return lines


def _transition_prefix(transition: str) -> str:
    return {
        "Direct cut": "cut directly to",
        "Cross-dissolve": "cross-dissolve to",
        "Fade": "fade to",
        "Wipe": "wipe to",
    }[transition]


def _voice_audio_for(speaker: str, plan: dict, catalog: dict) -> str | None:
    relationships = {entry["asset_id"]: entry for entry in plan["audio_relationships"]}
    for audio in catalog["audios"]:
        relationship = relationships[audio["asset_id"]]
        if relationship["use"] == AUDIO_VOICE and _alias_key(
            relationship["target_speaker"]
        ) == _alias_key(speaker):
            return catalog["audio_labels"][audio["asset_id"]]
    return None


def _dialogue_text(event: dict, plan: dict, catalog: dict) -> str:
    speaker = _speaker_reference(event["speaker"], catalog)
    audio_label = _voice_audio_for(event["speaker"], plan, catalog)
    voice_clause = (
        f" using the voice timbre and delivery referenced from {audio_label}"
        if audio_label
        else ""
    )
    exact = f"<d>[{event['language']}] {event['exact_text']}</d>"
    delivery = _sentence(event["delivery"])
    delivery_clause = f" Delivery: {delivery}" if delivery else ""
    if event["voice_mode"] == "Voiceover":
        return (
            f"{speaker} says in an off-screen voiceover{voice_clause}: {exact}. "
            f"The on-screen character's lips remain completely closed.{delivery_clause}"
        )
    if event["voice_mode"] == "Off-screen speech":
        return f"{speaker} speaks off-screen{voice_clause}: {exact}.{delivery_clause}"
    return f"{speaker} speaks{voice_clause}: {exact}.{delivery_clause}"


def _implicit_or_planned_shots(plan: dict) -> list[dict]:
    if plan["shots"]:
        return plan["shots"]
    return [
        {
            "shot_number": 1,
            "cut_at": 0.0,
            "description": plan["project"]["initial_prompt"],
            "camera_direction": "",
            "transition": "Direct cut",
        }
    ]


def _detailed_description(plan: dict, catalog: dict) -> str:
    project = plan["project"]
    style = _clean_inline(
        project["visual_style"], "a coherent cinematic audiovisual style"
    )
    lines = [
        f"The target video uses {style} and lasts {project['effective_duration']:.3f} seconds."
    ]
    if plan["shots"] and _clean_block(project["initial_prompt"]):
        lines.append(_sentence(project["initial_prompt"]))

    for video in catalog["videos"]:
        label = catalog["video_labels"][video["asset_id"]]
        if video["relationship"] == VIDEO_EDIT:
            lines.append(
                f"The visible timeline begins from {label} and applies only the requested edits."
            )
        elif video["relationship"] == VIDEO_CONTINUE:
            lines.append(
                f"The target begins immediately after {label} and continues its established state."
            )

    events_by_shot: dict[int, list[dict]] = {}
    for event in plan["dialogue_events"]:
        events_by_shot.setdefault(event["shot_number"], []).append(event)

    for shot in _implicit_or_planned_shots(plan):
        number = shot["shot_number"]
        description = _sentence(
            shot["description"], "Describe the target action and setting."
        )
        replacement_instructions = [
            _replacement_shot_instruction(replacement, plan, catalog)
            for replacement in plan["character_replacements"]
            if number in _replacement_scope(replacement, plan)
        ]
        if replacement_instructions:
            description = " ".join((description, *replacement_instructions))
        camera = _sentence(shot["camera_direction"])
        if number == 1:
            line = f"[Shot 1] {description}"
        else:
            transition = _transition_prefix(shot["transition"])
            line = (
                f"[Shot {number}] At {_format_timestamp(shot['cut_at'])}, "
                f"{transition} {description[0].lower() + description[1:] if description else description}"
            )
        if camera:
            line += f" Camera: {camera}"
        lines.append(line)
        lines.extend(
            _dialogue_text(event, plan, catalog)
            for event in events_by_shot.get(number, [])
        )
    return "\n".join(lines)


def _base_description(plan: dict, catalog: dict) -> str:
    project = plan["project"]
    style = _sentence(project["visual_style"])
    events_by_shot: dict[int, list[dict]] = {}
    for event in plan["dialogue_events"]:
        events_by_shot.setdefault(event["shot_number"], []).append(event)
    lines: list[str] = []
    for shot in _implicit_or_planned_shots(plan):
        number = shot["shot_number"]
        description = _sentence(
            shot["description"], "Describe the target action and setting."
        )
        if number == 1:
            prefix = f"{style} " if style else ""
            line = f"[Shot 1] {prefix}{description}"
        else:
            transition = _transition_prefix(shot["transition"])
            line = (
                f"[Shot {number}] At {_format_timestamp(shot['cut_at'])}, "
                f"{transition} {description[0].lower() + description[1:] if description else description}"
            )
        camera = _sentence(shot["camera_direction"])
        if camera:
            line += f" Camera: {camera}"
        lines.append(line)
        lines.extend(
            _dialogue_text(event, plan, catalog)
            for event in events_by_shot.get(number, [])
        )
    return "\n".join(lines)


def _audio_sections(plan: dict, catalog: dict) -> tuple[str, str]:
    project = plan["project"]
    relationships = {entry["asset_id"]: entry for entry in plan["audio_relationships"]}
    complete = [
        audio
        for audio in catalog["audios"]
        if relationships[audio["asset_id"]]["use"] == AUDIO_COPY_COMPLETE
    ]
    if complete:
        label = catalog["audio_labels"][complete[0]["asset_id"]]
        return (
            f"Reuse {label} as the complete final audio track; do not add, replace, remix, "
            "or synthesize any dialogue, ambience, effects, or music.",
            f"Contained entirely in {label}; do not add or replace any audience-only music.",
        )

    soundscape = _clean_block(project["overall_soundscape"], "N/A")
    music = _clean_block(project["non_diegetic_music"], "N/A")
    music_references: list[str] = []
    sound_references: list[str] = []
    for audio in catalog["audios"]:
        relationship = relationships[audio["asset_id"]]
        label = catalog["audio_labels"][audio["asset_id"]]
        if relationship["use"] == AUDIO_MUSIC:
            music_references.append(
                f"Use {label} as the style reference for {relationship['target_layer_or_event']} "
                "without copying its signal."
            )
        elif relationship["use"] in {
            AUDIO_BEAT,
            AUDIO_SFX,
            AUDIO_CONTINUITY,
            AUDIO_COPY_PARTIAL,
            AUDIO_BROAD,
        } and not _shot_prose_mentions_label(plan, label):
            paired_video_id = audio.get("paired_video_asset_id")
            paired_video = next(
                (
                    video
                    for video in catalog["videos"]
                    if video["asset_id"] == paired_video_id
                ),
                None,
            )
            if (
                relationship["use"] == AUDIO_CONTINUITY
                and paired_video is not None
                and paired_video["relationship"] == VIDEO_CONTINUE
            ):
                video_label = catalog["video_labels"][paired_video_id]
                sound_references.append(
                    f"Using {label}, continue {relationship['target_layer_or_event']} after "
                    f"{video_label}'s final audible state with newly generated, forward-developing "
                    "audio; do not restart, replay, repeat, or loop the source signal."
                )
            else:
                sound_references.append(
                    f"Apply {label} only according to its declared "
                    f"{relationship['use'].lower()} role."
                )
    if sound_references:
        soundscape = " ".join([soundscape, *sound_references])
    if music_references:
        music = " ".join([music, *music_references])
    return soundscape, music


def _endpoint_preamble(mode: str, plan: dict, catalog: dict) -> str:
    if mode == MODE_T2VA:
        return ""
    first = next(
        (
            catalog["picture_labels"][asset["asset_id"]]
            for asset in catalog["pictures"]
            if asset["relationship"] == IMAGE_FIRST_FRAME
        ),
        None,
    )
    last = next(
        (
            catalog["picture_labels"][asset["asset_id"]]
            for asset in catalog["pictures"]
            if asset["relationship"] == IMAGE_LAST_FRAME
        ),
        None,
    )
    final_shot = max(1, len(plan["shots"]))
    duration = plan["project"]["effective_duration"]
    if mode == MODE_I2VA:
        return (
            f"For the target video, at 0.00 seconds into the target video, "
            f"{first} (from [Shot 1]) is fully referenced."
        )
    if mode == MODE_L2VA:
        return (
            f"For the target video, at {duration:.2f} seconds into the target video, "
            f"{last} (from [Shot {final_shot}]) is fully referenced."
        )
    return (
        "How the reference pictures align with the target video — "
        f"{first} (from [Shot 1]) aligns with the 0.00-second mark of the target video; "
        f"{last} (from [Shot {final_shot}]) aligns with the {duration:.2f}-second mark "
        "of the target video."
    )


def _native_routes(mode: str, catalog: dict) -> list[dict]:
    if mode == MODE_REF2VA:
        return [dict(entry) for entry in catalog["presentation"]]
    routes: list[dict] = []
    for asset in catalog["pictures"]:
        relationship = asset["relationship"]
        route = "first_frame" if relationship == IMAGE_FIRST_FRAME else "last_frame"
        routes.append(
            {
                "asset_id": asset["asset_id"],
                "label": catalog["picture_labels"][asset["asset_id"]],
                "route": route,
                "media_kind": "image",
            }
        )
    return routes


def _problems_report(
    plan: dict,
    catalog: dict,
    mode: str,
    checkpoint: str,
    routes: list[dict],
) -> str:
    lines = [
        "Plan ready: 0 errors.",
        f"Mode: {mode}",
        f"Checkpoint: {checkpoint}",
        (
            f"Timing: requested {plan['project']['duration_seconds']:.3f}s; native "
            f"{plan['project']['h3_length']} frames at {H3_FPS} FPS = "
            f"{plan['project']['effective_duration']:.3f}s."
        ),
        (
            f"Inventory: {len(catalog['pictures'])} picture(s), "
            f"{len(catalog['videos'])} video(s), {len(catalog['audios'])} audio clip(s), "
            f"{len(plan['character_replacements'])} character replacement mapping(s), "
            f"{len(plan['shots']) or 1} shot(s), {len(plan['dialogue_events'])} vocal event(s)."
        ),
        "Native routes:",
    ]
    if routes:
        lines.extend(f"- {entry['label']} -> {entry['route']}" for entry in routes)
    else:
        lines.append("- none")
    lines.append(
        "Locked by the compiler: labels, roles, character replacements, retention, speakers, "
        "exact dialogue, and cut times."
    )
    return "\n".join(lines)


def _rewrite_request(prompt: str, report: str, mode: str) -> str:
    section_rule = (
        "Preserve exactly these six section names and their order: subject_definitions, "
        "summary, retention_analysis, detailed_description, overall_soundscape, "
        "non_diegetic_music."
        if mode == MODE_REF2VA
        else "Preserve the endpoint preamble when present and exactly these three fields: "
        "integrated_multimodal_description, overall_soundscape, non_diegetic_music."
    )
    return f"""Improve only the descriptive prose in this valid MiniMax H3 {mode} prompt.

{section_rule}
Do not add, remove, rename, or renumber any <Subject N>, <Picture N>, <Video N>, or <Audio N> label.
Do not change task types, retention markers, native roles, speaker IDs, exact <d> dialogue,
languages, Shot order, or cut timestamps. Do not infer audio meaning from imagery. Return only
the complete enhanced H3 prompt. If an instruction conflicts with a locked fact, preserve the
locked fact.

COMPILER REPORT
{report}

VALID DRAFT
{prompt}"""


def compile_h3_plan(plan: Any) -> tuple[str, str, str, dict, int]:
    """Compile one completed Plan v2 chain into an H3 prompt package."""

    validated = validated_plan(
        plan,
        allowed_phases={PHASE_SETUP, PHASE_TIMELINE},
    )
    catalog = _catalog(validated)
    mode, checkpoint = _determine_mode(validated, catalog)
    _validate_final_plan(validated, catalog, mode)
    routes = _native_routes(mode, catalog)
    soundscape, music = _audio_sections(validated, catalog)

    if mode == MODE_REF2VA:
        prompt = (
            "subject_definitions:\n"
            + "\n".join(_subject_definitions(validated, catalog))
            + "\n\nsummary:\n"
            + _summary(validated, catalog)
            + "\n\nretention_analysis:\n"
            + "\n".join(_retention_analysis(validated, catalog))
            + "\n\ndetailed_description:\n"
            + _detailed_description(validated, catalog)
            + "\n\noverall_soundscape:\n"
            + soundscape
            + "\n\nnon_diegetic_music:\n"
            + music
        )
    else:
        preamble = _endpoint_preamble(mode, validated, catalog)
        body = (
            "integrated_multimodal_description: "
            + _base_description(validated, catalog)
            + "\n\noverall_soundscape: "
            + soundscape
            + "\n\nnon_diegetic_music: "
            + music
        )
        prompt = f"{preamble}\n\n{body}" if preamble else body

    report = _problems_report(validated, catalog, mode, checkpoint, routes)
    rewrite = _rewrite_request(prompt, report, mode)
    compiled_plan = _copy_plan(validated)
    compiled_plan["phase"] = PHASE_COMPILED
    compiled_plan["compiled"] = {
        "mode": mode,
        "checkpoint": checkpoint,
        "routes": [dict(entry) for entry in routes],
        "subject_labels": {
            group["subject_name"]: group["label"] for group in catalog["subject_groups"]
        },
        "speaker_ids": {
            next(
                event["speaker"]
                for event in validated["dialogue_events"]
                if _alias_key(event["speaker"]) == key
            ): value
            for key, value in catalog["speaker_ids"].items()
        },
    }
    return prompt, rewrite, report, compiled_plan, validated["project"]["h3_length"]


class MiniMaxH3PlanV2ProjectSetup:
    """Start one ordered H3 semantic plan."""

    CATEGORY = "MiniMax H3/Plan v2"
    FUNCTION = "start"
    RETURN_TYPES = (PLAN_TYPE, "INT", "STRING")
    RETURN_NAMES = ("h3_plan", "h3_length", "project_preview")
    OUTPUT_TOOLTIPS = (
        "Connect to the first Plan v2 Image, Video, Audio, or Shot node.",
        "Native H3 target length on the required 17k+5 frame grid at 24 FPS.",
        "Requested and effective timing plus the global audiovisual choices.",
    )
    DESCRIPTION = (
        "Starts the ordered Plan v2 workflow. It owns global intent and duration only; "
        "references and Shots are added by following nodes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "initial_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": "Global creative intent for the complete target video.",
                        "tooltip": (
                            "Describe the overall target and requested change. Shot nodes later hold "
                            "the chronological detail; this field may be enough for a simple one-shot video."
                        ),
                    },
                ),
                "duration_seconds": (
                    "FLOAT",
                    {
                        "default": 6.0,
                        "min": 4.0,
                        "max": 15.0,
                        "step": 0.01,
                        "tooltip": (
                            "Authoritative requested duration. Prompt Merge reports the slightly "
                            "adjusted native duration after upward 17k+5 alignment."
                        ),
                    },
                ),
                "visual_style": (
                    "STRING",
                    {
                        "default": "cinematic, live-action",
                        "placeholder": "Example: cinematic, live-action",
                        "tooltip": "Optional global visual treatment shared by every Shot.",
                    },
                ),
                "overall_soundscape": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": "Ambience and synchronized physical sounds.",
                        "tooltip": (
                            "Global ambience and physical sounds. Leave blank when a complete copied "
                            "audio signal will become the entire final track."
                        ),
                    },
                ),
                "non_diegetic_music": (
                    "STRING",
                    {
                        "default": "N/A",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": "Audience-only score, or N/A.",
                        "tooltip": "Audience-only score. Use N/A when none is requested.",
                    },
                ),
            }
        }

    def start(
        self,
        initial_prompt: str,
        duration_seconds: float,
        visual_style: str,
        overall_soundscape: str,
        non_diegetic_music: str,
    ):
        plan = _new_plan(
            initial_prompt,
            duration_seconds,
            visual_style,
            overall_soundscape,
            non_diegetic_music,
        )
        project = plan["project"]
        preview = (
            f"Project ready. Requested {project['duration_seconds']:.3f}s; native "
            f"{project['h3_length']} frames at {H3_FPS} FPS = "
            f"{project['effective_duration']:.3f}s.\n"
            f"Style: {project['visual_style'] or 'derive from intent/references'}\n"
            f"Soundscape: {project['overall_soundscape'] or 'not specified'}\n"
            f"Non-diegetic music: {project['non_diegetic_music']}"
        )
        return plan, project["h3_length"], preview


class MiniMaxH3PlanV2ImageReference:
    """Register one image and its exact primary H3 relationship."""

    CATEGORY = "MiniMax H3/Plan v2"
    FUNCTION = "add_image"
    RETURN_TYPES = (PLAN_TYPE, REFERENCE_HANDLE_TYPE, "IMAGE", "STRING")
    RETURN_NAMES = ("h3_plan", "reference_handle", "h3_image", "reference_preview")
    OUTPUT_TOOLTIPS = (
        "Continue the ordered setup chain from this output.",
        "Connect to Subject Binding when this one image defines additional Subjects or roles.",
        "Original image for the native route shown by Prompt Merge.",
        "Provisional Picture number, exact relationship, Subject alias, and scope.",
    )
    DESCRIPTION = (
        "Registers one real image and one exact role. It creates a Subject only when "
        "Define reusable visible content is explicitly selected."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_plan": (
                    PLAN_TYPE,
                    {
                        "tooltip": (
                            "Connect the preceding Project or reference/binding node. "
                            "All references must be added before the first Shot."
                        )
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Exactly one source image. Image batches are intentionally rejected."
                    },
                ),
                "image_use": (
                    IMAGE_USES,
                    {
                        "default": UNASSIGNED_IMAGE_USE,
                        "tooltip": (
                            "Choose what this image actually controls. Identity/object/scene/style "
                            "roles create reusable content; endpoint/keyframe roles remain Pictures."
                        ),
                    },
                ),
                "reference_name": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Example: woman portrait",
                        "tooltip": "Human-readable asset name used in previews and error messages.",
                    },
                ),
                "description": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": "What is important in this reference?",
                        "tooltip": "Visible source facts or the exact composition to preserve.",
                    },
                ),
                "content_type": (
                    CONTENT_TYPES,
                    {
                        "default": UNASSIGNED_CONTENT_TYPE,
                        "tooltip": (
                            "Required only for Define reusable visible content. The browser UI hides "
                            "this control for direct Picture roles."
                        ),
                    },
                ),
                "subject_name": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Example: woman, truck, Arizona road",
                        "tooltip": (
                            "Stable human alias for reusable visible content. Prompt Merge converts "
                            "it to a final Subject number."
                        ),
                    },
                ),
                "retention": (
                    VISUAL_RETENTIONS,
                    {
                        "default": RETENTION_AUTO,
                        "tooltip": (
                            "Auto uses full preservation for identity/object/scene and exact frames, "
                            "weak reference for style/storyboard, and transfer for action/motion."
                        ),
                    },
                ),
                "shot_scope": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Examples: 3, 3,4, 3-4, all",
                        "tooltip": (
                            "Optional numeric scope only. Do not rewrite Shot prose here. Scoped "
                            "Subjects must be cited in those Shot descriptions."
                        ),
                    },
                ),
                "transfer_target_subject": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Existing upstream Subject alias",
                        "tooltip": (
                            "Used only for attribute_transfer. The target Subject must already exist "
                            "upstream in this setup chain."
                        ),
                    },
                ),
            }
        }

    def add_image(
        self,
        h3_plan,
        image,
        image_use: str,
        reference_name: str,
        description: str,
        content_type: str,
        subject_name: str,
        retention: str,
        shot_scope: str,
        transfer_target_subject: str,
    ):
        plan = validated_plan(h3_plan, allowed_phases={PHASE_SETUP})
        if image_use not in IMAGE_USES[1:]:
            raise ValueError("Choose an explicit image relationship before queuing.")
        _validate_image(image)
        if image_use == IMAGE_DEFINE_VISIBLE:
            if content_type not in CONTENT_TYPES[1:]:
                raise ValueError(
                    "Choose a visible content type for this reusable Subject."
                )
            if not _clean_inline(subject_name):
                raise ValueError(
                    "Define reusable visible content requires subject_name."
                )
        elif content_type != UNASSIGNED_CONTENT_TYPE or _clean_inline(subject_name):
            raise ValueError(
                "content_type and subject_name are only used by Define reusable visible content. "
                "Clear them for a direct Picture role."
            )
        if image_use == IMAGE_KEYFRAME and not _clean_inline(shot_scope):
            raise ValueError("A concrete keyframe needs a numeric shot_scope.")
        if image_use in {IMAGE_FIRST_FRAME, IMAGE_LAST_FRAME} and _clean_inline(
            shot_scope
        ):
            raise ValueError(
                "Exact first/last frames derive their Shot automatically; clear shot_scope."
            )
        resolved_retention = _validate_image_retention(
            image_use,
            content_type,
            retention,
            transfer_target_subject,
        )
        asset = {
            "asset_id": _next_asset_id(plan, "image"),
            "media_kind": "image",
            "media": image,
            "relationship": image_use,
            "reference_name": _clean_inline(reference_name, "image reference"),
            "description": _clean_block(description),
            "retention": resolved_retention,
            "shot_scope": _clean_inline(shot_scope),
            "target_subject": "",
        }
        updated = _copy_plan(plan)
        updated["assets"].append(asset)
        if image_use == IMAGE_DEFINE_VISIBLE:
            updated = _append_binding(
                updated,
                asset,
                subject_name,
                content_type,
                resolved_retention,
                shot_scope,
                description,
                transfer_target_subject,
            )
        _validate_reference_counts(updated)
        picture_number = len(
            [entry for entry in updated["assets"] if entry["media_kind"] == "image"]
        )
        subject_text = ""
        if image_use == IMAGE_DEFINE_VISIBLE:
            subject = _catalog(updated)["subjects_by_alias"][_alias_key(subject_name)]
            subject_text = f"; provisional {subject['label']}={subject['subject_name']}"
        preview = (
            f"Provisional <Picture {picture_number}>: {image_use}{subject_text}. "
            "Prompt Merge assigns the authoritative label and native route."
        )
        return updated, _reference_handle(asset), image, preview


class MiniMaxH3PlanV2SubjectBinding:
    """Attach another explicit visible-content role to registered media."""

    CATEGORY = "MiniMax H3/Plan v2"
    FUNCTION = "bind_subject"
    RETURN_TYPES = (PLAN_TYPE, REFERENCE_HANDLE_TYPE, "STRING")
    RETURN_NAMES = ("h3_plan", "reference_handle", "binding_preview")
    OUTPUT_TOOLTIPS = (
        "Continue the setup chain after adding this advanced binding.",
        "Pass-through handle so another binding can target the same physical media.",
        "Subject alias, exact role, retention, source media, transfer target, and scope.",
    )
    DESCRIPTION = (
        "Advanced node for one image/video that defines several Subjects or roles. "
        "It never registers or routes the physical media a second time."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_plan": (
                    PLAN_TYPE,
                    {
                        "tooltip": "Connect the plan containing the referenced image or video."
                    },
                ),
                "reference_handle": (
                    REFERENCE_HANDLE_TYPE,
                    {
                        "tooltip": (
                            "Connect reference_handle from the Image/Video Reference whose media "
                            "provides this additional role."
                        )
                    },
                ),
                "subject_name": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Example: jacket",
                        "tooltip": "Human alias for the reusable visible Subject created by this binding.",
                    },
                ),
                "content_type": (
                    CONTENT_TYPES,
                    {
                        "default": UNASSIGNED_CONTENT_TYPE,
                        "tooltip": "The exact visible property this media defines for the Subject.",
                    },
                ),
                "retention": (
                    VISUAL_RETENTIONS,
                    {
                        "default": RETENTION_AUTO,
                        "tooltip": "One explicit retention relationship for this Subject binding.",
                    },
                ),
                "shot_scope": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Examples: 3,4 or 3-4",
                        "tooltip": "Optional numeric Shot scope; no Shot prose belongs in this field.",
                    },
                ),
                "notes": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Visible facts that belong to this role only.",
                    },
                ),
                "transfer_target_subject": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Existing upstream Subject alias",
                        "tooltip": "Required only for attribute_transfer.",
                    },
                ),
            }
        }

    def bind_subject(
        self,
        h3_plan,
        reference_handle,
        subject_name: str,
        content_type: str,
        retention: str,
        shot_scope: str,
        notes: str,
        transfer_target_subject: str,
    ):
        plan = validated_plan(h3_plan, allowed_phases={PHASE_SETUP})
        asset = _resolved_handle(
            plan,
            reference_handle,
            allowed_kinds={"image", "video"},
        )
        if asset["media_kind"] == "image" and asset.get("relationship") in {
            IMAGE_FIRST_FRAME,
            IMAGE_LAST_FRAME,
        }:
            raise ValueError(
                "Exact first/last-frame images cannot receive Subject Bindings because "
                "that would switch the plan away from native endpoint conditioning. "
                "Register a separate Ref2VA image instead."
            )
        updated = _append_binding(
            plan,
            asset,
            subject_name,
            content_type,
            retention,
            shot_scope,
            notes,
            transfer_target_subject,
        )
        binding = updated["bindings"][-1]
        subject = _catalog(updated)["subjects_by_alias"][_alias_key(subject_name)]
        preview = (
            f"Provisional {subject['label']}={binding['subject_name']}: "
            f"{binding['content_type']} "
            f"[{binding['retention']}] from {asset['reference_name']}"
            + (f"; shots={binding['shot_scope']}" if binding["shot_scope"] else "")
            + (
                f"; transfer target={binding['transfer_target_subject']}"
                if binding["transfer_target_subject"]
                else ""
            )
        )
        return updated, dict(reference_handle), preview


class MiniMaxH3PlanV2VideoReference:
    """Register one reference-video frame batch and its exact relationship."""

    CATEGORY = "MiniMax H3/Plan v2"
    FUNCTION = "add_video"
    RETURN_TYPES = (PLAN_TYPE, REFERENCE_HANDLE_TYPE, "IMAGE", "STRING")
    RETURN_NAMES = ("h3_plan", "reference_handle", "h3_video", "reference_preview")
    OUTPUT_TOOLTIPS = (
        "Continue the ordered setup chain.",
        "Connect to Subject Binding or an Audio Reference paired-video input.",
        "Frames resampled to 24 FPS and aligned/truncated to the native H3 grid.",
        "Effective media timing, relationship, target, and provisional native route.",
    )
    DESCRIPTION = (
        "Registers an IMAGE frame batch as one H3 reference video. Editing, continuation, "
        "motion transfer, structure transfer, and visible-content definition remain distinct."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_plan": (
                    PLAN_TYPE,
                    {"tooltip": "Connect the preceding Plan v2 setup node."},
                ),
                "video_frames": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Decoded reference-video frames. source_fps describes this incoming batch; "
                            "the node returns the native 24-FPS aligned batch."
                        )
                    },
                ),
                "video_use": (
                    VIDEO_USES,
                    {
                        "default": UNASSIGNED_VIDEO_USE,
                        "tooltip": "Choose the exact whole-video or visible-content relationship.",
                    },
                ),
                "reference_name": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Example: source truck video",
                        "tooltip": "Human-readable source name for reports.",
                    },
                ),
                "description": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": "What should be edited, continued, transferred, or retained?",
                        "tooltip": "Source facts and the exact requested relationship.",
                    },
                ),
                "source_fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 0.01,
                        "max": 240.0,
                        "step": 0.01,
                        "tooltip": "Frame rate represented by the connected IMAGE batch.",
                    },
                ),
                "content_type": (
                    CONTENT_TYPES,
                    {
                        "default": UNASSIGNED_CONTENT_TYPE,
                        "tooltip": "Required only when this video defines reusable visible content.",
                    },
                ),
                "subject_name": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Reusable Subject alias",
                        "tooltip": "Required only when the video defines reusable visible content.",
                    },
                ),
                "target_subject": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Existing upstream Subject alias",
                        "tooltip": (
                            "Required for Motion or action reference, and for reusable visible "
                            "content whose retention resolves to attribute_transfer. The target "
                            "must already exist upstream."
                        ),
                    },
                ),
                "retention": (
                    VISUAL_RETENTIONS,
                    {
                        "default": RETENTION_AUTO,
                        "tooltip": "Used by a visible-content Subject binding; whole-video roles derive it.",
                    },
                ),
                "shot_scope": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Examples: 2, 2-4, all",
                        "tooltip": "Optional numeric scope for the declared role.",
                    },
                ),
            }
        }

    def add_video(
        self,
        h3_plan,
        video_frames,
        video_use: str,
        reference_name: str,
        description: str,
        source_fps: float,
        content_type: str,
        subject_name: str,
        target_subject: str,
        retention: str,
        shot_scope: str,
    ):
        plan = validated_plan(h3_plan, allowed_phases={PHASE_SETUP})
        if video_use not in VIDEO_USES[1:]:
            raise ValueError("Choose an explicit video relationship before queuing.")
        h3_video, source_duration, resampled_count, native_count = (
            _prepare_video_frames(
                video_frames,
                source_fps,
                plan["project"]["h3_length"],
            )
        )
        target = _clean_inline(target_subject)
        if video_use == VIDEO_DEFINE_VISIBLE:
            if content_type not in CONTENT_TYPES[1:] or not _clean_inline(subject_name):
                raise ValueError(
                    "Define reusable visible content requires content_type and subject_name."
                )
        elif content_type != UNASSIGNED_CONTENT_TYPE or _clean_inline(subject_name):
            raise ValueError(
                "content_type and subject_name are only used when the video defines reusable content."
            )
        whole_retention = {
            VIDEO_EDIT: RETENTION_PARTIAL,
            VIDEO_CONTINUE: RETENTION_PARTIAL,
            VIDEO_MOTION: RETENTION_TRANSFER,
            VIDEO_STRUCTURE: RETENTION_WEAK,
        }.get(video_use)
        resolved_retention = (
            _resolve_binding_retention(retention, content_type, target)
            if video_use == VIDEO_DEFINE_VISIBLE
            else whole_retention
        )

        if video_use == VIDEO_MOTION:
            if not target:
                raise ValueError("Motion or action reference requires target_subject.")
            aliases = {_alias_key(entry["subject_name"]) for entry in plan["bindings"]}
            if _alias_key(target) not in aliases:
                raise ValueError(
                    f"Motion target {target!r} is not an upstream Subject. "
                    "Define the target image/Subject first."
                )
        elif video_use != VIDEO_DEFINE_VISIBLE and target:
            raise ValueError(
                "target_subject is only used by Motion or action reference or by "
                "reusable visible content with attribute_transfer retention."
            )
        asset = {
            "asset_id": _next_asset_id(plan, "video"),
            "media_kind": "video",
            "media": h3_video,
            "relationship": video_use,
            "reference_name": _clean_inline(reference_name, "video reference"),
            "description": _clean_block(description),
            "retention": resolved_retention,
            "shot_scope": _clean_inline(shot_scope),
            "target_subject": target,
            "source_duration": source_duration,
            "source_fps": float(source_fps),
            "resampled_frame_count": resampled_count,
            "native_frame_count": native_count,
            "native_duration": native_count / H3_FPS,
        }
        updated = _copy_plan(plan)
        updated["assets"].append(asset)
        if video_use == VIDEO_DEFINE_VISIBLE:
            updated = _append_binding(
                updated,
                asset,
                subject_name,
                content_type,
                resolved_retention,
                shot_scope,
                description,
                target,
            )
        _validate_reference_counts(updated)
        video_number = len(
            [entry for entry in updated["assets"] if entry["media_kind"] == "video"]
        )
        preview = (
            f"Provisional <Video {video_number}>: {video_use}. Source "
            f"{source_duration:.3f}s -> native {native_count} frames/"
            f"{native_count / H3_FPS:.3f}s at 24 FPS; route ref_video_{video_number - 1}."
        )
        return updated, _reference_handle(asset), h3_video, preview


class MiniMaxH3PlanV2CharacterReplacement:
    """Map one performer in a source edit or continuation to one referenced Subject."""

    CATEGORY = "MiniMax H3/Plan v2"
    FUNCTION = "add_replacement"
    RETURN_TYPES = (PLAN_TYPE, "STRING")
    RETURN_NAMES = ("h3_plan", "replacement_preview")
    OUTPUT_TOOLTIPS = (
        "Continue the ordered setup chain before adding Shots.",
        "Resolved source performer, replacement Subject, appearance policy, and Shot scope.",
    )
    DESCRIPTION = (
        "Declares an exact character replacement inside a source video edit or newly generated "
        "continuation. Select the source Video handle and an upstream identity Subject; Prompt "
        "Merge writes and locks the relationship into every scoped Shot."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_plan": (
                    PLAN_TYPE,
                    {
                        "tooltip": (
                            "Connect the same setup chain that already contains the replacement "
                            "Subject and source Video Reference."
                        )
                    },
                ),
                "source_video": (
                    REFERENCE_HANDLE_TYPE,
                    {
                        "tooltip": (
                            "Connect reference_handle from the Video Reference whose video_use "
                            "is Source video to edit or Source video to continue. For continuation, "
                            "the mapping affects only newly generated frames after the endpoint."
                        )
                    },
                ),
                "replacement_subject": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Choose an upstream identity Subject",
                        "tooltip": (
                            "Human alias of the referenced character that replaces the source "
                            "performer. The browser picker lists upstream Subjects."
                        ),
                    },
                ),
                "source_character_description": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": (
                            "Example: the woman in the red jacket seated beside the driver"
                        ),
                        "tooltip": (
                            "Identify exactly one performer already visible in the source video. "
                            "Use plain visual language, not <Subject N> or <Video N> labels."
                        ),
                    },
                ),
                "appearance_policy": (
                    CHARACTER_REPLACEMENT_APPEARANCE_POLICIES,
                    {
                        "default": REPLACEMENT_IDENTITY_KEEP_BODY_WARDROBE,
                        "tooltip": (
                            "Choose which visible properties come from the replacement Subject "
                            "and which remain from the source performer."
                        ),
                    },
                ),
                "preserve_performance": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "For an edit, keep the source performance and timing. For a "
                            "continuation, begin from the endpoint pose, position, motion "
                            "direction, gaze, expression, and interactions, then move forward."
                        ),
                    },
                ),
                "preserve_scene": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "For an edit, keep the source scene and cuts. For a continuation, "
                            "carry other people, environment, props, lighting, camera, framing, "
                            "and spatial state forward from the endpoint without a reset."
                        ),
                    },
                ),
                "shot_scope": (
                    "STRING",
                    {
                        "default": "all",
                        "placeholder": "Examples: 2, 2-4, all",
                        "tooltip": (
                            "Generated Shots where the source performer is replaced. For a "
                            "continuation this never modifies frames inside the supplied source. "
                            "The compiler inserts the typed mapping into every selected Shot."
                        ),
                    },
                ),
                "instructions": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": "Optional continuity exception or replacement constraint",
                        "tooltip": (
                            "Optional plain-language constraint specific to this replacement. "
                            "Do not type numbered H3 labels; the compiler owns them."
                        ),
                    },
                ),
            }
        }

    def add_replacement(
        self,
        h3_plan,
        source_video,
        replacement_subject: str,
        source_character_description: str,
        appearance_policy: str,
        preserve_performance: bool,
        preserve_scene: bool,
        shot_scope: str,
        instructions: str,
    ):
        plan = validated_plan(h3_plan, allowed_phases={PHASE_SETUP})
        video = _resolved_handle(plan, source_video, allowed_kinds={"video"})
        if video["relationship"] not in {VIDEO_EDIT, VIDEO_CONTINUE}:
            raise ValueError(
                "Character Replacement requires a Video Reference set to Source video to edit "
                "or Source video to continue."
            )
        alias = _clean_inline(replacement_subject)
        catalog = _catalog(plan)
        group = catalog["subjects_by_alias"].get(_alias_key(alias))
        if group is None:
            raise ValueError(
                f"replacement_subject {alias!r} is not an upstream Subject. "
                "Define the character image and Subject first."
            )
        if not any(
            binding["content_type"] == CONTENT_IDENTITY
            for binding in group["bindings"]
        ):
            raise ValueError(
                f"{group['label']} ({group['subject_name']}) needs an Identity or "
                "appearance binding before it can replace a character."
            )
        source_character = _clean_inline(source_character_description)
        if not source_character:
            raise ValueError(
                "Describe the exact source performer to replace, such as "
                "'the woman in the red jacket'."
            )
        extra = _clean_block(instructions)
        if _REFERENCE_TOKEN_RE.search(source_character) or _REFERENCE_TOKEN_RE.search(
            extra
        ):
            raise ValueError(
                "Use plain language in Character Replacement fields; Prompt Merge assigns "
                "the numbered H3 labels."
            )
        if appearance_policy not in CHARACTER_REPLACEMENT_APPEARANCE_POLICIES:
            raise ValueError(
                "Choose a supported Character Replacement appearance policy."
            )
        scope = _clean_inline(shot_scope)
        if not scope:
            raise ValueError(
                "Character Replacement requires shot_scope, such as 2-4 or all."
            )
        source_key = (video["asset_id"], _alias_key(source_character))
        if any(
            (
                entry["source_video_asset_id"],
                _alias_key(entry["source_character_description"]),
            )
            == source_key
            for entry in plan["character_replacements"]
        ):
            raise ValueError(
                "That source performer already has a Character Replacement in this plan."
            )
        replacement = {
            "replacement_id": _next_character_replacement_id(plan),
            "source_video_asset_id": video["asset_id"],
            "replacement_subject": group["subject_name"],
            "source_character_description": source_character,
            "appearance_policy": appearance_policy,
            "preserve_performance": bool(preserve_performance),
            "preserve_scene": bool(preserve_scene),
            "shot_scope": scope,
            "instructions": extra,
        }
        updated = _copy_plan(plan)
        updated["character_replacements"].append(replacement)
        video_label = catalog["video_labels"][video["asset_id"]]
        preview = (
            f"{video_label} performer ({source_character}) -> {group['label']} "
            f"({group['subject_name']}); shots={scope}; policy={appearance_policy}."
        )
        return updated, preview


class MiniMaxH3PlanV2AudioReference:
    """Register one audio clip with one exact semantic relationship."""

    CATEGORY = "MiniMax H3/Plan v2"
    FUNCTION = "add_audio"
    RETURN_TYPES = (PLAN_TYPE, "AUDIO", "STRING")
    RETURN_NAMES = ("h3_plan", "h3_audio", "reference_preview")
    OUTPUT_TOOLTIPS = (
        "Continue the ordered setup chain.",
        "Exact input audio segment for the native standalone or paired route assigned by Prompt Merge.",
        "Exact audio role, duration, target, retention, and paired/standalone route.",
    )
    DESCRIPTION = (
        "Registers one real audio clip as voice, music, beat, effect, content, continuity, "
        "copy, or weak inspiration. These meanings are never combined into one generic option."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_plan": (
                    PLAN_TYPE,
                    {"tooltip": "Connect the preceding Plan v2 setup node."},
                ),
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "One 2-15 second ComfyUI AUDIO segment. All Audio References in the "
                            "same Plan v2 chain share a 15-second cumulative limit; Reference Sheet "
                            "can trim its selected_audio output non-destructively."
                        )
                    },
                ),
                "audio_use": (
                    AUDIO_USES,
                    {
                        "default": UNASSIGNED_AUDIO_USE,
                        "tooltip": (
                            "Choose exactly one relationship. The browser UI shows only the controls "
                            "required by the selected relationship."
                        ),
                    },
                ),
                "reference_name": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Example: woman's voice",
                        "tooltip": "Human-readable source name for previews.",
                    },
                ),
                "target_speaker": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Subject alias or nonvisual speaker name",
                        "tooltip": "Required only for voice timbre or dialogue/lyric content.",
                    },
                ),
                "language": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Example: English",
                        "tooltip": "Required only when the source provides exact dialogue or lyric content.",
                    },
                ),
                "transcript": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": "Exact source words; write [unclear] where necessary.",
                        "tooltip": (
                            "Required only for dialogue/lyric content. Voice timbre alone must leave "
                            "this blank so source words are not accidentally reused."
                        ),
                    },
                ),
                "target_layer_or_event": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Music layer, action, edit rhythm, sound event, or continuity phase",
                        "tooltip": "Required for music, beat, effect, continuity, and broad inspiration.",
                    },
                ),
                "instructions": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": "Delivery notes, copied range/layers, or exact sonic guidance.",
                        "tooltip": "Relationship-specific detail. Partial copy requires exact range/layers.",
                    },
                ),
                "shot_scope": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Examples: 1, 2-3, all",
                        "tooltip": (
                            "Optional numeric Shot scope for this audio relationship. For a sound "
                            "effect, also insert its <Audio N> tag inside the exact Shot sentence "
                            "where the sound occurs; scope validates placement but does not write it."
                        ),
                    },
                ),
            },
            "optional": {
                "paired_video": (
                    REFERENCE_HANDLE_TYPE,
                    {
                        "tooltip": (
                            "Connect a Video Reference handle only when this is that video's paired "
                            "soundtrack. Otherwise it is routed as standalone ref_audio_N."
                        )
                    },
                )
            },
        }

    def add_audio(
        self,
        h3_plan,
        audio,
        audio_use: str,
        reference_name: str,
        target_speaker: str,
        language: str,
        transcript: str,
        target_layer_or_event: str,
        instructions: str,
        shot_scope: str,
        paired_video=None,
    ):
        plan = validated_plan(h3_plan, allowed_phases={PHASE_SETUP})
        if audio_use not in AUDIO_USES[1:]:
            raise ValueError("Choose an explicit audio relationship before queuing.")
        duration = _audio_duration(audio)
        speaker = _clean_inline(target_speaker)
        source_language = _clean_inline(language)
        source_transcript = _clean_block(transcript)
        layer = _clean_inline(target_layer_or_event)
        detail = _clean_block(instructions)
        if audio_use in {AUDIO_VOICE, AUDIO_CONTENT} and not speaker:
            raise ValueError(f"{audio_use} requires target_speaker.")
        if audio_use == AUDIO_VOICE and source_transcript:
            raise ValueError(
                "Voice timbre and delivery must not include a transcript; use Dialogue Event "
                "for target words or choose Dialogue or lyric content."
            )
        if audio_use == AUDIO_CONTENT:
            if not source_language or not source_transcript:
                raise ValueError(
                    "Dialogue or lyric content requires both language and exact transcript."
                )
            if (
                "<d>" in source_transcript.casefold()
                or "</d>" in source_transcript.casefold()
            ):
                raise ValueError(
                    "Enter transcript words without H3 <d> tags; Prompt Merge adds them."
                )
        if (
            audio_use
            in {
                AUDIO_MUSIC,
                AUDIO_BEAT,
                AUDIO_SFX,
                AUDIO_CONTINUITY,
                AUDIO_BROAD,
            }
            and not layer
        ):
            raise ValueError(f"{audio_use} requires target_layer_or_event.")
        if audio_use == AUDIO_COPY_PARTIAL and not detail:
            raise ValueError(
                "Copy selected part or layers requires exact range/layer instructions."
            )

        paired_video_asset_id = ""
        if paired_video is not None:
            video_asset = _resolved_handle(
                plan,
                paired_video,
                allowed_kinds={"video"},
            )
            paired_video_asset_id = video_asset["asset_id"]
            if any(
                asset.get("paired_video_asset_id") == paired_video_asset_id
                for asset in plan["assets"]
                if asset["media_kind"] == "audio"
            ):
                raise ValueError(
                    "That Video Reference already has a paired soundtrack."
                )

        asset = {
            "asset_id": _next_asset_id(plan, "audio"),
            "media_kind": "audio",
            "media": audio,
            "relationship": audio_use,
            "reference_name": _clean_inline(reference_name, "audio reference"),
            "duration": duration,
            "paired_video_asset_id": paired_video_asset_id,
        }
        relationship = {
            "asset_id": asset["asset_id"],
            "use": audio_use,
            "target_speaker": (
                speaker if audio_use in {AUDIO_VOICE, AUDIO_CONTENT} else ""
            ),
            "language": source_language if audio_use == AUDIO_CONTENT else "",
            "transcript": source_transcript if audio_use == AUDIO_CONTENT else "",
            "target_layer_or_event": (
                layer
                if audio_use
                in {AUDIO_MUSIC, AUDIO_BEAT, AUDIO_SFX, AUDIO_CONTINUITY, AUDIO_BROAD}
                else ""
            ),
            "instructions": detail if audio_use != AUDIO_COPY_COMPLETE else "",
            "shot_scope": (
                _clean_inline(shot_scope) if audio_use != AUDIO_COPY_COMPLETE else ""
            ),
            "retention": _audio_retention(audio_use),
        }
        updated = _copy_plan(plan)
        updated["assets"].append(asset)
        updated["audio_relationships"].append(relationship)
        _validate_reference_counts(updated)
        route = (
            "paired soundtrack; final ref_video_audio_N route assigned by Prompt Merge"
            if paired_video_asset_id
            else (
                "standalone "
                f"ref_audio_{sum(1 for entry in updated['assets'] if entry['media_kind'] == 'audio' and not entry.get('paired_video_asset_id')) - 1}"
            )
        )
        preview = (
            f"{asset['reference_name']}: {audio_use} [{relationship['retention']}], "
            f"{duration:.3f}s; {route}. Final <Audio N> follows native presentation order; "
            "insert that tag in the intended Shot sound sentence for exact placement."
        )
        return updated, audio, preview


def _shot_chain_preview(plan: dict) -> str:
    shots = plan["shots"]
    effective_end = plan["project"]["effective_duration"]
    lines: list[str] = []
    for index, shot in enumerate(shots):
        end = shots[index + 1]["cut_at"] if index + 1 < len(shots) else effective_end
        lines.append(
            f"[Shot {shot['shot_number']}] "
            f"{_format_timestamp(shot['cut_at'])}-{_format_timestamp(end)}: "
            f"{_clean_inline(shot['description'])}"
        )
    return "\n".join(lines)


class MiniMaxH3PlanV2Shot:
    """Append one Shot start to the ordered semantic plan."""

    CATEGORY = "MiniMax H3/Plan v2"
    FUNCTION = "add_shot"
    RETURN_TYPES = (PLAN_TYPE, "STRING")
    RETURN_NAMES = ("h3_plan", "shot_preview")
    OUTPUT_TOOLTIPS = (
        "Connect to Dialogue Event, the next Shot, or Prompt Merge.",
        "Complete timeline with each end computed from the next cut or Project duration.",
    )
    DESCRIPTION = (
        "Adds one Shot in playback order. Shot 1 begins at zero; every later Shot uses "
        "one cut_at float and the Project duration closes the final range."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_plan": (
                    PLAN_TYPE,
                    {
                        "tooltip": (
                            "Connect the final reference/binding node for Shot 1, then the preceding "
                            "Shot or Dialogue Event for later Shots."
                        )
                    },
                ),
                "cut_at": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 15.0,
                        "step": 0.001,
                        "tooltip": (
                            "Shot 1 must be 0.000. Later values are the single H3 cut timestamp; "
                            "the next cut or Project duration supplies the end."
                        ),
                    },
                ),
                "description": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": "Visible and audible events during this Shot only.",
                        "tooltip": (
                            "Describe visible and audible events in chronological order. Insert an "
                            "upstream <Audio N> label directly in the sentence where its sound is "
                            "heard. The browser UI adds the < autocomplete menu."
                        ),
                    },
                ),
                "camera_direction": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Example: slow, small-amplitude push-in",
                        "tooltip": "Optional camera movement for this Shot only.",
                    },
                ),
                "transition": (
                    SHOT_TRANSITIONS,
                    {
                        "default": SHOT_TRANSITIONS[0],
                        "tooltip": "How this Shot begins. Shot 1 ignores this value.",
                    },
                ),
            }
        }

    def add_shot(
        self,
        h3_plan,
        cut_at: float,
        description: str,
        camera_direction: str,
        transition: str,
    ):
        plan = validated_plan(
            h3_plan,
            allowed_phases={PHASE_SETUP, PHASE_TIMELINE},
        )
        description_text = _clean_block(description)
        if not description_text:
            raise ValueError("Every explicit H3 Shot needs a description.")
        cut = float(cut_at)
        if not math.isfinite(cut) or cut < 0.0:
            raise ValueError("Shot cut_at must be a finite non-negative number.")
        if transition not in SHOT_TRANSITIONS:
            raise ValueError("Choose a supported Shot transition.")
        if not plan["shots"]:
            if not math.isclose(cut, 0.0, abs_tol=0.0005):
                raise ValueError("Shot 1 must use cut_at=0.000.")
            cut = 0.0
        elif cut <= float(plan["shots"][-1]["cut_at"]) + 0.0005:
            raise ValueError(
                "Every later Shot cut_at must be greater than the preceding cut."
            )
        if cut >= float(plan["project"]["duration_seconds"]):
            raise ValueError("Shot cut_at must be before the Project duration.")
        shot = {
            "shot_number": len(plan["shots"]) + 1,
            "cut_at": cut,
            "description": description_text,
            "camera_direction": _clean_block(camera_direction),
            "transition": transition,
        }
        updated = _copy_plan(plan)
        updated["phase"] = PHASE_TIMELINE
        updated["shots"].append(shot)
        return updated, _shot_chain_preview(updated)


class MiniMaxH3PlanV2DialogueEvent:
    """Append exact speech to the most recently opened Shot."""

    CATEGORY = "MiniMax H3/Plan v2"
    FUNCTION = "add_dialogue"
    RETURN_TYPES = (PLAN_TYPE, "STRING")
    RETURN_NAMES = ("h3_plan", "dialogue_preview")
    OUTPUT_TOOLTIPS = (
        "Connect to another Dialogue Event, the next Shot, or Prompt Merge.",
        "Shot attachment, exact text, delivery, and provisional first-vocal-event speaker ID.",
    )
    DESCRIPTION = (
        "Attaches one exact vocal event to the current Shot. Prompt Merge assigns S1, S2, "
        "and later IDs from this actual playback order."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_plan": (
                    PLAN_TYPE,
                    {
                        "tooltip": "Connect a Shot or earlier Dialogue Event from the same current Shot."
                    },
                ),
                "speaker": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Upstream Subject alias or nonvisual speaker name",
                        "tooltip": (
                            "Use the same human alias selected by a voice Audio Reference. "
                            "On-screen speech must resolve to an upstream Subject."
                        ),
                    },
                ),
                "language": (
                    "STRING",
                    {
                        "default": "English",
                        "placeholder": "Exact spoken language",
                        "tooltip": "Written inside the final H3 dialogue tag.",
                    },
                ),
                "exact_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": "Exact words, without H3 tags.",
                        "tooltip": "Only these exact words are placed inside the H3 dialogue tag.",
                    },
                ),
                "delivery": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Example: relieved, slightly breathless",
                        "tooltip": "Optional target delivery; this does not replace a connected voice timbre.",
                    },
                ),
                "voice_mode": (
                    DIALOGUE_MODES,
                    {
                        "default": DIALOGUE_MODES[0],
                        "tooltip": "On-screen speech, off-screen speech, or explicit voiceover.",
                    },
                ),
            }
        }

    def add_dialogue(
        self,
        h3_plan,
        speaker: str,
        language: str,
        exact_text: str,
        delivery: str,
        voice_mode: str,
    ):
        plan = validated_plan(h3_plan, allowed_phases={PHASE_TIMELINE})
        if not plan["shots"]:
            raise ValueError("Dialogue Event must follow the Shot it belongs to.")
        speaker_name = _clean_inline(speaker)
        source_language = _clean_inline(language)
        words = _clean_block(exact_text)
        if not speaker_name or not source_language or not words:
            raise ValueError(
                "Dialogue Event requires speaker, language, and exact_text."
            )
        if "<d>" in words.casefold() or "</d>" in words.casefold():
            raise ValueError(
                "Enter exact_text without H3 <d> tags; Prompt Merge adds them."
            )
        if voice_mode not in DIALOGUE_MODES:
            raise ValueError("Choose a supported Dialogue Event voice_mode.")
        aliases = {_alias_key(entry["subject_name"]) for entry in plan["bindings"]}
        if voice_mode == "On-screen speech" and _alias_key(speaker_name) not in aliases:
            raise ValueError(
                f"On-screen speaker {speaker_name!r} is not an upstream Subject. "
                "Use the Subject alias or choose an off-screen mode."
            )
        event = {
            "event_number": len(plan["dialogue_events"]) + 1,
            "shot_number": len(plan["shots"]),
            "speaker": speaker_name,
            "language": source_language,
            "exact_text": words,
            "delivery": _clean_inline(delivery),
            "voice_mode": voice_mode,
        }
        updated = _copy_plan(plan)
        updated["dialogue_events"].append(event)
        ordered_speakers: list[str] = []
        for item in updated["dialogue_events"]:
            if _alias_key(item["speaker"]) not in {
                _alias_key(existing) for existing in ordered_speakers
            }:
                ordered_speakers.append(item["speaker"])
        speaker_id = next(
            f"S{index}"
            for index, name in enumerate(ordered_speakers, start=1)
            if _alias_key(name) == _alias_key(speaker_name)
        )
        preview = (
            f"[Shot {event['shot_number']}] {speaker_name} ({speaker_id}), "
            f"{voice_mode}: <d>[{source_language}] {words}</d>"
        )
        return updated, preview


class MiniMaxH3PlanV2PromptMerge:
    """Compile a completed semantic plan without an LLM."""

    CATEGORY = "MiniMax H3/Plan v2"
    FUNCTION = "merge"
    RETURN_TYPES = ("STRING", "STRING", PLAN_TYPE, "STRING", "INT")
    RETURN_NAMES = (
        "h3_prompt",
        "rewrite_request",
        "plan_context",
        "problems_report",
        "h3_length",
    )
    OUTPUT_TOOLTIPS = (
        "Immediately usable deterministic H3 prompt with three or six correct sections.",
        "Self-contained prose-only enhancement request with semantic fields locked.",
        "Compiled plan for the structured enhancer and native adapter.",
        "Readiness, mode, inventory, native routes, timing, and locked fields.",
        "Same native 17k+5 target length produced by Project Setup.",
    )
    DESCRIPTION = (
        "Assigns all Subject/Picture/Video/Audio labels, speaker IDs, task types, "
        "retention rows, timing, and native routes deterministically. No model is called."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_plan": (
                    PLAN_TYPE,
                    {
                        "tooltip": (
                            "Connect the final Shot/Dialogue node, or the final setup node for "
                            "an implicit one-shot Project prompt."
                        )
                    },
                )
            }
        }

    def merge(self, h3_plan):
        prompt, rewrite, report, compiled_plan, h3_length = compile_h3_plan(h3_plan)
        return prompt, rewrite, compiled_plan, report, h3_length


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PlanV2ProjectSetup": MiniMaxH3PlanV2ProjectSetup,
    "MiniMaxH3PlanV2ImageReference": MiniMaxH3PlanV2ImageReference,
    "MiniMaxH3PlanV2SubjectBinding": MiniMaxH3PlanV2SubjectBinding,
    "MiniMaxH3PlanV2VideoReference": MiniMaxH3PlanV2VideoReference,
    "MiniMaxH3PlanV2CharacterReplacement": MiniMaxH3PlanV2CharacterReplacement,
    "MiniMaxH3PlanV2AudioReference": MiniMaxH3PlanV2AudioReference,
    "MiniMaxH3PlanV2Shot": MiniMaxH3PlanV2Shot,
    "MiniMaxH3PlanV2DialogueEvent": MiniMaxH3PlanV2DialogueEvent,
    "MiniMaxH3PlanV2PromptMerge": MiniMaxH3PlanV2PromptMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PlanV2ProjectSetup": "MiniMax H3 Project Setup (Plan v2)",
    "MiniMaxH3PlanV2ImageReference": "MiniMax H3 Image Reference (Plan v2)",
    "MiniMaxH3PlanV2SubjectBinding": "MiniMax H3 Subject Binding (Plan v2)",
    "MiniMaxH3PlanV2VideoReference": "MiniMax H3 Video Reference (Plan v2)",
    "MiniMaxH3PlanV2CharacterReplacement": "MiniMax H3 Character Replacement (Plan v2)",
    "MiniMaxH3PlanV2AudioReference": "MiniMax H3 Audio Reference (Plan v2)",
    "MiniMaxH3PlanV2Shot": "MiniMax H3 Shot (Plan v2)",
    "MiniMaxH3PlanV2DialogueEvent": "MiniMax H3 Dialogue Event (Plan v2)",
    "MiniMaxH3PlanV2PromptMerge": "MiniMax H3 Prompt Merge (Plan v2)",
}
