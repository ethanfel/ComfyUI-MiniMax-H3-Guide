"""Chainable, role-aware visual context for MiniMax H3 prompt enhancement."""

from __future__ import annotations

import math


REFERENCE_CONTEXT_TYPE = "MINIMAX_H3_ENHANCER_REFERENCE_CONTEXT"
ROLE_CHAIN_TYPE = "MINIMAX_H3_VISUAL_ROLE_CHAIN"

PICTURE_MEDIA = "Picture"
VIDEO_MEDIA = "Video frames"
MEDIA_TYPES = [PICTURE_MEDIA, VIDEO_MEDIA]

UNASSIGNED_ROLE = "Unassigned - choose a reference role"
IDENTITY_ROLE = "Identity or appearance"
ITEM_ROLE = "Object, prop, clothing, interface, or effect"
SCENE_ROLE = "Scene or environment"
STYLE_ROLE = "Visual style"
MOTION_ROLE = "Motion or action"
CAMERA_ROLE = "Camera, cuts, or rhythm"
EDIT_ROLE = "Source video to edit"
CONTINUE_ROLE = "Source video to continue"
FIRST_FRAME_ROLE = "Exact first frame (I2VA or FL2VA)"
LAST_FRAME_ROLE = "Exact last frame (L2VA or FL2VA)"
CONCRETE_KEYFRAME_ROLE = "Concrete Ref2VA keyframe or composition anchor"
STORYBOARD_ROLE = "Storyboard or shot planning"
KEYFRAME_ROLE = "Storyboard or keyframe"  # Serialized 0.6.x compatibility.
REFERENCE_ROLES = [
    UNASSIGNED_ROLE,
    FIRST_FRAME_ROLE,
    LAST_FRAME_ROLE,
    IDENTITY_ROLE,
    ITEM_ROLE,
    SCENE_ROLE,
    STYLE_ROLE,
    CONCRETE_KEYFRAME_ROLE,
    STORYBOARD_ROLE,
    MOTION_ROLE,
    CAMERA_ROLE,
    EDIT_ROLE,
    CONTINUE_ROLE,
    KEYFRAME_ROLE,
]

AUTO_RELATION = "Auto - choose from this role"
FULL_RELATION = "fully_preserved"
PARTIAL_RELATION = "partially_preserved"
TRANSFER_RELATION = "attribute_transfer"
WEAK_RELATION = "weak_reference"
VISUAL_RELATIONS = [
    AUTO_RELATION,
    FULL_RELATION,
    PARTIAL_RELATION,
    TRANSFER_RELATION,
    WEAK_RELATION,
]

ENDPOINT_ROLES = {FIRST_FRAME_ROLE, LAST_FRAME_ROLE}
PICTURE_ONLY_ROLES = {
    FIRST_FRAME_ROLE,
    LAST_FRAME_ROLE,
    CONCRETE_KEYFRAME_ROLE,
    STORYBOARD_ROLE,
    KEYFRAME_ROLE,
}
VIDEO_ONLY_ROLES = {CAMERA_ROLE, EDIT_ROLE, CONTINUE_ROLE}
DIRECT_PICTURE_ROLES = {CONCRETE_KEYFRAME_ROLE, STORYBOARD_ROLE, KEYFRAME_ROLE}
DIRECT_VIDEO_ROLES = {CAMERA_ROLE, EDIT_ROLE, CONTINUE_ROLE}
SUBJECT_ROLES = {IDENTITY_ROLE, ITEM_ROLE, SCENE_ROLE, STYLE_ROLE, MOTION_ROLE}

H3_VIDEO_FPS = 24.0
NATIVE_FRAME_MODULUS = 17
NATIVE_FRAME_OFFSET = 5
CONTEXT_VERSION = 2
ROLE_CHAIN_VERSION = 1

ENDPOINT_NODE_ID = "MiniMaxH3ImageToVideo"
ENDPOINT_NODE_NAME = "MiniMax H3 Image to Video"
REFERENCE_NODE_ID = "MiniMaxH3ReferenceToVideo"
REFERENCE_NODE_NAME = "MiniMax H3 Reference to Video"


def _clean_text(value) -> str:
    return " ".join(str(value or "").strip().split())


def _default_relation(role: str) -> str:
    if role in {STYLE_ROLE, STORYBOARD_ROLE, KEYFRAME_ROLE, CAMERA_ROLE}:
        return WEAK_RELATION
    if role == EDIT_ROLE:
        return PARTIAL_RELATION
    return FULL_RELATION


def _validate_binding(binding: dict, media_type: str | None = None) -> dict:
    if not isinstance(binding, dict):
        raise ValueError("Every visual role binding must be a dictionary from the role node.")
    role = binding.get("role")
    if role == UNASSIGNED_ROLE:
        raise ValueError(
            "Choose an explicit visual reference role before running the workflow. "
            "A media file alone does not imply identity, Subject, endpoint, or motion semantics."
        )
    if role not in REFERENCE_ROLES:
        raise ValueError(f"Unsupported visual reference role: {role!r}.")
    if media_type == PICTURE_MEDIA and role in VIDEO_ONLY_ROLES:
        raise ValueError(f"'{role}' requires media_type=Video frames.")
    if media_type == VIDEO_MEDIA and role in PICTURE_ONLY_ROLES:
        raise ValueError(f"'{role}' requires media_type=Picture.")

    relation = binding.get("retention", AUTO_RELATION)
    if relation not in VISUAL_RELATIONS:
        raise ValueError(f"Unsupported visual retention relationship: {relation!r}.")
    resolved_relation = _default_relation(role) if relation == AUTO_RELATION else relation
    if role in ENDPOINT_ROLES and resolved_relation != FULL_RELATION:
        raise ValueError(f"'{role}' is an exact endpoint and requires fully_preserved retention.")
    if resolved_relation == TRANSFER_RELATION and role not in SUBJECT_ROLES:
        raise ValueError(f"'{role}' cannot use attribute_transfer; choose a reusable-content role.")

    content_group = _clean_text(binding.get("content_group"))
    transfer_target = _clean_text(binding.get("transfer_target"))
    if content_group and role not in SUBJECT_ROLES:
        raise ValueError(
            f"'{role}' is tracked directly and cannot join content_group={content_group!r}; "
            "content groups are only for reusable <Subject N> content."
        )
    if transfer_target and resolved_relation != TRANSFER_RELATION:
        raise ValueError(
            "transfer_target is only valid when retention=attribute_transfer."
        )
    if resolved_relation == TRANSFER_RELATION:
        if not content_group:
            raise ValueError(
                "attribute_transfer requires a non-empty source content_group."
            )
        if not transfer_target:
            raise ValueError(
                "attribute_transfer requires a non-empty transfer_target content-group ID."
            )
        if transfer_target == content_group:
            raise ValueError(
                "attribute_transfer requires transfer_target to differ from its source "
                "content_group."
            )
    return {
        "role": role,
        "retention": resolved_relation,
        "content_group": content_group,
        "transfer_target": transfer_target,
        "shot_scope": _clean_text(binding.get("shot_scope")),
        "notes": _clean_text(binding.get("notes")),
    }


def role_binding_entries(role_chain) -> list[dict]:
    """Validate and copy a repeatable role-binding chain."""

    if role_chain is None:
        return []
    if not isinstance(role_chain, dict) or role_chain.get("version") != ROLE_CHAIN_VERSION:
        raise ValueError("role_bindings must come from a MiniMax H3 Visual Reference Role node.")
    raw_bindings = role_chain.get("bindings")
    if not isinstance(raw_bindings, (list, tuple)) or not raw_bindings:
        raise ValueError("The visual role chain contains no role bindings.")
    bindings = [_validate_binding(binding) for binding in raw_bindings]
    if len({tuple(binding.items()) for binding in bindings}) != len(bindings):
        raise ValueError("The visual role chain contains an identical duplicate binding.")
    return bindings


# Kept as a public alias for early context-v2 callers.
role_bindings = role_binding_entries


def _validate_binding_set(bindings: list[dict], media_type: str) -> list[dict]:
    validated = [_validate_binding(binding, media_type) for binding in bindings]
    roles = {binding["role"] for binding in validated}
    if EDIT_ROLE in roles and CONTINUE_ROLE in roles:
        raise ValueError("One video cannot be both the source to edit and the source to continue.")
    endpoint_roles = roles & ENDPOINT_ROLES
    if endpoint_roles and len(validated) != 1:
        raise ValueError(
            "An exact first/last-frame binding cannot be mixed with Ref2VA roles on the same media. "
            "Use a separate Visual Reference node for each endpoint."
        )
    return validated


def _binding_preview(bindings: list[dict]) -> str:
    lines = []
    for number, binding in enumerate(bindings, start=1):
        details = [binding["role"], binding["retention"]]
        if binding["content_group"]:
            details.append(f"content group: {binding['content_group']}")
        if binding["transfer_target"]:
            details.append(f"transfer target: {binding['transfer_target']}")
        if binding["shot_scope"]:
            details.append(f"shots: {binding['shot_scope']}")
        if binding["notes"]:
            details.append(f"notes: {binding['notes']}")
        lines.append(f"Role {number} | " + " | ".join(details))
    return "\n".join(lines)


def _is_image_tensor(value) -> bool:
    return value is not None and hasattr(value, "shape") and len(value.shape) == 4


def _validate_analysis_tensor(value, name: str, expected_frames: int | None = None):
    if not _is_image_tensor(value):
        raise ValueError(f"{name} must be a ComfyUI IMAGE tensor with shape [frames, H, W, C].")
    if value.shape[0] < 1:
        raise ValueError(f"{name} contains no frames.")
    if expected_frames is not None and value.shape[0] != expected_frames:
        raise ValueError(
            f"{name} contains {value.shape[0]} frames but its timestamp list contains "
            f"{expected_frames}."
        )


def _validate_timestamps(values, name: str, expected_count: int) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) != expected_count:
        raise ValueError(f"{name} must contain one timestamp per analysis frame.")
    timestamps = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in timestamps):
        raise ValueError(f"{name} must contain finite, non-negative seconds.")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{name} must be strictly increasing.")
    return timestamps


def _resize_long_edge(images, long_edge: int):
    """Create a smaller RGB analysis copy on an approximately aspect-safe 32 px grid."""

    import torch.nn.functional as functional

    images = images[..., :3]
    height, width = images.shape[1:3]
    longest = max(height, width)
    if longest <= long_edge:
        return images
    scale = long_edge / longest
    target_height = max(32, round(height * scale / 32) * 32)
    target_width = max(32, round(width * scale / 32) * 32)
    channels_first = images.movedim(-1, 1)
    resized = functional.interpolate(
        channels_first,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.movedim(1, -1)


def _video_analysis_frames(frames, source_fps: float, analysis_fps: float, max_frames: int):
    """Sample configurable generic-Qwen evidence across the effective H3 clip."""

    import torch

    step = source_fps / analysis_fps
    indices = torch.arange(
        0,
        frames.shape[0],
        step,
        dtype=torch.float32,
        device=frames.device,
    ).round().long()
    indices = torch.unique(indices.clamp(max=frames.shape[0] - 1), sorted=True)
    if indices.numel() > max_frames:
        keep = torch.linspace(
            0,
            indices.numel() - 1,
            max_frames,
            device=frames.device,
        ).round().long()
        indices = indices[keep]
    timestamps = [float(index) / source_fps for index in indices.tolist()]
    return frames[indices], timestamps


def _minimax_video_analysis_frames(frames):
    """Mirror native H3's fixed 2 FPS Qwen presentation."""

    import torch

    indices = torch.arange(
        0,
        frames.shape[0],
        int(H3_VIDEO_FPS // 2),
        dtype=torch.long,
        device=frames.device,
    )
    timestamps = [int(index) / H3_VIDEO_FPS for index in indices.tolist()]
    return frames[indices], timestamps


def _resample_video_for_h3(frames, source_fps: float):
    """Return a nearest-neighbour 24 FPS frame batch for native H3."""

    import torch

    if math.isclose(source_fps, H3_VIDEO_FPS, rel_tol=0.0, abs_tol=0.0005):
        return frames
    duration = frames.shape[0] / source_fps
    target_count = max(1, round(duration * H3_VIDEO_FPS))
    source_positions = torch.arange(
        target_count,
        dtype=torch.float32,
        device=frames.device,
    )
    source_positions = (source_positions / H3_VIDEO_FPS * source_fps).round().long()
    source_positions = source_positions.clamp(max=frames.shape[0] - 1)
    return frames[source_positions]


def _align_target_frame_count(length: int) -> int:
    value = max(NATIVE_FRAME_OFFSET, int(length))
    while value % NATIVE_FRAME_MODULUS != NATIVE_FRAME_OFFSET:
        value += 1
    return value


def _align_reference_frame_count(length: int) -> int:
    if length < NATIVE_FRAME_OFFSET:
        raise ValueError("MiniMax H3 reference videos need at least 5 frames after resampling.")
    return length - ((length - NATIVE_FRAME_OFFSET) % NATIVE_FRAME_MODULUS)


def _align_reference_video(frames, h3_length: int | None):
    """Mirror native truncation followed by downward 17k+5 alignment."""

    target_frame_count = None
    if h3_length is not None:
        if isinstance(h3_length, bool) or int(h3_length) < NATIVE_FRAME_OFFSET:
            raise ValueError("h3_length must be at least 5 frames when connected.")
        if int(h3_length) > 3600:
            raise ValueError("h3_length cannot exceed the native H3 input maximum of 3600.")
        target_frame_count = _align_target_frame_count(int(h3_length))
        frames = frames[:target_frame_count]

    count = _align_reference_frame_count(frames.shape[0])
    return frames[:count], target_frame_count


def _route_family(bindings: list[dict]) -> str:
    return "endpoint" if bindings[0]["role"] in ENDPOINT_ROLES else "ref2va"


def _context_routing_mode(entries: list[dict]) -> str:
    if not entries:
        return "ref2va"
    families = {_route_family(entry["bindings"]) for entry in entries}
    if len(families) != 1:
        raise ValueError(
            "Exact first/last-frame media uses MiniMax H3 Image to Video and cannot share a "
            "reference_context chain with Ref2VA media. Build separate workflows for those modes."
        )
    mode = families.pop()
    if mode == "endpoint":
        roles = [entry["bindings"][0]["role"] for entry in entries]
        if any(entry["kind"] != "image" for entry in entries):
            raise ValueError("First/last endpoints must be pictures, not video frame batches.")
        for endpoint in ENDPOINT_ROLES:
            if roles.count(endpoint) > 1:
                raise ValueError(f"The endpoint chain contains more than one '{endpoint}' picture.")
        if len(entries) > 2:
            raise ValueError("Endpoint mode accepts at most one first frame and one last frame.")
    return mode


def _mode_hint(entries: list[dict], routing_mode: str) -> str:
    if routing_mode == "ref2va":
        return "Ref2VA"
    roles = {entry["bindings"][0]["role"] for entry in entries}
    if roles == {FIRST_FRAME_ROLE, LAST_FRAME_ROLE}:
        return "FL2VA"
    return "I2VA" if FIRST_FRAME_ROLE in roles else "L2VA"


def _native_order(entries: list[dict], routing_mode: str) -> list[dict]:
    if routing_mode == "endpoint":
        endpoint_order = {FIRST_FRAME_ROLE: 0, LAST_FRAME_ROLE: 1}
        return sorted(entries, key=lambda entry: endpoint_order[entry["bindings"][0]["role"]])
    return sorted(entries, key=lambda entry: entry["kind"] == "video")


def _normalized_entries(entries: list[dict]) -> tuple[list[dict], str, str]:
    """Assign native semantic labels and exact downstream socket routes."""

    routing_mode = _context_routing_mode(entries)
    hint = _mode_hint(entries, routing_mode)
    normalized = [dict(entry) for entry in entries]

    if routing_mode == "endpoint":
        ordered = _native_order(normalized, routing_mode)
        both = len(ordered) == 2
        for index, entry in enumerate(ordered, start=1):
            role = entry["bindings"][0]["role"]
            number = index if both else 1
            input_name = "first_frame" if role == FIRST_FRAME_ROLE else "last_frame"
            entry["label"] = f"<Picture {number}>"
            entry["h3_node"] = ENDPOINT_NODE_NAME
            entry["h3_node_id"] = ENDPOINT_NODE_ID
            entry["h3_input"] = input_name
            entry["route_family"] = routing_mode
            entry["routes"] = [
                {"node": ENDPOINT_NODE_NAME, "node_id": ENDPOINT_NODE_ID, "input": input_name}
            ]
    else:
        image_number = video_number = 0
        for entry in normalized:
            if entry["kind"] == "image":
                image_number += 1
                label = f"<Picture {image_number}>"
                input_name = f"ref_image_{image_number - 1}"
            else:
                video_number += 1
                label = f"<Video {video_number}>"
                input_name = f"ref_video_{video_number - 1}"
            entry["label"] = label
            entry["h3_node"] = REFERENCE_NODE_NAME
            entry["h3_node_id"] = REFERENCE_NODE_ID
            entry["h3_input"] = input_name
            entry["route_family"] = routing_mode
            entry["routes"] = [
                {"node": REFERENCE_NODE_NAME, "node_id": REFERENCE_NODE_ID, "input": input_name}
            ]
    return normalized, routing_mode, hint


def _validate_transfer_targets(entries: list[dict]) -> None:
    """Require every explicit transfer to name a visible destination group.

    This is intentionally a final-context check. A source may be added before its
    destination while the user is still building a Visual Reference chain.
    """

    visible_groups = {
        binding["content_group"]
        for entry in entries
        for binding in entry["bindings"]
        if binding["content_group"] and binding["retention"] != TRANSFER_RELATION
    }
    missing = []
    for entry in entries:
        for binding in entry["bindings"]:
            if (
                binding["retention"] == TRANSFER_RELATION
                and binding["transfer_target"] not in visible_groups
            ):
                missing.append(
                    f"{entry.get('label', 'reference')} role '{binding['role']}' targets "
                    f"content_group={binding['transfer_target']!r}"
                )
    if missing:
        raise ValueError(
            "attribute_transfer needs a visible non-transfer target binding in the final "
            "reference context: " + "; ".join(missing) + "."
        )


def _adapt_v1_entry(raw_entry: dict) -> dict:
    if not isinstance(raw_entry, dict) or raw_entry.get("kind") not in {"image", "video"}:
        raise ValueError("reference_context contains an invalid legacy visual-reference entry.")
    kind = raw_entry["kind"]
    media_type = PICTURE_MEDIA if kind == "image" else VIDEO_MEDIA
    analysis_media = raw_entry.get("analysis_media")
    _validate_analysis_tensor(analysis_media, "legacy analysis_media")
    if kind == "image" and analysis_media.shape[0] != 1:
        raise ValueError("A legacy picture entry must contain exactly one analysis image.")
    role = raw_entry.get("role", IDENTITY_ROLE)
    binding = _validate_binding(
        {
            "role": role,
            "retention": AUTO_RELATION,
            "content_group": "",
            "transfer_target": "",
            "notes": raw_entry.get("notes", ""),
        },
        media_type,
    )
    entry = dict(raw_entry)
    entry.update(
        {
            "schema_version": CONTEXT_VERSION,
            "media_type": media_type,
            "bindings": [binding],
            "role": binding["role"],
            "notes": _clean_text(raw_entry.get("notes")),
            "minimax_analysis_media": analysis_media,
            "legacy_context": True,
        }
    )
    if kind == "video":
        timestamps = _validate_timestamps(
            raw_entry.get("timestamps"), "legacy timestamps", analysis_media.shape[0]
        )
        duration = float(raw_entry.get("duration_seconds", 0.0))
        if not math.isfinite(duration) or not 2.0 <= duration <= 15.0:
            raise ValueError("A legacy reference video duration must remain within 2-15 seconds.")
        entry.update(
            {
                "timestamps": timestamps,
                "minimax_timestamps": timestamps,
                "source_duration_seconds": duration,
                "duration_seconds": duration,
                "native_frame_count": max(
                    NATIVE_FRAME_OFFSET, round(duration * H3_VIDEO_FPS)
                ),
                "resampled_frame_count": max(
                    NATIVE_FRAME_OFFSET, round(duration * H3_VIDEO_FPS)
                ),
                "native_was_trimmed": False,
            }
        )
    return entry


def _validate_context_limits(entries: list[dict]) -> None:
    image_count = sum(entry["kind"] == "image" for entry in entries)
    video_entries = [entry for entry in entries if entry["kind"] == "video"]
    if image_count > 9 or len(video_entries) > 3:
        raise ValueError("The reference context exceeds MiniMax H3 visual-reference count limits.")
    total_source_duration = sum(
        float(entry["source_duration_seconds"]) for entry in video_entries
    )
    if total_source_duration > 15.0005:
        raise ValueError("The reference context exceeds the 15-second total source-video limit.")


def _validate_v2_entry(raw_entry: dict) -> dict:
    if not isinstance(raw_entry, dict) or raw_entry.get("schema_version") != CONTEXT_VERSION:
        raise ValueError("Every context-v2 entry must declare schema_version=2.")
    kind = raw_entry.get("kind")
    if kind not in {"image", "video"}:
        raise ValueError("Every context-v2 entry needs kind=image or kind=video.")
    media_type = PICTURE_MEDIA if kind == "image" else VIDEO_MEDIA
    if raw_entry.get("media_type") != media_type:
        raise ValueError("A context-v2 entry has an inconsistent media_type.")
    raw_bindings = raw_entry.get("bindings")
    if not isinstance(raw_bindings, (list, tuple)) or not raw_bindings:
        raise ValueError("Every context-v2 entry needs at least one role binding.")
    bindings = _validate_binding_set(list(raw_bindings), media_type)
    entry = dict(raw_entry)
    entry["bindings"] = bindings
    if not isinstance(entry.get("legacy_context"), bool):
        raise ValueError("context-v2 legacy_context must be true or false.")
    if entry.get("role") != bindings[0]["role"]:
        raise ValueError("context-v2 role must mirror the first role binding for compatibility.")
    if not isinstance(entry.get("notes", ""), str):
        raise ValueError("context-v2 notes must be text.")

    analysis_media = entry.get("analysis_media")
    minimax_media = entry.get("minimax_analysis_media")
    _validate_analysis_tensor(analysis_media, "analysis_media")
    _validate_analysis_tensor(minimax_media, "minimax_analysis_media")
    if kind == "image":
        if analysis_media.shape[0] != 1 or minimax_media.shape[0] != 1:
            raise ValueError("A context-v2 picture must contain one generic and one MiniMax image.")
    else:
        entry["timestamps"] = _validate_timestamps(
            entry.get("timestamps"), "timestamps", analysis_media.shape[0]
        )
        entry["minimax_timestamps"] = _validate_timestamps(
            entry.get("minimax_timestamps"),
            "minimax_timestamps",
            minimax_media.shape[0],
        )
        source_duration = float(entry.get("source_duration_seconds", 0.0))
        if not math.isfinite(source_duration) or not 2.0 <= source_duration <= 15.0:
            raise ValueError("context-v2 source video duration must remain within 2-15 seconds.")
        frame_count = entry.get("native_frame_count")
        if not isinstance(frame_count, int) or frame_count < NATIVE_FRAME_OFFSET:
            raise ValueError("context-v2 video needs a valid native_frame_count.")
        duration = float(entry.get("duration_seconds", 0.0))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("context-v2 video duration must be positive and finite.")
        if not entry["legacy_context"]:
            if len(entry["timestamps"]) > 32 or not math.isclose(
                entry["timestamps"][0], 0.0, abs_tol=0.0005
            ):
                raise ValueError(
                    "context-v2 generic video evidence must start at 0 seconds and contain "
                    "at most 32 samples."
                )
            expected_count = math.ceil(frame_count / (H3_VIDEO_FPS / 2.0))
            if minimax_media.shape[0] != expected_count:
                raise ValueError(
                    "context-v2 MiniMax video evidence must contain every fixed 2-FPS sample."
                )
            expected_minimax = [index / 2.0 for index in range(expected_count)]
            if any(
                not math.isclose(actual, expected, abs_tol=0.0005)
                for actual, expected in zip(entry["minimax_timestamps"], expected_minimax)
            ):
                raise ValueError(
                    "context-v2 MiniMax video evidence must be sampled at fixed 2 FPS."
                )
            if frame_count % NATIVE_FRAME_MODULUS != NATIVE_FRAME_OFFSET:
                raise ValueError("context-v2 native_frame_count must be on the 17k+5 grid.")
            if not math.isclose(duration, frame_count / H3_VIDEO_FPS, abs_tol=0.0005):
                raise ValueError(
                    "context-v2 video duration must match native_frame_count at 24 FPS."
                )
            if any(
                timestamp > (frame_count - 1) / H3_VIDEO_FPS + 0.0005
                for timestamp in [*entry["timestamps"], *entry["minimax_timestamps"]]
            ):
                raise ValueError("context-v2 analysis timestamps exceed the effective video.")
            source_fps = float(entry.get("source_fps", 0.0))
            if not math.isfinite(source_fps) or source_fps <= 0:
                raise ValueError("context-v2 source_fps must be positive and finite.")
            resampled_count = entry.get("resampled_frame_count")
            if not isinstance(resampled_count, int) or resampled_count < frame_count:
                raise ValueError(
                    "context-v2 resampled_frame_count must cover native_frame_count."
                )
            if resampled_count != max(1, round(source_duration * H3_VIDEO_FPS)):
                raise ValueError(
                    "context-v2 resampled_frame_count does not match source duration at 24 FPS."
                )
            if entry.get("native_was_trimmed") is not (frame_count < resampled_count):
                raise ValueError(
                    "context-v2 native_was_trimmed does not match its frame counts."
                )
            target_count = entry.get("target_frame_count")
            if target_count is not None:
                if (
                    not isinstance(target_count, int)
                    or target_count < NATIVE_FRAME_OFFSET
                    or target_count > _align_target_frame_count(3600)
                    or target_count % NATIVE_FRAME_MODULUS != NATIVE_FRAME_OFFSET
                    or frame_count > target_count
                ):
                    raise ValueError("context-v2 target_frame_count is not a valid H3 length.")
            capped_count = min(resampled_count, target_count or resampled_count)
            if frame_count != _align_reference_frame_count(capped_count):
                raise ValueError(
                    "context-v2 native_frame_count does not match native truncation/alignment."
                )
    return entry


def reference_entries(reference_context, *, validate_transfers: bool = True) -> list[dict]:
    """Adapt legacy v1 payloads and strictly validate native-aware context v2."""

    if reference_context is None:
        return []
    if not isinstance(reference_context, dict) or not isinstance(
        reference_context.get("entries"), (list, tuple)
    ):
        raise ValueError(
            "reference_context must come from a MiniMax H3 Enhancer Visual Reference node."
        )
    version = reference_context.get("version", 1)
    if version == 1:
        entries = [_adapt_v1_entry(entry) for entry in reference_context["entries"]]
        normalized, _, _ = _normalized_entries(entries)
        _validate_context_limits(normalized)
        if validate_transfers:
            _validate_transfer_targets(normalized)
        return normalized
    if version != CONTEXT_VERSION:
        raise ValueError(f"Unsupported reference_context version: {version!r}.")

    entries = [_validate_v2_entry(entry) for entry in reference_context["entries"]]
    normalized, routing_mode, hint = _normalized_entries(entries)
    if reference_context.get("routing_mode") != routing_mode:
        raise ValueError("context-v2 routing_mode does not match its role bindings.")
    if reference_context.get("mode_hint") != hint:
        raise ValueError("context-v2 mode_hint does not match its endpoint/reference roles.")
    for supplied, expected in zip(entries, normalized):
        for key in (
            "label",
            "h3_node",
            "h3_node_id",
            "h3_input",
            "route_family",
            "routes",
        ):
            if supplied.get(key) != expected.get(key):
                raise ValueError(f"context-v2 contains stale or invalid routing data for {key}.")
    _validate_context_limits(normalized)
    if validate_transfers:
        _validate_transfer_targets(normalized)
    return normalized


def _label_policy(entry: dict, binding: dict) -> str:
    role = binding["role"]
    label = entry["label"]
    if role in ENDPOINT_ROLES:
        return f"track {label} directly as the exact native {entry['h3_input']} endpoint"
    if role in SUBJECT_ROLES:
        if binding["retention"] == TRANSFER_RELATION:
            return (
                f"derive only the '{binding['content_group']}' attribute from {label} and apply "
                f"it to the separately defined '{binding['transfer_target']}' <Subject N>; "
                "do not replace the target's identity or unrelated attributes"
            )
        group = (
            f" and merge it into content group '{binding['content_group']}'"
            if binding["content_group"]
            else ""
        )
        return (
            f"derive the requested reusable visible content as <Subject N> citing {label}{group}; "
            f"do not also define {label} unless another binding gives it a direct frame/video role"
        )
    if entry["kind"] == "image":
        return f"track {label} directly as a concrete frame, storyboard, or composition anchor"
    return f"track {label} directly as an editing, continuation, or temporal-structure video"


def reference_inventory(entries: list[dict]) -> str:
    """Describe every role binding and the exact visual evidence Qwen receives."""

    if not entries:
        return "No chained visual references are attached."
    _validate_transfer_targets(entries)
    routing_mode = _context_routing_mode(entries)
    lines = [
        f"Context route={routing_mode}; mode hint={_mode_hint(entries, routing_mode)}."
    ]
    for entry in _native_order(entries, routing_mode):
        route = f"{entry['h3_node']}.{entry['h3_input']}"
        if entry["kind"] == "video":
            heading = (
                f"{entry['label']}: route={route}; source duration="
                f"{entry['source_duration_seconds']:.3f}s; native effective duration="
                f"{entry['duration_seconds']:.3f}s/{entry['native_frame_count']} frames; "
                f"generic samples={len(entry['timestamps'])}; MiniMax samples="
                f"{len(entry['minimax_timestamps'])}."
            )
        else:
            heading = f"{entry['label']}: route={route}."
        lines.append(heading)
        for number, binding in enumerate(entry["bindings"], start=1):
            details = [f"role={binding['role']}", f"retention={binding['retention']}"]
            if binding["content_group"]:
                details.append(f"content_group={binding['content_group']}")
            if binding["transfer_target"]:
                details.append(f"transfer_target={binding['transfer_target']}")
            if binding["shot_scope"]:
                details.append(f"shot_scope={binding['shot_scope']}")
            if binding["notes"]:
                details.append(f"notes={binding['notes']}")
            details.append(f"H3 label policy={_label_policy(entry, binding)}")
            lines.append(f"  Binding {number}: " + "; ".join(details) + ".")
        entry_notes = _clean_text(entry.get("notes"))
        if entry_notes and not any(entry_notes == binding["notes"] for binding in entry["bindings"]):
            lines.append(f"  Media notes: {entry_notes}.")
        if entry.get("legacy_context"):
            lines.append("  Legacy v1 evidence: MiniMax samples could not be reconstructed at 2 FPS.")
    return "\n".join(lines)


def _routing_report(entries: list[dict], routing_mode: str, hint: str) -> str:
    lines = [f"Visual-reference routing: {hint} via {routing_mode}."]
    for entry in _native_order(entries, routing_mode):
        roles = ", ".join(binding["role"] for binding in entry["bindings"])
        line = (
            f"- {entry['label']} ({roles}) -> {entry['h3_node']}.{entry['h3_input']}"
        )
        if entry["kind"] == "video":
            line += (
                f"; {entry['native_frame_count']} effective 24-FPS frames/"
                f"{entry['duration_seconds']:.3f}s"
            )
            if entry["native_was_trimmed"]:
                line += f" (trimmed from {entry['resampled_frame_count']} resampled frames)"
        lines.append(line)
    if routing_mode == "endpoint":
        lines.append(
            "Connect h3_media outputs to the listed first_frame/last_frame inputs on MiniMax H3 "
            "Image to Video. Do not connect these pictures to Reference to Video."
        )
    else:
        lines.append(
            "Connect every h3_media output separately to the listed native Reference to Video "
            "autogrow input."
        )
    lines.append(
        "Fan out only the final reference_context output to Prompt Guide.reference_context and "
        "Prompt Enhancer.reference_context."
    )
    return "\n".join(lines)


class MiniMaxH3VisualReferenceRole:
    """Build repeatable semantic bindings for one picture or video."""

    CATEGORY = "MiniMax H3/Prompting"
    FUNCTION = "append_role"
    RETURN_TYPES = (ROLE_CHAIN_TYPE, "STRING")
    RETURN_NAMES = ("role_bindings", "role_preview")
    OUTPUT_TOOLTIPS = (
        "Connect to another Visual Reference Role.previous_roles input, or connect the final chain to a Visual Reference.role_bindings input. The role chain belongs to one media asset; the Visual Reference node adds it to the ordered context used by the Guide and Enhancer.",
        "Readable preview of every role, retention relation, content group, shot scope, and note for this media.",
    )
    DESCRIPTION = (
        "Assigns one semantic use to a visual asset. Chain roles when the same picture or video "
        "supplies more than one independently constrained piece of evidence."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_role": (
                    REFERENCE_ROLES,
                    {
                        "default": UNASSIGNED_ROLE,
                        "tooltip": (
                            "Choose the exact job this media performs; the unassigned value intentionally "
                            "stops execution instead of inventing a Subject or endpoint. Endpoint roles "
                            "route to Image to Video; reusable/direct reference roles route to Ref2VA."
                        ),
                    },
                ),
                "retention": (
                    VISUAL_RELATIONS,
                    {
                        "default": AUTO_RELATION,
                        "tooltip": (
                            "How strongly H3 should retain this role. Auto is conservative: exact "
                            "endpoints and motion are fully_preserved, while style/storyboard/camera "
                            "are weak_reference. Select attribute_transfer explicitly and identify "
                            "both its source and destination groups. Bindings merged into the same "
                            "content_group must resolve to one shared retention marker because H3 "
                            "allows one retention row per tracked Subject."
                        ),
                    },
                ),
                "content_group": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Example: hero, red car, coastal environment",
                        "tooltip": (
                            "Optional reusable-content key. Give related roles on different assets the same "
                            "value when they jointly define one <Subject N>. Those merged bindings must "
                            "also use the same retention marker. Direct frame/video roles cannot use it."
                        ),
                    },
                ),
                "transfer_target": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Example: hero (destination group)",
                        "tooltip": (
                            "Only for attribute_transfer: name the different content_group that "
                            "receives this source attribute. The final context must also contain a "
                            "visible, non-transfer binding with that destination group."
                        ),
                    },
                ),
                "shot_scope": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Example: all shots, Shot 2 only, Shots 1-3",
                        "tooltip": "Optional shots where this binding applies. Empty means wherever relevant.",
                    },
                ),
                "notes": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": "",
                        "placeholder": "Example: copy only the orbit; ignore the actor and background.",
                        "tooltip": "Binding-specific instructions about what to preserve, transfer, ignore, or change.",
                    },
                ),
            },
            "optional": {
                "previous_roles": (
                    ROLE_CHAIN_TYPE,
                    {
                        "tooltip": (
                            "Connect role_bindings from the preceding role node. Leave disconnected for "
                            "the first role assigned to this media."
                        )
                    },
                )
            },
        }

    def append_role(
        self,
        reference_role: str,
        retention: str,
        content_group: str,
        transfer_target: str,
        shot_scope: str,
        notes: str,
        previous_roles=None,
    ):
        previous = role_binding_entries(previous_roles) if previous_roles is not None else []
        candidate = _validate_binding(
            {
                "role": reference_role,
                "retention": retention,
                "content_group": content_group,
                "transfer_target": transfer_target,
                "shot_scope": shot_scope,
                "notes": notes,
            }
        )
        bindings = [*previous, candidate]
        if len({tuple(binding.items()) for binding in bindings}) != len(bindings):
            raise ValueError("The visual role chain contains an identical duplicate binding.")
        return (
            {"version": ROLE_CHAIN_VERSION, "bindings": bindings},
            _binding_preview(bindings),
        )


class MiniMaxH3EnhancerVisualReference:
    """Add one picture or video to context and pass native-ready media through."""

    CATEGORY = "MiniMax H3/Prompting"
    FUNCTION = "add_reference"
    RETURN_TYPES = (REFERENCE_CONTEXT_TYPE, "IMAGE", "STRING")
    RETURN_NAMES = ("reference_context", "h3_media", "routing_report")
    OUTPUT_TOOLTIPS = (
        "Chain into the next Visual Reference. Fan the final context out to both Prompt Guide.reference_context and Prompt Enhancer.reference_context so their labels, roles, grouping, retention, and mode remain identical.",
        "Original endpoint/reference picture, or video resampled and aligned to native H3's 24-FPS 17k+5 contract. Follow routing_report for the exact destination socket.",
        "Exact semantic labels, mode family, native input routes, and effective video duration after native alignment/truncation.",
    )
    DESCRIPTION = (
        "Adds one picture or reference-video frame batch to Qwen's prompt-writing context while "
        "passing native-ready H3 media through. Chain Visual Reference nodes to order media; use "
        "Visual Reference Role nodes when one asset has multiple roles."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "media": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "For Picture, connect exactly one image. For Video frames, connect the full "
                            "batch and set its real FPS; the node creates separate generic and native "
                            "MiniMax analysis samples plus an H3-ready 24-FPS pass-through."
                        )
                    },
                ),
                "media_type": (
                    MEDIA_TYPES,
                    {
                        "default": PICTURE_MEDIA,
                        "tooltip": (
                            "Picture can be an exact first/last endpoint or a Ref2VA picture. Video frames "
                            "is always Ref2VA. Labels and socket routes are assigned from the role bindings."
                        ),
                    },
                ),
                "reference_role": (
                    REFERENCE_ROLES,
                    {
                        "default": UNASSIGNED_ROLE,
                        "tooltip": (
                            "Single-role compatibility setting. Choose a role explicitly, or connect "
                            "role_bindings below to replace this value with a repeatable role chain. The "
                            "unassigned default stops execution and never invents a Subject or endpoint."
                        ),
                    },
                ),
                "notes": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": "",
                        "placeholder": "Optional: preserve her red jacket; copy only the orbit, not the actor.",
                        "tooltip": (
                            "For the single-role path, these are role notes. With role_bindings connected, "
                            "this becomes a media-wide note while each role keeps its own notes."
                        ),
                    },
                ),
                "source_fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 240.0,
                        "step": 0.001,
                        "tooltip": (
                            "Video only: actual FPS represented by the input batch. h3_media is resampled "
                            "to native H3's fixed 24 FPS. Ignored for Picture."
                        ),
                    },
                ),
                "analysis_fps": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.25,
                        "max": 2.0,
                        "step": 0.25,
                        "tooltip": (
                            "Video only: configurable evidence rate for a complete generic Qwen. MiniMax's "
                            "native tokenizer receives a separate fixed 2-FPS sample regardless of this value."
                        ),
                    },
                ),
                "max_analysis_frames": (
                    "INT",
                    {
                        "default": 16,
                        "min": 2,
                        "max": 32,
                        "step": 1,
                        "tooltip": (
                            "Video only: cap for generic-Qwen evidence. Native MiniMax evidence remains "
                            "fixed at 2 FPS, up to 30 samples for a 15-second source."
                        ),
                    },
                ),
                "analysis_long_edge": (
                    "INT",
                    {
                        "default": 768,
                        "min": 256,
                        "max": 1536,
                        "step": 32,
                        "tooltip": (
                            "Maximum long edge of both private Qwen analysis copies. Aspect ratio is "
                            "kept approximately after 32-pixel size rounding, and small media is never "
                            "enlarged; H3 routing uses a separate copy."
                        ),
                    },
                ),
            },
            "optional": {
                "previous_context": (
                    REFERENCE_CONTEXT_TYPE,
                    {
                        "tooltip": (
                            "Connect reference_context from the preceding Visual Reference node. This "
                            "chains media entries; it is separate from the per-media role chain."
                        )
                    },
                ),
                "role_bindings": (
                    ROLE_CHAIN_TYPE,
                    {
                        "tooltip": (
                            "Optional: connect the final Visual Reference Role chain for this media. It "
                            "replaces reference_role and supplies repeatable role-specific metadata."
                        )
                    },
                ),
                "h3_length": (
                    "INT",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "Recommended for videos: connect MiniMax H3 Target Timing.h3_length. Native "
                            "H3 first truncates each reference video to this target length, then aligns it "
                            "downward to 17k+5; this node mirrors that behavior before Qwen analysis. Do "
                            "not feed Prompt Guide.h3_length backward when this reference_context also "
                            "feeds the Guide, because that would create a graph cycle."
                        ),
                    },
                ),
            },
        }

    def add_reference(
        self,
        media,
        media_type: str,
        reference_role: str,
        notes: str,
        source_fps: float,
        analysis_fps: float,
        max_analysis_frames: int,
        analysis_long_edge: int,
        previous_context=None,
        role_bindings=None,
        h3_length=None,
    ):
        if not _is_image_tensor(media):
            raise ValueError("media must be a ComfyUI IMAGE tensor with shape [frames, H, W, C].")
        if media.shape[0] < 1:
            raise ValueError("media contains no image or video frames.")
        if media.shape[-1] < 3:
            raise ValueError("media must contain at least RGB channels in its final dimension.")
        if (
            isinstance(analysis_long_edge, bool)
            or not 256 <= int(analysis_long_edge) <= 1536
        ):
            raise ValueError("analysis_long_edge must be an integer from 256 through 1536.")

        entries = (
            reference_entries(previous_context, validate_transfers=False)
            if previous_context is not None
            else []
        )
        clean_notes = _clean_text(notes)
        if role_bindings is None:
            bindings = [
                _validate_binding(
                    {
                        "role": reference_role,
                        "retention": AUTO_RELATION,
                        "notes": clean_notes,
                    },
                    media_type,
                )
            ]
        else:
            bindings = _validate_binding_set(role_binding_entries(role_bindings), media_type)

        if media_type == PICTURE_MEDIA:
            if media.shape[0] != 1:
                raise ValueError(
                    "Picture expects exactly one image. Use one Visual Reference node per picture, "
                    "or select Video frames for a temporal IMAGE batch."
                )
            if sum(entry["kind"] == "image" for entry in entries) >= 9:
                raise ValueError("MiniMax H3 accepts at most 9 reference pictures.")
            analysis_media = _resize_long_edge(media[:1], int(analysis_long_edge))
            h3_media = media[:1]
            entry = {
                "schema_version": CONTEXT_VERSION,
                "kind": "image",
                "media_type": PICTURE_MEDIA,
                "bindings": bindings,
                "role": bindings[0]["role"],
                "notes": clean_notes,
                "analysis_media": analysis_media,
                "minimax_analysis_media": analysis_media,
                "legacy_context": False,
            }
        elif media_type == VIDEO_MEDIA:
            source_fps = float(source_fps)
            analysis_fps = float(analysis_fps)
            if not math.isfinite(source_fps) or source_fps <= 0:
                raise ValueError("source_fps must be a positive finite value.")
            if not math.isfinite(analysis_fps) or analysis_fps <= 0 or analysis_fps > 2.0:
                raise ValueError("analysis_fps must be a finite value greater than 0 and at most 2.")
            if (
                isinstance(max_analysis_frames, bool)
                or not 2 <= int(max_analysis_frames) <= 32
            ):
                raise ValueError("max_analysis_frames must be an integer from 2 through 32.")
            source_duration = media.shape[0] / source_fps
            if source_duration < 2.0 or source_duration > 15.0:
                raise ValueError(
                    f"Reference video duration is {source_duration:.3f}s; MiniMax H3 expects "
                    "2-15 seconds. Trim the frame batch or correct source_fps."
                )
            if sum(entry["kind"] == "video" for entry in entries) >= 3:
                raise ValueError("MiniMax H3 accepts at most 3 reference videos.")
            total_source_duration = source_duration + sum(
                entry.get("source_duration_seconds", entry.get("duration_seconds", 0.0))
                for entry in entries
                if entry["kind"] == "video"
            )
            if total_source_duration > 15.0005:
                raise ValueError(
                    f"The reference-video chain totals {total_source_duration:.3f}s; MiniMax H3 "
                    "accepts up to 15 seconds of source video in total. Trim one or more clips."
                )

            resampled = _resample_video_for_h3(media, source_fps)
            resampled_count = resampled.shape[0]
            h3_media, target_frame_count = _align_reference_video(resampled, h3_length)
            sampled, timestamps = _video_analysis_frames(
                h3_media[..., :3], H3_VIDEO_FPS, analysis_fps, int(max_analysis_frames)
            )
            minimax_sampled, minimax_timestamps = _minimax_video_analysis_frames(
                h3_media[..., :3]
            )
            entry = {
                "schema_version": CONTEXT_VERSION,
                "kind": "video",
                "media_type": VIDEO_MEDIA,
                "bindings": bindings,
                "role": bindings[0]["role"],
                "notes": clean_notes,
                "analysis_media": _resize_long_edge(sampled, int(analysis_long_edge)),
                "timestamps": timestamps,
                "minimax_analysis_media": _resize_long_edge(
                    minimax_sampled, int(analysis_long_edge)
                ),
                "minimax_timestamps": minimax_timestamps,
                "source_duration_seconds": source_duration,
                "duration_seconds": h3_media.shape[0] / H3_VIDEO_FPS,
                "source_fps": source_fps,
                "native_frame_count": h3_media.shape[0],
                "resampled_frame_count": resampled_count,
                "target_frame_count": target_frame_count,
                "native_was_trimmed": h3_media.shape[0] < resampled_count,
                "legacy_context": False,
            }
        else:
            raise ValueError(f"Unsupported media_type: {media_type}")

        updated_entries, routing_mode, hint = _normalized_entries([*entries, entry])
        context = {
            "version": CONTEXT_VERSION,
            "routing_mode": routing_mode,
            "mode_hint": hint,
            "entries": updated_entries,
        }
        return (context, h3_media, _routing_report(updated_entries, routing_mode, hint))


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3VisualReferenceRole": MiniMaxH3VisualReferenceRole,
    "MiniMaxH3EnhancerVisualReference": MiniMaxH3EnhancerVisualReference,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3VisualReferenceRole": "MiniMax H3 Visual Reference Role",
    "MiniMaxH3EnhancerVisualReference": "MiniMax H3 Enhancer Visual Reference",
}
