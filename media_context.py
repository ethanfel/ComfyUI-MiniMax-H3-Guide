"""Chainable visual-reference context for MiniMax H3 prompt enhancement."""

from __future__ import annotations

import math


REFERENCE_CONTEXT_TYPE = "MINIMAX_H3_ENHANCER_REFERENCE_CONTEXT"

PICTURE_MEDIA = "Picture"
VIDEO_MEDIA = "Video frames"
MEDIA_TYPES = [PICTURE_MEDIA, VIDEO_MEDIA]

IDENTITY_ROLE = "Identity or appearance"
ITEM_ROLE = "Object, prop, clothing, interface, or effect"
SCENE_ROLE = "Scene or environment"
STYLE_ROLE = "Visual style"
KEYFRAME_ROLE = "Storyboard or keyframe"
MOTION_ROLE = "Motion or action"
CAMERA_ROLE = "Camera, cuts, or rhythm"
EDIT_ROLE = "Source video to edit"
CONTINUE_ROLE = "Source video to continue"
REFERENCE_ROLES = [
    IDENTITY_ROLE,
    ITEM_ROLE,
    SCENE_ROLE,
    STYLE_ROLE,
    KEYFRAME_ROLE,
    MOTION_ROLE,
    CAMERA_ROLE,
    EDIT_ROLE,
    CONTINUE_ROLE,
]

H3_VIDEO_FPS = 24.0


def _label_policy(entry: dict) -> str:
    """Explain which H3 semantic label the declared asset role normally needs."""

    role = entry["role"]
    label = entry["label"]
    if role in {IDENTITY_ROLE, ITEM_ROLE, SCENE_ROLE, STYLE_ROLE, MOTION_ROLE}:
        return (
            f"derive only the reusable visible content as one or more <Subject N> items citing "
            f"{label} as their source; do not also define {label} unless the asset has a separate "
            "frame or whole-video role"
        )
    if entry["kind"] == "image":
        return (
            f"track {label} directly as a concrete frame, storyboard, or composition anchor; "
            "do not invent a <Subject N> unless reusable visible content is separately requested"
        )
    return (
        f"track {label} directly as a whole-video editing, continuation, or temporal-structure "
        "reference; do not invent a <Subject N> unless reusable visible content is separately requested"
    )


def reference_entries(reference_context) -> list[dict]:
    """Validate and copy the lightweight chain payload."""

    if reference_context is None:
        return []
    if not isinstance(reference_context, dict) or not isinstance(
        reference_context.get("entries"), (list, tuple)
    ):
        raise ValueError(
            "reference_context must come from a MiniMax H3 Enhancer Visual Reference node."
        )
    entries = []
    for raw_entry in reference_context["entries"]:
        if not isinstance(raw_entry, dict) or raw_entry.get("kind") not in {"image", "video"}:
            raise ValueError("reference_context contains an invalid visual-reference entry.")
        if not raw_entry.get("label") or raw_entry.get("analysis_media") is None:
            raise ValueError("Every visual-reference entry needs a label and analysis media.")
        entries.append(dict(raw_entry))
    return entries


def reference_inventory(entries: list[dict]) -> str:
    """Describe the labeled visual evidence Qwen receives."""

    if not entries:
        return "No chained visual references are attached."
    lines = []
    ordered_entries = sorted(entries, key=lambda entry: entry["kind"] == "video")
    for entry in ordered_entries:
        detail = f"role={entry['role']}"
        if entry["kind"] == "video":
            detail += (
                f"; duration={entry['duration_seconds']:.3f}s; "
                f"Qwen samples={len(entry['timestamps'])}"
            )
        if entry.get("notes"):
            detail += f"; user notes={entry['notes']}"
        detail += f"; H3 label policy={_label_policy(entry)}"
        lines.append(f"{entry['label']}: {detail}.")
    return "\n".join(lines)


def _resize_long_edge(images, long_edge: int):
    """Create a smaller RGB analysis copy while preserving aspect ratio."""

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
    """Uniformly sample a visual timeline without changing the H3 pass-through."""

    import torch

    step = source_fps / analysis_fps
    indices = torch.arange(0, frames.shape[0], step, dtype=torch.float64).round().long()
    indices = torch.unique(indices.clamp(max=frames.shape[0] - 1), sorted=True)
    if indices.numel() > max_frames:
        keep = torch.linspace(0, indices.numel() - 1, max_frames).round().long()
        indices = indices[keep]
    timestamps = [float(index) / source_fps for index in indices.tolist()]
    return frames[indices], timestamps


def _resample_video_for_h3(frames, source_fps: float):
    """Return a 24 FPS frame batch for the native H3 reference-video input."""

    import torch

    if math.isclose(source_fps, H3_VIDEO_FPS, rel_tol=0.0, abs_tol=0.0005):
        return frames
    duration = frames.shape[0] / source_fps
    target_count = max(1, round(duration * H3_VIDEO_FPS))
    source_positions = torch.arange(target_count, dtype=torch.float64)
    source_positions = (source_positions / H3_VIDEO_FPS * source_fps).round().long()
    source_positions = source_positions.clamp(max=frames.shape[0] - 1)
    return frames[source_positions]


def _routing_report(entries: list[dict]) -> str:
    lines = ["Visual-reference chain routing:"]
    lines.extend(
        f"- {entry['label']} ({entry['role']}) -> MiniMax H3 Reference to Video.{entry['h3_input']}"
        for entry in entries
    )
    lines.append(
        "Connect every h3_media output separately to the listed native H3 input; connect only "
        "the final reference_context output to Prompt Enhancer.reference_context."
    )
    return "\n".join(lines)


class MiniMaxH3EnhancerVisualReference:
    """Add one picture or video to the enhancer context and pass it through for H3."""

    CATEGORY = "MiniMax H3/Prompting"
    FUNCTION = "add_reference"
    RETURN_TYPES = (REFERENCE_CONTEXT_TYPE, "IMAGE", "STRING")
    RETURN_NAMES = ("reference_context", "h3_media", "routing_report")
    OUTPUT_TOOLTIPS = (
        "Chain into the next visual-reference node, or connect the final node to Prompt Enhancer.reference_context.",
        "Original picture, or video resampled to the 24 FPS expected by H3. Connect this output to the native ref_image_N or ref_video_N socket named in routing_report.",
        "Exact <Picture N>/<Video N> numbering and corresponding native H3 autogrow input names.",
    )
    DESCRIPTION = (
        "Adds one picture or reference-video frame batch to Qwen's prompt-writing context while "
        "passing H3-ready media through separately. Chain nodes to keep labels and routing ordered."
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
                            "frame batch from a video loader; the node makes a small analysis sample but "
                            "keeps an H3-ready pass-through."
                        )
                    },
                ),
                "media_type": (
                    MEDIA_TYPES,
                    {
                        "default": PICTURE_MEDIA,
                        "tooltip": (
                            "Picture assigns the next <Picture N> label and routes to ref_image_N. "
                            "Video frames assigns <Video N>, analyzes timestamped samples, and routes to ref_video_N."
                        ),
                    },
                ),
                "reference_role": (
                    REFERENCE_ROLES,
                    {
                        "default": IDENTITY_ROLE,
                        "tooltip": (
                            "Tell Qwen what it may learn from this reference. This role is evidence for "
                            "rewriting the prompt; it does not silently turn a picture into a first/last frame. "
                            "In H3 vocabulary, an object, prop, environment, style, or action reused as "
                            "visible content is a <Subject N>; Subject does not mean person only."
                        ),
                    },
                ),
                "notes": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": "",
                        "placeholder": "Optional: preserve her red jacket; copy only the camera orbit, not the actor.",
                        "tooltip": (
                            "Optional constraints about what to keep, transfer, ignore, or change. Qwen "
                            "receives these notes beside the automatically assigned reference label."
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
                            "Video only: actual FPS represented by the input frame batch. The h3_media "
                            "output is resampled to 24 FPS when necessary. Ignored for Picture."
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
                            "Video only: frames per second shown to Qwen. Use 1 FPS for scene/style/edit "
                            "analysis and 2 FPS for fast motion. H3 still receives the 24 FPS pass-through."
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
                            "Video only: maximum timestamped frames shown to Qwen. Frames are selected "
                            "across the full clip instead of truncating its ending."
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
                            "Maximum long edge of Qwen's private analysis copy. Aspect ratio is preserved "
                            "and small media is never enlarged. This does not resize the H3 pass-through; "
                            "768 is balanced, while 1024 helps faces or small visible text."
                        ),
                    },
                ),
            },
            "optional": {
                "previous_context": (
                    REFERENCE_CONTEXT_TYPE,
                    {
                        "tooltip": (
                            "Connect reference_context from the preceding visual-reference node. Leave "
                            "disconnected only for the first reference in the chain."
                        )
                    },
                )
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
    ):
        if media is None or not hasattr(media, "shape") or len(media.shape) != 4:
            raise ValueError("media must be a ComfyUI IMAGE tensor with shape [frames, H, W, C].")
        if media.shape[0] < 1:
            raise ValueError("media contains no image or video frames.")

        entries = reference_entries(previous_context) if previous_context is not None else []
        clean_notes = " ".join((notes or "").strip().split())
        if media_type == PICTURE_MEDIA:
            if media.shape[0] != 1:
                raise ValueError(
                    "Picture expects exactly one image. Use one visual-reference node per picture, "
                    "or select Video frames for a temporal IMAGE batch."
                )
            if reference_role in {EDIT_ROLE, CONTINUE_ROLE}:
                raise ValueError(f"'{reference_role}' requires media_type=Video frames.")
            number = sum(entry["kind"] == "image" for entry in entries) + 1
            if number > 9:
                raise ValueError("MiniMax H3 accepts at most 9 reference pictures.")
            analysis_media = _resize_long_edge(media[:1], analysis_long_edge)
            h3_media = media[:1]
            entry = {
                "kind": "image",
                "label": f"<Picture {number}>",
                "h3_input": f"ref_image_{number - 1}",
                "role": reference_role,
                "notes": clean_notes,
                "analysis_media": analysis_media,
            }
        elif media_type == VIDEO_MEDIA:
            source_fps = float(source_fps)
            analysis_fps = float(analysis_fps)
            if not math.isfinite(source_fps) or source_fps <= 0:
                raise ValueError("source_fps must be a positive finite value.")
            duration = media.shape[0] / source_fps
            if duration < 2.0 or duration > 15.0:
                raise ValueError(
                    f"Reference video duration is {duration:.3f}s; MiniMax H3 expects 2-15 seconds. "
                    "Trim the frame batch or correct source_fps."
                )
            number = sum(entry["kind"] == "video" for entry in entries) + 1
            if number > 3:
                raise ValueError("MiniMax H3 accepts at most 3 reference videos.")
            total_video_duration = duration + sum(
                entry.get("duration_seconds", 0.0)
                for entry in entries
                if entry["kind"] == "video"
            )
            if total_video_duration > 15.0005:
                raise ValueError(
                    f"The reference-video chain totals {total_video_duration:.3f}s; MiniMax H3 "
                    "accepts up to 15 seconds of reference video in total. Trim one or more clips."
                )
            sampled, timestamps = _video_analysis_frames(
                media[..., :3], source_fps, analysis_fps, int(max_analysis_frames)
            )
            analysis_media = _resize_long_edge(sampled, analysis_long_edge)
            h3_media = _resample_video_for_h3(media, source_fps)
            entry = {
                "kind": "video",
                "label": f"<Video {number}>",
                "h3_input": f"ref_video_{number - 1}",
                "role": reference_role,
                "notes": clean_notes,
                "analysis_media": analysis_media,
                "timestamps": timestamps,
                "duration_seconds": duration,
                "source_fps": source_fps,
            }
        else:
            raise ValueError(f"Unsupported media_type: {media_type}")

        updated_entries = [*entries, entry]
        context = {"version": 1, "entries": updated_entries}
        return (context, h3_media, _routing_report(updated_entries))


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3EnhancerVisualReference": MiniMaxH3EnhancerVisualReference,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3EnhancerVisualReference": "MiniMax H3 Enhancer Visual Reference",
}
