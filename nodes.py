"""Guided MiniMax H3 prompt preparation for ComfyUI.

The node intentionally has no ComfyUI imports, which keeps the decision and
formatting logic easy to test and lets ComfyUI discover it through the normal
NODE_CLASS_MAPPINGS interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable

if __package__:
    from .media_context import (
        CAMERA_ROLE,
        CONCRETE_KEYFRAME_ROLE,
        CONTINUE_ROLE,
        EDIT_ROLE,
        IDENTITY_ROLE,
        ITEM_ROLE,
        KEYFRAME_ROLE,
        MOTION_ROLE,
        REFERENCE_CONTEXT_TYPE,
        SCENE_ROLE,
        STORYBOARD_ROLE,
        STYLE_ROLE,
        SUBJECT_ROLES,
        TRANSFER_RELATION,
        reference_entries,
        reference_inventory,
    )
    from .reference_sheet import (
        AUDIO_REFERENCE_CONTEXT_TYPE,
        audio_reference_entries,
        audio_reference_inventory,
    )
else:  # Allows direct imports from the repository during tests.
    from media_context import (
        CAMERA_ROLE,
        CONCRETE_KEYFRAME_ROLE,
        CONTINUE_ROLE,
        EDIT_ROLE,
        IDENTITY_ROLE,
        ITEM_ROLE,
        KEYFRAME_ROLE,
        MOTION_ROLE,
        REFERENCE_CONTEXT_TYPE,
        SCENE_ROLE,
        STORYBOARD_ROLE,
        STYLE_ROLE,
        SUBJECT_ROLES,
        TRANSFER_RELATION,
        reference_entries,
        reference_inventory,
    )
    from reference_sheet import (
        AUDIO_REFERENCE_CONTEXT_TYPE,
        audio_reference_entries,
        audio_reference_inventory,
    )


AUTO_GOAL = "Auto - decide from the reference roles"
TEXT_GOAL = "Create a video from text only"
START_GOAL = "Animate a starting image"
END_GOAL = "Reach a supplied ending image"
CONNECT_GOAL = "Connect a first and last image"
REFERENCE_GOAL = "Generate from reference assets"
EDIT_GOAL = "Edit an existing video"
CONTINUE_GOAL = "Continue an existing video"
MOTION_GOAL = "Transfer motion to a different subject"

GOALS = [
    AUTO_GOAL,
    TEXT_GOAL,
    START_GOAL,
    END_GOAL,
    CONNECT_GOAL,
    REFERENCE_GOAL,
    EDIT_GOAL,
    CONTINUE_GOAL,
    MOTION_GOAL,
]

NO_IMAGE = "No image"
FIRST_IMAGE = "Use as the first frame"
LAST_IMAGE = "Use as the last frame"
FIRST_LAST_IMAGES = "Use as first and last frames"
APPEARANCE_IMAGE = "Reference appearance, scene, or style"
STORYBOARD_IMAGE = "Use as storyboard or concrete keyframe"
STORYBOARD_REFERENCE_IMAGE = "Use as storyboard or shot-planning reference"
KEYFRAME_IMAGE = "Use as a concrete keyframe"
MOTION_TARGET_IMAGE = "Target subject for motion transfer"

IMAGE_USES = [
    NO_IMAGE,
    FIRST_IMAGE,
    LAST_IMAGE,
    FIRST_LAST_IMAGES,
    APPEARANCE_IMAGE,
    KEYFRAME_IMAGE,
    STORYBOARD_REFERENCE_IMAGE,
    MOTION_TARGET_IMAGE,
    STORYBOARD_IMAGE,
]

NO_VIDEO = "No reference video"
EDIT_VIDEO = "Directly edit the source video"
CONTINUE_VIDEO = "Continue from the source video"
MOTION_VIDEO = "Transfer its motion or action"
STRUCTURE_VIDEO = "Reference its camera, cuts, or rhythm"
CONTENT_VIDEO = "Reference its subject, scene, or style"

VIDEO_USES = [
    NO_VIDEO,
    EDIT_VIDEO,
    CONTINUE_VIDEO,
    MOTION_VIDEO,
    STRUCTURE_VIDEO,
    CONTENT_VIDEO,
]

NO_AUDIO = "No reference audio"
COPY_ALL_AUDIO = "Reuse the complete audio signal"
COPY_PART_AUDIO = "Reuse only part or selected layers"
REFERENCE_AUDIO = "Reference voice, music, beat, or sound"
WEAK_AUDIO = "Use only broad audio mood"

AUDIO_USES = [
    NO_AUDIO,
    COPY_ALL_AUDIO,
    COPY_PART_AUDIO,
    REFERENCE_AUDIO,
    WEAK_AUDIO,
]

AUTO_FIDELITY = "Auto - choose per reference role"
FULL_FIDELITY = "Fully preserve the reference"
PARTIAL_FIDELITY = "Partly preserve while editing"
TRANSFER_FIDELITY = "Transfer attributes or motion"
WEAK_FIDELITY = "Use as weak inspiration"

FIDELITIES = [
    AUTO_FIDELITY,
    FULL_FIDELITY,
    PARTIAL_FIDELITY,
    TRANSFER_FIDELITY,
    WEAK_FIDELITY,
]

AUTO_VISUAL_STYLE = "Auto - derive from references and intent"

SHOT_PLAN_TYPE = "MINIMAX_H3_SHOT_PLAN"
TIMING_CONTEXT_TYPE = "MINIMAX_H3_TARGET_TIMING"
SHOT_TRANSITIONS = ["Direct cut", "Cross-dissolve", "Fade", "Wipe"]
H3_FPS = 24
H3_FRAME_MODULUS = 17
H3_FRAME_OFFSET = 5

_ASSET_RE = re.compile(
    r"^\s*<?(Subject|Picture|Video|Audio)\s*(\d+)>?\s*(?::|=|\s-\s)\s*(.+?)\s*$",
    re.IGNORECASE,
)
_MANUAL_SHOT_RE = re.compile(r"(?:\[\s*)?\bShot\s+(\d+)(?:\s*\])?", re.IGNORECASE)
_TIMECODE_RE = r"(?:\d{1,3}:\d{1,2}(?:\.\d{1,3})?|\d+(?:\.\d{1,3})?)"


@dataclass(frozen=True)
class Asset:
    kind: str
    number: int
    description: str

    @property
    def label(self) -> str:
        return f"<{self.kind} {self.number}>"


@dataclass(frozen=True)
class ModeDecision:
    mode: str
    checkpoint: str
    reason: str


@dataclass(frozen=True)
class ReferenceItem:
    label: str
    kind: str
    role: str
    description: str
    retention: str = ""
    shot_scope: str = ""
    transfer_target: str = ""
    roles: tuple[str, ...] = ()
    role_scopes: tuple[tuple[str, str], ...] = ()
    definition: str = ""
    from_context: bool = False


def _clean(text: str, fallback: str = "") -> str:
    value = " ".join((text or "").strip().split())
    return value or fallback


def _sentence(text: str, fallback: str = "") -> str:
    value = _clean(text, fallback)
    if value and value[-1] not in ".!?":
        value += "."
    return value


def _format_timestamp(seconds: float) -> str:
    """Format a float second value as the timestamp syntax used by H3."""

    total_milliseconds = round(float(seconds) * 1000)
    minutes, remainder = divmod(total_milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _native_frame_count(duration_seconds: float) -> int:
    """Snap a requested duration upward to ComfyUI's native H3 17k+5 grid."""

    requested_frames = max(H3_FRAME_OFFSET, math.ceil(float(duration_seconds) * H3_FPS - 1e-9))
    remainder = requested_frames % H3_FRAME_MODULUS
    return requested_frames + (H3_FRAME_OFFSET - remainder) % H3_FRAME_MODULUS


def _parse_timecode(value: str) -> float:
    """Parse ``MM:SS.mmm`` or plain seconds from the legacy shot widget."""

    text = value.strip()
    if ":" in text:
        minutes_text, seconds_text = text.split(":", 1)
        minutes = int(minutes_text)
        seconds = float(seconds_text)
        if seconds >= 60.0:
            raise ValueError(f"Invalid shot timestamp '{value}': seconds must be below 60.")
        result = minutes * 60.0 + seconds
    else:
        result = float(text)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"Invalid shot timestamp '{value}'.")
    return result


def _manual_shot_parts(fragment: str, number: int) -> dict:
    """Parse the header and body following one legacy ``Shot N`` marker."""

    range_match = re.fullmatch(
        rf"\s*,?\s*(?P<start>{_TIMECODE_RE})\s*[-–—]\s*"
        rf"(?P<end>{_TIMECODE_RE})\s*:\s*(?P<description>.+?)\s*",
        fragment,
        re.IGNORECASE | re.DOTALL,
    )
    if range_match:
        return {
            "number": number,
            "declared_start": _parse_timecode(range_match.group("start")),
            "declared_end": _parse_timecode(range_match.group("end")),
            "description": _clean(range_match.group("description")),
            "transition": "Direct cut",
        }

    cut_match = re.fullmatch(
        rf"\s*,?\s*(?:(?P<transition>direct\s+cut|cut|cross[- ]dissolve|fade|wipe)\s+at|at)\s+"
        rf"(?P<start>{_TIMECODE_RE})\s*[:,]\s*(?P<description>.+?)\s*",
        fragment,
        re.IGNORECASE | re.DOTALL,
    )
    if cut_match:
        transition_text = _clean(cut_match.group("transition")).lower()
        transition = {
            "": "Direct cut",
            "cut": "Direct cut",
            "direct cut": "Direct cut",
            "cross-dissolve": "Cross-dissolve",
            "cross dissolve": "Cross-dissolve",
            "fade": "Fade",
            "wipe": "Wipe",
        }[transition_text]
        return {
            "number": number,
            "declared_start": _parse_timecode(cut_match.group("start")),
            "declared_end": None,
            "description": _clean(cut_match.group("description")),
            "transition": transition,
        }

    simple_match = re.fullmatch(
        r"\s*:\s*(?P<description>.+?)\s*",
        fragment,
        re.IGNORECASE | re.DOTALL,
    )
    if simple_match:
        return {
            "number": number,
            "declared_start": None,
            "declared_end": None,
            "description": _clean(simple_match.group("description")),
            "transition": "Direct cut",
        }

    raise ValueError(
        f"Could not parse Shot {number}. Use 'Shot 1: description', "
        "'Shot 1, 00:00-00:02.500: description', or "
        "'Shot 2, cut at 00:02.500: description'."
    )


def _parse_manual_shots(manual_plan: str, duration: float) -> list[dict]:
    """Turn the legacy text widget into the same validated payload as shot nodes."""

    text = (manual_plan or "").strip()
    if not text:
        return []
    markers = list(_MANUAL_SHOT_RE.finditer(text))
    if not markers:
        raise ValueError(
            "Manual shot plan must contain sequential 'Shot N' markers. "
            "For free-form notes, leave the shot field empty and use target_description."
        )
    prefix = text[: markers[0].start()].strip(" \t\r\n.;")
    if prefix:
        raise ValueError("Manual shot plan must begin with Shot 1.")

    parsed: list[dict] = []
    for index, marker in enumerate(markers, start=1):
        number = int(marker.group(1))
        if number != index:
            raise ValueError(
                f"Manual shot numbers must be sequential from Shot 1; expected Shot {index}, "
                f"found Shot {number}."
            )
        end = markers[index].start() if index < len(markers) else len(text)
        parsed.append(_manual_shot_parts(text[marker.end() : end], number))

    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0.0 or duration > 15.0:
        raise ValueError("Manual shot duration must stay within the H3 range above 0 and at most 15 seconds.")

    starts: list[float] = []
    for index, entry in enumerate(parsed):
        declared_start = entry["declared_start"]
        if index == 0:
            start = 0.0 if declared_start is None else declared_start
        elif declared_start is not None:
            start = declared_start
        else:
            previous_end = parsed[index - 1]["declared_end"]
            if previous_end is None:
                raise ValueError(
                    f"Shot {index + 1} needs a cut time, because Shot {index} has no explicit end time."
                )
            start = previous_end
        starts.append(start)

    shots: list[dict] = []
    for index, (entry, start) in enumerate(zip(parsed, starts)):
        declared_end = entry["declared_end"]
        if index + 1 < len(parsed):
            end = starts[index + 1] if declared_end is None else declared_end
        else:
            end = duration if declared_end is None else declared_end
        timeline_end = end if index + 1 == len(parsed) else duration
        if start >= timeline_end:
            raise ValueError(
                f"Shot {index + 1} starts at {_format_timestamp(start)}, which must be before "
                f"its ending {_format_timestamp(timeline_end)}."
            )
        shots.append(
            {
                "start_time": start,
                "end_time": end,
                "description": entry["description"],
                "camera_direction": "",
                "transition": entry["transition"],
            }
        )
    return _shots_from_plan({"shots": shots})


def _extend_final_shot(shots: list[dict], effective_duration: float) -> list[dict]:
    """Return a copied timeline whose last shot reaches the native playback end."""

    if not shots:
        return []
    extended = [dict(shot) for shot in shots]
    if effective_duration <= extended[-1]["start_time"]:
        raise ValueError("The native duration must end after the final shot begins.")
    extended[-1]["end_time"] = float(effective_duration)
    return extended


def _shots_from_plan(shot_plan) -> list[dict]:
    """Validate and copy the serializable shot-chain payload."""

    if shot_plan is None:
        return []
    if not isinstance(shot_plan, dict) or not isinstance(shot_plan.get("shots"), (list, tuple)):
        raise ValueError("shot_plan must come from a MiniMax H3 Shot node.")

    shots: list[dict] = []
    previous_end = 0.0
    for index, raw_shot in enumerate(shot_plan["shots"], start=1):
        if not isinstance(raw_shot, dict):
            raise ValueError("shot_plan contains an invalid shot entry.")
        try:
            start = float(raw_shot["start_time"])
            end = float(raw_shot["end_time"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Every shot needs numeric start_time and end_time values.") from error
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Shot times must be finite numbers.")
        if index == 1 and not math.isclose(start, 0.0, abs_tol=0.0005):
            raise ValueError("Shot 1 must start at 0.000 seconds.")
        if index > 1 and not math.isclose(start, previous_end, abs_tol=0.0005):
            raise ValueError(
                f"Shot {index} must start at {_format_timestamp(previous_end)}, exactly when "
                f"Shot {index - 1} ends; its current start is {_format_timestamp(start)}."
            )
        if end <= start:
            raise ValueError(f"Shot {index} end_time must be greater than its start_time.")
        if start < 0.0 or end > 15.0:
            raise ValueError(f"Shot {index} must stay within the H3 range of 0 to 15 seconds.")
        description = _clean(str(raw_shot.get("description", "")))
        if not description:
            raise ValueError(f"Shot {index} needs a visible action or composition description.")
        transition = str(raw_shot.get("transition", SHOT_TRANSITIONS[0]))
        if transition not in SHOT_TRANSITIONS:
            raise ValueError(
                f"Shot {index} transition must be one of: {', '.join(SHOT_TRANSITIONS)}."
            )
        shots.append(
            {
                "start_time": start,
                "end_time": end,
                "description": description,
                "camera_direction": _clean(str(raw_shot.get("camera_direction", ""))),
                "transition": transition,
            }
        )
        previous_end = end
    if not shots:
        raise ValueError("shot_plan contains no shots.")
    return shots


def _build_timing_context(
    duration_seconds: float,
    shot_plan=None,
    manual_shot_plan: str = "",
) -> dict:
    """Resolve target timing once so upstream media preparation cannot form a graph cycle."""

    connected_shots = _shots_from_plan(shot_plan) if shot_plan is not None else []
    manual_shots = (
        _parse_manual_shots(manual_shot_plan, duration_seconds)
        if not connected_shots and manual_shot_plan.strip()
        else []
    )
    planned_shots = connected_shots or manual_shots
    requested_duration = (
        planned_shots[-1]["end_time"] if planned_shots else float(duration_seconds)
    )
    if not math.isfinite(requested_duration) or not 4.0 <= requested_duration <= 15.0:
        if planned_shots and requested_duration < 4.0:
            raise ValueError(
                "The shot plan ends before H3's 4-second minimum. Extend the final shot "
                "so its end_time is at least 4.000 seconds."
            )
        raise ValueError("MiniMax H3 target duration must be a finite value from 4 to 15 seconds.")
    h3_length = _native_frame_count(requested_duration)
    return {
        "version": 1,
        "source": "shot_plan" if connected_shots else "manual_shot_plan" if manual_shots else "duration",
        "requested_duration": requested_duration,
        "h3_length": h3_length,
        "effective_duration": h3_length / H3_FPS,
        "shots": [dict(shot) for shot in planned_shots],
    }


def _validated_timing_context(timing_context) -> dict:
    """Validate and copy a Target Timing payload connected to the Guide."""

    if not isinstance(timing_context, dict) or timing_context.get("version") != 1:
        raise ValueError("timing_context must come from a MiniMax H3 Target Timing node.")
    source = timing_context.get("source")
    if source not in {"duration", "shot_plan"}:
        raise ValueError("Target Timing contains an unsupported timing source.")
    try:
        requested_duration = float(timing_context["requested_duration"])
        effective_duration = float(timing_context["effective_duration"])
        h3_length = timing_context["h3_length"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Target Timing contains invalid duration or frame-count data.") from error
    if (
        not math.isfinite(requested_duration)
        or not 4.0 <= requested_duration <= 15.0
        or not math.isfinite(effective_duration)
    ):
        raise ValueError("Target Timing durations must remain finite and within H3's 4-15s range.")
    if isinstance(h3_length, bool) or not isinstance(h3_length, int):
        raise ValueError("Target Timing h3_length must be an integer.")
    expected_length = _native_frame_count(requested_duration)
    if h3_length != expected_length or not math.isclose(
        effective_duration, h3_length / H3_FPS, abs_tol=0.0005
    ):
        raise ValueError("Target Timing contains stale or inconsistent native H3 timing data.")
    raw_shots = timing_context.get("shots")
    if not isinstance(raw_shots, (list, tuple)):
        raise ValueError("Target Timing shots must be a list.")
    shots = _shots_from_plan({"shots": raw_shots}) if raw_shots else []
    if source == "shot_plan":
        if not shots or not math.isclose(
            shots[-1]["end_time"], requested_duration, abs_tol=0.0005
        ):
            raise ValueError("Target Timing shot_plan does not end at its requested duration.")
    elif shots:
        raise ValueError("Duration-only Target Timing cannot contain a shot plan.")
    return {
        "version": 1,
        "source": source,
        "requested_duration": requested_duration,
        "h3_length": h3_length,
        "effective_duration": effective_duration,
        "shots": shots,
    }


def _shot_body(shot: dict) -> str:
    parts = [_sentence(shot["description"])]
    if shot["camera_direction"]:
        parts.append(_sentence(shot["camera_direction"]))
    return " ".join(parts)


def _render_structured_shots(shots: list[dict], first_context: str = "") -> str:
    first_parts = [part for part in (first_context.strip(), _shot_body(shots[0])) if part]
    lines = [f"[Shot 1] {' '.join(first_parts)}"]
    transitions = {
        "Direct cut": "the video cuts directly to the next composition",
        "Cross-dissolve": "the previous composition cross-dissolves into the next",
        "Fade": "the previous composition fades into the next",
        "Wipe": "a wipe reveals the next composition",
    }
    for number, shot in enumerate(shots[1:], start=2):
        transition = transitions.get(shot["transition"], transitions["Direct cut"])
        lines.append(
            f"[Shot {number}] At {_format_timestamp(shot['start_time'])}, {transition}. "
            f"{_shot_body(shot)}"
        )
    return "\n".join(lines)


def _shot_plan_preview(shots: list[dict]) -> str:
    lines = []
    for number, shot in enumerate(shots, start=1):
        lines.append(
            f"Shot {number} | {_format_timestamp(shot['start_time'])}–"
            f"{_format_timestamp(shot['end_time'])} | {shot['description']}"
        )
    return "\n".join(lines)


def parse_assets(reference_assets: str) -> tuple[list[Asset], list[str]]:
    """Parse friendly ``Picture 1: ...`` lines while retaining free-form notes."""

    assets: list[Asset] = []
    notes: list[str] = []
    seen: set[tuple[str, int]] = set()
    for raw_line in (reference_assets or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _ASSET_RE.match(line)
        if not match:
            notes.append(line)
            continue
        kind = match.group(1).title()
        number = int(match.group(2))
        if number < 1:
            raise ValueError(f"{kind} labels are one-based; use <{kind} 1> instead of <{kind} {number}>.")
        key = (kind, number)
        if key in seen:
            raise ValueError(
                f"Duplicate <{kind} {number}> definition. Keep one definition per label and "
                "move extra details onto that same line."
            )
        seen.add(key)
        assets.append(Asset(kind, number, match.group(3).strip()))
    return assets, notes


def _asset_label_warnings(assets: list[Asset]) -> list[str]:
    """Report labels that cannot match H3's independently ordered media sockets."""

    warnings: list[str] = []
    for kind in ("Subject", "Picture", "Video", "Audio"):
        numbers = [asset.number for asset in assets if asset.kind == kind]
        if not numbers:
            continue
        expected = list(range(1, len(numbers) + 1))
        if sorted(numbers) != expected:
            warnings.append(
                f"{kind} labels must be contiguous from 1 in native input order; "
                f"found {', '.join(str(number) for number in numbers)}."
            )
        elif numbers != expected:
            warnings.append(
                f"{kind} labels are out of order; list them as "
                f"{', '.join(f'{kind} {number}' for number in expected)}."
            )
    return warnings


def _validate_active_asset_labels(
    assets: list[Asset], image_use: str, video_use: str, audio_use: str
) -> None:
    """Reject active media labels that cannot bind to native one-based sockets."""

    for kind, selected in (
        ("Picture", image_use != NO_IMAGE),
        ("Video", video_use != NO_VIDEO),
        ("Audio", audio_use != NO_AUDIO),
    ):
        if not selected:
            continue
        numbers = [asset.number for asset in assets if asset.kind == kind]
        if numbers and sorted(numbers) != list(range(1, len(numbers) + 1)):
            raise ValueError(
                f"Active {kind} labels must be contiguous from <{kind} 1> so they bind to "
                "the native H3 input order."
            )


def choose_mode(
    goal: str,
    image_use: str,
    video_use: str,
    audio_use: str,
) -> ModeDecision:
    """Choose the H3 prompt family from the user's intent and asset roles."""

    explicit = {
        TEXT_GOAL: ("T2VA", "The request starts from text without a frame anchor."),
        START_GOAL: ("I2VA", "The supplied image is the exact first frame."),
        END_GOAL: ("L2VA", "The supplied image is the exact final frame."),
        CONNECT_GOAL: ("FL2VA", "Two images anchor the first and last frames."),
        REFERENCE_GOAL: ("Ref2VA", "Assets guide generation without only acting as endpoint frames."),
        EDIT_GOAL: ("Ref2VA", "An existing source video is directly modified."),
        CONTINUE_GOAL: ("Ref2VA", "New content continues from an existing source video."),
        MOTION_GOAL: ("Ref2VA", "Motion from a reference is transferred to another visible subject."),
    }
    if goal in explicit:
        mode, reason = explicit[goal]
    elif video_use == EDIT_VIDEO:
        mode, reason = "Ref2VA", "The source video is directly edited."
    elif video_use == CONTINUE_VIDEO:
        mode, reason = "Ref2VA", "The target continues from a source video."
    elif video_use != NO_VIDEO or audio_use != NO_AUDIO:
        mode, reason = "Ref2VA", "Video or audio is used as a multimodal reference."
    elif image_use == FIRST_LAST_IMAGES:
        mode, reason = "FL2VA", "The images define both endpoint frames."
    elif image_use == FIRST_IMAGE:
        mode, reason = "I2VA", "The image defines the opening frame."
    elif image_use == LAST_IMAGE:
        mode, reason = "L2VA", "The image defines the final frame."
    elif image_use != NO_IMAGE:
        mode, reason = "Ref2VA", "The image guides content rather than only anchoring an endpoint."
    else:
        mode, reason = "T2VA", "No media reference role is selected."

    checkpoint = "H3-Base-Ref2VA" if mode == "Ref2VA" else "H3-Base-FL2VA"
    return ModeDecision(mode, checkpoint, reason)


def _resolve_roles(goal: str, image_use: str, video_use: str) -> tuple[str, str]:
    """Infer the obvious media roles when the top-level goal already states them."""

    if image_use == NO_IMAGE:
        image_use = {
            START_GOAL: FIRST_IMAGE,
            END_GOAL: LAST_IMAGE,
            CONNECT_GOAL: FIRST_LAST_IMAGES,
            MOTION_GOAL: MOTION_TARGET_IMAGE,
        }.get(goal, image_use)
    if video_use == NO_VIDEO:
        video_use = {
            EDIT_GOAL: EDIT_VIDEO,
            CONTINUE_GOAL: CONTINUE_VIDEO,
            MOTION_GOAL: MOTION_VIDEO,
        }.get(goal, video_use)
    return image_use, video_use


def _task_types(goal: str, image_use: str, video_use: str, audio_use: str) -> list[str]:
    task_types: list[str] = []

    if video_use == EDIT_VIDEO:
        task_types.append("video editing")
    elif video_use == CONTINUE_VIDEO:
        task_types.append("video continuation")

    if image_use in {FIRST_IMAGE, LAST_IMAGE, FIRST_LAST_IMAGES, KEYFRAME_IMAGE, STORYBOARD_IMAGE}:
        task_types.append("keyframe completion")

    reference_generation = (
        image_use in {APPEARANCE_IMAGE, MOTION_TARGET_IMAGE, STORYBOARD_REFERENCE_IMAGE}
        or video_use in {MOTION_VIDEO, STRUCTURE_VIDEO, CONTENT_VIDEO}
    )
    if reference_generation:
        task_types.append("reference generation")

    if audio_use in {COPY_ALL_AUDIO, COPY_PART_AUDIO}:
        task_types.append("audio reuse")
    elif audio_use in {REFERENCE_AUDIO, WEAK_AUDIO}:
        task_types.append("audio reference")

    if not task_types:
        task_types.append("reference generation")
    return task_types


def _assets_with_defaults(
    assets: list[Asset], image_use: str, video_use: str, audio_use: str
) -> list[Asset]:
    result = list(assets)
    pictures = [asset for asset in result if asset.kind == "Picture"]
    picture_numbers = {asset.number for asset in pictures}
    required_pictures = range(1, 3) if image_use == FIRST_LAST_IMAGES else range(1, 2)
    if image_use != NO_IMAGE:
        if not pictures:
            result.extend(
                Asset("Picture", number, "the supplied reference image")
                for number in required_pictures
            )
        elif image_use == FIRST_LAST_IMAGES and picture_numbers == {1}:
            result.append(Asset("Picture", 2, "the supplied ending reference image"))
    if video_use != NO_VIDEO and not any(asset.kind == "Video" for asset in result):
        result.append(Asset("Video", 1, "the supplied reference video"))
    if audio_use != NO_AUDIO and not any(asset.kind == "Audio" for asset in result):
        result.append(Asset("Audio", 1, "the supplied reference audio"))
    return result


def _next_subject_number(items: Iterable[ReferenceItem]) -> int:
    numbers = []
    for item in items:
        match = re.match(r"<Subject (\d+)>", item.label)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def _build_reference_items(
    assets: list[Asset],
    image_use: str,
    video_use: str,
    audio_use: str,
) -> list[ReferenceItem]:
    items: list[ReferenceItem] = []

    for asset in assets:
        if asset.kind == "Subject":
            items.append(ReferenceItem(asset.label, "Subject", "visible content", asset.description))

    pictures = [asset for asset in assets if asset.kind == "Picture" and image_use != NO_IMAGE]
    videos = [asset for asset in assets if asset.kind == "Video" and video_use != NO_VIDEO]
    audios = [asset for asset in assets if asset.kind == "Audio" and audio_use != NO_AUDIO]

    if image_use in {FIRST_IMAGE, LAST_IMAGE}:
        pictures = [picture for picture in pictures if picture.number == 1]
    elif image_use == FIRST_LAST_IMAGES:
        pictures = [picture for picture in pictures if picture.number in {1, 2}]
    if video_use in {EDIT_VIDEO, CONTINUE_VIDEO}:
        videos = [video for video in videos if video.number == 1]
    if audio_use == COPY_ALL_AUDIO:
        audios = [audio for audio in audios if audio.number == 1]

    if image_use in {APPEARANCE_IMAGE, MOTION_TARGET_IMAGE}:
        for picture in pictures:
            number = _next_subject_number(items)
            role = "motion target" if image_use == MOTION_TARGET_IMAGE else "appearance reference"
            description = (
                f"the target visible subject or scene defined by {picture.label}: "
                f"{picture.description}"
            )
            items.append(ReferenceItem(f"<Subject {number}>", "Subject", role, description))
    else:
        for picture in pictures:
            if image_use == FIRST_IMAGE:
                role = "first frame"
            elif image_use == LAST_IMAGE:
                role = "last frame"
            elif image_use == FIRST_LAST_IMAGES:
                if picture.number == 1:
                    role = "first frame"
                elif picture.number == 2:
                    role = "last frame"
                else:
                    role = "picture reference"
            elif image_use == KEYFRAME_IMAGE:
                role = "concrete keyframe"
            elif image_use == STORYBOARD_REFERENCE_IMAGE:
                role = "storyboard or shot-planning reference"
            elif image_use == STORYBOARD_IMAGE:
                role = "concrete keyframe (legacy combined role)"
            else:
                role = "picture reference"
            items.append(ReferenceItem(picture.label, "Picture", role, picture.description))

    if video_use == MOTION_VIDEO:
        for video in videos:
            number = _next_subject_number(items)
            description = f"the action or motion performance in {video.label}: {video.description}"
            items.append(ReferenceItem(f"<Subject {number}>", "Subject", "motion source", description))
    elif video_use == CONTENT_VIDEO:
        for video in videos:
            number = _next_subject_number(items)
            description = f"the reusable visible content in {video.label}: {video.description}"
            items.append(ReferenceItem(f"<Subject {number}>", "Subject", "content reference", description))
    else:
        for video in videos:
            if video_use == EDIT_VIDEO:
                role = "source video editing"
            elif video_use == CONTINUE_VIDEO:
                role = "continuation starting point"
            elif video_use == STRUCTURE_VIDEO:
                role = "camera, cut, and rhythm structure"
            else:
                role = "video reference"
            items.append(ReferenceItem(video.label, "Video", role, video.description))

    for audio in audios:
        if audio_use == COPY_ALL_AUDIO:
            role = "complete synchronized audio reuse"
        elif audio_use == COPY_PART_AUDIO:
            role = "partial audio or layer reuse"
        elif audio_use == REFERENCE_AUDIO:
            role = "voice, music, beat, or sound reference"
        elif audio_use == WEAK_AUDIO:
            role = "broad audio mood reference"
        else:
            role = "audio reference"
        items.append(ReferenceItem(audio.label, "Audio", role, audio.description))
    return items


_CONTEXT_SUBJECT_PHRASES = {
    IDENTITY_ROLE: "identity or appearance",
    ITEM_ROLE: "object, prop, clothing, interface, or effect",
    SCENE_ROLE: "scene or environment",
    STYLE_ROLE: "visual style",
    MOTION_ROLE: "motion or action",
}
_CONTEXT_DIRECT_DEFINITIONS = {
    CONCRETE_KEYFRAME_ROLE: "a concrete Ref2VA keyframe or composition anchor",
    STORYBOARD_ROLE: "a storyboard or shot-planning reference",
    KEYFRAME_ROLE: "a storyboard or keyframe reference (legacy combined role)",
    CAMERA_ROLE: "the reference video whose camera movement, cuts, rhythm, and temporal structure are tracked",
    EDIT_ROLE: "the source video for the target video edit",
    CONTINUE_ROLE: "the source video whose ending starts the target continuation",
}
_CONTEXT_INTERNAL_ROLES = {
    CONCRETE_KEYFRAME_ROLE: "concrete keyframe",
    STORYBOARD_ROLE: "storyboard or shot-planning reference",
    KEYFRAME_ROLE: "concrete keyframe (legacy combined role)",
    CAMERA_ROLE: "camera, cut, and rhythm structure",
    EDIT_ROLE: "source video editing",
    CONTINUE_ROLE: "continuation starting point",
}


def _joined_phrases(values: list[str]) -> str:
    values = [value for value in values if value]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _context_binding_clause(entry: dict, binding: dict, description: str) -> str:
    role_phrase = _CONTEXT_SUBJECT_PHRASES[binding["role"]]
    clause = f"the {role_phrase} supplied by {entry['label']}"
    detail = _clean(description).rstrip(".!?")
    if detail:
        clause += f", described as {detail}"
    notes = _clean(binding.get("notes", "")).rstrip(".!?")
    if notes:
        clause += f"; role instruction: {notes}"
    media_notes = _clean(entry.get("notes", "")).rstrip(".!?")
    if media_notes and media_notes != notes:
        clause += f"; media instruction: {media_notes}"
    scope = _display_shot_scope(binding.get("shot_scope", "")).rstrip(".!?")
    if scope:
        clause += f"; applies to {scope}"
    return clause


def _one_context_retention(bindings: list[dict], tracked_label: str) -> str:
    markers = {binding["retention"] for binding in bindings}
    if len(markers) != 1:
        raise ValueError(
            f"{tracked_label} combines role bindings with different retention markers. "
            "One H3 tracked item needs one retention_analysis relationship; use separate "
            "content_group values or align the binding retention values."
        )
    return markers.pop()


def _context_reference_items(entries: list[dict], parsed_assets: list[Asset]) -> list[ReferenceItem]:
    """Build deterministic Ref2VA rows from authoritative visual-role bindings."""

    descriptions = {
        asset.label: asset.description
        for asset in parsed_assets
        if asset.kind in {"Picture", "Video"}
    }
    grouped_subjects: dict[tuple, dict] = {}
    direct_entries: list[tuple[dict, list[dict]]] = []

    for entry_index, entry in enumerate(entries):
        direct_bindings = []
        for binding_index, binding in enumerate(entry["bindings"]):
            if binding["role"] in SUBJECT_ROLES:
                content_group = binding.get("content_group", "")
                key = (
                    ("content_group", content_group)
                    if content_group
                    else ("binding", entry_index, binding_index)
                )
                bucket = grouped_subjects.setdefault(
                    key,
                    {
                        "content_group": content_group,
                        "sources": [],
                    },
                )
                bucket["sources"].append((entry, binding))
            else:
                direct_bindings.append(binding)
        if direct_bindings:
            direct_entries.append((entry, direct_bindings))

    subject_labels = {
        key: f"<Subject {number}>"
        for number, key in enumerate(grouped_subjects, start=1)
    }
    group_labels = {
        bucket["content_group"]: subject_labels[key]
        for key, bucket in grouped_subjects.items()
        if bucket["content_group"]
    }

    items: list[ReferenceItem] = []
    for key, bucket in grouped_subjects.items():
        label = subject_labels[key]
        sources = bucket["sources"]
        bindings = [binding for _, binding in sources]
        clauses = [
            _context_binding_clause(entry, binding, descriptions.get(entry["label"], ""))
            for entry, binding in sources
        ]
        content_group = bucket["content_group"]
        if content_group:
            definition = f"content group '{content_group}', combining {_joined_phrases(clauses)}"
        else:
            definition = clauses[0]
        target_groups = {
            binding["transfer_target"]
            for binding in bindings
            if binding.get("transfer_target")
        }
        if len(target_groups) > 1:
            raise ValueError(
                f"{label} transfers to more than one content group. Split the source roles into "
                "separate content_group values so each tracked Subject has one destination."
            )
        target_group = next(iter(target_groups), "")
        transfer_target = group_labels.get(target_group, "")
        if target_group and not transfer_target:
            raise ValueError(
                f"{label} targets content_group={target_group!r}, but that group does not resolve "
                "to a visible Subject in the final reference context."
            )
        scopes = list(
            dict.fromkeys(
                _display_shot_scope(binding.get("shot_scope", ""))
                for binding in bindings
                if _clean(binding.get("shot_scope", ""))
            )
        )
        items.append(
            ReferenceItem(
                label=label,
                kind="Subject",
                role="context subject",
                description=definition,
                retention=_one_context_retention(bindings, label),
                shot_scope="; ".join(scopes),
                transfer_target=transfer_target,
                roles=tuple(binding["role"] for binding in bindings),
                role_scopes=tuple(
                    (binding["role"], _clean(binding.get("shot_scope", "")))
                    for binding in bindings
                ),
                definition=definition,
                from_context=True,
            )
        )

    for entry, bindings in direct_entries:
        label = entry["label"]
        roles = tuple(binding["role"] for binding in bindings)
        role_phrases = [_CONTEXT_DIRECT_DEFINITIONS[role] for role in roles]
        if len(role_phrases) == 1:
            definition = role_phrases[0]
        else:
            definition = (
                f"a directly tracked {entry['kind']} reference used as "
                + _joined_phrases(role_phrases)
            )
        description = _clean(descriptions.get(label, "")).rstrip(".!?")
        if description:
            definition += f": {description}"
        binding_notes = [
            f"{binding['role']}: {_clean(binding.get('notes', '')).rstrip('.!?')}"
            for binding in bindings
            if _clean(binding.get("notes", ""))
        ]
        media_notes = _clean(entry.get("notes", "")).rstrip(".!?")
        if media_notes and media_notes not in {
            _clean(binding.get("notes", "")).rstrip(".!?") for binding in bindings
        }:
            binding_notes.append(f"media instruction: {media_notes}")
        if binding_notes:
            definition += "; " + "; ".join(binding_notes)
        scopes = list(
            dict.fromkeys(
                _display_shot_scope(binding.get("shot_scope", ""))
                for binding in bindings
                if _clean(binding.get("shot_scope", ""))
            )
        )
        primary_role = next(
            (
                _CONTEXT_INTERNAL_ROLES[role]
                for role in (EDIT_ROLE, CONTINUE_ROLE, CAMERA_ROLE, CONCRETE_KEYFRAME_ROLE, STORYBOARD_ROLE, KEYFRAME_ROLE)
                if role in roles
            ),
            "direct reference",
        )
        items.append(
            ReferenceItem(
                label=label,
                kind="Picture" if entry["kind"] == "image" else "Video",
                role=primary_role,
                description=definition,
                retention=_one_context_retention(bindings, label),
                shot_scope="; ".join(scopes),
                roles=roles,
                role_scopes=tuple(
                    (binding["role"], _clean(binding.get("shot_scope", "")))
                    for binding in bindings
                ),
                definition=definition,
                from_context=True,
            )
        )
    return items


def _audio_context_assets(entries: list[dict]) -> list[Asset]:
    return [
        Asset(
            "Audio",
            number,
            _clean(entry.get("description"), "the saved reference audio"),
        )
        for number, entry in enumerate(entries, start=1)
    ]


def _audio_context_reference_items(entries: list[dict]) -> list[ReferenceItem]:
    roles = {
        COPY_ALL_AUDIO: "complete synchronized audio reuse",
        COPY_PART_AUDIO: "partial audio or layer reuse",
        REFERENCE_AUDIO: "voice, music, beat, or sound reference",
        WEAK_AUDIO: "broad audio mood reference",
    }
    items = []
    for entry in entries:
        description = _clean(entry.get("description"), "the saved reference audio")
        notes = _clean(entry.get("notes"))
        if notes:
            description += f"; workflow instruction: {notes}"
        items.append(
            ReferenceItem(
                label=entry["label"],
                kind="Audio",
                role=roles[entry["use"]],
                description=description,
                retention=entry["retention"],
                shot_scope=_display_shot_scope(entry.get("shot_scope", "")),
                from_context=True,
            )
        )
    return items


def _context_task_types(entries: list[dict], audio_use: str) -> list[str]:
    roles = {
        binding["role"]
        for entry in entries
        for binding in entry["bindings"]
    }
    task_types: list[str] = []
    if EDIT_ROLE in roles:
        task_types.append("video editing")
    if CONTINUE_ROLE in roles:
        task_types.append("video continuation")
    if roles & {CONCRETE_KEYFRAME_ROLE, KEYFRAME_ROLE}:
        task_types.append("keyframe completion")
    if roles & (SUBJECT_ROLES | {STORYBOARD_ROLE, CAMERA_ROLE}):
        task_types.append("reference generation")
    if audio_use in {COPY_ALL_AUDIO, COPY_PART_AUDIO}:
        task_types.append("audio reuse")
    elif audio_use in {REFERENCE_AUDIO, WEAK_AUDIO}:
        task_types.append("audio reference")
    return task_types or ["reference generation"]


def _shot_numbers_from_scope(scope: str, shot_count: int) -> list[int] | None:
    """Resolve common role scopes without guessing from arbitrary prose."""

    value = _clean(scope).casefold()
    if not value:
        return []
    if ";" in value:
        combined = []
        for part in value.split(";"):
            resolved = _shot_numbers_from_scope(part, shot_count)
            if resolved is None:
                return None
            combined.extend(resolved)
        return list(dict.fromkeys(combined))
    if value in {"all", "all shots", "every shot"}:
        return list(range(1, shot_count + 1))
    range_match = re.fullmatch(
        r"(?:shots?\s+)?(\d+)\s*(?:-|–|through|to)\s*(\d+)(?:\s+only)?",
        value,
    )
    if range_match:
        start, end = (int(part) for part in range_match.groups())
        if start <= end and 1 <= start and end <= shot_count:
            return list(range(start, end + 1))
        return None
    if re.fullmatch(r"(?:shots?\s+)?[\d,\s]+(?:\s+only)?", value):
        numbers = [int(number) for number in re.findall(r"\d+", value)]
        if numbers and len(set(numbers)) == len(numbers) and all(
            1 <= number <= shot_count for number in numbers
        ):
            return numbers
        return None
    return None


def _display_shot_scope(scope: str) -> str:
    """Normalize compact scope input for readable generated prompt text."""

    value = _clean(scope)
    resolved_form = value.casefold()
    if resolved_form in {"all", "all shots", "every shot"}:
        return "all shots"
    range_match = re.fullmatch(
        r"(?:shots?\s+)?(\d+)\s*(?:-|–|through|to)\s*(\d+)(?:\s+only)?",
        resolved_form,
    )
    if range_match:
        return f"Shots {range_match.group(1)}-{range_match.group(2)}"
    if re.fullmatch(r"(?:shots?\s+)?[\d,\s]+(?:\s+only)?", resolved_form):
        numbers = re.findall(r"\d+", resolved_form)
        if len(numbers) == 1:
            return f"Shot {numbers[0]}"
        return "Shots " + ", ".join(numbers)
    return value


def _looks_like_numbered_shot_scope(scope: str) -> bool:
    value = _clean(scope).casefold()
    return bool(
        re.fullmatch(
            r"(?:shots?\s+)?\d+(?:\s*(?:,|-|–|through|to)\s*\d+)*(?:\s+only)?",
            value,
        )
    )


def _context_item_directive(item: ReferenceItem, roles: tuple[str, ...]) -> str:
    if item.kind == "Subject":
        role_phrases = list(
            dict.fromkeys(
                _CONTEXT_SUBJECT_PHRASES[role]
                for role in roles
                if role in _CONTEXT_SUBJECT_PHRASES
            )
        )
        role_text = _joined_phrases(role_phrases) or "reusable visible content"
        return f"Apply {item.label}'s defined {role_text} in this shot."
    role_phrases = list(
        dict.fromkeys(
            _CONTEXT_INTERNAL_ROLES[role]
            for role in roles
            if role in _CONTEXT_INTERNAL_ROLES
        )
    )
    role_text = _joined_phrases(role_phrases) or item.role
    return f"Use {item.label} in this shot as its defined {role_text}."


def _apply_context_shot_scopes(
    structured_shots: list[dict] | None,
    items: list[ReferenceItem],
) -> tuple[list[dict], str]:
    """Place explicitly scoped reference labels in their declared Shot bodies."""

    shot_count = len(structured_shots) if structured_shots else 1
    directives: dict[int, list[str]] = {}
    for item in items:
        if not item.from_context:
            continue
        if item.kind == "Audio":
            numbers = (
                _shot_numbers_from_scope(item.shot_scope, shot_count)
                if item.shot_scope
                else []
            )
            for number in numbers or []:
                directives.setdefault(number, []).append(
                    _context_item_directive(item, ())
                )
            continue
        roles_by_shot: dict[int, list[str]] = {}
        role_scopes = item.role_scopes or tuple((role, item.shot_scope) for role in item.roles)
        for role, scope in role_scopes:
            if scope:
                numbers = _shot_numbers_from_scope(scope, shot_count)
            elif role in {EDIT_ROLE, CONTINUE_ROLE}:
                numbers = [1]
            else:
                numbers = []
            for number in numbers or []:
                roles_by_shot.setdefault(number, []).append(role)
        for number, roles in roles_by_shot.items():
            scoped_roles = tuple(dict.fromkeys(roles))
            directives.setdefault(number, []).append(
                _context_item_directive(item, scoped_roles)
            )

    if not structured_shots:
        return [], " ".join(directives.get(1, []))
    scoped_shots = [dict(shot) for shot in structured_shots]
    for number, shot in enumerate(scoped_shots, start=1):
        additions = directives.get(number)
        if additions:
            shot["description"] = f"{_sentence(shot['description'])} {' '.join(additions)}"
    return scoped_shots, ""


def _definition_line(item: ReferenceItem, final_shot_number: int = 1) -> str:
    if item.definition:
        return _sentence(f"{item.label} is {item.definition}")
    description = _clean(item.description)
    if item.kind == "Subject":
        return _sentence(f"{item.label} is {description}")
    if item.kind == "Picture":
        if item.role == "first frame":
            return _sentence(f"{item.label} is the first frame of [Shot 1], showing {description}")
        if item.role == "last frame":
            return _sentence(
                f"{item.label} is the final frame of [Shot {final_shot_number}], showing {description}"
            )
        if item.role in {"concrete keyframe", "concrete keyframe (legacy combined role)"}:
            return _sentence(f"{item.label} is a concrete keyframe anchor showing {description}")
        if item.role == "storyboard or shot-planning reference":
            return _sentence(
                f"{item.label} is a storyboard reference defining viewpoint, placement, or shot order: "
                f"{description}"
            )
        return _sentence(f"{item.label} is a picture reference showing {description}")
    if item.kind == "Video":
        if item.role == "source video editing":
            return _sentence(f"{item.label} is the source video for the target video edit: {description}")
        if item.role == "continuation starting point":
            return _sentence(f"{item.label} is the source video whose ending starts the target continuation: {description}")
        if item.role == "camera, cut, and rhythm structure":
            return _sentence(f"{item.label} provides the camera movement, cuts, rhythm, and temporal structure: {description}")
        return _sentence(f"{item.label} is a reference video: {description}")
    return _sentence(f"{item.label} is the {item.role}: {description}")


def _visual_marker(fidelity: str, role: str, allow_attribute_transfer: bool = False) -> str:
    if role == "motion source":
        return "attribute_transfer" if allow_attribute_transfer else "weak_reference"
    if role in {
        "first frame",
        "last frame",
        "concrete keyframe",
        "concrete keyframe (legacy combined role)",
    }:
        return "fully_preserved"
    if role in {"source video editing", "continuation starting point"}:
        return "fully_preserved" if fidelity == FULL_FIDELITY else "partially_preserved"
    if role == "motion target" and fidelity == TRANSFER_FIDELITY:
        return "fully_preserved"
    if role == "camera, cut, and rhythm structure":
        if fidelity == FULL_FIDELITY:
            return "fully_preserved"
        if fidelity == PARTIAL_FIDELITY:
            return "partially_preserved"
        return "weak_reference"
    if role == "storyboard or shot-planning reference":
        if fidelity == PARTIAL_FIDELITY:
            return "partially_preserved"
        if fidelity in {TRANSFER_FIDELITY, WEAK_FIDELITY}:
            return "weak_reference"
        return "fully_preserved"
    if fidelity == FULL_FIDELITY:
        return "fully_preserved"
    if fidelity == PARTIAL_FIDELITY:
        return "partially_preserved"
    if fidelity == TRANSFER_FIDELITY:
        return "fully_preserved"
    if fidelity == WEAK_FIDELITY:
        return "weak_reference"
    if role in {"edited visible content", "picture reference"}:
        return "partially_preserved"
    return "fully_preserved"


def _retention_line(
    item: ReferenceItem,
    fidelity: str,
    audio_use: str,
    final_shot_number: int = 1,
    shot_count: int = 1,
    allow_attribute_transfer: bool = False,
) -> str:
    if item.kind == "Audio":
        marker = {
            COPY_ALL_AUDIO: "fully_copy",
            COPY_PART_AUDIO: "partially_copy",
            REFERENCE_AUDIO: "reference",
            WEAK_AUDIO: "weak_reference",
        }.get(audio_use, "reference")
        explanation = {
            "fully_copy": "the complete source signal is reused as the target video's complete audio track",
            "partially_copy": "selected source audio or layers are copied and mixed with the target audio",
            "reference": "the signal guides the requested audio characteristics without being copied directly",
            "weak_reference": "only broad sonic category or atmosphere is retained",
        }[marker]
        context = f" ({item.shot_scope})" if item.shot_scope else ""
        return _sentence(f"{item.label}{context}: {marker} - {explanation}")

    marker = item.retention or _visual_marker(fidelity, item.role, allow_attribute_transfer)
    if marker == "attribute_transfer":
        explanation = (
            f"the defined characteristics or motion are transferred to {item.transfer_target} "
            "without replacing its unrelated identity or attributes"
            if item.transfer_target
            else "the referenced characteristics or motion are transferred to the identifiable target subject"
        )
    elif marker == "fully_preserved":
        explanation = "the defined appearance, composition, or reference role is retained"
    elif marker == "partially_preserved":
        explanation = "the reference remains identifiable while the requested changes are applied"
    else:
        explanation = "only the requested broad structure, category, style, or atmosphere is retained"

    if item.shot_scope:
        context = f" ({item.shot_scope})"
    elif item.kind == "Subject":
        context = "" if item.from_context else " (appears in [Shot 1])" if shot_count == 1 else ""
    elif item.kind == "Picture":
        if item.role == "first frame":
            context = " ([Shot 1] first frame)"
        elif item.role == "last frame":
            context = f" ([Shot {final_shot_number}] last frame)"
        elif item.role in {"concrete keyframe", "concrete keyframe (legacy combined role)"}:
            context = " ([Shot 1] concrete keyframe)" if shot_count == 1 else " (concrete keyframe)"
        else:
            context = f" ({item.role})"
    else:
        context = f" ({item.role})"
    return _sentence(f"{item.label}{context}: {marker} - {explanation}")


def _role_sentences(items: list[ReferenceItem], video_use: str) -> str:
    parts: list[str] = []
    edit_sources = [
        item.label
        for item in items
        if item.role == "source video editing" or EDIT_ROLE in item.roles
    ]
    continuation_sources = [
        item.label
        for item in items
        if item.role == "continuation starting point" or CONTINUE_ROLE in item.roles
    ]
    if video_use == EDIT_VIDEO or edit_sources:
        source = edit_sources[0] if edit_sources else "the source video"
        parts.append(
            f"The visible timeline begins from {source} and applies only the requested edits."
        )
    elif video_use == CONTINUE_VIDEO or continuation_sources:
        source = continuation_sources[0] if continuation_sources else "the source video"
        parts.append(
            f"The new timeline begins from the ending state of {source} and continues naturally."
        )

    targets = [item.label for item in items if item.role == "motion target"]
    motions = [item.label for item in items if item.role == "motion source"]
    if targets and motions:
        parts.append(
            f"{targets[0]} keeps its visual identity while receiving the action, pose changes, timing, "
            f"and motion trajectory from {motions[0]}."
        )
    context_transfers = [
        item for item in items if item.retention == TRANSFER_RELATION and item.transfer_target
    ]
    for source in context_transfers:
        parts.append(
            f"{source.label} supplies only the declared transferred attribute or motion to "
            f"{source.transfer_target}; keep the destination's unrelated identity and attributes."
        )
    if not (targets and motions) and items:
        labels = ", ".join(item.label for item in items)
        parts.append(f"Apply the defined roles of {labels} at the points where they become visible or audible.")
    return " ".join(parts)


def _detail_body(
    target_description: str,
    visual_style: str,
    shot_plan: str,
    camera_direction: str,
    dialogue_and_text: str,
    role_sentences: str,
    reference_notes: list[str],
    complete_audio_label: str = "",
) -> str:
    parts = [_sentence(target_description, "Create a coherent audiovisual scene from the supplied intent")]
    if role_sentences:
        parts.append(role_sentences)
    if shot_plan.strip():
        parts.append(_sentence(f"Shot and timing plan: {shot_plan}"))
    if camera_direction.strip():
        parts.append(_sentence(camera_direction))
    if dialogue_and_text.strip():
        audio_constraint = (
            f" Any spoken or sung wording must already exist in the unchanged {complete_audio_label} "
            "track; do not synthesize, replace, or remix audio."
            if complete_audio_label
            else ""
        )
        parts.append(
            _sentence(
                "Dialogue and visible-text requirements: "
                f"{dialogue_and_text} Preserve exact wording and language; put spoken words inside "
                "<d>[Language] ...</d> and visible text in double quotation marks"
                + audio_constraint
            )
        )
    if reference_notes:
        parts.append(_sentence("Additional reference notes: " + " ".join(reference_notes)))
    return " ".join(parts)


def _section_sentence(text: str, fallback: str) -> str:
    value = _clean(text, fallback)
    return "N/A" if value.upper().rstrip(".") == "N/A" else _sentence(value)


def _reference_audio_sections(
    items: list[ReferenceItem],
    audio_use: str,
    soundscape: str,
    music: str,
) -> tuple[str, str]:
    """Cite reference audio in the semantic audio phase without inventing a new mix."""

    labels = [item.label for item in items if item.kind == "Audio"]
    if not labels:
        return (
            _section_sentence(soundscape, "Use coherent ambience and synchronized physical sounds"),
            _section_sentence(music, "N/A"),
        )
    label_text = ", ".join(labels)
    sound_detail = _clean(soundscape)
    music_detail = _clean(music)

    if audio_use == COPY_ALL_AUDIO:
        sound = (
            f"{label_text} is reused 1:1 as the target video's complete final audio track, "
            "with no added, removed, or replaced sound layers."
        )
        if sound_detail:
            sound += " The copied ambience and physical-sound content is described as: " + _sentence(
                sound_detail
            )
        score = (
            f"Any audience-only music already present in {label_text} remains inside the unchanged "
            "copied track; no new score is added."
        )
        if music_detail and music_detail.upper().rstrip(".") != "N/A":
            score += " The copied score is described as: " + _sentence(music_detail)
        return _sentence(sound), _sentence(score)

    relationship = {
        COPY_PART_AUDIO: "Selected time ranges or layers are copied and mixed with the target audio",
        REFERENCE_AUDIO: "Its declared timbre, rhythm, voice, music, or sound texture guides the target without copying samples",
        WEAK_AUDIO: "Only its broad sonic category or atmosphere guides the target",
    }.get(audio_use, "It guides the target audio without an undeclared copy relationship")
    sound = _sentence(f"{label_text}: {relationship}")
    if sound_detail:
        sound += " " + _sentence(sound_detail)
    elif audio_use == COPY_PART_AUDIO:
        sound += " Other target sounds remain synchronized to the described actions."

    if not music_detail or music_detail.upper().rstrip(".") == "N/A":
        score = "N/A"
    else:
        score_relationship = {
            COPY_PART_AUDIO: "Any copied audience-only music layer is taken from",
            REFERENCE_AUDIO: "The requested audience-only score follows the declared music characteristics of",
            WEAK_AUDIO: "The requested audience-only score uses only broad inspiration from",
        }.get(audio_use, "The requested audience-only score is related to")
        score = _sentence(f"{score_relationship} {label_text}") + " " + _sentence(music_detail)
    return sound, score


def _base_style_opening(mode: str, visual_style: str) -> str:
    value = _clean(visual_style)
    if value and value != AUTO_VISUAL_STYLE:
        return value
    return {
        "I2VA": "Preserve <Picture 1>'s existing visual style",
        "FL2VA": "Derive a compatible visual treatment from Picture 1 and Picture 2",
        "L2VA": "Use a visual treatment that converges on <Picture 1>'s existing style",
    }.get(mode, "Use the visual style specified by the target description")


def _reference_style_sentence(visual_style: str) -> str:
    value = _clean(visual_style)
    if not value or value == AUTO_VISUAL_STYLE:
        return (
            "The target video derives its visual style from the active visual references and "
            "the user's requested intent."
        )
    return _sentence(f"The target video uses a {value} style")


def _base_prompt(
    mode: str,
    duration: float,
    target_description: str,
    visual_style: str,
    shot_plan: str,
    camera_direction: str,
    dialogue_and_text: str,
    soundscape: str,
    music: str,
    reference_notes: list[str],
    structured_shots: list[dict] | None = None,
    final_shot_number: int = 1,
) -> str:
    duration_text = f"{duration:.2f}"
    if mode == "I2VA":
        instruction = (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
        )
        anchor = "Begin from the exact subject, composition, scene, and style in <Picture 1>."
    elif mode == "FL2VA":
        instruction = (
            "How the reference pictures align with the target video — Picture 1 (from Shot 1) "
            f"aligns with the 0.00-second mark of the target video; Picture 2 (from Shot {final_shot_number}) "
            f"aligns with the {duration_text}-second mark of the target video.\n\n"
        )
        anchor = "Begin at Picture 1 and show a continuous observable path that lands exactly on Picture 2."
    elif mode == "L2VA":
        instruction = (
            "How the reference pictures align with the target video — "
            f"<Picture 1> (from [Shot {final_shot_number}]) "
            f"aligns with the {duration_text}-second mark of the target video.\n\n"
        )
        anchor = "Infer a plausible preceding state and converge exactly on <Picture 1> at the end."
    else:
        instruction = ""
        anchor = ""

    body = _detail_body(
        target_description,
        visual_style,
        "" if structured_shots else shot_plan,
        camera_direction,
        dialogue_and_text,
        anchor,
        reference_notes,
    )
    style = _base_style_opening(mode, visual_style)
    timeline = (
        _render_structured_shots(structured_shots, f"{style}. {body}")
        if structured_shots
        else f"[Shot 1] {style}. {body}"
    )
    sound = _section_sentence(soundscape, "Use coherent ambience and synchronized physical sounds")
    score = _section_sentence(music, "N/A")
    return (
        f"{instruction}integrated_multimodal_description: {timeline}\n\n"
        f"overall_soundscape: {sound}\n\n"
        f"non_diegetic_music: {score}"
    )


def _reference_prompt(
    goal: str,
    duration: float,
    items: list[ReferenceItem],
    target_description: str,
    visual_style: str,
    shot_plan: str,
    camera_direction: str,
    dialogue_and_text: str,
    soundscape: str,
    music: str,
    fidelity: str,
    image_use: str,
    video_use: str,
    audio_use: str,
    reference_notes: list[str],
    structured_shots: list[dict] | None = None,
    final_shot_number: int = 1,
    task_types: list[str] | None = None,
) -> str:
    definitions = "\n".join(_definition_line(item, final_shot_number) for item in items)
    resolved_task_types = task_types or _task_types(goal, image_use, video_use, audio_use)
    types = " + ".join(resolved_task_types)
    labels = ", ".join(item.label for item in items)

    if "video editing" in types:
        source = next(
            (
                item.label
                for item in items
                if item.role == "source video editing" or EDIT_ROLE in item.roles
            ),
            "<Video 1>",
        )
        summary_start = f"The target video is an edited version of {source}."
    elif "video continuation" in types:
        source = next(
            (
                item.label
                for item in items
                if item.role == "continuation starting point" or CONTINUE_ROLE in item.roles
            ),
            "<Video 1>",
        )
        summary_start = f"The target video continues from the ending state of {source}."
    else:
        summary_start = "The target video is generated from the defined reference relationships."
    reference_clause = f" The reference roles use {labels}." if labels else ""
    summary = f"[{types}] {summary_start} {_sentence(target_description)}{reference_clause}"

    shot_count = len(structured_shots) if structured_shots else 1
    allow_attribute_transfer = image_use == MOTION_TARGET_IMAGE and video_use == MOTION_VIDEO
    retention = "\n".join(
        _retention_line(
            item,
            fidelity,
            audio_use,
            final_shot_number,
            shot_count,
            allow_attribute_transfer,
        )
        for item in items
    )
    has_context_items = any(item.from_context for item in items)
    role_sentences = "" if has_context_items else _role_sentences(items, video_use)
    scoped_shots, single_shot_directives = _apply_context_shot_scopes(
        structured_shots,
        items,
    )
    complete_audio_label = next(
        (item.label for item in items if item.role == "complete synchronized audio reuse"),
        "",
    )
    detail = _detail_body(
        target_description,
        visual_style,
        "" if structured_shots else shot_plan,
        camera_direction,
        dialogue_and_text,
        role_sentences,
        reference_notes,
        complete_audio_label,
    )
    if single_shot_directives:
        detail = f"{detail} {single_shot_directives}"
    style_sentence = _reference_style_sentence(visual_style)
    timeline = (
        _render_structured_shots(scoped_shots, detail)
        if scoped_shots
        else f"[Shot 1] {detail}"
    )
    sound, score = _reference_audio_sections(items, audio_use, soundscape, music)
    definitions_block = f"{definitions}\n" if definitions else ""
    retention_block = f"{retention}\n" if retention else ""

    return (
        f"subject_definitions:\n{definitions_block}\n"
        f"summary:\n{summary}\n\n"
        f"retention_analysis:\n{retention_block}\n"
        "detailed_description:\n"
        f"{style_sentence} The target video lasts {duration:.2f} seconds.\n"
        f"{timeline}\n\n"
        f"overall_soundscape:\n{sound}\n\n"
        f"non_diegetic_music:\n{score}"
    )


def _warnings(
    decision: ModeDecision,
    goal: str,
    image_use: str,
    video_use: str,
    audio_use: str,
    assets: list[Asset],
    target_description: str,
    reference_fidelity: str,
    shot_count: int,
    dialogue_and_text: str,
) -> list[str]:
    warnings = _asset_label_warnings(assets)
    if not target_description.strip():
        warnings.append(
            "Target description is empty. Describe the actual subject, action or edit, setting, and ending before enhancement."
        )
    counts = {kind: sum(asset.kind == kind for asset in assets) for kind in ("Picture", "Video", "Audio")}
    active_picture_count = counts["Picture"] if image_use != NO_IMAGE else 0
    active_video_count = counts["Video"] if video_use != NO_VIDEO else 0
    active_audio_count = counts["Audio"] if audio_use != NO_AUDIO else 0

    if decision.mode == "Ref2VA":
        if counts["Picture"] > 9:
            warnings.append("Ref2VA accepts at most 9 reference images.")
        if counts["Video"] > 3:
            warnings.append("Ref2VA accepts at most 3 reference videos.")
        if counts["Audio"] > 3:
            warnings.append("Ref2VA accepts at most 3 reference audio clips.")
        if sum(counts.values()) > 12:
            warnings.append("Ref2VA accepts at most 12 media files in total.")

    expected_picture_count = {
        FIRST_IMAGE: 1,
        LAST_IMAGE: 1,
        FIRST_LAST_IMAGES: 2,
    }.get(image_use)
    if expected_picture_count is not None and active_picture_count != expected_picture_count:
        warnings.append(
            f"'{image_use}' requires exactly {expected_picture_count} ordered Picture "
            f"label(s); found {active_picture_count}."
        )

    if video_use in {EDIT_VIDEO, CONTINUE_VIDEO} and active_video_count != 1:
        warnings.append(
            f"'{video_use}' currently requires exactly one source Video label; "
            f"found {active_video_count}."
        )
    if audio_use == COPY_ALL_AUDIO and active_audio_count != 1:
        warnings.append(
            "Complete 1:1 audio reuse requires exactly one Audio label; use partial reuse when "
            "mixing multiple source signals."
        )
    if audio_use == COPY_ALL_AUDIO and dialogue_and_text.strip():
        warnings.append(
            "Complete audio reuse cannot create a new spoken or sung signal. Treat dialogue/lyrics "
            "in the combined text field as a transcription of the unchanged copied Audio track "
            "(new visible text is still allowed), or choose partial reuse/audio reference."
        )

    if audio_use != NO_AUDIO and active_picture_count == 0 and active_video_count == 0:
        warnings.append("Reference audio cannot be the only Ref2VA media input; add an image or video.")
    if decision.mode != "Ref2VA" and (video_use != NO_VIDEO or audio_use != NO_AUDIO):
        warnings.append(
            "The explicit goal selects a frame/text mode, but video or audio reference roles require Ref2VA. "
            "Choose Auto or a reference-based goal if those assets should be used."
        )
    if goal == TEXT_GOAL and image_use != NO_IMAGE:
        warnings.append("Text-only was selected while an image role is active; the explicit goal takes priority.")
    expected_image_use = {
        "I2VA": FIRST_IMAGE,
        "FL2VA": FIRST_LAST_IMAGES,
        "L2VA": LAST_IMAGE,
    }.get(decision.mode)
    if expected_image_use and image_use != expected_image_use:
        warnings.append(
            f"{decision.mode} expects '{expected_image_use}', but '{image_use}' is selected. "
            "Choose Auto or align the image role with the explicit goal."
        )
    if decision.mode == "Ref2VA" and not (
        active_picture_count or active_video_count or active_audio_count
    ):
        warnings.append("Ref2VA is selected but no image, video, or audio reference is described.")
    if (image_use == MOTION_TARGET_IMAGE) != (video_use == MOTION_VIDEO):
        warnings.append(
            "Motion transfer requires both a target-picture role and a motion-source video role."
        )
    if reference_fidelity == TRANSFER_FIDELITY and not (
        image_use == MOTION_TARGET_IMAGE and video_use == MOTION_VIDEO
    ):
        warnings.append(
            "attribute_transfer requires the explicit motion target + motion source pair. "
            "The global transfer fidelity is downgraded for all other reference roles."
        )
    if image_use == STORYBOARD_IMAGE:
        warnings.append(
            "The legacy combined storyboard/keyframe role is interpreted as a concrete keyframe. "
            "Choose the dedicated storyboard role when the picture only plans shots."
        )
    if image_use in {KEYFRAME_IMAGE, STORYBOARD_IMAGE} and shot_count > 1:
        warnings.append(
            "A concrete keyframe is selected for a multi-shot plan. Name its <Picture N> label in "
            "the intended shot description so the enhancer can preserve the mapping."
        )
    if image_use in {FIRST_IMAGE, LAST_IMAGE, FIRST_LAST_IMAGES, KEYFRAME_IMAGE, STORYBOARD_IMAGE} and (
        reference_fidelity not in {AUTO_FIDELITY, FULL_FIDELITY}
    ):
        warnings.append(
            "Concrete frame anchors are always fully_preserved; the selected global fidelity is "
            "ignored for those Picture entries."
        )
    if video_use in {EDIT_VIDEO, CONTINUE_VIDEO} and reference_fidelity in {
        TRANSFER_FIDELITY,
        WEAK_FIDELITY,
    }:
        warnings.append(
            "Editing and continuation source videos cannot use attribute_transfer or weak_reference; "
            "their retention marker is constrained to partially_preserved."
        )
    for kind, selected in (
        ("Picture", image_use != NO_IMAGE),
        ("Video", video_use != NO_VIDEO),
        ("Audio", audio_use != NO_AUDIO),
    ):
        if not selected and any(asset.kind == kind for asset in assets):
            warnings.append(
                f"{kind} labels are listed in the reference inventory but their role is set to none, "
                "so they are not used in the prompt."
            )
    return warnings


def _context_mode_decision(reference_context: dict, entries: list[dict]) -> ModeDecision:
    if not entries:
        raise ValueError(
            "The connected reference_context contains no visual references. Connect the final "
            "output of a populated MiniMax H3 Enhancer Visual Reference chain."
        )
    route_families = {entry.get("route_family") for entry in entries}
    if route_families == {"endpoint"}:
        endpoint_inputs = {entry.get("h3_input") for entry in entries}
        if endpoint_inputs == {"first_frame", "last_frame"}:
            mode = "FL2VA"
        elif endpoint_inputs == {"first_frame"}:
            mode = "I2VA"
        elif endpoint_inputs == {"last_frame"}:
            mode = "L2VA"
        else:
            raise ValueError("The connected endpoint context has invalid first/last-frame routes.")
    else:
        mode = "Ref2VA"
    if mode not in {"I2VA", "L2VA", "FL2VA", "Ref2VA"}:
        raise ValueError(f"The connected reference_context has unsupported mode_hint={mode!r}.")
    checkpoint = "H3-Base-Ref2VA" if mode == "Ref2VA" else "H3-Base-FL2VA"
    return ModeDecision(
        mode,
        checkpoint,
        "The connected Visual Reference chain is authoritative for the H3 family, labels, roles, and native routes.",
    )


def _context_reference_notes(entries: list[dict], parsed_assets: list[Asset]) -> list[str]:
    descriptions = {
        asset.label: asset.description
        for asset in parsed_assets
        if asset.kind in {"Picture", "Video"}
    }
    notes = []
    for entry in entries:
        parts = []
        description = _clean(descriptions.get(entry["label"], ""))
        if description:
            parts.append(description)
        media_notes = _clean(entry.get("notes", ""))
        if media_notes:
            parts.append(media_notes)
        for binding in entry["bindings"]:
            binding_notes = _clean(binding.get("notes", ""))
            if binding_notes and binding_notes not in parts:
                parts.append(f"{binding['role']}: {binding_notes}")
        if parts:
            notes.append(f"{entry['label']}: " + "; ".join(parts))
    return notes


def _context_warnings(
    decision: ModeDecision,
    goal: str,
    image_use: str,
    video_use: str,
    audio_use: str,
    reference_fidelity: str,
    parsed_assets: list[Asset],
    entries: list[dict],
    target_description: str,
    shot_count: int,
    dialogue_and_text: str,
    h3_length: int,
    timing_is_connected: bool,
) -> list[str]:
    warnings: list[str] = []
    if not target_description.strip():
        warnings.append(
            "Target description is empty. Describe the actual subject, action or edit, setting, "
            "and ending before enhancement."
        )

    roles = {
        binding["role"]
        for entry in entries
        for binding in entry["bindings"]
    }
    if EDIT_ROLE in roles and CONTINUE_ROLE in roles:
        raise ValueError(
            "One target cannot simultaneously use chained source-video edit and continuation roles. "
            "Choose the intended operation in the Visual Reference Role chain."
        )
    for role, display in (
        (EDIT_ROLE, "source-video edit"),
        (CONTINUE_ROLE, "source-video continuation"),
    ):
        source_count = sum(
            any(binding["role"] == role for binding in entry["bindings"])
            for entry in entries
        )
        if source_count > 1:
            raise ValueError(
                f"The final reference_context contains {source_count} {display} sources. "
                "The target summary can track exactly one direct source video for this operation."
            )
    if decision.mode != "Ref2VA" and audio_use != NO_AUDIO:
        raise ValueError(
            "Exact endpoint reference_context routes use MiniMax H3 Image to Video, while reference "
            "audio requires MiniMax H3 Reference to Video. Remove the audio reference role or build "
            "a separate Ref2VA workflow."
        )

    explicit_modes = {
        TEXT_GOAL: "T2VA",
        START_GOAL: "I2VA",
        END_GOAL: "L2VA",
        CONNECT_GOAL: "FL2VA",
        REFERENCE_GOAL: "Ref2VA",
        EDIT_GOAL: "Ref2VA",
        CONTINUE_GOAL: "Ref2VA",
        MOTION_GOAL: "Ref2VA",
    }
    explicit_mode = explicit_modes.get(goal)
    if explicit_mode and explicit_mode != decision.mode:
        warnings.append(
            f"The goal dropdown suggests {explicit_mode}, but the connected reference_context "
            f"authoritatively routes as {decision.mode}; the context takes priority."
        )
    if image_use != NO_IMAGE or video_use != NO_VIDEO:
        warnings.append(
            "The legacy image/video role dropdowns are ignored because reference_context is "
            "connected; edit roles on the Visual Reference Role chain instead."
        )
    if reference_fidelity != AUTO_FIDELITY:
        warnings.append(
            "The global reference_fidelity widget is ignored for chained visuals; each role "
            "binding's validated retention relationship takes priority."
        )

    context_labels = {entry["label"] for entry in entries}
    ignored_visuals = [
        asset.label
        for asset in parsed_assets
        if asset.kind in {"Picture", "Video"} and asset.label not in context_labels
    ]
    if ignored_visuals:
        warnings.append(
            "Visual inventory label(s) not present in the authoritative context are ignored: "
            + ", ".join(ignored_visuals)
            + "."
        )
    manual_subjects = [asset.label for asset in parsed_assets if asset.kind == "Subject"]
    if manual_subjects:
        warnings.append(
            "Manual Subject inventory rows are ignored while reference_context is connected; "
            "Subjects are derived only from explicit reusable-content role bindings: "
            + ", ".join(manual_subjects)
            + "."
        )

    audio_assets = [asset for asset in parsed_assets if asset.kind == "Audio"]
    warnings.extend(_asset_label_warnings(audio_assets))
    active_audio_count = len(audio_assets) if audio_use != NO_AUDIO else 0
    if active_audio_count > 3:
        warnings.append("Ref2VA accepts at most 3 reference audio clips.")
    if audio_use == COPY_ALL_AUDIO and active_audio_count != 1:
        warnings.append(
            "Complete 1:1 audio reuse requires exactly one Audio label; use partial reuse when "
            "mixing multiple source signals."
        )
    if audio_use == COPY_ALL_AUDIO and dialogue_and_text.strip():
        warnings.append(
            "Complete audio reuse cannot create a new spoken or sung signal. Treat dialogue/lyrics "
            "as a transcription of the unchanged copied Audio track, or choose another audio role."
        )
    picture_count = sum(entry["kind"] == "image" for entry in entries)
    video_entries = [entry for entry in entries if entry["kind"] == "video"]
    video_count = len(video_entries)
    if decision.mode == "Ref2VA" and picture_count + video_count + active_audio_count > 12:
        warnings.append("Ref2VA accepts at most 12 media files in total, including audio.")
    for entry in video_entries:
        target_frame_count = entry.get("target_frame_count")
        if target_frame_count is not None and target_frame_count != h3_length:
            raise ValueError(
                f"{entry['label']} was prepared for h3_length={target_frame_count}, but the Guide "
                f"uses h3_length={h3_length}. Feed the same MiniMax H3 Target Timing.h3_length "
                "to every video Visual Reference and the native H3 node."
            )
        if target_frame_count is None and entry.get("native_frame_count", 0) > h3_length:
            prefix = (
                "Target Timing is connected, but its h3_length is not connected to"
                if timing_is_connected
                else "Use MiniMax H3 Target Timing and connect its h3_length to"
            )
            warnings.append(
                f"{prefix} {entry['label']}'s Visual Reference. Its current analysis includes "
                "frames beyond the target duration that native H3 will discard."
            )
    if audio_use == NO_AUDIO and audio_assets:
        warnings.append(
            "Audio labels are listed in the reference inventory but their role is set to none, "
            "so they are not used in the prompt."
        )
    if roles & {CONCRETE_KEYFRAME_ROLE, KEYFRAME_ROLE} and shot_count > 1:
        if any(
            binding["role"] in {CONCRETE_KEYFRAME_ROLE, KEYFRAME_ROLE}
            and not binding.get("shot_scope")
            for entry in entries
            for binding in entry["bindings"]
        ):
            warnings.append(
                "A chained concrete keyframe has no shot_scope in a multi-shot plan. Set its "
                "Visual Reference Role.shot_scope or cite its Picture label in the intended shot."
            )
    if KEYFRAME_ROLE in roles:
        warnings.append(
            "A legacy combined storyboard/keyframe binding is treated as a concrete keyframe. "
            "Choose the dedicated storyboard role when the picture only plans shots."
        )
    invalid_scope_entries = [
        (f"{entry['label']} ({binding['shot_scope']})", binding["shot_scope"])
        for entry in entries
        for binding in entry["bindings"]
        if binding.get("shot_scope")
        and _shot_numbers_from_scope(binding["shot_scope"], shot_count) is None
    ]
    invalid_scopes = [display for display, _scope in invalid_scope_entries]
    if invalid_scopes:
        invalid_numbered_scopes = [
            display
            for display, scope in invalid_scope_entries
            if _looks_like_numbered_shot_scope(scope)
        ]
        if invalid_numbered_scopes:
            raise ValueError(
                "Role shot_scope refers to a Shot that does not exist in the current plan: "
                + ", ".join(invalid_numbered_scopes)
                + ". Update the scope or connect the intended Shot chain."
            )
        warnings.append(
            "These role shot_scope values cannot be mapped to the current Shot plan and remain "
            "global constraints: " + ", ".join(invalid_scopes) + ". Use compact forms such as "
            "'2', '1,3', '1-3', or 'all'."
        )
    return warnings


def _audio_context_scope_warnings(entries: list[dict], shot_count: int) -> list[str]:
    invalid = [
        (f"{entry['label']} ({entry['shot_scope']})", entry["shot_scope"])
        for entry in entries
        if entry.get("shot_scope")
        and _shot_numbers_from_scope(entry["shot_scope"], shot_count) is None
    ]
    numbered = [
        display
        for display, scope in invalid
        if _looks_like_numbered_shot_scope(scope)
    ]
    if numbered:
        raise ValueError(
            "Audio shot_scope refers to a Shot that does not exist in the current plan: "
            + ", ".join(numbered)
            + ". Update the scope or connect the intended Shot chain."
        )
    if not invalid:
        return []
    return [
        "These audio shot_scope values cannot be mapped to the current Shot plan and remain "
        "global constraints: "
        + ", ".join(display for display, _scope in invalid)
        + ". Use compact forms such as '2', '1,3', '1-3', or 'all'."
    ]


def _context_mode_report(
    decision: ModeDecision,
    entries: list[dict],
    audio_use: str,
    task_types: list[str],
    warnings: list[str],
) -> str:
    lines = [
        f"Recommended mode: {decision.mode}",
        f"Checkpoint: {decision.checkpoint}",
        f"Why: {decision.reason}",
        "Visual role source: connected reference_context (authoritative).",
        f"Resolved audio role: {audio_use}",
    ]
    if decision.mode == "Ref2VA":
        lines.append(f"Task-type prefix: [{' + '.join(task_types)}]")
        lines.append(
            "Ref2VA limits: up to 9 images, 3 videos, 3 audio clips, and 12 media files total; "
            "video/audio clips are 2-15 seconds and total up to 15 seconds."
        )
    else:
        lines.append(f"Frame routing: connected context declares {decision.mode} endpoints.")
    lines.append("Authoritative visual bindings:")
    for entry in entries:
        bindings = ", ".join(
            f"{binding['role']} [{binding['retention']}]" for binding in entry["bindings"]
        )
        lines.append(f"- {entry['label']}: {bindings} -> {entry['h3_node']}.{entry['h3_input']}")
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("Warnings: none.")
    return "\n".join(lines)


def _mode_report(
    decision: ModeDecision,
    goal: str,
    image_use: str,
    video_use: str,
    audio_use: str,
    assets: list[Asset],
    warnings: list[str],
) -> str:
    lines = [
        f"Recommended mode: {decision.mode}",
        f"Checkpoint: {decision.checkpoint}",
        f"Why: {decision.reason}",
        f"Resolved image role: {image_use}",
        f"Resolved video role: {video_use}",
        f"Resolved audio role: {audio_use}",
    ]
    if decision.mode == "Ref2VA":
        lines.append(f"Task-type prefix: [{' + '.join(_task_types(goal, image_use, video_use, audio_use))}]")
        lines.append(
            "Ref2VA limits: up to 9 images, 3 videos, 3 audio clips, and 12 media files total; "
            "video/audio clips are 2-15 seconds and total up to 15 seconds."
        )
    else:
        route = {
            "T2VA": "no image anchor",
            "I2VA": "Picture 1 is the exact first frame",
            "FL2VA": "Picture 1 is first and Picture 2 is last",
            "L2VA": "Picture 1 is the exact last frame",
        }[decision.mode]
        lines.append(f"Frame routing: {route}.")
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("Warnings: none.")
    return "\n".join(lines)


def _rewrite_request(
    decision: ModeDecision,
    draft: str,
    report: str,
    target_description: str,
    reference_assets: str,
) -> str:
    if decision.mode == "Ref2VA":
        format_rules = """Return exactly these six English sections in order:
subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music.
Use stable <Subject N>, <Picture N>, <Video N>, and <Audio N> labels. In summary, use only applicable fixed task types: keyframe completion, reference generation, video editing, video continuation, audio reuse, audio reference. In retention_analysis, use only fully_preserved, partially_preserved, attribute_transfer, or weak_reference for visual references, and fully_copy, partially_copy, reference, or weak_reference for audio. A fully_copy source is the complete final track: do not add, replace, remix, or synthesize any dialogue, lyrics, ambience, effects, or music, and cite it in every applicable audio section. Make detailed_description explicit and chronological, normally 350-500 English words for generation tasks. Establish style before [Shot 1]."""
    else:
        format_rules = """Return the applicable image-alignment instruction first when this is I2VA, FL2VA, or L2VA, followed by one blank line. Then return exactly these three fields: integrated_multimodal_description, overall_soundscape, non_diegetic_music. Start the body with [Shot 1]. Put no timestamp on Shot 1; later cuts use [Shot N] At MM:SS.mmm with strictly increasing times inside the duration."""

    return f"""Rewrite the user's rough plan as a production-ready MiniMax H3 {decision.mode} prompt.

MODE DECISION
{report}

USER INTENT
{_clean(target_description, 'No additional intent supplied.')}

REFERENCE INVENTORY
{reference_assets.strip() or 'Use the reference labels and roles already present in the draft.'}

FORMAT REQUIREMENTS
{format_rules}
Write camera motion naturally as motion type plus meaningful amplitude and speed. Assign stable (S1), (S2), and later IDs only to actual vocal sources, in first-vocal-event order. Put only the user's exact spoken or sung words inside <d>[Language] ...</d>. For voice-over, use the exact phrase `says in an off-screen voiceover` and immediately state after the <d> block that the on-screen character's lips remain completely closed. If one utterance crosses a cut, put <scenetrans> at both connecting points and explicitly say its audio continues across the cut; use <cutoff> when speech is truncated by the video ending. Preserve dialogue, lyrics, and visible text in their original language. Keep ambience and physical sounds in overall_soundscape. Put only audience-only score in non_diegetic_music. Do not invent unsupported reference assets or change the user's requested identity, action, dialogue, or endpoint frames. Output only the finished H3 prompt.

STRUCTURED DRAFT TO EXPAND
{draft}"""


class MiniMaxH3Shot:
    """Build one validated shot and optionally append it to an earlier shot chain."""

    CATEGORY = "MiniMax H3/Prompting"
    FUNCTION = "append_shot"
    RETURN_TYPES = (SHOT_PLAN_TYPE, "STRING")
    RETURN_NAMES = ("shot_plan", "shot_plan_preview")
    OUTPUT_TOOLTIPS = (
        "Connect to the next MiniMax H3 Shot.previous_shots input. For a role-aware reference workflow, connect the final shot to Target Timing.shot_plan; connect directly to Prompt Guide.shot_plan only in the simple legacy timing path.",
        "Readable list of the accumulated float ranges and descriptions. Connect to a text viewer to inspect the complete chain.",
    )
    DESCRIPTION = (
        "Defines one H3 shot with an exact float time range. Chain nodes in playback order, then "
        "connect the final shot_plan to Target Timing for a reference-context workflow, or directly "
        "to Prompt Guide when no context feeds back into it."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_time": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 15.0,
                        "step": 0.001,
                        "tooltip": (
                            "Start time in seconds. Shot 1 must start at 0.000. Every later shot must "
                            "start exactly when the previous connected shot ends; this becomes its H3 cut timestamp."
                        ),
                    },
                ),
                "end_time": (
                    "FLOAT",
                    {
                        "default": 6.0,
                        "min": 0.001,
                        "max": 15.0,
                        "step": 0.001,
                        "tooltip": (
                            "End time in seconds; it must be greater than start_time. The final node's "
                            "end_time becomes the requested duration when connected to Target Timing "
                            "or directly to Prompt Guide; the resolved timeline may extend slightly "
                            "to native H3's frame grid."
                        ),
                    },
                ),
                "description": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": "",
                        "placeholder": "Example: A medium shot establishes the woman entering the rainy market.",
                        "tooltip": (
                            "Describe what is visible and what changes during only this time range: "
                            "composition, subject action, setting, lighting, and ending state."
                        ),
                    },
                ),
                "camera_direction": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Example: Slow, small-amplitude tracking move from the left.",
                        "tooltip": (
                            "Optional camera behavior for this shot. State motion type, meaningful "
                            "speed/amplitude, or 'static camera'. It overrides no other shot."
                        ),
                    },
                ),
                "transition": (
                    SHOT_TRANSITIONS,
                    {
                        "default": SHOT_TRANSITIONS[0],
                        "tooltip": (
                            "How this shot begins when it follows another shot. Shot 1 ignores this setting. "
                            "A direct cut is the clearest default; use dissolves/fades/wipes only when intended."
                        ),
                    },
                ),
            },
            "optional": {
                "previous_shots": (
                    SHOT_PLAN_TYPE,
                    {
                        "tooltip": (
                            "Connect shot_plan from the preceding MiniMax H3 Shot. Leave disconnected "
                            "only for Shot 1. Chains are validated to prevent gaps, overlaps, or reversed ranges."
                        )
                    },
                )
            },
        }

    def append_shot(
        self,
        start_time: float,
        end_time: float,
        description: str,
        camera_direction: str,
        transition: str,
        previous_shots=None,
    ):
        shots = _shots_from_plan(previous_shots) if previous_shots is not None else []
        candidate = {
            "start_time": float(start_time),
            "end_time": float(end_time),
            "description": description,
            "camera_direction": camera_direction,
            "transition": transition,
        }
        validated = _shots_from_plan({"shots": [*shots, candidate]})
        return ({"version": 1, "shots": validated}, _shot_plan_preview(validated))


class MiniMaxH3TargetTiming:
    """Resolve target duration before both reference preparation and prompt writing."""

    CATEGORY = "MiniMax H3/Prompting"
    FUNCTION = "resolve"
    RETURN_TYPES = (TIMING_CONTEXT_TYPE, "INT", "STRING")
    RETURN_NAMES = ("timing_context", "h3_length", "timing_report")
    OUTPUT_TOOLTIPS = (
        "Connect to Prompt Guide.timing_context. This keeps timing upstream when a final visual reference_context also feeds the Guide.",
        "Connect to every video Visual Reference.h3_length and the official H3 conditioning node.length input.",
        "Resolved requested duration, native 17k+5 length, effective duration, and whether a connected Shot chain overrode the duration widget.",
    )
    DESCRIPTION = (
        "Resolves the target duration once, before visual-reference video trimming and prompt writing. "
        "Use it whenever the final Visual Reference context is connected to the Prompt Guide, so the "
        "workflow remains a directed graph instead of feeding Guide.h3_length back into an upstream node."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration_seconds": (
                    "FLOAT",
                    {
                        "default": 6.0,
                        "min": 4.0,
                        "max": 15.0,
                        "step": 0.01,
                        "tooltip": (
                            "Target playback duration when no Shot chain is connected. It is rounded "
                            "upward to native H3's 17k+5 frame grid at 24 FPS."
                        ),
                    },
                )
            },
            "optional": {
                "shot_plan": (
                    SHOT_PLAN_TYPE,
                    {
                        "tooltip": (
                            "Connect the final MiniMax H3 Shot node. Its last end_time replaces "
                            "duration_seconds and the complete plan travels to the Prompt Guide inside "
                            "timing_context."
                        )
                    },
                )
            },
        }

    def resolve(self, duration_seconds: float, shot_plan=None):
        context = _build_timing_context(duration_seconds, shot_plan=shot_plan)
        source = (
            f"connected Shot chain ({len(context['shots'])} shot(s))"
            if context["source"] == "shot_plan"
            else "duration_seconds"
        )
        report = (
            f"Timing source: {source}. Requested duration: "
            f"{context['requested_duration']:.3f}s. Native duration: {context['h3_length']} "
            f"frames at {H3_FPS} FPS = {context['effective_duration']:.3f}s."
        )
        return (context, context["h3_length"], report)


class MiniMaxH3PromptGuide:
    """Select an H3 mode and turn a rough audiovisual idea into guided prompts."""

    CATEGORY = "MiniMax H3/Prompting"
    FUNCTION = "build"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("h3_prompt", "rewrite_request", "mode_report", "h3_length")
    OUTPUT_TOOLTIPS = (
        "Structured pre-LLM H3 draft. Connect it to MiniMax H3 Prompt Enhancer.manual_prompt for the recommended detailed, source-grounded rewrite; use it directly only after reviewing every reference and timeline detail.",
        "Self-contained instructions for a different LLM node. It includes the selected H3 format, fixed labels, rules, and this structured draft.",
        "Read this first when results look wrong. It shows the chosen H3 mode/checkpoint, resolved media roles, task prefix, input limits, and conflicts to fix.",
        "Native ComfyUI H3 frame count on the required 17k+5 grid at 24 FPS. Connect it to the official H3 conditioning node when using the Guide alone. With a Visual Reference context feeding back into this Guide, use Target Timing.h3_length upstream instead and never create a backward cycle.",
    )
    DESCRIPTION = (
        "Start here: describe the intended video and explain what each supplied image, video, or audio file does. "
        "The node selects T2VA/I2VA/FL2VA/L2VA/Ref2VA, prepares the correct prompt structure, "
        "and reports mismatched choices before generation. For role-chained media, connect the final "
        "Visual Reference.reference_context here as the authoritative role source."
    )

    @classmethod
    def INPUT_TYPES(cls):
        multiline = {"multiline": True, "dynamicPrompts": False}
        return {
            "required": {
                "what_do_you_want": (
                    GOALS,
                    {
                        "default": AUTO_GOAL,
                        "tooltip": (
                            "Start here. Auto is recommended: it chooses the H3 mode from the image/video/audio roles below. "
                            "Use Text only for no media; Animate/Reach/Connect when images are exact endpoint frames; "
                            "Generate from references for appearance/style guidance; Edit only when changing an existing video; "
                            "Continue when adding new footage after it; Transfer motion when one subject must perform another asset's action."
                        ),
                    },
                ),
                "target_description": (
                    "STRING",
                    {
                        **multiline,
                        "default": "",
                        "placeholder": "Example: Replace the blue car in the source video with a red vintage coupe while preserving the road, camera movement, and lighting.",
                        "tooltip": (
                            "Write the actual creative request, not an instruction such as 'describe the video.' Include the target subject, "
                            "visible action, setting, requested edit, and final state. Example: 'Replace the blue car with a red vintage coupe; "
                            "keep the original road, lighting, and camera movement.' Put exact spoken words in the dialogue field."
                        ),
                    },
                ),
                "how_images_are_used": (
                    IMAGE_USES,
                    {
                        "default": NO_IMAGE,
                        "tooltip": (
                            "Choose the image's job. First frame = exact image at 0.00s (I2VA). Last frame = exact final composition (L2VA). "
                            "First + last = continuous path between two exact frames (FL2VA). Appearance/scene/style guides generation but is not an exact frame. "
                            "Concrete keyframe anchors an exact internal composition and uses keyframe completion. Storyboard only plans viewpoint, placement, or shot order "
                            "and uses reference generation. Motion target is the subject that receives motion from a video. The combined storyboard/keyframe choice remains only for old workflows. "
                            "This whole dropdown is a legacy one-role shortcut and is ignored when reference_context is connected."
                        ),
                    },
                ),
                "how_video_is_used": (
                    VIDEO_USES,
                    {
                        "default": NO_VIDEO,
                        "tooltip": (
                            "Choose the video's job. Direct edit modifies its existing frames/content. Continue creates new footage after its ending. "
                            "Transfer motion copies action, pose timing, or trajectory to another subject without editing the source clip. "
                            "Camera/cuts/rhythm borrows only temporal structure. Subject/scene/style reuses visible reference content. All use Ref2VA. "
                            "This whole dropdown is a legacy one-role shortcut and is ignored when reference_context is connected."
                        ),
                    },
                ),
                "how_audio_is_used": (
                    AUDIO_USES,
                    {
                        "default": NO_AUDIO,
                        "tooltip": (
                            "Choose whether H3 copies or only imitates the audio. Complete reuse copies the whole signal 1:1. Partial reuse copies selected time/layers "
                            "and permits a new mix. Reference regenerates voice timbre, music style, beat, dialogue content, or sound texture without copying samples. "
                            "Broad mood is weak reference. Audio cannot be the only Ref2VA media input; add an image or video."
                        ),
                    },
                ),
                "reference_fidelity": (
                    FIDELITIES,
                    {
                        "default": AUTO_FIDELITY,
                        "tooltip": (
                            "Controls Ref2VA retention markers. Auto chooses per role. Fully preserve keeps defined identity/composition. Partly preserve allows visible edits. "
                            "Transfer applies attributes or motion to a different identifiable target. Weak inspiration keeps only broad style/category/atmosphere. "
                            "A motion source uses attribute_transfer only when the target-picture role is also selected; otherwise it is reported as incomplete and kept non-transfer."
                        ),
                    },
                ),
                "reference_assets": (
                    "STRING",
                    {
                        **multiline,
                        "default": "",
                        "placeholder": "Picture 1: target character appearance\nVideo 1: walking motion to transfer\nAudio 1: voice timbre",
                        "tooltip": (
                            "Inventory the actual downstream media in label order, one per line: 'Picture 1: red ceramic robot', "
                            "'Video 1: dancer performing a spin', 'Audio 1: calm voice timbre'. Angle brackets are optional. "
                            "Use Subject N only in a legacy text-only workflow. Unlabelled lines become extra notes. "
                            "When reference_context is connected, its visual labels/roles are authoritative; matching Picture/Video lines here add descriptions only, "
                            "unmatched visual labels and manual Subject rows are ignored, and Audio lines remain active."
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
                            "Requested playback duration, from 4 to 15 seconds. Native ComfyUI rounds it upward to a 17k+5 frame count at 24 FPS; "
                            "the effective rounded time becomes the exact landing time for last-frame tasks. Every later-shot cut must be earlier than the requested end."
                        ),
                    },
                ),
                "visual_style": (
                    "STRING",
                    {
                        "default": AUTO_VISUAL_STYLE,
                        "tooltip": (
                            "Auto derives the treatment from endpoint/reference pictures and the written intent, so it does not force live action onto animation or artwork. "
                            "Enter an explicit override such as 'cinematic, live-action', '2D animation', '3D CG', 'claymation', or 'vintage film' only when you want that change."
                        ),
                    },
                ),
                "shot_and_timing_plan": (
                    "STRING",
                    {
                        **multiline,
                        "default": "",
                        "placeholder": "Shot 1, 00:00-00:02.500: establish the room.\nShot 2, cut at 00:02.500: close-up of the letter.\nShot 3, cut at 00:04.250: wide ending shot.",
                        "tooltip": (
                            "Advanced legacy fallback when no MiniMax H3 Shot chain is connected. Give Shot 1's content but no H3 cut timestamp; for later shots provide the exact cut time. "
                            "Example: 'Shot 1, 00:00-00:02.500: medium entrance. Shot 2, cut at 00:02.500: close-up. "
                            "Shot 3, cut at 00:04.250: wide ending.' The guide converts later cuts to '[Shot 2] At 00:02.500, ...'. "
                            "Times must strictly increase and remain inside the duration. A connected shot_plan takes priority."
                        ),
                        "advanced": True,
                    },
                ),
                "camera_direction": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Example: The camera pushes in with small amplitude at slow speed toward the letter.",
                        "tooltip": (
                            "Describe only intended camera behavior. Use movement type plus meaningful range/speed: static, push/pull, pan, truck, tilt, pedestal, arc, tracking, shake, POV, or roll. "
                            "Example: 'pushes in with small amplitude at slow speed.' For different motion per shot, put each instruction in the shot plan instead."
                        ),
                    },
                ),
                "dialogue_lyrics_and_visible_text": (
                    "STRING",
                    {
                        **multiline,
                        "default": "",
                        "placeholder": "S1, young woman, English: I get off at the next station.\nVisible sign: 营业中",
                        "tooltip": (
                            "Enter exact words and identify who says them, their language, voice, and shot when known. Example: 'Shot 2, S1, quiet young woman, French: Je reviens demain.' "
                            "The enhancer converts speech to <d>[French] Je reviens demain.</d> and keeps S1 stable. List visible signs/subtitles separately; their original text is preserved in double quotes."
                        ),
                    },
                ),
                "overall_soundscape": (
                    "STRING",
                    {
                        **multiline,
                        "default": "",
                        "placeholder": "Steady rain, low room tone, wet footsteps, paper rustle, and one window latch click.",
                        "tooltip": (
                            "Describe sounds that exist in the scene: ambience, footsteps, impacts, mechanisms, fabric, breathing, laughter, and other physical/non-verbal sounds. "
                            "Do not repeat dialogue, singing, or audience-only score here. Use complete silence only when intentionally requested."
                        ),
                    },
                ),
                "non_diegetic_music": (
                    "STRING",
                    {
                        **multiline,
                        "default": "N/A",
                        "tooltip": (
                            "Audience-only background score that characters cannot hear. Describe instrumentation, tempo/rhythm, and volume changes: "
                            "'Sparse piano at a slow tempo, joined by sustained low strings, then fading.' Use N/A for no score. "
                            "A radio, performer, phone, or instrument audible in the scene belongs in the shot description instead."
                        ),
                    },
                ),
            },
            "optional": {
                "shot_plan": (
                    SHOT_PLAN_TYPE,
                    {
                        "tooltip": (
                            "Recommended for multiple shots: connect the final MiniMax H3 Shot node. "
                            "The guide writes real [Shot N] markers, validates contiguous float ranges, "
                            "and uses the last end_time as the target duration. This overrides the manual shot field. "
                            "When timing_context is connected, its embedded Shot plan takes priority instead."
                        )
                    },
                ),
                "timing_context": (
                    TIMING_CONTEXT_TYPE,
                    {
                        "tooltip": (
                            "Recommended with chained references: connect MiniMax H3 Target Timing.timing_context. "
                            "It overrides duration_seconds and carries any connected Shot plan. Its upstream h3_length "
                            "can safely feed video Visual References and native H3 without creating a graph cycle."
                        )
                    },
                ),
                "reference_context": (
                    REFERENCE_CONTEXT_TYPE,
                    {
                        "tooltip": (
                            "Connect the final MiniMax H3 Enhancer Visual Reference.reference_context. "
                            "It becomes authoritative for mode, labels, per-asset roles, Subject grouping, retention, "
                            "shot scope, and native routes; legacy image/video role dropdowns no longer create generic Subjects. "
                            "Fan this same final context out to Prompt Enhancer.reference_context."
                        )
                    },
                ),
                "audio_context": (
                    AUDIO_REFERENCE_CONTEXT_TYPE,
                    {
                        "tooltip": (
                            "Connect the final MiniMax H3 Reference Sheet Audio Reference.audio_context. "
                            "Its saved descriptions, actual audio relationships, labels, Shot scopes, and "
                            "native ref_audio_N routes replace the legacy audio dropdown/inventory path. "
                            "A visual reference_context is also required because H3 does not accept audio "
                            "as the only Ref2VA media."
                        )
                    },
                ),
            },
        }

    def build(
        self,
        what_do_you_want: str,
        target_description: str,
        how_images_are_used: str,
        how_video_is_used: str,
        how_audio_is_used: str,
        reference_fidelity: str,
        reference_assets: str,
        duration_seconds: float,
        visual_style: str,
        shot_and_timing_plan: str,
        camera_direction: str,
        dialogue_lyrics_and_visible_text: str,
        overall_soundscape: str,
        non_diegetic_music: str,
        shot_plan=None,
        timing_context=None,
        reference_context=None,
        audio_context=None,
    ):
        timing_is_connected = timing_context is not None
        if timing_is_connected:
            timing = _validated_timing_context(timing_context)
            connected_shots = timing["shots"] if timing["source"] == "shot_plan" else []
            manual_shots = []
        else:
            timing = _build_timing_context(
                duration_seconds,
                shot_plan=shot_plan,
                manual_shot_plan=shot_and_timing_plan,
            )
            connected_shots = timing["shots"] if timing["source"] == "shot_plan" else []
            manual_shots = (
                timing["shots"] if timing["source"] == "manual_shot_plan" else []
            )
        planned_shots = timing["shots"]
        requested_duration = timing["requested_duration"]
        h3_length = timing["h3_length"]
        effective_duration = timing["effective_duration"]
        structured_shots = _extend_final_shot(planned_shots, effective_duration)
        final_shot_number = len(structured_shots) or 1
        parsed_assets, notes = parse_assets(reference_assets)
        sheet_audio_entries = (
            audio_reference_entries(audio_context) if audio_context is not None else []
        )
        if sheet_audio_entries and reference_context is None:
            raise ValueError(
                "Reference Sheet audio_context also needs a visual reference_context because "
                "MiniMax H3 does not accept standalone audio as the only Ref2VA media."
            )
        effective_audio_use = (
            sheet_audio_entries[0]["use"] if sheet_audio_entries else how_audio_is_used
        )
        context_entries = (
            reference_entries(reference_context) if reference_context is not None else []
        )
        if reference_context is not None:
            decision = _context_mode_decision(reference_context, context_entries)
            effective_image_use = NO_IMAGE
            effective_video_use = NO_VIDEO
            task_types = _context_task_types(context_entries, effective_audio_use)
            context_audio_assets = _audio_context_assets(sheet_audio_entries)
            audio_assets = context_audio_assets or [
                asset for asset in parsed_assets if asset.kind == "Audio"
            ]
            assets = _assets_with_defaults(
                audio_assets,
                NO_IMAGE,
                NO_VIDEO,
                effective_audio_use,
            )
            _validate_active_asset_labels(assets, NO_IMAGE, NO_VIDEO, effective_audio_use)
            warning_assets = [asset for asset in parsed_assets if asset.kind != "Audio"]
            warning_assets.extend(audio_assets)
            warnings = _context_warnings(
                decision,
                what_do_you_want,
                how_images_are_used,
                how_video_is_used,
                effective_audio_use,
                reference_fidelity,
                warning_assets,
                context_entries,
                target_description,
                len(structured_shots) or 1,
                dialogue_lyrics_and_visible_text,
                h3_length,
                timing_is_connected,
            )
            warnings.extend(
                _audio_context_scope_warnings(
                    sheet_audio_entries,
                    len(structured_shots) or 1,
                )
            )
            if sheet_audio_entries and (
                how_audio_is_used != NO_AUDIO
                or any(asset.kind == "Audio" for asset in parsed_assets)
            ):
                warnings.append(
                    "The legacy audio role dropdown and Audio inventory rows are ignored because "
                    "audio_context is connected; edit the Reference Sheet Audio Reference nodes instead."
                )
            context_notes = _context_reference_notes(context_entries, parsed_assets)
            if decision.mode != "Ref2VA":
                notes = [*notes, *context_notes]
            rewrite_inventory = reference_inventory(context_entries)
            if sheet_audio_entries:
                rewrite_inventory += "\n\n" + audio_reference_inventory(sheet_audio_entries)
            context_labels = {entry["label"] for entry in context_entries}
            supplemental_inventory = [
                f"{asset.label}: {asset.description}"
                for asset in parsed_assets
                if asset.kind in {"Picture", "Video"} and asset.label in context_labels
            ]
            if not sheet_audio_entries:
                supplemental_inventory.extend(
                    f"{asset.label}: {asset.description}"
                    for asset in parsed_assets
                    if asset.kind == "Audio" and effective_audio_use != NO_AUDIO
                )
            if supplemental_inventory:
                rewrite_inventory += (
                    "\n\nUSER-SUPPLIED LABEL DESCRIPTIONS\n"
                    + "\n".join(supplemental_inventory)
                )
        else:
            task_types = None
            effective_image_use, effective_video_use = _resolve_roles(
                what_do_you_want,
                how_images_are_used,
                how_video_is_used,
            )
            decision = choose_mode(
                what_do_you_want,
                effective_image_use,
                effective_video_use,
                effective_audio_use,
            )
            assets = _assets_with_defaults(
                parsed_assets,
                effective_image_use,
                effective_video_use,
                effective_audio_use,
            )
            _validate_active_asset_labels(
                assets,
                effective_image_use,
                effective_video_use,
                effective_audio_use,
            )
            warnings = _warnings(
                decision,
                what_do_you_want,
                effective_image_use,
                effective_video_use,
                effective_audio_use,
                assets,
                target_description,
                reference_fidelity,
                len(structured_shots) or 1,
                dialogue_lyrics_and_visible_text,
            )
            rewrite_inventory = reference_assets

        declared_asset_keys = {(asset.kind, asset.number) for asset in parsed_assets}
        declared_asset_keys.update(
            ("Audio", number) for number, _entry in enumerate(sheet_audio_entries, start=1)
        )
        placeholder_labels = [
            asset.label
            for asset in assets
            if asset.kind in {"Picture", "Video", "Audio"}
            and (asset.kind, asset.number) not in declared_asset_keys
        ]
        if placeholder_labels:
            warnings.append(
                "The selected roles required draft placeholder label(s) "
                + ", ".join(placeholder_labels)
                + ". A placeholder is not an attached file; connect real media to every corresponding "
                "native H3 input."
            )
        if reference_context is not None:
            report = _context_mode_report(
                decision,
                context_entries,
                effective_audio_use,
                task_types,
                warnings,
            )
            if sheet_audio_entries:
                report += "\n" + audio_reference_inventory(sheet_audio_entries)
        else:
            report = _mode_report(
                decision,
                what_do_you_want,
                effective_image_use,
                effective_video_use,
                effective_audio_use,
                assets,
                warnings,
            )
        report += (
            f"\nNative duration: {h3_length} frames at {H3_FPS} FPS = "
            f"{effective_duration:.3f} seconds."
        )
        if not math.isclose(requested_duration, effective_duration, abs_tol=0.0005):
            report += (
                f" Requested {requested_duration:.3f} seconds was snapped upward to ComfyUI's "
                f"17k+5 frame grid; connect h3_length={h3_length} to the native H3 node."
            )
        if timing_is_connected:
            source = (
                f"embedded {len(structured_shots)}-shot plan"
                if connected_shots
                else "duration_seconds"
            )
            report += (
                f"\nTarget Timing: using its {source}; the Guide's duration and direct shot inputs "
                "are legacy fallbacks."
            )
            if shot_plan is not None:
                report += "\nDirect shot_plan input: ignored because timing_context is connected."
            if shot_and_timing_plan.strip():
                report += "\nManual shot fallback: ignored because timing_context is connected."
        elif connected_shots:
            report += (
                f"\nShot chain: {len(structured_shots)} shot(s), 00:00.000–"
                f"{_format_timestamp(requested_duration)} planned; final end_time overrides "
                "duration_seconds."
            )
            if shot_and_timing_plan.strip():
                report += "\nManual shot fallback: ignored because shot_plan is connected."
        elif manual_shots:
            report += (
                f"\nManual shot plan: parsed and validated {len(structured_shots)} shot(s), "
                f"00:00.000–{_format_timestamp(requested_duration)} planned."
            )
            if not math.isclose(requested_duration, duration_seconds, abs_tol=0.0005):
                report += " Its explicit final end overrides duration_seconds."
        if structured_shots and not math.isclose(
            requested_duration, effective_duration, abs_tol=0.0005
        ):
            report += (
                f"\nFinal [Shot {final_shot_number}] is extended through "
                f"{_format_timestamp(effective_duration)} so the described timeline reaches the "
                "native playback end."
            )

        if decision.mode == "Ref2VA":
            if reference_context is not None:
                visual_items = _context_reference_items(context_entries, parsed_assets)
                audio_items = (
                    _audio_context_reference_items(sheet_audio_entries)
                    if sheet_audio_entries
                    else _build_reference_items(
                        assets,
                        NO_IMAGE,
                        NO_VIDEO,
                        effective_audio_use,
                    )
                )
                items = [*visual_items, *audio_items]
            else:
                items = _build_reference_items(
                    assets,
                    effective_image_use,
                    effective_video_use,
                    effective_audio_use,
                )
            draft = _reference_prompt(
                what_do_you_want,
                effective_duration,
                items,
                target_description,
                visual_style,
                shot_and_timing_plan,
                camera_direction,
                dialogue_lyrics_and_visible_text,
                overall_soundscape,
                non_diegetic_music,
                reference_fidelity,
                effective_image_use,
                effective_video_use,
                effective_audio_use,
                notes,
                structured_shots,
                final_shot_number,
                task_types,
            )
        else:
            draft = _base_prompt(
                decision.mode,
                effective_duration,
                target_description,
                visual_style,
                shot_and_timing_plan,
                camera_direction,
                dialogue_lyrics_and_visible_text,
                overall_soundscape,
                non_diegetic_music,
                notes,
                structured_shots,
                final_shot_number,
            )

        rewrite = _rewrite_request(
            decision,
            draft,
            report,
            target_description,
            rewrite_inventory,
        )
        return (draft, rewrite, report, h3_length)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PromptGuide": MiniMaxH3PromptGuide,
    "MiniMaxH3Shot": MiniMaxH3Shot,
    "MiniMaxH3TargetTiming": MiniMaxH3TargetTiming,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PromptGuide": "MiniMax H3 Prompt Guide (Legacy)",
    "MiniMaxH3Shot": "MiniMax H3 Shot (Legacy)",
    "MiniMaxH3TargetTiming": "MiniMax H3 Target Timing (Legacy)",
}
