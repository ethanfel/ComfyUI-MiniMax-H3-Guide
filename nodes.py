"""Guided MiniMax H3 prompt preparation for ComfyUI.

The node intentionally has no ComfyUI imports, which keeps the decision and
formatting logic easy to test and lets ComfyUI discover it through the normal
NODE_CLASS_MAPPINGS interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


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
MOTION_TARGET_IMAGE = "Target subject for motion transfer"

IMAGE_USES = [
    NO_IMAGE,
    FIRST_IMAGE,
    LAST_IMAGE,
    FIRST_LAST_IMAGES,
    APPEARANCE_IMAGE,
    STORYBOARD_IMAGE,
    MOTION_TARGET_IMAGE,
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

_ASSET_RE = re.compile(
    r"^\s*<?(Subject|Picture|Video|Audio)\s*(\d+)>?\s*(?::|=|\s-\s)\s*(.+?)\s*$",
    re.IGNORECASE,
)


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


def _clean(text: str, fallback: str = "") -> str:
    value = " ".join((text or "").strip().split())
    return value or fallback


def _sentence(text: str, fallback: str = "") -> str:
    value = _clean(text, fallback)
    if value and value[-1] not in ".!?":
        value += "."
    return value


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
        key = (kind, number)
        if key in seen:
            notes.append(f"Additional note for <{kind} {number}>: {match.group(3).strip()}")
            continue
        seen.add(key)
        assets.append(Asset(kind, number, match.group(3).strip()))
    return assets, notes


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

    if goal == EDIT_GOAL or video_use == EDIT_VIDEO:
        task_types.append("video editing")
    elif goal == CONTINUE_GOAL or video_use == CONTINUE_VIDEO:
        task_types.append("video continuation")

    if image_use in {FIRST_IMAGE, LAST_IMAGE, FIRST_LAST_IMAGES, STORYBOARD_IMAGE}:
        task_types.append("keyframe completion")

    reference_generation = (
        goal in {REFERENCE_GOAL, MOTION_GOAL}
        or image_use in {APPEARANCE_IMAGE, MOTION_TARGET_IMAGE}
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
    picture_numbers = {asset.number for asset in result if asset.kind == "Picture"}
    required_pictures = range(1, 3) if image_use == FIRST_LAST_IMAGES else range(1, 2)
    if image_use != NO_IMAGE:
        result.extend(
            Asset("Picture", number, "the supplied reference image")
            for number in required_pictures
            if number not in picture_numbers
        )
    if video_use != NO_VIDEO and not any(
        asset.kind == "Video" and asset.number == 1 for asset in result
    ):
        result.append(Asset("Video", 1, "the supplied reference video"))
    if audio_use != NO_AUDIO and not any(
        asset.kind == "Audio" and asset.number == 1 for asset in result
    ):
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

    if video_use in {EDIT_VIDEO, CONTINUE_VIDEO} and not items and videos:
        source_role = "edited" if video_use == EDIT_VIDEO else "continued"
        item_role = "edited visible content" if video_use == EDIT_VIDEO else "visible content"
        items.append(
            ReferenceItem(
                "<Subject 1>",
                "Subject",
                item_role,
                f"the principal visible subject or scene from {videos[0].label} that is {source_role} "
                "while remaining identifiable",
            )
        )

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
            elif image_use == STORYBOARD_IMAGE:
                role = "storyboard or concrete keyframe"
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


def _definition_line(item: ReferenceItem) -> str:
    description = _clean(item.description)
    if item.kind == "Subject":
        return _sentence(f"{item.label} is {description}")
    if item.kind == "Picture":
        if item.role == "first frame":
            return _sentence(f"{item.label} is the first frame of [Shot 1], showing {description}")
        if item.role == "last frame":
            return _sentence(f"{item.label} is the target video's final frame, showing {description}")
        if item.role == "storyboard or concrete keyframe":
            return _sentence(f"{item.label} is a storyboard or concrete keyframe anchor showing {description}")
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


def _visual_marker(fidelity: str, role: str) -> str:
    if role == "motion source":
        return "attribute_transfer"
    if fidelity == FULL_FIDELITY:
        return "fully_preserved"
    if fidelity == PARTIAL_FIDELITY:
        return "partially_preserved"
    if fidelity == TRANSFER_FIDELITY:
        return "attribute_transfer"
    if fidelity == WEAK_FIDELITY or role == "camera, cut, and rhythm structure":
        return "weak_reference"
    if role in {"edited visible content", "source video editing", "continuation starting point", "picture reference"}:
        return "partially_preserved"
    return "fully_preserved"


def _retention_line(item: ReferenceItem, fidelity: str, audio_use: str) -> str:
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
        return _sentence(f"{item.label}: {marker} - {explanation}")

    marker = _visual_marker(fidelity, item.role)
    if marker == "attribute_transfer":
        explanation = "the referenced characteristics or motion are transferred to the identifiable target subject"
    elif marker == "fully_preserved":
        explanation = "the defined appearance, composition, or reference role is retained"
    elif marker == "partially_preserved":
        explanation = "the reference remains identifiable while the requested changes are applied"
    else:
        explanation = "only the requested broad structure, category, style, or atmosphere is retained"

    if item.kind == "Subject":
        context = " (appears in [Shot 1])"
    elif item.kind == "Picture":
        context = f" ([Shot 1] {item.role})"
    else:
        context = f" ({item.role})"
    return _sentence(f"{item.label}{context}: {marker} - {explanation}")


def _role_sentences(items: list[ReferenceItem], video_use: str) -> str:
    parts: list[str] = []
    if video_use == EDIT_VIDEO:
        parts.append("The visible timeline begins from the source video and applies only the requested edits.")
    elif video_use == CONTINUE_VIDEO:
        parts.append("The new timeline begins from the ending state of the source video and continues naturally.")

    targets = [item.label for item in items if item.role == "motion target"]
    motions = [item.label for item in items if item.role == "motion source"]
    if targets and motions:
        parts.append(
            f"{targets[0]} keeps its visual identity while receiving the action, pose changes, timing, "
            f"and motion trajectory from {motions[0]}."
        )
    elif items:
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
) -> str:
    parts = [_sentence(target_description, "Create a coherent audiovisual scene from the supplied intent")]
    if role_sentences:
        parts.append(role_sentences)
    if shot_plan.strip():
        parts.append(_sentence(f"Shot and timing plan: {shot_plan}"))
    if camera_direction.strip():
        parts.append(_sentence(f"Camera direction: {camera_direction}"))
    if dialogue_and_text.strip():
        parts.append(
            _sentence(
                "Dialogue and visible-text requirements: "
                f"{dialogue_and_text} Preserve exact wording and language; put spoken words inside "
                "<d>[Language] ...</d> and visible text in double quotation marks"
            )
        )
    if reference_notes:
        parts.append(_sentence("Additional reference notes: " + " ".join(reference_notes)))
    return " ".join(parts)


def _section_sentence(text: str, fallback: str) -> str:
    value = _clean(text, fallback)
    return "N/A" if value.upper().rstrip(".") == "N/A" else _sentence(value)


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
            "aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) "
            f"aligns with the {duration_text}-second mark of the target video.\n\n"
        )
        anchor = "Begin at Picture 1 and show a continuous observable path that lands exactly on Picture 2."
    elif mode == "L2VA":
        instruction = (
            "How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) "
            f"aligns with the {duration_text}-second mark of the target video.\n\n"
        )
        anchor = "Infer a plausible preceding state and converge exactly on <Picture 1> at the end."
    else:
        instruction = ""
        anchor = ""

    body = _detail_body(
        target_description,
        visual_style,
        shot_plan,
        camera_direction,
        dialogue_and_text,
        anchor,
        reference_notes,
    )
    style = _clean(visual_style, "Cinematic")
    sound = _section_sentence(soundscape, "Use coherent ambience and synchronized physical sounds")
    score = _section_sentence(music, "N/A")
    return (
        f"{instruction}integrated_multimodal_description: [Shot 1] {style}. {body}\n\n"
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
) -> str:
    definitions = "\n".join(_definition_line(item) for item in items)
    types = " + ".join(_task_types(goal, image_use, video_use, audio_use))
    labels = ", ".join(item.label for item in items)

    if "video editing" in types:
        summary_start = "The target video is an edited version of <Video 1>."
    elif "video continuation" in types:
        summary_start = "The target video continues from the ending state of <Video 1>."
    else:
        summary_start = "The target video is generated from the defined reference relationships."
    summary = (
        f"[{types}] {summary_start} {_sentence(target_description)} "
        f"The reference roles use {labels}."
    )

    retention = "\n".join(_retention_line(item, fidelity, audio_use) for item in items)
    role_sentences = _role_sentences(items, video_use)
    detail = _detail_body(
        target_description,
        visual_style,
        shot_plan,
        camera_direction,
        dialogue_and_text,
        role_sentences,
        reference_notes,
    )
    style = _clean(visual_style, "cinematic audiovisual")
    sound = _section_sentence(soundscape, "Use coherent ambience and synchronized physical sounds")
    score = _section_sentence(music, "N/A")

    return (
        f"subject_definitions:\n{definitions}\n\n"
        f"summary:\n{summary}\n\n"
        f"retention_analysis:\n{retention}\n\n"
        "detailed_description:\n"
        f"The target video uses a {style} style and lasts {duration:.2f} seconds.\n"
        f"[Shot 1] {detail}\n\n"
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
) -> list[str]:
    warnings: list[str] = []
    counts = {kind: sum(asset.kind == kind for asset in assets) for kind in ("Picture", "Video", "Audio")}
    if counts["Picture"] > 9:
        warnings.append("Ref2VA accepts at most 9 reference images.")
    if counts["Video"] > 3:
        warnings.append("Ref2VA accepts at most 3 reference videos.")
    if counts["Audio"] > 3:
        warnings.append("Ref2VA accepts at most 3 reference audio clips.")
    if sum(counts.values()) > 12:
        warnings.append("Ref2VA accepts at most 12 media files in total.")
    if audio_use != NO_AUDIO and image_use == NO_IMAGE and video_use == NO_VIDEO and not (
        counts["Picture"] or counts["Video"]
    ):
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
    if decision.mode == "Ref2VA" and not assets:
        warnings.append("Ref2VA is selected but no image, video, or audio reference is described.")
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
Use stable <Subject N>, <Picture N>, <Video N>, and <Audio N> labels. In summary, use only applicable fixed task types: keyframe completion, reference generation, video editing, video continuation, audio reuse, audio reference. In retention_analysis, use only fully_preserved, partially_preserved, attribute_transfer, or weak_reference for visual references, and fully_copy, partially_copy, reference, or weak_reference for audio. Make detailed_description explicit and chronological, normally 350-500 English words for generation tasks. Establish style before [Shot 1]."""
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
Write camera motion naturally as motion type plus meaningful amplitude and speed. Keep speaker IDs stable. Put only exact spoken words inside <d>[Language] ...</d>; preserve dialogue, lyrics, and visible text in their original language. Keep ambience and physical sounds in overall_soundscape. Put only audience-only score in non_diegetic_music. Do not invent unsupported reference assets or change the user's requested identity, action, dialogue, or endpoint frames. Output only the finished H3 prompt.

STRUCTURED DRAFT TO EXPAND
{draft}"""


class MiniMaxH3PromptGuide:
    """Select an H3 mode and turn a rough audiovisual idea into guided prompts."""

    CATEGORY = "MiniMax H3/Prompting"
    FUNCTION = "build"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("h3_prompt", "rewrite_request", "mode_report")
    OUTPUT_TOOLTIPS = (
        "A valid structured draft that can be sent to H3 or edited further.",
        "A self-contained request for an LLM or H3 Context-IR-style rewrite step.",
        "The selected mode, checkpoint, reasoning, limits, and warnings.",
    )
    DESCRIPTION = (
        "Answers the key H3 workflow questions, selects T2VA/I2VA/FL2VA/L2VA/Ref2VA, "
        "and prepares a guide-compliant prompt draft plus an optional LLM rewrite request."
    )

    @classmethod
    def INPUT_TYPES(cls):
        multiline = {"multiline": True, "dynamicPrompts": False}
        return {
            "required": {
                "what_do_you_want": (
                    GOALS,
                    {"default": AUTO_GOAL, "tooltip": "Start here. Auto resolves the mode from the three media-role questions."},
                ),
                "target_description": (
                    "STRING",
                    {
                        **multiline,
                        "default": "Describe the target video, its subjects, actions, setting, and ending.",
                        "tooltip": "Your rough creative brief. Exact dialogue can be entered separately below.",
                    },
                ),
                "how_images_are_used": (
                    IMAGE_USES,
                    {"default": NO_IMAGE, "tooltip": "An endpoint frame uses FL2VA; appearance/style and motion targets use Ref2VA."},
                ),
                "how_video_is_used": (
                    VIDEO_USES,
                    {"default": NO_VIDEO, "tooltip": "Direct edits, continuations, motion transfer, and structural references use Ref2VA."},
                ),
                "how_audio_is_used": (
                    AUDIO_USES,
                    {"default": NO_AUDIO, "tooltip": "Audio reuse/reference uses Ref2VA and requires an accompanying image or video."},
                ),
                "reference_fidelity": (
                    FIDELITIES,
                    {"default": AUTO_FIDELITY, "tooltip": "Controls the fixed retention marker used by full-reference prompts."},
                ),
                "reference_assets": (
                    "STRING",
                    {
                        **multiline,
                        "default": "",
                        "placeholder": "Picture 1: target character appearance\nVideo 1: walking motion to transfer\nAudio 1: voice timbre",
                        "tooltip": "One asset per line using Subject/Picture/Video/Audio N: description. Missing labels are supplied from the chosen roles.",
                    },
                ),
                "duration_seconds": (
                    "FLOAT",
                    {"default": 6.0, "min": 4.0, "max": 15.0, "step": 0.01, "tooltip": "H3 supports 4-15 seconds."},
                ),
                "visual_style": (
                    "STRING",
                    {"default": "cinematic, live-action", "tooltip": "For keyframe tasks, match the supplied frame unless an intentional change is requested."},
                ),
                "shot_and_timing_plan": (
                    "STRING",
                    {**multiline, "default": "", "tooltip": "Optional. Describe cuts and times; the rewrite request enforces H3 timestamp syntax."},
                ),
                "camera_direction": (
                    "STRING",
                    {"default": "", "tooltip": "Example: slowly push in with small amplitude toward the subject."},
                ),
                "dialogue_lyrics_and_visible_text": (
                    "STRING",
                    {**multiline, "default": "", "tooltip": "Exact words only. They remain in their original language."},
                ),
                "overall_soundscape": (
                    "STRING",
                    {**multiline, "default": "", "tooltip": "Ambience, physical sounds, and non-verbal human sounds; not dialogue or score."},
                ),
                "non_diegetic_music": (
                    "STRING",
                    {**multiline, "default": "N/A", "tooltip": "Audience-only score: instrumentation, tempo, rhythm, and dynamics. Use N/A for none."},
                ),
            }
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
    ):
        parsed_assets, notes = parse_assets(reference_assets)
        effective_image_use, effective_video_use = _resolve_roles(
            what_do_you_want,
            how_images_are_used,
            how_video_is_used,
        )
        effective_audio_use = how_audio_is_used
        if what_do_you_want == AUTO_GOAL:
            kinds = {asset.kind for asset in parsed_assets}
            if effective_image_use == NO_IMAGE and "Picture" in kinds:
                effective_image_use = APPEARANCE_IMAGE
            if effective_video_use == NO_VIDEO and "Video" in kinds:
                effective_video_use = CONTENT_VIDEO
            if effective_audio_use == NO_AUDIO and "Audio" in kinds:
                effective_audio_use = REFERENCE_AUDIO
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
        warnings = _warnings(
            decision,
            what_do_you_want,
            effective_image_use,
            effective_video_use,
            effective_audio_use,
            assets,
        )
        report = _mode_report(
            decision,
            what_do_you_want,
            effective_image_use,
            effective_video_use,
            effective_audio_use,
            assets,
            warnings,
        )

        if decision.mode == "Ref2VA":
            items = _build_reference_items(
                assets,
                effective_image_use,
                effective_video_use,
                effective_audio_use,
            )
            if not items:
                items = [
                    ReferenceItem(
                        "<Subject 1>",
                        "Subject",
                        "visible content",
                        "the main target subject described in the user brief",
                    )
                ]
            draft = _reference_prompt(
                what_do_you_want,
                duration_seconds,
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
            )
        else:
            draft = _base_prompt(
                decision.mode,
                duration_seconds,
                target_description,
                visual_style,
                shot_and_timing_plan,
                camera_direction,
                dialogue_lyrics_and_visible_text,
                overall_soundscape,
                non_diegetic_music,
                notes,
            )

        rewrite = _rewrite_request(
            decision,
            draft,
            report,
            target_description,
            reference_assets,
        )
        return (draft, rewrite, report)


NODE_CLASS_MAPPINGS = {"MiniMaxH3PromptGuide": MiniMaxH3PromptGuide}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3PromptGuide": "MiniMax H3 Prompt Guide"}
