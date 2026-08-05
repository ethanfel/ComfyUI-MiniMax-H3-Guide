"""Qwen-family prompt enhancement using ComfyUI's loaded CLIP interface."""

from __future__ import annotations

from collections import Counter
import hashlib
import re

try:
    from .media_context import (
        REFERENCE_CONTEXT_TYPE,
        reference_entries,
        reference_inventory,
    )
except ImportError:
    from media_context import (
        REFERENCE_CONTEXT_TYPE,
        reference_entries,
        reference_inventory,
    )


NO_TAIL = "[none — connected CLIP is already complete]"
TAIL_TYPE = "MINIMAX_H3_GENERATION_TAIL"


def _tail_choices() -> list[str]:
    """List only compatible generation-tail files when ComfyUI is available."""

    try:
        import folder_paths
    except ImportError:
        return [NO_TAIL]
    tails = [
        name
        for name in folder_paths.get_filename_list("text_encoders")
        if "generation_tail_50_63" in name.casefold()
    ]
    return [NO_TAIL, *sorted(tails, key=str.casefold)]


def _generate_with_tail(clip, tail_name: str, tokens, generation_options: dict):
    """Lazy import keeps this node pack importable outside ComfyUI for tests."""

    try:
        from .hybrid_tail import generate_with_tail
    except ImportError:
        from hybrid_tail import generate_with_tail
    return generate_with_tail(clip, tail_name, tokens, generation_options)


def _clip_offload_report(status: str) -> str:
    if status == "suppressed":
        return (
            "A legacy explicit connected-CLIP unload request was suppressed for safety; "
            "ComfyUI manages the connected model's residency. Any temporary H3 generation "
            "tail was still unloaded internally."
        )
    return ""


DEFAULT_SYSTEM_PROMPT = """You are an expert prompt engineer for MiniMax H3 audiovisual generation. Rewrite the supplied manual H3 draft into a production-ready prompt. Do not discuss the task, explain your choices, or add a preface. Return only the finished H3 prompt.

Follow these rules:
- Preserve the user's requested subjects, identities, actions, dialogue, lyrics, visible text, reference roles, endpoint frames, timing, and audio intent. Never replace or contradict them.
- Write in English except for dialogue and lyrics inside <d>[Language] ...</d> and text visibly present in the scene. Preserve their original wording and language.
- Treat attached pictures and timestamped video samples as optional visual evidence. Use each reference only for its declared label and role. Do not silently turn a picture into a first/last frame, confuse sampled video frames with separate pictures, or change reference numbering.
- When REFERENCE CONTEXT contains chained visual references, its routing mode, labels, and role bindings are authoritative. Endpoint routing uses the resolved I2VA, L2VA, or FL2VA base format; ref2va routing uses the six Ref2VA sections. Reconcile stale generic wording in the draft without changing the user's creative request.
- `subject_definitions` is the required Ref2VA section title, not a requirement to invent a <Subject N>. Create <Subject N> only for reusable visible content abstracted from an asset: a person, animal, object, prop, clothing, interface, effect, scene, environment, style, action, expression, or pose. In H3 vocabulary, Subject does not mean person only.
- If a picture only supplies reusable visible content, cite <Picture N> inside the corresponding <Subject N> definition and do not define that <Picture N> on a separate line. Track <Picture N> separately only when the image itself is a concrete first frame, keyframe, last frame, edited keyframe, storyboard, or composition anchor.
- Track <Video N> separately for direct editing, continuation, or whole-video camera movement, cuts, rhythm, and temporal structure. If a video only supplies reusable visible content or motion, cite <Video N> inside the corresponding <Subject N> definition instead. A reference asset may define multiple subjects, and one subject may combine multiple assets.
- Never add a generic main <Subject 1> merely because Ref2VA or reference generation is selected. Remove unsupported placeholder subjects from the manual draft when chained roles do not define reusable visible content.
- Every supplied <Picture N> or <Video N> source label must still be cited at least once. When it only supplies a <Subject N>, omit only a separate definition and retention row for the asset label; never erase its citation from the subject definition.
- Choose summary task types from actual relationships, not file presence. Concrete target-frame anchors use keyframe completion. Character, object, scene, style, action, motion, camera, cut, rhythm, and storyboard guidance use reference generation. Use video editing only when a source video is directly modified, and video continuation only when new content continues its ending. Use audio reuse only when signal content is copied, and audio reference when only timbre, delivery, music style, beat, dialogue or lyric content, continuity, or sound texture guides new audio. Combine distinct types with ` + ` and never repeat one.
- In Ref2VA, define every separately tracked item on its own line, then give that same item exactly one retention_analysis row using only its declared role. Insert each reference label at its first visible or audible use in detailed_description and again wherever its role takes effect. Do not introduce a new reference label outside subject_definitions.
- Make the video chronological and physically observable. For every shot, establish composition, subject appearance and position, environment and lighting, action and state changes, camera behavior, and synchronized sound.
- [Shot 1] has no timestamp. Later shots begin with [Shot N] At MM:SS.mmm and strictly increasing cut times inside the stated duration.
- Describe camera movement naturally. Include movement type and meaningful speed or amplitude, but do not invent camera movement that conflicts with a static-camera request.
- Give only actual vocal sources stable (S1), (S2), etc., assigned in first-vocal-event order and reused across shots. A non-vocal character gets no speaker ID. Put only spoken or sung words inside <d>[Language] ...</d>, preserving the user's words and language. Put visible text in English double quotation marks.
- For voice-over, use the exact phrase `says in an off-screen voiceover` and state immediately after its <d> block that the corresponding on-screen character's lips remain completely closed. When one utterance crosses a cut, place <scenetrans> at both connecting points and explicitly state that its audio continues across the cut. Use <cutoff> when speech is truncated by the video ending. Never duplicate or restart audio that is declared continuous.
- Keep ambience, physical action sounds, and non-verbal human sounds in overall_soundscape. Put only audience-only score in non_diegetic_music. Use N/A when there is no such score.
- If an <Audio N> row uses fully_copy, that source is the complete final audio track. Do not add, replace, remix, or newly synthesize dialogue, lyrics, ambience, effects, or music. Any audible event described elsewhere must already exist in that copied track. Cite <Audio N> in overall_soundscape and in non_diegetic_music unless that section is N/A.

Choose the output structure from the supplied draft and authoritative REFERENCE CONTEXT routing mode:

For T2VA, I2VA, FL2VA, or L2VA, output exactly these fields:
integrated_multimodal_description
overall_soundscape
non_diegetic_music

For I2VA, FL2VA, and L2VA, the official image-alignment instruction must be the first line followed by one blank line. Preserve a correct line, or repair a missing/incorrect one from the resolved mode, supplied pictures, actual final shot, and effective duration in the draft or mode report. I2VA anchors <Picture 1> at 0.00 seconds in [Shot 1]. FL2VA aligns Picture 1 with Shot 1 at 0.00 seconds and Picture 2 with the actual final [Shot N] at the effective duration. L2VA aligns <Picture 1> with the actual final [Shot N] at the effective duration. FL2VA and L2VA alignment durations use exactly two decimal places. Never assume the final anchor belongs to Shot 1 when later shots exist, and never invent a missing picture or duration.

For Ref2VA, output exactly these six sections in this order:
subject_definitions
summary
retention_analysis
detailed_description
overall_soundscape
non_diegetic_music

In Ref2VA, establish visual style in one or two English sentences at the start of detailed_description before [Shot 1]. Keep every applicable <Subject N>, <Picture N>, <Video N>, and <Audio N> meaning stable. Do not invent media assets. Use only the fixed summary task types keyframe completion, reference generation, video editing, video continuation, audio reuse, and audio reference. Use only fully_preserved, partially_preserved, attribute_transfer, or weak_reference for visible retention, and fully_copy, partially_copy, reference, or weak_reference for audio retention. For generation tasks, normally make detailed_description 350-500 English words unless dialogue timing requires otherwise. Video-editing detail instead scales with the source edit's complexity.

If the manual draft is already detailed and correctly formatted, make only useful corrections. Never wrap the result in Markdown or code fences."""


# ComfyUI serializes multiline widget defaults into workflows. These are the
# stripped SHA256 values of every built-in prompt shipped before the current
# one (0.1-0.5, 0.6.0, and 0.6.1). Exact matches are safe to upgrade; a single
# user edit changes the hash and is therefore preserved as a custom prompt.
HISTORICAL_SYSTEM_PROMPT_SHA256 = frozenset(
    {
        "2c60735ca7a5ffa8b8195b690f08bad9300ab63d4c4bf73ce3254e840480509a",
        "107b26269e6b23db5c71ecca4593834950dff9ad5b763f5ecdd160e2be02b230",
        "8c1214e96dca59f94a7d18ab87cbc2255f371327d4e68eff660f075eff6357c1",
    }
)


def resolve_system_prompt(system_prompt: str | None) -> tuple[str, bool]:
    """Resolve blank/current/custom text and upgrade exact historical defaults."""

    supplied = (system_prompt or "").strip()
    if not supplied:
        return DEFAULT_SYSTEM_PROMPT, False
    digest = hashlib.sha256(supplied.encode("utf-8")).hexdigest()
    if digest in HISTORICAL_SYSTEM_PROMPT_SHA256:
        return DEFAULT_SYSTEM_PROMPT, True
    return supplied, False


def build_llm_user_prompt(
    manual_prompt: str,
    mode_report: str = "",
    has_image: bool = False,
    reference_context: str = "",
) -> str:
    """Build the user turn without mixing it into the editable system prompt."""

    if reference_context.strip():
        image_note = (
            "Chained pictures and/or timestamped video samples are attached as visual evidence. "
            "Analyze them according to the exact labels, roles, and reference inventory below."
        )
    elif has_image:
        image_note = (
            "One legacy image is attached as optional visual context. Analyze it, but follow the "
            "role assigned by the manual draft."
        )
    else:
        image_note = "No visual media is attached. Work only from the text and declared reference roles."
    report = (
        mode_report.strip()
        or "No separate mode report was supplied; infer the mode from the draft."
    )
    draft = manual_prompt.strip() or "No manual prompt was supplied."
    context = reference_context.strip() or "No chained reference context was supplied."
    return f"""Enhance the following MiniMax H3 prompt draft.

VISUAL CONTEXT
{image_note}

REFERENCE CONTEXT
{context}

MODE REPORT
{report}

MANUAL H3 DRAFT
{draft}"""


def _is_minimax_h3_tokenizer(clip) -> bool:
    tokenizer = getattr(clip, "tokenizer", None)
    tokenizer_type = type(tokenizer)
    return (
        tokenizer_type.__name__ == "MiniMaxH3Tokenizer"
        or tokenizer_type.__module__.endswith(".minimax")
    )


def _is_supported_qwen_tokenizer(clip) -> bool:
    """Recognize ComfyUI's generative Qwen3-VL and Qwen3.5 tokenizers."""

    tokenizer = getattr(clip, "tokenizer", None)
    tokenizer_type = type(tokenizer)
    identity = f"{tokenizer_type.__module__}.{tokenizer_type.__name__}".casefold()
    return any(
        marker in identity for marker in ("qwen3vl", "qwen3_vl", "qwen35", "qwen3_5")
    )


def validate_clip_compatibility(clip, tail_name: str | None = None) -> None:
    """Reject definitely incompatible Comfy CLIPs without blocking custom wrappers."""

    required = ["tokenize", "decode"]
    if tail_name is None:
        required.append("generate")
    missing = [name for name in required if not callable(getattr(clip, name, None))]
    if missing:
        raise RuntimeError(
            "Prompt Enhancer needs a generative Qwen3-VL or Qwen3.5 CLIP with callable "
            f"{', '.join(required)} methods; the connected object is missing {', '.join(missing)}."
        )

    tokenizer = getattr(clip, "tokenizer", None)
    tokenizer_type = type(tokenizer)
    identity = f"{tokenizer_type.__module__}.{tokenizer_type.__name__}".casefold()
    known_qwen = _is_supported_qwen_tokenizer(clip)
    known_minimax = _is_minimax_h3_tokenizer(clip)
    # Unknown third-party/fake wrappers remain allowed. Reject only tokenizer
    # classes known to come from ComfyUI's non-Qwen text-encoder modules.
    known_comfy_tokenizer = identity.startswith("comfy.")
    if known_comfy_tokenizer and not (known_qwen or known_minimax):
        raise RuntimeError(
            "Prompt Enhancer formats Qwen chat and vision tokens, but the connected CLIP uses "
            f"{tokenizer_type.__module__}.{tokenizer_type.__name__}. Connect a complete "
            "Qwen3-VL/Qwen3.5 CLIP, or MiniMax H3's CLIP with its Generation Tail Loader."
        )


def format_chat_prompt(
    system_prompt: str,
    user_prompt: str,
    has_image: bool,
    minimax_clip: bool,
    visual_count: int | None = None,
) -> str:
    """Format a Qwen chat while accommodating H3's raw multimodal tokenizer."""

    image_block = ""
    count = int(has_image) if visual_count is None else max(0, int(visual_count))
    if count and not minimax_clip:
        image_block = "<|vision_start|><|image_pad|><|vision_end|>" * count + "\n"
    return (
        f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n"
        f"<|im_start|>user\n{image_block}{user_prompt.strip()}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _prepare_reference_visuals(entries: list[dict], minimax_reference_payload: bool):
    """Build tokenizer payloads and an explicit visual-order map."""

    generic_images = []
    minimax_items = []
    visual_map = []
    # Native H3 presents all pictures first, then reference videos. Preserve
    # order within each media type even when the UI chain interleaves them.
    endpoint_order = {"first_frame": 0, "last_frame": 1}
    ordered_entries = sorted(
        entries,
        key=lambda entry: (
            entry["kind"] == "video",
            endpoint_order.get(str(entry.get("h3_input", "")), 2),
        ),
    )
    for entry in ordered_entries:
        media = (
            entry.get("minimax_analysis_media", entry["analysis_media"])
            if minimax_reference_payload
            else entry["analysis_media"]
        )
        if entry["kind"] == "image":
            if minimax_reference_payload:
                minimax_items.append({"type": "image", "data": media[:1]})
            else:
                generic_images.append(media[:1])
                visual_map.append(
                    f"Visual input {len(generic_images)} is {entry['label']}."
                )
            continue

        timestamps = (
            entry.get("minimax_timestamps", entry["timestamps"])
            if minimax_reference_payload
            else entry["timestamps"]
        )
        if minimax_reference_payload:
            minimax_items.append(
                {"type": "video", "data": media, "timestamps": timestamps}
            )
        else:
            start = len(generic_images) + 1
            generic_images.extend(
                media[index : index + 1] for index in range(media.shape[0])
            )
            end = len(generic_images)
            formatted_times = ", ".join(f"{timestamp:.3f}s" for timestamp in timestamps)
            visual_map.append(
                f"Visual inputs {start}-{end} are chronological samples from {entry['label']} "
                f"at {formatted_times}."
            )
    return generic_images, minimax_items, "\n".join(visual_map)


def _entry_role_bindings(entry: dict) -> list[dict]:
    """Read v2 role bindings while keeping v1 single-role contexts usable."""

    bindings = entry.get("bindings")
    if not isinstance(bindings, (list, tuple)):
        bindings = entry.get("roles")
    if isinstance(bindings, (list, tuple)):
        return [dict(binding) for binding in bindings if isinstance(binding, dict)]
    role = entry.get("role")
    if not role:
        return []
    return [
        {
            "role": role,
            "retention": entry.get("retention", ""),
            "content_group": entry.get("content_group", ""),
            "shot_scope": entry.get("shot_scope", ""),
            "transfer_target": entry.get("transfer_target", ""),
            "notes": entry.get("notes", ""),
        }
    ]


def validate_reference_bindings(entries: list[dict]) -> None:
    """Reject unresolved v2 attribute transfers before asking Qwen to guess."""

    groups = {
        str(binding.get("content_group", "")).strip()
        for entry in entries
        for binding in _entry_role_bindings(entry)
        if str(binding.get("content_group", "")).strip()
        and binding.get("retention") != "attribute_transfer"
    }
    for entry in entries:
        for binding in _entry_role_bindings(entry):
            if binding.get("retention") != "attribute_transfer":
                continue
            source = str(binding.get("content_group", "")).strip()
            target = str(binding.get("transfer_target", "")).strip()
            if not source or not target:
                raise ValueError(
                    f"{entry.get('label', 'A reference')} uses attribute_transfer but its "
                    "source content_group or transfer_target is blank. Define both groups in "
                    "the Visual Reference Role chain."
                )
            if source == target:
                raise ValueError(
                    f"{entry.get('label', 'A reference')} transfers attributes from {source!r} "
                    "back to the same group. Choose a different transfer_target."
                )
            if target not in groups:
                raise ValueError(
                    f"{entry.get('label', 'A reference')} targets content group {target!r}, "
                    "but no chained role binding defines that group."
                )


def enhancer_reference_inventory(entries: list[dict]) -> str:
    """Render every v2 binding while retaining media_context's policy details."""

    inventory_entries = []
    for entry in entries:
        inventory_entry = dict(entry)
        if not isinstance(inventory_entry.get("bindings"), (list, tuple)):
            aliased_bindings = _entry_role_bindings(inventory_entry)
            if aliased_bindings:
                inventory_entry["bindings"] = aliased_bindings
        inventory_entries.append(inventory_entry)
    rendered = reference_inventory(inventory_entries)
    if not entries or not any(_entry_role_bindings(entry) for entry in entries):
        return rendered

    missing_lines = []
    for entry in entries:
        label = entry.get("label", "<unlabeled reference>")
        for index, binding in enumerate(_entry_role_bindings(entry), start=1):
            signature = (
                f"role={binding.get('role', 'missing')}; "
                f"retention={binding.get('retention', 'missing')}"
            )
            if signature in rendered:
                continue
            details = [signature]
            for key, display in (
                ("content_group", "content group"),
                ("transfer_target", "transfer target"),
                ("shot_scope", "shot scope"),
                ("notes", "user notes"),
            ):
                value = str(binding.get(key, "")).strip()
                if value:
                    details.append(f"{display}={value}")
            missing_lines.append(
                f"{label} binding {index}: " + "; ".join(details) + "."
            )
    if missing_lines:
        rendered += "\n\nAUTHORITATIVE ROLE BINDINGS\n" + "\n".join(missing_lines)
    return rendered


def clean_generated_prompt(text: str, fallback: str) -> str:
    """Remove common chat/fence residue without altering the H3 body."""

    value = (text or "").strip()
    value = re.sub(r"^<\|im_start\|>assistant\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(?:assistant\s*:?[\r\n]+)", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^<think>.*?</think>\s*", "", value, flags=re.DOTALL)
    fence = re.match(
        r"^```(?:text|markdown)?\s*\n?(.*?)\n?```$",
        value,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence:
        value = fence.group(1).strip()
    return value or fallback.strip()


def generation_collapse_reason(text: str) -> str | None:
    """Recognize common LLM decoding collapse without rejecting short valid prompts."""

    value = (text or "").strip()
    if not value:
        return "the model returned no text"
    if re.search(r"<think(?:\s[^>]*)?>", value, flags=re.IGNORECASE) and not re.search(
        r"</think>", value, flags=re.IGNORECASE
    ):
        return "the model ended inside an unfinished reasoning block"

    non_space = [char for char in value if not char.isspace()]
    alphanumeric = sum(char.isalnum() for char in non_space)
    if non_space and alphanumeric / len(non_space) < 0.05:
        return "the output collapsed into punctuation"

    if len(value) < 40:
        return None

    tokens = re.findall(r"[\w<>/.-]+|[^\w\s]", value.casefold())
    if len(tokens) >= 20:
        _, repeated_count = Counter(tokens).most_common(1)[0]
        if repeated_count / len(tokens) >= 0.75:
            return "the output repeated one token almost exclusively"
    return None


BASE_H3_SECTIONS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)
REFERENCE_H3_SECTIONS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
ALL_H3_SECTIONS = tuple(dict.fromkeys((*REFERENCE_H3_SECTIONS, *BASE_H3_SECTIONS)))
SECTION_PATTERN = re.compile(
    rf"^({'|'.join(map(re.escape, ALL_H3_SECTIONS))})\s*:",
    flags=re.IGNORECASE | re.MULTILINE,
)
REFERENCE_LABEL_PATTERN = re.compile(r"<(Subject|Picture|Video|Audio)\s+(\d+)>")
ALLOWED_TASK_TYPES = frozenset(
    {
        "keyframe completion",
        "reference generation",
        "video editing",
        "video continuation",
        "audio reuse",
        "audio reference",
    }
)
VISUAL_RETENTION_MARKERS = frozenset(
    {"fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference"}
)
AUDIO_RETENTION_MARKERS = frozenset(
    {"fully_copy", "partially_copy", "reference", "weak_reference"}
)


def reference_context_mode_hint(reference_context, entries: list[dict]) -> str | None:
    """Read v2 routing metadata and infer endpoint mode when only entries remain."""

    if isinstance(reference_context, dict):
        supplied_hint = str(reference_context.get("mode_hint", "")).strip()
        for mode in ("T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA"):
            if supplied_hint.casefold() == mode.casefold():
                return mode
        routing_mode = str(reference_context.get("routing_mode", "")).casefold()
        if routing_mode == "ref2va":
            return "Ref2VA"

    families = {
        str(entry.get("route_family", "")).casefold()
        for entry in entries
        if entry.get("route_family")
    }
    if families == {"ref2va"}:
        return "Ref2VA"
    if families == {"endpoint"}:
        inputs = {str(entry.get("h3_input", "")) for entry in entries}
        if {"first_frame", "last_frame"} <= inputs:
            return "FL2VA"
        if "first_frame" in inputs:
            return "I2VA"
        if "last_frame" in inputs:
            return "L2VA"
    return None


def expected_h3_mode(
    manual_prompt: str,
    mode_report: str,
    has_references: bool,
    reference_mode_hint: str | None = None,
) -> str:
    """Resolve the expected output family while allowing an absent mode report."""

    if reference_mode_hint:
        return reference_mode_hint
    if has_references:
        # Legacy v1 visual chains predate endpoint routing and are Ref2VA-only.
        return "Ref2VA"
    match = re.search(
        r"Recommended\s+mode\s*:\s*(T2VA|I2VA|FL2VA|L2VA|Ref2VA)",
        mode_report or "",
        flags=re.IGNORECASE,
    )
    if match:
        value = match.group(1).upper()
        return "Ref2VA" if value == "REF2VA" else value
    if re.search(r"^\s*subject_definitions\s*:", manual_prompt or "", re.IGNORECASE):
        return "Ref2VA"
    return "T2VA"


def _section_matches(text: str):
    return list(SECTION_PATTERN.finditer(text or ""))


def _section_body(text: str, section: str, matches=None) -> str:
    matches = _section_matches(text) if matches is None else matches
    for index, match in enumerate(matches):
        if match.group(1).casefold() != section.casefold():
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        return text[match.end() : end].strip()
    return ""


def _shot_warnings(body: str) -> list[str]:
    warnings = []
    markers = list(
        re.finditer(
            r"\[Shot\s+(\d+)\](?:\s+At\s+(\d{2}):(\d{2})\.(\d{3}))?",
            body,
            flags=re.IGNORECASE,
        )
    )
    if not markers:
        return ["the main description has no [Shot 1] marker"]

    numbers = [int(marker.group(1)) for marker in markers]
    if numbers != list(range(1, len(numbers) + 1)):
        warnings.append("shot numbers are not contiguous from [Shot 1]")
    if markers[0].group(2) is not None:
        warnings.append("[Shot 1] must not have a timestamp")

    cut_times = []
    for marker in markers[1:]:
        if marker.group(2) is None:
            warnings.append(f"[Shot {marker.group(1)}] is missing At MM:SS.mmm")
            continue
        cut_times.append(
            int(marker.group(2)) * 60
            + int(marker.group(3))
            + int(marker.group(4)) / 1000.0
        )
    if any(later <= earlier for earlier, later in zip(cut_times, cut_times[1:])):
        warnings.append("later-shot timestamps are not strictly increasing")
    return warnings


def _expected_reference_label_is_present(
    text: str, label: str, expected_mode: str
) -> bool:
    """Accept FL2VA's official bare Picture labels without weakening other modes."""

    if label in text:
        return True
    if expected_mode != "FL2VA":
        return False
    picture = re.fullmatch(r"<Picture\s+(\d+)>", label)
    return bool(
        picture
        and re.search(
            rf"(?<![\w])Picture\s+{re.escape(picture.group(1))}(?!\d)",
            text,
        )
    )


def h3_structure_warnings(
    text: str,
    expected_mode: str,
    expected_labels: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """Return non-destructive warnings for malformed or truncated H3 output."""

    value = (text or "").strip()
    reference_mode = expected_mode.casefold() == "ref2va"
    required = REFERENCE_H3_SECTIONS if reference_mode else BASE_H3_SECTIONS
    matches = _section_matches(value)
    names = [match.group(1).casefold() for match in matches]
    warnings = []

    missing = [name for name in required if name.casefold() not in names]
    if missing:
        warnings.append("missing required section(s): " + ", ".join(missing))
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        warnings.append("duplicate section(s): " + ", ".join(duplicates))
    unexpected = [name for name in dict.fromkeys(names) if name not in required]
    if unexpected:
        warnings.append("unexpected section(s) for this mode: " + ", ".join(unexpected))
    present_required = [name for name in names if name in required]
    expected_present = [name for name in required if name in present_required]
    if present_required != expected_present:
        warnings.append("required sections are out of order")
    empty = [
        name
        for name in required
        if name.casefold() in names and not _section_body(value, name, matches)
    ]
    if empty:
        warnings.append("empty required section(s): " + ", ".join(empty))

    main_section = (
        "detailed_description"
        if reference_mode
        else "integrated_multimodal_description"
    )
    main_body = _section_body(value, main_section, matches)
    if main_body:
        warnings.extend(_shot_warnings(main_body))
        if reference_mode:
            first_shot = re.search(r"\[Shot\s+1\]", main_body, flags=re.IGNORECASE)
            style_opening = main_body[: first_shot.start()] if first_shot else ""
            if not any(char.isalpha() for char in style_opening):
                warnings.append(
                    "Ref2VA detailed_description needs a style opening before [Shot 1]"
                )

    if reference_mode:
        summary = _section_body(value, "summary", matches)
        task_prefix = re.match(r"^\[([^\]]+)\]", summary)
        if summary and not task_prefix:
            warnings.append("summary does not begin with a bracketed task-type prefix")
        elif task_prefix:
            task_types = [
                item.strip().casefold() for item in task_prefix.group(1).split("+")
            ]
            invalid = [item for item in task_types if item not in ALLOWED_TASK_TYPES]
            if invalid:
                warnings.append(
                    "summary uses unsupported task type(s): " + ", ".join(invalid)
                )
            if len(task_types) != len(set(task_types)):
                warnings.append("summary repeats a task type")

        definitions = _section_body(value, "subject_definitions", matches)
        retention = _section_body(value, "retention_analysis", matches)
        defined = re.findall(
            r"^\s*(<(?:Subject|Picture|Video|Audio)\s+\d+>)",
            definitions,
            flags=re.MULTILINE,
        )
        retained = re.findall(
            r"^\s*(<(?:Subject|Picture|Video|Audio)\s+\d+>)",
            retention,
            flags=re.MULTILINE,
        )
        invalid_markers = []
        missing_markers = []
        fully_copy_labels = []
        for label, row_body in re.findall(
            r"^\s*(<(?:Subject|Picture|Video|Audio)\s+\d+>)([^\n]*)",
            retention,
            flags=re.MULTILINE,
        ):
            marker_match = re.match(
                r"(?:\s*\([^\n)]*\))?\s*:\s*([A-Za-z_]+)\b",
                row_body,
            )
            if marker_match is None:
                missing_markers.append(label)
                continue
            marker = marker_match.group(1)
            allowed_markers = (
                AUDIO_RETENTION_MARKERS
                if label.startswith("<Audio ")
                else VISUAL_RETENTION_MARKERS
            )
            if marker not in allowed_markers:
                invalid_markers.append(f"{label}={marker}")
            elif marker == "fully_copy":
                fully_copy_labels.append(label)
        if missing_markers:
            warnings.append(
                "retention row(s) missing a recognizable marker: "
                + ", ".join(missing_markers)
            )
        if invalid_markers:
            warnings.append(
                "invalid retention marker(s): " + ", ".join(invalid_markers)
            )
        if fully_copy_labels:
            soundscape = _section_body(value, "overall_soundscape", matches)
            music = _section_body(value, "non_diegetic_music", matches)
            uncited = [label for label in fully_copy_labels if label not in soundscape]
            if music.strip().upper().rstrip(".") != "N/A":
                uncited.extend(
                    label for label in fully_copy_labels if label not in music
                )
            if uncited:
                warnings.append(
                    "fully_copy audio is not cited in every applicable audio section: "
                    + ", ".join(dict.fromkeys(uncited))
                )
            if not re.search(
                r"\b(?:complete final audio|reused 1:1|unchanged copied|no (?:new|added|removed|replaced))\b",
                f"{soundscape}\n{music}",
                flags=re.IGNORECASE,
            ):
                warnings.append(
                    "fully_copy audio sections do not state that the copied track remains exclusive"
                )
        defined_counts = Counter(defined)
        retained_counts = Counter(retained)
        duplicate_definitions = [
            label for label, count in defined_counts.items() if count > 1
        ]
        if duplicate_definitions:
            warnings.append(
                "duplicate tracked definition row(s): "
                + ", ".join(duplicate_definitions)
            )
        missing_retention = [
            label for label in defined_counts if label not in retained_counts
        ]
        if missing_retention:
            warnings.append(
                "tracked definition(s) missing a retention row: "
                + ", ".join(missing_retention)
            )
        duplicate_retention = [
            label for label, count in retained_counts.items() if count > 1
        ]
        if duplicate_retention:
            warnings.append(
                "duplicate retention row(s): " + ", ".join(duplicate_retention)
            )
        undefined_retention = [
            label for label in retained_counts if label not in defined_counts
        ]
        if undefined_retention:
            warnings.append(
                "retention row(s) without a tracked definition: "
                + ", ".join(undefined_retention)
            )

    expected_labels = tuple(label for label in expected_labels if label)
    absent_labels = [
        label
        for label in expected_labels
        if not _expected_reference_label_is_present(value, label, expected_mode)
    ]
    if absent_labels:
        warnings.append(
            "supplied reference label(s) are absent: " + ", ".join(absent_labels)
        )
    if expected_labels:
        supplied_visuals = {
            label
            for label in expected_labels
            if label.startswith(("<Picture ", "<Video "))
        }
        found_visuals = {
            f"<{kind} {number}>"
            for kind, number in REFERENCE_LABEL_PATTERN.findall(value)
            if kind in {"Picture", "Video"}
        }
        invented = sorted(found_visuals - supplied_visuals)
        if invented:
            warnings.append("unsupplied visual asset label(s): " + ", ".join(invented))

    if not reference_mode:
        leading = value.lstrip()
        if expected_mode == "I2VA" and not leading.startswith(
            "For the target video, at 0.00 seconds into the target video"
        ):
            warnings.append("I2VA is missing its first-frame alignment instruction")
        elif expected_mode in {"FL2VA", "L2VA"} and not leading.startswith(
            "How the reference pictures align with the target video"
        ):
            warnings.append(
                f"{expected_mode} is missing its endpoint alignment instruction"
            )

    return list(dict.fromkeys(warnings))


def _decorate_report(
    report: str,
    migrated_system_prompt: bool = False,
    structure_warnings: list[str] | tuple[str, ...] = (),
) -> str:
    additions = []
    if migrated_system_prompt:
        additions.append(
            "The serialized historical built-in system prompt was upgraded to the current version."
        )
    if structure_warnings:
        additions.append("H3 structure warning: " + "; ".join(structure_warnings) + ".")
    return " ".join((report, *additions)).strip()


def clip_generation_issue(clip) -> str | None:
    """Explain why MiniMax's truncated conditioning encoder should not generate text."""

    if not _is_minimax_h3_tokenizer(clip):
        return None
    stage = getattr(clip, "cond_stage_model", None)
    if stage is None or not hasattr(stage, "clip"):
        return None
    inner_clip = getattr(stage, stage.clip, None)
    transformer = getattr(inner_clip, "transformer", None)
    config = getattr(getattr(transformer, "model", None), "config", None)
    layers = getattr(
        config, "num_hidden_layers", getattr(transformer, "num_layers", None)
    )
    has_lm_head = getattr(config, "lm_head", None)
    has_final_norm = getattr(config, "final_norm", None)
    if layers == 50 and has_lm_head is False and has_final_norm is False:
        return (
            "Enhancement skipped: the connected MiniMax H3 CLIP is the conditioning-only "
            "Qwen3-VL-32B checkpoint truncated to 50 layers, without a final normalization or "
            "language-model head. It can condition H3 but cannot reliably generate instructions. "
            "Connect a MiniMax H3 Generation Tail Loader to clip_tail, connect a complete "
            "instruction-tuned Qwen3-VL or Qwen3.5 model, or send the textual llm_prompt plus any visual "
            "inputs through another multimodal LLM node's own visual-token format. manual_prompt "
            "was returned unchanged."
        )
    return None


def _resolve_tail_name(clip_tail) -> str | None:
    """Resolve the lightweight loader output, accepting old string inputs too."""

    if clip_tail is None or clip_tail == NO_TAIL:
        return None
    if isinstance(clip_tail, str):
        return clip_tail
    if isinstance(clip_tail, dict):
        tail_name = clip_tail.get("tail_name")
        if isinstance(tail_name, str) and tail_name and tail_name != NO_TAIL:
            return tail_name
    raise ValueError(
        "clip_tail must come from MiniMax H3 Generation Tail Loader, or be left disconnected "
        "when the connected CLIP is already generation-capable."
    )


def _generate_with_clip(
    clip,
    tokens,
    generation_options: dict,
    use_minimax_image_path: bool,
):
    """Generate like ComfyUI's TextGenerate, preserving Qwen3-VL image inputs.

    MiniMax's H3 CLIP uses the base generation wrapper, which does not forward
    Qwen3-VL MRoPE/DeepStack image metadata. When that exact CLIP receives an
    image, use the same image-aware arguments as ComfyUI's Qwen3VLClipModel.
    """

    stage = getattr(clip, "cond_stage_model", None)
    if not use_minimax_image_path or stage is None or not hasattr(stage, "clip"):
        return clip.generate(tokens, **generation_options)

    import comfy.model_management  # Imported only inside a running ComfyUI instance.

    stage.reset_clip_options()
    clip.load_model(tokens)
    device = clip.patcher.load_device
    stage.set_clip_options({"layer": None})
    stage.set_clip_options({"execution_device": device})
    inner_clip = getattr(stage, stage.clip)

    token_batches = next(iter(tokens.values())) if isinstance(tokens, dict) else tokens
    tokens_only = [[token[0] for token in batch] for batch in token_batches]
    embeds, _, _, embeds_info = inner_clip.process_tokens(tokens_only, device)
    position_ids, visual_pos_masks, deepstack = (
        inner_clip.transformer.build_image_inputs(embeds, embeds_info)
    )
    with comfy.model_management.cuda_device_context(device):
        return inner_clip.transformer.generate(
            embeds,
            **generation_options,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_embeds=deepstack,
        )


class MiniMaxH3GenerationTailLoader:
    """Select a compatible H3 generation tail without loading it persistently."""

    CATEGORY = "MiniMax H3/Prompting"
    FUNCTION = "select_tail"
    RETURN_TYPES = (TAIL_TYPE,)
    RETURN_NAMES = ("clip_tail",)
    OUTPUT_TOOLTIPS = (
        "Connect this to Prompt Enhancer.clip_tail. This lightweight descriptor lets the enhancer "
        "temporarily load layers 50-63, final norm, and LM head only while it writes the prompt.",
    )
    DESCRIPTION = (
        "Side loader for MiniMax H3's optional Qwen3-VL generation tail. Use it only with the "
        "bundled 50-layer conditioning CLIP; complete Qwen3-VL and Qwen3.5 CLIPs need no tail."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tail_file": (
                    _tail_choices(),
                    {
                        "tooltip": (
                            "Select the layers 50-63 generation-tail safetensors stored under "
                            "ComfyUI/models/text_encoders. If only the 'none' entry is shown, install "
                            "the compatible MiniMax H3 tail and refresh ComfyUI. The weights are loaded "
                            "temporarily by the enhancer, then unloaded; this node itself uses no VRAM."
                        )
                    },
                )
            }
        }

    def select_tail(self, tail_file: str):
        if tail_file == NO_TAIL:
            raise RuntimeError(
                "No compatible MiniMax H3 generation tail was selected. Install a file whose name "
                "contains 'generation_tail_50_63' under models/text_encoders, then refresh ComfyUI."
            )
        return ({"tail_name": tail_file},)


class MiniMaxH3PromptEnhancer:
    """Generate an enhanced H3 prompt with a supported generative Qwen CLIP."""

    CATEGORY = "MiniMax H3/Prompting"
    FUNCTION = "enhance"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "system_prompt", "llm_prompt", "enhancer_report")
    OUTPUT_TOOLTIPS = (
        "Cleaned candidate text produced by Qwen. Review enhancer_report and correct any structural warning before saving it or connecting it to an official MiniMax H3 conditioning node.",
        "Exact resolved system instructions used for enhancement. Connect to a text viewer to inspect/copy them; edit the system_prompt widget to customize behavior.",
        "Textual Qwen chat template, including system, inventory, and user turns. Pixel tensors remain external tokenizer inputs. MiniMax's tokenizer prepends actual <Picture N> blocks for endpoint or Ref2VA pictures and <Video N> blocks for Ref2VA video. Use this for text debugging only, never as H3's video prompt; an external multimodal LLM still needs the pixels attached in its own visual-token format.",
        "Resolved H3 output family, generation status, and H3 structure warnings. It explains success, conditioning-only skips, historical system-prompt upgrades, or fallback after empty/repetitive/punctuation output.",
    )
    DESCRIPTION = (
        "Second step after MiniMax H3 Prompt Guide. Connect h3_prompt, mode_report, and a generative Qwen3-VL or Qwen3.5 CLIP. "
        "A complete generative CLIP needs no tail; MiniMax H3's 50-layer conditioning CLIP uses the optional 50-63 tail. "
        "The node can analyze a chained set of role-labeled pictures and video samples, or one legacy image. "
        "ComfyUI manages the connected CLIP's residency; only the temporary generation tail is explicitly unloaded. "
        "Connect enhanced_prompt to ComfyUI's official H3 conditioning node."
    )

    @classmethod
    def INPUT_TYPES(cls):
        multiline = {"multiline": True, "dynamicPrompts": False}
        return {
            "required": {
                "clip": (
                    "CLIP",
                    {
                        "tooltip": (
                            "Connect a complete instruction-tuned Qwen3-VL/Qwen3.5 model or MiniMax H3's normal 50-layer conditioning CLIP. "
                            "For the 50-layer MiniMax CLIP, connect MiniMax H3 Generation Tail Loader to the optional clip_tail input. "
                            "This input means an LLM-capable ComfyUI CLIP, not an OpenAI CLIP vision model."
                        )
                    },
                ),
                "manual_prompt": (
                    "STRING",
                    {
                        **multiline,
                        "default": "",
                        "tooltip": (
                            "Connect h3_prompt from MiniMax H3 Prompt Guide. This is intentionally the pre-LLM structured draft. "
                            "You may instead paste a manual H3 prompt. Include real creative details; empty/default descriptions give Qwen little useful material to enhance."
                        ),
                    },
                ),
                "mode_report": (
                    "STRING",
                    {
                        **multiline,
                        "default": "",
                        "tooltip": (
                            "Recommended: connect mode_report from MiniMax H3 Prompt Guide. It tells Qwen the selected mode/checkpoint, resolved image/video/audio roles, task prefix, and warnings. "
                            "Leave blank only when the manual prompt already makes its H3 structure unambiguous."
                        ),
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        **multiline,
                        "default": DEFAULT_SYSTEM_PROMPT,
                        "tooltip": (
                            "Editable base behavior for Qwen. It contains H3 section order, shot/timestamp syntax, reference-label markers, dialogue, and audio rules. "
                            "Keep it unchanged initially. Add project-specific rules here, not scene content. If emptied, the built-in default is restored. Exact unedited defaults from older releases are upgraded automatically, while customized text is preserved. The resolved text is echoed from system_prompt output."
                        ),
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 1200,
                        "min": 64,
                        "max": 4096,
                        "step": 1,
                        "tooltip": (
                            "Maximum tokens Qwen may generate, not input length. 800-1200 is a practical range for the guide's detailed Ref2VA body plus other sections. "
                            "Lower values are faster but may truncate sections; increase only for dialogue-heavy prompts."
                        ),
                    },
                ),
                "sampling": (
                    ["deterministic", "sample"],
                    {
                        "default": "sample",
                        "tooltip": (
                            "Deterministic always chooses the most likely next token and ignores randomness, giving repeatable formatting. "
                            "Sample uses temperature/top-k/top-p/min-p and seed, usually producing richer descriptions."
                        ),
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.01,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": (
                            "Sampling creativity. Around 0.6-0.8 balances detail and strict formatting. Lower is more literal; high values can damage labels, timestamps, or section order. Used only when sampling=sample."
                        ),
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Limits sampling to the K most likely tokens. 64 is a stable default; 0 disables this filter. Used only when sampling=sample.",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Nucleus sampling threshold. 0.95 keeps likely alternatives while avoiding the long tail; 1.0 disables top-p filtering. Used only when sampling=sample.",
                    },
                ),
                "min_p": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Drops tokens whose probability is too small relative to the best token. 0.05 is conservative; 0 disables it. Used only when sampling=sample.",
                    },
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.05,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": (
                            "Discourages repeated phrases. Keep close to 1.0 because H3 deliberately repeats fixed labels such as <Subject 1> across sections; an excessive penalty can corrupt required structure."
                        ),
                    },
                ),
                "presence_penalty": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": (
                            "Additional penalty for tokens already used. Leave at 0 for H3 because reference labels and field names must recur. Raise only if Qwen is looping badly."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Controls repeatable random sampling. The same inputs, settings, and seed should reproduce the same text. It has no effect in deterministic mode.",
                    },
                ),
                "thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Off is recommended for faster direct rewriting. On allows Qwen to reason before answering, which may help complex reference relationships but costs more tokens/time. "
                            "Any decoded <think>...</think> block is removed from enhanced_prompt but remains conceptually part of generation cost."
                        ),
                    },
                ),
            },
            "optional": {
                "clip_tail": (
                    TAIL_TYPE,
                    {
                        "tooltip": (
                            "Connect MiniMax H3 Generation Tail Loader only when clip is H3's bundled "
                            "50-layer conditioning encoder. Leave disconnected for a complete "
                            "generation-capable Qwen3-VL or Qwen3.5 CLIP. The tail is temporary and does not alter clip."
                        )
                    },
                ),
                "reference_context": (
                    REFERENCE_CONTEXT_TYPE,
                    {
                        "tooltip": (
                            "Recommended visual-input route: connect the final reference_context from a "
                            "MiniMax H3 Enhancer Visual Reference chain. Qwen receives role-labeled pictures "
                            "and timestamped video samples. Follow that chain's routing_report to connect the "
                            "original media separately to native H3 Image to Video endpoint inputs or "
                            "Reference to Video reference inputs."
                        )
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Legacy compatibility input for one Qwen context image. Use a Visual Reference "
                            "chain for multiple pictures, explicit roles, correct H3 routing, or video. "
                            "Do not connect image and reference_context simultaneously."
                        ),
                    },
                ),
                "offload_after_generation": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Compatibility switch retained for existing workflows. The enhancer never "
                            "explicitly unloads a connected CLIP because a synchronous unload can stall "
                            "large complete Qwen models; ComfyUI manages its residency instead. If an old "
                            "workflow enables this switch, the request is safely ignored. The temporary "
                            "H3 generation tail is always unloaded internally."
                        ),
                    },
                ),
            },
        }

    def enhance(
        self,
        clip,
        manual_prompt: str,
        mode_report: str,
        system_prompt: str,
        max_new_tokens: int,
        sampling: str,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
        presence_penalty: float,
        seed: int,
        thinking: bool,
        clip_tail=None,
        reference_context=None,
        image=None,
        offload_after_generation: bool = False,
    ):
        if clip is None:
            raise RuntimeError("A generation-capable CLIP input is required.")
        if reference_context is not None and image is not None:
            raise ValueError(
                "Use either reference_context or the legacy image input, not both. The chain is "
                "required when visual labels and H3 routing must remain unambiguous."
            )

        tail_name = _resolve_tail_name(clip_tail)
        validate_clip_compatibility(clip, tail_name)
        resolved_system_prompt, migrated_system_prompt = resolve_system_prompt(
            system_prompt
        )
        minimax_clip = _is_minimax_h3_tokenizer(clip)
        entries = (
            reference_entries(reference_context)
            if reference_context is not None
            else []
        )
        validate_reference_bindings(entries)
        context_mode_hint = reference_context_mode_hint(reference_context, entries)
        resolved_mode = expected_h3_mode(
            manual_prompt,
            mode_report,
            bool(entries),
            context_mode_hint,
        )
        minimax_reference_payload = (
            minimax_clip and bool(entries) and resolved_mode == "Ref2VA"
        )
        generic_images, minimax_items, visual_map = _prepare_reference_visuals(
            entries, minimax_reference_payload
        )
        inventory = enhancer_reference_inventory(entries) if entries else ""
        if visual_map and not minimax_clip:
            inventory = f"{inventory}\n\nGENERIC QWEN VISUAL ORDER\n{visual_map}"
        has_visual = bool(entries) or image is not None
        expected_labels = tuple(str(entry.get("label", "")) for entry in entries)
        user_prompt = build_llm_user_prompt(
            manual_prompt,
            mode_report,
            has_image=has_visual,
            reference_context=inventory,
        )
        visual_count = (
            len(generic_images)
            if entries and not minimax_clip
            else int(image is not None)
        )
        llm_prompt = format_chat_prompt(
            resolved_system_prompt,
            user_prompt,
            has_image=has_visual,
            minimax_clip=minimax_clip,
            visual_count=visual_count,
        )
        if not thinking:
            llm_prompt += "<think>\n\n</think>\n\n"

        compatibility_issue = clip_generation_issue(clip) if tail_name is None else None
        if compatibility_issue:
            return (
                manual_prompt.strip(),
                resolved_system_prompt,
                llm_prompt,
                _decorate_report(
                    f"Resolved H3 output family: {resolved_mode}. {compatibility_issue}",
                    migrated_system_prompt,
                ),
            )

        tokenize_options = {
            "skip_template": True,
            "min_length": 1,
            "thinking": thinking,
        }
        if entries and minimax_reference_payload:
            tokenize_options["minimax_ref_items"] = minimax_items
        elif entries:
            tokenize_options["images"] = generic_images
        elif image is not None:
            first_image = image[:1]
            tokenize_options["image"] = first_image
            tokenize_options["images"] = [first_image]
        tokens = clip.tokenize(llm_prompt, **tokenize_options)
        generation_options = {
            "do_sample": sampling == "sample",
            "max_length": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "seed": seed,
        }
        clip_offload_status = "suppressed" if offload_after_generation else "disabled"
        if tail_name is None:
            generated_ids = _generate_with_clip(
                clip,
                tokens,
                generation_options,
                use_minimax_image_path=minimax_clip and has_visual,
            )
        else:
            generated_ids = _generate_with_tail(
                clip, tail_name, tokens, generation_options
            )
        raw_generated_prompt = clip.decode(generated_ids)
        enhanced_prompt = clean_generated_prompt(raw_generated_prompt, "")
        collapse = generation_collapse_reason(raw_generated_prompt)
        if collapse is None:
            collapse = generation_collapse_reason(enhanced_prompt)
        if collapse:
            report = (
                f"Resolved H3 output family: {resolved_mode}. Enhancement fallback: {collapse}. "
                "manual_prompt was returned unchanged. "
                "Try deterministic sampling, lower temperature, or a complete instruction-tuned "
                "Qwen3-VL or Qwen3.5 model."
            )
            enhanced_prompt = manual_prompt.strip()
            report = _decorate_report(report, migrated_system_prompt)
        else:
            generation_report = (
                f"Resolved H3 output family: {resolved_mode}. Enhancement completed successfully "
                "with the temporary MiniMax generation tail; "
                "the connected conditioning CLIP was left unchanged."
                if tail_name is not None
                else f"Resolved H3 output family: {resolved_mode}. Enhancement completed "
                "successfully with the connected complete CLIP."
            )
            if entries:
                picture_count = sum(entry["kind"] == "image" for entry in entries)
                video_count = sum(entry["kind"] == "video" for entry in entries)
                report = (
                    f"{generation_report} Qwen analyzed {picture_count} chained picture(s) and "
                    f"{video_count} timestamped reference video(s)."
                )
            else:
                report = generation_report
            structure_warnings = h3_structure_warnings(
                enhanced_prompt,
                resolved_mode,
                expected_labels,
            )
            report = _decorate_report(
                report,
                migrated_system_prompt,
                structure_warnings,
            )

        clip_offload_note = _clip_offload_report(clip_offload_status)
        if clip_offload_note:
            report = f"{report} {clip_offload_note}"

        return (enhanced_prompt, resolved_system_prompt, llm_prompt, report)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3GenerationTailLoader": MiniMaxH3GenerationTailLoader,
    "MiniMaxH3PromptEnhancer": MiniMaxH3PromptEnhancer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3GenerationTailLoader": "MiniMax H3 Generation Tail Loader",
    "MiniMaxH3PromptEnhancer": "MiniMax H3 Prompt Enhancer (Legacy Qwen LLM)",
}
