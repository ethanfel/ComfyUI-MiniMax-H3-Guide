"""Structured Qwen enhancement for the locked MiniMax H3 Plan v2 compiler.

The model never writes the final H3 document directly. It may only return a
small versioned JSON object containing descriptive prose. Python applies that
patch to a copy of the semantic plan, recompiles the prompt, and verifies that
all structural facts remained unchanged.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

try:
    from .enhancer import (
        TAIL_TYPE,
        _clip_offload_report,
        _generate_with_clip,
        _generate_with_tail,
        _is_minimax_h3_tokenizer,
        _prepare_reference_visuals,
        _resolve_tail_name,
        clip_generation_issue,
        format_chat_prompt,
        generation_collapse_reason,
        validate_clip_compatibility,
    )
    from .media_context import (
        _minimax_video_analysis_frames,
        _resize_long_edge,
        _video_analysis_frames,
    )
    from .plan_v2 import (
        H3_FPS,
        PHASE_COMPILED,
        PHASE_SETUP,
        PHASE_TIMELINE,
        PLAN_TYPE,
        _catalog,
        _copy_plan,
        compile_h3_plan,
        validated_plan,
    )
except ImportError:
    from enhancer import (
        TAIL_TYPE,
        _clip_offload_report,
        _generate_with_clip,
        _generate_with_tail,
        _is_minimax_h3_tokenizer,
        _prepare_reference_visuals,
        _resolve_tail_name,
        clip_generation_issue,
        format_chat_prompt,
        generation_collapse_reason,
        validate_clip_compatibility,
    )
    from media_context import (
        _minimax_video_analysis_frames,
        _resize_long_edge,
        _video_analysis_frames,
    )
    from plan_v2 import (
        H3_FPS,
        PHASE_COMPILED,
        PHASE_SETUP,
        PHASE_TIMELINE,
        PLAN_TYPE,
        _catalog,
        _copy_plan,
        compile_h3_plan,
        validated_plan,
    )


PROSE_SCHEMA_VERSION = 1
VISUAL_ANALYSIS_DISABLED = "Text metadata only (no visual analysis)"
VISUAL_ANALYSIS_ENABLED = "Analyze reference images and sampled video"
VISUAL_ANALYSIS_MODES = [VISUAL_ANALYSIS_DISABLED, VISUAL_ANALYSIS_ENABLED]
ENHANCEMENT_MODE_INTENT_LOCKED = "Intent-locked expansion"
ENHANCEMENT_MODE_CREATIVE = "Creative expansion"
ENHANCEMENT_MODES = [ENHANCEMENT_MODE_INTENT_LOCKED, ENHANCEMENT_MODE_CREATIVE]

PLAN_ENHANCER_SYSTEM_PROMPT = """You are a descriptive-prose editor for a compiled MiniMax H3 audiovisual plan.

You do not write an H3 prompt. You return only the JSON object requested by the user message. Improve clarity, chronology, physical observability, composition, lighting, motion, camera language, ambience, and synchronized action sounds while preserving the user's intent.

Enhancement standard:
- Follow the authoritative enhancement-mode contract appended to this prompt. It determines whether strings are addenda or complete rewrites.
- Add only grounded production detail: concrete blocking, spatial relationships, observable motion, continuity from opening state to ending state, and relevant synchronized physical sounds.
- Do not create new story facts such as characters, props, locations, actions, state changes, dialogue, lyrics, visible text, or independent audio sources. A visible attribute may be added only when supported by the supplied prose or visual evidence assigned to that exact role.
- Prefer information density over padding, repetition, ornamental adjectives, generic cinematic atmosphere, or restating the global premise.

Rules:
- Never introduce, remove, rename, renumber, paraphrase, move, or reinterpret a reference label such as <Subject 1>, <Picture 1>, <Video 1>, or <Audio 1>. An existing label may repeat naturally inside the same editable field.
- Never write H3 section names, [Shot N] markers, cut timestamps, H3 dialogue-tag markup, speaker IDs, task types, retention markers, or native routes. Python reconstructs them from locked data.
- Preserve the original creative emphasis and intensity as well as every requested identity, object, action, state change, relationship, setting, visible text, dialogue intent, and negative constraint. Never euphemize, sanitize, soften, intensify, moralize, or otherwise reinterpret it.
- Describe only observable audiovisual facts. Do not add motivation, judgment, emotional interpretation, or phrases such as an intimate atmosphere unless the source explicitly requests them.
- Treat attached visual media only as supplementary evidence for the exact declared image/video role. Textual scene intent is authoritative; pixels may clarify assigned identity or appearance but never override the requested action, emphasis, framing, or tone. Never infer audio, speech, music, or personality from pixels.
- Audio meaning comes only from supplied metadata and transcripts. Never claim to have listened to audio.
- Keep exact reference labels present in the same shot prose where they currently occur. They may be woven into a clearer sentence, but may not be removed or moved to another shot.
- Return valid UTF-8 JSON with double-quoted keys and strings. Do not use Markdown fences, comments, trailing commas, explanations, or additional keys."""


_STRUCTURAL_PROSE_PATTERNS = (
    (re.compile(r"\[\s*Shot\s+\d+\s*\]", re.IGNORECASE), "[Shot N] markers"),
    (re.compile(r"\bAt\s+\d{2}:\d{2}\.\d{3}\b", re.IGNORECASE), "cut timestamps"),
    (re.compile(r"</?d>", re.IGNORECASE), "<d> dialogue tags"),
    (
        re.compile(
            r"^\s*(?:subject_definitions|summary|retention_analysis|detailed_description|"
            r"integrated_multimodal_description|overall_soundscape|non_diegetic_music)\s*:",
            re.IGNORECASE | re.MULTILINE,
        ),
        "H3 section headers",
    ),
)
_REFERENCE_LABEL_RE = re.compile(
    r"<\s*(Subject|Picture|Video|Audio)\s+(\d+)\s*>",
    re.IGNORECASE,
)


def _source_plan(plan_context: Any) -> dict:
    """Return a mutable compiler input from a locked compiled Plan v2 value."""

    compiled = validated_plan(plan_context, allowed_phases={PHASE_COMPILED})
    source = _copy_plan(compiled)
    source["phase"] = PHASE_TIMELINE if source["shots"] else PHASE_SETUP
    source.pop("compiled", None)
    return source


def _base_package(plan_context: Any) -> tuple[str, str, dict, int]:
    """Recompile the locked plan so all enhancer fallbacks are immediately usable."""

    prompt, _rewrite, report, compiled, length = compile_h3_plan(
        _source_plan(plan_context)
    )
    return prompt, report, compiled, length


def editable_prose_payload(plan_context: Any) -> dict:
    """Extract the complete and only prose surface the LLM may modify."""

    plan = _source_plan(plan_context)
    return {
        "schema_version": PROSE_SCHEMA_VERSION,
        "visual_style": str(plan["project"]["visual_style"] or ""),
        "initial_prompt": str(plan["project"]["initial_prompt"] or ""),
        "shots": [
            {
                "shot_number": int(shot["shot_number"]),
                "description": str(shot["description"] or ""),
                "camera_direction": str(shot["camera_direction"] or ""),
            }
            for shot in plan["shots"]
        ],
        "overall_soundscape": str(plan["project"]["overall_soundscape"] or ""),
        "non_diegetic_music": str(plan["project"]["non_diegetic_music"] or ""),
    }


def editable_prose_json(plan_context: Any) -> str:
    return json.dumps(
        editable_prose_payload(plan_context),
        ensure_ascii=False,
        indent=2,
    )


def intent_addenda_payload(plan_context: Any) -> dict:
    """Return the intent-locked schema whose strings are additions, not rewrites."""

    source = editable_prose_payload(plan_context)
    return {
        "schema_version": PROSE_SCHEMA_VERSION,
        "visual_style": "",
        "initial_prompt": "",
        "shots": [
            {
                "shot_number": shot["shot_number"],
                "description": "",
                "camera_direction": "",
            }
            for shot in source["shots"]
        ],
        "overall_soundscape": "",
        "non_diegetic_music": "",
    }


def intent_addenda_json(plan_context: Any) -> str:
    return json.dumps(
        intent_addenda_payload(plan_context),
        ensure_ascii=False,
        indent=2,
    )


def _without_verbatim_source(candidate: str, source: str) -> str:
    """Tolerate a model copying the source before its requested addendum."""

    value = str(candidate or "").strip()
    original = str(source or "").strip()
    if original and value.startswith(original):
        return value[len(original) :].lstrip(" \t\r\n-–—:;.")
    return value


def _canonical_reference_label(match: re.Match) -> str:
    return f"<{match.group(1).title()} {int(match.group(2))}>"


def _reference_labels(value: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                _canonical_reference_label(match)
                for match in _REFERENCE_LABEL_RE.finditer(str(value or ""))
            }
        )
    )


def _prepare_intent_addendum(
    candidate: str, source: str
) -> tuple[str, tuple[str, ...]]:
    """Normalize same-field labels or reject an addendum that introduces one."""

    addition = _without_verbatim_source(candidate, source)
    allowed = set(_reference_labels(source))
    unexpected = tuple(
        label for label in _reference_labels(addition) if label not in allowed
    )
    if unexpected:
        return "", unexpected
    normalized = _REFERENCE_LABEL_RE.sub(_canonical_reference_label, addition)
    return normalized, ()


def _append_prose(source: str, addendum: str) -> str:
    original = str(source or "").strip()
    addition = _without_verbatim_source(addendum, original)
    if not addition:
        return original
    if not original:
        return addition
    separator = " " if original[-1] in ".!?" else ". "
    return original + separator + addition


def _compose_intent_locked_prose(
    plan_context: Any, model_output: str
) -> tuple[str, str]:
    """Append model addenda while keeping every original creative field intact."""

    source = editable_prose_payload(plan_context)
    addenda = parse_editable_prose(model_output, plan_context)
    combined = json.loads(json.dumps(source, ensure_ascii=False))
    applied = 0
    ignored = []

    for key in ("visual_style", "initial_prompt", "non_diegetic_music"):
        if str(addenda[key]).strip():
            ignored.append(key)

    for source_shot, addendum_shot, combined_shot in zip(
        source["shots"], addenda["shots"], combined["shots"]
    ):
        description_addendum, unexpected_labels = _prepare_intent_addendum(
            addendum_shot["description"], source_shot["description"]
        )
        if unexpected_labels:
            ignored.append(
                f"Shot {source_shot['shot_number']} description "
                f"(introduced {', '.join(unexpected_labels)})"
            )
        elif description_addendum:
            combined_shot["description"] = _append_prose(
                source_shot["description"], description_addendum
            )
            applied += 1

        camera_addendum, unexpected_labels = _prepare_intent_addendum(
            addendum_shot["camera_direction"], source_shot["camera_direction"]
        )
        if unexpected_labels:
            ignored.append(
                f"Shot {source_shot['shot_number']} camera_direction "
                f"(introduced {', '.join(unexpected_labels)})"
            )
        elif camera_addendum and source_shot["camera_direction"].strip():
            combined_shot["camera_direction"] = _append_prose(
                source_shot["camera_direction"], camera_addendum
            )
            applied += 1
        elif camera_addendum:
            ignored.append(f"Shot {source_shot['shot_number']} camera_direction")

    soundscape_addendum, unexpected_labels = _prepare_intent_addendum(
        addenda["overall_soundscape"], source["overall_soundscape"]
    )
    if unexpected_labels:
        ignored.append(
            f"overall_soundscape (introduced {', '.join(unexpected_labels)})"
        )
    elif (
        soundscape_addendum and source["overall_soundscape"].strip().casefold() != "n/a"
    ):
        combined["overall_soundscape"] = _append_prose(
            source["overall_soundscape"], soundscape_addendum
        )
        applied += 1
    elif soundscape_addendum:
        ignored.append("overall_soundscape")

    report = f"Intent lock preserved every original prose field and applied {applied} addendum(s)."
    if ignored:
        report += " Ignored model edits to locked field(s): " + ", ".join(ignored) + "."
    return json.dumps(combined, ensure_ascii=False), report


def _editable_prose_fields(payload: dict) -> list[str]:
    fields = [
        str(payload["visual_style"]),
        str(payload["initial_prompt"]),
        str(payload["overall_soundscape"]),
        str(payload["non_diegetic_music"]),
    ]
    for shot in payload["shots"]:
        fields.extend((str(shot["description"]), str(shot["camera_direction"])))
    return fields


def _prose_word_count(payload: dict) -> int:
    return sum(
        len(re.findall(r"\b\w+(?:[’'-]\w+)*\b", value, flags=re.UNICODE))
        for value in _editable_prose_fields(payload)
    )


def _prose_delta_report(before: dict, after: dict) -> str:
    before_fields = _editable_prose_fields(before)
    after_fields = _editable_prose_fields(after)
    changed = sum(
        old.strip() != new.strip() for old, new in zip(before_fields, after_fields)
    )
    before_words = _prose_word_count(before)
    after_words = _prose_word_count(after)
    word_delta = after_words - before_words
    report = (
        f"Prose delta: {changed}/{len(before_fields)} editable fields changed; "
        f"{before_words} -> {after_words} words ({word_delta:+d})."
    )
    if changed == 0:
        return report + " Warning: Qwen returned a valid but unchanged prose patch."
    light_threshold = max(20, round(before_words * 0.15))
    if word_delta < light_threshold:
        return (
            report
            + " Note: Qwen made a light edit rather than a substantial expansion."
        )
    return report


def _clean_model_json(text: str) -> str:
    value = str(text or "").strip()
    value = re.sub(r"^<\|im_start\|>assistant\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(?:assistant\s*:?\s*[\r\n]+)", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^<think>.*?</think>\s*", "", value, flags=re.DOTALL)
    fence = re.fullmatch(
        r"\x60{3}(?:json|text)?\s*\n?(.*?)\n?\x60{3}",
        value,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence:
        value = fence.group(1).strip()
    return value


def _mask_compiler_dialogue_lines(prompt: str) -> str:
    """Keep full-scene context while hiding markup Qwen must not copy."""

    replacement = (
        "A compiler-managed dialogue event occurs here. Its exact speaker, wording, "
        "voice reference, and delivery are locked and omitted from editable prose."
    )
    lines = []
    dropping = False
    for line in str(prompt or "").splitlines():
        lowered = line.casefold()
        if not dropping and "<d>" not in lowered:
            lines.append(line)
            continue
        if not dropping:
            lines.append(replacement)
        dropping = "</d>" not in lowered
    return "\n".join(lines)


def _strip_dialogue_tagged_sentences(value: str) -> tuple[str, int]:
    """Remove model-copied dialogue sentences that the compiler will restore."""

    text = str(value)
    removed = 0
    while True:
        opening = re.search(r"<d>", text, flags=re.IGNORECASE)
        if opening is None:
            break
        closing = re.search(r"</d>", text[opening.end() :], flags=re.IGNORECASE)
        closing_end = (
            opening.end() + closing.end() if closing is not None else len(text)
        )

        start = 0
        for boundary in re.finditer(r"[.!?](?:\s+)|\n+", text[: opening.start()]):
            start = boundary.end()

        end = closing_end
        boundary = re.search(r"[.!?](?:\s+|$)|\n+", text[closing_end:])
        if boundary is not None:
            end = closing_end + boundary.end()

        delivery = re.match(r"Delivery\s*:", text[end:], flags=re.IGNORECASE)
        if delivery is not None:
            delivery_end = end + delivery.end()
            boundary = re.search(r"[.!?](?:\s+|$)|\n+", text[delivery_end:])
            end = delivery_end + boundary.end() if boundary is not None else len(text)

        text = text[:start] + text[end:]
        removed += 1

    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), removed


def _repair_model_dialogue_leak(text: str) -> tuple[str, int]:
    """Repair only compiler-owned dialogue markup inside otherwise valid JSON."""

    cleaned = _clean_model_json(text)
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return text, 0
    if not isinstance(payload, dict):
        return text, 0

    removed = 0
    for key in (
        "visual_style",
        "initial_prompt",
        "overall_soundscape",
        "non_diegetic_music",
    ):
        if isinstance(payload.get(key), str):
            payload[key], count = _strip_dialogue_tagged_sentences(payload[key])
            removed += count
    if isinstance(payload.get("shots"), list):
        for shot in payload["shots"]:
            if not isinstance(shot, dict):
                continue
            for key in ("description", "camera_direction"):
                if isinstance(shot.get(key), str):
                    shot[key], count = _strip_dialogue_tagged_sentences(shot[key])
                    removed += count
    if not removed:
        return text, 0
    return json.dumps(payload, ensure_ascii=False), removed


def _require_exact_keys(value: dict, expected: set[str], location: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    problems = []
    if missing:
        problems.append("missing " + ", ".join(missing))
    if extra:
        problems.append("unexpected " + ", ".join(extra))
    if problems:
        raise ValueError(f"{location} has " + " and ".join(problems) + ".")


def _prose_string(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a string.")
    text = value.strip()
    for pattern, name in _STRUCTURAL_PROSE_PATTERNS:
        if pattern.search(text):
            raise ValueError(
                f"{location} contains {name}; leave those to the locked compiler."
            )
    return _REFERENCE_LABEL_RE.sub(_canonical_reference_label, text)


def parse_editable_prose(text: str, plan_context: Any) -> dict:
    """Parse and strictly validate one model- or user-edited prose patch."""

    cleaned = _clean_model_json(text)
    if not cleaned:
        raise ValueError("editable_prose is empty.")
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"editable_prose is not valid JSON: {error.msg} at line {error.lineno}, "
            f"column {error.colno}."
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("editable_prose must be one JSON object.")
    root_keys = {
        "schema_version",
        "visual_style",
        "initial_prompt",
        "shots",
        "overall_soundscape",
        "non_diegetic_music",
    }
    _require_exact_keys(payload, root_keys, "editable_prose")
    if payload["schema_version"] != PROSE_SCHEMA_VERSION:
        raise ValueError(
            f"editable_prose schema_version must be {PROSE_SCHEMA_VERSION}."
        )

    normalized = {
        "schema_version": PROSE_SCHEMA_VERSION,
        "visual_style": _prose_string(payload["visual_style"], "visual_style"),
        "initial_prompt": _prose_string(payload["initial_prompt"], "initial_prompt"),
        "overall_soundscape": _prose_string(
            payload["overall_soundscape"], "overall_soundscape"
        ),
        "non_diegetic_music": _prose_string(
            payload["non_diegetic_music"], "non_diegetic_music"
        ),
    }
    shots = payload["shots"]
    if not isinstance(shots, list):
        raise ValueError("editable_prose.shots must be a JSON array.")
    source = _source_plan(plan_context)
    if len(shots) != len(source["shots"]):
        raise ValueError(
            f"editable_prose.shots must contain exactly {len(source['shots'])} item(s); "
            "Shot structure is locked."
        )
    normalized_shots = []
    for index, (candidate, source_shot) in enumerate(
        zip(shots, source["shots"]), start=1
    ):
        if not isinstance(candidate, dict):
            raise ValueError(f"editable_prose.shots[{index - 1}] must be an object.")
        _require_exact_keys(
            candidate,
            {"shot_number", "description", "camera_direction"},
            f"editable_prose.shots[{index - 1}]",
        )
        if (
            isinstance(candidate["shot_number"], bool)
            or candidate["shot_number"] != source_shot["shot_number"]
        ):
            raise ValueError(
                f"editable_prose Shot {index} must keep shot_number "
                f"{source_shot['shot_number']}."
            )
        normalized_shots.append(
            {
                "shot_number": int(source_shot["shot_number"]),
                "description": _prose_string(
                    candidate["description"], f"Shot {index} description"
                ),
                "camera_direction": _prose_string(
                    candidate["camera_direction"], f"Shot {index} camera_direction"
                ),
            }
        )
    normalized["shots"] = normalized_shots
    return normalized


def _locked_signature(plan: dict) -> dict:
    """Build a tensor-free signature for every field outside editable prose."""

    return {
        "version": plan["version"],
        "project_timing": {
            key: plan["project"][key]
            for key in ("duration_seconds", "h3_length", "effective_duration")
        },
        "assets": [
            {key: value for key, value in asset.items() if key != "media"}
            for asset in plan["assets"]
        ],
        "bindings": [dict(binding) for binding in plan["bindings"]],
        "character_replacements": [
            dict(replacement) for replacement in plan["character_replacements"]
        ],
        "audio_relationships": [
            dict(relationship) for relationship in plan["audio_relationships"]
        ],
        "shot_structure": [
            {
                "shot_number": shot["shot_number"],
                "cut_at": shot["cut_at"],
                "transition": shot["transition"],
            }
            for shot in plan["shots"]
        ],
        "dialogue_events": [dict(event) for event in plan["dialogue_events"]],
    }


def _reference_label_signature(plan: dict) -> dict:
    """Keep every existing H3 label in the same editable field."""

    return {
        "visual_style": _reference_labels(plan["project"]["visual_style"]),
        "initial_prompt": _reference_labels(plan["project"]["initial_prompt"]),
        "overall_soundscape": _reference_labels(plan["project"]["overall_soundscape"]),
        "non_diegetic_music": _reference_labels(plan["project"]["non_diegetic_music"]),
        "shots": [
            {
                "description": _reference_labels(shot["description"]),
                "camera_direction": _reference_labels(shot["camera_direction"]),
            }
            for shot in plan["shots"]
        ],
    }


def apply_editable_prose(
    plan_context: Any, editable_prose: str
) -> tuple[str, dict, str, int, str]:
    """Apply a prose patch, rebuild the final prompt, and verify semantic locks."""

    original = validated_plan(plan_context, allowed_phases={PHASE_COMPILED})
    source = _source_plan(original)
    before = _locked_signature(source)
    labels_before = _reference_label_signature(source)
    prose_before = editable_prose_payload(original)
    payload = parse_editable_prose(editable_prose, original)

    updated = _copy_plan(source)
    updated["project"]["visual_style"] = payload["visual_style"]
    updated["project"]["initial_prompt"] = payload["initial_prompt"]
    updated["project"]["overall_soundscape"] = payload["overall_soundscape"]
    updated["project"]["non_diegetic_music"] = payload["non_diegetic_music"]
    for target, prose in zip(updated["shots"], payload["shots"]):
        target["description"] = prose["description"]
        target["camera_direction"] = prose["camera_direction"]

    if _locked_signature(updated) != before:
        raise RuntimeError("Internal error: applying prose changed locked plan data.")
    if _reference_label_signature(updated) != labels_before:
        raise ValueError(
            "Enhanced prose added, removed, renamed, or moved an H3 reference label. "
            "Each label must remain in the same editable field."
        )
    prompt, _rewrite, compiler_report, compiled, h3_length = compile_h3_plan(updated)
    if _locked_signature(_source_plan(compiled)) != before:
        raise RuntimeError("Internal error: recompilation changed locked plan data.")
    if compiled.get("compiled") != original.get("compiled"):
        raise ValueError(
            "Enhanced prose changed compiled labels, routes, mode, or speakers; "
            "the locked draft was preserved instead."
        )
    canonical_json = json.dumps(payload, ensure_ascii=False, indent=2)
    report = (
        "Structured prose applied and recompiled successfully. Locked labels, roles, "
        "retention, speakers, dialogue, cut times, mode, and native routes are unchanged.\n\n"
        + _prose_delta_report(prose_before, payload)
        + "\n\n"
        + compiler_report
    )
    return prompt, compiled, report, h3_length, canonical_json


def _apply_generated_prose(
    plan_context: Any,
    model_output: str,
    enhancement_mode: str,
) -> tuple[str, dict, str, int, str, str]:
    """Apply either a complete creative rewrite or intent-locked addenda."""

    enhancement_mode = _validate_enhancement_mode(enhancement_mode)
    if enhancement_mode == ENHANCEMENT_MODE_INTENT_LOCKED:
        editable_prose, mode_report = _compose_intent_locked_prose(
            plan_context, model_output
        )
    else:
        editable_prose = model_output
        mode_report = (
            "Creative expansion accepted Qwen's complete prose rewrite, subject to "
            "the structural and reference-label locks."
        )
    prompt, compiled, report, length, canonical = apply_editable_prose(
        plan_context, editable_prose
    )
    return prompt, compiled, report, length, canonical, mode_report


def _analysis_entries(
    plan_context: Any,
    analysis_long_edge: int,
    video_analysis_fps: float,
    max_analysis_frames: int,
) -> list[dict]:
    """Prepare visual evidence only; audio waveform data is never presented to Qwen."""

    plan = validated_plan(plan_context, allowed_phases={PHASE_COMPILED})
    catalog = _catalog(plan)
    entries = []
    long_edge = int(analysis_long_edge)
    fps = float(video_analysis_fps)
    maximum = int(max_analysis_frames)
    if long_edge < 256 or long_edge > 1536:
        raise ValueError("analysis_long_edge must be from 256 through 1536 pixels.")
    if not math.isfinite(fps) or fps <= 0.0 or fps > 2.0:
        raise ValueError("video_analysis_fps must be above 0 and at most 2 FPS.")
    if maximum < 2 or maximum > 32:
        raise ValueError("max_analysis_frames must be from 2 through 32.")

    for asset in catalog["pictures"]:
        media = asset["media"]
        evidence = _resize_long_edge(media[:1, ..., :3], long_edge)
        entries.append(
            {
                "kind": "image",
                "label": catalog["picture_labels"][asset["asset_id"]],
                "analysis_media": evidence,
                "minimax_analysis_media": evidence,
            }
        )
    for asset in catalog["videos"]:
        media = asset["media"][..., :3]
        sampled, timestamps = _video_analysis_frames(media, H3_FPS, fps, maximum)
        minimax_sampled, minimax_timestamps = _minimax_video_analysis_frames(media)
        entries.append(
            {
                "kind": "video",
                "label": catalog["video_labels"][asset["asset_id"]],
                "analysis_media": _resize_long_edge(sampled, long_edge),
                "timestamps": timestamps,
                "minimax_analysis_media": _resize_long_edge(minimax_sampled, long_edge),
                "minimax_timestamps": minimax_timestamps,
            }
        )
    return entries


def _plan_inventory(plan_context: Any) -> str:
    """Describe exact semantic roles without passing audio waveforms to the model."""

    plan = validated_plan(plan_context, allowed_phases={PHASE_COMPILED})
    catalog = _catalog(plan)
    bindings_by_asset: dict[str, list[dict]] = {}
    for binding in plan["bindings"]:
        bindings_by_asset.setdefault(binding["asset_id"], []).append(binding)
    audio_relationships = {
        relationship["asset_id"]: relationship
        for relationship in plan["audio_relationships"]
    }
    replacements_by_video: dict[str, list[dict]] = {}
    for replacement in plan["character_replacements"]:
        replacements_by_video.setdefault(
            replacement["source_video_asset_id"], []
        ).append(replacement)
    lines = []
    for asset in catalog["pictures"]:
        label = catalog["picture_labels"][asset["asset_id"]]
        lines.append(
            f"{label}: image; relationship={asset['relationship']}; "
            f"name={asset['reference_name']}; description={asset['description'] or 'not supplied'}; "
            f"retention={asset['retention']}; "
            f"shot_scope={asset['shot_scope'] or 'all applicable shots'}."
        )
        for binding in bindings_by_asset.get(asset["asset_id"], []):
            lines.append(
                f"  Subject binding: alias={binding['subject_name']}; "
                f"content_type={binding['content_type']}; retention={binding['retention']}; "
                f"transfer_target={binding['transfer_target_subject'] or 'none'}; "
                f"shot_scope={binding['shot_scope'] or 'all applicable shots'}; "
                f"notes={binding['notes'] or 'not supplied'}."
            )
    for asset in catalog["videos"]:
        label = catalog["video_labels"][asset["asset_id"]]
        lines.append(
            f"{label}: video; relationship={asset['relationship']}; "
            f"name={asset['reference_name']}; description={asset['description'] or 'not supplied'}; "
            f"duration={asset['native_duration']:.3f}s; retention={asset['retention']}; "
            f"target_subject={asset['target_subject'] or 'none'}; "
            f"shot_scope={asset['shot_scope'] or 'all applicable shots'}."
        )
        for binding in bindings_by_asset.get(asset["asset_id"], []):
            lines.append(
                f"  Subject binding: alias={binding['subject_name']}; "
                f"content_type={binding['content_type']}; retention={binding['retention']}; "
                f"transfer_target={binding['transfer_target_subject'] or 'none'}; "
                f"shot_scope={binding['shot_scope'] or 'all applicable shots'}; "
                f"notes={binding['notes'] or 'not supplied'}."
            )
        for replacement in replacements_by_video.get(asset["asset_id"], []):
            subject = catalog["subjects_by_alias"].get(
                str(replacement["replacement_subject"]).strip().casefold()
            )
            subject_label = subject["label"] if subject else "unresolved Subject"
            lines.append(
                f"  Character replacement: source_performer="
                f"{replacement['source_character_description']}; "
                f"replacement_subject={subject_label} "
                f"({replacement['replacement_subject']}); "
                f"appearance_policy={replacement['appearance_policy']}; "
                f"preserve_performance={replacement['preserve_performance']}; "
                f"preserve_scene={replacement['preserve_scene']}; "
                f"shot_scope={replacement['shot_scope']}; "
                f"instructions={replacement['instructions'] or 'none'}."
            )
    for asset in catalog["audios"]:
        relationship = audio_relationships[asset["asset_id"]]
        label = catalog["audio_labels"][asset["asset_id"]]
        lines.append(
            f"{label}: audio metadata only; relationship={relationship['use']}; "
            f"name={asset['reference_name']}; duration={asset['duration']:.3f}s; "
            f"target_speaker={relationship['target_speaker'] or 'none'}; "
            f"language={relationship['language'] or 'not supplied'}; "
            f"transcript={relationship['transcript'] or 'not supplied'}; "
            f"target_layer_or_event={relationship['target_layer_or_event'] or 'none'}; "
            f"instructions={relationship['instructions'] or 'none'}; "
            f"shot_scope={relationship['shot_scope'] or 'all applicable shots'}."
        )
    return "\n".join(lines) or "No reference media."


def _validate_enhancement_mode(enhancement_mode: str) -> str:
    if enhancement_mode not in ENHANCEMENT_MODES:
        raise ValueError("Choose a supported enhancement_mode.")
    return enhancement_mode


def _effective_system_prompt(
    base_prompt: str,
    plan_context: Any,
    enhancement_mode: str = ENHANCEMENT_MODE_INTENT_LOCKED,
) -> str:
    enhancement_mode = _validate_enhancement_mode(enhancement_mode)
    payload = editable_prose_payload(plan_context)
    shot_numbers = [shot["shot_number"] for shot in payload["shots"]]
    if enhancement_mode == ENHANCEMENT_MODE_INTENT_LOCKED:
        mode_contract = (
            "\n\nINTENT-LOCKED MODE CONTRACT:\n"
            "- Every returned string is an addendum only, never a replacement or paraphrase. "
            "Python preserves the original prose separately.\n"
            "- Keep visual_style, initial_prompt, and non_diegetic_music empty.\n"
            "- For each Shot description, add only 30-70 useful words of grounded, observable "
            "production detail. Do not restate the source or alter its emphasis.\n"
            "- Keep camera_direction empty when the original camera_direction is empty. When "
            "the original contains a camera request, return only a compatible addendum.\n"
            "- Normally omit reference labels from addenda because Python already preserves them. "
            "If a label is essential, repeat it only when that exact source field already contains "
            "the same label. Never mention an Audio label in Shot addenda; voice and dialogue "
            "relationships are compiler-managed.\n"
            "- overall_soundscape may contain only compatible ambience or synchronized physical-"
            "sound additions; never add dialogue, vocals, or music.\n"
            "- Do not invent colors, weather, lighting effects, lenses, atmosphere, motivations, "
            "or emotional interpretations unless supported by the original text or visual evidence "
            "assigned to that exact role."
        )
    else:
        mode_contract = (
            "\n\nCREATIVE EXPANSION MODE CONTRACT:\n"
            "- Return complete rewritten strings rather than addenda.\n"
            "- Perform a material production-detail rewrite and normally aim for 80-140 useful "
            "words per Shot description.\n"
            "- You may add a concise motivated camera_direction when it is blank, unless the "
            "source establishes a static camera or another conflicting constraint.\n"
            "- Presentation choices may be added only when they preserve every source fact, "
            "emphasis, intensity, role, and negative constraint."
        )
    return (
        base_prompt.strip()
        + "\n\nLOCKED OUTPUT CONTRACT — these instructions take precedence over any "
        "conflicting editable text:\n"
        + f"- schema_version must be {PROSE_SCHEMA_VERSION}.\n"
        + f"- shots must contain exactly these shot_number values in order: {shot_numbers}.\n"
        + "- Return exactly these root keys: schema_version, visual_style, initial_prompt, "
        "shots, overall_soundscape, non_diegetic_music.\n"
        + "- Every shot object must contain exactly: shot_number, description, "
        "camera_direction.\n"
        + "- Empty strings are structurally valid. You may fill blank descriptive fields with "
        "non-conflicting production detail, but preserve explicit N/A, silence, static-camera, "
        "and other negative constraints.\n"
        + "- Compiler-managed dialogue lines are masked in the read-only scene context. Never "
        "reconstruct dialogue wording, H3 dialogue-tag markup, speaker IDs, voice-reference clauses, or "
        "delivery clauses inside editable prose." + mode_contract
    )


def _llm_user_prompt(
    plan_context: Any,
    base_prompt: str,
    compiler_report: str,
    visual_analysis: str,
    enhancement_mode: str = ENHANCEMENT_MODE_INTENT_LOCKED,
) -> str:
    enhancement_mode = _validate_enhancement_mode(enhancement_mode)
    visual_note = (
        "Reference images and timestamped video samples are attached. Use them only according "
        "to REFERENCE INVENTORY."
        if visual_analysis == VISUAL_ANALYSIS_ENABLED
        else "No pixels are attached. Use text metadata only."
    )
    shot_count = len(editable_prose_payload(plan_context)["shots"])
    if enhancement_mode == ENHANCEMENT_MODE_INTENT_LOCKED:
        detail_target = (
            f"Write addenda for all {shot_count} Shot descriptions. Aim for 30-70 useful words "
            "per Shot without restating, replacing, softening, or intensifying the locked source."
        )
        output_material = f"""LOCKED ORIGINAL PROSE — context only; Python preserves it verbatim
{editable_prose_json(plan_context)}

ADDENDA JSON TEMPLATE — return this exact schema with additions only
{intent_addenda_json(plan_context)}"""
    else:
        detail_target = (
            f"Materially expand all {shot_count} Shot descriptions. Aim for 80-140 useful words per "
            "Shot plus one concise camera_direction per Shot, unless the supplied prose is already "
            "equally specific."
            if shot_count
            else "Materially expand the global scene prose without inventing a new event."
        )
        output_material = (
            "EDITABLE PROSE JSON — return the complete rewritten object\n"
            + editable_prose_json(plan_context)
        )
    return f"""Improve the editable prose JSON below, then return the complete JSON object only.

ENHANCEMENT MODE
{enhancement_mode}

PRODUCTION DETAIL TARGET
{detail_target}
This is a production-detail pass, not a summary. Preserve textual intent over optional visual evidence.

VISUAL EVIDENCE
{visual_note}

REFERENCE INVENTORY
{_plan_inventory(plan_context)}

LOCKED COMPILER REPORT
{compiler_report}

VALID COMPILED H3 CONTEXT
Dialogue lines are replaced by compiler-managed placeholders so they cannot leak into editable prose.
{_mask_compiler_dialogue_lines(base_prompt)}

{output_material}"""


class MiniMaxH3PlanV2ApplyProse:
    """Rebuild a locked Plan v2 prompt from manually edited structured prose."""

    CATEGORY = "MiniMax H3/Plan v2"
    FUNCTION = "apply_prose"
    RETURN_TYPES = ("STRING", PLAN_TYPE, "STRING", "INT", "STRING")
    RETURN_NAMES = (
        "h3_prompt",
        "plan_context",
        "validation_report",
        "h3_length",
        "canonical_editable_prose",
    )
    OUTPUT_TOOLTIPS = (
        "Recompiled final H3 prompt with every structural field restored by Python.",
        "Validated compiled plan for later native routing or another edit pass.",
        "Confirmation that semantic locks survived, followed by the compiler report.",
        "Native 17k+5 frame length.",
        "Normalized JSON after validation. Save or edit this instead of the final H3 structure.",
    )
    DESCRIPTION = (
        "Optional manual-edit step after the Plan v2 Structured Prompt Enhancer. It accepts only "
        "descriptive JSON, reconstructs the H3 document, and rejects changes to structural facts."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan_context": (
                    PLAN_TYPE,
                    {
                        "tooltip": "Connect compiled plan_context from Prompt Merge or the structured enhancer."
                    },
                ),
                "editable_prose": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": (
                            "Connect/edit the enhancer's JSON output. Structural markers, extra keys, "
                            "changed Shot numbers, or invalid labels are rejected."
                        ),
                    },
                ),
            }
        }

    def apply_prose(self, plan_context, editable_prose: str):
        prompt, compiled, report, length, canonical = apply_editable_prose(
            plan_context, editable_prose
        )
        return prompt, compiled, report, length, canonical


class MiniMaxH3PlanV2PromptEnhancer:
    """Ask Qwen for prose JSON, then reconstruct and validate the final H3 prompt."""

    CATEGORY = "MiniMax H3/Plan v2"
    FUNCTION = "enhance_plan"
    RETURN_TYPES = (
        "STRING",
        "STRING",
        PLAN_TYPE,
        "STRING",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "enhanced_prompt",
        "editable_prose",
        "enhanced_plan_context",
        "base_system_prompt",
        "effective_system_prompt",
        "llm_prompt",
        "enhancer_report",
    )
    OUTPUT_TOOLTIPS = (
        "Final prompt rebuilt and validated by Python. Intent-locked mode preserves the source wording and appends only accepted details.",
        "Complete validated prose JSON after the selected mode is applied. Connect it to Apply Structured Prose for manual refinement.",
        "Recompiled locked plan matching enhanced_prompt.",
        "Editable system instructions exactly as resolved from the widget.",
        "Actual system instructions sent to Qwen: base prompt plus the selected mode and non-editable schema contract.",
        "Complete Qwen chat template for debugging or use in another compatible LLM node.",
        "Selected-mode, generation, visual-analysis, fallback, validation, tail, and CLIP-residency safety status.",
    )
    DESCRIPTION = (
        "Structured Plan v2 production-detail enhancer. Intent-locked mode keeps every original "
        "prose field and appends Qwen details; Creative mode permits a full rewrite. Python "
        "restores and validates every H3 structural field."
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
                            "A complete generation-capable Qwen3-VL or Qwen3.5 CLIP, or MiniMax H3's "
                            "conditioning CLIP with Generation Tail Loader connected."
                        )
                    },
                ),
                "plan_context": (
                    PLAN_TYPE,
                    {
                        "tooltip": "Connect compiled plan_context from MiniMax H3 Prompt Merge (Plan v2)."
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        **multiline,
                        "default": PLAN_ENHANCER_SYSTEM_PROMPT,
                        "tooltip": (
                            "Editable prose behavior. Blank uses the built-in production-detail pass. "
                            "The node appends a non-editable schema contract and exposes both versions."
                        ),
                    },
                ),
                "visual_analysis": (
                    VISUAL_ANALYSIS_MODES,
                    {
                        "default": VISUAL_ANALYSIS_DISABLED,
                        "tooltip": (
                            "Disabled is safest and lowest-memory. Enabled attaches image pixels and "
                            "sampled video frames. Audio is never inferred from waveform data."
                        ),
                    },
                ),
                "analysis_long_edge": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 1536,
                        "step": 32,
                        "tooltip": "Maximum long edge for Qwen visual evidence; hidden when analysis is disabled.",
                    },
                ),
                "video_analysis_fps": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "Generic-Qwen video sampling rate; hidden when analysis is disabled.",
                    },
                ),
                "max_analysis_frames": (
                    "INT",
                    {
                        "default": 8,
                        "min": 2,
                        "max": 32,
                        "step": 1,
                        "tooltip": "Generic-Qwen frame cap per video; hidden when analysis is disabled.",
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 1200,
                        "min": 128,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Maximum generated JSON tokens. Increase for many long Shots.",
                    },
                ),
                "sampling": (
                    ["deterministic", "sample"],
                    {"default": "sample"},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.65, "min": 0.01, "max": 2.0, "step": 0.01},
                ),
                "top_k": (
                    "INT",
                    {"default": 64, "min": 0, "max": 1000, "step": 1},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "min_p": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.02, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "presence_penalty": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.01},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Off is recommended for direct schema-following output.",
                    },
                ),
            },
            "optional": {
                "clip_tail": (
                    TAIL_TYPE,
                    {
                        "tooltip": (
                            "Connect Generation Tail Loader only for MiniMax's truncated "
                            "50-layer conditioning CLIP. The tail is always temporary."
                        )
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
                "enhancement_mode": (
                    ENHANCEMENT_MODES,
                    {
                        "default": ENHANCEMENT_MODE_INTENT_LOCKED,
                        "tooltip": (
                            "Intent-locked expansion (recommended) preserves your original prose "
                            "verbatim and lets Qwen append only compatible production detail. "
                            "Creative expansion lets Qwen rewrite the editable prose while Python "
                            "still locks references, timing, dialogue, roles, and routes."
                        ),
                    },
                ),
            },
        }

    def enhance_plan(
        self,
        clip,
        plan_context,
        system_prompt: str,
        visual_analysis: str,
        analysis_long_edge: int,
        video_analysis_fps: float,
        max_analysis_frames: int,
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
        offload_after_generation: bool = False,
        enhancement_mode: str = ENHANCEMENT_MODE_INTENT_LOCKED,
    ):
        if clip is None:
            raise RuntimeError("A generation-capable CLIP input is required.")
        if visual_analysis not in VISUAL_ANALYSIS_MODES:
            raise ValueError("Choose a supported visual_analysis mode.")
        enhancement_mode = _validate_enhancement_mode(enhancement_mode)

        base_prompt, compiler_report, base_compiled, _h3_length = _base_package(
            plan_context
        )
        fallback_json = editable_prose_json(base_compiled)
        base_system = str(system_prompt or "").strip() or PLAN_ENHANCER_SYSTEM_PROMPT
        effective_system = _effective_system_prompt(
            base_system, base_compiled, enhancement_mode
        )
        tail_name = _resolve_tail_name(clip_tail)
        validate_clip_compatibility(clip, tail_name)
        minimax_clip = _is_minimax_h3_tokenizer(clip)
        mode = base_compiled["compiled"]["mode"]
        entries = (
            _analysis_entries(
                base_compiled,
                analysis_long_edge,
                video_analysis_fps,
                max_analysis_frames,
            )
            if visual_analysis == VISUAL_ANALYSIS_ENABLED
            else []
        )
        minimax_reference_payload = minimax_clip and bool(entries) and mode == "Ref2VA"
        generic_images, minimax_items, visual_map = _prepare_reference_visuals(
            entries, minimax_reference_payload
        )
        user_prompt = _llm_user_prompt(
            base_compiled,
            base_prompt,
            compiler_report,
            visual_analysis,
            enhancement_mode,
        )
        if visual_map and not minimax_clip:
            user_prompt += "\n\nGENERIC QWEN VISUAL ORDER\n" + visual_map
        llm_prompt = format_chat_prompt(
            effective_system,
            user_prompt,
            has_image=bool(entries),
            minimax_clip=minimax_clip,
            visual_count=len(generic_images) if entries and not minimax_clip else 0,
        )
        if not thinking:
            llm_prompt += "<think>\n\n</think>\n\n"

        compatibility_issue = clip_generation_issue(clip) if tail_name is None else None
        if compatibility_issue:
            return (
                base_prompt,
                fallback_json,
                base_compiled,
                base_system,
                effective_system,
                llm_prompt,
                f"Structured enhancement skipped in {enhancement_mode}. "
                f"{compatibility_issue}",
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
                use_minimax_image_path=minimax_clip and bool(entries),
            )
        else:
            generated_ids = _generate_with_tail(
                clip, tail_name, tokens, generation_options
            )
        raw_output = clip.decode(generated_ids)

        collapse = generation_collapse_reason(raw_output)
        fallback_reason = collapse or ""
        repair_note = ""
        if not fallback_reason:
            try:
                (
                    enhanced_prompt,
                    enhanced_plan,
                    apply_report,
                    _length,
                    canonical,
                    mode_report,
                ) = _apply_generated_prose(base_compiled, raw_output, enhancement_mode)
            except (ValueError, RuntimeError) as error:
                if "<d> dialogue tags" not in str(error):
                    fallback_reason = str(error)
                else:
                    repaired_output, removed = _repair_model_dialogue_leak(raw_output)
                    if not removed:
                        fallback_reason = str(error)
                    else:
                        try:
                            (
                                enhanced_prompt,
                                enhanced_plan,
                                apply_report,
                                _length,
                                canonical,
                                mode_report,
                            ) = _apply_generated_prose(
                                base_compiled, repaired_output, enhancement_mode
                            )
                        except (ValueError, RuntimeError) as repair_error:
                            fallback_reason = (
                                f"{error} Automatic dialogue-leak repair also failed: "
                                f"{repair_error}"
                            )
                        else:
                            repair_note = (
                                f" Qwen copied {removed} compiler-owned dialogue segment(s); "
                                "the node removed them before validation and the locked compiler "
                                "restored the exact dialogue once."
                            )

        if fallback_reason:
            enhanced_prompt = base_prompt
            enhanced_plan = base_compiled
            canonical = fallback_json
            report = (
                f"Structured enhancement fallback in {enhancement_mode}: "
                + fallback_reason
                + " The deterministic compiler draft and valid original prose JSON were returned."
            )
        else:
            media_note = (
                f" Qwen analyzed {sum(entry['kind'] == 'image' for entry in entries)} image(s) "
                f"and {sum(entry['kind'] == 'video' for entry in entries)} sampled video(s)."
                if entries
                else " Qwen used text metadata only; no pixels or audio waveforms were attached."
            )
            tail_note = (
                " The temporary MiniMax generation tail was used and unloaded."
                if tail_name is not None
                else " The connected complete CLIP generated the prose patch."
            )
            report = (
                mode_report
                + "\n\n"
                + apply_report
                + media_note
                + tail_note
                + repair_note
            )

        offload_note = _clip_offload_report(clip_offload_status)
        if offload_note:
            report += " " + offload_note
        return (
            enhanced_prompt,
            canonical,
            enhanced_plan,
            base_system,
            effective_system,
            llm_prompt,
            report,
        )


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PlanV2ApplyProse": MiniMaxH3PlanV2ApplyProse,
    "MiniMaxH3PlanV2PromptEnhancer": MiniMaxH3PlanV2PromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PlanV2ApplyProse": "MiniMax H3 Apply Structured Prose (Plan v2)",
    "MiniMaxH3PlanV2PromptEnhancer": "MiniMax H3 Structured Prompt Enhancer (Plan v2)",
}
