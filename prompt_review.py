"""Interactive, prompt-only approval gate for compiled MiniMax H3 Plan v2 flows.

The node deliberately keeps the media-bearing plan in memory and sends only
prompt text plus small history metadata to the browser.  Manual edits are
accepted only when the compiler-owned H3 structure still matches the connected
plan.  The native adapter independently verifies the approval before using an
edited prompt.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import uuid
from typing import Any, Callable

try:
    from .plan_v2 import (
        MODE_REF2VA,
        PHASE_COMPILED,
        PHASE_SETUP,
        PHASE_TIMELINE,
        PLAN_TYPE,
        _catalog,
        _copy_plan,
        _dialogue_text,
        _format_timestamp,
        _replacement_scope,
        _replacement_shot_instruction,
        _transition_prefix,
        compile_h3_plan,
        validated_plan,
    )
except ImportError:
    from plan_v2 import (
        MODE_REF2VA,
        PHASE_COMPILED,
        PHASE_SETUP,
        PHASE_TIMELINE,
        PLAN_TYPE,
        _catalog,
        _copy_plan,
        _dialogue_text,
        _format_timestamp,
        _replacement_scope,
        _replacement_shot_instruction,
        _transition_prefix,
        compile_h3_plan,
        validated_plan,
    )


REVIEW_APPROVAL_VERSION = 1
REVIEW_MODE_PAUSE = "Pause for approval"
REVIEW_MODE_PASSTHROUGH = "Pass through without pausing"
REVIEW_MODES = [REVIEW_MODE_PAUSE, REVIEW_MODE_PASSTHROUGH]
MAX_ACTIVE_REVIEWS = 8
MAX_PROMPT_CHARACTERS = 262_144
MAX_HISTORY_FILE_BYTES = 16 * 1024 * 1024

_APPROVAL_KEY = "prompt_review_approval"
_HISTORY_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
_REFERENCE_RE = re.compile(
    r"<\s*(Subject|Picture|Video|Audio)\s+(\d+)\s*>", re.IGNORECASE
)
_REFERENCE_START_RE = re.compile(
    r"<\s*(?:Subject|Picture|Video|Audio)\b", re.IGNORECASE
)
_SHOT_TOKEN_RE = re.compile(r"\[\s*Shot\s+(\d+)\s*\]", re.IGNORECASE)
_SHOT_START_RE = re.compile(r"\[\s*Shot\b", re.IGNORECASE)
_SHOT_LINE_RE = re.compile(
    r"(?im)^\[\s*Shot\s+(\d+)\s*\](?=\s)"
)
_DIALOGUE_RE = re.compile(r"<d>.*?</d>", re.IGNORECASE | re.DOTALL)
_DIALOGUE_TAG_RE = re.compile(r"</?d>", re.IGNORECASE)
_DIALOGUE_TAG_START_RE = re.compile(r"<\s*/?\s*d\b", re.IGNORECASE)
_DURATION_RE = re.compile(r"\blasts\s+\d+(?:\.\d+)?\s+seconds\b", re.IGNORECASE)

_REF_HEADERS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
_ENDPOINT_HEADERS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)
_KNOWN_HEADERS = _REF_HEADERS + (_ENDPOINT_HEADERS[0],)
_HEADER_RE = re.compile(
    r"(?m)^(" + "|".join(re.escape(value) for value in _KNOWN_HEADERS) + r"):[ \t]*"
)


def _normalized_prompt(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tensor_free_plan(plan_context: Any) -> dict:
    """Return all approval-relevant plan data without runtime media or approvals."""

    plan = validated_plan(plan_context, allowed_phases={PHASE_COMPILED})
    return {
        "version": plan["version"],
        "phase": plan["phase"],
        "project": dict(plan["project"]),
        "target": {
            key: value
            for key, value in plan.get("target", {}).items()
            if key != "media"
        },
        "assets": [
            {key: value for key, value in asset.items() if key != "media"}
            for asset in plan["assets"]
        ],
        "bindings": [dict(entry) for entry in plan["bindings"]],
        "character_replacements": [
            dict(entry) for entry in plan["character_replacements"]
        ],
        "audio_relationships": [
            dict(entry) for entry in plan["audio_relationships"]
        ],
        "shots": [dict(entry) for entry in plan["shots"]],
        "dialogue_events": [dict(entry) for entry in plan["dialogue_events"]],
        "compiled": dict(plan["compiled"]),
    }


def plan_review_signature(plan_context: Any) -> str:
    payload = json.dumps(
        _tensor_free_plan(plan_context),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return _sha256(payload)


def _canonical_package(plan_context: Any) -> tuple[dict, str, int]:
    compiled = validated_plan(plan_context, allowed_phases={PHASE_COMPILED})
    source = _copy_plan(compiled)
    source["phase"] = PHASE_TIMELINE if source["shots"] else PHASE_SETUP
    source.pop("compiled", None)
    source.pop(_APPROVAL_KEY, None)
    prompt, _rewrite, _report, canonical, length = compile_h3_plan(source)
    if compiled.get("compiled") != canonical.get("compiled"):
        raise ValueError(
            "plan_context contains stale compiler labels or native routes. Re-run "
            "MiniMax H3 Prompt Merge before reviewing it."
        )
    return compiled, prompt, length


def _split_sections(prompt: str, expected_headers: tuple[str, ...]) -> tuple[str, dict]:
    matches = list(_HEADER_RE.finditer(prompt))
    names = [match.group(1) for match in matches]
    if tuple(names) != expected_headers:
        raise ValueError(
            "Keep the exact H3 section names and order: "
            + ", ".join(expected_headers)
            + "."
        )
    sections = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        sections[match.group(1)] = prompt[match.end() : end].strip()
    return prompt[: matches[0].start()].strip(), sections


def _reference_tokens(value: str) -> frozenset[str]:
    # Exact spelling is intentional: the review gate is not a label formatter.
    # Repetition inside the same editable region is harmless and useful in long
    # prose, so ownership is set-based rather than occurrence-count-based.
    return frozenset(match.group(0) for match in _REFERENCE_RE.finditer(value))


def _validate_reserved_token_syntax(value: str) -> None:
    for match in _REFERENCE_START_RE.finditer(value):
        if _REFERENCE_RE.match(value, match.start()) is None:
            raise ValueError(
                "Malformed H3 reference label. Use exact forms such as <Subject 1> or <Audio 2>."
            )
    for match in _SHOT_START_RE.finditer(value):
        if _SHOT_TOKEN_RE.match(value, match.start()) is None:
            raise ValueError("Malformed H3 Shot marker. Use the exact form [Shot N].")
    for match in _DIALOGUE_TAG_START_RE.finditer(value):
        if _DIALOGUE_TAG_RE.match(value, match.start()) is None:
            raise ValueError("Malformed H3 dialogue tag. Only exact <d> and </d> tags are valid.")


def _shot_regions(value: str) -> tuple[str, list[tuple[int, str]]]:
    matches = list(_SHOT_LINE_RE.finditer(value))
    if not matches:
        raise ValueError("The detailed prompt must keep every [Shot N] marker.")
    regions = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        regions.append((int(match.group(1)), value[match.start() : end].strip()))
    return value[: matches[0].start()].strip(), regions


def _expected_shot_prefixes(plan: dict) -> list[str]:
    if not plan["shots"]:
        return ["[Shot 1] "]
    prefixes = []
    for shot in plan["shots"]:
        number = int(shot["shot_number"])
        if number == 1:
            prefixes.append("[Shot 1] ")
        else:
            prefixes.append(
                f"[Shot {number}] At {_format_timestamp(shot['cut_at'])}, "
                f"{_transition_prefix(shot['transition'])} "
            )
    return prefixes


def _validate_dialogue_events(
    expected: str,
    candidate: str,
    number: int,
    plan: dict,
    catalog: dict,
) -> None:
    """Lock dialogue payload, Shot ownership, order, and explicit timing anchors."""

    expected_tags = _DIALOGUE_RE.findall(expected)
    candidate_tags = _DIALOGUE_RE.findall(candidate)
    if candidate_tags != expected_tags:
        raise ValueError(
            f"[Shot {number}] must keep its exact <d> dialogue events in their "
            "compiler-managed order."
        )

    cursor = 0
    for event in plan["dialogue_events"]:
        if int(event["shot_number"]) != int(number):
            continue
        compiled_clause = _dialogue_text(event, plan, catalog)
        tag_match = _DIALOGUE_RE.search(compiled_clause)
        if tag_match is None:
            raise ValueError(
                f"[Shot {number}] has an invalid compiler-managed dialogue event."
            )
        tag = tag_match.group(0)
        position = candidate.find(tag, cursor)
        if position < 0:
            raise ValueError(
                f"[Shot {number}] must keep its exact <d> dialogue events in their "
                "compiler-managed order."
            )
        timing_match = re.match(r"At \d{2}:\d{2}\.\d{3},", compiled_clause)
        if timing_match is not None:
            timing_anchor = timing_match.group(0)
            if timing_anchor not in candidate[cursor:position]:
                raise ValueError(
                    f"[Shot {number}] must keep the dialogue timing anchor "
                    f"{timing_anchor!r} before its <d> event."
                )
        cursor = position + len(tag)


def _validate_detailed_section(canonical: str, edited: str, plan: dict) -> None:
    expected_prelude, expected_regions = _shot_regions(canonical)
    edited_prelude, edited_regions = _shot_regions(edited)
    expected_numbers = [number for number, _region in expected_regions]
    edited_numbers = [number for number, _region in edited_regions]
    if edited_numbers != expected_numbers:
        raise ValueError("Shot markers, numbering, and order are compiler-locked.")

    expected_tokens = [match.group(0) for match in _SHOT_TOKEN_RE.finditer(canonical)]
    edited_tokens = [match.group(0) for match in _SHOT_TOKEN_RE.finditer(edited)]
    if edited_tokens != expected_tokens:
        raise ValueError("Do not add, remove, reformat, or move [Shot N] markers.")
    if _DURATION_RE.findall(edited) != _DURATION_RE.findall(canonical):
        raise ValueError("The compiler-derived target duration is locked.")
    if _DIALOGUE_RE.findall(edited) != _DIALOGUE_RE.findall(canonical):
        raise ValueError("Exact H3 dialogue tags, languages, and words are locked.")
    if _reference_tokens(edited_prelude) != _reference_tokens(expected_prelude):
        raise ValueError("A reference label was added, removed, or moved in the scene prelude.")

    prefixes = _expected_shot_prefixes(plan)
    catalog = _catalog(plan)
    for index, ((number, expected), (_edited_number, candidate)) in enumerate(
        zip(expected_regions, edited_regions)
    ):
        if not candidate.startswith(prefixes[index]):
            raise ValueError(
                f"[Shot {number}] must keep its compiler-owned marker, cut timestamp, "
                "and transition."
            )
        if _reference_tokens(candidate) != _reference_tokens(expected):
            raise ValueError(
                f"[Shot {number}] added, removed, renamed, or moved an H3 reference label."
            )
        _validate_dialogue_events(expected, candidate, number, plan, catalog)
        for replacement in plan["character_replacements"]:
            if number not in _replacement_scope(replacement, plan):
                continue
            locked = _replacement_shot_instruction(replacement, plan, catalog)
            if locked not in candidate:
                raise ValueError(
                    f"[Shot {number}] must keep the compiler-managed character-replacement "
                    "instruction unchanged."
                )


def validate_reviewed_prompt(
    plan_context: Any, canonical_prompt: str, reviewed_prompt: str
) -> str:
    """Validate a full H3 prompt edit and return its normalized text."""

    plan = validated_plan(plan_context, allowed_phases={PHASE_COMPILED})
    canonical = _normalized_prompt(canonical_prompt)
    reviewed = _normalized_prompt(reviewed_prompt)
    if not reviewed:
        raise ValueError("The reviewed prompt is empty.")
    if len(reviewed) > MAX_PROMPT_CHARACTERS:
        raise ValueError(
            f"The reviewed prompt exceeds the {MAX_PROMPT_CHARACTERS:,}-character safety limit."
        )
    _validate_reserved_token_syntax(reviewed)

    mode = plan["compiled"].get("mode")
    headers = _REF_HEADERS if mode == MODE_REF2VA else _ENDPOINT_HEADERS
    canonical_preamble, canonical_sections = _split_sections(canonical, headers)
    reviewed_preamble, reviewed_sections = _split_sections(reviewed, headers)
    if reviewed_preamble != canonical_preamble:
        raise ValueError("The endpoint/keyframe alignment preamble is compiler-locked.")

    if mode == MODE_REF2VA:
        for name in ("subject_definitions", "summary", "retention_analysis"):
            if reviewed_sections[name] != canonical_sections[name]:
                raise ValueError(
                    f"{name} is compiler-owned. Edit descriptive scene prose instead."
                )
        detail_name = "detailed_description"
    else:
        detail_name = "integrated_multimodal_description"

    _validate_detailed_section(
        canonical_sections[detail_name], reviewed_sections[detail_name], plan
    )
    for name in ("overall_soundscape", "non_diegetic_music"):
        if _reference_tokens(reviewed_sections[name]) != _reference_tokens(
            canonical_sections[name]
        ):
            raise ValueError(
                f"{name} added, removed, renamed, or moved an H3 reference label."
            )

    canonical_shots = [match.group(0) for match in _SHOT_TOKEN_RE.finditer(canonical)]
    reviewed_shots = [match.group(0) for match in _SHOT_TOKEN_RE.finditer(reviewed)]
    if reviewed_shots != canonical_shots:
        raise ValueError("[Shot N] markers may appear only in their compiler-owned locations.")
    canonical_dialogue_tags = [
        match.group(0) for match in _DIALOGUE_TAG_RE.finditer(canonical)
    ]
    reviewed_dialogue_tags = [
        match.group(0) for match in _DIALOGUE_TAG_RE.finditer(reviewed)
    ]
    if reviewed_dialogue_tags != canonical_dialogue_tags:
        raise ValueError("Do not add, remove, reformat, or move H3 <d> dialogue tags.")

    # This catches labels moved across editable sections even if individual
    # section counters happened to balance in an unexpected future template.
    if _reference_tokens(reviewed) != _reference_tokens(canonical):
        raise ValueError("The reviewed prompt must keep every exact H3 reference label.")
    return reviewed


def approve_reviewed_prompt(
    plan_context: Any, source_prompt: str, reviewed_prompt: str
) -> tuple[str, dict, str]:
    """Create a plan-bound approval for one structurally valid full-prompt edit."""

    compiled, canonical_prompt, _length = _canonical_package(plan_context)
    source = _normalized_prompt(source_prompt)
    canonical = _normalized_prompt(canonical_prompt)
    if source != canonical:
        raise ValueError(
            "The gate received a prompt that does not match plan_context. Connect both outputs "
            "from the same Prompt Merge, Structured Prompt Enhancer, or Apply Structured Prose node."
        )
    reviewed = validate_reviewed_prompt(compiled, canonical, reviewed_prompt)
    approved = _copy_plan(compiled)
    approved[_APPROVAL_KEY] = {
        "version": REVIEW_APPROVAL_VERSION,
        "plan_signature": plan_review_signature(compiled),
        "source_prompt_sha256": _sha256(canonical),
        "approved_prompt_sha256": _sha256(reviewed),
        "edited": reviewed != canonical,
    }
    state = "edited prompt" if reviewed != canonical else "unchanged prompt"
    report = (
        f"Prompt review approved the {state}. Compiler-owned sections, reference labels, "
        "retention, shot structure, cut times, dialogue, character replacements, and native "
        "media routes remain locked. The media-bearing plan_context kept every image, video, "
        "and audio value in place."
    )
    return reviewed, approved, report


def verify_review_approval(
    plan_context: Any, canonical_prompt: str, supplied_prompt: str
) -> None:
    """Independently verify a gate approval at the native adapter boundary."""

    plan = validated_plan(plan_context, allowed_phases={PHASE_COMPILED})
    approval = plan.get(_APPROVAL_KEY)
    if not isinstance(approval, dict) or approval.get("version") != REVIEW_APPROVAL_VERSION:
        raise ValueError(
            "Manual prompt text needs a MiniMax H3 Prompt Review Gate approval."
        )
    canonical = _normalized_prompt(canonical_prompt)
    supplied = _normalized_prompt(supplied_prompt)
    if approval.get("plan_signature") != plan_review_signature(plan):
        raise ValueError("Prompt review approval belongs to a different or modified plan_context.")
    if approval.get("source_prompt_sha256") != _sha256(canonical):
        raise ValueError("Prompt review approval belongs to a different compiler prompt.")
    if approval.get("approved_prompt_sha256") != _sha256(supplied):
        raise ValueError("The prompt changed after it was approved by Prompt Review Gate.")
    validate_reviewed_prompt(plan, canonical, supplied)


def _safe_history_key(value: Any, fallback: str = "") -> str:
    candidate = str(value or "").strip() or fallback
    if not _HISTORY_KEY_RE.fullmatch(candidate):
        raise ValueError(
            "history_key must contain only letters, numbers, underscores, or hyphens."
        )
    return candidate


def _node_history_key(configured_key: Any, unique_id: Any) -> str:
    """Keep copied gates independent while preserving history across workflow reloads."""

    base = _safe_history_key(configured_key, fallback="review")
    node_digest = _sha256(str(unique_id or "unknown"))[:12]
    return _safe_history_key(f"{base[:83]}-{node_digest}")


def _default_history_root() -> Path:
    try:
        import folder_paths

        root = Path(folder_paths.get_user_directory())
    except (ImportError, AttributeError):
        root = Path(tempfile.gettempdir())
    return root / "MiniMax-H3-Guide" / "prompt-review-history"


class PromptReviewHistory:
    """Small atomic JSON history store containing prompt text only."""

    _lock = threading.Lock()

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root is not None else _default_history_root()

    def _path(self, history_key: str) -> Path:
        return self.root / f"{_safe_history_key(history_key)}.json"

    def _load_unlocked(self, history_key: str) -> list[dict]:
        path = self._path(history_key)
        try:
            if path.stat().st_size > MAX_HISTORY_FILE_BYTES:
                return []
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return []
        entries = payload.get("entries", []) if isinstance(payload, dict) else []
        if not isinstance(entries, list):
            return []
        validated = []
        for entry in entries[-50:]:
            if not isinstance(entry, dict):
                continue
            revision = entry.get("revision")
            prompt = entry.get("prompt")
            if (
                isinstance(revision, bool)
                or not isinstance(revision, int)
                or revision < 1
                or not isinstance(prompt, str)
                or len(prompt) > MAX_PROMPT_CHARACTERS
                or not isinstance(entry.get("approved_at"), str)
                or not isinstance(entry.get("edited"), bool)
                or not isinstance(entry.get("source_prompt_sha256"), str)
                or not isinstance(entry.get("approved_prompt_sha256"), str)
                or not isinstance(entry.get("plan_signature"), str)
            ):
                continue
            validated.append(dict(entry))
        return validated

    def load(self, history_key: str) -> list[dict]:
        with self._lock:
            return self._load_unlocked(history_key)

    def append(
        self,
        history_key: str,
        source_prompt: str,
        approved_prompt: str,
        plan_signature: str,
        limit: int,
    ) -> dict:
        key = _safe_history_key(history_key)
        resolved_limit = max(1, min(50, int(limit)))
        with self._lock:
            entries = self._load_unlocked(key)
            entry = {
                "revision": (
                    max(
                        (int(item.get("revision", 0)) for item in entries),
                        default=0,
                    )
                    + 1
                ),
                "approved_at": datetime.now(timezone.utc).isoformat(),
                "edited": approved_prompt != source_prompt,
                "prompt": approved_prompt,
                "source_prompt_sha256": _sha256(source_prompt),
                "approved_prompt_sha256": _sha256(approved_prompt),
                "plan_signature": plan_signature,
            }
            entries.append(entry)
            entries = entries[-resolved_limit:]
            payload = {"version": 1, "history_key": key, "entries": entries}
            serialized = json.dumps(payload, ensure_ascii=False, indent=2)
            while (
                len(serialized.encode("utf-8")) > MAX_HISTORY_FILE_BYTES
                and len(entries) > 1
            ):
                entries.pop(0)
                payload["entries"] = entries
                serialized = json.dumps(payload, ensure_ascii=False, indent=2)
            if len(serialized.encode("utf-8")) > MAX_HISTORY_FILE_BYTES:
                raise ValueError(
                    "The approved prompt is too large for the prompt-review history file."
                )
            self.root.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{key}-", suffix=".tmp", dir=self.root
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self._path(key))
            except Exception:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
        return entry


class PromptReviewCancelled(Exception):
    pass


@dataclass
class _ReviewSession:
    node_id: str
    token: str
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future
    approver: Callable[[str], tuple[str, dict, str]]
    payload: dict | None = None
    resolving: bool = False
    resolved: bool = False


class PromptReviewBus:
    """Thread-safe bridge from aiohttp's loop to the prompt worker's async loop."""

    _lock = threading.Lock()
    _sessions: dict[str, _ReviewSession] = {}

    @staticmethod
    def _finish(session: _ReviewSession, *, result=None, error=None) -> None:
        def resolve():
            if session.future.done():
                return
            if error is not None:
                session.future.set_exception(error)
            else:
                session.future.set_result(result)

        try:
            session.loop.call_soon_threadsafe(resolve)
        except RuntimeError:
            pass

    @classmethod
    def arm(
        cls, node_id: Any, approver: Callable[[str], tuple[str, dict, str]]
    ) -> _ReviewSession:
        sid = str(node_id)
        loop = asyncio.get_running_loop()
        with cls._lock:
            old = cls._sessions.get(sid)
            if old is not None and old.resolving:
                raise RuntimeError(
                    "This prompt-review node already has a decision being validated."
                )
            active_without_same = len(cls._sessions) - (1 if old is not None else 0)
            if active_without_same >= MAX_ACTIVE_REVIEWS:
                raise RuntimeError(
                    f"At most {MAX_ACTIVE_REVIEWS} prompt reviews may wait at once. "
                    "Approve or reject an existing paused job first."
                )
            session = _ReviewSession(
                sid, uuid.uuid4().hex, loop, loop.create_future(), approver
            )
            cls._sessions[sid] = session
        if old is not None:
            cls._finish(old, error=PromptReviewCancelled("Superseded by a newer review run."))
        return session

    @classmethod
    def submit(cls, node_id: Any, token: Any, prompt: Any) -> tuple[bool, str]:
        sid, candidate_token = str(node_id), str(token)
        with cls._lock:
            session = cls._sessions.get(sid)
            if (
                session is None
                or session.token != candidate_token
                or session.resolving
                or session.resolved
            ):
                return False, "This prompt-review run is no longer active."
            session.resolving = True
        try:
            result = session.approver(str(prompt or ""))
        except Exception as error:
            with cls._lock:
                if cls._sessions.get(sid) is session and not session.resolved:
                    session.resolving = False
            return False, str(error)
        with cls._lock:
            current = cls._sessions.get(sid)
            if current is not session or session.resolved:
                return False, "This prompt-review run is no longer active."
            session.resolving = False
            session.resolved = True
        cls._finish(session, result=result)
        return True, "approved"

    @classmethod
    def reject(cls, node_id: Any, token: Any) -> tuple[bool, str]:
        sid, candidate_token = str(node_id), str(token)
        with cls._lock:
            session = cls._sessions.get(sid)
            if (
                session is None
                or session.token != candidate_token
                or session.resolving
                or session.resolved
            ):
                return False, "This prompt-review run is no longer active."
            session.resolved = True
        cls._finish(session, error=PromptReviewCancelled("Prompt review rejected by user."))
        return True, "rejected"

    @classmethod
    def publish(cls, node_id: Any, token: Any, payload: dict) -> bool:
        """Attach the text-only browser payload for reconnect recovery."""

        sid, candidate_token = str(node_id), str(token)
        with cls._lock:
            session = cls._sessions.get(sid)
            if session is None or session.token != candidate_token or session.resolved:
                return False
            session.payload = dict(payload)
            return True

    @classmethod
    def active_payloads(cls) -> list[dict]:
        """Return text-only snapshots for re-sending to the execution client."""

        with cls._lock:
            return [
                dict(session.payload)
                for session in cls._sessions.values()
                if session.payload is not None and not session.resolved
            ]

    @classmethod
    def disarm(cls, session: _ReviewSession) -> None:
        with cls._lock:
            if cls._sessions.get(session.node_id) is session:
                cls._sessions.pop(session.node_id, None)


async def _await_review(session: _ReviewSession):
    try:
        import comfy.model_management as model_management
    except ImportError:  # Unit-test/direct-import fallback.
        model_management = None
    while True:
        done, _pending = await asyncio.wait({session.future}, timeout=0.25)
        if done:
            return session.future.result()
        if model_management is not None and model_management.processing_interrupted():
            raise PromptReviewCancelled("ComfyUI interrupted the paused prompt review.")


class MiniMaxH3PlanV2PromptReview:
    """Pause generation for a safe, full-prompt manual review."""

    CATEGORY = "MiniMax H3/Plan v2"
    FUNCTION = "review_prompt"
    RETURN_TYPES = ("STRING", PLAN_TYPE, "STRING")
    RETURN_NAMES = ("approved_prompt", "approved_plan_context", "review_report")
    OUTPUT_TOOLTIPS = (
        "The approved original or manually edited H3 prompt.",
        "The unchanged media-bearing plan plus a plan-bound prompt approval.",
        "Approval status and a summary of the structural locks that were verified.",
    )
    DESCRIPTION = (
        "Prompt-only manual gate. It pauses before native H3 conditioning, displays the full "
        "compiled/enhanced prompt, permits safe descriptive-prose edits, and passes the same "
        "image/video/audio-bearing plan after approval. History stores text only."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_prompt": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": (
                            "Connect Prompt Merge.h3_prompt, Structured Prompt Enhancer.enhanced_prompt, "
                            "or Apply Structured Prose.h3_prompt. No image selection happens here."
                        ),
                    },
                ),
                "plan_context": (
                    PLAN_TYPE,
                    {
                        "tooltip": (
                            "Connect the matching plan_context output. It already contains and routes "
                            "all reference image, video, and audio values."
                        )
                    },
                ),
                "review_mode": (
                    REVIEW_MODES,
                    {
                        "default": REVIEW_MODE_PAUSE,
                        "tooltip": (
                            "Pause for approval opens the prompt editor on every queued run. Pass "
                            "through validates and forwards the pair without waiting."
                        ),
                    },
                ),
                "history_limit": (
                    "INT",
                    {
                        "default": 20,
                        "min": 1,
                        "max": 50,
                        "step": 1,
                        "tooltip": "Maximum approved text revisions saved for this review node.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    async def review_prompt(
        self,
        h3_prompt: str,
        plan_context,
        review_mode: str,
        history_limit: int,
        unique_id=None,
    ):
        if review_mode not in REVIEW_MODES:
            raise ValueError("Choose a supported prompt review mode.")
        source = _normalized_prompt(h3_prompt)
        if review_mode == REVIEW_MODE_PASSTHROUGH:
            reviewed, approved, report = approve_reviewed_prompt(
                plan_context, source, source
            )
            return reviewed, approved, "Prompt review bypassed without pausing. " + report

        # The ComfyUI node id is already stable within a saved workflow and unique
        # after copy/paste. Keeping history identity out of widgets avoids custom DOM
        # widgets shifting serialized review_mode/history_limit values.
        key = _node_history_key("review", unique_id)
        history = PromptReviewHistory()

        def approve(candidate: str):
            reviewed, approved, report = approve_reviewed_prompt(
                plan_context, source, candidate
            )
            try:
                history.append(
                    key,
                    source,
                    reviewed,
                    plan_review_signature(approved),
                    history_limit,
                )
            except (OSError, TypeError, ValueError) as error:
                report += f" Prompt approved, but text history could not be saved: {error}."
            return reviewed, approved, report

        # Validate the source pair before opening a browser editor.
        _compiled, canonical, _length = _canonical_package(plan_context)
        if source != _normalized_prompt(canonical):
            raise ValueError(
                "Prompt Review Gate received a mismatched h3_prompt and plan_context pair."
            )

        session = PromptReviewBus.arm(unique_id, approve)
        try:
            try:
                if __package__:
                    from .prompt_review_server import send_prompt_review
                else:
                    from prompt_review_server import send_prompt_review
                send_prompt_review(
                    unique_id,
                    session.token,
                    source,
                    key,
                    history.load(key),
                )
                return await _await_review(session)
            except PromptReviewCancelled as error:
                try:
                    import comfy.model_management as model_management
                except ImportError:
                    raise error
                raise model_management.InterruptProcessingException() from error
        finally:
            PromptReviewBus.disarm(session)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PlanV2PromptReview": MiniMaxH3PlanV2PromptReview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PlanV2PromptReview": "MiniMax H3 Prompt Review Gate (Plan v2)",
}
