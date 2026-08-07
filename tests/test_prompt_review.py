import asyncio
import json
from pathlib import Path
import threading

import pytest
import torch

import prompt_review as prompt_review_module
from plan_adapter import prepare_native_h3_call
from plan_v2 import (
    CONTENT_IDENTITY,
    IMAGE_DEFINE_VISIBLE,
    RETENTION_AUTO,
    MiniMaxH3PlanV2DialogueEvent,
    MiniMaxH3PlanV2ImageReference,
    MiniMaxH3PlanV2ProjectSetup,
    MiniMaxH3PlanV2Shot,
    compile_h3_plan,
)
from prompt_review import (
    NODE_CLASS_MAPPINGS,
    REVIEW_MODE_PAUSE,
    REVIEW_MODE_PASSTHROUGH,
    MiniMaxH3PlanV2PromptOverride,
    MiniMaxH3PlanV2PromptReview,
    PromptReviewBus,
    PromptReviewCancelled,
    PromptReviewHistory,
    _node_history_key,
    approve_reviewed_prompt,
    plan_review_signature,
    validate_reviewed_prompt,
    verify_review_approval,
)


ROOT = Path(__file__).resolve().parents[1]


def compiled_endpoint():
    plan = MiniMaxH3PlanV2ProjectSetup().start(
        "A fox crosses fresh snow.",
        6.0,
        "cinematic, live-action",
        "Wind through trees and synchronized footsteps.",
        "N/A",
    )[0]
    prompt, _rewrite, _report, compiled, _length = compile_h3_plan(plan)
    return prompt, compiled


def compiled_reference_scene():
    plan = MiniMaxH3PlanV2ProjectSetup().start(
        "A traveler crosses a quiet station and greets the camera.",
        6.0,
        "cinematic, live-action",
        "Quiet station ambience and synchronized footsteps.",
        "N/A",
    )[0]
    plan, _handle, _image, _preview = MiniMaxH3PlanV2ImageReference().add_image(
        plan,
        torch.zeros(1, 32, 48, 3),
        IMAGE_DEFINE_VISIBLE,
        "traveler portrait",
        "The traveler's identity and visible appearance.",
        CONTENT_IDENTITY,
        "traveler",
        RETENTION_AUTO,
        "1,2",
        "",
    )
    plan = MiniMaxH3PlanV2Shot().add_shot(
        plan,
        0.0,
        "<Subject 1> walks through the quiet station.",
        "A stable medium tracking shot.",
        "Direct cut",
    )[0]
    plan = MiniMaxH3PlanV2DialogueEvent().add_dialogue(
        plan,
        "traveler",
        "English",
        "Good morning.",
        "calm and friendly",
        "On-screen speech",
    )[0]
    plan = MiniMaxH3PlanV2Shot().add_shot(
        plan,
        3.0,
        "<Subject 1> stops beside the platform clock.",
        "A close medium composition.",
        "Direct cut",
    )[0]
    prompt, _rewrite, _report, compiled, _length = compile_h3_plan(plan)
    return prompt, compiled


def compiled_inline_dialogue_scene():
    plan = MiniMaxH3PlanV2ProjectSetup().start(
        "A traveler addresses the camera in a quiet station.",
        6.0,
        "cinematic, live-action",
        "Quiet station ambience.",
        "N/A",
    )[0]
    plan, _handle, _image, _preview = MiniMaxH3PlanV2ImageReference().add_image(
        plan,
        torch.zeros(1, 32, 48, 3),
        IMAGE_DEFINE_VISIBLE,
        "traveler portrait",
        "The traveler's identity and visible appearance.",
        CONTENT_IDENTITY,
        "traveler",
        RETENTION_AUTO,
        "1",
        "",
    )
    plan = MiniMaxH3PlanV2Shot().add_shot(
        plan,
        0.0,
        "<Subject 1> looks up. [d] The traveler then walks away.",
        "A stable medium shot.",
        "Direct cut",
    )[0]
    plan = MiniMaxH3PlanV2DialogueEvent().add_dialogue(
        plan,
        "traveler",
        "English",
        "Good morning.",
        "calm and friendly",
        "On-screen speech",
        "Complete in this Shot",
        1.25,
    )[0]
    prompt, _rewrite, _report, compiled, _length = compile_h3_plan(plan)
    return prompt, compiled


def test_descriptive_full_prompt_edit_gets_plan_bound_adapter_approval():
    prompt, compiled = compiled_endpoint()
    edited = prompt.replace(
        "A fox crosses fresh snow.",
        "A fox carefully crosses fresh snow, leaving a visible trail of prints.",
    )

    approved_prompt, approved_plan, report = approve_reviewed_prompt(
        compiled, prompt, edited
    )
    package = prepare_native_h3_call(approved_plan, approved_prompt, 1344, 768)

    assert approved_prompt == edited
    assert package["prompt"] == edited
    assert package["kwargs"]["prompt"] == edited
    assert "structurally approved" in package["report"]
    assert "media-bearing plan" in report
    verify_review_approval(approved_plan, prompt, edited)


def test_inline_prompt_override_outputs_a_matching_adapter_pair():
    prompt, compiled = compiled_endpoint()
    edited = prompt.replace(
        "A fox crosses fresh snow.",
        "A fox cautiously crosses fresh snow while powder drifts through the frame.",
    )

    output, output_plan, report = MiniMaxH3PlanV2PromptOverride().apply_override(
        prompt,
        compiled,
        edited,
        True,
    )
    package = prepare_native_h3_call(output_plan, output, 1344, 768)

    assert output == edited
    assert package["prompt"] == edited
    assert "Inline override applied" in report
    verify_review_approval(output_plan, prompt, edited)


@pytest.mark.parametrize(
    ("override", "enabled", "message"),
    [
        ("", True, "override is empty"),
        ("unused experiment", False, "override is disabled"),
    ],
)
def test_inline_prompt_override_can_bypass_without_deleting_experiment(
    override, enabled, message
):
    prompt, compiled = compiled_endpoint()

    output, output_plan, report = MiniMaxH3PlanV2PromptOverride().apply_override(
        prompt,
        compiled,
        override,
        enabled,
    )

    assert output == prompt
    assert message in report
    verify_review_approval(output_plan, prompt, prompt)


def test_inline_prompt_override_rejects_compiler_owned_changes():
    prompt, compiled = compiled_reference_scene()
    edited = prompt.replace("<Subject 1>", "<Subject 2>", 1)

    with pytest.raises(ValueError, match="subject_definitions is compiler-owned"):
        MiniMaxH3PlanV2PromptOverride().apply_override(
            prompt,
            compiled,
            edited,
            True,
        )


def test_prompt_changed_after_approval_is_rejected_by_adapter():
    prompt, compiled = compiled_endpoint()
    edited = prompt.replace("fox crosses", "fox carefully crosses")
    approved_prompt, approved_plan, _report = approve_reviewed_prompt(
        compiled, prompt, edited
    )

    with pytest.raises(ValueError, match="changed after it was approved"):
        prepare_native_h3_call(
            approved_plan, approved_prompt + " Extra unapproved prose.", 1344, 768
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda prompt: prompt.replace("subject_definitions:", "subjects:"),
            "section names and order",
        ),
        (
            lambda prompt: prompt.replace("fully_preserved", "weak_reference", 1),
            "retention_analysis is compiler-owned",
        ),
        (
            lambda prompt: prompt.replace("[Shot 2] At", "[Shot 4] At"),
            "Shot markers",
        ),
        (
            lambda prompt: prompt.replace("00:03.000", "00:03.500"),
            "cut timestamp",
        ),
        (
            lambda prompt: prompt.replace("Good morning.", "Good evening."),
            "dialogue tags",
        ),
        (
            lambda prompt: prompt.replace("<Subject 1>", "<Subject 2>", 1),
            "subject_definitions is compiler-owned",
        ),
    ],
)
def test_compiler_owned_h3_structure_cannot_be_manually_edited(mutation, message):
    prompt, compiled = compiled_reference_scene()

    with pytest.raises(ValueError, match=message):
        validate_reviewed_prompt(compiled, prompt, mutation(prompt))


def test_labels_cannot_move_between_shots_but_shot_prose_can_change():
    prompt, compiled = compiled_reference_scene()
    safe = prompt.replace(
        "walks through the quiet station",
        "walks slowly through the quiet station while her steps remain visible",
    )
    assert validate_reviewed_prompt(compiled, prompt, safe) == safe

    moved = prompt.replace(
        "<Subject 1> walks through the quiet station.",
        "<Subject 1> <Subject 1> walks through the quiet station.",
    ).replace(
        "<Subject 1> stops beside the platform clock.",
        "The traveler stops beside the platform clock.",
    )
    with pytest.raises(ValueError, match="Shot 2.*reference label"):
        validate_reviewed_prompt(compiled, prompt, moved)


def test_inline_dialogue_locks_payload_without_freezing_generated_prose_or_timing():
    prompt, compiled = compiled_inline_dialogue_scene()
    safe = prompt.replace(
        "At 00:01.250, ",
        "",
    ).replace(
        "looks up.",
        "looks up and briefly meets the camera's gaze.",
    ).replace(
        "The traveler then walks away.",
        "The traveler then turns and walks slowly away.",
    ).replace(
        "<Subject 1> (S1) speaks:",
        "<Subject 1> (S1) delivers the line:",
    ).replace(
        "Delivery: calm and friendly.",
        "Delivery: warm, relaxed, and friendly.",
    )

    assert validate_reviewed_prompt(compiled, prompt, safe) == safe


def test_dialogue_event_cannot_move_to_another_shot_during_review():
    prompt, compiled = compiled_reference_scene()
    tag = "<d>[English] Good morning.</d>"
    moved = prompt.replace(tag, "", 1).replace(
        "<Subject 1> stops beside the platform clock.",
        f"<Subject 1> stops beside the platform clock. {tag}",
        1,
    )

    with pytest.raises(ValueError, match=r"\[Shot 1\].*<d> dialogue events"):
        validate_reviewed_prompt(compiled, prompt, moved)


def test_reference_labels_may_repeat_and_descriptive_action_timing_may_be_added():
    prompt, compiled = compiled_reference_scene()
    reviewed = prompt.replace(
        "<Subject 1> stops beside the platform clock.",
        "<Subject 1> stops beside the platform clock; at 00:04.250, "
        "the camera returns to <Subject 1> for the reaction.",
    )

    assert validate_reviewed_prompt(compiled, prompt, reviewed) == reviewed


def test_review_keeps_runtime_reference_media_and_native_route_allocation():
    prompt, compiled = compiled_reference_scene()
    image = compiled["assets"][0]["media"]
    edited = prompt.replace(
        "walks through the quiet station",
        "walks carefully through the quiet station",
    )

    approved_prompt, approved_plan, _report = approve_reviewed_prompt(
        compiled, prompt, edited
    )
    package = prepare_native_h3_call(approved_plan, approved_prompt, 1344, 768)

    assert approved_plan["assets"][0]["media"] is image
    assert package["kwargs"]["ref_images"]["ref_image_0"] is image


def test_history_is_text_only_atomic_and_capped(tmp_path):
    history = PromptReviewHistory(tmp_path)
    prompt, compiled = compiled_endpoint()
    signature = plan_review_signature(compiled)
    for index in range(4):
        history.append(
            "review-safe-key",
            prompt,
            prompt.replace("fresh snow", f"fresh snow revision {index}"),
            signature,
            limit=2,
        )

    entries = history.load("review-safe-key")
    raw = (tmp_path / "review-safe-key.json").read_text(encoding="utf-8")

    assert [entry["revision"] for entry in entries] == [3, 4]
    assert all(set(entry) == {
        "revision",
        "approved_at",
        "edited",
        "prompt",
        "source_prompt_sha256",
        "approved_prompt_sha256",
        "plan_signature",
    } for entry in entries)
    assert "waveform" not in raw and "media" not in raw
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads(raw)["history_key"] == "review-safe-key"


def test_corrupt_history_entries_are_ignored_and_do_not_block_new_revision(tmp_path):
    path = tmp_path / "review-corrupt.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [
                    {"revision": "not-an-integer", "prompt": "broken"},
                    ["not", "an", "entry"],
                ],
            }
        ),
        encoding="utf-8",
    )
    history = PromptReviewHistory(tmp_path)

    assert history.load("review-corrupt") == []
    entry = history.append(
        "review-corrupt", "source", "approved", "signature", limit=20
    )

    assert entry["revision"] == 1
    assert history.load("review-corrupt")[0]["prompt"] == "approved"


def test_history_prunes_old_revisions_before_exceeding_file_byte_cap(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(prompt_review_module, "MAX_HISTORY_FILE_BYTES", 1_600)
    history = PromptReviewHistory(tmp_path)
    for index in range(4):
        history.append(
            "review-byte-cap",
            "source",
            f"latest-{index}-" + ("é" * 350),
            "signature",
            limit=50,
        )

    path = tmp_path / "review-byte-cap.json"
    entries = history.load("review-byte-cap")
    assert path.stat().st_size <= prompt_review_module.MAX_HISTORY_FILE_BYTES
    assert entries[-1]["prompt"].startswith("latest-3-")
    assert len(entries) < 4


def test_copied_review_nodes_get_independent_stable_history_keys():
    first = _node_history_key("shared-widget-key", "24")
    assert first == _node_history_key("shared-widget-key", "24")
    assert first != _node_history_key("shared-widget-key", "25")
    assert len(_node_history_key("x" * 96, "24")) == 96


def test_async_review_bus_resumes_the_waiting_prompt_without_blocking():
    async def scenario():
        session = PromptReviewBus.arm(
            "node-42", lambda prompt: (prompt, {"plan": True}, "approved")
        )
        try:
            accepted, message = PromptReviewBus.submit(
                "node-42", session.token, "edited prompt"
            )
            result = await asyncio.wait_for(session.future, timeout=1.0)
            return accepted, message, result
        finally:
            PromptReviewBus.disarm(session)

    accepted, message, result = asyncio.run(scenario())
    assert accepted is True and message == "approved"
    assert result == ("edited prompt", {"plan": True}, "approved")


def test_failed_review_validation_can_be_corrected_on_the_same_active_run():
    async def scenario():
        def approve(prompt):
            if prompt == "invalid":
                raise ValueError("fix the prompt")
            return prompt, {"plan": True}, "approved"

        session = PromptReviewBus.arm("node-retry", approve)
        try:
            failed = PromptReviewBus.submit("node-retry", session.token, "invalid")
            corrected = PromptReviewBus.submit("node-retry", session.token, "corrected")
            result = await asyncio.wait_for(session.future, timeout=1.0)
            return failed, corrected, result
        finally:
            PromptReviewBus.disarm(session)

    failed, corrected, result = asyncio.run(scenario())
    assert failed == (False, "fix the prompt")
    assert corrected == (True, "approved")
    assert result[0] == "corrected"


def test_review_bus_allows_only_one_concurrent_approval_consumer():
    async def scenario():
        started = threading.Event()
        release = threading.Event()
        calls = []

        def approve(prompt):
            calls.append(prompt)
            started.set()
            assert release.wait(timeout=2.0)
            return prompt, {"plan": True}, "approved"

        session = PromptReviewBus.arm("node-race", approve)
        try:
            first = asyncio.create_task(
                asyncio.to_thread(
                    PromptReviewBus.submit,
                    "node-race",
                    session.token,
                    "first prompt",
                )
            )
            assert await asyncio.to_thread(started.wait, 1.0)
            second = PromptReviewBus.submit(
                "node-race", session.token, "duplicate prompt"
            )
            with pytest.raises(RuntimeError, match="decision being validated"):
                PromptReviewBus.arm("node-race", lambda prompt: (prompt, {}, "new"))
            release.set()
            first_result = await asyncio.wait_for(first, timeout=2.0)
            approved = await asyncio.wait_for(session.future, timeout=1.0)
            return calls, first_result, second, approved
        finally:
            release.set()
            PromptReviewBus.disarm(session)

    calls, first, second, approved = asyncio.run(scenario())
    assert calls == ["first prompt"]
    assert first == (True, "approved")
    assert second == (False, "This prompt-review run is no longer active.")
    assert approved[0] == "first prompt"


def test_review_bus_keeps_a_text_only_payload_available_for_socket_recovery():
    async def scenario():
        session = PromptReviewBus.arm("node-recover", lambda prompt: (prompt, {}, "ok"))
        payload = {
            "id": "node-recover",
            "display_id": "28",
            "token": session.token,
            "prompt": "text only",
            "history": [],
        }
        try:
            assert PromptReviewBus.publish("node-recover", session.token, payload)
            snapshots = PromptReviewBus.active_payloads()
            snapshots[0]["prompt"] = "mutated copy"
            assert PromptReviewBus.active_payloads()[0]["prompt"] == "text only"
            accepted, _message = PromptReviewBus.reject(
                "node-recover", session.token
            )
            with pytest.raises(PromptReviewCancelled):
                await session.future
            return accepted, PromptReviewBus.active_payloads()
        finally:
            PromptReviewBus.disarm(session)

    accepted, snapshots = asyncio.run(scenario())
    assert accepted is True
    assert snapshots == []


def test_pass_through_mode_validates_and_approves_without_arming_a_session():
    prompt, compiled = compiled_endpoint()

    result = asyncio.run(
        MiniMaxH3PlanV2PromptReview().review_prompt(
            prompt,
            compiled,
            REVIEW_MODE_PASSTHROUGH,
            20,
            unique_id="88",
        )
    )

    assert result[0] == prompt
    verify_review_approval(result[1], prompt, prompt)
    assert "bypassed without pausing" in result[2]


@pytest.mark.parametrize(
    "suffix",
    [
        "\n<d>unclosed compiler tag",
        "\n[Shot 99] invalid marker outside the timeline",
    ],
)
def test_compiler_tokens_cannot_be_smuggled_into_audio_sections(suffix):
    prompt, compiled = compiled_endpoint()
    reviewed = prompt.replace("N/A", "N/A" + suffix)

    with pytest.raises(ValueError, match="dialogue tags|Shot N"):
        validate_reviewed_prompt(compiled, prompt, reviewed)


@pytest.mark.parametrize(
    "malformed",
    [
        "<Subject one>",
        "<Audio 2 extra>",
        "[Shot final]",
        "<d emphasis>",
        "</ d>",
    ],
)
def test_malformed_reserved_h3_tokens_are_rejected(malformed):
    prompt, compiled = compiled_endpoint()
    reviewed = prompt.replace("N/A", f"N/A\n{malformed}")

    with pytest.raises(ValueError, match="Malformed H3"):
        validate_reviewed_prompt(compiled, prompt, reviewed)


def test_review_node_and_frontend_contract_are_registered():
    assert set(NODE_CLASS_MAPPINGS) == {
        "MiniMaxH3PlanV2PromptOverride",
        "MiniMaxH3PlanV2PromptReview",
    }
    override_schema = MiniMaxH3PlanV2PromptOverride.INPUT_TYPES()
    assert override_schema["required"]["h3_prompt"][1]["forceInput"] is True
    assert override_schema["required"]["override_prompt"][1]["default"] == ""
    assert override_schema["required"]["use_override"][1]["default"] is True
    assert MiniMaxH3PlanV2PromptOverride.RETURN_NAMES[:2] == (
        "h3_prompt",
        "plan_context",
    )
    schema = MiniMaxH3PlanV2PromptReview.INPUT_TYPES()
    assert schema["required"]["h3_prompt"][1]["forceInput"] is True
    assert schema["required"]["review_mode"][1]["default"] == REVIEW_MODE_PAUSE
    assert "history_key" not in schema["required"]
    assert schema["hidden"] == {"unique_id": "UNIQUE_ID"}
    assert MiniMaxH3PlanV2PromptReview.RETURN_NAMES[:2] == (
        "approved_prompt",
        "approved_plan_context",
    )

    source = (ROOT / "web" / "prompt_review.js").read_text(encoding="utf-8")
    assert 'const NODE = "MiniMaxH3PlanV2PromptReview"' in source
    assert "Approve edited prompt & continue" in source
    assert "Restore input" in source
    assert "Reject run" in source
    assert "minimax-h3-prompt-review" in source
    assert "resolvedToken" in source
    assert "prompt_review/recover" in source
    assert "recoveryAgain" in source
    assert f'const PASS_THROUGH_MODE = "{REVIEW_MODE_PASSTHROUGH}"' in source
    assert 'const SETTINGS_PROPERTY = "minimax_h3_prompt_review_settings"' in source
    assert 'const UI_STATE_ID_PROPERTY = "minimax_h3_prompt_review_ui_state_id"' in source
    assert "const reviewStateCache = new Map()" in source
    assert "cacheReviewState(node)" in source
    assert "restoreCachedReviewState(node)" in source
    assert "serializeReviewState(this, serialized)" in source
    assert "node.__h3ReviewWidget.serialize = false" in source
    assert "serialized.widgets_values = [settings.review_mode, settings.history_limit]" in source
    assert "serialized.widgets_values_named = {" in source
    assert "const settings = configuredSettings(this, arguments[0])" in source
    plan_source = (ROOT / "web" / "plan_v2.js").read_text(encoding="utf-8")
    assert 'const PROMPT_OVERRIDE = "MiniMaxH3PlanV2PromptOverride"' in plan_source
    assert "Inline prompt experiment · plan-bound structural validation" in plan_source
    assert "if (className(node) === PROMPT_REVIEW) return;" in plan_source
    assert "place [d] where the next Dialogue Event must appear" in plan_source
    server_source = (ROOT / "prompt_review_server.py").read_text(encoding="utf-8")
    assert '"prompt": prompt' in server_source
    assert "plan_context" not in server_source
    assert '@routes.post("/minimax_h3/prompt_review/recover")' in server_source
    assert '@routes.get("/minimax_h3/prompt_review/active")' not in server_source
    assert "if not isinstance(body, dict):" in server_source
