"""ComfyUI routes and socket event for the Plan v2 prompt review gate."""

from aiohttp import web
from server import PromptServer

try:
    from .prompt_review import PromptReviewBus
except ImportError:
    from prompt_review import PromptReviewBus


routes = PromptServer.instance.routes


def _display_id(server, node_id):
    return str(getattr(server, "last_node_id", None) or node_id)


def send_prompt_review(
    node_id, token, prompt, history_key, history, timeout_seconds=0
):
    server = PromptServer.instance
    payload = {
        "id": str(node_id),
        "display_id": _display_id(server, node_id),
        "token": token,
        "prompt": prompt,
        "history_key": history_key,
        "history": history,
        "timeout_seconds": timeout_seconds,
    }
    if PromptReviewBus.publish(node_id, token, payload):
        server.send_sync(
            "minimax-h3-prompt-review",
            payload,
            getattr(server, "client_id", None),
        )


def send_passthrough_prompt(node_id, prompt):
    """Display a prompt without creating a review session or blocking the queue."""

    server = PromptServer.instance
    server.send_sync(
        "minimax-h3-prompt-review",
        {
            "id": str(node_id),
            "display_id": _display_id(server, node_id),
            "prompt": prompt,
            "passthrough": True,
        },
        getattr(server, "client_id", None),
    )


def send_prompt_review_timeout(node_id, token, timeout_seconds):
    """Tell the editor that its unanswered review continued automatically."""

    server = PromptServer.instance
    server.send_sync(
        "minimax-h3-prompt-review-timeout",
        {
            "id": str(node_id),
            "display_id": _display_id(server, node_id),
            "token": token,
            "timeout_seconds": timeout_seconds,
        },
        getattr(server, "client_id", None),
    )


@routes.post("/minimax_h3/prompt_review/recover")
async def recover_prompt_reviews(_request):
    """Re-send active reviews to the execution client without HTTP prompt egress."""

    server = PromptServer.instance
    payloads = PromptReviewBus.active_payloads()
    for payload in payloads:
        server.send_sync(
            "minimax-h3-prompt-review",
            payload,
            getattr(server, "client_id", None),
        )
    return web.json_response({"status": "sent", "count": len(payloads)})


@routes.post("/minimax_h3/prompt_review/decision")
async def prompt_review_decision(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Expected a JSON request body."}, status=400)
    if not isinstance(body, dict):
        return web.json_response({"error": "Expected a JSON object request body."}, status=400)
    node_id = body.get("id")
    token = body.get("token")
    action = body.get("action")
    if node_id is None or token is None:
        return web.json_response({"error": "Missing review node id or token."}, status=400)
    if action == "reject":
        accepted, message = PromptReviewBus.reject(node_id, token)
    elif action == "approve":
        if not isinstance(body.get("prompt"), str):
            return web.json_response({"error": "Missing reviewed prompt text."}, status=400)
        accepted, message = PromptReviewBus.submit(node_id, token, body["prompt"])
    else:
        return web.json_response({"error": "Action must be approve or reject."}, status=400)
    if not accepted:
        return web.json_response({"error": message}, status=409)
    return web.json_response({"status": message})
