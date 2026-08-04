"""Native MiniMax H3 conditioning adapter for compiled Plan v2 workflows.

The semantic compiler owns reference meaning and canonical media order.  This
module is deliberately a narrow bridge to ComfyUI's native H3 conditioning
nodes: it verifies that the prompt and compiled plan still match, converts the
route table into the native dictionaries, then calls the installed native
implementation instead of copying its conditioning logic.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

try:
    from .plan_v2 import (
        MODE_FL2VA,
        MODE_I2VA,
        MODE_L2VA,
        MODE_REF2VA,
        MODE_T2VA,
        PHASE_COMPILED,
        PHASE_SETUP,
        PHASE_TIMELINE,
        PLAN_TYPE,
        _copy_plan,
        compile_h3_plan,
        validated_plan,
    )
except ImportError:
    from plan_v2 import (
        MODE_FL2VA,
        MODE_I2VA,
        MODE_L2VA,
        MODE_REF2VA,
        MODE_T2VA,
        PHASE_COMPILED,
        PHASE_SETUP,
        PHASE_TIMELINE,
        PLAN_TYPE,
        _copy_plan,
        compile_h3_plan,
        validated_plan,
    )


NATIVE_H3_MODULE = "comfy_extras.nodes_minimax_h3"
NATIVE_H3_CONTRACT = "2026-08-04 autogrow H3 API"
ENDPOINT_NODE_ID = "MiniMaxH3ImageToVideo"
REFERENCE_NODE_ID = "MiniMaxH3ReferenceToVideo"

_ENDPOINT_MODES = {MODE_T2VA, MODE_I2VA, MODE_L2VA, MODE_FL2VA}
_NATIVE_PARAMETERS = {
    ENDPOINT_NODE_ID: {
        "clip",
        "vae",
        "prompt",
        "width",
        "height",
        "length",
        "first_frame",
        "last_frame",
    },
    REFERENCE_NODE_ID: {
        "clip",
        "vae",
        "audio_vae",
        "prompt",
        "width",
        "height",
        "length",
        "ref_image_size",
        "ref_images",
        "ref_videos",
        "ref_video_audios",
        "ref_audios",
    },
}


def _normalized_prompt(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _canonical_package(plan_context: Any) -> tuple[dict, str, int]:
    """Recompile a compiled plan and reject stale/tampered route metadata."""

    compiled = validated_plan(plan_context, allowed_phases={PHASE_COMPILED})
    source = _copy_plan(compiled)
    source["phase"] = PHASE_TIMELINE if source["shots"] else PHASE_SETUP
    source.pop("compiled", None)
    prompt, _rewrite, _report, canonical, h3_length = compile_h3_plan(source)
    if compiled.get("compiled") != canonical.get("compiled"):
        raise ValueError(
            "plan_context contains stale native labels or routes. Re-run MiniMax H3 "
            "Prompt Merge (Plan v2) and reconnect its current plan_context output."
        )
    return compiled, prompt, h3_length


def _target_size(width: Any, height: Any) -> tuple[int, int]:
    try:
        resolved_width, resolved_height = int(width), int(height)
    except (TypeError, ValueError) as error:
        raise ValueError("H3 width and height must be whole numbers.") from error
    if (
        resolved_width < 32
        or resolved_height < 32
        or resolved_width % 32
        or resolved_height % 32
    ):
        raise ValueError("H3 width and height must be positive multiples of 32.")
    return resolved_width, resolved_height


def _route_payload(compiled: dict, mode: str) -> tuple[dict, bool]:
    assets = {asset["asset_id"]: asset for asset in compiled["assets"]}
    routes = compiled["compiled"].get("routes")
    if not isinstance(routes, list) or not all(
        isinstance(route, dict) for route in routes
    ):
        raise ValueError("plan_context contains an invalid native route table.")

    endpoint: dict[str, Any] = {"first_frame": None, "last_frame": None}
    references: dict[str, dict[str, Any]] = {
        "ref_images": {},
        "ref_videos": {},
        "ref_video_audios": {},
        "ref_audios": {},
    }
    seen_assets: set[str] = set()
    seen_routes: set[str] = set()

    for entry in routes:
        asset_id = entry.get("asset_id")
        route = entry.get("route")
        if asset_id not in assets or not isinstance(route, str):
            raise ValueError(
                "plan_context contains a route for unknown reference media."
            )
        if asset_id in seen_assets or route in seen_routes:
            raise ValueError(
                "plan_context routes one reference asset or socket more than once."
            )
        seen_assets.add(asset_id)
        seen_routes.add(route)

        asset = assets[asset_id]
        media_kind = asset.get("media_kind")
        media = asset.get("media")
        if media is None:
            raise ValueError(
                f"Reference asset {asset.get('reference_name', asset_id)!r} has no runtime media. "
                "Queue from the connected Image, Video, or Audio Reference node."
            )
        if entry.get("media_kind") != media_kind:
            raise ValueError("plan_context contains a stale route media type.")

        if route in endpoint:
            if mode not in _ENDPOINT_MODES or media_kind != "image":
                raise ValueError(
                    f"Native endpoint route {route!r} needs an endpoint image plan."
                )
            endpoint[route] = media
        elif route.startswith("ref_image_"):
            if mode != MODE_REF2VA or media_kind != "image":
                raise ValueError(f"Native route {route!r} needs a Ref2VA image.")
            references["ref_images"][route] = media
        elif route.startswith("ref_video_audio_"):
            if mode != MODE_REF2VA or media_kind != "audio":
                raise ValueError(
                    f"Native route {route!r} needs a paired Ref2VA audio clip."
                )
            references["ref_video_audios"][route] = media
        elif route.startswith("ref_video_"):
            if mode != MODE_REF2VA or media_kind != "video":
                raise ValueError(
                    f"Native route {route!r} needs a Ref2VA video frame batch."
                )
            references["ref_videos"][route] = media
        elif route.startswith("ref_audio_"):
            if mode != MODE_REF2VA or media_kind != "audio":
                raise ValueError(
                    f"Native route {route!r} needs a standalone Ref2VA audio clip."
                )
            references["ref_audios"][route] = media
        else:
            raise ValueError(f"Unsupported native H3 route {route!r} in plan_context.")

    if seen_assets != set(assets):
        missing = sorted(set(assets) - seen_assets)
        raise ValueError(
            "plan_context does not route every registered reference asset: "
            + ", ".join(missing)
            + ". Re-run Prompt Merge."
        )

    if mode == MODE_REF2VA:
        needs_audio_vae = bool(
            references["ref_video_audios"] or references["ref_audios"]
        )
        return references, needs_audio_vae
    return endpoint, False


def prepare_native_h3_call(
    plan_context: Any,
    h3_prompt: str,
    width: int,
    height: int,
    ref_image_size: str = "match",
) -> dict:
    """Build one verified call package for the installed native H3 node."""

    compiled, canonical_prompt, h3_length = _canonical_package(plan_context)
    supplied_prompt = _normalized_prompt(h3_prompt)
    if not supplied_prompt:
        raise ValueError("h3_prompt is empty.")
    if supplied_prompt != _normalized_prompt(canonical_prompt):
        raise ValueError(
            "h3_prompt does not match the connected plan_context. Connect both outputs from "
            "the same Prompt Merge, Structured Prompt Enhancer, or Apply Structured Prose node. "
            "For manual changes, edit editable_prose and recompile it first."
        )

    resolved_width, resolved_height = _target_size(width, height)
    if ref_image_size not in {"match", "max"}:
        raise ValueError("ref_image_size must be 'match' or 'max'.")

    metadata = compiled["compiled"]
    mode = metadata.get("mode")
    checkpoint = metadata.get("checkpoint")
    if mode not in _ENDPOINT_MODES | {MODE_REF2VA}:
        raise ValueError(f"Unsupported compiled H3 mode {mode!r}.")
    route_values, needs_audio_vae = _route_payload(compiled, mode)
    node_id = REFERENCE_NODE_ID if mode == MODE_REF2VA else ENDPOINT_NODE_ID

    kwargs = {
        "prompt": canonical_prompt,
        "width": resolved_width,
        "height": resolved_height,
        "length": h3_length,
    }
    if mode == MODE_REF2VA:
        kwargs.update(route_values)
        kwargs["ref_image_size"] = ref_image_size
    else:
        kwargs.update(route_values)

    route_lines = [
        f"- {entry['label']} -> {entry['route']}"
        for entry in metadata.get("routes", [])
    ] or ["- none"]
    report = "\n".join(
        [
            "Native H3 reference plan ready.",
            f"Mode: {mode}",
            f"Required checkpoint family: {checkpoint}",
            f"Native conditioning node: {node_id}",
            f"Target: {resolved_width}x{resolved_height}, {h3_length} frames at 24 FPS.",
            f"Compatibility contract: {NATIVE_H3_CONTRACT}",
            "Applied routes:",
            *route_lines,
        ]
    )
    return {
        "node_id": node_id,
        "mode": mode,
        "checkpoint": checkpoint,
        "h3_length": h3_length,
        "prompt": canonical_prompt,
        "kwargs": kwargs,
        "needs_audio_vae": needs_audio_vae,
        "report": report,
    }


def _native_h3_class(node_id: str):
    try:
        module = importlib.import_module(NATIVE_H3_MODULE)
    except Exception as error:
        raise RuntimeError(
            "ComfyUI's native MiniMax H3 nodes are unavailable. Update ComfyUI to a build "
            f"that provides {NATIVE_H3_MODULE} ({NATIVE_H3_CONTRACT})."
        ) from error
    native_class = getattr(module, node_id, None)
    if native_class is None or not callable(getattr(native_class, "execute", None)):
        raise RuntimeError(
            f"Installed ComfyUI does not provide the expected native {node_id} class. "
            f"This adapter targets the {NATIVE_H3_CONTRACT}."
        )
    parameters = set(inspect.signature(native_class.execute).parameters)
    missing = sorted(_NATIVE_PARAMETERS[node_id] - parameters)
    if missing:
        raise RuntimeError(
            f"Installed {node_id} has an incompatible API (missing {', '.join(missing)}). "
            f"This adapter targets the {NATIVE_H3_CONTRACT}; update this node pack or use "
            "ComfyUI's native conditioning node manually."
        )
    return native_class


def native_h3_compatibility_report() -> str:
    """Validate both native call signatures without loading model weights."""

    checked = []
    for node_id in (ENDPOINT_NODE_ID, REFERENCE_NODE_ID):
        _native_h3_class(node_id)
        checked.append(node_id)
    return f"Compatible with {NATIVE_H3_CONTRACT}: " + ", ".join(checked) + "."


def _native_outputs(result: Any, node_id: str) -> tuple[Any, Any]:
    values = getattr(result, "result", result)
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise RuntimeError(
            f"Native {node_id} returned an incompatible output contract. Expected "
            "(CONDITIONING, LATENT); update this node pack or use the native node manually."
        )
    return values[0], values[1]


class MiniMaxH3PlanV2ApplyReferencePlan:
    """Apply one compiled Plan v2 package through ComfyUI's native H3 node."""

    CATEGORY = "MiniMax H3/Plan v2"
    FUNCTION = "apply_reference_plan"
    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING", "INT", "STRING")
    RETURN_NAMES = (
        "positive",
        "latent",
        "applied_prompt",
        "h3_length",
        "adapter_report",
    )
    OUTPUT_TOOLTIPS = (
        "Native MiniMax H3 positive conditioning for Basic Guider or another sampler path.",
        "Native empty joint audio/video latent, ready for sampling.",
        "The exact prompt verified against and sent with plan_context.",
        "Native 17k+5 target frame count at 24 FPS.",
        "Mode, checkpoint family, native implementation, target size, and applied routes.",
    )
    DESCRIPTION = (
        "End-to-end Plan v2 handoff. Verifies that prompt and plan_context match, routes every "
        "stored reference in native order, and delegates conditioning to ComfyUI's official "
        "MiniMax H3 implementation. It does not duplicate the native encoder."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": (
                    "CLIP",
                    {
                        "tooltip": "MiniMax H3 Qwen3-VL conditioning CLIP from the official workflow."
                    },
                ),
                "vae": (
                    "VAE",
                    {
                        "tooltip": "MiniMax H3 video VAE used by native keyframe/reference encoding."
                    },
                ),
                "h3_prompt": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": (
                            "Connect h3_prompt/enhanced_prompt from the same node that supplies "
                            "plan_context. A text-only manual edit is rejected to prevent stale labels."
                        ),
                    },
                ),
                "plan_context": (
                    PLAN_TYPE,
                    {
                        "tooltip": (
                            "Connect compiled plan_context from Prompt Merge, Structured Prompt "
                            "Enhancer, or Apply Structured Prose."
                        )
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1344,
                        "min": 32,
                        "max": 16384,
                        "step": 32,
                        "tooltip": "Target width. Use the same H3-native size as the sampler workflow.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 768,
                        "min": 32,
                        "max": 16384,
                        "step": 32,
                        "tooltip": "Target height. Width and height must be multiples of 32.",
                    },
                ),
                "ref_image_size": (
                    ["match", "max"],
                    {
                        "default": "match",
                        "tooltip": (
                            "Ref2VA only. match is faster; max preserves up to a 2048px short edge "
                            "for stronger identity fidelity and much higher reference-token cost."
                        ),
                    },
                ),
            },
            "optional": {
                "audio_vae": (
                    "VAE",
                    {
                        "tooltip": (
                            "MiniMax H3 audio VAE. Required only when the plan contains standalone "
                            "or video-paired reference audio."
                        )
                    },
                )
            },
        }

    def apply_reference_plan(
        self,
        clip,
        vae,
        h3_prompt: str,
        plan_context,
        width: int,
        height: int,
        ref_image_size: str,
        audio_vae=None,
    ):
        package = prepare_native_h3_call(
            plan_context,
            h3_prompt,
            width,
            height,
            ref_image_size,
        )
        if package["needs_audio_vae"] and audio_vae is None:
            raise ValueError(
                "This plan contains reference audio. Connect the MiniMax H3 audio VAE to "
                "Apply Reference Plan.audio_vae."
            )

        native_class = _native_h3_class(package["node_id"])
        kwargs = dict(package["kwargs"])
        kwargs.update({"clip": clip, "vae": vae})
        if package["node_id"] == REFERENCE_NODE_ID:
            kwargs["audio_vae"] = audio_vae
        result = native_class.execute(**kwargs)
        conditioning, latent = _native_outputs(result, package["node_id"])
        return (
            conditioning,
            latent,
            package["prompt"],
            package["h3_length"],
            package["report"],
        )


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PlanV2ApplyReferencePlan": MiniMaxH3PlanV2ApplyReferencePlan,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PlanV2ApplyReferencePlan": "MiniMax H3 Apply Reference Plan (Plan v2)",
}
