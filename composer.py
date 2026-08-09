"""Compact Plan v2 composers built from the authoritative semantic nodes.

The first composer intentionally targets one proven workflow shape: animate an
opening keyframe with a separate identity reference, a motion-only video, and
the original soundtrack.  It does not duplicate Plan v2 validation.  Instead,
it calls the same public node methods as the expanded graph so both surfaces
produce the same plan and compiler output.
"""

from __future__ import annotations

try:
    from .plan_v2 import (
        AUDIO_COPY_COMPLETE,
        CONTENT_IDENTITY,
        IMAGE_DEFINE_VISIBLE,
        KEYFRAME_SHOT_OPENING,
        PLAN_TYPE,
        RETENTION_AUTO,
        SHOT_TRANSITIONS,
        MiniMaxH3PlanV2AudioReference,
        MiniMaxH3PlanV2ImageReference,
        MiniMaxH3PlanV2ProjectSetup,
        MiniMaxH3PlanV2Shot,
        MiniMaxH3PlanV2ShotKeyframe,
        MiniMaxH3PlanV2ShotMotionReference,
    )
except ImportError:
    from plan_v2 import (
        AUDIO_COPY_COMPLETE,
        CONTENT_IDENTITY,
        IMAGE_DEFINE_VISIBLE,
        KEYFRAME_SHOT_OPENING,
        PLAN_TYPE,
        RETENTION_AUTO,
        SHOT_TRANSITIONS,
        MiniMaxH3PlanV2AudioReference,
        MiniMaxH3PlanV2ImageReference,
        MiniMaxH3PlanV2ProjectSetup,
        MiniMaxH3PlanV2Shot,
        MiniMaxH3PlanV2ShotKeyframe,
        MiniMaxH3PlanV2ShotMotionReference,
    )


DEFAULT_ANIMATION_GOAL = (
    "Animate the supplied opening keyframe as one continuous shot. The identity "
    "reference defines the character's stable identity and appearance. The opening "
    "keyframe defines the initial pose, subject placement, wardrobe, camera, framing, "
    "composition, and target visual appearance. Transfer only motion and action timing "
    "from the structural control video to the referenced character, beginning smoothly "
    "from the keyframe pose. Preserve <Audio 1> as the complete final soundtrack and "
    "synchronize visible facial reactions to it. Do not copy the control video's identity, "
    "appearance, wardrobe, environment, lighting, camera, framing, or control-map style. "
    "Avoid an initial pose snap, duplicated action, scene change, or cut."
)

DEFAULT_IDENTITY_DESCRIPTION = (
    "One consistent adult character shown clearly enough to define stable facial identity, "
    "hairstyle, skin appearance, body proportions, and distinctive visible features. Use "
    "identity and appearance only. Do not reproduce a reference-sheet layout, neutral "
    "backdrop, labels, or multiple copies in the target."
)

DEFAULT_SHOT_DESCRIPTION = (
    "Begin in the supplied opening keyframe's pose and composition. <Subject 1> performs "
    "the motion transferred from <Video 1>, adapted smoothly from the keyframe pose while "
    "preserving the keyframe's subject placement, wardrobe, camera framing, and spatial "
    "relationships. Preserve <Audio 1> as the complete final soundtrack with unchanged "
    "timing, and synchronize the character's mouth, eyes, brows, facial expression, and "
    "visible reactions to it. Keep the action physically coherent and continuous."
)

DEFAULT_CAMERA_DIRECTION = (
    "Keep the opening keyframe's camera position, lens character, framing, and composition "
    "stable throughout the Shot. Treat global translation or scale changes in <Video 1> "
    "as subject motion, not camera movement."
)

DEFAULT_KEYFRAME_DESCRIPTION = (
    "This image defines the target Shot's opening subject pose, subject placement, wardrobe, "
    "camera, framing, composition, and target visual appearance. Use the reusable character "
    "reference for identity; do not reproduce a reference-sheet layout or its background."
)

DEFAULT_MOTION_DESCRIPTION = (
    "Transfer only pose progression, action order, movement cadence, body mechanics, balance "
    "shifts, contact timing, and broad facial timing from the structural control video. "
    "Begin from the opening keyframe pose and adapt the motion smoothly. Treat line art, "
    "depth, or other control-map rendering as structural guidance only, never as target "
    "appearance. Do not copy source identity, anatomy, wardrobe, environment, lighting, "
    "camera, framing, control colors, background edges, or exact facial and clothing contours."
)


class MiniMaxH3PlanV2KeyframeMotionComposer:
    """Build the proven one-shot keyframe-to-motion Plan v2 graph in one node."""

    CATEGORY = "MiniMax H3/Plan v2/Composer"
    FUNCTION = "compose"
    RETURN_TYPES = (PLAN_TYPE, "INT", "STRING", "IMAGE", "AUDIO")
    RETURN_NAMES = (
        "h3_plan",
        "h3_length",
        "composer_preview",
        "h3_motion_video",
        "h3_audio",
    )
    OUTPUT_TOOLTIPS = (
        "Complete timeline-phase Plan v2 package; connect directly to Prompt Merge.",
        "Native 17k+5 output frame count at 24 FPS.",
        "Resolved identity, audio, keyframe, motion, Shot, and route summary.",
        "Motion frames prepared on H3's native 24-FPS temporal grid.",
        "Normalized complete-copy soundtrack stored in the Plan.",
    )
    DESCRIPTION = (
        "Compact equivalent of Project Setup -> identity reference -> complete-copy audio -> "
        "Shot 1 -> opening keyframe -> motion attachment. It uses the authoritative Plan v2 "
        "nodes internally and produces the same validated plan as the expanded graph."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "identity_image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Character identity/appearance reference. This becomes <Picture 1> "
                            "and defines <Subject 1>."
                        )
                    },
                ),
                "opening_keyframe": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Exact opening composition. This becomes <Picture 2> and the Shot "
                            "opening keyframe."
                        )
                    },
                ),
                "motion_video_frames": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Decoded motion/control frames. This becomes <Video 1> and transfers "
                            "motion only to the named Subject."
                        )
                    },
                ),
                "original_audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Original soundtrack copied as the complete final track through "
                            "standalone <Audio 1>."
                        )
                    },
                ),
                "frame_count": (
                    "INT",
                    {
                        "default": 345,
                        "min": 107,
                        "max": 362,
                        "step": 17,
                        "tooltip": "Native 17k+5 output frame count at the fixed 24 FPS.",
                    },
                ),
                "animation_goal": (
                    "STRING",
                    {
                        "default": DEFAULT_ANIMATION_GOAL,
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Global intent for this one-shot keyframe animation.",
                    },
                ),
                "visual_style": (
                    "STRING",
                    {
                        "default": "cinematic, photorealistic live-action",
                        "tooltip": "Target appearance; control-map rendering is excluded.",
                    },
                ),
                "subject_name": (
                    "STRING",
                    {
                        "default": "character",
                        "tooltip": "Stable alias for the identity and motion-transfer target.",
                    },
                ),
                "shot_description": (
                    "STRING",
                    {
                        "default": DEFAULT_SHOT_DESCRIPTION,
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Visible and audible events for the single target Shot.",
                    },
                ),
                "camera_direction": (
                    "STRING",
                    {
                        "default": DEFAULT_CAMERA_DIRECTION,
                        "multiline": True,
                        "dynamicPrompts": False,
                        "tooltip": "Target camera behavior; independent from control motion.",
                    },
                ),
                "advanced_options": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Canvas-only disclosure control. The hidden semantic fields remain "
                            "serialized and authoritative."
                        ),
                    },
                ),
                "identity_reference_name": (
                    "STRING",
                    {"default": "character identity"},
                ),
                "identity_description": (
                    "STRING",
                    {
                        "default": DEFAULT_IDENTITY_DESCRIPTION,
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "audio_reference_name": (
                    "STRING",
                    {"default": "original control-video soundtrack"},
                ),
                "keyframe_reference_name": (
                    "STRING",
                    {"default": "target opening keyframe"},
                ),
                "keyframe_description": (
                    "STRING",
                    {
                        "default": DEFAULT_KEYFRAME_DESCRIPTION,
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "motion_reference_name": (
                    "STRING",
                    {"default": "character motion"},
                ),
                "motion_description": (
                    "STRING",
                    {
                        "default": DEFAULT_MOTION_DESCRIPTION,
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "source_fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 0.01,
                        "max": 240.0,
                        "step": 0.01,
                    },
                ),
            }
        }

    def compose(
        self,
        identity_image,
        opening_keyframe,
        motion_video_frames,
        original_audio,
        frame_count: int,
        animation_goal: str,
        visual_style: str,
        subject_name: str,
        shot_description: str,
        camera_direction: str,
        advanced_options: bool,
        identity_reference_name: str,
        identity_description: str,
        audio_reference_name: str,
        keyframe_reference_name: str,
        keyframe_description: str,
        motion_reference_name: str,
        motion_description: str,
        source_fps: float,
    ):
        del advanced_options  # Presentation state only; serialized by ComfyUI.

        plan, h3_length, _project_preview = MiniMaxH3PlanV2ProjectSetup().start(
            animation_goal,
            frame_count,
            visual_style,
            "",
            "N/A",
        )
        plan, _identity_handle, _identity, identity_preview = (
            MiniMaxH3PlanV2ImageReference().add_image(
                plan,
                identity_image,
                IMAGE_DEFINE_VISIBLE,
                identity_reference_name,
                identity_description,
                CONTENT_IDENTITY,
                subject_name,
                RETENTION_AUTO,
                "all",
                "",
            )
        )
        plan, h3_audio, audio_preview = MiniMaxH3PlanV2AudioReference().add_audio(
            plan,
            original_audio,
            AUDIO_COPY_COMPLETE,
            audio_reference_name,
            "",
            "",
            "",
            "",
            "",
            "",
        )
        plan, shot_preview, shot_handle = MiniMaxH3PlanV2Shot().add_shot(
            plan,
            0.0,
            shot_description,
            camera_direction,
            SHOT_TRANSITIONS[0],
        )
        plan, shot_handle, _keyframe, keyframe_preview = (
            MiniMaxH3PlanV2ShotKeyframe().attach_keyframe(
                plan,
                shot_handle,
                opening_keyframe,
                keyframe_reference_name,
                keyframe_description,
                KEYFRAME_SHOT_OPENING,
            )
        )
        plan, _shot_handle, h3_motion_video, motion_preview = (
            MiniMaxH3PlanV2ShotMotionReference().attach_motion(
                plan,
                shot_handle,
                motion_video_frames,
                subject_name,
                motion_reference_name,
                motion_description,
                source_fps,
            )
        )

        preview = "\n".join(
            (
                f"Keyframe-to-motion Composer ready · {h3_length}f · "
                f"{h3_length / 24:.3f}s at 24 FPS.",
                f"Identity: {identity_preview}",
                f"Audio: {audio_preview}",
                f"Timeline: {shot_preview}",
                f"Keyframe: {keyframe_preview}",
                f"Motion: {motion_preview}",
                "Next: connect h3_plan directly to Prompt Merge (Plan v2).",
            )
        )
        return plan, h3_length, preview, h3_motion_video, h3_audio


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3PlanV2KeyframeMotionComposer": MiniMaxH3PlanV2KeyframeMotionComposer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3PlanV2KeyframeMotionComposer": (
        "MiniMax H3 Keyframe + Motion Composer (Plan v2)"
    ),
}
