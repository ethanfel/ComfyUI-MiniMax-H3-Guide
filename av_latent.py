"""MiniMax H3-specific audio/video latent plumbing.

ComfyUI represents an H3 sample as a nested pair, but its video and audio
streams do not share a temporal grid:

* video: ``[B, 24, Tv, H/16, W/16]`` on the ``5k+2`` latent grid;
* audio: ``[B, 32, 2, Ta]`` at 40 latent steps per second.

The stock LTX AV helper can package those tensors, but a split H3 video latent
is plain.  In that route the LTX joiner does not reconcile the audio length to
the H3 video timeline.  These nodes make that reconciliation explicit and keep
the video latent's metadata authoritative.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch


FPS = 24
AUDIO_LATENT_FPS = 40
VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
AUDIO_CHANNEL_LAYOUT = 2
VIDEO_TIME_DIM = 2
AUDIO_TIME_DIM = 3
VIDEO_LATENT_BASE = 2
VIDEO_LATENTS_PER_GROUP = 5
FRAMES_BASE = 5
FRAMES_PER_GROUP = 17


def h3_frame_count(video_latents: int) -> int:
    """Return the exact pixel-frame count represented by an H3 video latent."""

    latent_count = int(video_latents)
    if (
        latent_count < VIDEO_LATENT_BASE
        or (latent_count - VIDEO_LATENT_BASE) % VIDEO_LATENTS_PER_GROUP
    ):
        raise ValueError(
            "MiniMax H3 video length must be on the 5k+2 latent grid; "
            f"received {latent_count} video latents."
        )
    groups = (latent_count - VIDEO_LATENT_BASE) // VIDEO_LATENTS_PER_GROUP
    return FRAMES_BASE + groups * FRAMES_PER_GROUP


def h3_audio_latent_count(video_latents: int) -> int:
    """Return the H3 40 Hz audio length paired with a video latent length."""

    frames = h3_frame_count(video_latents)
    return round(frames / FPS * AUDIO_LATENT_FPS)


def _is_nested(value: Any) -> bool:
    return bool(getattr(value, "is_nested", False))


def _nested_streams(value: Any, name: str) -> tuple[Any, Any]:
    if not _is_nested(value):
        raise ValueError(
            f"{name} is a plain latent. Expected a MiniMax H3 nested video/audio latent."
        )
    try:
        streams = tuple(value.unbind())
    except Exception as error:
        raise ValueError(f"Could not read the nested streams in {name}.") from error
    if len(streams) != 2:
        raise ValueError(
            f"{name} must contain exactly two streams (video, audio); received {len(streams)}."
        )
    return streams[0], streams[1]


def _latent_samples(latent: Any, name: str) -> Any:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError(f"{name} is not a ComfyUI LATENT dictionary.")
    return latent["samples"]


def _video_samples(latent: Any, name: str) -> tuple[Any, Any | None]:
    samples = _latent_samples(latent, name)
    if _is_nested(samples):
        video, _audio = _nested_streams(samples, name)
        return video, samples
    return samples, None


def _audio_samples(latent: Any, name: str) -> tuple[Any, Any | None]:
    samples = _latent_samples(latent, name)
    if _is_nested(samples):
        _video, audio = _nested_streams(samples, name)
        return audio, samples
    return samples, None


def _validate_video(video: Any, name: str = "video_latent") -> None:
    if getattr(video, "ndim", None) != 5 or int(video.shape[1]) != VIDEO_CHANNELS:
        shape = tuple(video.shape) if hasattr(video, "shape") else type(video).__name__
        raise ValueError(
            f"{name} is not an H3 video latent. Expected [B,24,T,H,W], received {shape}."
        )
    h3_frame_count(int(video.shape[VIDEO_TIME_DIM]))


def _validate_audio(audio: Any, name: str = "audio_latent") -> None:
    if (
        getattr(audio, "ndim", None) != 4
        or int(audio.shape[1]) != AUDIO_CHANNELS
        or int(audio.shape[2]) != AUDIO_CHANNEL_LAYOUT
    ):
        shape = tuple(audio.shape) if hasattr(audio, "shape") else type(audio).__name__
        raise ValueError(
            f"{name} is not an H3 audio latent. Expected [B,32,2,T40], received {shape}."
        )


def _new_nested(template: Any | None, video: Any, audio: Any) -> Any:
    if template is not None:
        try:
            return type(template)((video, audio))
        except Exception:
            pass
    try:
        nested_module = importlib.import_module("comfy.nested_tensor")
        return nested_module.NestedTensor((video, audio))
    except Exception as error:
        raise RuntimeError(
            "Could not construct MiniMax H3's nested video/audio latent."
        ) from error


def _fit_time(value: Any, length: int, dim: int, pad_value: float) -> tuple[Any, str]:
    current = int(value.shape[dim])
    if current == length:
        return value.contiguous(), "unchanged"
    if current > length:
        return value.narrow(dim, 0, length).contiguous(), f"trimmed {current}->{length}"

    pad_shape = list(value.shape)
    pad_shape[dim] = length - current
    pad = value.new_full(pad_shape, pad_value)
    return torch.cat((value, pad), dim=dim).contiguous(), f"padded {current}->{length}"


def _modality_mask(latent: Any, modality: int, name: str) -> Any | None:
    mask = latent.get("noise_mask") if isinstance(latent, dict) else None
    if mask is None:
        return None
    if _is_nested(mask):
        video_mask, audio_mask = _nested_streams(mask, f"{name}.noise_mask")
        return video_mask if modality == 0 else audio_mask

    ndim = getattr(mask, "ndim", None)
    expected = 5 if modality == 0 else 4
    if ndim != expected:
        raise ValueError(
            f"{name}.noise_mask has {ndim} dimensions; expected {expected} for the "
            f"{'video' if modality == 0 else 'audio'} stream."
        )
    return mask


def _validate_mask_time(mask: Any, samples: Any, dim: int, name: str) -> None:
    if int(mask.shape[dim]) != int(samples.shape[dim]):
        raise ValueError(
            f"{name} has {int(mask.shape[dim])} temporal steps but its latent has "
            f"{int(samples.shape[dim])}."
        )


class MiniMaxH3SplitAVLatent:
    """Losslessly split a native H3 AV latent along modality."""

    CATEGORY = "MiniMax H3/Latent"
    FUNCTION = "split"
    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("video_latent", "audio_latent", "sync_report")
    DESCRIPTION = (
        "Split a MiniMax H3 nested AV latent into its 24-channel video and "
        "32-channel stereo audio latents without applying LTX timing assumptions."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Native MiniMax H3 AV latent containing nested video and audio streams."
                        )
                    },
                )
            }
        }

    def split(self, av_latent):
        samples = _latent_samples(av_latent, "av_latent")
        video, audio = _nested_streams(samples, "av_latent")
        _validate_video(video)
        _validate_audio(audio)

        video_output = dict(av_latent)
        audio_output = dict(av_latent)
        video_output["samples"] = video.contiguous()
        audio_output["samples"] = audio.contiguous()

        mask = av_latent.get("noise_mask")
        if mask is not None:
            video_mask, audio_mask = _nested_streams(mask, "av_latent.noise_mask")
            _validate_mask_time(video_mask, video, VIDEO_TIME_DIM, "video noise mask")
            _validate_mask_time(audio_mask, audio, AUDIO_TIME_DIM, "audio noise mask")
            video_output["noise_mask"] = video_mask.contiguous()
            audio_output["noise_mask"] = audio_mask.contiguous()

        video_latents = int(video.shape[VIDEO_TIME_DIM])
        frames = h3_frame_count(video_latents)
        expected_audio = h3_audio_latent_count(video_latents)
        actual_audio = int(audio.shape[AUDIO_TIME_DIM])
        status = (
            "aligned"
            if actual_audio == expected_audio
            else f"MISMATCH: expected {expected_audio} audio latents"
        )
        report = (
            f"{video_latents} video latents = {frames} frames = {frames / FPS:.3f}s; "
            f"audio {actual_audio}/40s = {actual_audio / AUDIO_LATENT_FPS:.3f}s; {status}."
        )
        return video_output, audio_output, report


class MiniMaxH3PackAVLatent:
    """Pack H3 video and audio while locking audio length to the video clock."""

    CATEGORY = "MiniMax H3/Latent"
    FUNCTION = "pack"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("av_latent", "sync_report")
    DESCRIPTION = (
        "Pack MiniMax H3 video and audio latents. Audio is trimmed or silence-padded "
        "to the exact 40 Hz length implied by the H3 video grid."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Plain H3 video latent from Split AV/VAE Encode, or an H3 AV "
                            "latent whose video stream should be kept."
                        )
                    },
                )
            },
            "optional": {
                "audio_latent": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Plain H3 audio latent from Split AV/VAE Encode Audio, or an H3 "
                            "AV latent whose audio stream should be kept. Omit for silence."
                        )
                    },
                )
            },
        }

    def pack(self, video_latent, audio_latent=None):
        video, video_template = _video_samples(video_latent, "video_latent")
        _validate_video(video)
        video_latents = int(video.shape[VIDEO_TIME_DIM])
        frames = h3_frame_count(video_latents)
        wanted_audio = h3_audio_latent_count(video_latents)

        audio_template = None
        if audio_latent is None:
            audio = video.new_zeros(
                (int(video.shape[0]), AUDIO_CHANNELS, AUDIO_CHANNEL_LAYOUT, wanted_audio)
            )
            adjustment = f"generated {wanted_audio} silent audio latents"
        else:
            audio, audio_template = _audio_samples(audio_latent, "audio_latent")
            _validate_audio(audio)
            if int(audio.shape[0]) != int(video.shape[0]):
                raise ValueError(
                    "H3 video and audio batch sizes must match; received "
                    f"{int(video.shape[0])} and {int(audio.shape[0])}."
                )
            audio, adjustment = _fit_time(
                audio.to(device=video.device, dtype=video.dtype),
                wanted_audio,
                AUDIO_TIME_DIM,
                0.0,
            )

        output = dict(video_latent)
        output["samples"] = _new_nested(
            video_template if video_template is not None else audio_template,
            video.contiguous(),
            audio.contiguous(),
        )

        video_mask = _modality_mask(video_latent, 0, "video_latent")
        audio_mask = (
            _modality_mask(audio_latent, 1, "audio_latent")
            if audio_latent is not None
            else None
        )
        if video_mask is not None:
            _validate_mask_time(video_mask, video, VIDEO_TIME_DIM, "video noise mask")
        if audio_mask is not None:
            original_audio, _template = _audio_samples(audio_latent, "audio_latent")
            _validate_mask_time(audio_mask, original_audio, AUDIO_TIME_DIM, "audio noise mask")
            audio_mask, _mask_adjustment = _fit_time(
                audio_mask.to(device=audio.device),
                wanted_audio,
                AUDIO_TIME_DIM,
                1.0,
            )

        if video_mask is not None or audio_mask is not None:
            if video_mask is None:
                video_mask = torch.ones(
                    (int(video.shape[0]), 1, int(video.shape[2]), int(video.shape[3]), int(video.shape[4])),
                    dtype=torch.float32,
                    device=video.device,
                )
            if audio_mask is None:
                audio_mask = torch.ones(
                    (int(audio.shape[0]), 1, int(audio.shape[2]), int(audio.shape[3])),
                    dtype=torch.float32,
                    device=audio.device,
                )
            output["noise_mask"] = _new_nested(
                video_template if video_template is not None else audio_template,
                video_mask.contiguous(),
                audio_mask.contiguous(),
            )
        else:
            output.pop("noise_mask", None)

        report = (
            f"{video_latents} video latents = {frames} frames = {frames / FPS:.3f}s; "
            f"audio locked to {wanted_audio}/40s = {wanted_audio / AUDIO_LATENT_FPS:.3f}s; "
            f"{adjustment}."
        )
        return output, report


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3GuideSplitAVLatent": MiniMaxH3SplitAVLatent,
    "MiniMaxH3GuidePackAVLatent": MiniMaxH3PackAVLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3GuideSplitAVLatent": "MiniMax H3 Split AV Latent",
    "MiniMaxH3GuidePackAVLatent": "MiniMax H3 Pack AV Latent (Frame-Locked)",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
