import pytest
import torch

import av_latent
from av_latent import (
    MiniMaxH3PackAVLatent,
    MiniMaxH3SplitAVLatent,
    h3_audio_latent_count,
    h3_frame_count,
)


class FakeNested:
    is_nested = True

    def __init__(self, streams):
        self.streams = tuple(streams)

    def unbind(self):
        return self.streams


@pytest.fixture(autouse=True)
def fake_nested_constructor(monkeypatch):
    monkeypatch.setattr(
        av_latent,
        "_new_nested",
        lambda _template, video, audio: FakeNested((video, audio)),
    )


def test_h3_grid_converts_video_latents_to_exact_frame_and_audio_lengths():
    assert h3_frame_count(2) == 5
    assert h3_frame_count(47) == 158
    assert h3_frame_count(107) == 362
    assert h3_audio_latent_count(47) == 263
    assert h3_audio_latent_count(107) == 603

    with pytest.raises(ValueError, match=r"5k\+2"):
        h3_frame_count(108)


def test_split_is_lossless_and_reports_an_existing_audio_mismatch():
    video = torch.randn(1, 24, 107, 2, 3)
    audio = torch.randn(1, 32, 2, 604)
    video_mask = torch.zeros(1, 1, 107, 2, 3)
    audio_mask = torch.ones(1, 1, 2, 604)
    source = {
        "samples": FakeNested((video, audio)),
        "noise_mask": FakeNested((video_mask, audio_mask)),
        "batch_index": [4],
    }

    video_output, audio_output, report = MiniMaxH3SplitAVLatent().split(source)

    assert video_output["samples"] is video
    assert audio_output["samples"] is audio
    assert video_output["noise_mask"] is video_mask
    assert audio_output["noise_mask"] is audio_mask
    assert video_output["batch_index"] == audio_output["batch_index"] == [4]
    assert "362 frames" in report
    assert "expected 603 audio latents" in report


def test_pack_locks_plain_split_audio_to_the_video_clock_and_keeps_video_metadata():
    video = torch.randn(1, 24, 107, 2, 3)
    audio = torch.randn(1, 32, 2, 604)
    video_input = {"samples": video, "batch_index": [1], "owner": "video"}
    audio_input = {"samples": audio, "batch_index": [99], "owner": "audio"}

    output, report = MiniMaxH3PackAVLatent().pack(video_input, audio_input)
    packed_video, packed_audio = output["samples"].unbind()

    assert packed_video is video
    assert packed_audio.shape == (1, 32, 2, 603)
    assert torch.equal(packed_audio, audio[:, :, :, :603])
    assert output["batch_index"] == [1]
    assert output["owner"] == "video"
    assert "audio locked to 603/40s" in report
    assert "trimmed 604->603" in report


def test_pack_silence_pads_short_audio_and_marks_the_new_mask_tail_for_denoising():
    video = torch.randn(1, 24, 47, 2, 3)
    audio = torch.randn(1, 32, 2, 262)
    video_mask = torch.zeros(1, 1, 47, 2, 3)
    audio_mask = torch.zeros(1, 1, 2, 262)
    video_input = {"samples": video, "noise_mask": video_mask}
    audio_input = {"samples": audio, "noise_mask": audio_mask}

    output, report = MiniMaxH3PackAVLatent().pack(video_input, audio_input)
    _packed_video, packed_audio = output["samples"].unbind()
    packed_video_mask, packed_audio_mask = output["noise_mask"].unbind()

    assert packed_audio.shape[-1] == 263
    assert torch.count_nonzero(packed_audio[:, :, :, -1]) == 0
    assert packed_video_mask is video_mask
    assert torch.equal(packed_audio_mask[:, :, :, :-1], audio_mask)
    assert torch.all(packed_audio_mask[:, :, :, -1] == 1)
    assert "padded 262->263" in report


def test_pack_without_audio_generates_exact_silence():
    video = torch.randn(2, 24, 47, 2, 3)

    output, report = MiniMaxH3PackAVLatent().pack({"samples": video})
    _packed_video, packed_audio = output["samples"].unbind()

    assert packed_audio.shape == (2, 32, 2, 263)
    assert torch.count_nonzero(packed_audio) == 0
    assert "generated 263 silent audio latents" in report


def test_pack_rejects_wrong_h3_shapes_and_batch_mismatch():
    node = MiniMaxH3PackAVLatent()

    with pytest.raises(ValueError, match="H3 video latent"):
        node.pack({"samples": torch.zeros(1, 128, 47, 2, 3)})

    with pytest.raises(ValueError, match="batch sizes must match"):
        node.pack(
            {"samples": torch.zeros(2, 24, 47, 2, 3)},
            {"samples": torch.zeros(1, 32, 2, 263)},
        )
