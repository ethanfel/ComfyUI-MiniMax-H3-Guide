# H3 AV latent split and repack

Use these nodes when a workflow keeps the video stream from one MiniMax H3
latent and the audio stream from another—for example, retaining first-pass
audio after a video-only refinement pass.

```text
first-pass H3 AV ── MiniMax H3 Split AV Latent ── audio ─┐
                                                        ├─ MiniMax H3 Pack AV Latent (Frame-Locked)
refined H3 AV ───── MiniMax H3 Split AV Latent ── video ┘
```

Replace `LTXVSeparateAVLatent` and `LTXVConcatAVLatent` on this route. The LTX
nodes can nest the H3 tensors, but after separation the video input is plain,
so the joiner does not necessarily reconcile the audio length.

## Why the lengths differ

H3 uses separate clocks:

| Stream | Shape | Time rule |
| --- | --- | --- |
| Video | `[B, 24, Tv, H, W]` | `Tv = 5k+2` represents `17k+5` frames at 24 FPS |
| Audio | `[B, 32, 2, Ta]` | 40 latent steps per second |

The frame-locked pack node calculates the output duration from the video
latent, then sets audio to `round(frames / 24 × 40)`. Extra audio is trimmed
from the tail; short audio is silence-padded at the tail. The origin remains
at time zero, so lip-sync does not receive a compensating offset.

If either input carries a noise mask, the node constructs a matching nested
mask. A silence-padded audio tail receives mask value `1`, allowing the model
to generate that missing tail instead of pinning silence.

## Validation

The nodes reject:

- non-H3 channel layouts;
- video lengths outside the `5k+2` latent grid;
- mismatched video/audio batch sizes;
- masks whose temporal length no longer matches their source latent.

The pack output keeps metadata from the video latent. Audio metadata cannot
silently overwrite the video stream's batch or sampling metadata.
