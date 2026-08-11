# Video-to-audio Foley

Foley mode keeps the source picture track and asks H3 to generate a new
synchronized audio stream.

## Graph

```text
Project Setup
  → Foley Target: decoded source frames and real source FPS
  → optional Audio References
  → one Shot per source cut
  → Prompt Merge
  → optional Structured Enhancer / Review Gate
  → Apply Reference Plan
  → sampler
```

Foley Target carries the source frames as target media, not as a Ref2VA Video
Reference. Presenting the same video through Video Reference adds an expensive
reference stream without helping the locked latent.

## Latent behavior

Apply Reference Plan obtains positive conditioning from the normal H3 prompt
node, replaces its empty video stream with the encoded source frames, and
constructs:

```text
video latent: source video, noise mask 0 → preserve
audio latent: empty target, noise mask 1 → generate
```

The generated audio replaces the complete soundtrack. Optional Audio
References guide qualities such as timbre, material, texture, voice, or beat;
they do not preserve the source signal.

## Writing the Foley timeline

Create one Shot per real source cut. Write visible timing anchor → audible
result in chronological order.

```text
initial_prompt: Generate realistic production Foley synchronized to the locked source video. No dialogue and no music.

Shot 1: A person crosses the tiled room. Each visible heel and sole contact produces a short dry indoor footstep at the exact contact frame; clothing produces light rustle during each stride.

Shot 2: At the cut, a hand sets a glass on a wooden table. A brief glass-on-wood contact occurs when the base touches the surface, followed by a faint settling tick.

overall_soundscape: Low continuous indoor room tone remains beneath the synchronized footsteps, cloth movement, and object contacts.
non_diegetic_music: N/A
```

Guidelines:

- Use actions visibly supported by the locked video.
- Keep continuous ambience in `overall_soundscape`.
- Keep discrete events inside the Shot where they occur.
- Set `non_diegetic_music` to `N/A` unless an audience-only score is intended.
- Avoid visual/camera requests; the picture track is latent-locked.

When visual analysis is enabled, the enhancer samples timestamped source
frames as evidence. It may infer ordinary physical sounds from visible action,
but cannot add unseen events, speech, or music.

## Sound references

For a footstep texture reference, register an Audio Reference as
**Sound-effect texture**, then place its assigned `<Audio N>` tag in the Shot
sentence at the exact contact event. Source timing remains controlled by the
video; the reference supplies sonic character.

Complete-copy and partial-copy audio relationships are rejected for Foley
because the target audio stream is fully regenerated. Preserving only part of
an original soundtrack requires a separate audio-time mask workflow.

## Requirements

Foley requires MiniMax H3 per-token mask support from
[ComfyUI PR #15375](https://github.com/Comfy-Org/ComfyUI/pull/15375) or an
equivalent compatibility patch, such as a compatible per-row mask patch on the
MODEL branch.

Match Apply Reference Plan's width and height to the source aspect ratio. The
adapter resizes pixels and pads the temporal length to the H3 frame grid.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Picture changes | Ensure the video is connected only through Foley Target |
| Audio is noisy or muted | Test deterministic/ODE sampling or `eta=0` |
| Events drift after cuts | Create one Shot per real source cut |
| Reference texture appears at the wrong time | Put `<Audio N>` in the exact Shot sentence |
| Source audio disappears | Expected: Foley regenerates the complete audio stream |
