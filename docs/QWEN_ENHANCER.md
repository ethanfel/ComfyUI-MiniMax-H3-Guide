# Qwen enhancement and prompt review

The optional structured enhancer expands Plan v2 prose without allowing a
language model to redefine the compiled plan.

## Data flow

```text
Prompt Merge
  → Structured Prompt Enhancer
  → optional Apply Structured Prose
  → optional Prompt Review Gate or Inline Prompt Override
  → Apply Reference Plan
```

The enhancer receives the compiled scene, references, roles, Shots, timing,
audio metadata, native routes, compiler report, and optional visual evidence
in one request.

## Enhancement modes

### Intent-locked expansion

This is the default. Original prose stays intact. Qwen may add short compatible
Shot, camera, or soundscape details. Attempts to replace global intent, style,
music, timing, or compiler-owned content are ignored and reported.

### Creative expansion

Qwen may rewrite editable prose and introduce additional presentation choices.
Labels, roles, timing, dialogue, Shot order, and native routing remain locked.

## Structured output

Qwen returns a versioned JSON object. Python validates it, reconstructs the H3
prompt, and checks:

- reference labels and roles;
- Subject retention and transfer relationships;
- native media routes;
- Shot order and cut times;
- speaker order and exact dialogue;
- required H3 structure.

Invalid, incomplete, or collapsed output falls back to the deterministic
compiler draft. `editable_prose` is always the final validated JSON. Use
**Apply Structured Prose** after a text editor to refine and recompile that JSON
without running Qwen again.

## Visual and audio context

The enhancer may analyze selected images and timestamped video samples. Audio
waveforms are not presented as understandable model input; audio meaning comes
from explicit Audio Reference metadata and transcripts.

## Model options

The `CLIP` input accepts:

1. a complete generation-capable Qwen3-VL or Qwen3.5 model; or
2. MiniMax H3's normal 50-layer conditioning CLIP plus the optional 50–63
   generation tail.

For option 2, use **MiniMax H3 Generation Tail Loader** with
`qwen3vl_32b_minimax_h3_generation_tail_50_63_int8_convrot.safetensors`.
Download it from
[`ethanfel/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot`](https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot)
and place it under:

```text
ComfyUI/models/text_encoders/MiniMax-H3/
```

The tail loader itself holds only a descriptor. During generation, the
enhancer reuses the connected embedding, vision tower, and language layers
0–49, then loads layers 50–63, final norm, and LM head. The tail is released
after decoding; the connected CLIP remains under ComfyUI's normal model manager.

DynamicVRAM can stream the model, but sub-32-GB operation is hardware-dependent
and may be slow. Keep DynamicVRAM enabled. `--highvram` and `--gpu-only` defeat
the low-VRAM path.

## Prompt Review Gate

The Review Gate pauses the queued job and opens the full H3 prompt in an editor.
It supports approval, input restoration, revision history, rejection, and an
optional timeout.

Editable content includes scene, camera, ambience, and music prose. The gate
rejects changes to compiler-owned sections, labels, retention, Shot markers,
cut timestamps, dialogue, replacement mappings, and media routes.

History is stored under ComfyUI user data and contains prompt text and hashes,
not reference media or tensors. Pass-through mode leaves the gate in the graph
without pausing.

## Inline Prompt Override

Inline Prompt Override provides the same plan-bound descriptive validation
without the interactive pause. Paste a complete experimental prompt, connect
the matching `plan_context`, and toggle `use_override`. Clear or disable it to
bypass the experiment.
