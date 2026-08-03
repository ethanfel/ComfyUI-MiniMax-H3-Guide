# ComfyUI MiniMax H3 Prompt Guide

A dependency-free ComfyUI node pack that turns a rough video idea into the prompt structure expected by MiniMax H3. It asks how each image, video, and audio asset is meant to be used, selects the appropriate H3 mode, and can use ComfyUI's loaded Qwen3-VL CLIP to enhance the result.

The node is based on MiniMax's official [base prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md), [full-reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md), and [H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3).

## Nodes

### MiniMax H3 Prompt Guide

`MiniMax H3 Prompt Guide` appears under `MiniMax H3/Prompting`. It produces:

1. `h3_prompt` — a structured draft that can be edited or sent directly to H3.
2. `rewrite_request` — a self-contained instruction for an LLM or Context-IR-style rewrite step. This is recommended when the starting notes are short because the official guide expects a detailed chronological description.
3. `mode_report` — the selected mode and checkpoint, the reason for the selection, Ref2VA limits, and warnings about contradictory options.

No model, API key, or Python dependency is required.

### MiniMax H3 Prompt Enhancer (Qwen3-VL)

This node follows ComfyUI's native **Generate Text** execution model. Connect
`h3_prompt` and optionally `mode_report` from the guide node. Its `CLIP` input
accepts either a complete generation-capable Qwen3-VL model or MiniMax H3's
normal 50-layer conditioning CLIP plus the optional 50–63 generation tail:

```text
MiniMax H3 Prompt Guide
    h3_prompt ─────┐
    mode_report ───┼─> MiniMax H3 Prompt Enhancer ─> enhanced_prompt
standard CLIPLoader ┤                              ├─> system_prompt
optional IMAGE ────┤                              ├─> llm_prompt
Generation Tail Loader ─> clip_tail               └─> enhancer_report
```

It produces:

1. `enhanced_prompt` — Qwen's cleaned and properly formatted H3 prompt.
2. `system_prompt` — the resolved base enhancer instructions, exposed so they can be reused, inspected, or edited.
3. `llm_prompt` — the complete Qwen chat text used for generation.
4. `enhancer_report` — success, compatibility, or generation-collapse status.

The full base system prompt is visible in the node's editable `system_prompt` widget. If that widget is blank, the built-in default is restored. Sampling controls match the important controls from ComfyUI's Generate Text node: maximum generated tokens, deterministic or sampled decoding, temperature, top-k, top-p, min-p, repetition/presence penalties, seed, and thinking mode.

The optional `image` is used only as visual context while Qwen rewrites the prompt. Only the first image in a batch is used. Its presence does not automatically turn the task into image-to-video or make it an H3 keyframe.

For a complete generative Qwen3-VL CLIP, leave the enhancer's optional
`clip_tail` socket disconnected. The enhancer calls the connected CLIP's
normal `generate()` path, so this mode does not require or load the MiniMax
tail.

For MiniMax H3's bundled conditioning CLIP, add **MiniMax H3 Generation Tail
Loader**, select
`qwen3vl_32b_minimax_h3_generation_tail_50_63_int8_convrot.safetensors` in
the loader, and connect its `clip_tail` output to the enhancer. The loader
passes a lightweight descriptor and consumes no VRAM by itself. During
enhancement, the enhancer reuses the connected embedding, vision tower, and
language layers 0–49, loads only layers 50–63 plus the final norm and LM head,
then unloads that tail when generation finishes. The connected 50-layer CLIP
is never merged or modified and remains suitable for official H3 conditioning.
If the truncated CLIP is connected without the side loader, enhancement is
safely skipped and the manual prompt is returned unchanged.

Download the INT8 tail from
[`ethanfel/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot`](https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot)
and place it under `ComfyUI/models/text_encoders/MiniMax-H3/`.

Connect `enhanced_prompt` to ComfyUI's official **MiniMax H3 Image to Video** or **MiniMax H3 Reference to Video** node. Those nodes encode the prompt and attach the correct AV latent plus any keyframe/reference VAE latents and media metadata. The enhancer deliberately does not emit a separate `CONDITIONING` output because it would duplicate the official node for T2VA and be incomplete for image/reference tasks.

### Planning multiple shots

Add one **MiniMax H3 Shot** node per shot. Connect each `shot_plan` output to
the next node's `previous_shots` input, then connect only the last node to the
Prompt Guide's optional `shot_plan` input:

```text
MiniMax H3 Shot (0.000–2.500)
    shot_plan ─> MiniMax H3 Shot (2.500–4.250)
                    shot_plan ─> MiniMax H3 Shot (4.250–6.000)
                                    shot_plan ─> Prompt Guide.shot_plan
```

Each node has float `start_time` and `end_time` controls with millisecond
steps, a shot description, per-shot camera direction, and transition. The
chain rejects gaps, overlaps, reversed ranges, a first shot that does not start
at zero, and times above 15 seconds. Its final `end_time` becomes the prompt's
duration. The generated draft uses `[Shot 1]` without a timestamp, followed by
`[Shot 2] At 00:02.500, ...` and later exact cut markers.

The Prompt Guide's `shot_and_timing_plan` text widget remains available under
advanced controls for old workflows or a quick manual plan. A connected shot
chain takes priority.

## Choosing the right route

| What you want | Mode | Checkpoint |
| --- | --- | --- |
| Generate from text | T2VA | H3-Base-FL2VA |
| Animate an exact first frame | I2VA | H3-Base-FL2VA |
| Land on an exact final frame | L2VA | H3-Base-FL2VA |
| Connect exact first and last frames | FL2VA | H3-Base-FL2VA |
| Use appearance, style, motion, video editing/continuation, or audio references | Ref2VA | H3-Base-Ref2VA |

The important distinction is the role of the asset, not merely its file type:

- A picture used as the exact first or last frame is an endpoint anchor.
- A picture used only for a character's appearance or scene style is a reference-generation asset.
- A video being modified is `video editing`; a video that only supplies motion, camera movement, cuts, or rhythm is `reference generation`.
- Copying an audio signal is `audio reuse`; borrowing its timbre, beat, music style, or sound texture is `audio reference`.

For the common “transfer motion to an image” case, select:

- `Transfer motion to a different subject`
- `Target subject for motion transfer`
- `Transfer its motion or action`

This creates a Ref2VA prompt where the target image keeps its visible identity and the action reference receives the fixed `attribute_transfer` relationship.

## Reference inventory

Enter one asset per line. Labels correspond to the order in which media is supplied to the downstream H3 workflow:

```text
Picture 1: a red ceramic robot, front three-quarter view
Video 1: a dancer performing a quick clockwise spin
Audio 1: a dry studio recording of a calm female voice
```

Angle brackets are optional. `Picture 1: ...` and `<Picture 1>: ...` are equivalent. Unlabelled lines are retained as additional reference notes. If a selected role needs a missing label, the node creates a generic one so the draft remains structurally complete.

## Example: edit a video and keep its soundtrack

Set `how_video_is_used` to `Directly edit the source video` and `how_audio_is_used` to `Reuse the complete audio signal`. The node selects Ref2VA and starts the summary with:

```text
[video editing + audio reuse] The target video is an edited version of <Video 1>.
```

The output also distinguishes the video retention marker from the audio `fully_copy` marker.

## Install

Clone or copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide
```

Then add **MiniMax H3 Prompt Guide**, **MiniMax H3 Shot**, **MiniMax H3
Generation Tail Loader**, and **MiniMax H3 Prompt Enhancer (Qwen3-VL)** from
**MiniMax H3 → Prompting** as needed.

## Practical notes

- H3 output duration is 4–15 seconds.
- Ref2VA accepts up to 9 images, 3 videos, 3 audio clips, and 12 media files in total.
- Each reference video/audio clip must be 2–15 seconds; each media type has a total duration limit of 15 seconds.
- Reference audio cannot be the sole media input; it must be accompanied by an image or video.
- The Prompt Guide node prepares text and labels only. Connect the actual media to the H3 generation nodes used by your workflow in the same label order.
- For a polished generation prompt, send `rewrite_request` to your preferred LLM node and use its response as the final H3 prompt.
- The included enhancer can use the same MiniMax H3 CLIP as conditioning when the Generation Tail Loader is connected. The 32B model is large, so prompt enhancement can require substantial VRAM and time.

## Test

```bash
pytest -q
```
