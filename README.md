# ComfyUI MiniMax H3 Prompt Guide

A dependency-free ComfyUI node pack that turns a rough video idea into the prompt structure expected by MiniMax H3. It asks how each image, video, and audio asset is meant to be used, selects the appropriate H3 mode, and can use ComfyUI's loaded Qwen3-VL CLIP to enhance and encode the result.

The node is based on MiniMax's official [base prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md), [full-reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md), and [H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3).

## Nodes

### MiniMax H3 Prompt Guide

`MiniMax H3 Prompt Guide` appears under `MiniMax H3/Prompting`. It produces:

1. `h3_prompt` — a structured draft that can be edited or sent directly to H3.
2. `rewrite_request` — a self-contained instruction for an LLM or Context-IR-style rewrite step. This is recommended when the starting notes are short because the official guide expects a detailed chronological description.
3. `mode_report` — the selected mode and checkpoint, the reason for the selection, Ref2VA limits, and warnings about contradictory options.

No model, API key, or Python dependency is required.

### MiniMax H3 Prompt Enhancer (Qwen3-VL)

This node follows ComfyUI's native **Generate Text** execution model. Connect a generation-capable Qwen3-VL `CLIP`, then connect `h3_prompt` and optionally `mode_report` from the guide node:

```text
MiniMax H3 Prompt Guide
    h3_prompt ─────┐
    mode_report ───┼─> MiniMax H3 Prompt Enhancer ─> enhanced_prompt
Qwen3-VL CLIP ─────┤                              └─> conditioning
optional IMAGE ────┘
```

It produces:

1. `enhanced_prompt` — Qwen's cleaned and properly formatted H3 prompt.
2. `conditioning` — the enhanced text encoded with the connected CLIP.
3. `system_prompt` — the resolved base enhancer instructions, exposed so they can be reused, inspected, or edited.
4. `llm_prompt` — the complete Qwen chat text used for generation.

The full base system prompt is visible in the node's editable `system_prompt` widget. If that widget is blank, the built-in default is restored. Sampling controls match the important controls from ComfyUI's Generate Text node: maximum generated tokens, deterministic or sampled decoding, temperature, top-k, top-p, min-p, repetition/presence penalties, seed, and thinking mode.

The optional `image` is used only as visual context while Qwen rewrites the prompt. Only the first image in a batch is used. Its presence does not automatically turn the task into image-to-video or make it an H3 keyframe.

For text-only T2VA, `conditioning` can be connected directly to a MiniMax H3 sampling workflow. For I2VA, FL2VA, L2VA, and Ref2VA, connect `enhanced_prompt` to ComfyUI's official **MiniMax H3 Image to Video** or **MiniMax H3 Reference to Video** node. Those nodes attach the required keyframe/reference VAE latents and media metadata; plain text conditioning cannot contain them.

### Planning multiple shots

Use `shot_and_timing_plan` for cuts. Write a human-readable plan; the enhancer converts it to H3 syntax:

```text
Shot 1, 00:00-00:02.500: medium shot establishing the room.
Shot 2, cut at 00:02.500: close-up of the letter being opened.
Shot 3, cut at 00:04.250: wide ending shot held through 00:06.000.
```

The enhanced prompt uses `[Shot 1]` without a timestamp, followed by `[Shot 2] At 00:02.500, ...` and `[Shot 3] At 00:04.250, ...`. Cut times must strictly increase and remain inside `duration_seconds`.

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
git clone <repository-url> ComfyUI-MiniMax-H3-Guide
```

Then add **MiniMax H3 Prompt Guide** and **MiniMax H3 Prompt Enhancer (Qwen3-VL)** from **MiniMax H3 → Prompting**.

## Practical notes

- H3 output duration is 4–15 seconds.
- Ref2VA accepts up to 9 images, 3 videos, 3 audio clips, and 12 media files in total.
- Each reference video/audio clip must be 2–15 seconds; each media type has a total duration limit of 15 seconds.
- Reference audio cannot be the sole media input; it must be accompanied by an image or video.
- The Prompt Guide node prepares text and labels only. Connect the actual media to the H3 generation nodes used by your workflow in the same label order.
- For a polished generation prompt, send `rewrite_request` to your preferred LLM node and use its response as the final H3 prompt.
- Alternatively, use the included Qwen3-VL enhancer with the MiniMax H3 CLIP. The 32B text encoder is large, so prompt enhancement can require substantial VRAM and time.

## Test

```bash
pytest -q
```
