# ComfyUI MiniMax H3 Prompt Guide

A dependency-free ComfyUI node that turns a rough video idea into the prompt structure expected by MiniMax H3. It asks how each image, video, and audio asset is meant to be used, selects the appropriate H3 mode, and hides most of the guide's bookkeeping.

The node is based on MiniMax's official [base prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md), [full-reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md), and [H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3).

## What the node does

`MiniMax H3 Prompt Guide` appears under `MiniMax H3/Prompting`. It produces:

1. `h3_prompt` — a structured draft that can be edited or sent directly to H3.
2. `rewrite_request` — a self-contained instruction for an LLM or Context-IR-style rewrite step. This is recommended when the starting notes are short because the official guide expects a detailed chronological description.
3. `mode_report` — the selected mode and checkpoint, the reason for the selection, Ref2VA limits, and warnings about contradictory options.

No model, API key, or Python dependency is required.

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

Then add **MiniMax H3 Prompt Guide** from **MiniMax H3 → Prompting**.

## Practical notes

- H3 output duration is 4–15 seconds.
- Ref2VA accepts up to 9 images, 3 videos, 3 audio clips, and 12 media files in total.
- Each reference video/audio clip must be 2–15 seconds; each media type has a total duration limit of 15 seconds.
- Reference audio cannot be the sole media input; it must be accompanied by an image or video.
- The node prepares text only. Connect the actual media to the H3 generation nodes used by your workflow in the same label order.
- For a polished generation prompt, send `rewrite_request` to your preferred LLM node and use its response as the final H3 prompt.

## Test

```bash
pytest -q
```
