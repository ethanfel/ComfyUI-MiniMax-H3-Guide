# ComfyUI MiniMax H3 Prompt Guide

A dependency-free ComfyUI node pack that turns a rough video idea into the prompt structure expected by MiniMax H3. It separates endpoint frames from full-reference media, assigns explicit roles instead of guessing from file type, selects the H3 prompt family, and can use ComfyUI's loaded Qwen3-VL CLIP to analyze visual references and enhance the result.

The node is based on MiniMax's official [base prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md), [full-reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md), and [H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3).

## Nodes

### MiniMax H3 Prompt Guide

`MiniMax H3 Prompt Guide` appears under `MiniMax H3/Prompting`. It produces:

1. `h3_prompt` — a deterministic, structured pre-LLM draft. When a final Visual Reference `reference_context` is connected, the Guide derives role-correct Subject grouping, direct Picture/Video rows, retention relationships, and the H3 family from that context. Run the draft through the enhancer when the creative notes are short or visual analysis would help.
2. `rewrite_request` — a self-contained instruction for an LLM or Context-IR-style rewrite step. This is recommended when the starting notes are short because the official guide expects a detailed chronological description.
3. `mode_report` — the selected mode and checkpoint, the reason for the selection, Ref2VA limits, and warnings about contradictory options.
4. `h3_length` — the requested duration rounded upward to native ComfyUI's `17k+5` frame grid at 24 FPS. In a simple workflow it can feed the official H3 node directly. When a final Visual Reference context returns to the Guide, use the upstream Target Timing node described below; the Guide then echoes the same resolved value without creating a graph cycle.

No model, API key, or extra Python dependency is required for the guide itself.

### MiniMax H3 Target Timing

Use **MiniMax H3 Target Timing** whenever a final Visual Reference
`reference_context` will feed the Prompt Guide. It is especially important for
video references because it resolves duration before video preparation and
exposes:

1. `timing_context` — connect to `Prompt Guide.timing_context`; it carries the
   requested/effective duration and any connected Shot chain.
2. `h3_length` — connect to every video Visual Reference `h3_length` input and
   to the official H3 node's `length` input.
3. `timing_report` — the selected timing source and native `17k+5` result.

This keeps every edge pointing downstream: Target Timing prepares the length,
video references use it to trim their analysis/native batches, and only then
does the final reference context reach the Guide. Do not feed
`Prompt Guide.h3_length` back into a video Visual Reference that contributes to
the Guide's own `reference_context`.

### MiniMax H3 Prompt Enhancer (Qwen3-VL)

This node follows ComfyUI's native **Generate Text** execution model. Connect
`h3_prompt` and optionally `mode_report` from the guide node. Its `CLIP` input
accepts either a complete generation-capable Qwen3-VL model or MiniMax H3's
normal 50-layer conditioning CLIP plus the optional 50–63 generation tail:

```text
MiniMax H3 Target Timing
    timing_context ──────────────────────> Prompt Guide.timing_context
    h3_length ─────┬─────────────────────> each video Visual Reference.h3_length
                  └─────────────────────> native H3.length

final Visual Reference.reference_context ─┬─> Prompt Guide.reference_context
                                          └─> Prompt Enhancer.reference_context

Prompt Guide.h3_prompt ────┐
Prompt Guide.mode_report ──┼─> MiniMax H3 Prompt Enhancer ─> enhanced_prompt
standard CLIPLoader.CLIP ──┤                              ├─> system_prompt
legacy optional IMAGE ─────┤                              ├─> llm_prompt
Generation Tail Loader ────┘                              └─> enhancer_report
```

It produces:

1. `enhanced_prompt` — Qwen's cleaned candidate H3 prompt. Generation residue is
   removed, but the text is not silently rewritten after decoding; review
   `enhancer_report` for structural warnings before generation.
2. `system_prompt` — the resolved base enhancer instructions, exposed so they can be reused, inspected, or edited.
3. `llm_prompt` — the serialized text/chat portion sent to Qwen. Pixel tensors and MiniMax reference blocks are tokenizer inputs and therefore are not embedded in this string. An external multimodal LLM must receive the pixels separately in its own visual-token format.
4. `enhancer_report` — the resolved H3 family, generation status,
   compatibility/fallback details, and structural H3 warnings.

The full base system prompt is visible in the node's editable `system_prompt` widget. If that widget is blank, the built-in default is restored. Exact unmodified defaults serialized by older releases are upgraded automatically; any customized prompt is preserved. Sampling controls match the important controls from ComfyUI's Generate Text node: maximum generated tokens, deterministic or sampled decoding, temperature, top-k, top-p, min-p, repetition/presence penalties, seed, and thinking mode.

The optional `image` remains as a compatibility route for one context image.
For labeled pictures, multiple images, or video understanding, use the Visual
Reference chain below. Visual context helps Qwen write the prompt; it never
silently turns a picture into an endpoint frame or replaces the media inputs
on the native H3 node.

For a complete generative Qwen3-VL CLIP, leave the enhancer's optional
`clip_tail` socket disconnected. ComfyUI's supported full generative Qwen3-VL
routes are normally the 4B/8B families. The native MiniMax 32B text encoder is
the deliberately truncated conditioning model described below; loading that
checkpoint does not by itself create a complete 32B LLM.

For MiniMax H3's bundled conditioning CLIP, add **MiniMax H3 Generation Tail
Loader**, select
`qwen3vl_32b_minimax_h3_generation_tail_50_63_int8_convrot.safetensors` in
the loader, and connect its `clip_tail` output to the enhancer. The loader
passes a lightweight descriptor and consumes no VRAM by itself. During
enhancement, the enhancer reuses the connected embedding, vision tower, and
language layers 0–49, loads only layers 50–63 plus the final norm and LM head,
then unloads that tail when generation finishes. Tail KV caches and embeddings
are released before the managed unload and CUDA cache flush. The connected
50-layer CLIP is never merged or modified and remains suitable for official H3
conditioning. The enhancer's `offload_after_generation` option defaults to on
and explicitly moves that connected CLIP to its configured ComfyUI offload
device after decoding. Turn it off only when the same CLIP is consumed
immediately downstream and avoiding a reload is more important than freeing
VRAM.
If the truncated CLIP is connected without the side loader, enhancement is
safely skipped and the manual prompt is returned unchanged.

The tail loader accepts only the published split layout. Its chunked LM head
supports ComfyUI tensor-wise INT8 scalar/per-row scales and rejects other
quantized layouts explicitly. The complete model does not have to fit in VRAM:
the base and tail are both registered with ComfyUI's managed patchers, so
DynamicVRAM streams/caches weights on demand and legacy Normal VRAM can
partially load them. This means a card below 32 GB may run the enhancer when it
has enough VRAM for the largest active layer, KV cache, vision tensors, and
runtime headroom, plus enough system RAM for offloaded weights; that full path
has not yet been hardware-verified below 32 GB. It will be much slower because
autoregressive generation revisits every language layer for each token.
`--highvram` and especially `--gpu-only` defeat this low-VRAM behavior; with
`--gpu-only`, the configured load and offload devices are identical.

High `nvidia-smi` usage on a larger card does not itself mean full residency is
required: ComfyUI opportunistically uses available VRAM and may retain allocator
cache. The explicit post-generation cleanup is what returns the transient tail
and, by default, the connected CLIP afterward. A real INT8 tail artifact has
now been smoke-tested for successful text generation; available hardware still
determines practical speed and maximum visual/prompt context.

Download the INT8 tail from
[`ethanfel/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot`](https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot)
and place it under `ComfyUI/models/text_encoders/MiniMax-H3/`.

Connect `enhanced_prompt` to ComfyUI's official **MiniMax H3 Image to Video** or **MiniMax H3 Reference to Video** node. Those nodes encode the prompt and attach the correct AV latent plus any keyframe/reference VAE latents and media metadata. The enhancer deliberately does not emit a separate `CONDITIONING` output because it would duplicate the official node for T2VA and be incomplete for image/reference tasks.

### Visual references, roles, and native routing

Use one **MiniMax H3 Enhancer Visual Reference** node per picture or reference
video. `previous_context` records assets in chain order. The backend then
numbers pictures and videos independently and recommends sockets in native H3
category order: pictures first, then videos, while preserving chain order
within each category. A separate **MiniMax H3 Visual Reference Role** chain
assigns one or more semantic jobs to a single media file:

```text
Role: identity ─> Role: clothing ─> Visual Reference.role_bindings
                                       │ media: Picture
Previous Visual Reference.context ─────┤
                                       ├─ reference_context ─> next Visual Reference.previous_context
                                       └─ h3_media ─────────> socket recommended by routing_report

final Visual Reference.reference_context ─┬─> Prompt Guide.reference_context
                                          └─> Prompt Enhancer.reference_context
```

New Visual Reference nodes start with `Unassigned - choose a reference role`.
Before running, either select one simple role in the compatibility
`reference_role` dropdown or connect a completed `role_bindings` chain. Use
role nodes when one asset has several roles, when several assets should provide
evidence for one Subject, or when retention/shot mapping must be explicit. Fan
the final Visual Reference `reference_context` out to both the Prompt Guide and
Prompt Enhancer. The Guide deterministically writes the role-correct Subject or
direct Picture/Video rows; Qwen then analyzes the supplied pixels and expands
the creative description without being asked to invent the role mapping.

The role fields mean:

| Field | Meaning |
| --- | --- |
| `reference_role` | What content the asset provides: endpoint, identity, object, scene, style, keyframe, storyboard, motion, temporal structure, edit source, or continuation source. The `Unassigned - choose a reference role` new-node default must be replaced before execution. |
| `retention` | One official visible marker: `fully_preserved`, `partially_preserved`, `attribute_transfer`, or `weak_reference`. Auto chooses a role-safe non-transfer default. |
| `content_group` | A stable user key for reusable visible content. Give bindings on different files the same key when the Guide should combine them as evidence for one `<Subject N>`. |
| `transfer_target` | Required only for explicit `attribute_transfer`; names a different content group that receives the attribute or motion. |
| `shot_scope` | Optional explicit location such as `Shot 2` or `Shots 1-3`. Leave blank when the location is not known instead of inventing Shot 1. |
| `notes` | What to preserve, transfer, ignore, or change for this binding. |

Two route families are intentionally exclusive:

- **Endpoint context:** `Exact first frame` and/or `Exact last frame` stays in
  I2VA/L2VA/FL2VA. The report maps each `h3_media` output to native **MiniMax
  H3 Image to Video** `first_frame` / `last_frame`. Two endpoint pictures are
  analyzed in native Picture 1/Picture 2 order.
- **Ref2VA context:** reusable content, concrete keyframes, storyboards, motion,
  temporal structure, edit sources, and continuation sources receive recommended
  native **MiniMax H3 Reference to Video** `ref_image_N` / `ref_video_N` inputs.

The backend `routing_report` is the authoritative description of the intended
mapping. It cannot create native-node links: connect every `h3_media` output to
the listed socket yourself. Canvas output labels are convenience hints,
especially when role chains, reroutes, or bypassed nodes are present. Do not
mix endpoint and Ref2VA roles in one context chain.

The media paths deliberately have different representations:

- **Picture passed to H3:** the original picture is unchanged. In Ref2VA,
  native `ref_image_size=match|max` remains authoritative.
- **Video passed to H3:** the source batch is resampled to 24 FPS, optionally
  truncated to connected `h3_length`, and rounded downward to native H3's
  `17k+5` reference grid. Set `source_fps` to the real batch rate and connect
  Target Timing's `h3_length` to every video reference node when the final
  context also feeds the Guide. Use the Guide's output only in a legacy path
  where doing so cannot form a cycle.
- **Generic-Qwen analysis:** a reduced long-edge copy with configurable
  `analysis_fps` and frame cap, sampled only from the effective native clip.
- **MiniMax-Qwen analysis:** a separate fixed-2-FPS sequence, matching native
  MiniMax temporal pairs. It is not affected by the generic frame cap.

Reference videos must be 2–15 seconds before native alignment and total at most
15 seconds. The report shows both source and effective duration, and Qwen's
visual evidence excludes the discarded tail. Free-form text can still mention
discarded events, so review the candidate prompt when the source was trimmed. A
48-frame 24-FPS source, for example, becomes 39 native frames and four MiniMax
samples at 0.0, 0.5, 1.0, and 1.5 seconds.

H3's `<Subject N>` is reusable visible content, not a synonym for a person. It
may represent an object, environment, style, action, expression, or pose. From
the connected context, the Guide cites a picture/video used only as reusable
Subject evidence inside that Subject's definition without adding an unnecessary
standalone definition/retention row. Concrete frames remain `<Picture N>`;
edit/continuation/whole-video temporal sources remain `<Video N>`. The enhancer
can expand those definitions from visual evidence and checks the resulting
structure, but the explicit bindings remain authoritative. Review
`enhancer_report` and the candidate text before generation. Audio analysis is
intentionally outside this visual chain for now.

### Planning multiple shots

Add one **MiniMax H3 Shot** node per shot and connect each `shot_plan` output to
the next node's `previous_shots` input. With a chained Visual Reference context,
connect the final Shot to Target Timing so the complete plan stays upstream:

```text
MiniMax H3 Shot (0.000–2.500)
    shot_plan ─> MiniMax H3 Shot (2.500–4.250)
                    shot_plan ─> MiniMax H3 Shot (4.250–6.000)
                                    shot_plan ─> Target Timing.shot_plan

Target Timing.timing_context ─> Prompt Guide.timing_context
Target Timing.h3_length ──────┬─> every video Visual Reference.h3_length
                              └─> native H3.length
```

Without a connected `reference_context`, the older direct route remains valid:
connect the final Shot to `Prompt Guide.shot_plan`, and use
`Prompt Guide.h3_length` downstream.

Each node has float `start_time` and `end_time` controls with millisecond
steps, a shot description, per-shot camera direction, and transition. The
chain rejects gaps, overlaps, reversed ranges, a first shot that does not start
at zero, and times above 15 seconds. Its final `end_time` is the requested
duration. Target Timing—or the Guide in the legacy direct path—rounds that
duration to native H3 frames and extends the last described shot through the
effective playback end. Camera instructions are written as natural shot prose,
never as a `Camera direction:` metadata label.

The Prompt Guide's `shot_and_timing_plan` text widget remains available under
advanced controls for old workflows or a quick manual plan. It now parses the
same common syntax into real H3 markers, for example:

```text
Shot 1, 00:00-00:02.500: medium entrance.
Shot 2, cut at 00:02.500: close-up reaction.
Shot 3, cut at 00:04.250: wide ending.
```

Numbers, gaps, ranges, descriptions, and cut order are validated. A connected
Shot chain takes priority. Final-frame alignment always cites a Shot marker
that actually exists.

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

With per-asset role nodes, the equivalent explicit mapping is:

| Media | Role | Content group | Retention | Transfer target |
| --- | --- | --- | --- | --- |
| Picture 1 | Identity or appearance | `hero` | `fully_preserved` | — |
| Video 1 | Motion or action | `reference-motion` | `attribute_transfer` | `hero` |

To combine rather than transfer evidence, give bindings the same group. For
example, Picture 1 `Identity or appearance` and Video 1 `Motion or action` can
both use `hero`; the Guide then defines one Subject whose appearance comes from
the picture and whose motion evidence comes from the video, without inventing
an unrelated second Subject. The enhancer may add details found in the media,
but it receives the same fixed grouping.

## Reference inventory

Enter one asset per line. Labels describe the native sockets to which you intend
to connect the corresponding media:

```text
Picture 1: a red ceramic robot, front three-quarter view
Video 1: a dancer performing a quick clockwise spin
Audio 1: a dry studio recording of a calm female voice
```

Angle brackets are optional. `Picture 1: ...` and `<Picture 1>: ...` are
equivalent. Labels must be positive and unique. Active gaps are rejected;
out-of-order entries are reported in `mode_report` so you can match native
category order. Unlabelled lines are retained as additional reference notes.

Inventory text describes expected files; it does not decide their role. A
listed Picture with image role `No image`, for example, stays unused and
produces a warning instead of silently becoming an appearance reference or
Subject. In the legacy dropdown path, selecting a role may make the text-only
Guide synthesize a required placeholder label to keep the draft structurally
complete, including Picture 2 when a partially listed first-and-last-frame task
needs it. A placeholder is not a media file: `mode_report` calls it out, and you
must verify that every generated label has a real downstream connection.

The Guide's global image/video role dropdowns remain a legacy shortcut when no
`reference_context` is connected. In that path they select one role per media
type and may create structurally required placeholder labels, exactly as older
workflows expect.

For chained references, the final Visual Reference `reference_context` is the
single authoritative visual model. Connect it to both the Guide and Enhancer.
The Guide ignores its legacy image/video role dropdowns, derives Subject
grouping, direct Picture/Video rows, retention, task prefix, and H3 family from
the explicit bindings, and uses matching inventory lines only as descriptions.
The enhancer analyzes and expands that already aligned draft instead of
reconciling two conflicting role models. Audio still uses the Guide's audio
dropdown and inventory because audio is outside the visual chain.

## Example: edit a video and keep its soundtrack

Set `how_video_is_used` to `Directly edit the source video` and `how_audio_is_used` to `Reuse the complete audio signal`. The node selects Ref2VA and starts the summary with:

```text
[video editing + audio reuse] The target video is an edited version of <Video 1>.
```

The output also distinguishes the video retention marker from the audio
`fully_copy` marker. The Guide states that no new layer may be added and warns
when its combined dialogue/text field could imply a new vocal signal. The
enhancer instructs Qwen to keep the copy exclusive, and its structural check
requires applicable audio sections to cite the copied label and state that
exclusivity. Free-form target descriptions and custom system prompts still
cannot be semantically proven compatible; review them and use partial copy when
the target adds or replaces sound.

## Install

Clone or copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide
```

Then add **MiniMax H3 Prompt Guide**, **MiniMax H3 Shot**, **MiniMax H3 Target
Timing**, **MiniMax H3 Visual Reference Role**, **MiniMax H3 Enhancer Visual
Reference**, **MiniMax H3 Generation Tail Loader**, and **MiniMax H3 Prompt
Enhancer (Qwen3-VL)** from **MiniMax H3 → Prompting** as needed.

This release expects a ComfyUI build containing native MiniMax H3 support
(introduced by ComfyUI commit `57500fc5bc92`). Update ComfyUI if the official
**MiniMax H3 Image to Video** / **MiniMax H3 Reference to Video** nodes or the
MiniMax tokenizer are absent.

## Practical notes

- H3's requested output range is 4–15 seconds. Native ComfyUI rounds upward to
  `17k+5` frames, so the effective value shown in the prompt/report can be
  slightly longer (for example, 7.25 seconds becomes 175 frames / 7.292 seconds).
- H3 Ref2VA policy allows up to 9 images, 3 videos, 3 audio clips, and 12 media
  files in total. The visual chain validates its image/video portion, but cannot
  inspect separately wired audio or enforce the mixed total against those audio
  connections.
- H3 policy requires each reference video/audio clip to be 2–15 seconds and
  limits each media type to 15 seconds total. Video duration is validated by the
  visual chain; audio duration remains the user's responsibility.
- H3 policy does not allow reference audio as the sole media input. The Guide
  can warn from its inventory, but the visual chain cannot validate actual audio
  wiring.
- Native Ref2VA ordering is pictures first; then each enabled video soundtrack
  `<Audio N>` immediately before its `<Video N>`; then standalone audio. Audio
  and video labels are independently numbered, so equal numbers do not imply a
  pairing. The visual chain does not create or analyze audio entries yet.
- Native Image to Video stretches a first frame to the target canvas and
  center-cover-crops a last frame. Match endpoint aspect ratio to output
  width/height when exact composition is important.
- The Prompt Guide and Visual Reference chain prepare text, labels, and intended
  routes; they do not wire native H3 inputs. Connect the actual media yourself
  according to the final `routing_report` and label order.
- For an external/general LLM node, send `rewrite_request` and use its response
  as the candidate H3 prompt. The included Prompt Enhancer instead expects
  `h3_prompt` plus `mode_report`.
- The included enhancer can use the same MiniMax H3 CLIP as conditioning when the Generation Tail Loader is connected. Managed partial loading is designed to support sub-32-GB cards, but that hardware tier remains unverified and the 32B autoregressive pass can be substantially slower when weights stream from system RAM. Leave DynamicVRAM enabled and avoid `--highvram` / `--gpu-only` for that use case.
- See [AUDIT_REPORT.md](AUDIT_REPORT.md) for the source-grounded findings,
  compatibility decisions, and remaining limitations.

## Test

```bash
pytest -q
```
