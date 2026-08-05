# ComfyUI MiniMax H3 Prompt Guide

A dependency-free ComfyUI node pack that turns a rough video idea into the prompt structure expected by MiniMax H3. It separates endpoint frames from full-reference media, assigns explicit roles instead of guessing from file type, stores reusable image/audio Reference Sheets under ComfyUI user data, selects the H3 prompt family, and can use a loaded Qwen3-VL or Qwen3.5 CLIP to analyze visual references and enhance the result.

The node is based on MiniMax's official [base prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md), [full-reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md), and [H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3).

## Nodes

### Plan v2 ordered workflow

For new reference-heavy workflows, the Plan v2 nodes provide a typed semantic
chain instead of asking one large form or an LLM to infer what each file means:

    MiniMax H3 Project Setup (Plan v2)
        -> Image / Video / Audio Reference nodes
        -> optional Subject Binding nodes
        -> Shot 1
        -> optional Dialogue Event nodes
        -> Shot 2 ...
        -> MiniMax H3 Prompt Merge (Plan v2)
        -> optional Structured Prompt Enhancer (Plan v2)
        -> optional Apply Structured Prose (Plan v2)
        -> MiniMax H3 Apply Reference Plan (Plan v2)
        -> sampler

Every node consumes and returns one **MINIMAX_H3_PLAN_V2** value. Reference
nodes are accepted only before the first Shot. A Shot uses one cut time; the
next cut or Project duration computes its end. Dialogue Events attach to the
most recently opened Shot, allowing Prompt Merge to assign S1, S2, and later
speaker IDs from actual vocal-event order.

The three media nodes require an exact relationship:

- Image Reference distinguishes reusable visible content, exact first/last
  frames, concrete keyframes, and storyboard planning. It creates a Subject
  only when reusable visible content is explicitly selected.
- Video Reference distinguishes source editing, continuation, visible content,
  motion/action transfer, and camera/cut/temporal structure.
- Audio Reference distinguishes voice, music style, beat, sound-effect
  texture, exact dialogue/lyric content, continuity, complete copy, partial
  copy, and broad inspiration. Voice requires a target speaker; exact content
  requires language and transcript. Its optional paired-video handle routes
  that audio as the soundtrack belonging to the selected Video Reference.

Subject names are human aliases such as woman, truck, or wristwatch. Prompt
Merge assigns the final Subject/Picture/Video/Audio numbers, validates compact
scopes such as 3,4 or 3-4, canonicalizes native media order, and returns:

1. **h3_prompt** — an immediately usable deterministic three- or six-section
   prompt;
2. **rewrite_request** — prose-enhancement instructions that explicitly lock
   labels, roles, retention, speakers, dialogue, and cut times;
3. **plan_context** — the compiled typed plan for the structured enhancer and
   native Apply Reference Plan adapter;
4. **problems_report** — readiness, mode, timing, inventory, and exact native
   routes;
5. **h3_length** — the Project Setup native frame length.

Reference Sheet remains the reusable media library: connect its selected image
or selected audio output to the matching Plan v2 reference node, where the
workflow-specific role is declared.

The Plan v2 browser extension hides irrelevant role fields, supplies upstream
Subject pickers, validates numeric scope syntax, shows live label/route/timing
badges, fixes the first Shot at 0.000 seconds, and replaces the Shot description
box with an editor that opens an upstream reference menu when `<` is typed.
These conveniences do not replace Python validation and are not required in
API/headless mode.

The optional **Structured Prompt Enhancer (Plan v2)** gives Qwen the complete
compiled scene in one request: the valid H3 draft, all references and roles,
every Shot, exact dialogue, audio metadata/transcripts, timing, routes, and the
compiler report. When visual analysis is enabled, the same request also
contains image pixels and timestamped video samples. Audio waveforms are never
presented as something Qwen can understand; audio meaning remains explicit
metadata.

Qwen returns only a versioned JSON object containing visual style, global
intent, Shot descriptions/camera prose, soundscape, and music prose. Python
then reconstructs the complete H3 prompt and verifies that labels, reference
roles, retention, native routes, speaker order, exact dialogue, Shot order, and
cut times did not change. Invalid, incomplete, or collapsed model output falls
back to the deterministic compiler draft. The node exposes:

1. the rebuilt `enhanced_prompt` and valid `editable_prose` JSON;
2. the matching compiled `enhanced_plan_context`;
3. both the editable `base_system_prompt` and actual
   `effective_system_prompt` with its appended locked contract;
4. the complete `llm_prompt` and a validation/model-residency report.

The enhancers never synchronously unload a connected complete Qwen checkpoint.
ComfyUI's normal memory manager keeps it resident and reclaims it when another
model needs the VRAM. The serialized `offload_after_generation` switch remains
for workflow compatibility, defaults to off, and is safely ignored when enabled.
The temporary MiniMax generation tail still unloads itself after every request.

Use **Apply Structured Prose (Plan v2)** after a text editor when the returned
JSON should be refined manually and recompiled without running Qwen again.
The side-node Generation Tail Loader remains supported.

**Apply Reference Plan (Plan v2)** is the native handoff. Connect `h3_prompt`
and `plan_context` from the same Prompt Merge, Structured Prompt Enhancer, or
Apply Structured Prose result, plus the official H3 CLIP/video VAE and optional
audio VAE. The node verifies that the pair still matches, automatically routes
stored media as endpoint frames or canonical Ref2VA dictionaries, and delegates
conditioning to ComfyUI's installed `MiniMaxH3ImageToVideo` or
`MiniMaxH3ReferenceToVideo` implementation. It returns native positive
conditioning and the joint AV latent. Reference audio requires the audio VAE;
text-only and endpoint plans do not.

The adapter does not duplicate ComfyUI's encoder. It checks the installed
native call signature and fails with an actionable compatibility message when
that API changes. `adapter_report` states the selected mode, required checkpoint
family, target size, length, native implementation, and every applied route.

### Workflow presets and migration

Ready-to-open examples live in `example_workflows/` and appear in ComfyUI's
workflow template browser. The text-only prompt builder is preconfigured for
APP mode; reference starters keep their semantic spine visible so files, roles,
Shots, and dialogue can be inspected before generation. The identity-and-voice
example also includes the current official Ref2VA model, sampler, joint video/
audio decode, and Save Video path, with Apply Reference Plan replacing manual
reference-socket wiring.

For existing graphs, see [Migrating existing workflows to Plan v2](MIGRATION_TO_PLAN_V2.md).
Old node IDs remain registered so saved workflows load, but the monolithic
Prompt Guide, Target Timing/Shot chain, visual/audio context builders, and
free-form enhancer now include **Legacy** in their library names. Reference
Sheet and Generation Tail Loader remain supported components.

### Legacy: MiniMax H3 Prompt Guide

`MiniMax H3 Prompt Guide` appears under `MiniMax H3/Prompting`. It produces:

1. `h3_prompt` — a deterministic, structured pre-LLM draft. When a final Visual Reference `reference_context` is connected, the Guide derives role-correct Subject grouping, direct Picture/Video rows, retention relationships, and the H3 family from that context. Run the draft through the enhancer when the creative notes are short or visual analysis would help.
2. `rewrite_request` — a self-contained instruction for an LLM or Context-IR-style rewrite step. This is recommended when the starting notes are short because the official guide expects a detailed chronological description.
3. `mode_report` — the selected mode and checkpoint, the reason for the selection, Ref2VA limits, and warnings about contradictory options.
4. `h3_length` — the requested duration rounded upward to native ComfyUI's `17k+5` frame grid at 24 FPS. In a simple workflow it can feed the official H3 node directly. When a final Visual Reference context returns to the Guide, use the upstream Target Timing node described below; the Guide then echoes the same resolved value without creating a graph cycle.

No model, API key, or extra Python dependency is required for the guide itself.

### Legacy: MiniMax H3 Target Timing

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

### Legacy: MiniMax H3 Prompt Enhancer (Qwen LLM)

This node follows ComfyUI's native **Generate Text** execution model. Connect
`h3_prompt` and optionally `mode_report` from the guide node. Its `CLIP` input
accepts a complete generation-capable Qwen3-VL or Qwen3.5 model, or MiniMax H3's
normal 50-layer conditioning CLIP plus the optional 50–63 generation tail:

```text
MiniMax H3 Target Timing
    timing_context ──────────────────────> Prompt Guide.timing_context
    h3_length ─────┬─────────────────────> each video Visual Reference.h3_length
                  └─────────────────────> native H3.length

final Visual Reference.reference_context ─┬─> Prompt Guide.reference_context
                                          └─> Prompt Enhancer.reference_context
final Reference Sheet Audio.audio_context ──> Prompt Guide.audio_context

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

For a complete generative Qwen3-VL or Qwen3.5 CLIP, leave the enhancer's
optional `clip_tail` socket disconnected. Qwen3.5 uses ComfyUI's normal
`CLIPLoader` and the native Generate Text contract; its 4B model is a practical
general enhancer choice. The native MiniMax 32B text encoder is
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
conditioning. The connected CLIP remains under ComfyUI's normal memory manager;
the enhancer does not synchronously unload it after decoding. The legacy
`offload_after_generation` widget is retained only so older workflows continue
to load, and enabling it no longer forces a connected-model unload.
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
cache. Explicit post-generation cleanup returns only the transient tail; the
connected CLIP remains under ComfyUI's model manager. A real INT8 tail artifact
has now been smoke-tested for successful text generation; available hardware
still determines practical speed and maximum visual/prompt context.

Download the INT8 tail from
[`ethanfel/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot`](https://huggingface.co/ethanfel/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot)
and place it under `ComfyUI/models/text_encoders/MiniMax-H3/`.

Connect `enhanced_prompt` to ComfyUI's official **MiniMax H3 Image to Video** or **MiniMax H3 Reference to Video** node. Those nodes encode the prompt and attach the correct AV latent plus any keyframe/reference VAE latents and media metadata. The enhancer deliberately does not emit a separate `CONDITIONING` output because it would duplicate the official node for T2VA and be incomplete for image/reference tasks.

### Persistent Reference Sheets

A **Reference Sheet** is a reusable media library entry rather than a
character-only profile. One sheet can describe a person, object, outfit,
location, style, voice, sound, or a mixed project collection. Saved sheets live
outside the custom-node repository under:

```text
ComfyUI/user/default/minimax_h3/reference_sheets/
└── <sheet-name>--<short-id>/
    ├── manifest.json
    ├── images/
    └── audio/
```

Set `MINIMAX_H3_REFERENCE_SHEETS_DIR` before starting ComfyUI only when a
different library root is required. Each manifest is versioned and contains a
UUID, descriptions, tags, relative media paths, and SHA-256 checksums. Connected
ComfyUI `IMAGE` tensors are saved as PNG and connected `AUDIO` tensors are saved
as WAV, making the sheet independent of its creation workflow and original
input filenames. Create never overwrites another sheet; Update requires
`confirm_update` and replaces the selected sheet atomically.

Build, display, and save a sheet with one integrated node:

```text
Load Image.IMAGE ───> Reference Sheet.image_1
Load Image.IMAGE ───> Reference Sheet.image_2      optional
Load Audio.AUDIO ───> Reference Sheet.audio_1      optional

Reference Sheet
    operation: Create new
    sheet_name: reusable display name
    └─ embedded thumbnail/audio gallery
```

Connect up to four image sources and three audio sources directly. An image
batch is expanded into separate saved pictures, up to H3's nine-image limit.
Queue once to create the sheet. Later choose `Load existing`, select the sheet,
and click the desired thumbnail or audio player in the embedded gallery. The
selection is carried in `reference_sheet`; users never type or remember an
input filename or internal asset key. Update with no connected media changes
metadata only and preserves the saved media. Update with connected media
replaces the saved media after `confirm_update` is enabled.

Use saved assets directly with Plan v2:

```text
Reference Sheet.selected_image -> Plan v2 Image Reference.image
Reference Sheet.selected_audio -> Plan v2 Audio Reference.audio
```

The Plan v2 reference node assigns the actual workflow relationship, retention,
scope, speaker/layer binding, and native route. Duplicate Reference Sheet when
several saved assets must be selected independently. The old Reference Sheet
Visual/Audio Reference context nodes remain registered for existing Prompt
Guide workflows and are labeled Legacy.

The structured Qwen enhancer can analyze the selected image after it enters the
Plan v2 image inventory. It does not analyze the audio waveform: audio meaning
comes from the exact Audio Reference metadata, while native H3 receives the real
`AUDIO` value through Apply Reference Plan. Reference Sheet stores images and
standalone audio; use Video Reference for decoded video frame batches.

### Legacy: visual references, roles, and native routing

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
| `retention` | One official visible marker: `fully_preserved`, `partially_preserved`, `attribute_transfer`, or `weak_reference`. Auto uses full preservation for identity/object/scene, weak reference for style/storyboard/temporal structure, and attribute transfer for action/motion. |
| `content_group` | A stable user key for reusable visible content. Give bindings on different files the same key when the Guide should combine them as evidence for one `<Subject N>`. |
| `transfer_target` | Required whenever retention resolves to `attribute_transfer`, including Auto action/motion bindings; names a different upstream Subject that receives the attribute or motion. |
| `shot_scope` | Optional Shot numbers: `3`, `3,4`, `3-4`, or `all`. Older wording such as `Shot 3` remains supported. Leave blank when the location is not known instead of inventing Shot 1. |
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
`enhancer_report` and the candidate text before generation. Reference Sheet
audio uses a separate typed context because the enhancer model receives its
saved text description rather than the waveform.

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
reconciling two conflicting role models. Legacy audio still uses the Guide's
dropdown and inventory. When a final Reference Sheet `audio_context` is
connected, its saved descriptions and per-workflow audio relationship replace
that legacy audio path and provide exact standalone native routes.

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

For new work, add nodes from **MiniMax H3 → Plan v2**, beginning with Project
Setup and ending with Prompt Merge plus Apply Reference Plan. Reference Sheet
appears under **MiniMax H3 → Reference Sheets**, and Generation Tail Loader
remains under **MiniMax H3 → Prompting**. Other Prompting/context nodes include
Legacy in their displayed names for old-workflow compatibility. The previous
filename-based Reference Sheet Image Asset, Audio Asset, and Library builder
nodes remain disabled.

This release expects a ComfyUI build containing native MiniMax H3 support
(introduced by ComfyUI commit `57500fc5bc92`). Update ComfyUI if the official
**MiniMax H3 Image to Video** / **MiniMax H3 Reference to Video** nodes or the
MiniMax tokenizer are absent.

## Practical notes

- H3's requested output range is 4–15 seconds. Native ComfyUI rounds upward to
  `17k+5` frames, so the effective value shown in the prompt/report can be
  slightly longer (for example, 7.25 seconds becomes 175 frames / 7.292 seconds).
- H3 Ref2VA policy allows up to 9 images, 3 videos, 3 audio clips, and 12 media
  files in total. Plan v2 validates the complete mixed inventory before
  conditioning.
- H3 policy requires each reference video/audio clip to be 2–15 seconds and
  limits each media type to 15 seconds total. Video Reference and Audio
  Reference validate each asset when registered, and Prompt Merge validates
  their totals.
- H3 policy does not allow reference audio as the sole media input. Prompt Merge
  rejects that plan before Apply Reference Plan can run.
- Native Ref2VA ordering is pictures first; then each enabled video soundtrack
  `<Audio N>` immediately before its `<Video N>`; then standalone audio. Audio
  and video labels are independently numbered, so equal numbers do not imply a
  pairing. Audio Reference creates a paired `ref_video_audio_N` route only when
  its optional Video Reference handle is explicitly connected.
- Native Image to Video stretches a first frame to the target canvas and
  center-cover-crops a last frame. Match endpoint aspect ratio to output
  width/height when exact composition is important.
- Apply Reference Plan wires Plan v2 media automatically. Expert users may keep
  the official native conditioning node and connect sockets manually according
  to Prompt Merge's route report.
- For an external/general LLM node, use the structured editable-prose contract
  and Apply Structured Prose. A raw full-prompt rewrite can no longer be paired
  safely with the native adapter unless it is represented by matching compiled
  plan data.
- The included enhancer can use the same MiniMax H3 CLIP as conditioning when the Generation Tail Loader is connected. Managed partial loading is designed to support sub-32-GB cards, but that hardware tier remains unverified and the 32B autoregressive pass can be substantially slower when weights stream from system RAM. Leave DynamicVRAM enabled and avoid `--highvram` / `--gpu-only` for that use case.
- See [AUDIT_REPORT.md](AUDIT_REPORT.md) for the source-grounded findings,
  compatibility decisions, and remaining limitations.

## Test

```bash
pytest -q
```
