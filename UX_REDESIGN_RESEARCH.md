# MiniMax H3 prompt workflow UX redesign

Research date: 2026-08-04
Development branches: `dev/reference-workflow-ux`,
`dev/reference-workflow-ux-phase2`, `dev/reference-workflow-ux-phase3`

Implementation status: Phase 1 provides the isolated `MINIMAX_H3_PLAN_V2`
compiler and eight Plan v2 planning nodes. Phase 2 adds the interaction
extension, hard-locked full-context structured enhancer, and manual Apply
Structured Prose node. Phase 3 adds the native Apply Reference Plan adapter,
workflow onboarding assets, direct Reference Sheet audio output, and an
explicit legacy migration path.

## Recommendation

Replace the current large Prompt Guide form with a small, ordered planning
pipeline:

```text
H3 Project Setup
    -> H3 Image / Video / Audio Reference nodes
    -> H3 Shot nodes (+ optional structured Dialogue events)
    -> H3 Prompt Merge
    -> H3 Prompt Enhancer (optional Qwen rewrite)
```

The chain should be the source of truth. The merger should assign all
`<Subject N>`, `<Picture N>`, `<Video N>`, `<Audio N>`, and native socket
routes deterministically. The LLM should expand the creative description, not
decide what a reference means.

The existing Reference Sheet should remain a reusable media library and
gallery. Selecting or loading media is a different responsibility from
declaring how that media is used in one generation.

The recommended graph is a **semantic spine**, not merely a string-building
chain:

```text
Loaders / Reference Sheet outputs
            |
            v
H3 Project Setup
  -> H3 Image Reference
  -> H3 Subject Binding                 (advanced, optional)
  -> H3 Video Reference
  -> H3 Audio Reference
  -> H3 Shot 1
  -> H3 Dialogue Event                  (optional; attaches to Shot 1)
  -> H3 Shot 2
  -> H3 Prompt Merge
       |-> immediately usable H3 prompt
       |-> enhancer request + locked plan
       `-> native route plan
              |
              v
       H3 Prompt Enhancer                (optional)
              |
              v
       H3 Apply Reference Plan           (optional native adapter)
```

References intentionally precede shots. This gives every Shot node a complete
catalog for autocomplete and validation. A Shot opens the current timeline
segment; any following Dialogue Event attaches to that Shot until the next
Shot opens. The merger closes the final shot at the global duration.

The chain should carry one versioned custom type, `MINIMAX_H3_PLAN_V2`. Media
nodes append structured facts to this plan instead of appending prose. The
workflow JSON stores the nodes and links as usual; runtime media tensors are
passed by reference inside the plan and are not serialized into the workflow.

## Why the current workflow is unreliable

### One audio option represents several incompatible intentions

The current value `Reference voice, music, beat, or sound` becomes a generic
definition and generic audio-section text. Those alternatives do not have the
same prompt behavior:

- A voice reference must identify its target speaker and use that speaker's
  stable `(Sx)` ID.
- A music-style reference belongs in the audible music layer it controls.
- A beat reference controls timing or rhythm without necessarily controlling
  instrumentation.
- A sound-effect reference belongs to a particular event or sound layer.
- Dialogue or lyric content requires exact words and the original language.
- Audio continuity is a temporal relationship, not a voice or music style.

The official guide allows different audio assets in the same workflow to have
different relationships. The current audio context rejects mixed relationships,
so it cannot express the official example of partially copied background music
plus a separate voice-timbre reference.

### The large Guide form mixes unrelated levels of work

The current Prompt Guide simultaneously asks for:

- the global creative intent;
- mode selection;
- media inventory;
- global media roles;
- retention strength;
- timing and shots;
- camera instructions;
- dialogue and visible text;
- ambience and music;
- legacy and connected-context behavior.

Several of these fields are ignored when a reference context is connected.
The user must therefore understand implementation precedence before they can
predict the result. The form also exposes broad media-type roles even though
the official format tracks the purpose of each individual asset.

### Labels are shown too late

The user writes Shot prose before having a visible, local list of the labels
created by the reference setup. They must remember whether the woman is
`<Subject 1>` or `<Subject 2>`, whether a saved image creates a Subject or a
standalone Picture, and which Audio number belongs to a voice. This is a
recognition-versus-recall failure and makes detached instructions such as
"apply Subject 1" likely.

### A model rewrite cannot repair missing semantics reliably

The Qwen enhancer can inspect connected image/video pixels, but it does not
hear standalone audio waveforms. It cannot reliably infer that an audio clip is
a particular woman's voice, a score reference, a beat source, or a copied
layer. Text metadata and explicit bindings must establish those facts before
generation.

MiniMax also describes H3 Context-IR as a multi-stage preprocessing and
orchestration system. A single free-form local LLM rewrite is useful for prose
expansion, but it is not a substitute for typed relationship setup.

## Proposed v2 nodes

### 1. MiniMax H3 Project Setup

Purpose: define only properties of the target video as a whole.

Inputs:

- `initial_prompt`: the user's global creative intent.
- `duration_seconds`: authoritative target duration, 4-15 seconds.
- `visual_style`: optional explicit override; blank derives it from intent and
  references.
- `overall_soundscape`: optional global ambience and physical sounds.
- `non_diegetic_music`: optional audience-only score; `N/A` means none.

Outputs:

- `h3_plan`: the chain context used by every following v2 node.
- `h3_length`: the native `17k+5` frame count, available early enough for
  reference-video preparation and the native conditioning node.
- `project_preview`: concise target duration and global choices.

The global duration remains authoritative. Shot cuts must fall inside it, and
the final shot implicitly continues to the effective native duration. This
removes the current timing/reference feedback-cycle problem.

### 2. MiniMax H3 Image Reference

Purpose: append one actual image and one explicit primary relationship.

Suggested top-level uses:

- `Define reusable visible content`
- `Exact first frame`
- `Exact last frame`
- `Concrete keyframe / composition anchor`
- `Storyboard / shot planning`

When defining reusable content, expose a second exact content type:

- identity or appearance;
- object, prop, clothing, interface, or visual effect;
- scene or environment;
- visual style;
- pose, expression, action, or motion.

Use a human-readable `subject_name`, such as `woman`, `truck`, or `Arizona
road`, rather than asking users to invent `content_group` identifiers. The
merger maps stable subject names to `<Subject N>`. Several assets with the same
subject name jointly define one Subject. Different names remain different
Subjects.

For the common case, these fields define one role directly on the Image
Reference node. An optional advanced `H3 Subject Binding` chain should let one
image define several separately named Subjects, or contribute several distinct
roles, without registering or connecting the native image more than once.

### 3. MiniMax H3 Video Reference

Purpose: append one actual video and its exact relationship.

Suggested uses:

- `Source video to edit`
- `Source video to continue`
- `Define reusable visible content`
- `Motion or action reference`
- `Camera, cuts, rhythm, or temporal-structure reference`

The common form defines one relationship. It accepts the same optional Subject
Binding chain when one video supplies several separately tracked people,
objects, environments, actions, or styles.

An optional paired soundtrack input should be declared explicitly. The native
route is `ref_video_audio_N`, and its `<Audio N>` label is inserted immediately
before the corresponding `<Video N>` in the native presentation order.

### 4. MiniMax H3 Audio Reference

Purpose: append one actual audio clip with one exact relationship. Do not use
an option containing "voice, music, beat, or sound".

Suggested exact uses and required metadata:

| Use | Required binding | Deterministic definition pattern |
| --- | --- | --- |
| Voice timbre and delivery | target `subject_name`, or a stable nonvisual speaker name | `<Audio N> is the voice-timbre and delivery reference for <Subject M> (Sx).` |
| Background-music style | diegetic or non-diegetic destination | `<Audio N> is the background-music style reference, guiding ... without copying the signal.` |
| Beat or rhythm | target action, edit, or music layer | `<Audio N> is the beat-and-rhythm reference for ...` |
| Sound-effect texture | named event or sound layer | `<Audio N> is the sound-effect texture reference for ...` |
| Dialogue or lyric content | exact transcript and language | `<Audio N> provides the referenced spoken/lyric content ...` |
| Audio continuity | source/target phase or continuation | `<Audio N> is the audio-continuity reference for ...` |
| Copy complete signal | no competing new audio | `<Audio N> is reused as the complete final audio track.` |
| Copy selected part/layers | exact range/layer instructions | `<Audio N> provides the copied ... layer/range.` |
| Broad inspiration | described category/atmosphere | `<Audio N> is a weak ... reference.` |

Conditional widgets should appear only for the selected use. A voice clip
should ask who owns the voice. A music reference should not show a speaker
field. Direct dialogue/lyric reuse should require a transcript because Qwen
does not inspect the waveform and the official format forbids guessing unclear
words.

Mixed audio relationships must be valid in one chain. Each `<Audio N>` keeps
its own retention marker and section placement.

The merger must canonicalize native presentation order even if users arrange
setup nodes differently: all pictures first, then each reference video's
enabled soundtrack immediately before that video, then standalone audio.
Picture, Video, and Audio numbering remains independent.

### 5. MiniMax H3 Shot

Purpose: append one shot in playback order.

The node receives `h3_plan`, so it already knows every upstream reference. It
does not ask the user to type `[Shot N]`; numbering follows chain order.

Inputs:

- `cut_at`: hidden or fixed at `0.000` for the first shot; later shots use one
  float cut time. The next cut or global duration determines the shot's end.
- `description`: only the visible/audible events in this shot.
- `camera_direction`: optional natural camera motion.
- `transition`: direct cut by default, with explicit dissolve/fade/wipe options.

Using one cut time per shot matches the official output format and removes the
redundant start/end pair that currently creates gaps and overlaps.

#### `<` reference autocomplete

When the user types `<` in the Shot description, a popup should list the labels
already available from the upstream setup chain, for example:

```text
<Subject 1>  Woman - identity from Picture 1
<Subject 2>  Truck - object from Picture 2
<Video 1>    Source video to edit
<Audio 1>    Woman's voice - bound to Subject 1
```

The menu should filter while typing, support arrow keys, Enter/Tab, and Escape,
and insert only the selected H3 label. It may also offer an explicit speaker
snippet such as `<Subject 1> (S1)` after speaker IDs have been resolved.

The frontend can implement this through ComfyUI's JavaScript extension hooks
and the multiline STRING widget's input element. It can traverse the connected
upstream chain, as the existing route-label extension already does. Backend
validation must remain authoritative so API-mode workflows and manually typed
labels are equally safe.

### 6. MiniMax H3 Dialogue Event (optional)

Purpose: preserve exact words and assign speaker IDs from actual playback order
instead of guessing them from prose.

One or more Dialogue Event nodes may feed a Shot. Each event contains:

- `speaker`: an upstream `subject_name` or stable nonvisual speaker name;
- `language`;
- `exact_text`;
- `delivery`;
- `voice_mode`: on-screen speech, off-screen speech, or voiceover;
- optional timing/continuity choices for `<scenetrans>` and `<cutoff>`.

The merger assigns `(S1)`, `(S2)`, and later IDs by the order of these actual
vocal events across the completed Shot chain. It then resolves any Audio
Reference bound to that speaker and writes the same `(Sx)` in the Audio
definition. This is necessary because the official guide explicitly forbids
assigning speaker IDs independently in audio definitions.

The free-form Shot description remains available for visual action and simple
expert-authored dialogue, but structured events are the recommended route when
voice references, multiple speakers, voiceover, dialogue across cuts, or exact
word preservation matter.

### 7. MiniMax H3 Prompt Merge

Purpose: consume the completed plan and produce the H3 prompt package.

Outputs:

- `h3_prompt`: deterministic pre-LLM prompt with the correct three- or
  six-section structure.
- `rewrite_request`: self-contained text for an external LLM.
- `plan_context`: authoritative structured data for the bundled enhancer.
- `problems_report`: only actionable conflicts, limits, missing bindings, and
  native routes.
- `h3_length`: the same native frame count created by Project Setup.

The merger automatically chooses T2VA, I2VA, FL2VA, L2VA, or Ref2VA from the
actual reference relationships. It should never create a generic Subject just
because Ref2VA is active.

For Ref2VA it deterministically creates:

1. `subject_definitions`
2. `summary`
3. `retention_analysis`
4. `detailed_description`
5. `overall_soundscape`
6. `non_diegetic_music`

The enhancer may expand descriptions, but labels, subject grouping, speaker
bindings, task types, retention markers, routes, and shot cut times are locked.

### 8. MiniMax H3 Prompt Enhancer

Purpose: improve only the descriptive fields that benefit from a language or
vision-language model.

The enhancer should receive both the draft prompt and the structured locked
plan. Its editable fields are limited to visual description, action clarity,
camera prose, transitions, ambience, and music description. It must not invent
or renumber references, change retention, assign speakers, change exact
dialogue, or move cut times.

The safest contract is structured output rather than asking the model to
rewrite the complete six-section prompt. The model returns enhanced global and
per-shot prose; the node validates that response and reconstructs the final H3
prompt from the locked plan. If parsing or semantic validation fails, the node
returns the valid pre-LLM prompt and an actionable warning instead of emitting
a corrupted prompt.

Inputs and outputs should preserve the existing editing workflow:

- optional `system_prompt` override input;
- optional side-node tail input;
- `base_system_prompt` output, so the built-in contract can be inspected or
  edited;
- `system_prompt_used` output, including the applied override/tail;
- `enhanced_h3_prompt` and `enhancer_warnings` outputs.

Qwen3-VL can use connected images and sampled video frames to improve visible
details and temporal descriptions. It is not an audio-understanding path in
this workflow. Voice ownership, transcript, language, copied layers, timing,
and other audio semantics must therefore arrive as text from the typed Audio
Reference and Dialogue nodes.

### 9. MiniMax H3 Apply Reference Plan (optional native adapter)

Purpose: eliminate the remaining manual synchronization between prompt labels
and the native H3 conditioning sockets.

The adapter accepts the compiled plan, final prompt, CLIP, VAE, optional audio
VAE, and target size. It presents media to H3 in the native canonical order and
returns conditioning and the empty latent. It should use a narrow compatibility
layer around ComfyUI's native H3 implementation and fail with a version-specific
message if that API changes.

This node is important for the best end-to-end UX, but it is not required for
the first compiler milestone. Until it is implemented, each Reference node can
provide a media pass-through output named with its resolved route and Prompt
Merge can expose the same route table for connection to the native node.

Keeping the adapter separate from Prompt Merge has three benefits:

- prompt planning and validation still work without H3 weights loaded;
- users can keep the official native conditioning node when they prefer it;
- compatibility changes in ComfyUI do not destabilize the semantic compiler.

## Example: woman with a voice reference

Setup chain:

```text
Project Setup
-> Image Reference
     use: Define reusable visible content
     content type: Identity or appearance
     subject name: woman
-> Audio Reference
     use: Voice timbre and delivery
     target subject: woman
-> Dialogue Event (connected to Shot 1)
     speaker: woman
     language: English
     exact text: Exact user-provided dialogue.
-> Shot 1
     description: As the truck moves, <Subject 1> sits beside the driver...
-> Prompt Merge
```

The merger should produce the exact relationship:

```text
<Subject 1> is the woman whose identity and appearance come from <Picture 1>.
<Audio 1> is the voice-timbre and delivery reference for <Subject 1> (S1).
```

When she speaks in a shot:

```text
<Subject 1> (S1) speaks using the voice timbre and delivery referenced from
<Audio 1>: <d>[English] Exact user-provided dialogue.</d>
```

No wording about music, beat, generic sound, or copying the source words should
appear for this setup.

## Validation and error prevention

The v2 merger should reject or clearly diagnose:

- audio as the only Ref2VA input;
- missing voice target;
- a voice target that has no matching vocal event, or a vocal event whose
  speaker has no resolvable subject/nonvisual speaker identity;
- dialogue/lyric-content reference without transcript and language;
- partial copy without a declared time range or layer;
- exact endpoint roles mixed into a Ref2VA native route;
- unknown labels typed in a Shot;
- a referenced label that never appears where its scoped role is needed;
- cut times that are not increasing or fall outside the global duration;
- unsupported media counts/durations or total file count;
- native media order that differs from the assigned prompt label order;
- complete audio copy combined with newly requested dialogue, effects, ambience,
  or music.

The report should state the fix in user language, for example: `Audio 1 is set
to Voice, but no target subject is selected. Choose Woman or another speaker.`

## Compatibility strategy

Build the v2 chain under new internal node IDs so existing workflows continue
to load. Mark the current Prompt Guide, Target Timing, Visual Reference Role,
and context-builder nodes as legacy after the v2 path reaches feature parity.
Do not make the old large form the recommended entry point.

Keep Reference Sheet persistence compatible. Its job is to save, load, preview,
and output selected media; the new Image/Audio Reference nodes assign the
workflow-specific semantics after loading.

## Architectural alternatives

### Plan A - ordered semantic spine (recommended)

All project, reference, shot, and dialogue nodes consume and return one plan.
References must be configured before the first Shot. Prompt Merge is a pure,
deterministic compiler, and an optional adapter performs native conditioning.

Strengths:

- lowest practical wire count without hiding the workflow;
- a complete upstream catalog is available in every Shot;
- order of shots and vocal events is visible directly on the canvas;
- works in API mode because the backend plan is authoritative;
- supports simple cases with one field per node and advanced bindings only when
  requested;
- one validation pass can check prompt labels, scopes, timing, and native routes
  together.

Tradeoffs:

- adding a new reference requires inserting it before the first Shot;
- changing an early node invalidates downstream execution cache, as expected
  for a compiled plan;
- the optional native adapter needs a maintained ComfyUI compatibility layer.

### Plan B - parallel specialist chains feeding a catalog hub

```text
Image Reference chain --\
Video Reference chain --- H3 Reference Catalog -> H3 Shot chain -> Prompt Merge
Audio Reference chain --/                         ^
Project Setup ------------------------------------|
```

This keeps media categories independent and is the most conservative extension
of the current code. It is a reasonable fallback if one-plan chaining proves
incompatible with third-party nodes.

Strengths:

- category-specific chains can be edited and tested independently;
- less media is carried through unrelated setup nodes;
- the compiler remains headless and deterministic.

Tradeoffs:

- more wires and collector nodes;
- audio-to-subject bindings cross chain boundaries;
- the Shot chain needs both project and reference-catalog context;
- precedence and partial-connection states are harder for users to understand;
- it trends toward the current collection of context sockets rather than the
  requested setup flow.

### Plan C - one structured H3 Composer UI

A custom node or side panel would contain tabs for Project, References, Shots,
Dialogue, and Validation, with sortable media cards and timeline rows. This can
be the smoothest novice interface when it is complete.

Strengths:

- almost no canvas wiring;
- drag-and-drop ordering, inline previews, and immediate validation are easy to
  present in one surface;
- arbitrary references and shots can be managed like a small editor.

Tradeoffs:

- highest implementation and maintenance cost;
- duplicates ComfyUI graph, widget, serialization, and media-management
  behavior inside one custom UI;
- weaker API/headless ergonomics unless a separate schema API is maintained;
- repeats the monolithic-node problem the redesign is intended to remove;
- a frontend failure can make the complete workflow difficult to edit.

It should not be the first implementation. It can later be built as an App-mode
or composer view over exactly the same `MINIMAX_H3_PLAN_V2` schema, so it does
not create a second prompt engine.

### Comparison

| Criterion | Plan A: semantic spine | Plan B: parallel hub | Plan C: composer UI |
| --- | --- | --- | --- |
| Semantic safety | High | High | High if schema is duplicated correctly |
| Canvas clarity | High | Medium-low | Very high |
| Wire count | Low | High | Very low |
| Arbitrary refs/shots | High | High | High |
| API/headless use | High | High | Low-medium |
| Inspectability | High | High | Medium |
| Frontend complexity | Medium | Low-medium | Very high |
| Native-route automation | High with adapter | Medium | Medium-high |
| Fit with requested workflow | Best | Acceptable fallback | Poor first step |

### Distribution layer - presets, subgraphs, and App mode

This complements Plan A rather than replacing it. Ship tested templates or
subgraph blueprints for common tasks:

- text-only;
- one identity plus one voice;
- an object appearing only in later shots;
- source-video edit with copied music and a new voice reference;
- motion transfer;
- exact first/last-frame completion.

ComfyUI can package reusable subgraph blueprints, and App mode can expose only
selected controls for a fixed template. These are useful onboarding layers,
but arbitrary reference counts and shot chains should remain editable as normal
nodes.

## Plan schema and invariants

The plan should use stable internal IDs and human aliases. H3 labels are derived
only during compilation so inserting a new reference cannot silently retarget a
stored numeric label.

```text
project
  initial_prompt, duration, style, soundscape, music
assets[]
  asset_id, media_kind, media_value, source_name, primary_relationship
bindings[]
  binding_id, asset_id, subject_alias, exact_role, retention, shot_scope
audio_relationships[]
  asset_id, exact_use, target_alias/layer/event, transcript, language, retention
shots[]
  shot_id, cut_at, description, transition, referenced_binding_ids
dialogue_events[]
  shot_id, sequence, speaker_alias, language, exact_text, delivery, voice_mode
```

Required invariants:

1. Project is first and Prompt Merge is last.
2. All media and bindings precede the first Shot.
3. A Dialogue Event follows the Shot it belongs to.
4. Subject and speaker references use internal IDs/aliases, never stored
   `<Subject N>` numbers.
5. Native labels are assigned in canonical category order at compile time.
6. Speaker IDs are assigned from actual vocal-event order at compile time.
7. A numeric `shot_scope`, such as `3,4` or `3-4`, never asks the user to
   rewrite "Shot 3" in prose. The Shot editor's reference picker inserts the
   corresponding label at the intended entity/action; the merger validates
   that scoped references are actually used.
8. The compiler never creates a Subject merely because Ref2VA is selected.
9. The enhancer cannot mutate any of these invariants.

## Concrete interaction details

- Each reference node displays a live badge such as `Picture 2 - truck` or
  `Audio 1 - woman's voice -> woman`. Backend output is authoritative; the
  badge is a preview.
- Selecting an audio role changes the visible fields. `Voice` shows speaker and
  delivery; `Music style` shows destination/layer; `Partial copy` shows range
  and layer. No field or tooltip contains an ambiguous list joined by "or".
- Subject selectors and `<` completion use upstream aliases, so users choose
  `Woman - Subject 1` rather than remembering a number.
- Unknown or out-of-scope references are errors before sampling, with a direct
  fix rather than a large diagnostic dump.
- The default report is a short readiness summary. Full generated definitions,
  route order, and retention analysis remain available on advanced outputs.
- Bypass behavior is explicit: a bypassed reference disappears from the catalog
  and labels are recomputed; stale labels in Shot prose are then reported.
- The legacy nodes remain loadable but are clearly labeled `Legacy`; there is
  no automatic positional widget migration into semantically different v2
  nodes.

## Why standard ComfyUI nodes plus a small frontend extension are enough

Use the V3 node schema for all new nodes. `DynamicCombo` is suitable for the
role-specific Image, Video, and Audio fields. Standard repeatable inputs are
not suitable for a compound Dialogue row because an Autogrow template cannot
itself be a dynamic compound input in the current API. Separate chainable
Dialogue Event nodes therefore give better serialization and headless behavior
than a custom repeated form.

Custom JavaScript is limited to conveniences that cannot be expressed by the
backend schema:

- upstream alias discovery and `<` completion;
- live label/route badges;
- first-Shot cut-time hiding and computed range preview;
- recursive traversal through reroutes, bypassed nodes, and subgraphs.

Every frontend-derived result is independently recomputed and validated by the
Python compiler.

## Phased implementation plan

### Phase 1 - deterministic compiler MVP

1. Define `MINIMAX_H3_PLAN_V2` as plain versioned data plus pure builder,
   compiler, and validator functions.
2. Add Project Setup, exact Image/Video/Audio Reference nodes, Shot, Dialogue
   Event, and Prompt Merge using ComfyUI's V3 schema.
3. Implement canonical media numbering, speaker ordering, mode selection, six-
   section generation, strict validation, and concise previews.
4. Cover the official task families and limits with golden tests before adding
   model inference.

This phase is already useful: it produces a valid prompt and route table with
no LLM and leaves all current nodes operational.

### Phase 2 - interaction and enhancement

1. Add conditional role widgets, upstream subject selectors, live badges,
   numeric shot-scope parsing, and `<` autocomplete.
2. Adapt Prompt Enhancer to consume the locked plan and return structured
   editable prose; reconstruct and validate the final prompt afterward.
3. Keep the side-node tail input and expose both base and effective system
   prompts.
4. Add image/video analysis selectively; describe audio only from explicit
   metadata and transcripts.

Implemented on `dev/reference-workflow-ux-phase2`. Qwen receives the complete
compiled scene and reference inventory in one request, optionally accompanied
by visual evidence. Its response is restricted to a versioned prose JSON
surface; Python recompiles the H3 document and compares all semantic locks
before accepting it. Invalid responses fall back to the deterministic draft.

### Phase 3 - end-to-end integration and onboarding

1. Add the optional H3 Apply Reference Plan adapter and compatibility tests
   against the current native H3 node.
2. Ship templates/subgraphs for the acceptance cases and an App-mode preset for
   the simplest workflows.
3. Mark the existing monolithic/context nodes as legacy in the node library and
   document a manual migration map.
4. Consider the Composer UI only after the plan schema and adapter have proven
   stable in real workflows.

Implemented on `dev/reference-workflow-ux-phase3`. Apply Reference Plan verifies
that the final prompt and compiled context are the same package, translates the
compiler's canonical routes into the installed native H3 call, and delegates to
ComfyUI's own conditioning implementation. The compatibility layer checks both
native signatures without loading weights. Workflow templates, an APP-mode
prompt preset, direct saved-audio selection, Legacy library labels, and a manual
migration map complete the onboarding layer. A Composer UI remains deliberately
deferred until this graph workflow has real-world use.

## Acceptance cases

The implementation is ready only when golden tests cover at least:

- text-only generation;
- exact first frame, exact last frame, and first-plus-last endpoints;
- one image used only for a person identity, with no standalone Picture row;
- one item reference appearing only in Shots 3 and 4;
- one image defining multiple visible subjects without duplicate native media;
- several images jointly defining one Subject;
- a voice clip bound to a referenced woman, with no generic "or" wording;
- mixed partially copied music and separate voice timbre, matching MiniMax's
  published Ref2VA example;
- music-style, beat, sound-effect, continuity, transcript, full-copy, and
  partial-copy audio paths;
- video editing, continuation, motion transfer, and temporal-structure use;
- Shot autocomplete through normal links, reroutes, and bypassed setup nodes;
- API-mode backend validation without relying on the browser extension;
- every official media-count, duration, and ordering constraint.

## Research sources

- [MiniMax H3 official model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)
  and its explanation of H3 Context-IR and media limits.
- [MiniMax H3 full-reference prompt writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md).
- [MiniMax H3 base prompt writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md).
- [ComfyUI's native MiniMax H3 implementation](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py)
  and its fixed media-presentation order.
- [Official ComfyUI MiniMax H3 Ref2VA template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json).
- [ComfyUI V3 node schema](https://docs.comfy.org/custom-nodes/v3_migration),
  [JavaScript extensions](https://docs.comfy.org/custom-nodes/js/javascript_overview),
  [subgraphs](https://docs.comfy.org/interface/features/subgraph), and
  [subgraph blueprints](https://docs.comfy.org/custom-nodes/subgraph_blueprints).
- [Qwen3-VL-32B-Instruct official model card](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
  for image/video comprehension scope.
- [Nielsen Norman Group's usability heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/),
  especially visibility of status, match with user language, error prevention,
  recognition rather than recall, and minimalist design.
