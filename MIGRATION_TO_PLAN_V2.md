# Migrating existing workflows to Plan v2

Existing node IDs remain registered, so old workflows still load. Their canvas
names now include **Legacy** because the old Guide/context path cannot express
per-reference roles as precisely as Plan v2. Migration is manual: positional
widgets from the large Guide do not have a safe one-to-one semantic conversion.

## Recommended replacement graph

```text
Project Setup
  -> Image / Video / Audio Reference nodes
  -> optional Subject Binding nodes
  -> Shot nodes (+ optional Dialogue Event nodes)
  -> Prompt Merge
  -> optional Structured Prompt Enhancer
  -> optional Prompt Review Gate
  -> Apply Reference Plan
  -> sampler
```

Connect `h3_prompt` and `plan_context` from the same Merge, Structured Prompt
Enhancer, Apply Structured Prose, or Prompt Review Gate node to Apply Reference
Plan. The adapter rejects mismatched pairs instead of silently using stale
labels or routes. Prompt Review Gate pauses only for prompt text; reference
media remains allocated in `plan_context` and needs no chooser.

## Node-by-node map

| Existing node or field | Plan v2 replacement |
| --- | --- |
| Prompt Guide target description | Project Setup `initial_prompt` |
| Prompt Guide duration, style, ambience, music | Project Setup fields |
| Target Timing | Project Setup `h3_length`; Prompt Merge and Apply Reference Plan carry the same authoritative length |
| Legacy Shot start/end text | One Shot node per segment; `cut_at` is the start and the next cut/global duration computes the end |
| Prompt Guide dialogue field | Dialogue Event after the Shot it belongs to |
| Visual Reference Role + Enhancer Visual Reference | One exact Image or Video Reference; add Subject Binding only when the same physical media defines another reusable Subject/role |
| Generic audio context | One exact Audio Reference per clip, including its speaker, layer/event, transcript, copied range, and optional paired-video handle |
| Legacy free-form Prompt Enhancer | Structured Prompt Enhancer (Plan v2), which returns prose JSON and a matching recompiled plan |
| Manual final-prompt text gate | Prompt Review Gate (Plan v2), which validates compiler locks and forwards the same media-bearing plan |
| Manual native reference sockets | Apply Reference Plan, or keep the official native node and follow Prompt Merge's route report |
| Generation Tail Loader | Keep it; both enhancer generations continue to support this side node |

## Reference Sheet migration

Reference Sheet itself is not legacy. It remains the reusable gallery and now
returns both `selected_image` and `selected_audio`:

```text
Reference Sheet.selected_image -> Image Reference.image
Reference Sheet.selected_audio -> Audio Reference.audio
```

Declare the workflow-specific role on the Plan v2 reference node. The old
Reference Sheet Visual/Audio Reference context nodes remain loadable only for
existing Guide workflows and are marked Legacy.

Duplicate Reference Sheet when a generation needs several independently
selected saved assets. Their order in the Plan v2 setup chain determines their
provisional inventory; Prompt Merge then assigns canonical native labels and
routes.

## Practical migration sequence

1. Leave the existing workflow intact and build the Plan v2 spine beside it.
2. Copy only the global creative intent, duration, style, ambience, and music
   into Project Setup.
3. Connect each real media output to its own exact reference node. Do not copy
   a generic media-type role.
4. Replace each old timing row with a Shot start. Add Dialogue Events for exact
   words, voice references, or multiple speakers.
5. Queue Prompt Merge and resolve every reported scope, label, timing, or media
   limit error.
6. Optionally enhance the compiled scene. For structured reusable edits, edit
   `editable_prose` and run Apply Structured Prose. For a final full-prompt
   check, connect both outputs through Prompt Review Gate.
7. Connect the matching prompt/context pair (or both approved gate outputs) to Apply Reference Plan. Load the
   checkpoint family named in `adapter_report`.
8. After a successful generation, remove or bypass the old branch.

There is intentionally no automatic conversion of broad options such as
"voice, music, beat, or sound." Choosing the exact new relationship is the
part that makes the migration correct.
