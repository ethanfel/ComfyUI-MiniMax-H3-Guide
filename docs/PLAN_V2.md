# Plan v2 workflow and node reference

Plan v2 represents an H3 project as one ordered `MINIMAX_H3_PLAN_V2` value.
Each node adds explicit semantic information; Prompt Merge assigns final labels
and compiles the native route.

## Recommended graph

```text
Project Setup
  → optional Foley Target
  → global Image / Video / Audio References
  → optional Subject Binding or Character Replacement
  → Shot 1
  → optional Attach Keyframe / Attach Motion / Dialogue Event
  → Shot 2 → its attachments
  → Prompt Merge
  → optional Structured Prompt Enhancer
  → optional Apply Structured Prose
  → optional Prompt Review Gate or Inline Prompt Override
  → Apply Reference Plan
  → sampler
```

Global references must be registered before the first Shot. Shot attachments
consume the stable handle returned by their Shot, so media never needs a
manually typed numeric scope.

## Project Setup

Project Setup defines target width, height, frame count, creative intent,
style, soundscape, and music. The frame selector advances through H3's native
`17k+5` grid at 24 FPS. The node displays the exact playback duration as
`frame_count / 24`, including 362 frames / 15.083 seconds.

## Reference nodes

### Image Reference

Image Reference distinguishes:

- reusable identity, wardrobe, object, scene, or style evidence;
- exact first and last endpoint frames;
- concrete keyframes;
- storyboard/planning frames.

Reusable visible content creates a Subject. Endpoint frames stay in the native
I2VA/L2VA/FL2VA route. A keyframe becomes a Picture attached to a specific
Shot.

### Video Reference

Video Reference distinguishes:

- a source video to edit;
- a source video to continue;
- reusable visible content;
- motion or action transfer;
- camera, cut, rhythm, or temporal structure.

Motion clips keep a native Video route while their reusable action is compiled
as a separate Subject. This lets one Subject receive another clip's pose or
action without asking H3 to reproduce the whole reference video.

### Audio Reference

Audio roles include voice, music style, beat, sound-effect texture, exact
dialogue/lyrics, continuity, complete copying, partial copying, and broad
inspiration.

- Voice requires a target Subject.
- Exact content requires a language and transcript.
- A paired-video handle marks audio as that Video Reference's soundtrack.
- Complete-copy and continuity audio must match the paired video interval
  within one 24 FPS frame.

The compiler places a tagged nonverbal audio reference at the exact sentence
location authored in Shot prose. Untagged texture/style references retain a
global soundscape fallback.

## Subject Binding

Subject aliases are human names such as `woman`, `watch`, or `truck`. Final
`<Subject N>` numbers are compiler-owned.

Use Subject Binding to:

- combine multiple references as evidence for one Subject; or
- transfer an attribute, pose, expression, or action to an existing Subject.

## Character Replacement

Character Replacement maps one precisely described performer in an edit-source
Video to one upstream identity Subject.

```text
Image Reference: identity
  → Video Reference: source video to edit
  → Character Replacement
       source character: the woman in the red jacket
       replacement: upstream identity Subject
  → Shots
```

The appearance policy controls identity, body, and wardrobe independently.
Separate preservation switches retain source performance and surroundings.

## Shots and attachments

Each Shot declares one cut time. The next cut or Project duration determines
its end. The chain rejects gaps, overlaps, reversed ranges, and a first Shot
that does not begin at `0.000`.

### Attach Keyframe

Registers one image as the Shot's opening, internal, or ending composition
anchor. The image receives a native Picture route and is cited only in that
Shot.

### Attach Motion

Transfers a Video's motion/action Subject to an upstream target Subject only
for the selected Shot. The compiler writes the relationship and assigns the
native labels.

### Dialogue Event

Dialogue attaches to the most recently opened Shot. Optional
`start_offset_seconds` uses:

- `-1`: automatic placement;
- `0`: Shot opening;
- positive value: seconds after the Shot opening.

Continuity can mark a line complete, split across the next cut, carried from
the previous cut, or interrupted by the video ending. Cross-cut lines receive
the required `<scenetrans>` markers; final interruptions receive `<cutoff>`.
Dialogue words and punctuation remain exact.

## Prompt Merge

Prompt Merge assigns Subject/Picture/Video/Audio numbers, validates scopes and
timing, canonicalizes native media order, and returns:

| Output | Purpose |
| --- | --- |
| `h3_prompt` | Deterministic H3 prompt |
| `rewrite_request` | Locked prose-enhancement contract |
| `plan_context` | Typed compiled plan |
| `problems_report` | Readiness, timing, inventory, routes, and warnings |
| `h3_length` | Native output frame count |

## Prompt refinement

The structured enhancer may expand editable prose, but Python restores and
checks all compiler-owned structure. Apply Structured Prose recompiles edited
JSON without rerunning Qwen. Prompt Review Gate provides an interactive final
approval. Inline Prompt Override provides a non-pausing A/B path.

See [Qwen enhancement and prompt review](QWEN_ENHANCER.md).

## Apply Reference Plan

Connect a prompt and `plan_context` from the same Prompt Merge, enhancer,
structured-prose, review, or override result. Apply Reference Plan verifies the
pair, routes stored media, and delegates conditioning to ComfyUI's installed
`MiniMaxH3ImageToVideo` or `MiniMaxH3ReferenceToVideo` implementation.

The node returns native positive conditioning and the joint audio/video latent.
Reference audio requires the audio VAE. Text-only, endpoint, and reference-free
Foley plans do not.

## Route selection

| Goal | Mode | Checkpoint family |
| --- | --- | --- |
| Text generation | T2VA | H3-Base-FL2VA |
| Exact first frame | I2VA | H3-Base-FL2VA |
| Exact final frame | L2VA | H3-Base-FL2VA |
| Exact first and final frames | FL2VA | H3-Base-FL2VA |
| Appearance, motion, editing, continuation, or audio references | Ref2VA | H3-Base-Ref2VA |

## Limits and native ordering

- Up to 9 images, 3 videos, 3 audio clips, and 12 mixed files.
- Each reference video/audio clip must be 2–15 seconds.
- Video references total at most 15 seconds.
- Audio references total at most 15 seconds.
- Reference audio cannot be the only Ref2VA media input.
- One complete-copy/continuity soundtrack paired to a matching 362-frame video
  may share the native 15.083-second boundary.

Native ordering is Pictures first, then each paired soundtrack immediately
before its Video, then standalone Audio. Picture, Video, and Audio labels are
numbered independently.
