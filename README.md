# ComfyUI MiniMax H3 Guide

A dependency-free ComfyUI node pack for building MiniMax H3 prompts from
explicit references, subjects, and shots. It compiles those choices into valid
H3 prose and routes the media to ComfyUI's native H3 nodes.

- Typed Plan v2 workflow instead of one large prompt form
- Explicit identity, keyframe, motion, edit-source, voice, and soundtrack roles
- Deterministic labels, timing, validation, and native media routing
- Optional Qwen enhancement with compiler-owned structure kept locked
- Reusable image/audio Reference Sheets and a locked-picture Foley mode
- H3-native AV split/repack with frame-locked audio length

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ethanfel/ComfyUI-MiniMax-H3-Guide
```

Restart ComfyUI, then look under **MiniMax H3 → Plan v2**. The pack has no
additional Python dependencies.

This release requires a ComfyUI build with native MiniMax H3 support. Update
ComfyUI if **MiniMax H3 Image to Video**, **MiniMax H3 Reference to Video**, or
the MiniMax tokenizer is missing.

## Start here

```text
Project Setup
  → optional Image / Video / Audio References
  → one Shot per real cut
  → Prompt Merge
  → Apply Reference Plan
  → sampler
```

1. **Project Setup** selects the output size and native frame count.
2. **Reference nodes** declare what each file means. The compiler assigns H3
   labels and routes automatically.
3. **Shot nodes** define the timeline. Attach keyframes, motion, and dialogue to
   the Shot that uses them.
4. **Prompt Merge** validates and compiles the plan.
5. **Apply Reference Plan** hands the matching prompt, media, and AV latent to
   ComfyUI's native H3 implementation.

For the complete node-by-node workflow, see the
[Plan v2 guide](docs/PLAN_V2.md).

## Pick roles by intent

| Intent | Use |
| --- | --- |
| Exact opening or ending composition | Image Reference as an endpoint |
| Preserve identity, wardrobe, object, scene, or style | Image Reference as reusable content |
| Transfer pose, action, or body mechanics | Video Reference + Attach Motion |
| Modify an existing video | Video Reference as an edit source |
| Continue an existing video | Video Reference as a continuation source |
| Keep or copy source audio | Paired Audio Reference with continuity/copy role |
| Borrow voice, beat, music style, or sound texture | Audio Reference with the matching guidance role |

The same file type can take different routes. Its intended job—not its
extension—decides the H3 mode.

## Example workflows

Examples are available in [`example_workflows/`](example_workflows/) and
ComfyUI's workflow template browser.

| Workflow | Status | Purpose |
| --- | --- | --- |
| Animate Keyframe with Motion Reference | Ready | Retarget motion onto a supplied opening keyframe |
| Prompt Builder App | WIP | Prompt-only Plan v2 interface |
| Identity and Voice | WIP | Identity, voice, dialogue, and Ref2VA generation |
| First and Last Frames | WIP | Exact endpoint composition |
| Five Shot Keyframe Composition | WIP | Per-shot keyframes and motion transfer |
| Character Replacement | WIP | Replace one performer in an edit-source video |
| Video Extension with Audio Continuity | WIP | Character transfer plus continuation |
| Video to Audio Foley | WIP | Locked picture track with generated audio |
| Foley with Sound Reference | WIP | Foley guided by a sound texture |
| Foley with Multiple Sound and Voice References | WIP | Multi-reference Foley and dialogue |

The ready motion-reference workflow is the reference implementation. WIP
templates expose their intended graph but still need task-specific media and
default refinement.

## Limits worth knowing

| Constraint | H3 limit |
| --- | --- |
| Output | 4–15 seconds on the native `17k+5` frame grid at 24 FPS |
| Images | 9 |
| Videos | 3 |
| Audio clips | 3 |
| Mixed reference files | 12 total |
| Each reference video or audio clip | 2–15 seconds |
| Cumulative video references | 15 seconds |
| Cumulative audio references | 15 seconds |

The 362-frame endpoint plays for 15.083 seconds. One complete-copy or
continuity soundtrack paired with a matching 362-frame video may share that
padded boundary; it does not increase the general audio limit.

## Guides

- [Plan v2 workflow and node reference](docs/PLAN_V2.md)
- [Video-to-audio Foley](docs/FOLEY.md)
- [Qwen enhancement and prompt review](docs/QWEN_ENHANCER.md)
- [Persistent Reference Sheets](docs/REFERENCE_SHEETS.md)
- [H3 AV latent split and frame-locked repack](docs/AV_LATENTS.md)
- [Preparing reference media](REFERENCE_MEDIA_PRACTICAL_GUIDE.md)
- [Migrating legacy workflows](MIGRATION_TO_PLAN_V2.md)
- [Audit and compatibility decisions](AUDIT_REPORT.md)

Legacy nodes remain registered so existing workflows load. New projects should
use Plan v2.

## Development

```bash
pytest -q
```

## Credits

The interaction direction was inspired by
[ComfyUI-MiniMaxH3-Easy](https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy),
used under its MIT license. This project retains an independent Plan v2 data
model, validator, router, and compiler.

The H3 AV timing nodes were cross-checked against ComfyUI's native H3 grid
formulas and the independent findings in
[ComfyUI-MMH3Tools](https://github.com/ckinpdx/ComfyUI-MMH3Tools).

Prompt structure follows MiniMax's official
[base guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md),
[reference guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md),
and [model card](https://huggingface.co/MiniMaxAI/MiniMax-H3).

## License

GNU General Public License version 3 only
([GPL-3.0-only](LICENSE)).
