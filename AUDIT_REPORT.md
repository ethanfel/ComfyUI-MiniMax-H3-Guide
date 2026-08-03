# MiniMax H3 workflow audit

Audit date: 2026-08-03
Branch: `fix/deep-h3-audit`

## Sources and scope

This audit reviews the public nodes, their current context payload contracts,
and their documented workflow against:

- MiniMax's official [base prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
- MiniMax's official [full-reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
- The official [MiniMax H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- ComfyUI `14b05228` (2026-08-02), especially `nodes_minimax_h3.py`,
  `text_encoders/minimax.py`, and `text_encoders/qwen3vl.py`. Native MiniMax H3
  support first appeared in ComfyUI commit `57500fc5bc92`.

The review covers prompt families, task prefixes, semantic labels, retention
markers, shots and timestamps, native frame alignment, reference-media order,
Qwen visual presentation, the optional generation tail, node contracts,
tooltips, compatibility behavior, and regression coverage.

## Native ComfyUI contract used by this pack

- Target video is 24 FPS. Native `length` is rounded upward to a frame count
  satisfying `frames % 17 == 5`.
- Base image-to-video accepts a first frame, a last frame, or both. The first
  frame is stretched to the target canvas; the last frame is center-cover
  cropped. Composition is exact only when source and target aspect ratios agree.
- Full-reference presentation order is all pictures first, then each video with
  its enabled soundtrack audio immediately before it, followed by standalone
  audio. Prompt labels are one-based; native sockets are zero-based.
- Reference videos are interpreted as 24 FPS, truncated to target length,
  rounded downward to the same `17k+5` grid, and shown to MiniMax's Qwen
  tokenizer at fixed 2 FPS.
- MiniMax's tokenizer prepends raw reference blocks rather than applying a chat
  template. Every two video samples form one temporal patch; audio is not visual
  input to Qwen.
- The bundled H3 Qwen3-VL-32B CLIP contains 50 language layers and no final norm
  or LM head. It is a conditioning encoder, not a complete text generator. The
  optional side tail supplies layers 50-63, final norm, and LM head temporarily.

## Findings and resolutions

| Priority | Finding | Resolution |
| --- | --- | --- |
| Critical | Unassigned inventory files silently became appearance/content references and could create unsupported Subjects. | Bare Guide inventory remains descriptive. New Visual Reference nodes default to `Unassigned - choose a reference role` and require either an explicit compatibility role or a role-binding chain before execution. Once selected, that role intentionally supplies semantic evidence. |
| Critical | A single asset role could not express multiple official relationships, and multiple assets could not jointly define one Subject. | Visual roles are repeatable bindings with a content group, retention relationship, optional shot scope, and notes. A shared group tells Qwen which assets may jointly define a Subject; the final Subject rows remain model output. The chain assigns labels and recommended sockets, while user wiring controls native file order. |
| Critical | Storyboards and concrete keyframes shared one ambiguous role. | They are separate roles. Concrete frames produce keyframe-completion semantics; storyboards produce reference-generation semantics. The old value remains readable for saved 0.6.x workflows. |
| Critical | Endpoint images were forced through Ref2VA when chained for enhancement, preventing two-frame FL2VA analysis. | First-frame and last-frame bindings form a base endpoint context, preserve I2VA, L2VA, or FL2VA analysis, and report `first_frame` / `last_frame` as the intended native sockets. The user still makes those links. Base endpoint and Ref2VA roles cannot share one context. |
| Critical | A hand-written multi-shot plan was embedded as prose inside Shot 1, while final-frame syntax could cite a Shot that did not exist. | The legacy text plan is parsed into real sequential shot markers and validated. A connected Shot chain remains authoritative. Final-image alignment always cites an emitted final Shot. |
| Critical | Prompt duration could disagree with native H3 output. | The Guide exposes `h3_length`, rounds requested time to native `17k+5` frames, uses effective time in the prompt, and extends the final described shot through the effective end. |
| Critical | Enhancer video samples did not match MiniMax's temporal-pair tokenizer and could describe footage native H3 discarded. | Reference video is normalized to 24 FPS and native effective length. Generic Qwen keeps configurable sparse samples; MiniMax receives a distinct fixed-2-FPS sample sequence from the effective clip. Visual evidence excludes a discarded tail, although free-form draft text can still mention it and requires review. |
| Critical | Empty or punctuation-only model output could be reported as successful after falling back to the manual prompt. | Collapse checks now inspect raw decoded output, including empty text, punctuation, and incomplete thinking blocks. Fallback is explicit in `enhancer_report`. |
| High | Old workflows serialized obsolete built-in system prompts indefinitely. | Exact known historical defaults are migrated to the current built-in prompt by hash; genuinely customized prompts are preserved verbatim. |
| High | Global fidelity could assign `attribute_transfer` to endpoints, editing sources, or structural references. | Retention is validated per role. Exact endpoints require `fully_preserved`; transfer is limited to reusable visible content with an identifiable grouping/target. |
| High | Retention locations and task types were inferred from file type or hard-coded to Shot 1. | Explicit bindings and optional shot scopes are supplied to Qwen, and unknown locations remain unspecified rather than being fabricated as Shot 1. Final Subject grouping and task-type selection are still model output; validation can diagnose structure but cannot prove semantic agreement with every binding. |
| High | Labels could be zero, gapped, duplicated, or inconsistent with native socket order. | Context labels are canonical. Guide inventory rejects zero, duplicates, and active gaps, and reports order mismatches. A selected role may intentionally synthesize a required placeholder label, including a partially missing endpoint; `mode_report` identifies every such placeholder as an expected native connection, not an attached file. |
| High | The enhancer prompt omitted several official formatting rules and accepted malformed output without diagnostics. | The built-in instructions cover section order, style placement, first-use labels, one retention row per tracked item, endpoint alignment, speaker IDs, voiceover, `<scenetrans>`, `<cutoff>`, and audio continuity. Post-generation validation diagnoses missing or empty sections, section order, basic shot markers, style opening, task-type and retention-marker vocabulary, definition/retention row cardinality, supplied visual labels, full-copy audio-section guards, and endpoint-instruction presence. Other semantic rules remain review requirements. |
| High | `Camera direction:` was emitted as a metadata label contrary to the base guide. | Camera movement is rendered as natural English inside each shot. |
| High | Full-copy audio could conflict with newly invented ambience/music. | Guide-generated audio sections state that a complete copy permits no additional audible layer and cite the copied track; the Guide warns when its dialogue/text field could imply a new signal. Enhancer instructions prohibit additions, and structural checks require applicable audio sections to cite the copied label and state exclusivity. Free-form target descriptions and custom system prompts still require review. Audio routing/analysis remains outside the visual chain. |
| Medium | UI route labels failed through reroutes/bypassed reference nodes. | Frontend traversal follows reroutes and skips bypassed chain nodes. Backend `routing_report` is authoritative for the intended mapping, but the user must make and verify every native connection. |
| Medium | The tail's chunked LM head assumed row-indexable INT8 scales and could mishandle tensor-wise scalar scales or other quantized layouts. | Scalar and per-row INT8 scales are handled explicitly. Unsupported storage/layout/ConvRot combinations fail before logits are calculated. |
| Medium | `llm_prompt` was described as the complete model input although visual tensors are injected separately. | Its tooltip now calls it the serialized text portion, explains MiniMax's prepended Picture/Video blocks for endpoint and Ref2VA media, and warns that an external multimodal LLM needs the pixels in its own visual-token format. |

## Input limits and routing decisions

- Target duration remains 4-15 seconds.
- H3 Ref2VA policy remains limited to 9 pictures, 3 videos, 3 audio items, and
  12 files total. Each video/audio item is 2-15 seconds and each media type
  totals at most 15 seconds. Native ComfyUI's executor is less strict.
- The visual chain validates picture/video counts and source-video durations.
  Because audio is separately wired, audio count/duration, the mixed 12-file
  total, and actual audio-only routing cannot be verified from that chain.
- H3 policy forbids audio as the sole Ref2VA input. The Guide can diagnose this
  from its declared inventory, but cannot inspect native audio connections.
- Picture and video numbering is independent and follows chain order within
  each media category. The report recommends native sockets; actual wiring is
  user-controlled.
- The native H3 node remains responsible for AV/keyframe/reference latent
  creation. This pack deliberately returns prompt text rather than incomplete
  `CONDITIONING`.
- Audio is deferred: the visual-reference chain neither decodes nor analyzes it.
  Native soundtrack/standalone-audio numbering is documented so a later audio
  node can preserve the correct order.

## Backward compatibility

- Serialized 0.6.x single-role visual-reference nodes remain accepted and are
  adapted into one role binding.
- Context version 1 remains readable. Newly produced contexts use a validated
  version 2 payload.
- The historical `Storyboard or keyframe` value remains accepted but is treated
  conservatively and reported as legacy/ambiguous.
- Existing custom enhancer system prompts are never rewritten; only exact known
  shipped defaults migrate.
- The new unassigned role is only the default for newly created nodes. Saved
  workflows with an explicit serialized `Identity or appearance` role keep that
  role.

## Remaining limitations

- Audio inspection and audio-role chaining are not implemented in this release.
- `enhanced_prompt` is a cleaned model candidate, not a guaranteed-valid prompt.
  Structural checks do not prove semantic first-use placement, speaker ordering,
  voice-over wording, dialogue continuity markers, audio continuity, reference
  meaning, or every timestamp against the declared duration.
- Role groups and task evidence constrain the enhancer instructions, but final
  Subject numbering/grouping and summary task types remain probabilistic Qwen
  output.
- Full-copy audio has cross-section citation/exclusivity checks, but this is not
  semantic proof: a free-form target description or custom instruction can still
  conflict and requires review.
- Selected Guide roles may create placeholder media labels. They keep a draft
  structurally complete but do not represent attached files; every placeholder
  must be matched to a real native connection.
- The optional 32B generation-tail path has deterministic unit coverage for its
  split and INT8 head assumptions, but a real end-to-end smoke test still
  requires the separately downloaded multi-gigabyte tail artifact and suitable
  hardware.
- Visual-reference output labels in the canvas are convenience hints. When a
  role-binding chain makes the route ambiguous to the browser, use the backend
  `routing_report` as the source of truth for intended sockets, then verify the
  native links manually.
- Native first-frame stretching and last-frame center cropping cannot preserve
  an unmatched source aspect ratio. Match endpoint images to the target canvas
  when exact composition matters.

## Validation

The integrated fix branch passed:

| Check | Command/result |
| --- | --- |
| Complete Python suite | `pytest -q -ra` — 108 passed |
| Lint | `ruff check .` — passed |
| Byte-code compilation | `python -m py_compile __init__.py nodes.py enhancer.py media_context.py hybrid_tail.py tests/test_nodes.py tests/test_enhancer.py tests/test_media_context.py tests/test_hybrid_tail.py` — passed |
| Frontend syntax | `node --check web/visual_reference_routes.js` — passed |
| Patch hygiene | `git diff --check` — passed |
| Dynamic mapping/schema and representative workflow smoke | 6/6 public nodes imported; callable schemas matched; reversed FL2VA endpoints normalized; Ref2VA transfer/routes validated; 48 source frames became 39 native frames and four fixed-2-FPS samples; legacy manual cuts normalized — passed |
