"""Qwen3-VL prompt enhancement using ComfyUI's loaded CLIP interface."""

from __future__ import annotations

from collections import Counter
import re


NO_TAIL = "[none — connected CLIP is already complete]"
TAIL_TYPE = "MINIMAX_H3_GENERATION_TAIL"


def _tail_choices() -> list[str]:
    """List only compatible generation-tail files when ComfyUI is available."""

    try:
        import folder_paths
    except ImportError:
        return [NO_TAIL]
    tails = [
        name
        for name in folder_paths.get_filename_list("text_encoders")
        if "generation_tail_50_63" in name.casefold()
    ]
    return [NO_TAIL, *sorted(tails, key=str.casefold)]


def _generate_with_tail(clip, tail_name: str, tokens, generation_options: dict):
    """Lazy import keeps this node pack importable outside ComfyUI for tests."""

    try:
        from .hybrid_tail import generate_with_tail
    except ImportError:
        from hybrid_tail import generate_with_tail
    return generate_with_tail(clip, tail_name, tokens, generation_options)


DEFAULT_SYSTEM_PROMPT = """You are an expert prompt engineer for MiniMax H3 audiovisual generation. Rewrite the supplied manual H3 draft into a production-ready prompt. Do not discuss the task, explain your choices, or add a preface. Return only the finished H3 prompt.

Follow these rules:
- Preserve the user's requested subjects, identities, actions, dialogue, lyrics, visible text, reference roles, endpoint frames, timing, and audio intent. Never replace or contradict them.
- Write in English except for dialogue and lyrics inside <d>[Language] ...</d> and text visibly present in the scene. Preserve their original wording and language.
- Treat an attached image as optional visual evidence. Use its visible appearance, composition, and style only where the manual draft assigns that role. Do not silently turn it into a first or last frame.
- Make the video chronological and physically observable. For every shot, establish composition, subject appearance and position, environment and lighting, action and state changes, camera behavior, and synchronized sound.
- [Shot 1] has no timestamp. Later shots begin with [Shot N] At MM:SS.mmm and strictly increasing cut times inside the stated duration.
- Describe camera movement naturally. Include movement type and meaningful speed or amplitude, but do not invent camera movement that conflicts with a static-camera request.
- Give every actual vocal source a stable (S1), (S2), etc. Put only spoken or sung words inside <d>[Language] ...</d>. Put visible text in English double quotation marks.
- Keep ambience, physical action sounds, and non-verbal human sounds in overall_soundscape. Put only audience-only score in non_diegetic_music. Use N/A when there is no such score.

Choose the output structure from the supplied draft:

For T2VA, I2VA, FL2VA, or L2VA, preserve the applicable image-alignment instruction at the top, followed by one blank line when present. Then output exactly these fields:
integrated_multimodal_description
overall_soundscape
non_diegetic_music

For Ref2VA, output exactly these six sections in this order:
subject_definitions
summary
retention_analysis
detailed_description
overall_soundscape
non_diegetic_music

In Ref2VA, keep <Subject N>, <Picture N>, <Video N>, and <Audio N> labels stable. Do not invent media assets. Use only the fixed summary task types keyframe completion, reference generation, video editing, video continuation, audio reuse, and audio reference. Use only fully_preserved, partially_preserved, attribute_transfer, or weak_reference for visible retention, and fully_copy, partially_copy, reference, or weak_reference for audio retention. For generation tasks, normally make detailed_description 350-500 English words unless dialogue timing requires otherwise.

If the manual draft is already detailed and correctly formatted, make only useful corrections. Never wrap the result in Markdown or code fences."""


def build_llm_user_prompt(manual_prompt: str, mode_report: str = "", has_image: bool = False) -> str:
    """Build the user turn without mixing it into the editable system prompt."""

    image_note = (
        "One image is attached as optional visual context. Analyze it, but follow the role assigned "
        "by the manual draft."
        if has_image
        else "No image is attached. Work only from the text and its declared reference roles."
    )
    report = mode_report.strip() or "No separate mode report was supplied; infer the mode from the draft."
    draft = manual_prompt.strip() or "No manual prompt was supplied."
    return f"""Enhance the following MiniMax H3 prompt draft.

IMAGE CONTEXT
{image_note}

MODE REPORT
{report}

MANUAL H3 DRAFT
{draft}"""


def _is_minimax_h3_tokenizer(clip) -> bool:
    tokenizer = getattr(clip, "tokenizer", None)
    tokenizer_type = type(tokenizer)
    return tokenizer_type.__name__ == "MiniMaxH3Tokenizer" or tokenizer_type.__module__.endswith(
        ".minimax"
    )


def format_chat_prompt(system_prompt: str, user_prompt: str, has_image: bool, minimax_clip: bool) -> str:
    """Format a Qwen chat while accommodating H3's raw multimodal tokenizer."""

    image_block = ""
    if has_image and not minimax_clip:
        image_block = "<|vision_start|><|image_pad|><|vision_end|>\n"
    return (
        f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n"
        f"<|im_start|>user\n{image_block}{user_prompt.strip()}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def clean_generated_prompt(text: str, fallback: str) -> str:
    """Remove common chat/fence residue without altering the H3 body."""

    value = (text or "").strip()
    value = re.sub(r"^<think>.*?</think>\s*", "", value, flags=re.DOTALL)
    value = re.sub(r"^(?:assistant\s*:?[\r\n]+)", "", value, flags=re.IGNORECASE)
    fence = re.match(r"^```(?:text|markdown)?\s*\n?(.*?)\n?```$", value, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        value = fence.group(1).strip()
    return value or fallback.strip()


def generation_collapse_reason(text: str) -> str | None:
    """Recognize common LLM decoding collapse without rejecting short valid prompts."""

    value = (text or "").strip()
    if not value:
        return "the model returned no text"
    if len(value) < 40:
        return None

    non_space = [char for char in value if not char.isspace()]
    alphanumeric = sum(char.isalnum() for char in non_space)
    if non_space and alphanumeric / len(non_space) < 0.05:
        return "the output collapsed into punctuation"

    tokens = re.findall(r"[\w<>/.-]+|[^\w\s]", value.casefold())
    if len(tokens) >= 20:
        _, repeated_count = Counter(tokens).most_common(1)[0]
        if repeated_count / len(tokens) >= 0.75:
            return "the output repeated one token almost exclusively"
    return None


def clip_generation_issue(clip) -> str | None:
    """Explain why MiniMax's truncated conditioning encoder should not generate text."""

    if not _is_minimax_h3_tokenizer(clip):
        return None
    stage = getattr(clip, "cond_stage_model", None)
    if stage is None or not hasattr(stage, "clip"):
        return None
    inner_clip = getattr(stage, stage.clip, None)
    transformer = getattr(inner_clip, "transformer", None)
    config = getattr(getattr(transformer, "model", None), "config", None)
    layers = getattr(config, "num_hidden_layers", getattr(transformer, "num_layers", None))
    has_lm_head = getattr(config, "lm_head", None)
    has_final_norm = getattr(config, "final_norm", None)
    if layers == 50 and has_lm_head is False and has_final_norm is False:
        return (
            "Enhancement skipped: the connected MiniMax H3 CLIP is the conditioning-only "
            "Qwen3-VL-32B checkpoint truncated to 50 layers, without a final normalization or "
            "language-model head. It can condition H3 but cannot reliably generate instructions. "
            "Connect a MiniMax H3 Generation Tail Loader to clip_tail, connect a complete "
            "instruction-tuned Qwen3-VL model, or send llm_prompt to "
            "another LLM node. manual_prompt was returned unchanged."
        )
    return None


def _resolve_tail_name(clip_tail) -> str | None:
    """Resolve the lightweight loader output, accepting old string inputs too."""

    if clip_tail is None or clip_tail == NO_TAIL:
        return None
    if isinstance(clip_tail, str):
        return clip_tail
    if isinstance(clip_tail, dict):
        tail_name = clip_tail.get("tail_name")
        if isinstance(tail_name, str) and tail_name and tail_name != NO_TAIL:
            return tail_name
    raise ValueError(
        "clip_tail must come from MiniMax H3 Generation Tail Loader, or be left disconnected "
        "when the connected CLIP is already generation-capable."
    )


def _generate_with_clip(
    clip,
    tokens,
    generation_options: dict,
    use_minimax_image_path: bool,
):
    """Generate like ComfyUI's TextGenerate, preserving Qwen3-VL image inputs.

    MiniMax's H3 CLIP uses the base generation wrapper, which does not forward
    Qwen3-VL MRoPE/DeepStack image metadata. When that exact CLIP receives an
    image, use the same image-aware arguments as ComfyUI's Qwen3VLClipModel.
    """

    stage = getattr(clip, "cond_stage_model", None)
    if not use_minimax_image_path or stage is None or not hasattr(stage, "clip"):
        return clip.generate(tokens, **generation_options)

    import comfy.model_management  # Imported only inside a running ComfyUI instance.

    stage.reset_clip_options()
    clip.load_model(tokens)
    device = clip.patcher.load_device
    stage.set_clip_options({"layer": None})
    stage.set_clip_options({"execution_device": device})
    inner_clip = getattr(stage, stage.clip)

    token_batches = next(iter(tokens.values())) if isinstance(tokens, dict) else tokens
    tokens_only = [[token[0] for token in batch] for batch in token_batches]
    embeds, _, _, embeds_info = inner_clip.process_tokens(tokens_only, device)
    position_ids, visual_pos_masks, deepstack = inner_clip.transformer.build_image_inputs(
        embeds, embeds_info
    )
    with comfy.model_management.cuda_device_context(device):
        return inner_clip.transformer.generate(
            embeds,
            **generation_options,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_embeds=deepstack,
        )


class MiniMaxH3GenerationTailLoader:
    """Select a compatible H3 generation tail without loading it persistently."""

    CATEGORY = "MiniMax H3/Prompting"
    FUNCTION = "select_tail"
    RETURN_TYPES = (TAIL_TYPE,)
    RETURN_NAMES = ("clip_tail",)
    OUTPUT_TOOLTIPS = (
        "Connect this to Prompt Enhancer.clip_tail. This lightweight descriptor lets the enhancer "
        "temporarily load layers 50-63, final norm, and LM head only while it writes the prompt.",
    )
    DESCRIPTION = (
        "Side loader for MiniMax H3's optional Qwen3-VL generation tail. Use it only with the "
        "bundled 50-layer conditioning CLIP; a complete generative Qwen3-VL CLIP needs no tail."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tail_file": (
                    _tail_choices(),
                    {
                        "tooltip": (
                            "Select the layers 50-63 generation-tail safetensors stored under "
                            "ComfyUI/models/text_encoders. If only the 'none' entry is shown, install "
                            "the compatible MiniMax H3 tail and refresh ComfyUI. The weights are loaded "
                            "temporarily by the enhancer, then unloaded; this node itself uses no VRAM."
                        )
                    },
                )
            }
        }

    def select_tail(self, tail_file: str):
        if tail_file == NO_TAIL:
            raise RuntimeError(
                "No compatible MiniMax H3 generation tail was selected. Install a file whose name "
                "contains 'generation_tail_50_63' under models/text_encoders, then refresh ComfyUI."
            )
        return ({"tail_name": tail_file},)


class MiniMaxH3PromptEnhancer:
    """Generate an enhanced H3 prompt with ComfyUI's loaded Qwen3-VL CLIP."""

    CATEGORY = "MiniMax H3/Prompting"
    FUNCTION = "enhance"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "system_prompt", "llm_prompt", "enhancer_report")
    OUTPUT_TOOLTIPS = (
        "Final text produced by Qwen after cleanup. For all H3 modes, this is the prompt to save or connect to an official MiniMax H3 conditioning node.",
        "Exact resolved system instructions used for enhancement. Connect to a text viewer to inspect/copy them; edit the system_prompt widget to customize behavior.",
        "Complete Qwen chat input, including system and user turns. Use this for debugging what the LLM received; do not send it to H3 as the video prompt.",
        "Generation status. It explains whether enhancement succeeded, was skipped because the CLIP is conditioning-only, or fell back after repetitive/punctuation output.",
    )
    DESCRIPTION = (
        "Second step after MiniMax H3 Prompt Guide. Connect h3_prompt, mode_report, and a Qwen3-VL CLIP. "
        "A complete generative CLIP needs no tail; MiniMax H3's 50-layer conditioning CLIP uses the optional 50-63 tail. "
        "The node optionally analyzes one image and rewrites the draft with an editable system prompt. "
        "Connect enhanced_prompt to ComfyUI's official H3 conditioning node."
    )

    @classmethod
    def INPUT_TYPES(cls):
        multiline = {"multiline": True, "dynamicPrompts": False}
        return {
            "required": {
                "clip": (
                    "CLIP",
                    {
                        "tooltip": (
                            "Connect either a complete instruction-tuned Qwen3-VL model or MiniMax H3's normal 50-layer conditioning CLIP. "
                            "For the 50-layer MiniMax CLIP, connect MiniMax H3 Generation Tail Loader to the optional clip_tail input. "
                            "This input means an LLM-capable ComfyUI CLIP, not an OpenAI CLIP vision model."
                        )
                    },
                ),
                "manual_prompt": (
                    "STRING",
                    {
                        **multiline,
                        "default": "",
                        "tooltip": (
                            "Connect h3_prompt from MiniMax H3 Prompt Guide. This is intentionally the pre-LLM structured draft. "
                            "You may instead paste a manual H3 prompt. Include real creative details; empty/default descriptions give Qwen little useful material to enhance."
                        ),
                    },
                ),
                "mode_report": (
                    "STRING",
                    {
                        **multiline,
                        "default": "",
                        "tooltip": (
                            "Recommended: connect mode_report from MiniMax H3 Prompt Guide. It tells Qwen the selected mode/checkpoint, resolved image/video/audio roles, task prefix, and warnings. "
                            "Leave blank only when the manual prompt already makes its H3 structure unambiguous."
                        ),
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        **multiline,
                        "default": DEFAULT_SYSTEM_PROMPT,
                        "tooltip": (
                            "Editable base behavior for Qwen. It contains H3 section order, shot/timestamp syntax, reference-label markers, dialogue, and audio rules. "
                            "Keep it unchanged initially. Add project-specific rules here, not scene content. If emptied, the built-in default is restored. The resolved text is echoed from system_prompt output."
                        ),
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 1400,
                        "min": 64,
                        "max": 4096,
                        "step": 1,
                        "tooltip": (
                            "Maximum tokens Qwen may generate, not input length. 1000-1400 is a practical range for the guide's detailed 350-500-word Ref2VA body plus other sections. "
                            "Lower values are faster but may truncate sections; increase only for dialogue-heavy prompts."
                        ),
                    },
                ),
                "sampling": (
                    ["deterministic", "sample"],
                    {
                        "default": "sample",
                        "tooltip": (
                            "Deterministic always chooses the most likely next token and ignores randomness, giving repeatable formatting. "
                            "Sample uses temperature/top-k/top-p/min-p and seed, usually producing richer descriptions."
                        ),
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.01,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": (
                            "Sampling creativity. Around 0.6-0.8 balances detail and strict formatting. Lower is more literal; high values can damage labels, timestamps, or section order. Used only when sampling=sample."
                        ),
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Limits sampling to the K most likely tokens. 64 is a stable default; 0 disables this filter. Used only when sampling=sample.",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Nucleus sampling threshold. 0.95 keeps likely alternatives while avoiding the long tail; 1.0 disables top-p filtering. Used only when sampling=sample.",
                    },
                ),
                "min_p": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Drops tokens whose probability is too small relative to the best token. 0.05 is conservative; 0 disables it. Used only when sampling=sample.",
                    },
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1.05,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": (
                            "Discourages repeated phrases. Keep close to 1.0 because H3 deliberately repeats fixed labels such as <Subject 1> across sections; an excessive penalty can corrupt required structure."
                        ),
                    },
                ),
                "presence_penalty": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.01,
                        "tooltip": (
                            "Additional penalty for tokens already used. Leave at 0 for H3 because reference labels and field names must recur. Raise only if Qwen is looping badly."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Controls repeatable random sampling. The same inputs, settings, and seed should reproduce the same text. It has no effect in deterministic mode.",
                    },
                ),
                "thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Off is recommended for faster direct rewriting. On allows Qwen to reason before answering, which may help complex reference relationships but costs more tokens/time. "
                            "Any decoded <think>...</think> block is removed from enhanced_prompt but remains conceptually part of generation cost."
                        ),
                    },
                ),
            },
            "optional": {
                "clip_tail": (
                    TAIL_TYPE,
                    {
                        "tooltip": (
                            "Connect MiniMax H3 Generation Tail Loader only when clip is H3's bundled "
                            "50-layer conditioning encoder. Leave disconnected for a complete "
                            "generation-capable Qwen3-VL CLIP. The tail is temporary and does not alter clip."
                        )
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional image for Qwen to inspect while rewriting. Only the first image in the batch is used. This image is LLM context only: it is not automatically an H3 first frame, last frame, or Ref2VA latent. "
                            "Assign its role in MiniMax H3 Prompt Guide, then separately connect the actual image to the official H3 Image/Reference node for generation."
                        ),
                    },
                )
            },
        }

    def enhance(
        self,
        clip,
        manual_prompt: str,
        mode_report: str,
        system_prompt: str,
        max_new_tokens: int,
        sampling: str,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
        presence_penalty: float,
        seed: int,
        thinking: bool,
        clip_tail=None,
        image=None,
    ):
        if clip is None:
            raise RuntimeError("A generation-capable CLIP input is required.")

        resolved_system_prompt = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
        has_image = image is not None
        user_prompt = build_llm_user_prompt(manual_prompt, mode_report, has_image)
        minimax_clip = _is_minimax_h3_tokenizer(clip)
        llm_prompt = format_chat_prompt(
            resolved_system_prompt,
            user_prompt,
            has_image=has_image,
            minimax_clip=minimax_clip,
        )
        if not thinking:
            llm_prompt += "<think>\n\n</think>\n\n"

        tail_name = _resolve_tail_name(clip_tail)
        compatibility_issue = clip_generation_issue(clip) if tail_name is None else None
        if compatibility_issue:
            return (
                manual_prompt.strip(),
                resolved_system_prompt,
                llm_prompt,
                compatibility_issue,
            )

        tokenize_options = {
            "skip_template": True,
            "min_length": 1,
            "thinking": thinking,
        }
        if has_image:
            first_image = image[:1]
            tokenize_options["image"] = first_image
            tokenize_options["images"] = [first_image]
        tokens = clip.tokenize(llm_prompt, **tokenize_options)
        generation_options = {
            "do_sample": sampling == "sample",
            "max_length": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "seed": seed,
        }
        if tail_name is None:
            generated_ids = _generate_with_clip(
                clip,
                tokens,
                generation_options,
                use_minimax_image_path=minimax_clip and has_image,
            )
        else:
            generated_ids = _generate_with_tail(
                clip, tail_name, tokens, generation_options
            )
        enhanced_prompt = clean_generated_prompt(clip.decode(generated_ids), manual_prompt)
        collapse = generation_collapse_reason(enhanced_prompt)
        if collapse:
            report = (
                f"Enhancement fallback: {collapse}. manual_prompt was returned unchanged. "
                "Try deterministic sampling, lower temperature, or a complete instruction-tuned "
                "Qwen3-VL model."
            )
            enhanced_prompt = manual_prompt.strip()
        else:
            report = (
                "Enhancement completed successfully with the temporary MiniMax generation tail; "
                "the connected conditioning CLIP was left unchanged."
                if tail_name is not None
                else "Enhancement completed successfully with the connected complete CLIP."
            )

        return (enhanced_prompt, resolved_system_prompt, llm_prompt, report)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3GenerationTailLoader": MiniMaxH3GenerationTailLoader,
    "MiniMaxH3PromptEnhancer": MiniMaxH3PromptEnhancer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3GenerationTailLoader": "MiniMax H3 Generation Tail Loader",
    "MiniMaxH3PromptEnhancer": "MiniMax H3 Prompt Enhancer (Qwen3-VL)",
}
