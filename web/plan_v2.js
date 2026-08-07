import { app } from "../../scripts/app.js";

const PROJECT = "MiniMaxH3PlanV2ProjectSetup";
const FOLEY = "MiniMaxH3PlanV2FoleyTarget";
const IMAGE = "MiniMaxH3PlanV2ImageReference";
const BINDING = "MiniMaxH3PlanV2SubjectBinding";
const VIDEO = "MiniMaxH3PlanV2VideoReference";
const REPLACEMENT = "MiniMaxH3PlanV2CharacterReplacement";
const AUDIO = "MiniMaxH3PlanV2AudioReference";
const SHOT = "MiniMaxH3PlanV2Shot";
const SHOT_KEYFRAME = "MiniMaxH3PlanV2ShotKeyframe";
const SHOT_MOTION = "MiniMaxH3PlanV2ShotMotionReference";
const DIALOGUE = "MiniMaxH3PlanV2DialogueEvent";
const MERGE = "MiniMaxH3PlanV2PromptMerge";
const APPLY_PROSE = "MiniMaxH3PlanV2ApplyProse";
const ENHANCER = "MiniMaxH3PlanV2PromptEnhancer";
const APPLY_REFERENCE = "MiniMaxH3PlanV2ApplyReferencePlan";
const PROMPT_REVIEW = "MiniMaxH3PlanV2PromptReview";

const PLAN_CLASSES = new Set([
    PROJECT,
    FOLEY,
    IMAGE,
    BINDING,
    VIDEO,
    REPLACEMENT,
    AUDIO,
    SHOT,
    SHOT_KEYFRAME,
    SHOT_MOTION,
    DIALOGUE,
    MERGE,
]);
const UI_CLASSES = new Set([
    ...PLAN_CLASSES,
    APPLY_PROSE,
    ENHANCER,
    PROMPT_REVIEW,
    APPLY_REFERENCE,
]);

const UNASSIGNED_IMAGE_USE = "Choose an image relationship";
const IMAGE_DEFINE_VISIBLE = "Define reusable visible content";
const IMAGE_FIRST_FRAME = "Exact first frame";
const IMAGE_LAST_FRAME = "Exact last frame";
const IMAGE_KEYFRAME = "Concrete keyframe / composition anchor";
const IMAGE_STORYBOARD = "Storyboard / shot planning";

const UNASSIGNED_VIDEO_USE = "Choose a video relationship";
const VIDEO_EDIT = "Source video to edit";
const VIDEO_CONTINUE = "Source video to continue";
const VIDEO_DEFINE_VISIBLE = "Define reusable visible content";
const VIDEO_MOTION = "Motion or action reference";
const VIDEO_STRUCTURE = "Camera, cuts, rhythm, or temporal-structure reference";

const UNASSIGNED_CONTENT_TYPE = "Choose visible content type";
const CONTENT_ACTION = "Pose, expression, action, or motion";
const RETENTION_AUTO = "Auto for this relationship";
const RETENTION_FULL = "fully_preserved";
const RETENTION_TRANSFER = "attribute_transfer";

const UNASSIGNED_AUDIO_USE = "Choose an audio relationship";
const AUDIO_VOICE = "Voice timbre and delivery";
const AUDIO_MUSIC = "Background-music style";
const AUDIO_BEAT = "Beat or rhythm";
const AUDIO_SFX = "Sound-effect texture";
const AUDIO_CONTENT = "Dialogue or lyric content";
const AUDIO_CONTINUITY = "Audio continuity";
const AUDIO_COPY_COMPLETE = "Copy complete signal";
const AUDIO_COPY_PARTIAL = "Copy selected part or layers";
const AUDIO_BROAD = "Broad audio inspiration";

const VISUAL_ANALYSIS_ENABLED = "Analyze reference images and sampled video";

const COLORS = {
    ready: "#277a52",
    info: "#245f8f",
    warning: "#9a6a1f",
    error: "#9b3434",
    bypass: "#555b62",
};

let refreshScheduled = false;

function className(node) {
    return String(node?.comfyClass || node?.type || "");
}

function widget(node, name) {
    return (node?.widgets || []).find((item) => item.name === name);
}

function input(node, name) {
    return (node?.inputs || []).find((item) => item.name === name);
}

function output(node, name) {
    return (node?.outputs || []).find((item) => item.name === name);
}

function isPictureNode(node) {
    return [IMAGE, SHOT_KEYFRAME].includes(className(node));
}

function isVideoNode(node) {
    return [VIDEO, SHOT_MOTION].includes(className(node));
}

function imageUse(node) {
    return className(node) === SHOT_KEYFRAME
        ? IMAGE_KEYFRAME
        : clean(widget(node, "image_use")?.value);
}

function videoUse(node) {
    return className(node) === SHOT_MOTION
        ? VIDEO_MOTION
        : clean(widget(node, "video_use")?.value);
}

function clean(value) {
    return String(value || "").trim();
}

function aliasKey(value) {
    return clean(value).toLocaleLowerCase();
}

function setWidgetVisible(target, visible) {
    if (!target) return;
    if (!Object.prototype.hasOwnProperty.call(target, "__h3OriginalComputeSize")) {
        target.__h3OriginalComputeSize = target.computeSize;
    }
    target.hidden = !visible;
    target.serialize = target.serialize !== false;
    if (target.element) target.element.style.display = visible ? "" : "none";
    if (visible) {
        if (target.__h3OriginalComputeSize) {
            target.computeSize = target.__h3OriginalComputeSize;
        } else {
            delete target.computeSize;
        }
    } else {
        target.computeSize = () => [0, -4];
    }
}

function setWidgetValue(node, name, value) {
    const target = widget(node, name);
    if (!target || target.value === value) return;
    target.value = value;
    if (target.inputEl) target.inputEl.value = value;
}

function bindingTransfers(node) {
    const retention = clean(widget(node, "retention")?.value);
    const contentType = clean(widget(node, "content_type")?.value);
    return (
        retention === RETENTION_TRANSFER ||
        (retention === RETENTION_AUTO && contentType === CONTENT_ACTION)
    );
}

function setOutputLabel(node, name, label) {
    const target = output(node, name);
    if (target) target.label = label || target.name;
}

function graphLink(graph, linkId) {
    if (linkId == null) return null;
    return graph?.links?.[linkId] || graph?.links?.get?.(linkId) || null;
}

function isReroute(node) {
    const name = className(node) || String(node?.title || "");
    return name === "Reroute" || name.endsWith("Reroute") || node?.isVirtualNode === true;
}

function connectedPassthroughInput(node, outputType) {
    const connected = (node?.inputs || []).filter((item) => item.link != null);
    return (
        connected.find((item) => item.type === outputType) ||
        connected.find((item) => item.name === "h3_plan") ||
        connected[0] ||
        null
    );
}

function resolveOrigin(graph, linkId, visited = new Set()) {
    const link = graphLink(graph, linkId);
    if (!link) return null;
    const origin = graph?.getNodeById?.(link.origin_id);
    if (!origin || visited.has(origin.id)) return null;
    visited.add(origin.id);

    if (isReroute(origin) || origin.mode === 4) {
        const outputType = origin.outputs?.[link.origin_slot]?.type || link.type;
        const previous = connectedPassthroughInput(origin, outputType);
        return previous?.link == null
            ? null
            : resolveOrigin(graph, previous.link, visited);
    }
    return origin;
}

function previousPlanNode(node) {
    const planInput = input(node, "h3_plan");
    if (planInput?.link == null || !node?.graph) return null;
    const origin = resolveOrigin(node.graph, planInput.link);
    return PLAN_CLASSES.has(className(origin)) ? origin : null;
}

function chainThrough(node) {
    const chain = [];
    const visited = new Set();
    let current = previousPlanNode(node);
    while (current && !visited.has(current.id)) {
        visited.add(current.id);
        chain.push(current);
        current = previousPlanNode(current);
    }
    chain.reverse();
    if (PLAN_CLASSES.has(className(node)) && node.mode !== 4) chain.push(node);
    return chain;
}

function bestContextChain(node) {
    let best = chainThrough(node);
    const graph = node?.graph;
    if (!graph) return best;
    for (const candidate of graph._nodes || []) {
        if (!PLAN_CLASSES.has(className(candidate)) || candidate.mode === 4) continue;
        const chain = chainThrough(candidate);
        if (
            chain.length > best.length &&
            chain.some((entry) => entry.id === node.id)
        ) {
            best = chain;
        }
    }
    return best;
}

function referenceHandleSource(node, inputName) {
    const sourceInput = input(node, inputName);
    if (sourceInput?.link == null || !node.graph) return null;
    let origin = resolveOrigin(node.graph, sourceInput.link);
    const visited = new Set();
    while (origin && !visited.has(origin.id)) {
        visited.add(origin.id);
        if (className(origin) !== BINDING) return origin;
        const bindingInput = input(origin, "reference_handle");
        if (bindingInput?.link == null) return null;
        origin = resolveOrigin(origin.graph, bindingInput.link, visited);
    }
    return origin;
}

function addSubject(subjects, aliases, node, name) {
    const alias = clean(name);
    const key = aliasKey(alias);
    if (!alias || aliases.has(key)) return;
    const subject = {
        node,
        alias,
        label: "<Subject " + (subjects.length + 1) + ">",
    };
    aliases.set(key, subject);
    subjects.push(subject);
}

function buildCatalog(chain) {
    const project = chain.find((node) => className(node) === PROJECT) || null;
    const foley = chain.find((node) => className(node) === FOLEY) || null;
    const pictures = chain.filter(isPictureNode);
    const videos = chain.filter(isVideoNode);
    const replacements = chain.filter((node) => className(node) === REPLACEMENT);
    const audios = chain.filter((node) => className(node) === AUDIO);
    const shots = chain.filter((node) => className(node) === SHOT);
    const dialogues = chain.filter((node) => className(node) === DIALOGUE);
    const subjects = [];
    const subjectsByAlias = new Map();

    for (const node of chain) {
        const type = className(node);
        if (
            type === IMAGE &&
            clean(widget(node, "image_use")?.value) === IMAGE_DEFINE_VISIBLE
        ) {
            addSubject(subjects, subjectsByAlias, node, widget(node, "subject_name")?.value);
        } else if (
            type === VIDEO &&
            clean(widget(node, "video_use")?.value) === VIDEO_DEFINE_VISIBLE
        ) {
            addSubject(subjects, subjectsByAlias, node, widget(node, "subject_name")?.value);
        } else if (type === BINDING) {
            addSubject(subjects, subjectsByAlias, node, widget(node, "subject_name")?.value);
        }
    }

    const pictureLabels = new Map();
    pictures.forEach((node, index) => {
        pictureLabels.set(node.id, "<Picture " + (index + 1) + ">");
    });
    const videoLabels = new Map();
    videos.forEach((node, index) => {
        videoLabels.set(node.id, "<Video " + (index + 1) + ">");
    });
    const motionSubjectLabels = new Map();
    let nextMotionSubject = subjects.length + 1;
    videos.forEach((node) => {
        if (videoUse(node) !== VIDEO_MOTION) return;
        motionSubjectLabels.set(node.id, "<Subject " + nextMotionSubject + ">");
        nextMotionSubject += 1;
    });

    const audioLabels = new Map();
    const routes = new Map();
    let audioNumber = 0;
    let standaloneNumber = 0;
    pictures.forEach((node, index) => {
        routes.set(node.id, "ref_image_" + index);
    });
    videos.forEach((videoNode, videoIndex) => {
        const paired = audios.find((audioNode) => {
            const source = referenceHandleSource(audioNode, "paired_video");
            return source?.id === videoNode.id;
        });
        if (paired) {
            audioNumber += 1;
            audioLabels.set(paired.id, "<Audio " + audioNumber + ">");
            routes.set(paired.id, "ref_video_audio_" + videoIndex);
        }
        routes.set(videoNode.id, "ref_video_" + videoIndex);
    });
    for (const audioNode of audios) {
        if (audioLabels.has(audioNode.id)) continue;
        audioNumber += 1;
        audioLabels.set(audioNode.id, "<Audio " + audioNumber + ">");
        routes.set(audioNode.id, "ref_audio_" + standaloneNumber);
        standaloneNumber += 1;
    }

    const speakers = [];
    const speakersByAlias = new Map();
    for (const node of dialogues) {
        const alias = clean(widget(node, "speaker")?.value);
        const key = aliasKey(alias);
        if (!alias || speakersByAlias.has(key)) continue;
        const speaker = { alias, id: "S" + (speakers.length + 1) };
        speakers.push(speaker);
        speakersByAlias.set(key, speaker);
    }

    const imageUses = pictures.map(imageUse);
    const unassignedReferenceIds = new Set([
        ...pictures
            .filter(
                (node) =>
                    imageUse(node) === UNASSIGNED_IMAGE_USE
            )
            .map((node) => node.id),
        ...videos
            .filter(
                (node) =>
                    videoUse(node) === UNASSIGNED_VIDEO_USE
            )
            .map((node) => node.id),
        ...audios
            .filter(
                (node) =>
                    clean(widget(node, "audio_use")?.value) === UNASSIGNED_AUDIO_USE
            )
            .map((node) => node.id),
    ]);
    const endpointsOnly = imageUses.every(
        (value) => value === IMAGE_FIRST_FRAME || value === IMAGE_LAST_FRAME
    );
    const hasEndpoint = imageUses.some(
        (value) => value === IMAGE_FIRST_FRAME || value === IMAGE_LAST_FRAME
    );
    const requiresRef2va = Boolean(
        videos.length ||
        audios.length ||
        subjects.length ||
        (pictures.length && !endpointsOnly)
    );
    const endpointConflict = hasEndpoint && (requiresRef2va || Boolean(foley));
    let mode = "T2VA";
    if (endpointConflict) {
        mode = "Invalid mixed routes";
    } else if (requiresRef2va) {
        mode = "Ref2VA";
    } else if (pictures.length) {
        const first = imageUses.includes(IMAGE_FIRST_FRAME);
        const last = imageUses.includes(IMAGE_LAST_FRAME);
        mode = first && last ? "FL2VA" : first ? "I2VA" : "L2VA";
    }
    if (endpointConflict) {
        for (const node of [...pictures, ...videos, ...audios]) {
            routes.set(node.id, "incompatible endpoint/Ref2VA mix");
        }
    } else if (mode !== "Ref2VA") {
        for (const node of pictures) {
            const use = imageUse(node);
            routes.set(
                node.id,
                use === IMAGE_FIRST_FRAME ? "first_frame" : "last_frame"
            );
        }
    }

    return {
        project,
        foley,
        pictures,
        videos,
        replacements,
        audios,
        shots,
        dialogues,
        subjects,
        subjectsByAlias,
        pictureLabels,
        videoLabels,
        motionSubjectLabels,
        audioLabels,
        routes,
        speakers,
        speakersByAlias,
        mode,
        endpointConflict,
        unassignedReferenceIds,
    };
}

function nativeFrameCount(duration) {
    const requested = Math.max(5, Math.ceil(Number(duration) * 24 - 1e-9));
    return requested + ((5 - requested) % 17 + 17) % 17;
}

function effectiveDuration(catalog) {
    const requested = Number(widget(catalog.project, "duration_seconds")?.value || 6);
    return nativeFrameCount(requested) / 24;
}

function scopeSyntax(value, shotCount) {
    const text = clean(value);
    if (!text) return { valid: true, text: "" };
    let normalized = text
        .replace(/\bshots?\b/gi, "")
        .trim()
        .replace(/^[\[\(]+|[\]\)]+$/g, "")
        .trim();
    if (normalized.toLocaleLowerCase() === "all" || normalized === "*") {
        if (shotCount === 0) {
            return { valid: false, text: "scope awaits a connected Shot chain" };
        }
        return { valid: true, text: "all " + shotCount + " Shots" };
    }
    const numbers = [];
    for (const fragment of normalized.split(",")) {
        const part = fragment.trim();
        if (!part) continue;
        const range = part.match(/^(\d+)\s*-\s*(\d+)$/);
        if (range) {
            const start = Number(range[1]);
            const end = Number(range[2]);
            if (start < 1 || end < start) {
                return { valid: false, text: "invalid or reversed scope " + part };
            }
            for (let value = start; value <= end; value += 1) numbers.push(value);
        } else if (/^\d+$/.test(part)) {
            const number = Number(part);
            if (number < 1) return { valid: false, text: "Shot numbers start at 1" };
            numbers.push(number);
        } else {
            return { valid: false, text: "use 3, 3,4, 3-4, or all" };
        }
    }
    if (!numbers.length) return { valid: false, text: "scope is empty or invalid" };
    if (shotCount && Math.max(...numbers) > shotCount) {
        return {
            valid: false,
            text: "scope exceeds the connected " + shotCount + "-Shot timeline",
        };
    }
    if (!shotCount) {
        return { valid: false, text: "scope awaits a connected Shot chain" };
    }
    return {
        valid: true,
        text: "Shots " + [...new Set(numbers)].join(","),
    };
}

function createAliasEditor(node, fieldName, labelText, strict) {
    const hidden = widget(node, fieldName);
    if (!hidden || !node.addDOMWidget) return null;
    setWidgetVisible(hidden, false);

    const root = document.createElement("label");
    root.style.cssText = [
        "box-sizing:border-box",
        "display:flex",
        "flex-direction:column",
        "gap:4px",
        "width:100%",
        "padding:4px 8px",
        "color:#bbb",
        "font-size:11px",
    ].join(";");
    const title = document.createElement("span");
    title.textContent = labelText;
    const editor = document.createElement("input");
    editor.type = "text";
    editor.value = clean(hidden.value);
    editor.autocomplete = "off";
    editor.style.cssText = [
        "box-sizing:border-box",
        "width:100%",
        "height:28px",
        "padding:4px 7px",
        "border:1px solid #555",
        "border-radius:5px",
        "background:#202020",
        "color:#eee",
    ].join(";");
    const list = document.createElement("datalist");
    const listId = "h3-subjects-" + node.id + "-" + fieldName;
    list.id = listId;
    editor.setAttribute("list", listId);
    root.append(title, editor, list);
    const domWidget = node.addDOMWidget(
        "h3_" + fieldName + "_picker",
        "div",
        root,
        {
            serialize: false,
            getMinHeight: () => 52,
        }
    );
    editor.addEventListener("input", () => {
        hidden.value = editor.value;
        hidden.callback?.(editor.value);
        node.graph?.setDirtyCanvas(true, true);
        scheduleRefresh();
    });
    return { fieldName, hidden, root, editor, list, domWidget, strict };
}

function refreshAliasEditor(aliasEditor, catalog, visible) {
    if (!aliasEditor) return;
    setWidgetVisible(aliasEditor.hidden, false);
    setWidgetVisible(aliasEditor.domWidget, visible);
    if (!visible) return;
    if (document.activeElement !== aliasEditor.editor) {
        aliasEditor.editor.value = clean(aliasEditor.hidden.value);
    }
    aliasEditor.list.replaceChildren();
    for (const subject of catalog.subjects) {
        const option = document.createElement("option");
        option.value = subject.alias;
        option.label = subject.label + " · " + subject.alias;
        aliasEditor.list.appendChild(option);
    }
    const value = aliasKey(aliasEditor.editor.value);
    const unknown = value && !catalog.subjectsByAlias.has(value);
    aliasEditor.editor.style.borderColor =
        aliasEditor.strict && unknown ? "#c45a5a" : "#555";
    aliasEditor.editor.placeholder = catalog.subjects.length
        ? aliasEditor.strict
            ? "Choose/type an upstream Subject alias"
            : "Choose an upstream Subject or type a nonvisual speaker"
        : aliasEditor.strict
          ? "No upstream Subject is connected yet"
          : "Type a nonvisual speaker name";
}

function labelSuggestions(catalog) {
    const suggestions = [];
    for (const subject of catalog.subjects) {
        suggestions.push({
            token: subject.label,
            detail: subject.alias,
        });
    }
    for (const node of catalog.pictures) {
        suggestions.push({
            token: catalog.pictureLabels.get(node.id),
            detail: clean(widget(node, "reference_name")?.value) || "image reference",
        });
    }
    for (const node of catalog.videos) {
        suggestions.push({
            token: catalog.videoLabels.get(node.id),
            detail: clean(widget(node, "reference_name")?.value) || "video reference",
        });
        const motionSubject = catalog.motionSubjectLabels.get(node.id);
        if (motionSubject) {
            suggestions.push({
                token: motionSubject,
                detail:
                    (clean(widget(node, "reference_name")?.value) || "referenced motion") +
                    " · reusable action from " +
                    catalog.videoLabels.get(node.id),
            });
        }
    }
    for (const node of catalog.audios) {
        const use = clean(widget(node, "audio_use")?.value);
        const event = clean(widget(node, "target_layer_or_event")?.value);
        const name = clean(widget(node, "reference_name")?.value) || "audio reference";
        suggestions.push({
            token: catalog.audioLabels.get(node.id),
            detail: [use, event || name, "insert in sound sentence"].filter(Boolean).join(" · "),
        });
    }
    return suggestions;
}

function createShotEditor(node) {
    const hidden = widget(node, "description");
    if (!hidden || !node.addDOMWidget) return null;
    setWidgetVisible(hidden, false);

    const root = document.createElement("div");
    root.style.cssText = [
        "box-sizing:border-box",
        "display:flex",
        "flex-direction:column",
        "gap:4px",
        "width:100%",
        "padding:5px 8px",
        "color:#bbb",
        "font-size:11px",
    ].join(";");
    const hint = document.createElement("div");
    hint.textContent = "Shot description · type < to insert a Subject or media tag; place <Audio N> in the exact sound sentence";
    const editor = document.createElement("textarea");
    editor.value = String(hidden.value || "");
    editor.rows = 6;
    editor.spellcheck = true;
    editor.style.cssText = [
        "box-sizing:border-box",
        "width:100%",
        "min-height:118px",
        "resize:vertical",
        "padding:7px",
        "border:1px solid #555",
        "border-radius:5px",
        "background:#202020",
        "color:#eee",
        "font:12px sans-serif",
        "line-height:1.35",
    ].join(";");
    const menu = document.createElement("div");
    menu.style.cssText = [
        "display:none",
        "max-height:145px",
        "overflow:auto",
        "border:1px solid #4d7290",
        "border-radius:5px",
        "background:#171b1f",
    ].join(";");
    root.append(hint, editor, menu);
    const domWidget = node.addDOMWidget("h3_shot_editor", "div", root, {
        serialize: false,
        getMinHeight: () => 170,
    });

    const state = {
        hidden,
        root,
        editor,
        menu,
        domWidget,
        suggestions: [],
        filtered: [],
        tokenStart: -1,
        selected: 0,
    };

    function closeMenu() {
        state.menu.style.display = "none";
        state.menu.replaceChildren();
        state.filtered = [];
        state.tokenStart = -1;
        state.selected = 0;
    }

    function insertSuggestion(suggestion) {
        const cursor = state.editor.selectionStart;
        const before = state.editor.value.slice(0, state.tokenStart);
        const after = state.editor.value.slice(cursor);
        const separator = after.startsWith(" ") ? "" : " ";
        state.editor.value = before + suggestion.token + separator + after;
        const next = before.length + suggestion.token.length + separator.length;
        state.editor.setSelectionRange(next, next);
        state.hidden.value = state.editor.value;
        state.hidden.callback?.(state.editor.value);
        closeMenu();
        state.editor.focus();
        node.graph?.setDirtyCanvas(true, true);
        scheduleRefresh();
    }

    function renderMenu() {
        state.menu.replaceChildren();
        if (!state.filtered.length) {
            state.menu.style.display = "none";
            return;
        }
        state.filtered.forEach((suggestion, index) => {
            const button = document.createElement("button");
            button.type = "button";
            button.textContent = suggestion.token + " · " + suggestion.detail;
            button.style.cssText = [
                "display:block",
                "width:100%",
                "padding:6px 8px",
                "border:0",
                "border-bottom:1px solid #333",
                "background:" + (index === state.selected ? "#2c536f" : "#171b1f"),
                "color:#eee",
                "text-align:left",
                "cursor:pointer",
                "font-size:11px",
            ].join(";");
            button.addEventListener("mousedown", (event) => {
                event.preventDefault();
                insertSuggestion(suggestion);
            });
            state.menu.appendChild(button);
        });
        state.menu.style.display = "block";
    }

    function updateMenu() {
        const cursor = state.editor.selectionStart;
        const prefix = state.editor.value.slice(0, cursor);
        const match = prefix.match(/<([A-Za-z]*)(?:\s*(\d*))?$/);
        if (!match) {
            closeMenu();
            return;
        }
        state.tokenStart = cursor - match[0].length;
        const query = match[0].slice(1).toLocaleLowerCase();
        state.filtered = state.suggestions.filter((suggestion) => {
            const haystack = (suggestion.token + " " + suggestion.detail).toLocaleLowerCase();
            return haystack.includes(query);
        });
        state.selected = Math.min(state.selected, Math.max(0, state.filtered.length - 1));
        renderMenu();
    }

    editor.addEventListener("input", () => {
        hidden.value = editor.value;
        hidden.callback?.(editor.value);
        updateMenu();
        node.graph?.setDirtyCanvas(true, true);
        scheduleRefresh();
    });
    editor.addEventListener("click", updateMenu);
    editor.addEventListener("keyup", (event) => {
        if (!["ArrowUp", "ArrowDown", "Enter", "Escape"].includes(event.key)) {
            updateMenu();
        }
    });
    editor.addEventListener("keydown", (event) => {
        if (!state.filtered.length) return;
        if (event.key === "ArrowDown") {
            event.preventDefault();
            state.selected = (state.selected + 1) % state.filtered.length;
            renderMenu();
        } else if (event.key === "ArrowUp") {
            event.preventDefault();
            state.selected =
                (state.selected - 1 + state.filtered.length) % state.filtered.length;
            renderMenu();
        } else if (event.key === "Enter" || event.key === "Tab") {
            event.preventDefault();
            insertSuggestion(state.filtered[state.selected]);
        } else if (event.key === "Escape") {
            event.preventDefault();
            closeMenu();
        }
    });
    editor.addEventListener("blur", () => {
        window.setTimeout(closeMenu, 100);
    });
    state.closeMenu = closeMenu;
    return state;
}

function refreshShotEditor(shotEditor, catalog) {
    if (!shotEditor) return;
    setWidgetVisible(shotEditor.hidden, false);
    setWidgetVisible(shotEditor.domWidget, true);
    shotEditor.suggestions = labelSuggestions(catalog);
    if (document.activeElement !== shotEditor.editor) {
        shotEditor.editor.value = String(shotEditor.hidden.value || "");
    }
}

function installWidgetCallbacks(node) {
    for (const target of node.widgets || []) {
        if (target.__h3PlanV2Callback) continue;
        const original = target.callback;
        target.callback = function () {
            const result = original?.apply(this, arguments);
            scheduleRefresh();
            return result;
        };
        target.__h3PlanV2Callback = true;
    }
}

function resizeNode(node) {
    if (!node?.computeSize || !node?.setSize) return;
    // The prompt review extension owns a freely resizable, fill-height DOM
    // editor. Recomputing its height on every Plan-v2 refresh would snap the
    // node back and make vertical resizing appear broken.
    if (className(node) === PROMPT_REVIEW) return;
    const computed = node.computeSize();
    const width = Math.max(node.size?.[0] || 0, className(node) === SHOT ? 390 : 340);
    const height = computed?.[1] || node.size?.[1] || 120;
    if (
        Math.abs(width - (node.size?.[0] || 0)) > 1 ||
        Math.abs(height - (node.size?.[1] || 0)) > 1
    ) {
        node.setSize([width, height]);
    }
}

function scopeWidgetStatus(node, catalog) {
    const target = widget(node, "shot_scope");
    if (!target) return { valid: true, text: "" };
    return scopeSyntax(target.value, catalog.shots.length);
}

function referenceBadge(node, catalog) {
    const type = className(node);
    if (isPictureNode(node)) {
        const picture = catalog.pictureLabels.get(node.id) || "<Picture ?>";
        const alias = clean(widget(node, "subject_name")?.value);
        const subject = catalog.subjectsByAlias.get(aliasKey(alias));
        const route = catalog.routes.get(node.id) || "route pending";
        const position =
            type === SHOT_KEYFRAME
                ? clean(widget(node, "keyframe_position")?.value)
                : "";
        return (
            picture +
            (position ? " · " + position : "") +
            (subject ? " · " + subject.label : "") +
            " → " +
            route
        );
    }
    if (isVideoNode(node)) {
        const video = catalog.videoLabels.get(node.id) || "<Video ?>";
        const alias = clean(
            widget(node, type === SHOT_MOTION ? "target_subject" : "subject_name")?.value
        );
        const subject = catalog.subjectsByAlias.get(aliasKey(alias));
        const route = catalog.routes.get(node.id) || "route pending";
        const motionSubject = catalog.motionSubjectLabels.get(node.id);
        if (motionSubject) {
            return (
                motionSubject +
                " from " +
                video +
                (subject ? " → " + subject.label : " → target required") +
                " · " +
                route
            );
        }
        return video + (subject ? " · " + subject.label : "") + " → " + route;
    }
    if (type === AUDIO) {
        const audio = catalog.audioLabels.get(node.id) || "<Audio ?>";
        return audio + " → " + (catalog.routes.get(node.id) || "route pending");
    }
    return "";
}

function setupNode(node) {
    const type = className(node);
    // Workflows saved before Shot handles existed may restore their serialized
    // two-output layout. Add only the missing typed output so they can use the
    // new attachment chain without recreating every Shot node.
    if (type === SHOT && !output(node, "shot_handle")) {
        node.addOutput?.("shot_handle", "MINIMAX_H3_SHOT_HANDLE_V2");
    }
    if (node.__h3PlanV2Setup) {
        installWidgetCallbacks(node);
        return;
    }
    node.__h3PlanV2Setup = true;
    node.__h3AliasEditors = {};
    if (type === IMAGE || type === BINDING) {
        node.__h3AliasEditors.transfer_target_subject = createAliasEditor(
            node,
            "transfer_target_subject",
            "Transfer target Subject",
            true
        );
    } else if (type === VIDEO || type === SHOT_MOTION) {
        node.__h3AliasEditors.target_subject = createAliasEditor(
            node,
            "target_subject",
            "Motion target Subject",
            true
        );
    } else if (type === REPLACEMENT) {
        node.__h3AliasEditors.replacement_subject = createAliasEditor(
            node,
            "replacement_subject",
            "Replacement Subject",
            true
        );
    } else if (type === AUDIO) {
        node.__h3AliasEditors.target_speaker = createAliasEditor(
            node,
            "target_speaker",
            "Target speaker",
            false
        );
    } else if (type === DIALOGUE) {
        node.__h3AliasEditors.speaker = createAliasEditor(
            node,
            "speaker",
            "Speaker",
            false
        );
    } else if (type === SHOT) {
        node.__h3ShotEditor = createShotEditor(node);
    }
    installWidgetCallbacks(node);
}

function refreshConditionalWidgets(node, catalog) {
    const type = className(node);
    const upstreamCatalog = buildCatalog(
        chainThrough(node).filter((entry) => entry.id !== node.id)
    );
    if (type === IMAGE) {
        const use = clean(widget(node, "image_use")?.value);
        const reusable = use === IMAGE_DEFINE_VISIBLE;
        if (!reusable) {
            setWidgetValue(node, "content_type", UNASSIGNED_CONTENT_TYPE);
            setWidgetValue(node, "subject_name", "");
        }
        const scopeVisible =
            reusable || use === IMAGE_KEYFRAME || use === IMAGE_STORYBOARD;
        if (!scopeVisible) setWidgetValue(node, "shot_scope", "");
        if (
            [IMAGE_FIRST_FRAME, IMAGE_LAST_FRAME, IMAGE_KEYFRAME].includes(use) &&
            ![RETENTION_AUTO, RETENTION_FULL].includes(
                clean(widget(node, "retention")?.value)
            )
        ) {
            setWidgetValue(node, "retention", RETENTION_AUTO);
        }
        if (
            use === IMAGE_STORYBOARD &&
            clean(widget(node, "retention")?.value) === RETENTION_TRANSFER
        ) {
            setWidgetValue(node, "retention", RETENTION_AUTO);
        }
        const transfers = reusable && bindingTransfers(node);
        if (!transfers) setWidgetValue(node, "transfer_target_subject", "");
        setWidgetVisible(widget(node, "content_type"), reusable);
        setWidgetVisible(widget(node, "subject_name"), reusable);
        setWidgetVisible(widget(node, "shot_scope"), scopeVisible);
        refreshAliasEditor(
            node.__h3AliasEditors?.transfer_target_subject,
            upstreamCatalog,
            transfers
        );
    } else if (type === BINDING) {
        const transfers = bindingTransfers(node);
        if (!transfers) setWidgetValue(node, "transfer_target_subject", "");
        refreshAliasEditor(
            node.__h3AliasEditors?.transfer_target_subject,
            upstreamCatalog,
            transfers
        );
    } else if (type === VIDEO) {
        const use = clean(widget(node, "video_use")?.value);
        const reusable = use === VIDEO_DEFINE_VISIBLE;
        const assigned = Boolean(use) && use !== UNASSIGNED_VIDEO_USE;
        if (!reusable) {
            setWidgetValue(node, "content_type", UNASSIGNED_CONTENT_TYPE);
            setWidgetValue(node, "subject_name", "");
            setWidgetValue(node, "retention", RETENTION_AUTO);
        }
        const transfers = use === VIDEO_MOTION || (reusable && bindingTransfers(node));
        if (!transfers) setWidgetValue(node, "target_subject", "");
        if (!assigned) setWidgetValue(node, "shot_scope", "");
        setWidgetVisible(widget(node, "content_type"), reusable);
        setWidgetVisible(widget(node, "subject_name"), reusable);
        setWidgetVisible(widget(node, "retention"), reusable);
        setWidgetVisible(widget(node, "shot_scope"), assigned);
        refreshAliasEditor(
            node.__h3AliasEditors?.target_subject,
            upstreamCatalog,
            transfers
        );
    } else if (type === SHOT_MOTION) {
        refreshAliasEditor(
            node.__h3AliasEditors?.target_subject,
            upstreamCatalog,
            true
        );
    } else if (type === REPLACEMENT) {
        refreshAliasEditor(
            node.__h3AliasEditors?.replacement_subject,
            upstreamCatalog,
            true
        );
    } else if (type === AUDIO) {
        const use = clean(widget(node, "audio_use")?.value);
        const assigned = Boolean(use) && use !== UNASSIGNED_AUDIO_USE;
        const speakerUse = use === AUDIO_VOICE || use === AUDIO_CONTENT;
        const contentUse = use === AUDIO_CONTENT;
        const layerUse = [
            AUDIO_MUSIC,
            AUDIO_BEAT,
            AUDIO_SFX,
            AUDIO_CONTINUITY,
            AUDIO_BROAD,
        ].includes(use);
        const instructionsUse = assigned && use !== AUDIO_COPY_COMPLETE;
        const scopeUse = assigned && use !== AUDIO_COPY_COMPLETE;
        if (!speakerUse) setWidgetValue(node, "target_speaker", "");
        if (!contentUse) {
            setWidgetValue(node, "language", "");
            setWidgetValue(node, "transcript", "");
        }
        if (!layerUse) setWidgetValue(node, "target_layer_or_event", "");
        if (!instructionsUse) setWidgetValue(node, "instructions", "");
        if (!scopeUse) setWidgetValue(node, "shot_scope", "");
        refreshAliasEditor(
            node.__h3AliasEditors?.target_speaker,
            upstreamCatalog,
            speakerUse
        );
        setWidgetVisible(widget(node, "language"), contentUse);
        setWidgetVisible(widget(node, "transcript"), contentUse);
        setWidgetVisible(widget(node, "target_layer_or_event"), layerUse);
        setWidgetVisible(widget(node, "instructions"), instructionsUse);
        setWidgetVisible(widget(node, "shot_scope"), scopeUse);
    } else if (type === DIALOGUE) {
        refreshAliasEditor(node.__h3AliasEditors?.speaker, upstreamCatalog, true);
    } else if (type === SHOT) {
        const localCatalog = buildCatalog(chainThrough(node));
        const shotNumber = localCatalog.shots.findIndex((entry) => entry.id === node.id) + 1;
        const first = shotNumber === 1;
        const cut = widget(node, "cut_at");
        if (first && cut) cut.value = 0;
        setWidgetVisible(cut, !first);
        setWidgetVisible(widget(node, "transition"), !first);
        refreshShotEditor(node.__h3ShotEditor, catalog);
    } else if (type === ENHANCER) {
        const analyze =
            clean(widget(node, "visual_analysis")?.value) === VISUAL_ANALYSIS_ENABLED;
        for (const name of [
            "analysis_long_edge",
            "video_analysis_fps",
            "max_analysis_frames",
        ]) {
            setWidgetVisible(widget(node, name), analyze);
        }
        const sampling = clean(widget(node, "sampling")?.value) === "sample";
        for (const name of [
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "seed",
        ]) {
            setWidgetVisible(widget(node, name), sampling);
        }
    }
}

function refreshBadgeAndOutputs(node, catalog) {
    const type = className(node);
    let text = "Plan v2";
    let color = COLORS.info;
    if (node.mode === 4) {
        text = "Bypassed · labels recompute downstream";
        color = COLORS.bypass;
    } else if (type === PROJECT) {
        const requested = Number(widget(node, "duration_seconds")?.value || 6);
        const frames = nativeFrameCount(requested);
        text =
            requested.toFixed(3) +
            "s requested · " +
            frames +
            "f · " +
            (frames / 24).toFixed(3) +
            "s native";
        color = COLORS.ready;
    } else if (type === FOLEY) {
        const requested = Number(
            widget(catalog.project, "duration_seconds")?.value || 6
        );
        const frames = nativeFrameCount(requested);
        text =
            "Foley · video mask 0 · audio mask 1 · " +
            frames +
            "f / " +
            (frames / 24).toFixed(3) +
            "s";
        setOutputLabel(node, "h3_video", "locked picture track");
        color = catalog.endpointConflict ? COLORS.error : COLORS.ready;
    } else if (isPictureNode(node) || isVideoNode(node) || type === AUDIO) {
        text = referenceBadge(node, catalog);
        const scope = scopeWidgetStatus(node, catalog);
        if (catalog.unassignedReferenceIds.has(node.id)) {
            const kind = isPictureNode(node)
                ? "image"
                : isVideoNode(node)
                  ? "video"
                  : "audio";
            text = "Choose a " + kind + " relationship before queueing";
            color = COLORS.warning;
        } else if (catalog.endpointConflict) {
            text += " · endpoint and Ref2VA roles cannot share one plan";
            color = COLORS.error;
        } else if (!scope.valid && clean(widget(node, "shot_scope")?.value)) {
            text += " · " + scope.text;
            color = COLORS.error;
        } else {
            color = COLORS.ready;
        }
        if (isPictureNode(node)) {
            setOutputLabel(node, "h3_image", catalog.routes.get(node.id));
        } else if (isVideoNode(node)) {
            setOutputLabel(node, "h3_video", catalog.routes.get(node.id));
        } else {
            setOutputLabel(node, "h3_audio", catalog.routes.get(node.id));
        }
    } else if (type === BINDING) {
        const alias = clean(widget(node, "subject_name")?.value);
        const subject = catalog.subjectsByAlias.get(aliasKey(alias));
        text = subject ? subject.label + " · " + subject.alias : "Subject alias required";
        const scope = scopeWidgetStatus(node, catalog);
        color = subject ? COLORS.ready : COLORS.warning;
        if (catalog.endpointConflict) {
            text += " · exact endpoints cannot receive Subject bindings";
            color = COLORS.error;
        } else if (!scope.valid && clean(widget(node, "shot_scope")?.value)) {
            text += " · " + scope.text;
            color = COLORS.error;
        }
    } else if (type === REPLACEMENT) {
        const alias = clean(widget(node, "replacement_subject")?.value);
        const subject = catalog.subjectsByAlias.get(aliasKey(alias));
        const source = referenceHandleSource(node, "source_video");
        const video = source ? catalog.videoLabels.get(source.id) : null;
        const sourceUse = clean(widget(source, "video_use")?.value);
        const sourceCharacter = clean(
            widget(node, "source_character_description")?.value
        );
        const scope = scopeWidgetStatus(node, catalog);
        text =
            (video || "Source Video required") +
            " performer → " +
            (subject ? subject.label + " · " + subject.alias : "Subject required");
        if (!video || !subject || !sourceCharacter) {
            color = COLORS.warning;
        } else if (![VIDEO_EDIT, VIDEO_CONTINUE].includes(sourceUse)) {
            text += " · source must use video edit or continuation";
            color = COLORS.error;
        } else if (!scope.valid) {
            text += " · " + scope.text;
            color = COLORS.error;
        } else {
            if (sourceUse === VIDEO_CONTINUE) {
                text += " · edit source → continue edited endpoint";
            }
            text += " · " + scope.text;
            color = COLORS.ready;
        }
    } else if (type === SHOT) {
        const local = buildCatalog(chainThrough(node));
        const number = local.shots.findIndex((entry) => entry.id === node.id) + 1;
        const fullIndex = catalog.shots.findIndex((entry) => entry.id === node.id);
        const start = number === 1 ? 0 : Number(widget(node, "cut_at")?.value || 0);
        const next = fullIndex >= 0 ? catalog.shots[fullIndex + 1] : null;
        const end = next
            ? Number(widget(next, "cut_at")?.value || 0)
            : effectiveDuration(catalog);
        text =
            "Shot " +
            number +
            " · " +
            start.toFixed(3) +
            "–" +
            end.toFixed(3) +
            "s";
        setOutputLabel(node, "shot_preview", text);
        color = end > start ? COLORS.ready : COLORS.error;
    } else if (type === DIALOGUE) {
        const local = buildCatalog(chainThrough(node));
        const speakerName = clean(widget(node, "speaker")?.value);
        const speaker = local.speakersByAlias.get(aliasKey(speakerName));
        text =
            "Shot " +
            local.shots.length +
            " · " +
            (speaker ? speaker.id + " · " + speaker.alias : "speaker required");
        color = speakerName ? COLORS.ready : COLORS.warning;
    } else if (type === MERGE) {
        if (catalog.unassignedReferenceIds.size) {
            text =
                "Incomplete · " +
                catalog.unassignedReferenceIds.size +
                " reference relationship(s) unassigned";
            color = COLORS.warning;
        } else if (catalog.endpointConflict) {
            text = "Invalid · endpoint and Ref2VA roles need separate plans";
            color = COLORS.error;
        } else {
            text =
                catalog.mode +
                " · " +
                (catalog.subjects.length + catalog.motionSubjectLabels.size) +
                " Subjects · " +
                catalog.shots.length +
                " Shots";
            color = COLORS.ready;
        }
    } else if (type === ENHANCER) {
        text = "Full-scene context · locked JSON prose";
        color = COLORS.ready;
    } else if (type === APPLY_PROSE) {
        text = "Recompile · validate all locks";
        color = COLORS.ready;
    } else if (type === PROMPT_REVIEW) {
        text = "Prompt-only approval · media stays in plan_context";
        color = COLORS.ready;
    } else if (type === APPLY_REFERENCE) {
        text = "Native H3 handoff · prompt/context pair verified at queue time";
        color = COLORS.ready;
    }
    node.__h3PlanV2Badge = { text, color };
    const planOutput = output(node, "h3_plan");
    if (planOutput) planOutput.label = text;
}

function refreshNode(node) {
    if (!UI_CLASSES.has(className(node))) return;
    setupNode(node);
    const chain = PLAN_CLASSES.has(className(node))
        ? bestContextChain(node)
        : [];
    const catalog = buildCatalog(chain);
    refreshConditionalWidgets(node, catalog);
    refreshBadgeAndOutputs(node, catalog);
    resizeNode(node);
}

function performRefresh() {
    refreshScheduled = false;
    for (const node of app.graph?._nodes || []) refreshNode(node);
    app.graph?.setDirtyCanvas(true, true);
}

function scheduleRefresh() {
    if (refreshScheduled) return;
    refreshScheduled = true;
    queueMicrotask(performRefresh);
}

function drawBadge(node, ctx) {
    const badge = node.__h3PlanV2Badge;
    if (!badge || node.flags?.collapsed) return;
    ctx.save();
    ctx.font = "bold 11px sans-serif";
    const display =
        badge.text.length > 58 ? badge.text.slice(0, 55) + "…" : badge.text;
    const width = Math.min(node.size[0] - 8, ctx.measureText(display).width + 14);
    const x = Math.max(4, node.size[0] - width - 4);
    const y = -20;
    ctx.fillStyle = badge.color;
    ctx.beginPath();
    ctx.roundRect(x, y, width, 17, 5);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(display, x + width / 2, y + 8.5, width - 8);
    ctx.restore();
}

function wrapLifecycle(nodeType, method, callback) {
    const original = nodeType.prototype[method];
    nodeType.prototype[method] = function () {
        const result = original?.apply(this, arguments);
        callback(this);
        return result;
    };
}

app.registerExtension({
    name: "MiniMaxH3Guide.planV2Interaction",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const name = nodeData.name;
        if (UI_CLASSES.has(name)) {
            wrapLifecycle(nodeType, "onNodeCreated", (node) => {
                queueMicrotask(() => {
                    setupNode(node);
                    scheduleRefresh();
                });
            });
            wrapLifecycle(nodeType, "onConfigure", (node) => {
                queueMicrotask(() => {
                    setupNode(node);
                    scheduleRefresh();
                });
            });
            wrapLifecycle(nodeType, "onConnectionsChange", () => scheduleRefresh());
            wrapLifecycle(nodeType, "onModeChange", () => scheduleRefresh());
            wrapLifecycle(nodeType, "onRemoved", () => scheduleRefresh());

            const originalDraw = nodeType.prototype.onDrawForeground;
            nodeType.prototype.onDrawForeground = function (ctx) {
                const result = originalDraw?.apply(this, arguments);
                drawBadge(this, ctx);
                return result;
            };
            return;
        }
        if (name === "Reroute" || name.endsWith("Reroute")) {
            wrapLifecycle(nodeType, "onConnectionsChange", () => scheduleRefresh());
            wrapLifecycle(nodeType, "onModeChange", () => scheduleRefresh());
            wrapLifecycle(nodeType, "onRemoved", () => scheduleRefresh());
        }
    },

    async setup() {
        scheduleRefresh();
    },
});
