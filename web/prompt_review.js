import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "MiniMaxH3PlanV2PromptReview";
const ROUTE = "/minimax_h3/prompt_review/decision";
const RECOVER_ROUTE = "/minimax_h3/prompt_review/recover";
const PAUSE_MODE = "Pause for approval";
const PASS_THROUGH_MODE = "Pass through without pausing";
const SETTINGS_PROPERTY = "minimax_h3_prompt_review_settings";
const UI_STATE_ID_PROPERTY = "minimax_h3_prompt_review_ui_state_id";
const DEFAULT_HISTORY_LIMIT = 20;
const MIN_HISTORY_LIMIT = 1;
const MAX_HISTORY_LIMIT = 50;
const MAX_CACHED_UI_STATES = 32;
const MIN_WIDTH = 540;
const MIN_EDITOR_HEIGHT = 300;
const reviewStateCache = new Map();
let recoveryRequest = null;
let recoveryTimer = null;
let recoveryAgain = false;

function widget(node, name) {
    return (node.widgets || []).find((entry) => entry.name === name);
}

function validReviewMode(value) {
    return value === PAUSE_MODE || value === PASS_THROUGH_MODE;
}

function validHistoryLimit(value) {
    return Number.isInteger(value) && value >= MIN_HISTORY_LIMIT && value <= MAX_HISTORY_LIMIT;
}

function currentSettings(node) {
    const mode = widget(node, "review_mode")?.value;
    const limit = Number(widget(node, "history_limit")?.value);
    return {
        review_mode: validReviewMode(mode) ? mode : PAUSE_MODE,
        history_limit: validHistoryLimit(limit) ? limit : DEFAULT_HISTORY_LIMIT,
    };
}

function configuredSettings(node, info) {
    const propertySettings = info?.properties?.[SETTINGS_PROPERTY];
    const namedSettings = info?.widgets_values_named;
    const values = Array.isArray(info?.widgets_values) ? info.widgets_values : [];
    const legacyMode = values.find((value) => validReviewMode(value));
    const legacyLimit = values.map(Number).find((value) => validHistoryLimit(value));
    const live = currentSettings(node);
    const modeCandidates = [
        namedSettings?.review_mode,
        propertySettings?.review_mode,
        legacyMode,
        live.review_mode,
    ];
    const limitCandidates = [
        Number(namedSettings?.history_limit),
        Number(propertySettings?.history_limit),
        legacyLimit,
        live.history_limit,
    ];
    return {
        review_mode: modeCandidates.find((value) => validReviewMode(value)) || PAUSE_MODE,
        history_limit:
            limitCandidates.find((value) => validHistoryLimit(value)) || DEFAULT_HISTORY_LIMIT,
    };
}

function persistSettings(node, settings = currentSettings(node)) {
    node.properties ||= {};
    node.properties[SETTINGS_PROPERTY] = { ...settings };
    return settings;
}

function applySettings(node, settings) {
    const mode = widget(node, "review_mode");
    const limit = widget(node, "history_limit");
    if (mode) mode.value = settings.review_mode;
    if (limit) limit.value = settings.history_limit;
    persistSettings(node, settings);
}

function trackSettings(node) {
    for (const name of ["review_mode", "history_limit"]) {
        const target = widget(node, name);
        if (!target || target.__h3PromptReviewTracked) continue;
        target.__h3PromptReviewTracked = true;
        const callback = target.callback;
        target.callback = function () {
            const result = callback?.apply(this, arguments);
            persistSettings(node);
            return result;
        };
    }
    persistSettings(node);
}

function serializeSettings(node, serialized) {
    const settings = persistSettings(node);
    serialized.properties ||= {};
    serialized.properties[SETTINGS_PROPERTY] = { ...settings };

    // ComfyUI 1.48 writes `serialize === false` widgets at their full array
    // position but restores the remaining widgets from a compacted array. A DOM
    // editor before/between real widgets therefore shifts values on workflow-tab
    // recreation. Emit both compact legacy values and named values explicitly.
    serialized.widgets_values = [settings.review_mode, settings.history_limit];
    serialized.widgets_values_named = {
        review_mode: settings.review_mode,
        history_limit: settings.history_limit,
    };
}

function makeReviewStateId() {
    const random = globalThis.crypto?.randomUUID?.().replaceAll("-", "");
    return "review-ui-" + (random || `${Date.now()}-${Math.random().toString(16).slice(2)}`);
}

function reviewStateId(node) {
    node.properties ||= {};
    let stateId = String(node.properties[UI_STATE_ID_PROPERTY] || "").trim();
    const cached = stateId ? reviewStateCache.get(stateId) : null;
    // A copied node inherits arbitrary LiteGraph properties. Give the copy its
    // own editor cache while allowing a recreated workflow-tab node (same id)
    // to recover the original cache entry.
    if (!stateId || (cached && cached.nodeId !== String(node.id))) {
        stateId = makeReviewStateId();
        node.properties[UI_STATE_ID_PROPERTY] = stateId;
    }
    return stateId;
}

function trimReviewStateCache() {
    while (reviewStateCache.size > MAX_CACHED_UI_STATES) {
        reviewStateCache.delete(reviewStateCache.keys().next().value);
    }
}

function cacheReviewState(node) {
    const state = node.__h3PromptReview;
    if (!state) return;
    const stateId = reviewStateId(node);
    const snapshot = {
        nodeId: String(node.id),
        areaValue: state.area.value,
        inputPrompt: state.inputPrompt,
        // `populateHistory` accepts the backend's oldest-to-newest order and
        // reverses it for display. Store that canonical order in the cache so
        // repeated workflow-tab reconstruction cannot flip the dropdown.
        entries: state.entries.slice().reverse().map((entry) => ({ ...entry })),
        historyValue: state.history.value,
        statusText: state.status.textContent,
        statusKind: state.status.dataset.kind || "idle",
        runId: state.runId,
        token: state.token,
        resolvedToken: state.resolvedToken,
        active: state.active,
        controlsDisabled: state.approve.disabled,
    };
    // Refresh insertion order so recently used workflow tabs are evicted last.
    reviewStateCache.delete(stateId);
    reviewStateCache.set(stateId, snapshot);
    trimReviewStateCache();
}

function restoreCachedReviewState(node) {
    const stateId = reviewStateId(node);
    const cached = reviewStateCache.get(stateId);
    const state = node.__h3PromptReview;
    if (!cached || !state) return false;

    state.inputPrompt = String(cached.inputPrompt || "");
    state.area.value = String(cached.areaValue || "");
    state.runId = cached.runId || null;
    state.token = cached.token || null;
    state.resolvedToken = cached.resolvedToken || null;
    state.active = Boolean(cached.active);
    populateHistory(node, cached.entries || []);
    const selected = String(cached.historyValue || "");
    if ([...state.history.options].some((option) => option.value === selected)) {
        state.history.value = selected;
    }
    setButtonsDisabled(node, Boolean(cached.controlsDisabled));
    if (state.active && !cached.controlsDisabled) {
        updateDirtyState(node);
    } else {
        setStatus(node, String(cached.statusText || "Waiting for a queued prompt"), cached.statusKind);
    }
    return true;
}

function serializeReviewState(node, serialized) {
    cacheReviewState(node);
    serialized.properties ||= {};
    serialized.properties[UI_STATE_ID_PROPERTY] = reviewStateId(node);
}

async function sendDecision(node, action, prompt = "") {
    const state = node.__h3PromptReview;
    if (!state?.runId || !state?.token) throw new Error("No prompt review is active.");
    const response = await api.fetchApi(ROUTE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            id: state.runId,
            token: state.token,
            action,
            prompt,
        }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `Review request failed (${response.status}).`);
    return payload;
}

function setStatus(node, message, kind = "idle") {
    const state = node.__h3PromptReview;
    if (!state) return;
    state.status.textContent = message;
    state.status.dataset.kind = kind;
    node.setDirtyCanvas?.(true, true);
}

function setButtonsDisabled(node, disabled) {
    const state = node.__h3PromptReview;
    if (!state) return;
    for (const button of [state.approve, state.restore, state.reject]) {
        button.disabled = disabled;
    }
    state.history.disabled = disabled || !state.entries.length;
}

function historyLabel(entry) {
    const revision = Number(entry.revision || 0);
    const date = String(entry.approved_at || "").replace("T", " ").replace(/\+00:00$/, "Z");
    const mode = entry.edited ? "edited" : "unchanged";
    return `r${revision} · ${mode}${date ? " · " + date : ""}`;
}

function populateHistory(node, entries) {
    const state = node.__h3PromptReview;
    state.entries = Array.isArray(entries) ? entries.slice().reverse() : [];
    state.history.replaceChildren();
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = state.entries.length
        ? `History (${state.entries.length}) — choose a revision to load`
        : "No approved prompt history yet";
    state.history.appendChild(placeholder);
    state.entries.forEach((entry, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = historyLabel(entry);
        state.history.appendChild(option);
    });
    state.history.disabled = !state.entries.length;
}

function updateDirtyState(node) {
    const state = node.__h3PromptReview;
    if (!state || !state.active) return;
    const edited = state.area.value !== state.inputPrompt;
    state.approve.textContent = edited ? "Approve edited prompt & continue" : "Approve prompt & continue";
    setStatus(
        node,
        edited
            ? "Manual edits pending structural validation"
            : "Generation is paused; review the prompt, then approve",
        edited ? "editing" : "paused"
    );
}

function syncSize(node) {
    if (!node.__h3ReviewWidget) return;
    node.__h3ReviewWidget.width = Math.max(node.size?.[0] || 0, MIN_WIDTH);
}

function findReviewNode(displayId) {
    let node = app.graph?.getNodeById?.(displayId);
    if (!node && /^\d+$/.test(String(displayId || ""))) {
        node = app.graph?.getNodeById?.(Number(displayId));
    }
    if (!node || String(node.comfyClass || node.type) !== NODE) return null;
    return node;
}

function activateReview(data) {
    const node = findReviewNode(data?.display_id);
    if (!node || !data?.id || !data?.token) return false;
    setupNode(node);
    const state = node.__h3PromptReview;
    const runId = String(data.id);
    const token = String(data.token);
    if (state.resolvedToken === token) return false;
    if (state.active && state.runId === runId && state.token === token) {
        // Recovery can duplicate the original socket event. Preserve any
        // manual text already typed into the active editor.
        return true;
    }
    state.runId = runId;
    state.token = token;
    state.inputPrompt = String(data.prompt || "");
    state.area.value = state.inputPrompt;
    state.active = true;
    populateHistory(node, data.history || []);
    setButtonsDisabled(node, false);
    updateDirtyState(node);
    cacheReviewState(node);
    state.area.focus();
    return true;
}

async function recoverActiveReviews() {
    if (recoveryRequest) {
        recoveryAgain = true;
        return recoveryRequest;
    }
    recoveryRequest = api.fetchApi(RECOVER_ROUTE, { method: "POST" })
        .catch(() => null)
        .finally(() => {
            recoveryRequest = null;
            if (recoveryAgain) {
                recoveryAgain = false;
                scheduleRecovery();
            }
        });
    return recoveryRequest;
}

function scheduleRecovery() {
    clearTimeout(recoveryTimer);
    recoveryTimer = setTimeout(() => recoverActiveReviews(), 100);
}

function setupNode(node) {
    if (node.__h3PromptReview) return;
    reviewStateId(node);

    const root = document.createElement("div");
    root.className = "h3-review-root";

    const area = document.createElement("textarea");
    area.className = "h3-review-editor";
    area.placeholder = "Queue the workflow to review the compiled or enhanced H3 prompt…";
    area.spellcheck = true;
    area.addEventListener("keydown", (event) => event.stopPropagation());

    const history = document.createElement("select");
    history.className = "h3-review-history";

    const controls = document.createElement("div");
    controls.className = "h3-review-controls";
    const approve = document.createElement("button");
    approve.className = "h3-review-approve";
    approve.textContent = "Approve prompt & continue";
    const restore = document.createElement("button");
    restore.className = "h3-review-restore";
    restore.textContent = "Restore input";
    const reject = document.createElement("button");
    reject.className = "h3-review-reject";
    reject.textContent = "Reject run";
    controls.append(approve, restore, reject);

    const status = document.createElement("div");
    status.className = "h3-review-status";
    status.textContent = "Waiting for a queued prompt";
    root.append(area, history, controls, status);

    node.__h3PromptReview = {
        root,
        area,
        history,
        approve,
        restore,
        reject,
        status,
        entries: [],
        inputPrompt: "",
        runId: null,
        token: null,
        resolvedToken: null,
        active: false,
    };
    node.__h3ReviewWidget = node.addDOMWidget("h3_prompt_review_editor", "div", root, {
        serialize: false,
        getMinHeight: () => MIN_EDITOR_HEIGHT + 104,
    });
    // `options.serialize` and the widget-level serialization flag are distinct
    // in LiteGraph. Set the latter explicitly so this display-only editor can
    // never be mistaken for a backend input.
    node.__h3ReviewWidget.serialize = false;
    trackSettings(node);

    area.addEventListener("input", () => {
        updateDirtyState(node);
        cacheReviewState(node);
    });
    history.addEventListener("change", () => {
        if (!history.value) return;
        const index = Number(history.value);
        if (!Number.isInteger(index) || !node.__h3PromptReview.entries[index]) return;
        area.value = String(node.__h3PromptReview.entries[index].prompt || "");
        updateDirtyState(node);
        cacheReviewState(node);
        area.focus();
    });
    restore.addEventListener("click", () => {
        area.value = node.__h3PromptReview.inputPrompt;
        history.value = "";
        updateDirtyState(node);
        cacheReviewState(node);
        area.focus();
    });
    approve.addEventListener("click", async () => {
        setButtonsDisabled(node, true);
        setStatus(node, "Validating prompt locks…", "working");
        cacheReviewState(node);
        try {
            await sendDecision(node, "approve", area.value);
            node.__h3PromptReview.active = false;
            node.__h3PromptReview.resolvedToken = node.__h3PromptReview.token;
            setStatus(node, "Approved — downstream H3 generation is resuming", "approved");
            cacheReviewState(node);
        } catch (error) {
            setButtonsDisabled(node, false);
            setStatus(node, error.message || String(error), "error");
            cacheReviewState(node);
        }
    });
    reject.addEventListener("click", async () => {
        setButtonsDisabled(node, true);
        setStatus(node, "Rejecting queued run…", "working");
        cacheReviewState(node);
        try {
            await sendDecision(node, "reject");
            node.__h3PromptReview.active = false;
            node.__h3PromptReview.resolvedToken = node.__h3PromptReview.token;
            setStatus(node, "Run rejected; no H3 generation was started", "rejected");
            cacheReviewState(node);
        } catch (error) {
            setButtonsDisabled(node, false);
            setStatus(node, error.message || String(error), "error");
            cacheReviewState(node);
        }
    });

    populateHistory(node, []);
    setButtonsDisabled(node, true);
    restoreCachedReviewState(node);
    const onResize = node.onResize;
    node.onResize = function () {
        const result = onResize?.apply(this, arguments);
        syncSize(node);
        return result;
    };
    node.setSize([
        Math.max(node.size?.[0] || 0, MIN_WIDTH),
        Math.max(node.size?.[1] || 0, node.computeSize()[1]),
    ]);
    syncSize(node);
}

function injectStyles() {
    if (document.getElementById("minimax-h3-prompt-review-styles")) return;
    const style = document.createElement("style");
    style.id = "minimax-h3-prompt-review-styles";
    style.textContent = `
      .h3-review-root { box-sizing:border-box; width:100%; height:100%; min-height:0;
        padding:5px 8px 8px; display:flex; flex-direction:column; gap:7px; }
      .h3-review-editor { box-sizing:border-box; width:100%; flex:1 1 auto; min-height:${MIN_EDITOR_HEIGHT}px;
        resize:none; padding:9px; border:1px solid #555; border-radius:6px; background:#17191b;
        color:#eee; font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace; white-space:pre-wrap; }
      .h3-review-editor:focus { border-color:#4c86ad; outline:none; }
      .h3-review-history { width:100%; min-height:28px; box-sizing:border-box; background:#24272a;
        color:#ddd; border:1px solid #555; border-radius:5px; padding:3px 6px; }
      .h3-review-controls { display:flex; flex-wrap:wrap; gap:6px; }
      .h3-review-controls button { min-height:30px; padding:4px 10px; border-radius:5px;
        border:1px solid #555; color:#fff; cursor:pointer; }
      .h3-review-controls button:disabled { opacity:.45; cursor:default; }
      .h3-review-approve { flex:1 1 240px; background:#267547; }
      .h3-review-restore { background:#3e566b; }
      .h3-review-reject { background:#7b3131; }
      .h3-review-status { min-height:18px; font-size:11px; color:#aaa; overflow-wrap:anywhere; }
      .h3-review-status[data-kind="paused"] { color:#e2bd63; }
      .h3-review-status[data-kind="editing"] { color:#72b7df; }
      .h3-review-status[data-kind="approved"] { color:#72cf94; }
      .h3-review-status[data-kind="error"], .h3-review-status[data-kind="rejected"] { color:#ef8585; }
    `;
    document.head.appendChild(style);
}

app.registerExtension({
    name: "MiniMaxH3Guide.promptReview",

    setup() {
        injectStyles();
        api.addEventListener("minimax-h3-prompt-review", (event) => {
            activateReview(event.detail || {});
        });
        // Initial WebSocket status includes a session id. ComfyUI sends it
        // again after reconnect, so ask the server to re-send any active
        // review through that same execution client's WebSocket.
        api.addEventListener("status", (event) => {
            if (event.detail?.sid) scheduleRecovery();
        });
        scheduleRecovery();
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            queueMicrotask(() => {
                setupNode(this);
                scheduleRecovery();
            });
            return result;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const settings = configuredSettings(this, arguments[0]);
            const result = onConfigure?.apply(this, arguments);
            queueMicrotask(() => {
                setupNode(this);
                applySettings(this, settings);
                trackSettings(this);
                const mode = widget(this, "review_mode")?.value;
                if (mode !== PAUSE_MODE) {
                    setStatus(this, "Pass-through mode: queued prompts will not pause", "idle");
                }
                scheduleRecovery();
            });
            return result;
        };
        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (serialized) {
            const result = onSerialize?.apply(this, arguments);
            serializeReviewState(this, serialized);
            serializeSettings(this, serialized);
            return result;
        };
    },
});
