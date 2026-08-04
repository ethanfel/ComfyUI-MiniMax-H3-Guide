import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "MiniMaxH3ReferenceSheet";
const LOAD = "Load existing";
const CREATE = "Create new";
const UPDATE = "Update existing";

function widget(node, name) {
    return (node.widgets || []).find((item) => item.name === name);
}

function setWidgetVisible(target, visible) {
    if (!target) return;
    if (!Object.prototype.hasOwnProperty.call(target, "__h3OriginalComputeSize")) {
        target.__h3OriginalComputeSize = target.computeSize;
    }
    target.hidden = !visible;
    target.serialize = true;
    if (visible) {
        if (target.__h3OriginalComputeSize) target.computeSize = target.__h3OriginalComputeSize;
        else delete target.computeSize;
    } else {
        target.computeSize = () => [0, -4];
    }
}

function setComboValues(target, values) {
    if (!target) return;
    target.options = target.options || {};
    target.options.values = values;
}

function mediaUrl(sheet, asset) {
    const query = new URLSearchParams({
        sheet_id: sheet.id,
        asset_key: asset.key,
        v: asset.sha256.slice(0, 12),
    });
    return api.apiURL(`/minimax_h3/reference_sheets/media?${query}`);
}

function selectAsset(node, kind, key) {
    const target = widget(node, kind === "image" ? "selected_image_key" : "selected_audio_key");
    if (!target) return;
    target.value = key;
    node.graph?.setDirtyCanvas(true, true);
    renderGallery(node);
}

function assetCard(node, sheet, asset) {
    const kind = asset.kind;
    const ordinal = String(asset.key).match(/_(\d+)$/)?.[1];
    const displayName = `${kind === "image" ? "Image" : "Audio"}${ordinal ? ` ${ordinal}` : ""}`;
    const selected = String(
        widget(node, kind === "image" ? "selected_image_key" : "selected_audio_key")?.value || ""
    ) === asset.key;
    const card = document.createElement("button");
    card.type = "button";
    card.title = `Select saved ${displayName.toLowerCase()}`;
    card.style.cssText = [
        "display:flex",
        "flex-direction:column",
        "gap:5px",
        "padding:6px",
        "min-width:0",
        `border:2px solid ${selected ? "#6fb7ff" : "#4b4b4b"}`,
        "border-radius:7px",
        "background:#252525",
        "color:#ddd",
        "cursor:pointer",
        "text-align:left",
    ].join(";");
    card.onclick = () => selectAsset(node, kind, asset.key);

    if (kind === "image") {
        const image = document.createElement("img");
        image.src = mediaUrl(sheet, asset);
        image.alt = displayName;
        image.loading = "lazy";
        image.style.cssText = "width:100%;height:112px;object-fit:contain;background:#111;border-radius:4px";
        card.appendChild(image);
    } else {
        const audio = document.createElement("audio");
        audio.src = mediaUrl(sheet, asset);
        audio.controls = true;
        audio.preload = "metadata";
        audio.style.cssText = "width:100%;height:34px";
        audio.onclick = (event) => event.stopPropagation();
        card.appendChild(audio);
    }

    const label = document.createElement("span");
    label.textContent = selected ? `Selected · ${displayName}` : displayName;
    label.style.cssText = "font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
    card.appendChild(label);
    return card;
}

function currentSheet(node) {
    const selection = String(widget(node, "saved_sheet")?.value || "");
    return (node.__h3SheetCatalog || []).find((sheet) => sheet.option === selection) || null;
}

function renderGallery(node) {
    const root = node.__h3SheetGallery;
    if (!root) return;
    root.replaceChildren();

    if (String(widget(node, "operation")?.value || LOAD) === CREATE) {
        const createHint = document.createElement("div");
        createHint.textContent = "Connect IMAGE/AUDIO inputs and queue once. The saved gallery will appear here automatically.";
        createHint.style.cssText = "padding:14px;color:#aaa;text-align:center;font-size:12px";
        root.appendChild(createHint);
        return;
    }

    const sheet = currentSheet(node);
    if (!sheet) {
        const empty = document.createElement("div");
        empty.textContent = "No saved sheet selected. Connect media and choose Create new, then queue once.";
        empty.style.cssText = "padding:14px;color:#aaa;text-align:center;font-size:12px";
        root.appendChild(empty);
        return;
    }

    const header = document.createElement("div");
    header.style.cssText = "display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:7px";
    const title = document.createElement("strong");
    title.textContent = sheet.name;
    title.style.cssText = "overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
    const refresh = document.createElement("button");
    refresh.type = "button";
    refresh.textContent = "Refresh";
    refresh.title = "Refresh saved sheets and gallery media";
    refresh.onclick = () => refreshCatalog(node);
    header.append(title, refresh);
    root.appendChild(header);

    if (sheet.description) {
        const description = document.createElement("div");
        description.textContent = sheet.description;
        description.style.cssText = "font-size:11px;color:#aaa;margin-bottom:8px;white-space:normal";
        root.appendChild(description);
    }

    const images = sheet.assets.filter((asset) => asset.kind === "image");
    const audios = sheet.assets.filter((asset) => asset.kind === "audio");
    for (const [kind, assets] of [["image", images], ["audio", audios]]) {
        if (!assets.length) continue;
        const selectedWidget = widget(node, kind === "image" ? "selected_image_key" : "selected_audio_key");
        if (!assets.some((asset) => asset.key === selectedWidget?.value)) {
            selectedWidget.value = assets[0].key;
        }
        const section = document.createElement("div");
        section.style.cssText = kind === "image"
            ? "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;margin-bottom:8px"
            : "display:flex;flex-direction:column;gap:6px";
        for (const asset of assets) section.appendChild(assetCard(node, sheet, asset));
        root.appendChild(section);
    }
}

async function refreshCatalog(node, preferredSelection = "") {
    if (!node) return;
    if (node.__h3SheetRefreshing) {
        if (preferredSelection) node.__h3PendingSheetSelection = preferredSelection;
        return;
    }
    node.__h3SheetRefreshing = true;
    try {
        const response = await api.fetchApi("/minimax_h3/reference_sheets/catalog", { cache: "no-store" });
        if (!response.ok) throw new Error(`catalog request failed (${response.status})`);
        const payload = await response.json();
        const sheets = Array.isArray(payload.sheets) ? payload.sheets : [];
        node.__h3SheetCatalog = sheets;
        const saved = widget(node, "saved_sheet");
        const choices = sheets.map((sheet) => sheet.option);
        if (!choices.length) choices.push("(no saved reference sheets)");
        setComboValues(saved, choices);
        const wanted = preferredSelection || String(saved?.value || "");
        if (choices.includes(wanted)) saved.value = wanted;
        else if (sheets.length) saved.value = sheets[0].option;
        renderGallery(node);
        node.setSize?.([Math.max(node.size?.[0] || 0, 360), node.computeSize?.()[1] || node.size?.[1] || 260]);
    } catch (error) {
        if (node.__h3SheetGallery) {
            node.__h3SheetGallery.textContent = `Reference Sheet gallery unavailable: ${error.message}`;
        }
    } finally {
        node.__h3SheetRefreshing = false;
        const pending = node.__h3PendingSheetSelection;
        node.__h3PendingSheetSelection = "";
        if (pending) queueMicrotask(() => refreshCatalog(node, pending));
    }
}

function refreshMode(node) {
    const mode = String(widget(node, "operation")?.value || LOAD);
    setWidgetVisible(widget(node, "saved_sheet"), mode !== CREATE);
    setWidgetVisible(widget(node, "sheet_name"), mode !== LOAD);
    setWidgetVisible(widget(node, "sheet_description"), mode !== LOAD);
    setWidgetVisible(widget(node, "tags"), mode !== LOAD);
    setWidgetVisible(widget(node, "confirm_update"), mode === UPDATE);
    setWidgetVisible(widget(node, "selected_image_key"), false);
    setWidgetVisible(widget(node, "selected_audio_key"), false);
    renderGallery(node);
    node.setSize?.([Math.max(node.size?.[0] || 0, 360), node.computeSize?.()[1] || node.size?.[1] || 260]);
}

function setupNode(node) {
    if (!node.__h3SheetGallery) {
        const root = document.createElement("div");
        root.style.cssText = [
            "box-sizing:border-box",
            "width:100%",
            "min-height:155px",
            "padding:8px",
            "overflow:auto",
            "background:#1b1b1b",
            "border:1px solid #444",
            "border-radius:6px",
            "color:#ddd",
        ].join(";");
        node.__h3SheetGallery = root;
        node.addDOMWidget("reference_sheet_gallery", "div", root, {
            serialize: false,
            getMinHeight: () => 180,
        });
    }

    const operation = widget(node, "operation");
    if (operation && !operation.__h3SheetCallback) {
        const original = operation.callback;
        operation.callback = function () {
            original?.apply(this, arguments);
            queueMicrotask(() => refreshMode(node));
        };
        operation.__h3SheetCallback = true;
    }
    const saved = widget(node, "saved_sheet");
    if (saved && !saved.__h3SheetCallback) {
        const original = saved.callback;
        saved.callback = function () {
            original?.apply(this, arguments);
            queueMicrotask(() => renderGallery(node));
        };
        saved.__h3SheetCallback = true;
    }

    refreshMode(node);
    refreshCatalog(node);
}

app.registerExtension({
    name: "MiniMaxH3Guide.referenceSheet",

    async setup() {
        api.addEventListener("executed", ({ detail }) => {
            const node = app.graph?.getNodeById?.(detail?.display_node ?? detail?.node);
            if (!node || (node.comfyClass !== NODE_CLASS && node.type !== NODE_CLASS)) return;
            const selection = detail?.output?.saved_sheet;
            const preferred = Array.isArray(selection) ? selection[0] : selection;
            const operation = widget(node, "operation");
            if (preferred && operation && operation.value !== LOAD) {
                operation.value = LOAD;
                refreshMode(node);
            }
            refreshCatalog(node, String(preferred || ""));
        });
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            queueMicrotask(() => setupNode(this));
            return result;
        };

        const originalConfigured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigured?.apply(this, arguments);
            queueMicrotask(() => setupNode(this));
            return result;
        };
    },
});
