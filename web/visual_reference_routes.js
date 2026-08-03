import { app } from "../../scripts/app.js";

const NODE_CLASS = "MiniMaxH3EnhancerVisualReference";
const PICTURE = "Picture";

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
        if (target.__h3OriginalComputeSize) {
            target.computeSize = target.__h3OriginalComputeSize;
        } else {
            delete target.computeSize;
        }
    } else {
        target.computeSize = () => [0, -4];
    }
}

function isReferenceNode(node) {
    return node?.comfyClass === NODE_CLASS || node?.type === NODE_CLASS;
}

function previousReference(node) {
    const input = (node.inputs || []).find((item) => item.name === "previous_context");
    if (input?.link == null || !node.graph) return null;
    const link = node.graph.links?.[input.link] || node.graph.links?.get?.(input.link);
    if (!link) return null;
    const previous = node.graph.getNodeById(link.origin_id);
    return isReferenceNode(previous) ? previous : null;
}

function routeFor(node) {
    const mediaType = String(widget(node, "media_type")?.value || PICTURE);
    let number = 1;
    let previous = previousReference(node);
    const visited = new Set([node.id]);
    while (previous && !visited.has(previous.id)) {
        visited.add(previous.id);
        const previousType = String(widget(previous, "media_type")?.value || PICTURE);
        if (previousType === mediaType) number += 1;
        previous = previousReference(previous);
    }
    return mediaType === PICTURE ? `ref_image_${number - 1}` : `ref_video_${number - 1}`;
}

function refreshNode(node) {
    if (!isReferenceNode(node)) return;
    const isVideo = String(widget(node, "media_type")?.value || PICTURE) !== PICTURE;
    for (const name of ["source_fps", "analysis_fps", "max_analysis_frames"]) {
        setWidgetVisible(widget(node, name), isVideo);
    }
    if (node.outputs?.[1]) node.outputs[1].label = routeFor(node);
    node.setSize([node.size[0], node.computeSize()[1]]);
}

function refreshAll() {
    for (const node of app.graph?._nodes || []) refreshNode(node);
    app.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "MiniMaxH3Guide.visualReferenceRoutes",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalCreated?.apply(this, arguments);
            const mediaType = widget(this, "media_type");
            if (mediaType && !mediaType.__h3RouteCallback) {
                const originalCallback = mediaType.callback;
                mediaType.callback = function () {
                    originalCallback?.apply(this, arguments);
                    queueMicrotask(refreshAll);
                };
                mediaType.__h3RouteCallback = true;
            }
            queueMicrotask(refreshAll);
        };

        const originalConfigured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            originalConfigured?.apply(this, arguments);
            queueMicrotask(refreshAll);
        };

        const originalConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            originalConnectionsChange?.apply(this, arguments);
            queueMicrotask(refreshAll);
        };

        const originalRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            originalRemoved?.apply(this, arguments);
            queueMicrotask(refreshAll);
        };
    },
});
