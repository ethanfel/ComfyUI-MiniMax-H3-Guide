import { app } from "../../scripts/app.js";

const NODE_CLASS = "MiniMaxH3EnhancerVisualReference";
const PICTURE = "Picture";
const UNASSIGNED_ROLE = "Unassigned - choose a reference role";
const FIRST_FRAME = "Exact first frame (I2VA or FL2VA)";
const LAST_FRAME = "Exact last frame (L2VA or FL2VA)";

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

function graphLink(graph, linkId) {
    return graph?.links?.[linkId] || graph?.links?.get?.(linkId) || null;
}

function isReroute(node) {
    const type = String(node?.comfyClass || node?.type || node?.title || "");
    return type === "Reroute" || type.endsWith("Reroute") || node?.isVirtualNode === true;
}

function passthroughInput(node, outputType) {
    const connected = (node?.inputs || []).filter((item) => item.link != null);
    return connected.find((item) => item.type === outputType) || connected[0] || null;
}

function upstreamReference(graph, linkId, visited = new Set()) {
    const link = graphLink(graph, linkId);
    if (!link) return null;
    const origin = graph?.getNodeById?.(link.origin_id);
    if (!origin || visited.has(origin.id)) return null;
    visited.add(origin.id);

    if (isReferenceNode(origin)) {
        // ComfyUI mode 4 is bypass. A bypassed reference contributes neither
        // media nor numbering, so continue through its context input.
        if (origin.mode !== 4) return origin;
        const previous = (origin.inputs || []).find(
            (item) => item.name === "previous_context"
        );
        return previous?.link == null
            ? null
            : upstreamReference(graph, previous.link, visited);
    }

    if (isReroute(origin) || origin.mode === 4) {
        const outputType = origin.outputs?.[link.origin_slot]?.type || link.type;
        const input = passthroughInput(origin, outputType);
        return input?.link == null
            ? null
            : upstreamReference(graph, input.link, visited);
    }
    return null;
}

function previousReference(node) {
    const input = (node.inputs || []).find((item) => item.name === "previous_context");
    if (input?.link == null || !node.graph) return null;
    return upstreamReference(node.graph, input.link);
}

function routeFor(node) {
    const roleInput = (node.inputs || []).find((item) => item.name === "role_bindings");
    if (roleInput?.link != null) return "h3_media · see routing_report";

    const role = String(widget(node, "reference_role")?.value || "");
    if (role === UNASSIGNED_ROLE) return "choose reference role";
    if (role === FIRST_FRAME) return "first_frame";
    if (role === LAST_FRAME) return "last_frame";

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
            const referenceRole = widget(this, "reference_role");
            if (referenceRole && !referenceRole.__h3RouteCallback) {
                const originalCallback = referenceRole.callback;
                referenceRole.callback = function () {
                    originalCallback?.apply(this, arguments);
                    queueMicrotask(refreshAll);
                };
                referenceRole.__h3RouteCallback = true;
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

        const originalModeChange = nodeType.prototype.onModeChange;
        nodeType.prototype.onModeChange = function () {
            originalModeChange?.apply(this, arguments);
            queueMicrotask(refreshAll);
        };

        const originalRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            originalRemoved?.apply(this, arguments);
            queueMicrotask(refreshAll);
        };
    },
});
