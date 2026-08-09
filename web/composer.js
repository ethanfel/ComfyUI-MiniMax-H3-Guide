import { app } from "../../scripts/app.js";

const KEYFRAME_MOTION_COMPOSER = "MiniMaxH3PlanV2KeyframeMotionComposer";

const ADVANCED_WIDGETS = [
    "identity_reference_name",
    "identity_description",
    "audio_reference_name",
    "keyframe_reference_name",
    "keyframe_description",
    "motion_reference_name",
    "motion_description",
    "source_fps",
];

function widget(node, name) {
    return (node?.widgets || []).find((item) => item.name === name);
}

function clean(value, fallback = "") {
    const text = String(value ?? "").trim();
    return text || fallback;
}

function setWidgetVisible(target, visible) {
    if (!target) return;
    if (!Object.prototype.hasOwnProperty.call(target, "__h3ComposerComputeSize")) {
        target.__h3ComposerComputeSize = target.computeSize;
    }
    target.hidden = !visible;
    target.serialize = true;
    if (target.element) target.element.style.display = visible ? "" : "none";
    if (visible) {
        if (target.__h3ComposerComputeSize) {
            target.computeSize = target.__h3ComposerComputeSize;
        } else {
            delete target.computeSize;
        }
    } else {
        target.computeSize = () => [0, -4];
    }
}

function roleRow(label, value, color) {
    const row = document.createElement("div");
    row.className = "h3-composer-role";

    const badge = document.createElement("span");
    badge.className = "h3-composer-role-badge";
    badge.style.setProperty("--h3-composer-role-color", color);
    badge.textContent = label;

    const text = document.createElement("span");
    text.className = "h3-composer-role-text";
    text.textContent = value;
    row.append(badge, text);
    return { row, text };
}

function addRoleSummary(node) {
    if (!node?.addDOMWidget || node.__h3ComposerSummary) return;

    const wrap = document.createElement("div");
    wrap.className = "h3-composer-summary";
    const heading = document.createElement("div");
    heading.className = "h3-composer-heading";
    heading.textContent = "Resolved Plan v2 roles";

    const identity = roleRow("Picture 1", "identity → Subject 1", "#69aee8");
    const keyframe = roleRow("Picture 2", "Shot 1 opening frame", "#8cbfe8");
    const motion = roleRow("Video 1", "motion only → Subject 1", "#79b995");
    const audio = roleRow("Audio 1", "complete final soundtrack", "#d5b86b");
    wrap.append(
        heading,
        identity.row,
        keyframe.row,
        motion.row,
        audio.row,
    );

    const summaryWidget = node.addDOMWidget(
        "h3_composer_role_summary",
        "h3_composer_role_summary",
        wrap,
        {
            serialize: false,
            getValue: () => "",
            setValue: () => {},
        },
    );
    summaryWidget.serialize = false;
    summaryWidget.computeSize = (width) => [width, 126];

    const summaryIndex = node.widgets?.indexOf(summaryWidget) ?? -1;
    if (summaryIndex > 0) {
        node.widgets.splice(summaryIndex, 1);
        node.widgets.unshift(summaryWidget);
    }
    node.__h3ComposerSummary = { wrap, identity, keyframe, motion, audio };
}

function refreshSummary(node) {
    const summary = node.__h3ComposerSummary;
    if (!summary) return;
    const subject = clean(widget(node, "subject_name")?.value, "character");
    summary.identity.text.textContent = `identity → ${subject} / <Subject 1>`;
    summary.motion.text.textContent = `motion only → ${subject} / <Subject 1>`;
}

function refreshComposer(node) {
    const advanced = Boolean(widget(node, "advanced_options")?.value);
    for (const name of ADVANCED_WIDGETS) {
        setWidgetVisible(widget(node, name), advanced);
    }
    refreshSummary(node);

    const computed = node.computeSize?.() || node.size || [560, 640];
    const width = Math.max(560, Number(node.size?.[0] || 0), Number(computed[0] || 0));
    const height = Math.max(420, Number(computed[1] || 0));
    node.setSize?.([width, height]);
    node.graph?.setDirtyCanvas?.(true, true);
}

function bindRefresh(node, name) {
    const target = widget(node, name);
    if (!target || target.__h3ComposerRefreshBound) return;
    target.__h3ComposerRefreshBound = true;
    const original = target.callback;
    target.callback = function h3ComposerRefresh(value) {
        const result = original?.apply(this, arguments);
        requestAnimationFrame(() => refreshComposer(node));
        return result;
    };
}

function installStyles() {
    if (document.getElementById("h3-composer-styles")) return;
    const style = document.createElement("style");
    style.id = "h3-composer-styles";
    style.textContent = `
      .h3-composer-summary {
        box-sizing: border-box; display: grid; grid-template-columns: 1fr 1fr;
        gap: 5px 8px; width: 100%; padding: 8px 10px; border-radius: 7px;
        background: rgba(17, 22, 27, .72); border: 1px solid rgba(129, 169, 202, .24);
        color: rgba(238, 244, 248, .94); font: 12px/1.25 sans-serif;
      }
      .h3-composer-heading {
        grid-column: 1 / -1; color: rgba(207, 226, 240, .82); font-weight: 600;
        letter-spacing: .02em; margin-bottom: 2px;
      }
      .h3-composer-role { display: flex; align-items: center; min-width: 0; gap: 6px; }
      .h3-composer-role-badge {
        flex: 0 0 auto; padding: 2px 5px; border-radius: 4px;
        background: color-mix(in srgb, var(--h3-composer-role-color) 22%, transparent);
        color: var(--h3-composer-role-color); font-weight: 600;
      }
      .h3-composer-role-text {
        min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        color: rgba(235, 240, 244, .82);
      }
    `;
    document.head.append(style);
}

app.registerExtension({
    name: "MiniMaxH3Guide.KeyframeMotionComposer",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== KEYFRAME_MOTION_COMPOSER) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreatedComposer() {
            const result = originalCreated?.apply(this, arguments);
            this.title = this.title || "Keyframe + Motion Composer";
            addRoleSummary(this);
            bindRefresh(this, "advanced_options");
            bindRefresh(this, "subject_name");
            requestAnimationFrame(() => refreshComposer(this));
            return result;
        };

        const originalConfigured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function onConfigureComposer() {
            const result = originalConfigured?.apply(this, arguments);
            requestAnimationFrame(() => refreshComposer(this));
            return result;
        };
    },

    setup() {
        installStyles();
    },
});
