#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const COMPOSER = "MiniMaxH3PlanV2KeyframeMotionComposer";
const PROJECT = "MiniMaxH3PlanV2ProjectSetup";
const IDENTITY = "MiniMaxH3PlanV2ImageReference";
const AUDIO = "MiniMaxH3PlanV2AudioReference";
const SHOT = "MiniMaxH3PlanV2Shot";
const KEYFRAME = "MiniMaxH3PlanV2ShotKeyframe";
const MOTION = "MiniMaxH3PlanV2ShotMotionReference";
const MERGE = "MiniMaxH3PlanV2PromptMerge";
const REVIEW = "MiniMaxH3PlanV2PromptReview";
const APPLY = "MiniMaxH3PlanV2ApplyReferencePlan";

const TEMPLATE_GOAL =
    "Animate the supplied opening keyframe as one continuous shot. The identity reference " +
    "defines the character's stable identity and appearance. The opening keyframe defines " +
    "the initial pose, placement, wardrobe, camera, framing, composition, and target visual " +
    "appearance. Transfer only motion and action timing from the structural control video " +
    "to the referenced character, beginning smoothly from the keyframe pose. Preserve " +
    "<Audio 1> as the complete final soundtrack and synchronize visible facial reactions " +
    "to it. Do not copy control-video appearance, identity, environment, camera, framing, " +
    "or control-map style.";

const TEMPLATE_SHOT =
    "Begin in the supplied opening keyframe's pose and composition. <Subject 1> performs " +
    "the motion transferred from <Video 1>, adapted smoothly while preserving the " +
    "keyframe's placement, wardrobe, camera framing, and spatial relationships. Preserve " +
    "<Audio 1> as the complete final soundtrack with unchanged timing and synchronize " +
    "visible facial reactions to it. Keep the action physically coherent and continuous.";

const TEMPLATE_MOTION =
    "Transfer only pose progression, action order, cadence, body mechanics, balance shifts, " +
    "contact timing, and broad facial timing from the structural control video. Begin from " +
    "the opening keyframe pose. Treat line art, depth, or another control-map rendering as " +
    "structural guidance only, never as target appearance. Do not copy source identity, " +
    "anatomy, wardrobe, environment, lighting, camera, framing, colors, or background edges.";

function usage() {
    console.error(
        "Usage: node scripts/compact_keyframe_motion_workflow.mjs " +
        "<expanded-workflow.json> <composer-workflow.json> [--template-safe]",
    );
    process.exit(2);
}

const [, , sourcePath, destinationPath, ...options] = process.argv;
if (!sourcePath || !destinationPath) usage();
const templateSafe = options.includes("--template-safe");

const workflow = JSON.parse(fs.readFileSync(sourcePath, "utf8"));
const findOne = (type) => {
    const matches = workflow.nodes.filter((node) => node.type === type);
    if (matches.length !== 1) {
        throw new Error(`Expected exactly one ${type}; found ${matches.length}.`);
    }
    return matches[0];
};
const inputLink = (node, name) => {
    const input = (node.inputs || []).find((entry) => entry.name === name);
    if (!input || input.link == null) {
        throw new Error(`${node.type}.${name} must be connected.`);
    }
    return Number(input.link);
};
const outputLinks = (node, name) => {
    const output = (node.outputs || []).find((entry) => entry.name === name);
    return (output?.links || []).map(Number);
};

const project = findOne(PROJECT);
const identity = findOne(IDENTITY);
const audio = findOne(AUDIO);
const shot = findOne(SHOT);
const keyframe = findOne(KEYFRAME);
const motion = findOne(MOTION);

const identityLink = inputLink(identity, "image");
const keyframeLink = inputLink(keyframe, "image");
const motionLink = inputLink(motion, "video_frames");
const audioLink = inputLink(audio, "audio");
const planLinks = outputLinks(motion, "h3_plan");
const lengthLinks = outputLinks(project, "h3_length");
if (planLinks.length !== 1) {
    throw new Error("The final motion attachment must connect once to Prompt Merge.");
}

const projectValues = project.widgets_values || [];
const identityValues = identity.widgets_values || [];
const audioValues = audio.widgets_values || [];
const shotValues = shot.widgets_values || [];
const keyframeValues = keyframe.widgets_values || [];
const motionValues = motion.widgets_values || [];

const composer = {
    id: project.id,
    type: COMPOSER,
    pos: [3456, -224],
    size: [640, 768],
    flags: { ...(project.flags || {}) },
    order: project.order,
    mode: project.mode ?? 0,
    inputs: [
        { name: "identity_image", type: "IMAGE", link: identityLink },
        { name: "opening_keyframe", type: "IMAGE", link: keyframeLink },
        { name: "motion_video_frames", type: "IMAGE", link: motionLink },
        { name: "original_audio", type: "AUDIO", link: audioLink },
    ],
    outputs: [
        {
            name: "h3_plan",
            type: "MINIMAX_H3_PLAN_V2",
            slot_index: 0,
            links: planLinks,
            label: "2 Pictures · 1 Video · 1 Audio · Shot 1",
        },
        {
            name: "h3_length",
            type: "INT",
            slot_index: 1,
            links: lengthLinks,
        },
        {
            name: "composer_preview",
            type: "STRING",
            slot_index: 2,
            links: null,
        },
        {
            name: "h3_motion_video",
            type: "IMAGE",
            slot_index: 3,
            links: null,
        },
        {
            name: "h3_audio",
            type: "AUDIO",
            slot_index: 4,
            links: null,
        },
    ],
    properties: {
        aux_id: "ethanfel/ComfyUI-MiniMax-H3-Guide",
        "Node name for S&R": COMPOSER,
    },
    widgets_values: [
        projectValues[1] ?? 345,
        templateSafe ? TEMPLATE_GOAL : projectValues[0],
        projectValues[2] || "cinematic, photorealistic live-action",
        identityValues[4] || "character",
        templateSafe ? TEMPLATE_SHOT : shotValues[1],
        shotValues[2] || "Keep the opening keyframe camera stable.",
        false,
        identityValues[1] || "character identity",
        identityValues[2] || "Stable character identity and appearance reference.",
        audioValues[1] || "original control-video soundtrack",
        keyframeValues[0] || "target opening keyframe",
        keyframeValues[1] || "The exact opening pose, composition, and target appearance.",
        motionValues[1] || "character motion",
        templateSafe ? TEMPLATE_MOTION : motionValues[2],
        motionValues[3] || 24,
    ],
    color: "#244f66",
    bgcolor: "#173444",
    title: "WIP — 1. EDIT: keyframe + motion Plan Composer",
};

const removed = new Set([
    project.id,
    identity.id,
    audio.id,
    shot.id,
    keyframe.id,
    motion.id,
]);
const remappedIds = new Set([
    identityLink,
    keyframeLink,
    motionLink,
    audioLink,
    planLinks[0],
    ...lengthLinks,
]);
const originalLinks = new Map(workflow.links.map((link) => [Number(link[0]), link]));
workflow.links = workflow.links.filter(
    (link) => !removed.has(link[1]) && !removed.has(link[3]),
);

const remap = (linkId, originId, originSlot, targetId, targetSlot, type) => {
    if (!originalLinks.has(linkId)) throw new Error(`Missing source link ${linkId}.`);
    return [linkId, originId, originSlot, targetId, targetSlot, type];
};
for (const linkId of remappedIds) {
    const link = originalLinks.get(linkId);
    if (!link) throw new Error(`Cannot remap missing link ${linkId}.`);
    if (linkId === identityLink) {
        workflow.links.push(remap(linkId, link[1], link[2], composer.id, 0, "IMAGE"));
    } else if (linkId === keyframeLink) {
        workflow.links.push(remap(linkId, link[1], link[2], composer.id, 1, "IMAGE"));
    } else if (linkId === motionLink) {
        workflow.links.push(remap(linkId, link[1], link[2], composer.id, 2, "IMAGE"));
    } else if (linkId === audioLink) {
        workflow.links.push(remap(linkId, link[1], link[2], composer.id, 3, "AUDIO"));
    } else if (linkId === planLinks[0]) {
        workflow.links.push(
            remap(linkId, composer.id, 0, link[3], link[4], "MINIMAX_H3_PLAN_V2"),
        );
    } else {
        workflow.links.push(remap(linkId, composer.id, 1, link[3], link[4], "INT"));
    }
}
workflow.links.sort((left, right) => Number(left[0]) - Number(right[0]));
workflow.nodes = workflow.nodes.filter((node) => !removed.has(node.id));
workflow.nodes.push(composer);

for (const node of workflow.nodes) {
    if (node.type === MERGE) node.title = "2. Compile compact Composer plan";
    if (node.type === REVIEW) node.title = "3. Review compiled roles and prompt";
    if (node.type === APPLY) node.title = "4. Apply native H3 reference plan";
    if (!templateSafe) continue;
    if (node.type === "LoadImage") {
        const isIdentity = outputLinks(node, "IMAGE").includes(identityLink);
        node.widgets_values = [
            isIdentity ? "replace_with_identity.png" : "replace_with_opening_keyframe.png",
            "image",
        ];
    }
    if (node.type === "VHS_LoadVideoPath" && node.widgets_values) {
        node.widgets_values.video = "replace_with_control_video.mp4";
        const preview = node.widgets_values.videopreview?.params;
        if (preview) preview.filename = "replace_with_control_video.mp4";
    }
}

const firstGroup = workflow.groups?.[0];
if (firstGroup) {
    firstGroup.title = "WIP — A. References and compact keyframe + motion Composer";
}
workflow.extra ||= {};
workflow.extra.workflowStatus = "WIP";
workflow.extra.composerSource = "expanded-keyframe-motion-plan-v2";

fs.mkdirSync(path.dirname(destinationPath), { recursive: true });
fs.writeFileSync(destinationPath, `${JSON.stringify(workflow, null, 2)}\n`);
console.log(
    `Wrote ${destinationPath}: ${workflow.nodes.length} nodes, ${workflow.links.length} links.`,
);
