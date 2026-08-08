# MiniMax H3 Practical Reference Media Guide

This document records practical guidance for preparing scene-composition and
motion references for MiniMax H3. It complements MiniMax's formal prompt format
guide; it is not a claim that rough references guarantee exact geometry or
motion.

## 1. Choose the reference role by intent

| Desired result | Reference role | Typical retention |
| --- | --- | --- |
| Broad layout, blocking, viewpoint, or shot order | `Storyboard or shot planning` | `weak_reference` |
| A composition that should be followed closely | `Concrete Ref2VA keyframe or composition anchor` | `fully_preserved` |
| The literal opening or ending frame | `Exact first frame` or `Exact last frame` | `fully_preserved` |
| Reuse a person's or object's appearance | `Identity or appearance` or `Object, prop, clothing, interface, or effect` | Usually `fully_preserved` |
| Transfer an action, trajectory, pose change, or timing | `Motion or action` | `attribute_transfer` when applied to another subject |

MiniMax's reference format distinguishes reusable visible content (`<Subject
N>`) from a concrete picture used as a keyframe, composition anchor, or
storyboard (`<Picture N>`). Do not use an exact endpoint role merely because an
image happens to resemble the intended shot.

## 2. Minimum useful scene-composition reference

A reference does not have to be photorealistic. A sketch, collage, blocked 3D
viewport, or simple render can work when it communicates these relationships
clearly:

- Target aspect ratio.
- Camera viewpoint, height, and approximate lens/framing.
- Number of visible subjects.
- Each subject's screen position, size, and facing direction.
- Foreground, middle-ground, and background separation.
- Horizon and major perspective lines.
- Important occlusion relationships and intentional empty space.

A rough image controls broad composition more reliably than identity, anatomy,
lighting, texture, or fine object details. Use separate identity/item reference
images when those details matter.

### Practical image preparation

- Use the target video's aspect ratio whenever possible.
- A short edge around 768 pixels is a useful practical baseline. This is not an
  official minimum, but very small references are not enlarged by ComfyUI's H3
  reference pipeline.
- Keep silhouettes and major shapes visually distinct.
- Avoid visible arrows, labels, grids, or bounding boxes unless their possible
  appearance in the result is acceptable.
- Put mapping and ignore instructions in prompt metadata rather than drawing
  them into the image.
- Use one composition image per shot when exact shot-to-image mapping matters.

Example role notes for a rough layout:

```text
Use only the viewpoint, subject placement, relative scale, depth layers, and
camera height. Ignore the sketch style, placeholder colors, unfinished anatomy,
and missing surface details.
```

## 3. Designating character positions

Define character identity separately from the composition image:

```text
Character image 1
  role: Identity or appearance
  content_group: woman

Character image 2
  role: Identity or appearance
  content_group: man

Rough composition image
  role: Storyboard or shot planning
  retention: weak_reference
  shot_scope: 3
  notes: Map the left foreground placeholder to woman and the right background
         placeholder to man. Preserve blocking, relative scale, and eyelines.
         Ignore placeholder appearance and color.
```

The Shot description should reinforce the intended blocking using the viewer's
screen coordinates:

```text
The woman occupies the left third in a tight eye-level close-up, centered around
25% of the frame width. The man occupies the right third in a lower medium
close-up, centered around 75% of the frame width. Both remain visible
simultaneously.
```

Useful position vocabulary:

- `screen-left`, `screen-right`, and `center`.
- `left third`, `center third`, and `right third`.
- Approximate percentages of frame width or height.
- `foreground`, `middle ground`, and `background`.
- `extreme close-up`, `head-and-shoulders`, `waist-up`, and `full body`.
- Facing direction, eyeline, relative scale, and occlusion.

For this node pack, compact shot scopes are:

```text
3       one shot
3,4     selected shots
3-5     an inclusive range
all     every shot
```

If the composition source is an animatic video rather than a still, use
`Camera, cuts, or rhythm` for its whole-video editing language and timing. Use
separate identity references for the final characters.

## 4. Minimum useful motion reference

The simplest adequate proxy depends on what must transfer:

| Proxy | Information it can communicate reliably |
| --- | --- |
| Moving point | Screen-space position, path, timing, speed, and easing |
| Oriented point, arrow, or circle with a direction line | Position plus facing direction or simple rotation |
| Rigid head/object proxy | Translation, yaw, pitch, roll, and scale change |
| Stick figure or simple rig | Joint articulation and recognizable body poses |
| Rough rendered animatic | Articulation, contacts, camera relationship, depth, and timing |

A moving point can guide the center of a head from left to right, but it cannot
by itself distinguish among:

- The whole head translating sideways.
- The head turning left-to-right around the neck (yaw).
- The head tilting sideways (roll).
- The head nodding up and down (pitch).
- The eyes merely tracking an off-screen target.

Use an oriented proxy whenever rotation matters. For a left-to-right head turn,
a circle with a nose/direction line is a reasonable minimum:

```text
Start: face points screen-left.
Middle: face points toward camera.
End: face points screen-right.
The head center and body remain stationary.
```

Every motion reference should make these properties explicit:

- Start and end states.
- Screen-space or 3D path.
- Speed, acceleration, pauses, and easing.
- Translation versus rotation versus deformation.
- The body part or object receiving the motion.
- What remains stationary.
- Whether the source camera moves.
- Contacts or constraints, such as a hand remaining on a table.

### Motion-transfer binding

```text
Target identity reference
  role: Identity or appearance
  content_group: woman
  retention: fully_preserved

Motion video
  role: Motion or action
  retention: attribute_transfer
  content_group: head_motion
  transfer_target: woman
  shot_scope: 3
  notes: Transfer only the head motion. Keep the body and camera stationary.
         Ignore the proxy's shape, color, identity, and background.
```

Example for a moving-point trajectory:

```text
Transfer only the point's screen-space path and timing to the center of the
woman's head. Her head translates smoothly from screen-left to screen-right
without rotating. Keep her body stationary and do not include the point.
```

Use a 2-15 second reference video for the normal H3 Ref2VA workflow. A short
motion can be padded with clear stationary holds before and after the action so
the initial and final states remain legible.

## 5. Reliability checklist

Before generating, verify:

- The selected role describes the intended relationship, not merely the file
  type.
- Every identity or reusable item has a stable `content_group`.
- An `attribute_transfer` source names a different visible target group.
- The role's `shot_scope` refers to an existing Shot.
- Rough reference colors and shapes are explicitly marked as disposable when
  they should not transfer.
- The prompt distinguishes position from orientation and object motion from
  camera motion.
- The native H3 inputs receive the same reference media described by the Guide
  and Enhancer context.

## Sources

- [MiniMax H3 full-reference prompt writing guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
- [ComfyUI MiniMax H3 native node implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py)

