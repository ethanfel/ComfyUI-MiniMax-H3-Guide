# Persistent Reference Sheets

A Reference Sheet is a reusable image/audio library entry. It can describe a
person, object, outfit, location, style, voice, sound, or a mixed project
collection.

## Storage

Sheets live outside the custom-node repository:

```text
ComfyUI/user/default/minimax_h3/reference_sheets/
└── <sheet-name>--<short-id>/
    ├── manifest.json
    ├── images/
    └── audio/
```

Each versioned manifest contains a UUID, descriptions, tags, relative paths,
and SHA-256 checksums. Connected `IMAGE` tensors are saved as PNG and connected
`AUDIO` tensors as WAV, so a sheet does not depend on the original workflow or
input filename.

Set `MINIMAX_H3_REFERENCE_SHEETS_DIR` before starting ComfyUI to use a custom
library root.

## Create a sheet

```text
Load Image.IMAGE → Reference Sheet.image_1
Load Image.IMAGE → Reference Sheet.image_2  (optional)
Load Audio.AUDIO → Reference Sheet.audio_1  (optional)
```

Choose **Create new**, provide a display name and metadata, then queue once.
The node accepts up to four image sources and three audio sources. An image
batch expands into separate saved pictures up to H3's nine-image limit.

Create never overwrites another sheet.

## Load and select media

Choose **Load existing**, select a sheet, then choose a thumbnail or audio item
from the embedded gallery. Selection is carried in the node output; users do
not type internal filenames or asset keys.

Audio uses a non-destructive trim window:

- `audio_start_seconds` selects the offset;
- `audio_duration_seconds` selects a 2–15 second output segment.

The stored WAV and player remain complete. Duplicate the Reference Sheet node
to use several assets or different segments from one saved clip in a workflow.

## Update behavior

Update requires confirmation and is atomic.

- Metadata-only updates preserve every media asset.
- Connected non-duplicate media is appended.
- Byte-identical media is skipped.
- Existing assets are not treated as a replaceable input list.

## Use with Plan v2

```text
Reference Sheet.selected_image → Image Reference.image
Reference Sheet.selected_audio → Audio Reference.audio
```

The Plan v2 reference node declares the current workflow's role, retention,
scope, speaker/layer binding, and native route. The sheet itself remains a
role-neutral media library.

The structured Qwen enhancer can analyze a selected image after it enters the
Plan v2 inventory. It does not analyze the audio waveform; audio meaning comes
from Audio Reference metadata. Video is handled by Video Reference rather than
Reference Sheet.
