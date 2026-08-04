"""Persistent, reusable media sheets for MiniMax H3 reference workflows."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

try:
    from .media_context import (
        AUTO_RELATION,
        REFERENCE_CONTEXT_TYPE,
        REFERENCE_ROLES,
        ROLE_CHAIN_TYPE,
        ROLE_CHAIN_VERSION,
        SUBJECT_ROLES,
        UNASSIGNED_ROLE,
        VISUAL_RELATIONS,
        PICTURE_MEDIA,
        MiniMaxH3EnhancerVisualReference,
        role_binding_entries,
    )
except ImportError:  # Direct module loading in tests.
    from media_context import (
        AUTO_RELATION,
        REFERENCE_CONTEXT_TYPE,
        REFERENCE_ROLES,
        ROLE_CHAIN_TYPE,
        ROLE_CHAIN_VERSION,
        SUBJECT_ROLES,
        UNASSIGNED_ROLE,
        VISUAL_RELATIONS,
        PICTURE_MEDIA,
        MiniMaxH3EnhancerVisualReference,
        role_binding_entries,
    )


REFERENCE_SHEET_DRAFT_TYPE = "MINIMAX_H3_REFERENCE_SHEET_DRAFT"
REFERENCE_SHEET_TYPE = "MINIMAX_H3_REFERENCE_SHEET"
AUDIO_REFERENCE_CONTEXT_TYPE = "MINIMAX_H3_AUDIO_REFERENCE_CONTEXT"

SHEET_SCHEMA_VERSION = 1
SHEET_DRAFT_VERSION = 1
AUDIO_CONTEXT_VERSION = 1

NO_SAVED_SHEET = "(no saved reference sheets)"
NO_IMAGE_FILE = "(no image files found in ComfyUI/input)"
NO_AUDIO_FILE = "(no audio files found in ComfyUI/input)"
USE_SHEET_ROLE = "Use the sheet asset's suggested role"
NO_SUGGESTED_ROLE = "No default - choose when used"

LOAD_SHEET = "Load existing"
CREATE_SHEET = "Create new"
UPDATE_SHEET = "Update existing"
SHEET_OPERATIONS = [LOAD_SHEET, CREATE_SHEET, UPDATE_SHEET]

AUDIO_COPY_ALL = "Reuse the complete audio signal"
AUDIO_COPY_PART = "Reuse only part or selected layers"
AUDIO_REFERENCE = "Reference voice, music, beat, or sound"
AUDIO_WEAK = "Use only broad audio mood"
AUDIO_USES = [AUDIO_REFERENCE, AUDIO_COPY_ALL, AUDIO_COPY_PART, AUDIO_WEAK]
AUDIO_RETENTION = {
    AUDIO_COPY_ALL: "fully_copy",
    AUDIO_COPY_PART: "partially_copy",
    AUDIO_REFERENCE: "reference",
    AUDIO_WEAK: "weak_reference",
}

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


def _clean_text(value) -> str:
    return " ".join(str(value or "").strip().split())


def _slug(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", _clean_text(value).casefold()).strip("-")
    return slug[:80] or fallback


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _folder_paths_module():
    try:
        import folder_paths
    except ImportError:
        return None
    return folder_paths


def _input_directory() -> Path:
    override = os.environ.get("MINIMAX_H3_INPUT_DIR")
    if override:
        return Path(override).expanduser().resolve()
    folder_paths = _folder_paths_module()
    if folder_paths is None:
        return Path.cwd().resolve() / ".missing-comfy-input"
    return Path(folder_paths.get_input_directory()).resolve()


def _reference_sheet_root() -> Path:
    override = os.environ.get("MINIMAX_H3_REFERENCE_SHEETS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    folder_paths = _folder_paths_module()
    if folder_paths is None:
        return Path.cwd().resolve() / ".minimax-h3-reference-sheets"
    return (
        Path(folder_paths.get_user_directory()).resolve()
        / "default"
        / "minimax_h3"
        / "reference_sheets"
    )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_input_file(filename: str, extensions: set[str]) -> Path:
    if not filename or filename in {NO_IMAGE_FILE, NO_AUDIO_FILE}:
        raise ValueError("Select a real media file from ComfyUI's input directory.")
    root = _input_directory()
    folder_paths = _folder_paths_module()
    if folder_paths is not None:
        try:
            resolved = Path(folder_paths.get_annotated_filepath(filename)).resolve(strict=True)
        except Exception as error:
            raise ValueError(f"Input media file cannot be resolved: {filename!r}.") from error
    else:
        resolved = (root / filename).resolve(strict=True)
    if not _is_within(resolved, root):
        raise ValueError("Reference Sheet media must come from ComfyUI's input directory.")
    if resolved.suffix.casefold() not in extensions:
        raise ValueError(f"Unsupported reference-sheet media extension: {resolved.suffix!r}.")
    return resolved


def _input_files(extensions: set[str], empty_label: str) -> list[str]:
    root = _input_directory()
    if not root.is_dir():
        return [empty_label]
    files = sorted(
        path.name
        for path in root.iterdir()
        if path.is_file() and path.suffix.casefold() in extensions
    )
    return files or [empty_label]


def _load_image(path: Path):
    import numpy
    import torch
    from PIL import Image, ImageOps

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        array = numpy.asarray(image, dtype=numpy.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _load_wav_fallback(path: Path):
    import numpy
    import torch

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)
    if sample_width == 1:
        samples = (numpy.frombuffer(raw, dtype=numpy.uint8).astype(numpy.float32) - 128.0) / 128.0
    elif sample_width == 2:
        samples = numpy.frombuffer(raw, dtype="<i2").astype(numpy.float32) / 32768.0
    elif sample_width == 4:
        samples = numpy.frombuffer(raw, dtype="<i4").astype(numpy.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes.")
    samples = samples.reshape(-1, channels).T.copy()
    waveform = torch.from_numpy(samples).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}


def _load_audio(path: Path) -> dict:
    try:
        from comfy_extras.nodes_audio import load as comfy_load_audio
    except ImportError:
        if path.suffix.casefold() != ".wav":
            raise ValueError(
                "This audio format needs ComfyUI's audio decoder. WAV is the only standalone "
                "fallback available during tests or direct module loading."
            )
        return _load_wav_fallback(path)
    waveform, sample_rate = comfy_load_audio(str(path))
    return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}


def _audio_duration(audio: dict) -> float:
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Audio must use ComfyUI's waveform/sample_rate AUDIO format.")
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    if not hasattr(waveform, "shape") or len(waveform.shape) != 3 or waveform.shape[-1] < 1:
        raise ValueError("Audio waveform must have shape [batch, channels, samples].")
    if isinstance(sample_rate, bool) or int(sample_rate) <= 0:
        raise ValueError("Audio sample_rate must be a positive integer.")
    return waveform.shape[-1] / int(sample_rate)


def _draft_entries(draft) -> list[dict]:
    if draft is None:
        return []
    if not isinstance(draft, dict) or draft.get("version") != SHEET_DRAFT_VERSION:
        raise ValueError("assets must come from a MiniMax H3 Reference Sheet Asset node.")
    raw_entries = draft.get("assets")
    if not isinstance(raw_entries, (list, tuple)):
        raise ValueError("The reference-sheet asset chain is malformed.")
    entries = []
    seen = set()
    for raw in raw_entries:
        if not isinstance(raw, dict) or raw.get("kind") not in {"image", "audio"}:
            raise ValueError("Every reference-sheet asset must be an image or audio entry.")
        key = _clean_text(raw.get("key"))
        if not key:
            raise ValueError("Every reference-sheet asset needs a non-empty asset key.")
        folded = key.casefold()
        if folded in seen:
            raise ValueError(f"Reference-sheet asset key {key!r} is duplicated.")
        seen.add(folded)
        source_path = Path(str(raw.get("source_path", ""))).resolve(strict=True)
        if not _is_within(source_path, _input_directory()):
            raise ValueError("Reference-sheet draft media must remain inside ComfyUI/input.")
        expected_extensions = IMAGE_EXTENSIONS if raw["kind"] == "image" else AUDIO_EXTENSIONS
        if source_path.suffix.casefold() not in expected_extensions:
            raise ValueError(f"Unsupported {raw['kind']} asset extension: {source_path.suffix!r}.")
        entries.append(
            {
                "key": key,
                "kind": raw["kind"],
                "source_path": str(source_path),
                "source_name": _clean_text(raw.get("source_name")) or source_path.name,
                "description": _clean_text(raw.get("description")),
                "suggested_role": _clean_text(raw.get("suggested_role")),
                "group_key": _clean_text(raw.get("group_key")) or "main",
            }
        )
    return entries


def _append_draft(previous_assets, candidate: dict) -> dict:
    entries = _draft_entries(previous_assets)
    entries.append(candidate)
    return {"version": SHEET_DRAFT_VERSION, "assets": _draft_entries({"version": 1, "assets": entries})}


def _manifest_asset_path(sheet_dir: Path, relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Reference-sheet manifests may contain only safe relative media paths.")
    path = (sheet_dir / relative).resolve()
    if not _is_within(path, sheet_dir.resolve()):
        raise ValueError("Reference-sheet media path escapes its sheet directory.")
    return path


def _validate_manifest(manifest, sheet_dir: Path, *, verify_files: bool = True) -> dict:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SHEET_SCHEMA_VERSION:
        raise ValueError("Unsupported or malformed Reference Sheet manifest.")
    try:
        uuid.UUID(str(manifest.get("id")))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("Reference Sheet manifest needs a valid UUID id.") from error
    name = _clean_text(manifest.get("name"))
    if not name:
        raise ValueError("Reference Sheet manifest needs a display name.")
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ValueError("Reference Sheet manifest contains no assets.")
    assets = []
    seen = set()
    for raw in raw_assets:
        if not isinstance(raw, dict) or raw.get("kind") not in {"image", "audio"}:
            raise ValueError("Reference Sheet manifest has an unsupported asset kind.")
        key = _clean_text(raw.get("key"))
        if not key or key.casefold() in seen:
            raise ValueError("Reference Sheet manifest asset keys must be non-empty and unique.")
        seen.add(key.casefold())
        relative_name = str(raw.get("file", ""))
        media_path = _manifest_asset_path(sheet_dir, relative_name)
        if verify_files:
            if not media_path.is_file():
                raise ValueError(f"Reference Sheet asset is missing: {relative_name!r}.")
            supplied_hash = str(raw.get("sha256", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", supplied_hash) or _sha256(media_path) != supplied_hash:
                raise ValueError(f"Reference Sheet asset checksum failed: {relative_name!r}.")
        assets.append(
            {
                "key": key,
                "kind": raw["kind"],
                "file": relative_name,
                "description": _clean_text(raw.get("description")),
                "suggested_role": _clean_text(raw.get("suggested_role")),
                "group_key": _clean_text(raw.get("group_key")) or "main",
                "source_name": _clean_text(raw.get("source_name")),
                "sha256": str(raw.get("sha256", "")),
            }
        )
    raw_tags = manifest.get("tags", [])
    if not isinstance(raw_tags, list):
        raise ValueError("Reference Sheet manifest tags must be a list of strings.")
    result = dict(manifest)
    result.update(
        {
            "name": name,
            "description": _clean_text(manifest.get("description")),
            "tags": [_clean_text(tag) for tag in raw_tags if _clean_text(tag)],
            "assets": assets,
        }
    )
    return result


def _read_manifest(sheet_dir: Path, *, verify_files: bool = True) -> dict:
    manifest_path = sheet_dir / "manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read Reference Sheet manifest at {manifest_path}.") from error
    return _validate_manifest(manifest, sheet_dir, verify_files=verify_files)


def _sheet_records() -> list[dict]:
    root = _reference_sheet_root()
    if not root.is_dir():
        return []
    records = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = _read_manifest(manifest_path.parent, verify_files=False)
        except ValueError:
            continue
        records.append(
            {
                "option": f"{manifest['name']} [{manifest['id'][:8]}]",
                "directory": manifest_path.parent,
                "manifest": manifest,
            }
        )
    return sorted(records, key=lambda record: record["option"].casefold())


def _sheet_options() -> list[str]:
    options = [record["option"] for record in _sheet_records()]
    return options or [NO_SAVED_SHEET]


def _selected_record(selection: str) -> dict:
    for record in _sheet_records():
        if selection in {record["option"], record["directory"].name, record["manifest"]["id"]}:
            return record
    raise ValueError(f"Saved Reference Sheet cannot be found: {selection!r}.")


def _sheet_runtime(sheet_dir: Path) -> dict:
    manifest = _read_manifest(sheet_dir, verify_files=True)
    return {
        "version": SHEET_SCHEMA_VERSION,
        "root": str(sheet_dir.resolve()),
        "manifest": manifest,
    }


def reference_sheet_manifest(sheet) -> tuple[dict, Path]:
    if not isinstance(sheet, dict) or sheet.get("version") != SHEET_SCHEMA_VERSION:
        raise ValueError("reference_sheet must come from MiniMax H3 Reference Sheet.")
    root = Path(str(sheet.get("root", ""))).resolve()
    library_root = _reference_sheet_root().resolve()
    if not _is_within(root, library_root):
        raise ValueError("Reference Sheet root is outside the configured sheet library.")
    manifest = _read_manifest(root, verify_files=True)
    supplied = sheet.get("manifest")
    if not isinstance(supplied, dict) or supplied.get("id") != manifest["id"]:
        raise ValueError("Reference Sheet runtime payload does not match its saved manifest.")
    return manifest, root


def _selected_asset_key(sheet, expected_kind: str) -> str:
    manifest, _root = reference_sheet_manifest(sheet)
    supplied = _clean_text(sheet.get(f"selected_{expected_kind}_key"))
    available = [asset["key"] for asset in manifest["assets"] if asset["kind"] == expected_kind]
    if supplied and supplied.casefold() in {key.casefold() for key in available}:
        return supplied
    if available:
        return available[0]
    raise ValueError(f"Reference Sheet {manifest['name']!r} contains no {expected_kind} assets.")


def _sheet_asset(sheet, asset_key: str | None, expected_kind: str) -> tuple[dict, dict, Path]:
    manifest, root = reference_sheet_manifest(sheet)
    resolved_key = _clean_text(asset_key) or _selected_asset_key(sheet, expected_kind)
    key = resolved_key.casefold()
    matches = [asset for asset in manifest["assets"] if asset["key"].casefold() == key]
    if not matches:
        available = ", ".join(asset["key"] for asset in manifest["assets"] if asset["kind"] == expected_kind)
        raise ValueError(
            f"Reference Sheet has no {expected_kind} asset named {asset_key!r}. "
            f"Available {expected_kind} keys: {available or 'none'}."
        )
    asset = matches[0]
    if asset["kind"] != expected_kind:
        raise ValueError(f"Reference Sheet asset {asset['key']!r} is {asset['kind']}, not {expected_kind}.")
    return manifest, asset, _manifest_asset_path(root, asset["file"])


def _sheet_report(sheet) -> str:
    manifest, root = reference_sheet_manifest(sheet)
    lines = [
        f"Reference Sheet: {manifest['name']} [{manifest['id'][:8]}]",
        f"Storage: {root}",
        f"Description: {manifest['description'] or 'N/A'}",
        f"Tags: {', '.join(manifest['tags']) or 'none'}",
        "Assets:",
    ]
    for asset in manifest["assets"]:
        role = asset["suggested_role"] or "no suggested role"
        lines.append(
            f"- {asset['key']} ({asset['kind']}; group={asset['group_key']}; {role}): "
            f"{asset['description'] or 'no description'}"
        )
    return "\n".join(lines)


def _save_sheet(
    operation: str,
    saved_sheet: str,
    sheet_name: str,
    sheet_description: str,
    tags: str,
    confirm_update: bool,
    draft,
) -> dict:
    entries = _draft_entries(draft)
    if not entries:
        raise ValueError("Create/update needs at least one connected Reference Sheet Asset.")
    root = _reference_sheet_root()
    root.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    if operation == CREATE_SHEET:
        display_name = _clean_text(sheet_name)
        if not display_name:
            raise ValueError("Create new requires a non-empty sheet_name.")
        sheet_id = str(uuid.uuid4())
        target = root / f"{_slug(display_name, 'sheet')}--{sheet_id[:8]}"
        created_at = now
        if target.exists():
            raise ValueError("The generated Reference Sheet directory already exists; retry creation.")
    elif operation == UPDATE_SHEET:
        if not confirm_update:
            raise ValueError("Update existing requires confirm_update=true to prevent accidental replacement.")
        record = _selected_record(saved_sheet)
        target = record["directory"]
        existing = _read_manifest(target, verify_files=True)
        sheet_id = existing["id"]
        display_name = _clean_text(sheet_name) or existing["name"]
        created_at = existing.get("created_at", now)
    else:
        raise ValueError(f"Unsupported save operation: {operation!r}.")

    staging = Path(tempfile.mkdtemp(prefix=".reference-sheet-", dir=root))
    backup = None
    try:
        manifest_assets = []
        for index, entry in enumerate(entries, start=1):
            source = Path(entry["source_path"])
            subdirectory = "images" if entry["kind"] == "image" else "audio"
            destination_directory = staging / subdirectory
            destination_directory.mkdir(parents=True, exist_ok=True)
            filename = f"{index:02d}-{_slug(entry['key'], 'asset')}{source.suffix.casefold()}"
            destination = destination_directory / filename
            shutil.copy2(source, destination)
            manifest_assets.append(
                {
                    "key": entry["key"],
                    "kind": entry["kind"],
                    "file": destination.relative_to(staging).as_posix(),
                    "description": entry["description"],
                    "suggested_role": entry["suggested_role"],
                    "group_key": entry["group_key"],
                    "source_name": entry["source_name"],
                    "sha256": _sha256(destination),
                }
            )
        resolved_description = _clean_text(sheet_description)
        resolved_tags = [part for value in tags.split(",") if (part := _clean_text(value))]
        if operation == UPDATE_SHEET:
            resolved_description = resolved_description or existing["description"]
            resolved_tags = resolved_tags or existing["tags"]
        manifest = {
            "schema_version": SHEET_SCHEMA_VERSION,
            "id": sheet_id,
            "name": display_name,
            "description": resolved_description,
            "tags": resolved_tags,
            "created_at": created_at,
            "updated_at": now,
            "assets": manifest_assets,
        }
        with (staging / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        _validate_manifest(manifest, staging, verify_files=True)

        if target.exists():
            backup = root / f".reference-sheet-backup-{uuid.uuid4().hex}"
            os.replace(target, backup)
        try:
            os.replace(staging, target)
        except Exception:
            if backup is not None and backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
        return _sheet_runtime(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


class MiniMaxH3ReferenceSheetImageAsset:
    """Add one input-directory image to a reference-sheet draft."""

    CATEGORY = "MiniMax H3/Reference Sheets"
    FUNCTION = "add_image"
    RETURN_TYPES = (REFERENCE_SHEET_DRAFT_TYPE, "IMAGE", "STRING")
    RETURN_NAMES = ("sheet_assets", "image", "asset_report")
    OUTPUT_TOOLTIPS = (
        "Connect to another Reference Sheet Image/Audio Asset.previous_assets, or to Reference Sheet Library.assets when this is the final asset.",
        "Decoded preview of the selected input image. Saving copies the original file, not this tensor.",
        "Asset key, source filename, intrinsic group, suggested role, and current draft size.",
    )
    DESCRIPTION = (
        "Loads one image directly from ComfyUI/input and appends it to a chainable Reference Sheet "
        "draft. Connect the final sheet_assets output to Reference Sheet Library when saving."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_file": (
                    _input_files(IMAGE_EXTENSIONS, NO_IMAGE_FILE),
                    {
                        "image_upload": True,
                        "tooltip": "Image stored in ComfyUI/input. The original file is copied into the saved sheet.",
                    },
                ),
                "asset_key": (
                    "STRING",
                    {
                        "default": "primary",
                        "tooltip": "Stable name used to select this asset later, such as face_front, full_body, item_detail, or location_wide.",
                    },
                ),
                "description": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": "",
                        "tooltip": "Intrinsic description of what this file shows. Do not put Shot numbers here.",
                    },
                ),
                "suggested_role": (
                    [NO_SUGGESTED_ROLE, *[role for role in REFERENCE_ROLES if role != UNASSIGNED_ROLE]],
                    {
                        "default": NO_SUGGESTED_ROLE,
                        "tooltip": "Optional reusable suggestion only. The workflow-use node may override it without changing the saved sheet.",
                    },
                ),
                "group_key": (
                    "STRING",
                    {
                        "default": "main",
                        "tooltip": "Intrinsic group inside this sheet. Related views can share main; independent objects can use different keys.",
                    },
                ),
            },
            "optional": {
                "previous_assets": (
                    REFERENCE_SHEET_DRAFT_TYPE,
                    {"tooltip": "Connect sheet_assets from the preceding image/audio asset node."},
                )
            },
        }

    def add_image(
        self,
        image_file: str,
        asset_key: str,
        description: str,
        suggested_role: str,
        group_key: str,
        previous_assets=None,
    ):
        path = _resolve_input_file(image_file, IMAGE_EXTENSIONS)
        role = "" if suggested_role == NO_SUGGESTED_ROLE else suggested_role
        candidate = {
            "key": _clean_text(asset_key),
            "kind": "image",
            "source_path": str(path),
            "source_name": path.relative_to(_input_directory()).as_posix(),
            "description": _clean_text(description),
            "suggested_role": role,
            "group_key": _clean_text(group_key) or "main",
        }
        draft = _append_draft(previous_assets, candidate)
        report = (
            f"Image asset {candidate['key']!r}: {candidate['source_name']}; "
            f"group={candidate['group_key']}; suggested role={role or 'none'}. "
            f"Draft now contains {len(draft['assets'])} asset(s)."
        )
        return (draft, _load_image(path), report)

    @classmethod
    def IS_CHANGED(cls, image_file, **_kwargs):
        return _sha256(_resolve_input_file(image_file, IMAGE_EXTENSIONS))

    @classmethod
    def VALIDATE_INPUTS(cls, image_file, **_kwargs):
        try:
            _resolve_input_file(image_file, IMAGE_EXTENSIONS)
        except ValueError as error:
            return str(error)
        return True


class MiniMaxH3ReferenceSheetAudioAsset:
    """Add one input-directory audio file to a reference-sheet draft."""

    CATEGORY = "MiniMax H3/Reference Sheets"
    FUNCTION = "add_audio"
    RETURN_TYPES = (REFERENCE_SHEET_DRAFT_TYPE, "AUDIO", "STRING")
    RETURN_NAMES = ("sheet_assets", "audio", "asset_report")
    OUTPUT_TOOLTIPS = (
        "Connect to another Reference Sheet Image/Audio Asset.previous_assets, or to Reference Sheet Library.assets when this is the final asset.",
        "Decoded ComfyUI AUDIO preview of the selected file. Saving copies the original file.",
        "Asset key, source filename, duration, intrinsic group, suggested use, and current draft size.",
    )
    DESCRIPTION = (
        "Loads one audio or audio-bearing media file directly from ComfyUI/input and appends it to "
        "a chainable Reference Sheet draft. The saved role is only a suggestion."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_file": (
                    _input_files(AUDIO_EXTENSIONS, NO_AUDIO_FILE),
                    {
                        "tooltip": "Audio or audio-bearing media in ComfyUI/input. The original file is copied into the saved sheet.",
                    },
                ),
                "asset_key": (
                    "STRING",
                    {
                        "default": "voice",
                        "tooltip": "Stable name used to select this audio later, such as voice, ambience, theme, or source_mix.",
                    },
                ),
                "description": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": "",
                        "tooltip": "Describe voice, accent, delivery, music, rhythm, or sound characteristics. Qwen receives this text, not waveform analysis.",
                    },
                ),
                "suggested_use": (
                    AUDIO_USES,
                    {
                        "default": AUDIO_REFERENCE,
                        "tooltip": "Reusable suggestion only. The workflow-use node chooses the actual audio relationship.",
                    },
                ),
                "group_key": (
                    "STRING",
                    {
                        "default": "main",
                        "tooltip": "Intrinsic group inside this sheet, for example main_voice or environment_audio.",
                    },
                ),
            },
            "optional": {
                "previous_assets": (
                    REFERENCE_SHEET_DRAFT_TYPE,
                    {"tooltip": "Connect sheet_assets from the preceding image/audio asset node."},
                )
            },
        }

    def add_audio(
        self,
        audio_file: str,
        asset_key: str,
        description: str,
        suggested_use: str,
        group_key: str,
        previous_assets=None,
    ):
        path = _resolve_input_file(audio_file, AUDIO_EXTENSIONS)
        candidate = {
            "key": _clean_text(asset_key),
            "kind": "audio",
            "source_path": str(path),
            "source_name": path.relative_to(_input_directory()).as_posix(),
            "description": _clean_text(description),
            "suggested_role": suggested_use,
            "group_key": _clean_text(group_key) or "main",
        }
        draft = _append_draft(previous_assets, candidate)
        audio = _load_audio(path)
        duration = _audio_duration(audio)
        report = (
            f"Audio asset {candidate['key']!r}: {candidate['source_name']}; "
            f"{duration:.3f}s; group={candidate['group_key']}; suggested use={suggested_use}. "
            f"Draft now contains {len(draft['assets'])} asset(s)."
        )
        return (draft, audio, report)

    @classmethod
    def IS_CHANGED(cls, audio_file, **_kwargs):
        return _sha256(_resolve_input_file(audio_file, AUDIO_EXTENSIONS))

    @classmethod
    def VALIDATE_INPUTS(cls, audio_file, **_kwargs):
        try:
            _resolve_input_file(audio_file, AUDIO_EXTENSIONS)
        except ValueError as error:
            return str(error)
        return True


class MiniMaxH3ReferenceSheetLibrary:
    """Explicitly create, update, or load a persistent Reference Sheet."""

    CATEGORY = "MiniMax H3/Reference Sheets"
    FUNCTION = "manage"
    RETURN_TYPES = (REFERENCE_SHEET_TYPE, "STRING")
    RETURN_NAMES = ("reference_sheet", "sheet_report")
    OUTPUT_TOOLTIPS = (
        "Connect to one or more Reference Sheet Visual/Audio Reference nodes. The payload points only to the validated saved sheet.",
        "Sheet ID, storage directory, description, tags, and exact image/audio asset keys available for selection.",
    )
    DESCRIPTION = (
        "Creates, explicitly updates, or loads a self-contained Reference Sheet under the ComfyUI "
        "user directory. Save operations require a connected asset chain; updates require confirmation."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (
                    SHEET_OPERATIONS,
                    {
                        "default": LOAD_SHEET,
                        "tooltip": "Load is read-only. Create makes a new sheet. Update replaces the selected sheet only when confirm_update is enabled.",
                    },
                ),
                "saved_sheet": (
                    _sheet_options(),
                    {"tooltip": "Existing sheet used by Load or Update. Refresh the browser/node list after creating a new sheet."},
                ),
                "sheet_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Required for Create. Optional replacement display name for Update. Ignored by Load.",
                    },
                ),
                "sheet_description": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": "",
                        "tooltip": (
                            "Reusable description of the sheet as a whole. Do not include workflow-specific "
                            "Shot numbers. Blank preserves the current value during Update."
                        ),
                    },
                ),
                "tags": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "character, red coat, studio project",
                        "tooltip": "Optional comma-separated library tags. Blank preserves current tags during Update.",
                    },
                ),
                "confirm_update": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Safety lock required only for Update existing. Create never overwrites another sheet.",
                    },
                ),
            },
            "optional": {
                "assets": (
                    REFERENCE_SHEET_DRAFT_TYPE,
                    {"tooltip": "Connect the final sheet_assets chain for Create or Update. Ignored by Load."},
                )
            },
        }

    def manage(
        self,
        operation: str,
        saved_sheet: str,
        sheet_name: str,
        sheet_description: str,
        tags: str,
        confirm_update: bool,
        assets=None,
    ):
        if operation == LOAD_SHEET:
            record = _selected_record(saved_sheet)
            sheet = _sheet_runtime(record["directory"])
        else:
            sheet = _save_sheet(
                operation,
                saved_sheet,
                sheet_name,
                sheet_description,
                tags,
                bool(confirm_update),
                assets,
            )
        return (sheet, _sheet_report(sheet))

    @classmethod
    def IS_CHANGED(cls, operation, saved_sheet, **_kwargs):
        if operation != LOAD_SHEET:
            return float("nan")
        try:
            record = _selected_record(saved_sheet)
        except ValueError:
            return float("nan")
        return _sha256(record["directory"] / "manifest.json")


def _connected_media_entries(image_inputs: list[tuple[str, object]], audio_inputs: list[tuple[str, object]]) -> list[dict]:
    entries = []
    image_number = 0
    for input_name, image in image_inputs:
        if image is None:
            continue
        if not hasattr(image, "shape") or len(image.shape) != 4 or image.shape[-1] < 3:
            raise ValueError(f"{input_name} must be a ComfyUI IMAGE batch with shape [B,H,W,C].")
        for batch_index in range(int(image.shape[0])):
            image_number += 1
            entries.append(
                {
                    "key": f"image_{image_number}",
                    "kind": "image",
                    "value": image[batch_index],
                    "source_name": f"{input_name} connection"
                    + (f", batch item {batch_index + 1}" if image.shape[0] > 1 else ""),
                }
            )
    if image_number > 9:
        raise ValueError("A Reference Sheet accepts at most 9 connected images, matching H3's image-reference limit.")

    audio_number = 0
    for input_name, audio in audio_inputs:
        if audio is None:
            continue
        _audio_duration(audio)
        waveform = audio["waveform"]
        if int(waveform.shape[0]) != 1:
            raise ValueError(f"{input_name} must contain exactly one AUDIO batch item.")
        audio_number += 1
        entries.append(
            {
                "key": f"audio_{audio_number}",
                "kind": "audio",
                "value": audio,
                "source_name": f"{input_name} connection",
            }
        )
    if audio_number > 3:
        raise ValueError("A Reference Sheet accepts at most 3 connected audio clips, matching H3's audio-reference limit.")
    return entries


def _write_connected_image(path: Path, image) -> None:
    import numpy
    from PIL import Image

    array = image.detach().to(device="cpu").float().clamp(0.0, 1.0).numpy()[..., :3]
    pixels = numpy.rint(array * 255.0).astype(numpy.uint8)
    Image.fromarray(pixels).save(path, format="PNG")


def _write_connected_audio(path: Path, audio: dict) -> None:
    import numpy

    waveform = audio["waveform"].detach().to(device="cpu").float()[0].clamp(-1.0, 1.0).numpy()
    if waveform.ndim != 2 or waveform.shape[0] < 1:
        raise ValueError("Connected AUDIO waveform must have shape [1, channels, samples].")
    samples = numpy.rint(waveform.T * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(int(samples.shape[1]))
        handle.setsampwidth(2)
        handle.setframerate(int(audio["sample_rate"]))
        handle.writeframes(samples.tobytes())


def _save_connected_sheet(
    operation: str,
    saved_sheet: str,
    sheet_name: str,
    sheet_description: str,
    tags: str,
    confirm_update: bool,
    entries: list[dict],
) -> dict:
    root = _reference_sheet_root()
    root.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    existing = None

    if operation == CREATE_SHEET:
        display_name = _clean_text(sheet_name)
        if not display_name:
            raise ValueError("Create new requires a non-empty sheet_name.")
        if not entries:
            raise ValueError("Create new requires at least one connected IMAGE or AUDIO input.")
        sheet_id = str(uuid.uuid4())
        target = root / f"{_slug(display_name, 'sheet')}--{sheet_id[:8]}"
        created_at = now
        if target.exists():
            raise ValueError("The generated Reference Sheet directory already exists; retry creation.")
    elif operation == UPDATE_SHEET:
        if not confirm_update:
            raise ValueError("Update existing requires confirm_update=true to prevent accidental replacement.")
        record = _selected_record(saved_sheet)
        target = record["directory"]
        existing = _read_manifest(target, verify_files=True)
        sheet_id = existing["id"]
        display_name = _clean_text(sheet_name) or existing["name"]
        created_at = existing.get("created_at", now)
    else:
        raise ValueError(f"Unsupported save operation: {operation!r}.")

    staging = Path(tempfile.mkdtemp(prefix=".reference-sheet-", dir=root))
    backup = None
    try:
        manifest_assets = []
        if entries:
            for entry in entries:
                subdirectory = "images" if entry["kind"] == "image" else "audio"
                extension = ".png" if entry["kind"] == "image" else ".wav"
                destination = staging / subdirectory / f"{entry['key']}{extension}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if entry["kind"] == "image":
                    _write_connected_image(destination, entry["value"])
                else:
                    _write_connected_audio(destination, entry["value"])
                manifest_assets.append(
                    {
                        "key": entry["key"],
                        "kind": entry["kind"],
                        "file": destination.relative_to(staging).as_posix(),
                        "description": "",
                        "suggested_role": "",
                        "group_key": "main",
                        "source_name": entry["source_name"],
                        "sha256": _sha256(destination),
                    }
                )
        else:
            # A metadata-only update preserves the current media byte-for-byte.
            for asset in existing["assets"]:
                source = _manifest_asset_path(target, asset["file"])
                destination = _manifest_asset_path(staging, asset["file"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied = dict(asset)
                copied["sha256"] = _sha256(destination)
                manifest_assets.append(copied)

        resolved_description = _clean_text(sheet_description)
        resolved_tags = [part for value in tags.split(",") if (part := _clean_text(value))]
        if existing is not None:
            resolved_description = resolved_description or existing["description"]
            resolved_tags = resolved_tags or existing["tags"]
        manifest = {
            "schema_version": SHEET_SCHEMA_VERSION,
            "id": sheet_id,
            "name": display_name,
            "description": resolved_description,
            "tags": resolved_tags,
            "created_at": created_at,
            "updated_at": now,
            "assets": manifest_assets,
        }
        with (staging / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        _validate_manifest(manifest, staging, verify_files=True)

        if target.exists():
            backup = root / f".reference-sheet-backup-{uuid.uuid4().hex}"
            os.replace(target, backup)
        try:
            os.replace(staging, target)
        except Exception:
            if backup is not None and backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
        return _sheet_runtime(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _with_selected_assets(sheet, selected_image_key: str, selected_audio_key: str) -> dict:
    manifest, _root = reference_sheet_manifest(sheet)
    selected = dict(sheet)
    for kind, supplied in (("image", selected_image_key), ("audio", selected_audio_key)):
        available = [asset["key"] for asset in manifest["assets"] if asset["kind"] == kind]
        folded = _clean_text(supplied).casefold()
        match = next((key for key in available if key.casefold() == folded), "")
        selected[f"selected_{kind}_key"] = match or (available[0] if available else "")
    return selected


class MiniMaxH3ReferenceSheet:
    """Create, load, preview, and select reusable reference media in one node."""

    CATEGORY = "MiniMax H3/Reference Sheets"
    FUNCTION = "manage"
    RETURN_TYPES = (REFERENCE_SHEET_TYPE, "IMAGE", "STRING", "AUDIO")
    RETURN_NAMES = (
        "reference_sheet",
        "selected_image",
        "sheet_report",
        "selected_audio",
    )
    OUTPUT_NODE = True
    OUTPUT_TOOLTIPS = (
        "Carries the selected saved image/audio for legacy Guide workflows that still use the Reference Sheet role nodes.",
        "The image currently selected in the embedded gallery. Connect it to a Plan v2 Image Reference and choose its exact workflow relationship there.",
        "Saved location, media inventory, and currently selected gallery items.",
        "The audio clip currently selected in the embedded gallery. Connect it directly to a Plan v2 Audio Reference node, where its exact workflow role is declared.",
    )
    DESCRIPTION = (
        "One integrated Reference Sheet. Connect ordinary Load Image/Load Audio nodes, save the sheet, "
        "then load it later and choose saved media from the embedded gallery. No filenames or asset keys are typed."
    )

    @classmethod
    def INPUT_TYPES(cls):
        image_tooltip = (
            "Optional IMAGE from Load Image or another image-producing node. IMAGE batches are expanded into "
            "separate saved references; the sheet accepts at most 9 images."
        )
        audio_tooltip = "Optional AUDIO from Load Audio; the sheet accepts at most 3 clips."
        return {
            "required": {
                "operation": (
                    SHEET_OPERATIONS,
                    {
                        "default": LOAD_SHEET,
                        "tooltip": "Load is read-only. Create saves connected media. Update replaces media only when new media is connected.",
                    },
                ),
                "saved_sheet": (
                    _sheet_options(),
                    {"tooltip": "Choose a saved sheet. The embedded gallery refreshes this list without typed paths."},
                ),
                "sheet_name": (
                    "STRING",
                    {"default": "", "tooltip": "Required only for Create new; optional rename for Update existing."},
                ),
                "sheet_description": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": "",
                        "tooltip": "Reusable description of this person, object, place, style, or project reference.",
                    },
                ),
                "tags": (
                    "STRING",
                    {"default": "", "placeholder": "character, project, location", "tooltip": "Optional comma-separated tags."},
                ),
                "confirm_update": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Safety lock required only for Update existing."},
                ),
                "selected_image_key": (
                    "STRING",
                    {"default": "", "tooltip": "Managed automatically by the embedded image gallery."},
                ),
                "selected_audio_key": (
                    "STRING",
                    {"default": "", "tooltip": "Managed automatically by the embedded audio gallery."},
                ),
            },
            "optional": {
                "image_1": ("IMAGE", {"tooltip": image_tooltip}),
                "image_2": ("IMAGE", {"tooltip": image_tooltip}),
                "image_3": ("IMAGE", {"tooltip": image_tooltip}),
                "image_4": ("IMAGE", {"tooltip": image_tooltip}),
                "audio_1": ("AUDIO", {"tooltip": audio_tooltip}),
                "audio_2": ("AUDIO", {"tooltip": audio_tooltip}),
                "audio_3": ("AUDIO", {"tooltip": audio_tooltip}),
            },
        }

    def manage(
        self,
        operation: str,
        saved_sheet: str,
        sheet_name: str,
        sheet_description: str,
        tags: str,
        confirm_update: bool,
        selected_image_key: str,
        selected_audio_key: str,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        audio_1=None,
        audio_2=None,
        audio_3=None,
    ):
        if operation == LOAD_SHEET:
            record = _selected_record(saved_sheet)
            sheet = _sheet_runtime(record["directory"])
        else:
            entries = _connected_media_entries(
                [("image_1", image_1), ("image_2", image_2), ("image_3", image_3), ("image_4", image_4)],
                [("audio_1", audio_1), ("audio_2", audio_2), ("audio_3", audio_3)],
            )
            sheet = _save_connected_sheet(
                operation,
                saved_sheet,
                sheet_name,
                sheet_description,
                tags,
                bool(confirm_update),
                entries,
            )
        sheet = _with_selected_assets(sheet, selected_image_key, selected_audio_key)
        manifest, _root = reference_sheet_manifest(sheet)
        option = f"{manifest['name']} [{manifest['id'][:8]}]"
        report = _sheet_report(sheet) + (
            f"\nSelected image: {sheet['selected_image_key'] or 'none'}"
            f"\nSelected audio: {sheet['selected_audio_key'] or 'none'}"
        )
        selected_image = None
        if sheet["selected_image_key"]:
            _manifest, _asset, selected_path = _sheet_asset(sheet, None, "image")
            selected_image = _load_image(selected_path)
        selected_audio = None
        if sheet["selected_audio_key"]:
            _manifest, _asset, selected_path = _sheet_asset(sheet, None, "audio")
            selected_audio = _load_audio(selected_path)
        return {
            "ui": {
                "saved_sheet": [option],
                "sheet_id": [manifest["id"]],
                "selected_image_key": [sheet["selected_image_key"]],
                "selected_audio_key": [sheet["selected_audio_key"]],
            },
            "result": (sheet, selected_image, report, selected_audio),
        }

    @classmethod
    def IS_CHANGED(cls, operation, saved_sheet, **_kwargs):
        if operation != LOAD_SHEET:
            return float("nan")
        try:
            record = _selected_record(saved_sheet)
        except ValueError:
            return float("nan")
        return _sha256(record["directory"] / "manifest.json")


class MiniMaxH3ReferenceSheetVisualReference:
    """Use the gallery-selected image as a role-aware, chainable H3 visual reference."""

    CATEGORY = "MiniMax H3/Reference Sheets"
    FUNCTION = "use_image"
    RETURN_TYPES = (REFERENCE_CONTEXT_TYPE, "IMAGE", "STRING")
    RETURN_NAMES = ("reference_context", "h3_media", "routing_report")
    OUTPUT_TOOLTIPS = (
        "Chain to another sheet/ordinary Visual Reference, then fan the final context to Prompt Guide and Prompt Enhancer.",
        "Original saved image for the native H3 socket listed in routing_report.",
        "Sheet asset, resolved workflow role, semantic H3 label, and exact native socket route.",
    )
    DESCRIPTION = (
        "Loads the image selected in a Reference Sheet gallery, assigns workflow-specific role/retention/Shot "
        "scope, and adds it to the same context used by Visual Reference nodes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_sheet": (
                    REFERENCE_SHEET_TYPE,
                    {"tooltip": "Connect MiniMax H3 Reference Sheet.reference_sheet. Its gallery selection is used automatically."},
                ),
                "reference_role": (
                    REFERENCE_ROLES,
                    {
                        "default": UNASSIGNED_ROLE,
                        "tooltip": "Choose what the selected gallery image means in this workflow.",
                    },
                ),
                "retention": (
                    VISUAL_RELATIONS,
                    {
                        "default": AUTO_RELATION,
                        "tooltip": "Workflow-specific H3 retention relationship; it is never written back into the sheet.",
                    },
                ),
                "content_group": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional workflow override. Blank uses a stable sheet/group-derived key for reusable Subject roles.",
                    },
                ),
                "transfer_target": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Destination content group only when retention=attribute_transfer.",
                    },
                ),
                "shot_scope": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Examples: 3, 3,4, 3-4, all",
                        "tooltip": "Workflow-only Shot scope. It is intentionally not saved in the reusable sheet.",
                    },
                ),
                "notes": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": "",
                        "tooltip": "Workflow-specific instructions about what to preserve, transfer, ignore, or change.",
                    },
                ),
                "analysis_long_edge": (
                    "INT",
                    {
                        "default": 768,
                        "min": 256,
                        "max": 1536,
                        "step": 32,
                        "tooltip": "Maximum long edge of the private Qwen analysis copy. H3 receives the original saved image.",
                    },
                ),
            },
            "optional": {
                "previous_context": (
                    REFERENCE_CONTEXT_TYPE,
                    {"tooltip": "Connect the preceding sheet/Visual Reference context to preserve native order."},
                ),
                "role_bindings": (
                    ROLE_CHAIN_TYPE,
                    {
                        "tooltip": (
                            "Optional final Visual Reference Role chain when this saved image has multiple "
                            "jobs. It replaces this node's single role/retention/group/scope widgets."
                        )
                    },
                ),
            },
        }

    def use_image(
        self,
        reference_sheet,
        reference_role: str,
        retention: str,
        content_group: str,
        transfer_target: str,
        shot_scope: str,
        notes: str,
        analysis_long_edge: int,
        previous_context=None,
        role_bindings=None,
    ):
        manifest, asset, path = _sheet_asset(reference_sheet, None, "image")
        if role_bindings is not None:
            resolved_bindings = role_binding_entries(role_bindings)
            role = resolved_bindings[0]["role"]
            binding = {"version": ROLE_CHAIN_VERSION, "bindings": resolved_bindings}
        else:
            role = reference_role
            if role == UNASSIGNED_ROLE:
                raise ValueError("Choose reference_role for the image selected in the Reference Sheet gallery.")
            resolved_group = _clean_text(content_group)
            if role in SUBJECT_ROLES and not resolved_group:
                resolved_group = f"sheet_{manifest['id'][:8]}_{_slug(asset['group_key'], 'main')}"
            binding = {
                "version": ROLE_CHAIN_VERSION,
                "bindings": [
                    {
                        "role": role,
                        "retention": retention,
                        "content_group": resolved_group,
                        "transfer_target": _clean_text(transfer_target),
                        "shot_scope": _clean_text(shot_scope),
                        "notes": _clean_text(notes),
                    }
                ],
            }
        media_notes = asset["description"] or manifest["description"]
        image = _load_image(path)
        context, h3_media, report = MiniMaxH3EnhancerVisualReference().add_reference(
            image,
            PICTURE_MEDIA,
            role,
            media_notes,
            24.0,
            1.0,
            16,
            int(analysis_long_edge),
            previous_context=previous_context,
            role_bindings=binding,
        )
        prefix = (
            f"Reference Sheet {manifest['name']} [{manifest['id'][:8]}], selected image "
            f"{asset['key']!r}; "
            f"workflow roles={', '.join(item['role'] for item in binding['bindings'])}."
        )
        return (context, h3_media, prefix + "\n" + report)


def audio_reference_entries(audio_context) -> list[dict]:
    if not isinstance(audio_context, dict) or audio_context.get("version") != AUDIO_CONTEXT_VERSION:
        raise ValueError(
            "audio_context must come from MiniMax H3 Reference Sheet Audio Reference."
        )
    raw_entries = audio_context.get("entries")
    if not isinstance(raw_entries, (list, tuple)) or not raw_entries:
        raise ValueError("Reference-sheet audio context contains no entries.")
    if len(raw_entries) > 3:
        raise ValueError("MiniMax H3 accepts at most 3 standalone reference-audio clips.")
    entries = []
    total_duration = 0.0
    for index, raw in enumerate(raw_entries, start=1):
        if not isinstance(raw, dict) or raw.get("use") not in AUDIO_USES:
            raise ValueError("Reference-sheet audio context contains an invalid audio relationship.")
        duration = float(raw.get("duration_seconds", 0.0))
        if not math.isfinite(duration) or not 2.0 <= duration <= 15.0:
            raise ValueError("Every H3 reference-audio clip must last from 2 through 15 seconds.")
        total_duration += duration
        expected = {
            "label": f"<Audio {index}>",
            "h3_node": "MiniMax H3 Reference to Video",
            "h3_node_id": "MiniMaxH3ReferenceToVideo",
            "h3_input": f"ref_audio_{index - 1}",
        }
        if any(raw.get(key) != value for key, value in expected.items()):
            raise ValueError("Reference-sheet audio context contains stale label or routing data.")
        entry = dict(raw)
        entry.update(
            {
                "description": _clean_text(raw.get("description")),
                "notes": _clean_text(raw.get("notes")),
                "shot_scope": _clean_text(raw.get("shot_scope")),
                "retention": AUDIO_RETENTION[raw["use"]],
                **expected,
            }
        )
        entries.append(entry)
    if total_duration > 15.0005:
        raise ValueError("Reference-audio clips total more than MiniMax H3's 15-second limit.")
    if len({entry["use"] for entry in entries}) != 1:
        raise ValueError(
            "The current Prompt Guide supports one audio relationship per audio_context chain. "
            "Use separate workflows or align the selected audio-use values."
        )
    if entries[0]["use"] == AUDIO_COPY_ALL and len(entries) != 1:
        raise ValueError("Complete 1:1 audio reuse requires exactly one reference-audio clip.")
    return entries


def audio_reference_inventory(entries: list[dict]) -> str:
    lines = ["Saved Reference Sheet audio references:"]
    for entry in entries:
        scope = f"; shot_scope={entry['shot_scope']}" if entry["shot_scope"] else ""
        notes = f"; notes={entry['notes']}" if entry["notes"] else ""
        lines.append(
            f"- {entry['label']}: route={entry['h3_node']}.{entry['h3_input']}; "
            f"use={entry['use']}; retention={entry['retention']}; "
            f"duration={entry['duration_seconds']:.3f}s; description={entry['description'] or 'N/A'}"
            f"{scope}{notes}."
        )
    return "\n".join(lines)


class MiniMaxH3ReferenceSheetAudioReference:
    """Use the gallery-selected audio in a chainable H3 audio-reference context."""

    CATEGORY = "MiniMax H3/Reference Sheets"
    FUNCTION = "use_audio"
    RETURN_TYPES = (AUDIO_REFERENCE_CONTEXT_TYPE, "AUDIO", "STRING")
    RETURN_NAMES = ("audio_context", "h3_audio", "routing_report")
    OUTPUT_TOOLTIPS = (
        "Chain to another Reference Sheet Audio Reference, then connect only the final context to Prompt Guide.audio_context.",
        "Decoded saved audio for the native standalone ref_audio_N socket listed in routing_report.",
        "Audio labels, relationships, durations, Shot scopes, and exact native standalone-audio routes.",
    )
    DESCRIPTION = (
        "Loads the audio selected in a Reference Sheet gallery, assigns its workflow relationship, "
        "and routes it to a native standalone ref_audio_N input. Qwen receives saved text metadata, "
        "not waveform analysis."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_sheet": (
                    REFERENCE_SHEET_TYPE,
                    {"tooltip": "Connect MiniMax H3 Reference Sheet.reference_sheet. Its gallery selection is used automatically."},
                ),
                "audio_use": (
                    AUDIO_USES,
                    {
                        "default": AUDIO_REFERENCE,
                        "tooltip": "Choose what the selected gallery audio means in this workflow.",
                    },
                ),
                "shot_scope": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Examples: 3, 3,4, 3-4, all",
                        "tooltip": "Optional workflow-only Shot scope for this audio relationship.",
                    },
                ),
                "notes": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": "",
                        "tooltip": "Workflow-specific audio instructions. The saved description remains unchanged.",
                    },
                ),
            },
            "optional": {
                "previous_audio_context": (
                    AUDIO_REFERENCE_CONTEXT_TYPE,
                    {"tooltip": "Connect audio_context from another sheet-audio reference to preserve order."},
                )
            },
        }

    def use_audio(
        self,
        reference_sheet,
        audio_use: str,
        shot_scope: str,
        notes: str,
        previous_audio_context=None,
    ):
        manifest, asset, path = _sheet_asset(reference_sheet, None, "audio")
        use = audio_use
        if use not in AUDIO_USES:
            raise ValueError(
                f"Sheet audio {asset['key']!r} has no valid suggested use. Choose audio_use explicitly."
            )
        previous = (
            audio_reference_entries(previous_audio_context)
            if previous_audio_context is not None
            else []
        )
        audio = _load_audio(path)
        duration = _audio_duration(audio)
        index = len(previous) + 1
        entry = {
            "sheet_id": manifest["id"],
            "sheet_name": manifest["name"],
            "asset_key": asset["key"],
            "use": use,
            "retention": AUDIO_RETENTION[use],
            "description": asset["description"] or manifest["description"],
            "notes": _clean_text(notes),
            "shot_scope": _clean_text(shot_scope),
            "duration_seconds": duration,
            "label": f"<Audio {index}>",
            "h3_node": "MiniMax H3 Reference to Video",
            "h3_node_id": "MiniMaxH3ReferenceToVideo",
            "h3_input": f"ref_audio_{index - 1}",
        }
        context = {"version": AUDIO_CONTEXT_VERSION, "entries": [*previous, entry]}
        entries = audio_reference_entries(context)
        report = audio_reference_inventory(entries) + (
            "\nConnect each h3_audio output to its listed native standalone ref_audio_N input. "
            "Connect only the final audio_context to Prompt Guide.audio_context."
        )
        return (context, audio, report)


def _reference_sheet_catalog() -> list[dict]:
    catalog = []
    for record in _sheet_records():
        try:
            manifest = _read_manifest(record["directory"], verify_files=True)
        except ValueError:
            continue
        catalog.append(
            {
                "id": manifest["id"],
                "option": f"{manifest['name']} [{manifest['id'][:8]}]",
                "name": manifest["name"],
                "description": manifest["description"],
                "tags": manifest["tags"],
                "assets": [
                    {
                        "key": asset["key"],
                        "kind": asset["kind"],
                        "description": asset["description"],
                        "sha256": asset["sha256"],
                    }
                    for asset in manifest["assets"]
                ],
            }
        )
    return catalog


def _register_reference_sheet_routes() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return
    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None or getattr(prompt_server, "_minimax_h3_reference_sheet_routes", False):
        return
    prompt_server._minimax_h3_reference_sheet_routes = True

    @prompt_server.routes.get("/minimax_h3/reference_sheets/catalog")
    async def reference_sheet_catalog_route(_request):
        return web.json_response({"sheets": _reference_sheet_catalog()})

    @prompt_server.routes.get("/minimax_h3/reference_sheets/media")
    async def reference_sheet_media_route(request):
        sheet_id = _clean_text(request.rel_url.query.get("sheet_id"))
        asset_key = _clean_text(request.rel_url.query.get("asset_key"))
        if not sheet_id or not asset_key:
            raise web.HTTPBadRequest(text="sheet_id and asset_key are required")
        try:
            record = _selected_record(sheet_id)
            manifest = _read_manifest(record["directory"], verify_files=True)
            asset = next(item for item in manifest["assets"] if item["key"] == asset_key)
            path = _manifest_asset_path(record["directory"], asset["file"])
        except (ValueError, StopIteration):
            raise web.HTTPNotFound(text="Reference Sheet media was not found") from None
        return web.FileResponse(
            path,
            headers={
                "Content-Disposition": "inline",
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "private, max-age=3600",
            },
        )


_register_reference_sheet_routes()


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ReferenceSheet": MiniMaxH3ReferenceSheet,
    "MiniMaxH3ReferenceSheetVisualReference": MiniMaxH3ReferenceSheetVisualReference,
    "MiniMaxH3ReferenceSheetAudioReference": MiniMaxH3ReferenceSheetAudioReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ReferenceSheet": "MiniMax H3 Reference Sheet",
    "MiniMaxH3ReferenceSheetVisualReference": "MiniMax H3 Reference Sheet Visual Reference (Legacy)",
    "MiniMaxH3ReferenceSheetAudioReference": "MiniMax H3 Reference Sheet Audio Reference (Legacy)",
}
