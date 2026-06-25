from __future__ import annotations

import argparse
import csv
import os
import shutil
import importlib.util
import zlib
import subprocess
import re
from datetime import datetime
from typing import Any
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

ALWAYS_EXPECTED_OUTPUTS = [
    "all_components_before_filter.csv",
    "rejected_objects.csv",
    "cell_features.csv",
    "blue_pixels_features.csv",
    "final_frame_summary.csv",
    "extended_qc_report.md",
    "run_parameters.txt",
    "run_parameters.csv",
    "macro_log.txt",
]

RUNNER_WRITTEN_OUTPUTS = {
    "run_parameters.txt",
    "run_parameters.csv",
    "macro_log.txt",
}

OVERLAY_EXPECTED_OUTPUTS = [
    "cellmask.tif",
    "vis_cellpixels.png",
    "roi_overlay.jpg",
    "selected_objects_overlay.jpg",
    "blue_inside_cells.tif",
]

DEBUG_OUTPUTS = [
    "debug_texture_evidence_mask.tif",
    "debug_edge_evidence_mask.tif",
    "debug_candidate_mask_raw.tif",
    "debug_candidate_mask_cleaned.tif",
    "debug_weka_probability_map.tif",
    "debug_weka_class_map.tif",
    "debug_weka_tile_mask_raw.tif",
    "weka_status.txt",
]

FINAL_OUTPUT_MAP = {
    "cellmask.tif": "cellmask_{image_stem}.tif",
    "vis_cellpixels.png": "vis_cellpixels_{image_stem}.png",
    "roi_overlay.jpg": "roi_overlay_{image_stem}.jpg",
    "selected_objects_overlay.jpg": "selected_objects_overlay_{image_stem}.jpg",
    "blue_inside_cells.tif": "blue_inside_cells_{image_stem}.tif",
    "blue_table.xlsx": "blue_table_{image_stem}.xlsx",
    "cell_features.csv": "cell_features_{image_stem}.csv",
    "final_frame_summary.csv": "frame_features_{image_stem}.csv",
}

DEFAULT_PARAMS = {
    "threshold_method": "Li",
    "threshold_mode": "bright",
    "background_rolling": "80",
    "median_radius": "2",
    "contrast_saturated": "0.35",
    "morph_open_iterations": "0",
    "morph_close_iterations": "2",
    "fill_holes": "true",
    "metadata_bar_height": "120",
    "particle_extract_min_area": "100",
    "particle_extract_max_area": "2000000",
    "min_noise_area": "100",
    "min_single_cell_area": "200",
    "max_single_cell_area": "3000",
    "min_aggregate_area": "3000",
    "max_aggregate_area": "2000000",
    "max_single_cell_aspect": "4",
    "max_aggregate_aspect": "8",
    "exclude_border_objects": "true",
    "border_margin_px": "20",
    "blue_min": "120",
    "blue_over_red": "20",
    "blue_over_green": "10",
    "save_overlays": "true",
    "label_objects": "true",
    "draw_rejected_objects": "false",
    "contour_width": "6",
    "final_overlay_preview_max_size": "1600",
    "min_expected_accepted_objects": "1",
    "min_stable_accepted_objects": "3",
    "max_stable_accepted_objects": "40",
    "min_stable_accepted_pixels": "500",
    "weka_tile_size": "768",
    "weka_tile_overlap": "64",
    "frame_select_top_size": "40",
    "frame_select_top_blue": "20",
    "frame_select_max_near_full_blue": "5",
}


def bool_param(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_project(value: Path | None) -> Path:
    # Default project root is the directory containing this runner, not a workstation-specific path.
    return (value or SCRIPT_DIR).resolve()


def resolve_fiji(project: Path, explicit: Path | None) -> Path:
    # Priority: CLI --fiji, env var, Fiji next to project, Fiji inside project. No user-machine hardcode.
    if explicit is not None:
        return explicit.resolve()
    env_value = os.environ.get("FIJI_PATH") or os.environ.get("FIJI_BAT")
    if env_value:
        return Path(env_value).resolve()
    candidates = [
        project.parent / "Fiji" / "fiji.bat",
        project.parent / "Fiji.app" / "fiji.bat",
        project / "Fiji" / "fiji.bat",
        project / "Fiji.app" / "fiji.bat",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise SystemExit(
        "Fiji runner not found. Pass --fiji <path-to-fiji-launcher> "
        "or set FIJI_PATH. Checked: " + "; ".join(str(c) for c in candidates)
    )


def discover_default_input(project: Path) -> Path:
    candidates = sorted((project / "input").glob("52*.bmp"))
    if candidates:
        for candidate in candidates:
            if "_2" not in candidate.stem:
                return candidate
        return candidates[0]
    return project / "input" / "52_proto1.bmp"


def expected_outputs(params: dict[str, str], image_stem: str | None = None) -> list[str]:
    outputs = list(ALWAYS_EXPECTED_OUTPUTS)
    if image_stem:
        outputs.extend([
            f"blue_table_{image_stem}.xlsx",
            f"cell_features_{image_stem}.csv",
            f"frame_features_{image_stem}.csv",
            f"selection_review_candidates_{image_stem}.csv",
            f"roi_proposals_{image_stem}.csv",
            f"reference_roi_regions_{image_stem}.csv",
        ])
    if bool_param(params.get("save_overlays", "true")):
        outputs.extend(OVERLAY_EXPECTED_OUTPUTS)
        if image_stem:
            outputs.extend([
                f"cellmask_{image_stem}.tif",
                f"vis_cellpixels_{image_stem}.png",
                f"roi_overlay_{image_stem}.jpg",
                f"selected_objects_overlay_{image_stem}.jpg",
                f"blue_inside_cells_{image_stem}.tif",
            ])
    return outputs


def final_expected_outputs(params: dict[str, str], image_stem: str) -> list[str]:
    outputs = [
        f"blue_table_{image_stem}.xlsx",
        f"cell_features_{image_stem}.csv",
        f"frame_features_{image_stem}.csv",
        f"selection_review_candidates_{image_stem}.csv",
    ]
    if bool_param(params.get("save_overlays", "true")):
        outputs.extend([
            f"cellmask_{image_stem}.tif",
            f"vis_cellpixels_{image_stem}.png",
            f"roi_overlay_{image_stem}.jpg",
            f"selected_objects_overlay_{image_stem}.jpg",
            f"selected_objects_contact_sheet_{image_stem}.jpg",
            f"review_candidates_contact_sheet_{image_stem}.jpg",
            f"roi_proposals_overlay_{image_stem}.jpg",
            f"reference_roi_regions_overlay_{image_stem}.jpg",
            f"blue_inside_cells_{image_stem}.tif",
        ])
    return outputs


def windows_short_path(path: Path) -> str:
    # Keep Fiji robust with Unicode filenames when Windows 8.3 short names are available.
    if os.name != "nt":
        return str(path)
    escaped = str(path).replace("'", "''")
    script = (
        "$fso = New-Object -ComObject Scripting.FileSystemObject; "
        f"$f = $fso.GetFile('{escaped}'); "
        "$f.ShortPath"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        text=True,
        capture_output=True,
    )
    value = completed.stdout.strip()
    if completed.returncode == 0 and value and Path(value).exists() and '"' not in value:
        return value
    return str(path)


def group_name_for(image: Path, project: Path) -> str:
    try:
        relative = image.resolve().relative_to((project / "input").resolve())
        if len(relative.parts) > 1:
            return relative.parts[0]
        return "mvp_input"
    except ValueError:
        return image.parent.name or "unknown"


def create_clean_output(root: Path, prefix: str, clean: bool) -> Path:
    if clean:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        return root
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = root / f"{prefix}_{run_id}"
    suffix = 2
    while output.exists():
        output = root / f"{prefix}_{run_id}_{suffix}"
        suffix += 1
    output.mkdir(parents=True)
    return output


def ascii_work_suffix(image: Path) -> str:
    suffix = image.suffix.lower()
    if suffix and suffix.isascii() and suffix.replace(".", "", 1).isalnum():
        return suffix
    return ".img"


def prepare_fiji_input(original_image: Path, output: Path) -> Path:
    work_dir = output / "fiji_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    fiji_input = work_dir / ("input" + ascii_work_suffix(original_image))
    shutil.copy2(original_image, fiji_input)
    return fiji_input


def build_macro_arg(fiji_input: Path, original_image: Path, output: Path, project: Path, params: dict[str, str]) -> tuple[str, str]:
    short_input = windows_short_path(fiji_input)
    short_used = "true" if short_input != str(fiji_input) else "false"
    macro_params = {
        "input": short_input,
        "output": str(output),
        "original_long_path": str(original_image),
        "original_file_name": original_image.name,
        "group_name": group_name_for(original_image, project),
        "short_path_used": short_used,
        **params,
    }
    return ";".join(f"{k}={v}" for k, v in macro_params.items()), short_used


def write_run_parameters(output: Path, project: Path, original_image: Path, fiji_input: Path, fiji: Path, macro: Path, macro_arg: str, params: dict[str, str]) -> None:
    outputs = expected_outputs(params, original_image.stem)
    lines = [
        f"project={project}",
        f"input={original_image}",
        f"fiji_input={fiji_input}",
        f"output={output}",
        f"macro={macro}",
        f"fiji={fiji}",
        f"macro_argument_string={macro_arg}",
        "expected_outputs=" + ";".join(outputs),
    ]
    lines.extend(f"{key}={value}" for key, value in params.items())
    (output / "run_parameters.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (output / "run_parameters.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        for key, value in [
            ("project", project),
            ("input", original_image),
            ("fiji_input", fiji_input),
            ("output", output),
            ("macro", macro),
            ("fiji", fiji),
            ("expected_outputs", ";".join(outputs)),
        ]:
            writer.writerow([key, str(value)])
        for key, value in params.items():
            writer.writerow([key, value])



def make_frame_id(image_stem: str) -> str:
    """Return a stable ASCII-safe frame id derived from the original Unicode stem."""
    ascii_slug = re.sub(r"[^A-Za-z0-9]+", "_", image_stem).strip("_").lower()
    if not ascii_slug:
        ascii_slug = "image"
    ascii_slug = ascii_slug[:80].strip("_") or "image"
    digest = zlib.crc32(image_stem.encode("utf-8")) & 0xFFFFFFFF
    return f"frame_{ascii_slug}_{digest:08x}"


def contains_replacement_questions(value: str | None) -> bool:
    return "????" in (value or "")


def rewrite_csv_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_public_csv_metadata(path: Path, original_image: Path, frame_id: str, short_path_used: str, require_object_ids: bool) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for column in ["image_name", "original_long_path", "short_path_used", "frame_id"]:
        if column not in fieldnames:
            fieldnames.append(column)

    for row in rows:
        row["image_name"] = original_image.name
        row["original_long_path"] = str(original_image)
        row["short_path_used"] = short_path_used
        row["frame_id"] = frame_id
        if require_object_ids and "feature_row_id" in fieldnames:
            object_id = str(row.get("object_id") or "").strip()
            if object_id and object_id.lower() != "frame":
                row["feature_row_id"] = f"{frame_id}_object_{object_id}"

    rewrite_csv_rows(path, rows, fieldnames)


def normalize_public_xlsx_metadata(path: Path, original_image: Path, frame_id: str, short_path_used: str) -> None:
    if importlib.util.find_spec("openpyxl") is None or not path.exists():
        return
    from openpyxl import load_workbook

    workbook = load_workbook(path)
    changed = False
    replacements = {
        "image_name": original_image.name,
        "original_long_path": str(original_image),
        "short_path_used": short_path_used,
        "frame_id": frame_id,
    }
    for worksheet in workbook.worksheets:
        headers = [cell.value for cell in worksheet[1]]
        for column_name, value in replacements.items():
            if column_name in headers:
                column_index = headers.index(column_name) + 1
                for row_index in range(2, worksheet.max_row + 1):
                    worksheet.cell(row=row_index, column=column_index).value = value
                changed = True
    if changed:
        workbook.save(path)


def validate_public_metadata(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row_number, row in enumerate(csv.DictReader(f), start=2):
            for column in ("image_name", "original_long_path"):
                if contains_replacement_questions(row.get(column)):
                    raise ValueError(f"Final public metadata contains replacement question marks in {path.name}:{row_number}:{column}")


def normalize_public_outputs_metadata(output: Path, image_stem: str, original_image: Path, short_path_used: str) -> None:
    frame_id = make_frame_id(image_stem)
    public_tables = [
        (output / f"cell_features_{image_stem}.csv", True),
        (output / f"frame_features_{image_stem}.csv", False),
    ]
    for path, require_object_ids in public_tables:
        if path.exists():
            normalize_public_csv_metadata(path, original_image, frame_id, short_path_used, require_object_ids)
            validate_public_metadata(path)

    blue_table = output / f"blue_table_{image_stem}.xlsx"
    normalize_public_xlsx_metadata(blue_table, original_image, frame_id, short_path_used)


def read_single_csv_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def validate_expected_outputs(output: Path, params: dict[str, str], started: datetime, image_stem: str | None = None) -> tuple[list[str], list[str], list[str]]:
    missing: list[str] = []
    stale: list[str] = []
    empty: list[str] = []
    for name in expected_outputs(params, image_stem):
        path = output / name
        if not path.exists():
            missing.append(name)
        elif name not in RUNNER_WRITTEN_OUTPUTS and datetime.fromtimestamp(path.stat().st_mtime) < started:
            stale.append(name)
        elif path.stat().st_size == 0:
            empty.append(name)
    return missing, stale, empty


def validate_named_outputs(output: Path, names: list[str], started: datetime) -> tuple[list[str], list[str], list[str]]:
    missing: list[str] = []
    stale: list[str] = []
    empty: list[str] = []
    for name in names:
        path = output / name
        if not path.exists():
            missing.append(name)
        elif datetime.fromtimestamp(path.stat().st_mtime) < started:
            stale.append(name)
        elif path.stat().st_size == 0:
            empty.append(name)
    return missing, stale, empty


def write_blue_pixels_xlsx(output: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    csv_path = output / "blue_pixels_features.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    object_rows = [
        row for row in rows
        if row.get("object_id") != "frame" and not row.get("object_type", "").startswith("all_")
    ]
    for row in object_rows:
        object_pixels = float(row.get("object_pixels") or 0)
        blue_pixels = float(row.get("blue_pixels") or 0)
        if blue_pixels > object_pixels:
            raise ValueError(f"blue_pixels exceeds object_pixels for object_id={row.get('object_id')}")
    object_rows.sort(key=lambda row: float(row.get("blue_pixel_percent") or 0), reverse=True)
    columns = [
        "image_name",
        "group_name",
        "object_type",
        "object_id",
        "object_pixels",
        "blue_pixels",
        "blue_pixel_fraction",
        "blue_pixel_percent",
    ]

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "blue_pixels_features"
    worksheet.append(columns)
    for row in object_rows:
        worksheet.append([coerce_cell(row.get(column, "")) for column in columns])

    total_object_pixels = sum(float(row.get("object_pixels") or 0) for row in object_rows)
    total_blue_pixels = sum(float(row.get("blue_pixels") or 0) for row in object_rows)
    total_fraction = total_blue_pixels / total_object_pixels if total_object_pixels else 0
    worksheet.append([
        "SUM",
        "",
        "all_accepted_cell_material",
        "",
        int(total_object_pixels),
        int(total_blue_pixels),
        total_fraction,
        100 * total_fraction,
    ])

    header_fill = PatternFill("solid", fgColor="D9EAF7")
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
    for cell in worksheet[worksheet.max_row]:
        cell.font = Font(bold=True)
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    for column_cells in worksheet.columns:
        width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells) + 2
        worksheet.column_dimensions[get_column_letter(column_cells[0].column)].width = min(max(width, 10), 40)
    for row in worksheet.iter_rows(min_row=2, min_col=7, max_col=8):
        row[0].number_format = "0.00000000"
        row[1].number_format = "0.0000"
    workbook.save(output / "blue_table.xlsx")


def coerce_cell(value: str) -> Any:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return int(number)
    return number


def read_image_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(32)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")
        if header[:2] == b"BM":
            return int.from_bytes(header[18:22], "little", signed=True), abs(int.from_bytes(header[22:26], "little", signed=True))
        if header[:2] in {b"II", b"MM"}:
            endian = "little" if header[:2] == b"II" else "big"
            f.seek(int.from_bytes(header[4:8], endian))
            count = int.from_bytes(f.read(2), endian)
            width = height = 0
            for _ in range(count):
                entry = f.read(12)
                tag = int.from_bytes(entry[0:2], endian)
                value = int.from_bytes(entry[8:12], endian)
                if tag == 256:
                    width = value
                elif tag == 257:
                    height = value
            if width and height:
                return width, height
        if header[:2] == b"\xff\xd8":
            f.seek(2)
            while True:
                marker = f.read(2)
                while marker[:1] != b"\xff":
                    marker = marker[1:] + f.read(1)
                code = marker[1]
                length = int.from_bytes(f.read(2), "big")
                if 0xC0 <= code <= 0xC3 or 0xC5 <= code <= 0xC7 or 0xC9 <= code <= 0xCB or 0xCD <= code <= 0xCF:
                    data = f.read(5)
                    return int.from_bytes(data[3:5], "big"), int.from_bytes(data[1:3], "big")
                f.seek(length - 2, 1)
    raise ValueError(f"Unsupported image format for dimension validation: {path}")


def read_tiff_mask_stats(path: Path) -> dict[str, int]:
    if importlib.util.find_spec("PIL") is not None:
        return read_tiff_mask_stats_with_pillow(path)
    if importlib.util.find_spec("tifffile") is not None:
        return read_tiff_mask_stats_with_tifffile(path)
    if importlib.util.find_spec("imageio") is not None:
        return read_tiff_mask_stats_with_imageio(path)
    return read_uncompressed_tiff_mask_stats(path)


def mask_array_stats(array: Any, path: Path) -> dict[str, int]:
    shape = array.shape
    if len(shape) == 3 and shape[-1] != 1:
        raise ValueError(f"Mask appears to have {shape[-1]} samples per pixel, expected one channel: {path}")
    if len(shape) == 3:
        array = array[..., 0]
    height, width = array.shape[:2]
    unique_values = set()
    for value in array.flat:
        unique_values.add(int(value))
        if len(unique_values) > 2:
            raise ValueError(f"Mask is not binary/binary-equivalent: {path}")
    nonzero = int((array != 0).sum())
    bits = int(getattr(array.dtype, "itemsize", 1) * 8)
    return {"width": int(width), "height": int(height), "nonzero": nonzero, "samples": 1, "bits": bits}


def read_tiff_mask_stats_with_tifffile(path: Path) -> dict[str, int]:
    import tifffile

    return mask_array_stats(tifffile.imread(path), path)


def read_tiff_mask_stats_with_imageio(path: Path) -> dict[str, int]:
    import imageio.v3 as iio

    return mask_array_stats(iio.imread(path), path)


def read_tiff_mask_stats_with_pillow(path: Path) -> dict[str, int]:
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
        bands = image.getbands()
        samples = len(bands)
        if samples != 1:
            raise ValueError(f"Mask appears to have {samples} samples per pixel, expected one channel: {path}")
        bits = image.tag_v2.get(258, 8)
        if isinstance(bits, tuple):
            bits = bits[0]
        gray = image.convert("L")
        histogram = gray.histogram()
        unique_values = sum(1 for count in histogram if count > 0)
        if unique_values > 2:
            raise ValueError(f"Mask is not binary/binary-equivalent: {path}")
        nonzero = sum(histogram[1:])
    return {"width": width, "height": height, "nonzero": nonzero, "samples": samples, "bits": int(bits)}


def read_uncompressed_tiff_mask_stats(path: Path) -> dict[str, int]:
    data = path.read_bytes()
    if data[:2] not in {b"II", b"MM"}:
        raise ValueError(f"Expected TIFF mask: {path}")
    endian = "little" if data[:2] == b"II" else "big"
    ifd = int.from_bytes(data[4:8], endian)
    count = int.from_bytes(data[ifd:ifd + 2], endian)
    tags: dict[int, tuple[int, int, int]] = {}
    pos = ifd + 2
    for _ in range(count):
        entry = data[pos:pos + 12]
        tag = int.from_bytes(entry[0:2], endian)
        typ = int.from_bytes(entry[2:4], endian)
        num = int.from_bytes(entry[4:8], endian)
        val = int.from_bytes(entry[8:12], endian)
        tags[tag] = (typ, num, val)
        pos += 12

    width = tags.get(256, (0, 0, 0))[2]
    height = tags.get(257, (0, 0, 0))[2]
    bits = tags.get(258, (0, 0, 8))[2]
    compression = tags.get(259, (0, 0, 1))[2]
    samples = tags.get(277, (0, 0, 1))[2]
    if compression != 1:
        raise ValueError(f"Compressed TIFF mask requires Pillow for QC validation: {path}")
    if samples != 1:
        raise ValueError(f"Mask appears to have {samples} samples per pixel, expected one channel: {path}")
    strip_offset = tags.get(273, (0, 0, 0))[2]
    strip_count = tags.get(279, (0, 0, 0))[2]
    if not strip_offset or not strip_count:
        raise ValueError(f"TIFF mask is missing strip offsets/counts: {path}")
    pixels = data[strip_offset:strip_offset + strip_count]
    unique_values = set()
    if bits == 8:
        nonzero = 0
        for value in pixels:
            unique_values.add(int(value))
            if len(unique_values) > 2:
                raise ValueError(f"Mask is not binary/binary-equivalent: {path}")
            if value != 0:
                nonzero += 1
    elif bits == 16:
        nonzero = 0
        for i in range(0, len(pixels), 2):
            value = int.from_bytes(pixels[i:i+2], endian)
            unique_values.add(value)
            if len(unique_values) > 2:
                raise ValueError(f"Mask is not binary/binary-equivalent: {path}")
            if value != 0:
                nonzero += 1
    else:
        raise ValueError(f"Unsupported TIFF mask bit depth {bits}: {path}")
    return {"width": width, "height": height, "nonzero": nonzero, "samples": samples, "bits": bits}


def images_effectively_same(path_a: Path, path_b: Path) -> bool:
    if importlib.util.find_spec("PIL") is None:
        return False
    from PIL import Image, ImageChops, ImageStat
    with Image.open(path_a) as image_a, Image.open(path_b) as image_b:
        image_a = image_a.convert("RGB")
        image_b = image_b.convert("RGB")
        if image_a.size != image_b.size:
            return False
        diff = ImageChops.difference(image_a, image_b)
        stat = ImageStat.Stat(diff)
        return max(stat.mean) < 0.5


def validate_frame_summary(output: Path) -> dict[str, float]:
    summary_path = output / "final_frame_summary.csv"
    if not summary_path.exists():
        summary_path = output / "_internal" / "final_frame_summary.csv"
    summary = read_single_csv_row(summary_path)
    accepted_objects = float(summary.get("accepted_object_count") or 0)
    accepted_pixels = float(summary.get("accepted_object_pixels") or 0)
    accepted_blue = float(summary.get("accepted_blue_pixels") or 0)
    if accepted_blue > accepted_pixels:
        raise ValueError("accepted_blue_pixels exceeds accepted_object_pixels")
    if accepted_objects == 0:
        raise ValueError("accepted_object_count is zero for this expected-positive single-frame run")
    return {
        "accepted_object_count": accepted_objects,
        "accepted_object_pixels": accepted_pixels,
        "accepted_blue_pixels": accepted_blue,
    }


def validate_output_image_dimensions(output: Path, image_stem: str, original_image: Path, params: dict[str, str]) -> None:
    if not bool_param(params.get("save_overlays", "true")):
        return
    expected = read_image_dimensions(original_image)
    final_names = [
        f"cellmask_{image_stem}.tif",
        f"blue_inside_cells_{image_stem}.tif",
        f"vis_cellpixels_{image_stem}.png",
        f"roi_overlay_{image_stem}.jpg",
        f"selected_objects_overlay_{image_stem}.jpg",
    ]
    for name in final_names:
        actual = read_image_dimensions(output / name)
        if actual != expected:
            raise ValueError(f"{name} dimensions {actual} do not match original image dimensions {expected}")
    cellmask_stats = read_tiff_mask_stats(output / f"cellmask_{image_stem}.tif")
    blue_stats = read_tiff_mask_stats(output / f"blue_inside_cells_{image_stem}.tif")
    if cellmask_stats["nonzero"] == 0:
        raise ValueError("accepted cellmask is empty")
    if blue_stats["nonzero"] > cellmask_stats["nonzero"]:
        raise ValueError("blue_inside_cells has more nonzero pixels than cellmask")
    if images_effectively_same(original_image, output / f"vis_cellpixels_{image_stem}.png"):
        raise ValueError("vis_cellpixels appears unchanged from the original input")
    if images_effectively_same(original_image, output / f"roi_overlay_{image_stem}.jpg"):
        raise ValueError("roi_overlay appears unchanged from the original input")


def validate_cell_feature_table(output: Path) -> None:
    path = output / "cell_features.csv"
    if not path.exists():
        candidates = sorted(output.glob("cell_features_*.csv"))
        if candidates:
            path = candidates[0]
        else:
            path = output / "_internal" / "cell_features.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            object_pixels = float(row.get("object_pixels") or 0)
            blue_pixels = float(row.get("blue_pixels") or 0)
            roi_area_pixels = float(row.get("roi_area_pixels") or 0)
            fraction = float(row.get("blue_pixel_fraction") or 0)
            percent = float(row.get("blue_pixel_percent") or 0)
            if blue_pixels > object_pixels:
                raise ValueError(f"blue_pixels exceeds object_pixels for object_id={row.get('object_id')}")
            if object_pixels > roi_area_pixels:
                raise ValueError(f"object_pixels exceeds roi_area_pixels for object_id={row.get('object_id')}")
            expected_fraction = blue_pixels / object_pixels if object_pixels else 0
            if abs(fraction - expected_fraction) > 1e-6 or abs(percent - 100 * expected_fraction) > 1e-3:
                raise ValueError(f"blue fraction/percent mismatch for object_id={row.get('object_id')}")


def copy_final_named_outputs(output: Path, image_stem: str) -> None:
    for source_name, target_template in FINAL_OUTPUT_MAP.items():
        source = output / source_name
        if source.exists():
            shutil.copy2(source, output / target_template.format(image_stem=image_stem))


def move_internal_outputs(output: Path) -> None:
    internal = output / "_internal"
    internal.mkdir(exist_ok=True)
    for name in set(ALWAYS_EXPECTED_OUTPUTS + OVERLAY_EXPECTED_OUTPUTS + DEBUG_OUTPUTS + ["blue_table.xlsx", "fiji_stdout.txt", "fiji_stderr.txt", "runner_command.txt", "fiji_work"]):
        source = output / name
        if source.exists():
            target = internal / name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(source), str(target))


def write_selected_contact_sheet(output: Path, image_stem: str, original_image: Path) -> None:
    from PIL import Image, ImageDraw

    if read_selection_unit_type(output, image_stem) == "reference_roi_regions":
        regions_path = output / f"reference_roi_regions_{image_stem}.csv"
        with regions_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        row_kind = "reference_roi"
    else:
        features_path = output / f"cell_features_{image_stem}.csv"
        if not features_path.exists():
            features_path = output / "cell_features.csv"
        with features_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = [row for row in csv.DictReader(f) if str(row.get("selected_for_frame_summary", "")).lower() == "true"]
        row_kind = "object"

    thumb_w = 220
    thumb_h = 160
    label_h = 72
    cols = 4
    rows_count = max(1, (len(rows) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows_count * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    with Image.open(original_image) as source:
        source = source.convert("RGB")
        for idx, row in enumerate(rows):
            if row_kind == "reference_roi":
                bbox_x = int(float(row.get("roi_inner_x") or 0))
                bbox_y = int(float(row.get("roi_inner_y") or 0))
                bbox_w = int(float(row.get("roi_inner_w") or 1))
                bbox_h = int(float(row.get("roi_inner_h") or 1))
            else:
                bbox_x = int(float(row.get("bbox_x") or 0))
                bbox_y = int(float(row.get("bbox_y") or 0))
                bbox_w = int(float(row.get("bbox_w") or row.get("bbox_width") or 1))
                bbox_h = int(float(row.get("bbox_h") or row.get("bbox_height") or 1))
            pad = 12
            left = max(0, bbox_x - pad)
            top = max(0, bbox_y - pad)
            right = min(source.width, bbox_x + bbox_w + pad)
            bottom = min(source.height, bbox_y + bbox_h + pad)
            crop = source.crop((left, top, right, bottom))
            crop.thumbnail((thumb_w, thumb_h))
            col = idx % cols
            row_index = idx // cols
            x0 = col * thumb_w
            y0 = row_index * (thumb_h + label_h)
            sheet.paste(crop, (x0 + (thumb_w - crop.width) // 2, y0))
            label = (
                f"#{row.get('object_id')} {row.get('feature_row_id')}\n"
                f"px={row.get('object_pixels')} blue%={row.get('blue_pixel_percent')}\n"
                f"bbox=({bbox_x},{bbox_y},{bbox_w},{bbox_h})"
            )
            draw.multiline_text((x0 + 4, y0 + thumb_h + 4), label, fill=(0, 0, 0), spacing=2)

    if not rows:
        draw.text((10, 10), "No selected frame-summary objects", fill=(0, 0, 0))
    sheet.save(output / f"selected_objects_contact_sheet_{image_stem}.jpg", quality=90)


def as_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def append_qc_status(qc_status: str, warning: str) -> str:
    if not warning:
        return qc_status
    parts = [part for part in str(qc_status or "PASS").split(";") if part and part != "PASS"]
    if warning not in parts:
        parts.append(warning)
    return ";".join(parts) if parts else "PASS"


def bbox_intersection_fraction(row: dict[str, str], roi: dict[str, Any]) -> float:
    x = as_float(row.get("bbox_x"))
    y = as_float(row.get("bbox_y"))
    w = max(1.0, as_float(row.get("bbox_w") or row.get("bbox_width"), 1.0))
    h = max(1.0, as_float(row.get("bbox_h") or row.get("bbox_height"), 1.0))
    rx = float(roi["roi_bbox_x"])
    ry = float(roi["roi_bbox_y"])
    rw = float(roi["roi_bbox_w"])
    rh = float(roi["roi_bbox_h"])
    ix = max(0.0, min(x + w, rx + rw) - max(x, rx))
    iy = max(0.0, min(y + h, ry + rh) - max(y, ry))
    return (ix * iy) / max(1.0, w * h)


def derive_auto_roi_proposals(rows: list[dict[str, str]], image_name: str, frame_id: str, rejected_rows: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    accepted = [row for row in rows if str(row.get("accepted_status", "")).strip().lower() == "accepted_cell_candidate"]
    seeds = list(accepted)
    for row in rejected_rows or []:
        classification = str(row.get("classification", "")).strip().lower()
        reject_reason = str(row.get("reject_reason", "")).strip().lower()
        area = as_float(row.get("area_px"))
        if reject_reason == "reject_large_rectangular_artifact" and area >= 100000 and classification != "border_object":
            seed = dict(row)
            seed["object_pixels"] = str(round(area))
            seed["blue_pixels"] = "0"
            seed["roi_seed_status"] = "roi_seed_candidate"
            seed["roi_seed_reason"] = "roi_seed_large_aggregate_candidate"
            seed["used_for_auto_roi_proposal"] = "true"
            seeds.append(seed)
    if not seeds:
        return []
    margin = 96.0
    parents = list(range(len(seeds)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    boxes: list[tuple[float, float, float, float]] = []
    expanded: list[tuple[float, float, float, float]] = []
    for row in seeds:
        x = as_float(row.get("bbox_x"))
        y = as_float(row.get("bbox_y"))
        w = max(1.0, as_float(row.get("bbox_w") or row.get("bbox_width"), 1.0))
        h = max(1.0, as_float(row.get("bbox_h") or row.get("bbox_height"), 1.0))
        boxes.append((x, y, x + w, y + h))
        expanded.append((x - margin, y - margin, x + w + margin, y + h + margin))
    for i in range(len(expanded)):
        ax1, ay1, ax2, ay2 = expanded[i]
        for j in range(i + 1, len(expanded)):
            bx1, by1, bx2, by2 = expanded[j]
            if min(ax2, bx2) >= max(ax1, bx1) and min(ay2, by2) >= max(ay1, by1):
                union(i, j)
    clusters: dict[int, list[int]] = {}
    for index in range(len(seeds)):
        clusters.setdefault(find(index), []).append(index)

    proposals: list[dict[str, Any]] = []
    for members in clusters.values():
        total_pixels = sum(as_float(seeds[index].get("object_pixels")) for index in members)
        total_blue = sum(as_float(seeds[index].get("blue_pixels")) for index in members)
        if len(members) < 3 and total_pixels < 3000:
            continue
        x1 = min(boxes[index][0] for index in members)
        y1 = min(boxes[index][1] for index in members)
        x2 = max(boxes[index][2] for index in members)
        y2 = max(boxes[index][3] for index in members)
        large_seed_count = 0
        seed_object_ids: list[str] = []
        seed_reasons: list[str] = []
        for index in members:
            seed_object_ids.append(str(seeds[index].get("object_id", "")))
            seed_reasons.append(str(seeds[index].get("roi_seed_reason", "roi_seed_clustered_cell_material") or "roi_seed_clustered_cell_material"))
            if str(seeds[index].get("roi_seed_reason", "")) == "roi_seed_large_aggregate_candidate":
                large_seed_count += 1
        near_full = total_pixels > 0 and (100 * total_blue / total_pixels) >= 99
        score = total_pixels + 500 * len(members) + 250000 * large_seed_count
        note = "roi_seed_clustered_cell_material"
        if large_seed_count > 0:
            note = "roi_seed_large_aggregate_candidate"
        if near_full and large_seed_count == 0:
            note = "roi_reject_diffuse_full_blue_region"
        proposals.append({
            "image_name": image_name,
            "frame_id": frame_id,
            "roi_bbox_x": round(x1),
            "roi_bbox_y": round(y1),
            "roi_bbox_w": max(1, round(x2 - x1)),
            "roi_bbox_h": max(1, round(y2 - y1)),
            "roi_candidate_object_count": len(members),
            "roi_total_object_pixels": round(total_pixels),
            "roi_total_blue_pixels": round(total_blue),
            "roi_blue_pixel_percent": 100 * total_blue / total_pixels if total_pixels > 0 else 0,
            "roi_selection_score": score,
            "roi_warn_near_full_blue_dominated": "true" if near_full else "false",
            "roi_seed_object_ids": "|".join(seed_object_ids),
            "roi_seed_reasons": "|".join(seed_reasons),
            "reference_roi_id": "",
            "overlaps_reference_roi": "false",
            "reference_roi_overlap_fraction": "0.0000",
            "roi_review_note": note,
        })
    proposals = [proposal for proposal in proposals if proposal.get("roi_review_note") != "roi_reject_diffuse_full_blue_region"]
    proposals.sort(key=lambda roi: (float(roi["roi_selection_score"]), int(roi["roi_candidate_object_count"])), reverse=True)
    proposals = proposals[:3]
    for index, proposal in enumerate(proposals, start=1):
        proposal["roi_id"] = f"auto_roi_{index}"
    return proposals


def write_roi_proposals_csv(output: Path, image_stem: str, proposals: list[dict[str, Any]]) -> None:
    fieldnames = [
        "image_name",
        "frame_id",
        "roi_id",
        "roi_bbox_x",
        "roi_bbox_y",
        "roi_bbox_w",
        "roi_bbox_h",
        "roi_candidate_object_count",
        "roi_total_object_pixels",
        "roi_total_blue_pixels",
        "roi_blue_pixel_percent",
        "roi_selection_score",
        "roi_warn_near_full_blue_dominated",
        "roi_seed_object_ids",
        "roi_seed_reasons",
        "reference_roi_id",
        "overlaps_reference_roi",
        "reference_roi_overlap_fraction",
        "roi_review_note",
    ]
    with (output / f"roi_proposals_{image_stem}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for proposal in proposals:
            writer.writerow({name: proposal.get(name, "") for name in fieldnames})


def write_roi_proposals_overlay(output: Path, image_stem: str, original_image: Path, proposals: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw

    with Image.open(original_image) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for proposal in proposals:
        x = int(proposal["roi_bbox_x"])
        y = int(proposal["roi_bbox_y"])
        w = int(proposal["roi_bbox_w"])
        h = int(proposal["roi_bbox_h"])
        draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 255), width=6)
        draw.text((x + 4, max(0, y - 18)), str(proposal["roi_id"]), fill=(255, 0, 255))
    image.save(output / f"roi_proposals_overlay_{image_stem}.jpg", quality=90)


def rewrite_frame_summary_selection(output: Path, image_stem: str, selected_rows: list[dict[str, str]], proposals: list[dict[str, Any]], primary_roi: dict[str, Any] | None = None) -> str:
    summary_path = output / f"frame_features_{image_stem}.csv"
    if not summary_path.exists():
        return ""
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        return ""
    for column in ["primary_auto_roi_id", "primary_auto_roi_reason", "auto_roi_overlaps_reference_roi", "selected_reference_roi_overlap_fraction"]:
        if column not in fieldnames:
            fieldnames.append(column)
    row = rows[0]
    selected_pixels = sum(as_float(item.get("object_pixels")) for item in selected_rows)
    selected_blue = sum(as_float(item.get("blue_pixels")) for item in selected_rows)
    selected_fraction = selected_blue / selected_pixels if selected_pixels > 0 else 0
    row["selected_frame_object_count"] = str(len(selected_rows))
    row["selected_frame_object_ids"] = "|".join(str(item.get("object_id", "")) for item in selected_rows)
    row["selected_object_pixels"] = f"{selected_pixels:.0f}"
    row["selected_blue_pixels"] = f"{selected_blue:.0f}"
    row["selected_blue_pixel_percent"] = f"{100 * selected_fraction:.4f}"
    qc_status = row.get("qc_status", "PASS")
    if not proposals:
        qc_status = append_qc_status(qc_status, "WARN_NO_AUTO_ROI_PROPOSALS")
    elif any(str(item.get("inside_auto_roi", "")).lower() != "true" for item in selected_rows):
        qc_status = append_qc_status(qc_status, "WARN_SELECTED_OUTSIDE_AUTO_ROI")
    row["primary_auto_roi_id"] = "" if primary_roi is None else str(primary_roi.get("roi_id", ""))
    row["primary_auto_roi_reason"] = "" if primary_roi is None else str(primary_roi.get("roi_review_note", ""))
    row["auto_roi_overlaps_reference_roi"] = "" if primary_roi is None else str(primary_roi.get("overlaps_reference_roi", "false"))
    row["selected_reference_roi_overlap_fraction"] = ""
    row["qc_status"] = qc_status
    rewrite_csv_rows(summary_path, rows, fieldnames)
    return qc_status


def derive_reference_roi_regions(rejected_rows: list[dict[str, str]], accepted_rows: list[dict[str, str]], image_name: str, frame_id: str) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for row in rejected_rows:
        classification = str(row.get("classification", "")).strip().lower()
        reject_reason = str(row.get("reject_reason", "")).strip().lower()
        area = as_float(row.get("area_px"))
        if reject_reason != "reject_large_rectangular_artifact" or area < 100000 or classification == "border_object":
            continue
        x = round(as_float(row.get("bbox_x")))
        y = round(as_float(row.get("bbox_y")))
        w = max(1, round(as_float(row.get("bbox_w") or row.get("bbox_width"), 1)))
        h = max(1, round(as_float(row.get("bbox_h") or row.get("bbox_height"), 1)))
        margin = max(8, min(24, round(min(w, h) * 0.03)))
        inner_x = x + margin
        inner_y = y + margin
        inner_w = max(1, w - 2 * margin)
        inner_h = max(1, h - 2 * margin)
        region = {
            "image_name": image_name,
            "frame_id": frame_id,
            "reference_roi_id": f"{frame_id}_reference_roi_{len(regions) + 1}",
            "source_object_id": str(row.get("object_id", "")),
            "source_reason": "large_rejected_aggregate_roi_seed",
            "roi_bbox_x": x,
            "roi_bbox_y": y,
            "roi_bbox_w": w,
            "roi_bbox_h": h,
            "roi_inner_x": inner_x,
            "roi_inner_y": inner_y,
            "roi_inner_w": inner_w,
            "roi_inner_h": inner_h,
            "roi_area_pixels": inner_w * inner_h,
            "roi_blue_pixels": 0.0,
            "roi_blue_pixel_percent": 0.0,
            "roi_review_note": "reference_roi_region_from_large_aggregate_seed",
        }
        blue_pixels = 0.0
        for accepted in accepted_rows:
            overlap = bbox_intersection_fraction(accepted, {"roi_bbox_x": inner_x, "roi_bbox_y": inner_y, "roi_bbox_w": inner_w, "roi_bbox_h": inner_h})
            if overlap > 0:
                blue_pixels += as_float(accepted.get("blue_pixels")) * overlap
        region["roi_blue_pixels"] = round(blue_pixels)
        region["roi_blue_pixel_percent"] = 100 * blue_pixels / max(1, inner_w * inner_h)
        regions.append(region)
    return regions


def write_reference_roi_regions_csv(output: Path, image_stem: str, regions: list[dict[str, Any]]) -> None:
    fieldnames = [
        "image_name", "frame_id", "reference_roi_id", "source_object_id", "source_reason",
        "roi_bbox_x", "roi_bbox_y", "roi_bbox_w", "roi_bbox_h",
        "roi_inner_x", "roi_inner_y", "roi_inner_w", "roi_inner_h",
        "roi_area_pixels", "roi_blue_pixels", "roi_blue_pixel_percent", "roi_review_note",
    ]
    with (output / f"reference_roi_regions_{image_stem}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for region in regions:
            writer.writerow({name: region.get(name, "") for name in fieldnames})


def write_reference_roi_regions_overlay(output: Path, image_stem: str, original_image: Path, regions: list[dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw

    with Image.open(original_image) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for region in regions:
        x = int(region["roi_bbox_x"]); y = int(region["roi_bbox_y"]); w = int(region["roi_bbox_w"]); h = int(region["roi_bbox_h"])
        ix = int(region["roi_inner_x"]); iy = int(region["roi_inner_y"]); iw = int(region["roi_inner_w"]); ih = int(region["roi_inner_h"])
        draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=5)
        draw.rectangle((ix, iy, ix + iw, iy + ih), outline=(255, 255, 0), width=4)
        draw.text((x + 4, max(0, y - 18)), str(region["reference_roi_id"]), fill=(255, 0, 0))
    image.save(output / f"reference_roi_regions_overlay_{image_stem}.jpg", quality=90)


def read_selection_unit_type(output: Path, image_stem: str) -> str:
    summary_path = output / f"frame_features_{image_stem}.csv"
    if not summary_path.exists():
        return "object_candidates"
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0].get("selection_unit_type", "object_candidates") if rows else "object_candidates"


def rewrite_frame_summary_reference_regions(output: Path, image_stem: str, regions: list[dict[str, Any]], accepted_rows: list[dict[str, str]], primary_roi: dict[str, Any] | None) -> str:
    summary_path = output / f"frame_features_{image_stem}.csv"
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        return ""
    for column in [
        "selected_reference_roi_count", "selected_reference_roi_ids", "selected_reference_roi_pixels",
        "selected_reference_roi_blue_pixels", "selected_reference_roi_blue_pixel_percent", "selection_unit_type",
        "primary_auto_roi_id", "primary_auto_roi_reason", "auto_roi_overlaps_reference_roi", "selected_reference_roi_overlap_fraction",
    ]:
        if column not in fieldnames:
            fieldnames.append(column)
    total_pixels = sum(as_float(str(region.get("roi_area_pixels"))) for region in regions)
    total_blue = sum(as_float(str(region.get("roi_blue_pixels"))) for region in regions)
    overlap_count = sum(1 for row in accepted_rows if row.get("inside_reference_roi") == "true")
    qc_status = append_qc_status(rows[0].get("qc_status", "PASS"), "INFO_REFERENCE_ROI_REGIONS_USED")
    if overlap_count == 0:
        qc_status = append_qc_status(qc_status, "WARN_NO_OBJECT_CANDIDATES_INSIDE_REFERENCE_ROI")
    rows[0]["selected_frame_object_count"] = str(len(regions))
    rows[0]["selected_frame_object_ids"] = "|".join(str(region.get("reference_roi_id", "")) for region in regions)
    rows[0]["selected_object_pixels"] = f"{total_pixels:.0f}"
    rows[0]["selected_blue_pixels"] = f"{total_blue:.0f}"
    rows[0]["selected_blue_pixel_percent"] = f"{(100 * total_blue / total_pixels) if total_pixels > 0 else 0:.4f}"
    rows[0]["selected_reference_roi_count"] = str(len(regions))
    rows[0]["selected_reference_roi_ids"] = "|".join(str(region.get("reference_roi_id", "")) for region in regions)
    rows[0]["selected_reference_roi_pixels"] = f"{total_pixels:.0f}"
    rows[0]["selected_reference_roi_blue_pixels"] = f"{total_blue:.0f}"
    rows[0]["selected_reference_roi_blue_pixel_percent"] = f"{(100 * total_blue / total_pixels) if total_pixels > 0 else 0:.4f}"
    rows[0]["selection_unit_type"] = "reference_roi_regions"
    rows[0]["primary_auto_roi_id"] = "" if primary_roi is None else str(primary_roi.get("roi_id", ""))
    rows[0]["primary_auto_roi_reason"] = "" if primary_roi is None else str(primary_roi.get("roi_review_note", ""))
    rows[0]["auto_roi_overlaps_reference_roi"] = "true"
    rows[0]["selected_reference_roi_overlap_fraction"] = "1.0000"
    rows[0]["qc_status"] = qc_status
    rewrite_csv_rows(summary_path, rows, fieldnames)
    return qc_status


def apply_auto_roi_selection(output: Path, image_stem: str, original_image: Path, params: dict[str, str]) -> tuple[list[dict[str, Any]], str]:
    features_path = output / f"cell_features_{image_stem}.csv"
    with features_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for column in ["auto_roi_id", "inside_auto_roi", "auto_roi_overlap_fraction", "auto_roi_selection_note", "roi_seed_status", "roi_seed_reason", "used_for_auto_roi_proposal", "reference_roi_id", "inside_reference_roi", "reference_roi_overlap_fraction", "reference_roi_selection_note"]:
        if column not in fieldnames:
            fieldnames.append(column)
    image_name = rows[0].get("image_name", original_image.name) if rows else original_image.name
    frame_id = rows[0].get("frame_id", make_frame_id(image_stem)) if rows else make_frame_id(image_stem)
    rejected_path = output / "rejected_objects.csv"
    rejected_rows: list[dict[str, str]] = []
    if rejected_path.exists():
        with rejected_path.open("r", encoding="utf-8-sig", newline="") as f:
            rejected_rows = list(csv.DictReader(f))
    proposals = derive_auto_roi_proposals(rows, image_name, frame_id, rejected_rows)
    accepted_source_rows = [row for row in rows if str(row.get("accepted_status", "")).strip().lower() == "accepted_cell_candidate"]
    reference_regions = derive_reference_roi_regions(rejected_rows, accepted_source_rows, image_name, frame_id)

    accepted_rows: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("accepted_status", "")).strip().lower() != "accepted_cell_candidate":
            row["auto_roi_id"] = ""
            row["inside_auto_roi"] = "false"
            row["auto_roi_overlap_fraction"] = "0.0000"
            row["auto_roi_selection_note"] = "not_accepted_cell_candidate"
            row["roi_seed_status"] = "not_roi_seed"
            row["roi_seed_reason"] = "not_accepted_cell_candidate"
            row["used_for_auto_roi_proposal"] = "false"
            row["reference_roi_id"] = ""
            row["inside_reference_roi"] = "false"
            row["reference_roi_overlap_fraction"] = "0.0000"
            row["reference_roi_selection_note"] = "not_accepted_cell_candidate"
            continue
        best_roi: dict[str, Any] | None = None
        best_overlap = 0.0
        for proposal in proposals:
            overlap = bbox_intersection_fraction(row, proposal)
            if overlap > best_overlap:
                best_overlap = overlap
                best_roi = proposal
        row["auto_roi_id"] = "" if best_roi is None else str(best_roi["roi_id"])
        row["inside_auto_roi"] = "true" if best_overlap >= 0.25 else "false"
        row["auto_roi_overlap_fraction"] = f"{best_overlap:.4f}"
        row["auto_roi_selection_note"] = "inside_auto_roi_selection_pool" if best_overlap >= 0.25 else "outside_auto_roi_not_selected"
        row["roi_seed_status"] = "roi_seed_candidate" if best_overlap >= 0.25 else "not_roi_seed"
        row["roi_seed_reason"] = "roi_seed_clustered_cell_material" if best_overlap >= 0.25 else "accepted_candidate_outside_auto_roi"
        row["used_for_auto_roi_proposal"] = "true" if best_overlap >= 0.25 else "false"
        best_reference: dict[str, Any] | None = None
        best_reference_overlap = 0.0
        for region in reference_regions:
            overlap = bbox_intersection_fraction(row, {"roi_bbox_x": region["roi_inner_x"], "roi_bbox_y": region["roi_inner_y"], "roi_bbox_w": region["roi_inner_w"], "roi_bbox_h": region["roi_inner_h"]})
            if overlap > best_reference_overlap:
                best_reference_overlap = overlap
                best_reference = region
        row["reference_roi_id"] = "" if best_reference is None else str(best_reference["reference_roi_id"])
        row["inside_reference_roi"] = "true" if best_reference_overlap >= 0.10 else "false"
        row["reference_roi_overlap_fraction"] = f"{best_reference_overlap:.4f}"
        row["reference_roi_selection_note"] = "inside_reference_roi" if best_reference_overlap >= 0.10 else "outside_reference_roi"
        row["selected_for_frame_summary"] = "false"
        accepted_rows.append(row)

    if reference_regions:
        for row in accepted_rows:
            if row.get("inside_reference_roi") == "true":
                row["reference_roi_selection_note"] = "object_overlaps_reference_roi_region"
            else:
                row["reference_roi_selection_note"] = "not_selected_outside_reference_roi_region"
        rewrite_csv_rows(features_path, rows, fieldnames)
        write_reference_roi_regions_csv(output, image_stem, reference_regions)
        write_reference_roi_regions_overlay(output, image_stem, original_image, reference_regions)
        qc_status = rewrite_frame_summary_reference_regions(output, image_stem, reference_regions, accepted_rows, proposals[0] if proposals else None)
        write_roi_proposals_csv(output, image_stem, proposals)
        write_roi_proposals_overlay(output, image_stem, original_image, proposals)
        return proposals, qc_status

    primary_roi = proposals[0] if proposals else None
    if primary_roi is not None:
        primary_roi_id = str(primary_roi.get("roi_id", ""))
        selection_pool = [row for row in accepted_rows if row.get("auto_roi_id") == primary_roi_id and str(row.get("inside_auto_roi", "")).lower() == "true"]
        for row in accepted_rows:
            if row.get("auto_roi_id") != primary_roi_id and str(row.get("inside_auto_roi", "")).lower() == "true":
                row["auto_roi_selection_note"] = "inside_secondary_auto_roi_not_primary_selection_pool"
        if not selection_pool:
            selection_pool = [row for row in accepted_rows if str(row.get("inside_auto_roi", "")).lower() == "true"]
            for row in selection_pool:
                row["auto_roi_selection_note"] = "fallback_any_auto_roi_no_primary_candidates"
    else:
        selection_pool = accepted_rows
        for row in accepted_rows:
            row["auto_roi_selection_note"] = "fallback_full_frame_no_auto_roi"
    selection_pool.sort(key=lambda row: (as_float(row.get("selection_score"), 999999999), as_float(row.get("selection_rank_blue"), 999999), -as_float(row.get("object_pixels"))))
    frame_select_top_blue = int(as_float(params.get("frame_select_top_blue"), 20))
    max_near_full_blue = int(as_float(params.get("frame_select_max_near_full_blue"), 5))
    max_near_full_pixel_fraction = 0.25
    selected_rows: list[dict[str, str]] = []
    near_full_selected = 0
    near_full_selected_pixels = 0.0
    near_full_selected_blue = 0.0

    for row in selection_pool:
        if len(selected_rows) >= frame_select_top_blue:
            break
        if as_float(row.get("blue_pixel_percent")) >= 99:
            continue
        selected_rows.append(row)

    selected_pixels = sum(as_float(row.get("object_pixels")) for row in selected_rows)
    selected_blue = sum(as_float(row.get("blue_pixels")) for row in selected_rows)
    for row in selection_pool:
        if len(selected_rows) >= frame_select_top_blue:
            break
        if as_float(row.get("blue_pixel_percent")) < 99:
            continue
        if str(row.get("selection_review_note", "")) == "strongly_deprioritized_nearly_full_blue_artifact_risk":
            row["auto_roi_selection_note"] = "not_selected_strong_near_full_blue_artifact_risk"
            continue
        if near_full_selected >= max_near_full_blue:
            row["auto_roi_selection_note"] = "not_selected_near_full_blue_cap_reached"
            continue
        candidate_pixels = as_float(row.get("object_pixels"))
        candidate_blue = as_float(row.get("blue_pixels"))
        next_pixels = selected_pixels + candidate_pixels
        next_blue = selected_blue + candidate_blue
        next_near_full_pixels = near_full_selected_pixels + candidate_pixels
        next_near_full_blue = near_full_selected_blue + candidate_blue
        if next_pixels > 0 and next_near_full_pixels / next_pixels > max_near_full_pixel_fraction:
            row["auto_roi_selection_note"] = "not_selected_near_full_blue_pixel_mass_cap"
            continue
        if next_blue > 0 and next_near_full_blue / next_blue > max_near_full_pixel_fraction:
            row["auto_roi_selection_note"] = "not_selected_near_full_blue_blue_mass_cap"
            continue
        selected_rows.append(row)
        selected_pixels = next_pixels
        selected_blue = next_blue
        near_full_selected_pixels = next_near_full_pixels
        near_full_selected_blue = next_near_full_blue
        near_full_selected += 1

    fallback_strong_near_full = False
    if not selected_rows and selection_pool:
        fallback = selection_pool[0]
        selected_rows.append(fallback)
        fallback_strong_near_full = str(fallback.get("selection_review_note", "")) == "strongly_deprioritized_nearly_full_blue_artifact_risk"
        fallback["auto_roi_selection_note"] = "fallback_selected_no_alternative_candidates"

    for row in selected_rows:
        row["selected_for_frame_summary"] = "true"
        if proposals and row.get("auto_roi_selection_note") != "fallback_selected_no_alternative_candidates":
            row["auto_roi_selection_note"] = "selected_inside_auto_roi"
    rewrite_csv_rows(features_path, rows, fieldnames)
    qc_status = rewrite_frame_summary_selection(output, image_stem, selected_rows, proposals, primary_roi)
    if fallback_strong_near_full:
        qc_status = append_qc_status(qc_status, "WARN_FALLBACK_SELECTED_STRONG_NEAR_FULL_BLUE_ARTIFACT_RISK")
        summary_path = output / f"frame_features_{image_stem}.csv"
        with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            summary_fieldnames = list(reader.fieldnames or [])
            summary_rows = list(reader)
        if summary_rows:
            summary_rows[0]["qc_status"] = qc_status
            rewrite_csv_rows(summary_path, summary_rows, summary_fieldnames)
    write_roi_proposals_csv(output, image_stem, proposals)
    write_roi_proposals_overlay(output, image_stem, original_image, proposals)
    write_reference_roi_regions_csv(output, image_stem, [])
    write_reference_roi_regions_overlay(output, image_stem, original_image, [])
    return proposals, qc_status


def read_qc_status(output: Path, image_stem: str) -> str:
    summary_path = output / f"frame_features_{image_stem}.csv"
    if not summary_path.exists():
        summary_path = output / "final_frame_summary.csv"
    if not summary_path.exists():
        return ""
    with summary_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0].get("qc_status", "") if rows else ""


def write_selection_review_candidates(output: Path, image_stem: str) -> None:
    features_path = output / f"cell_features_{image_stem}.csv"
    review_path = output / f"selection_review_candidates_{image_stem}.csv"
    qc_status = read_qc_status(output, image_stem)
    fieldnames = [
        "image_name",
        "frame_id",
        "object_id",
        "feature_row_id",
        "selected_for_frame_summary",
        "accepted_status",
        "candidate_status",
        "object_pixels",
        "blue_pixels",
        "blue_pixel_percent",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "selection_rank_size",
        "selection_rank_blue",
        "selection_score",
        "selection_penalty_full_blue",
        "warn_full_blue_candidate",
        "selection_review_note",
        "auto_roi_id",
        "inside_auto_roi",
        "auto_roi_overlap_fraction",
        "auto_roi_selection_note",
        "roi_seed_status",
        "roi_seed_reason",
        "used_for_auto_roi_proposal",
        "reference_roi_id",
        "inside_reference_roi",
        "reference_roi_overlap_fraction",
        "reference_roi_selection_note",
        "qc_status",
        "review_label",
        "review_reason",
        "review_note",
    ]
    with features_path.open("r", encoding="utf-8-sig", newline="") as f:
        accepted_rows = [
            row for row in csv.DictReader(f)
            if str(row.get("accepted_status", "")).strip().lower() == "accepted_cell_candidate"
        ]
    with review_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in accepted_rows:
            review_row = {name: row.get(name, "") for name in fieldnames}
            review_row["qc_status"] = qc_status
            review_row["review_label"] = ""
            review_row["review_reason"] = ""
            review_row["review_note"] = ""
            writer.writerow(review_row)
    validate_public_metadata(review_path)


def append_auto_roi_qc_report(output: Path, proposals: list[dict[str, Any]], qc_status: str) -> None:
    report_path = output / "extended_qc_report.md"
    if not report_path.exists():
        return
    lines = [
        "\n## Auto ROI proposal review\n\n",
        f"- auto_roi_proposal_count: {len(proposals)}\n",
        f"- selected_objects_constrained_to_auto_roi: {'true' if proposals else 'false'}\n",
        f"- primary_auto_roi_id: {proposals[0].get('roi_id', '') if proposals else ''}\n",
        f"- primary_auto_roi_reason: {proposals[0].get('roi_review_note', '') if proposals else ''}\n",
        "- selected overlay/contact sheet are regenerated from the final postprocessed selected_for_frame_summary flags.\n",
    ]
    large_seed_count = sum(1 for proposal in proposals if proposal.get("roi_review_note") == "roi_seed_large_aggregate_candidate")
    lines.append(f"- large_aggregate_roi_seed_proposal_count: {large_seed_count}\n")
    lines.append("- near-full-blue selected objects are capped by frame_select_max_near_full_blue; remaining slots are left unfilled rather than backfilled with full-blue candidates.\n")
    if not proposals:
        lines.append("- WARN_NO_AUTO_ROI_PROPOSALS: no deterministic accepted-candidate clusters met ROI proposal criteria; frame-summary selection used full-frame fallback.\n")
    if "WARN_SELECTED_OUTSIDE_AUTO_ROI" in qc_status:
        lines.append("- WARN_SELECTED_OUTSIDE_AUTO_ROI: one or more selected objects were not inside an auto ROI proposal.\n")
    lines.append("- Black rectangular boxes, when present, are optional reference guides only; auto ROI proposals are derived from accepted Weka/postfilter cell-material candidates.\n")
    with report_path.open("a", encoding="utf-8") as f:
        f.writelines(lines)


def write_selected_objects_overlay(output: Path, image_stem: str, original_image: Path) -> None:
    from PIL import Image, ImageDraw

    with Image.open(original_image) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    if read_selection_unit_type(output, image_stem) == "reference_roi_regions":
        with (output / f"reference_roi_regions_{image_stem}.csv").open("r", encoding="utf-8-sig", newline="") as f:
            regions = list(csv.DictReader(f))
        for region in regions:
            x = int(as_float(region.get("roi_inner_x"))); y = int(as_float(region.get("roi_inner_y")))
            w = int(as_float(region.get("roi_inner_w"), 1)); h = int(as_float(region.get("roi_inner_h"), 1))
            draw.rectangle((x, y, x + w, y + h), outline=(0, 255, 255), width=6)
            draw.text((x + 4, max(0, y - 18)), str(region.get("reference_roi_id")), fill=(0, 255, 255))
    else:
        features_path = output / f"cell_features_{image_stem}.csv"
        with features_path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = [row for row in csv.DictReader(f) if str(row.get("selected_for_frame_summary", "")).lower() == "true"]
        for row in rows:
            x = int(as_float(row.get("bbox_x"))); y = int(as_float(row.get("bbox_y")))
            w = int(as_float(row.get("bbox_w") or row.get("bbox_width"), 1)); h = int(as_float(row.get("bbox_h") or row.get("bbox_height"), 1))
            draw.rectangle((x, y, x + w, y + h), outline=(0, 255, 255), width=6)
            draw.text((x + 4, max(0, y - 18)), f"#{row.get('object_id')}", fill=(0, 255, 255))
    image.save(output / f"selected_objects_overlay_{image_stem}.jpg", quality=90)


def write_review_candidates_contact_sheet(output: Path, image_stem: str, original_image: Path) -> None:
    from PIL import Image, ImageDraw

    features_path = output / f"cell_features_{image_stem}.csv"
    with features_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [
            row for row in csv.DictReader(f)
            if str(row.get("accepted_status", "")).strip().lower() == "accepted_cell_candidate"
        ]
    rows.sort(key=lambda row: (float(row.get("selection_rank_size") or 999999), float(row.get("selection_rank_blue") or 999999)))

    thumb_w = 220
    thumb_h = 160
    label_h = 88
    cols = 4
    rows_count = max(1, (len(rows) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows_count * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    with Image.open(original_image) as source:
        source = source.convert("RGB")
        for idx, row in enumerate(rows):
            bbox_x = int(float(row.get("bbox_x") or 0))
            bbox_y = int(float(row.get("bbox_y") or 0))
            bbox_w = int(float(row.get("bbox_w") or row.get("bbox_width") or 1))
            bbox_h = int(float(row.get("bbox_h") or row.get("bbox_height") or 1))
            pad = 12
            left = max(0, bbox_x - pad)
            top = max(0, bbox_y - pad)
            right = min(source.width, bbox_x + bbox_w + pad)
            bottom = min(source.height, bbox_y + bbox_h + pad)
            crop = source.crop((left, top, right, bottom))
            crop.thumbnail((thumb_w, thumb_h))
            col = idx % cols
            row_index = idx // cols
            x0 = col * thumb_w
            y0 = row_index * (thumb_h + label_h)
            sheet.paste(crop, (x0 + (thumb_w - crop.width) // 2, y0))
            selected = str(row.get("selected_for_frame_summary", "")).lower() == "true"
            label = (
                f"#{row.get('object_id')} selected={selected}\n"
                f"size_rank={row.get('selection_rank_size')} blue_rank={row.get('selection_rank_blue')}\n"
                f"px={row.get('object_pixels')} blue%={row.get('blue_pixel_percent')}\n"
                f"bbox=({bbox_x},{bbox_y},{bbox_w},{bbox_h})"
            )
            draw.multiline_text((x0 + 4, y0 + thumb_h + 4), label, fill=(0, 0, 0), spacing=2)

    if not rows:
        draw.text((10, 10), "No accepted candidate objects", fill=(0, 0, 0))
    sheet.save(output / f"review_candidates_contact_sheet_{image_stem}.jpg", quality=90)


def postprocess_outputs(output: Path, image_stem: str, original_image: Path, params: dict[str, str], short_path_used: str) -> None:
    write_blue_pixels_xlsx(output)
    copy_final_named_outputs(output, image_stem)
    normalize_public_outputs_metadata(output, image_stem, original_image, short_path_used)
    proposals, qc_status = apply_auto_roi_selection(output, image_stem, original_image, params)
    append_auto_roi_qc_report(output, proposals, qc_status)
    write_selection_review_candidates(output, image_stem)
    if bool_param(params.get("save_overlays", "true")):
        write_selected_objects_overlay(output, image_stem, original_image)
        write_selected_contact_sheet(output, image_stem, original_image)
        write_review_candidates_contact_sheet(output, image_stem, original_image)
    move_internal_outputs(output)
    validate_frame_summary(output)
    validate_cell_feature_table(output)
    validate_output_image_dimensions(output, image_stem, original_image, params)
    # final_frame_summary.csv is written by the macro; Python copies it to the final frame_features_<original_stem>.csv name.

def append_runner_parameters_to_macro_log(output: Path, params: dict[str, str]) -> None:
    with (output / "macro_log.txt").open("a", encoding="utf-8", errors="replace") as f:
        f.write("\nRunner parameter summary:\n")
        for key, value in params.items():
            f.write(f"{key}={value}\n")


def read_last_checkpoint(output: Path) -> str:
    log_path = output / "macro_log.txt"
    if not log_path.exists():
        log_path = output / "_internal" / "macro_log.txt"
    if not log_path.exists():
        return ""
    last = ""
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("CHECKPOINT "):
            last = line
    return last


def write_simple_tiff(path: Path, width: int, height: int, fill: int = 0) -> None:
    pixels = bytes([fill]) * width * height
    entries = [
        (256, 4, 1, width),
        (257, 4, 1, height),
        (258, 3, 1, 8),
        (259, 3, 1, 1),
        (273, 4, 1, 0),
        (277, 3, 1, 1),
        (279, 4, 1, len(pixels)),
    ]
    ifd_len = 2 + 12 * len(entries) + 4
    pixel_offset = 8 + ifd_len
    data = b"II" + (42).to_bytes(2, "little") + (8).to_bytes(4, "little") + len(entries).to_bytes(2, "little")
    for tag, typ, count, value in entries:
        if tag == 273:
            value = pixel_offset
        data += tag.to_bytes(2, "little") + typ.to_bytes(2, "little") + count.to_bytes(4, "little") + value.to_bytes(4, "little")
    path.write_bytes(data + (0).to_bytes(4, "little") + pixels)


def write_simple_png(path: Path, width: int, height: int, rgb: tuple[int, int, int] = (0, 0, 0)) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return len(payload).to_bytes(4, "big") + kind + payload + zlib.crc32(kind + payload).to_bytes(4, "big")
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    ihdr = width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def copy_hs_err_logs(output: Path) -> None:
    internal = output / "_internal"
    internal.mkdir(exist_ok=True)
    roots = {Path.cwd(), output, output.parent}
    for root in roots:
        for log_path in root.glob("hs_err_pid*.log"):
            target = internal / log_path.name
            if log_path.resolve() != target.resolve():
                shutil.copy2(log_path, target)


def write_weka_failure_outputs(output: Path, image: Path, image_stem: str, weka_model: Path, params: dict[str, str], failure_status: str, detail: str) -> None:
    width, height = read_image_dimensions(image)
    write_simple_tiff(output / f"cellmask_{image_stem}.tif", width, height, 0)
    write_simple_tiff(output / f"blue_inside_cells_{image_stem}.tif", width, height, 0)
    write_simple_png(output / f"vis_cellpixels_{image_stem}.png", width, height, (80, 0, 80))
    write_simple_png(output / f"roi_overlay_{image_stem}.jpg", width, height, (80, 0, 0))
    write_simple_png(output / f"selected_objects_overlay_{image_stem}.jpg", width, height, (0, 80, 80))
    write_simple_png(output / f"selected_objects_contact_sheet_{image_stem}.jpg", max(1, min(width, 880)), max(1, min(height, 232)), (240, 240, 240))
    write_simple_png(output / f"review_candidates_contact_sheet_{image_stem}.jpg", max(1, min(width, 880)), max(1, min(height, 232)), (240, 240, 240))
    write_simple_png(output / f"roi_proposals_overlay_{image_stem}.jpg", width, height, (80, 0, 80))
    write_simple_png(output / f"reference_roi_regions_overlay_{image_stem}.jpg", width, height, (80, 80, 0))

    cell_header = "image_name,group_name,original_long_path,short_path_used,frame_id,object_id,feature_row_id,candidate_status,accepted_status,selected_for_frame_summary,reject_reason,selection_rank_size,selection_rank_blue,selection_score,selection_penalty_full_blue,warn_full_blue_candidate,selection_review_note,auto_roi_id,inside_auto_roi,auto_roi_overlap_fraction,auto_roi_selection_note,roi_seed_status,roi_seed_reason,used_for_auto_roi_proposal,reference_roi_id,inside_reference_roi,reference_roi_overlap_fraction,reference_roi_selection_note,object_type,roi_area_pixels,object_pixels,blue_pixels,blue_pixel_fraction,blue_pixel_percent,bbox_x,bbox_y,bbox_w,bbox_h,bbox_width,bbox_height,centroid_x,centroid_y,aspect_ratio,R_mean,G_mean,B_mean,R_std,G_std,B_std,R_min,G_min,B_min,R_max,G_max,B_max,R_div_G,B_div_R,B_div_RGB_sum,intensity_mean,intensity_std,intensity_min,intensity_max,cell_material_area_px,roi_area_reconstructed,roi_area_delta_percent,roi_reconstruction_status\n"
    (output / "cell_features.csv").write_text(cell_header, encoding="utf-8-sig")
    shutil.copy2(output / "cell_features.csv", output / f"cell_features_{image_stem}.csv")
    (output / "blue_pixels_features.csv").write_text("image_name,group_name,object_type,object_id,object_pixels,blue_pixels,blue_pixel_fraction,blue_pixel_percent\n", encoding="utf-8-sig")
    write_blue_pixels_xlsx(output)
    shutil.copy2(output / "blue_table.xlsx", output / f"blue_table_{image_stem}.xlsx")

    summary_header = "image_name,image_width,image_height,accepted_object_count,accepted_object_pixels,accepted_blue_pixels,blue_pixel_percent_all_accepted,qc_status\n"
    summary_row = f"{image.name},{width},{height},0,0,0,0,{failure_status}\n"
    (output / "final_frame_summary.csv").write_text(summary_header + summary_row, encoding="utf-8-sig")
    shutil.copy2(output / "final_frame_summary.csv", output / f"frame_features_{image_stem}.csv")
    write_roi_proposals_csv(output, image_stem, [])
    write_reference_roi_regions_csv(output, image_stem, [])
    write_selection_review_candidates(output, image_stem)
    report = (
        "# IronCellQuant single-frame QC report\n\n"
        f"## Status\n\n{failure_status}\n\n"
        "## Weka model\n\n"
        f"Model path: {weka_model}\n\n"
        f"{detail}\n\n"
        "No accepted biological ROI/cell-material regions were produced. Final placeholder outputs were written for QC review only.\n"
    )
    (output / "extended_qc_report.md").write_text(report, encoding="utf-8")
    move_internal_outputs(output)
    copy_hs_err_logs(output)


def run_fiji(project: Path, image: Path, output: Path, fiji: Path, macro: Path, params: dict[str, str], timeout_seconds: int, weka_model: Path | None = None) -> dict[str, str]:
    started = datetime.now()
    fiji_input = prepare_fiji_input(image, output)
    if weka_model is not None and not weka_model.exists():
        macro_arg, _ = build_macro_arg(fiji_input, image, output, project, {**params, "weka_model": str(weka_model)})
        write_run_parameters(output, project, image, fiji_input, fiji, macro, macro_arg, {**params, "weka_model": str(weka_model)})
        write_weka_failure_outputs(output, image, image.stem, weka_model, params, "FAIL_WEKA_MODEL_MISSING", f"Expected Weka model path does not exist: {weka_model}")
        finished = datetime.now()
        return {
            "project": str(project),
            "image": str(image),
            "fiji_input": str(fiji_input),
            "weka_model": str(weka_model),
            "weka_tile_size": params.get("weka_tile_size", ""),
            "weka_tile_overlap": params.get("weka_tile_overlap", ""),
            "output": str(output),
            "returncode": "0",
            "started": started.isoformat(timespec="seconds"),
            "finished": finished.isoformat(timespec="seconds"),
            "missing_outputs": "",
            "stale_outputs": "",
            "empty_outputs": "",
            "last_checkpoint": "",
            "timed_out": "False",
            "postprocess_error": "FAIL_WEKA_MODEL_MISSING",
            "ok": "False",
        }
    macro_params = params if weka_model is None else {**params, "weka_model": str(weka_model)}
    macro_arg, short_path_used = build_macro_arg(fiji_input, image, output, project, macro_params)
    cmd = [str(fiji), "--headless", "-macro", str(macro), macro_arg]
    write_run_parameters(output, project, image, fiji_input, fiji, macro, macro_arg, macro_params)

    timed_out = False
    returncode = 1
    stdout = ""
    stderr = ""
    postprocess_error = ""
    process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, text=True)
        else:
            process.kill()
        stdout, stderr = process.communicate()
        returncode = 124

    finished = datetime.now()
    (output / "fiji_stdout.txt").write_text(stdout or "", encoding="utf-8", errors="replace")
    (output / "fiji_stderr.txt").write_text(stderr or "", encoding="utf-8", errors="replace")
    (output / "runner_command.txt").write_text(" ".join(cmd), encoding="utf-8", errors="replace")
    append_runner_parameters_to_macro_log(output, macro_params)

    if weka_model is not None and returncode != 0:
        status = "FAIL_WEKA_INFERENCE_MEMORY" if ("memory" in (stdout + stderr).lower() or "paging file" in (stdout + stderr).lower()) else "FAIL_WEKA_INFERENCE_CRASH"
        detail = f"Weka inference failed before required outputs were complete. returncode={returncode}; last_checkpoint={read_last_checkpoint(output)}; tile_size={macro_params.get('weka_tile_size')}; tile_overlap={macro_params.get('weka_tile_overlap')}"
        write_weka_failure_outputs(output, image, image.stem, weka_model, macro_params, status, detail)
        postprocess_error = status
        missing, stale, empty = validate_named_outputs(output, final_expected_outputs(params, image.stem), started)
    else:
        missing, stale, empty = validate_expected_outputs(output, params, started)

    if weka_model is not None and returncode == 0 and (missing or empty):
        status = "FAIL_WEKA_MASK_MISSING" if "WekaCellMaskRaw" in (stdout + stderr) or "FAIL_WEKA_MASK_MISSING" in (stdout + stderr) else "FAIL_WEKA_TILE_INFERENCE"
        detail = f"Weka tile inference ended without the required macro outputs. last_checkpoint={read_last_checkpoint(output)}; missing_outputs={';'.join(missing)}; empty_outputs={';'.join(empty)}; tile_size={macro_params.get('weka_tile_size')}; tile_overlap={macro_params.get('weka_tile_overlap')}"
        write_weka_failure_outputs(output, image, image.stem, weka_model, macro_params, status, detail)
        postprocess_error = status
        missing, stale, empty = validate_named_outputs(output, final_expected_outputs(params, image.stem), started)

    if returncode == 0 and not missing and not stale and not empty and not postprocess_error:
        try:
            postprocess_outputs(output, image.stem, image, params, short_path_used)
        except Exception as exc:
            postprocess_error = repr(exc)
        missing, stale, empty = validate_named_outputs(output, final_expected_outputs(params, image.stem), started)

    last_checkpoint = read_last_checkpoint(output)
    ok = returncode == 0 and not missing and not stale and not empty and not postprocess_error
    return {
        "project": str(project),
        "image": str(image),
        "fiji_input": str(fiji_input),
        "weka_model": "" if weka_model is None else str(weka_model),
        "weka_tile_size": macro_params.get("weka_tile_size", ""),
        "weka_tile_overlap": macro_params.get("weka_tile_overlap", ""),
        "output": str(output),
        "returncode": str(returncode),
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "missing_outputs": ";".join(missing),
        "stale_outputs": ";".join(stale),
        "empty_outputs": ";".join(empty),
        "last_checkpoint": last_checkpoint,
        "timed_out": str(timed_out),
        "postprocess_error": postprocess_error,
        "ok": str(ok),
    }


def add_param_args(parser: argparse.ArgumentParser) -> None:
    for key, value in DEFAULT_PARAMS.items():
        parser.add_argument("--" + key.replace("_", "-"), default=value)


def params_from_args(args: argparse.Namespace) -> dict[str, str]:
    return {key: str(getattr(args, key)) for key in DEFAULT_PARAMS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Fiji/ImageJ IronCells analysis.")
    parser.add_argument("--project", type=Path, help="Project folder. Defaults to the folder containing this script.")
    parser.add_argument("--input", type=Path, help="Image path. Defaults to project/input/52*.bmp.")
    parser.add_argument("--output-root", type=Path, help="Output root. Defaults to project/output.")
    parser.add_argument("--output", type=Path, help="Exact output folder. Use with --clean-output to reuse it intentionally.")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--prefix", default="single_test")
    parser.add_argument("--macro", type=Path, help="Macro path. Defaults to project/macros/Main_IronCells_headless.ijm.")
    parser.add_argument("--fiji", type=Path, help="Path to fiji.bat. Can also be set by FIJI_PATH.")
    parser.add_argument("--weka-model", type=Path, help="Optional Fiji Trainable Weka Segmentation .model path for cell-material detection.")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    add_param_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = resolve_project(args.project)
    image = (args.input or discover_default_input(project)).resolve()
    output_root = (args.output_root or (project / "output")).resolve()
    macro = (args.macro or (project / "macros" / "Main_IronCells_headless.ijm")).resolve()
    fiji = resolve_fiji(project, args.fiji)

    if not project.exists():
        raise SystemExit(f"Project folder not found: {project}")
    if not image.exists():
        raise SystemExit(f"Input not found: {image}")
    if not macro.exists():
        raise SystemExit(f"Macro not found: {macro}")
    if not fiji.exists():
        raise SystemExit(f"Fiji runner not found: {fiji}")

    output = create_clean_output((args.output or output_root).resolve(), args.prefix, args.clean_output and args.output is not None)
    row = run_fiji(project, image, output, fiji, macro, params_from_args(args), args.timeout_seconds, args.weka_model.resolve() if args.weka_model else None)

    with (output / "runner_report.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    print(f"ok={row['ok']} returncode={row['returncode']} output={output}")
    if row["missing_outputs"]:
        print(f"missing={row['missing_outputs']}")
    if row["stale_outputs"]:
        print(f"stale={row['stale_outputs']}")
    if row["empty_outputs"]:
        print(f"empty={row['empty_outputs']}")
    if row["last_checkpoint"]:
        print(f"last_checkpoint={row['last_checkpoint']}")
    return 0 if row["ok"] == "True" else 1


if __name__ == "__main__":
    raise SystemExit(main())
