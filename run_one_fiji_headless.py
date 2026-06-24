from __future__ import annotations

import argparse
import csv
import os
import shutil
import importlib.util
import subprocess
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
    "blue_inside_cells.tif",
]

FINAL_OUTPUT_MAP = {
    "cellmask.tif": "cellmask_{image_stem}.tif",
    "vis_cellpixels.png": "vis_cellpixels_{image_stem}.png",
    "roi_overlay.jpg": "roi_overlay_{image_stem}.jpg",
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
    "border_margin_px": "2",
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
        ])
    if bool_param(params.get("save_overlays", "true")):
        outputs.extend(OVERLAY_EXPECTED_OUTPUTS)
        if image_stem:
            outputs.extend([
                f"cellmask_{image_stem}.tif",
                f"vis_cellpixels_{image_stem}.png",
                f"roi_overlay_{image_stem}.jpg",
                f"blue_inside_cells_{image_stem}.tif",
            ])
    return outputs


def final_expected_outputs(params: dict[str, str], image_stem: str) -> list[str]:
    outputs = [
        f"blue_table_{image_stem}.xlsx",
        f"cell_features_{image_stem}.csv",
        f"frame_features_{image_stem}.csv",
    ]
    if bool_param(params.get("save_overlays", "true")):
        outputs.extend([
            f"cellmask_{image_stem}.tif",
            f"vis_cellpixels_{image_stem}.png",
            f"roi_overlay_{image_stem}.jpg",
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
        raise ValueError(f"Compressed TIFF mask is not supported for QC validation: {path}")
    if samples != 1:
        raise ValueError(f"Mask appears to have {samples} samples per pixel, expected one channel: {path}")
    strip_offset = tags.get(273, (0, 0, 0))[2]
    strip_count = tags.get(279, (0, 0, 0))[2]
    if not strip_offset or not strip_count:
        raise ValueError(f"TIFF mask is missing strip offsets/counts: {path}")
    pixels = data[strip_offset:strip_offset + strip_count]
    if bits == 8:
        nonzero = sum(1 for value in pixels if value != 0)
    elif bits == 16:
        nonzero = 0
        for i in range(0, len(pixels), 2):
            if int.from_bytes(pixels[i:i+2], endian) != 0:
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
    summary = read_single_csv_row(output / "final_frame_summary.csv")
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
    for name in set(ALWAYS_EXPECTED_OUTPUTS + OVERLAY_EXPECTED_OUTPUTS + ["blue_table.xlsx", "fiji_stdout.txt", "fiji_stderr.txt", "runner_command.txt", "fiji_work"]):
        source = output / name
        if source.exists():
            target = internal / name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(source), str(target))


def postprocess_outputs(output: Path, image_stem: str, original_image: Path, params: dict[str, str]) -> None:
    validate_frame_summary(output)
    validate_cell_feature_table(output)
    write_blue_pixels_xlsx(output)
    copy_final_named_outputs(output, image_stem)
    validate_output_image_dimensions(output, image_stem, original_image, params)
    move_internal_outputs(output)
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


def run_fiji(project: Path, image: Path, output: Path, fiji: Path, macro: Path, params: dict[str, str], timeout_seconds: int) -> dict[str, str]:
    started = datetime.now()
    fiji_input = prepare_fiji_input(image, output)
    macro_arg, _ = build_macro_arg(fiji_input, image, output, project, params)
    cmd = [str(fiji), "--headless", "-macro", str(macro), macro_arg]
    write_run_parameters(output, project, image, fiji_input, fiji, macro, macro_arg, params)

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
    append_runner_parameters_to_macro_log(output, params)

    missing, stale, empty = validate_expected_outputs(output, params, started)
    if returncode == 0 and not missing and not stale and not empty:
        try:
            postprocess_outputs(output, image.stem, image, params)
        except Exception as exc:
            postprocess_error = repr(exc)
        missing, stale, empty = validate_named_outputs(output, final_expected_outputs(params, image.stem), started)

    last_checkpoint = read_last_checkpoint(output)
    ok = returncode == 0 and not missing and not stale and not empty and not postprocess_error
    return {
        "project": str(project),
        "image": str(image),
        "fiji_input": str(fiji_input),
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
    row = run_fiji(project, image, output, fiji, macro, params_from_args(args), args.timeout_seconds)

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
