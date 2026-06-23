from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import zipfile
from html import escape
from typing import Any
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

ALWAYS_EXPECTED_OUTPUTS = [
    "all_components_before_filter.csv",
    "rejected_objects.csv",
    "final_object_report.csv",
    "blue_pixels_features.csv",
    "blue_pixels_features.xlsx",
    "final_frame_summary.csv",
    "frame_features_for_pca.csv",
    "extended_qc_report.md",
    "run_parameters.txt",
    "run_parameters.csv",
    "macro_log.txt",
]

OVERLAY_EXPECTED_OUTPUTS = [
    "final_analysis_overlay.tif",
    "final_analysis_overlay_preview.jpg",
    "review_detection_overlay.tif",
    "review_detection_overlay_preview.jpg",
]

DEFAULT_PARAMS = {
    "threshold_method": "Li",
    "threshold_mode": "dark",
    "background_rolling": "80",
    "median_radius": "2",
    "contrast_saturated": "0.35",
    "morph_open_iterations": "0",
    "morph_close_iterations": "1",
    "fill_holes": "true",
    "metadata_bar_height": "120",
    "particle_extract_min_area": "10",
    "particle_extract_max_area": "2000000",
    "min_noise_area": "10",
    "min_single_cell_area": "40",
    "max_single_cell_area": "50000",
    "min_aggregate_area": "50000",
    "max_aggregate_area": "2000000",
    "max_single_cell_aspect": "10",
    "max_aggregate_aspect": "30",
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
        "Fiji runner not found. Pass --fiji \"R:\\Fiji\\fiji.bat\" "
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


def expected_outputs(params: dict[str, str]) -> list[str]:
    outputs = list(ALWAYS_EXPECTED_OUTPUTS)
    if bool_param(params.get("save_overlays", "true")):
        outputs.extend(OVERLAY_EXPECTED_OUTPUTS)
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


def build_macro_arg(image: Path, output: Path, project: Path, params: dict[str, str]) -> tuple[str, str]:
    short_input = windows_short_path(image)
    short_used = "true" if short_input != str(image) else "false"
    macro_params = {
        "input": short_input,
        "output": str(output),
        "original_long_path": str(image),
        "original_file_name": image.name,
        "group_name": group_name_for(image, project),
        "short_path_used": short_used,
        **params,
    }
    return ";".join(f"{k}={v}" for k, v in macro_params.items()), short_used


def write_run_parameters(output: Path, project: Path, image: Path, fiji: Path, macro: Path, macro_arg: str, params: dict[str, str]) -> None:
    outputs = expected_outputs(params)
    lines = [
        f"project={project}",
        f"input={image}",
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
            ("input", image),
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


def validate_expected_outputs(output: Path, params: dict[str, str], started: datetime) -> tuple[list[str], list[str], list[str]]:
    missing: list[str] = []
    stale: list[str] = []
    empty: list[str] = []
    for name in expected_outputs(params):
        path = output / name
        if not path.exists():
            missing.append(name)
        elif datetime.fromtimestamp(path.stat().st_mtime) < started:
            stale.append(name)
        elif path.stat().st_size == 0:
            empty.append(name)
    return missing, stale, empty


def write_blue_pixels_xlsx(output: Path) -> None:
    csv_path = output / "blue_pixels_features.csv"
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    object_rows = [row for row in rows if row.get("object_id") != "frame"]
    object_rows.sort(key=lambda row: float(row.get("blue_pixel_percent") or 0), reverse=True)
    columns = ["image_name", "group_name", "object_type", "object_id", "object_pixels", "blue_pixels", "blue_pixel_fraction", "blue_pixel_percent"]
    table: list[list[Any]] = [columns]
    table.extend([[coerce_cell(row.get(column, "")) for column in columns] for row in object_rows])

    total_object_pixels = sum(float(row.get("object_pixels") or 0) for row in object_rows)
    total_blue_pixels = sum(float(row.get("blue_pixels") or 0) for row in object_rows)
    total_fraction = total_blue_pixels / total_object_pixels if total_object_pixels else 0
    table.append(["SUM", "", "all_accepted_cell_material", "", int(total_object_pixels), int(total_blue_pixels), total_fraction, 100 * total_fraction])
    write_minimal_xlsx(output / "blue_pixels_features.xlsx", table, "blue_pixels_features")


def write_minimal_xlsx(path: Path, table: list[list[Any]], sheet_name: str) -> None:
    rows_xml = []
    for r_idx, row in enumerate(table, start=1):
        cells = []
        for c_idx, value in enumerate(row, start=1):
            ref = f"{excel_column(c_idx)}{r_idx}"
            style = ' s="1"' if r_idx == 1 or r_idx == len(table) else ""
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"{style}><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"{style}><is><t>{escape(str(value))}</t></is></c>')
        rows_xml.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    dimension = f"A1:{excel_column(len(table[0]))}{len(table)}" if table else "A1:A1"
    cols = "".join(f'<col min="{i}" max="{i}" width="18" customWidth="1"/>' for i in range(1, len(table[0]) + 1))
    worksheet = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><dimension ref="{dimension}"/><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><sheetFormatPr defaultRowHeight="15"/><cols>{cols}</cols><sheetData>{"".join(rows_xml)}</sheetData><autoFilter ref="{dimension}"/></worksheet>'
    workbook = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="{escape(sheet_name)}" sheetId="1" r:id="rId1"/></sheets></workbook>'
    styles = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font/><font><b/></font></fonts><fills count="1"><fill><patternFill patternType="none"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs></styleSheet>'
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>')
        zf.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        zf.writestr("xl/_rels/workbook.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet)
        zf.writestr("xl/styles.xml", styles)


def excel_column(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


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


def write_frame_features_for_pca(output: Path) -> None:
    summary = read_single_csv_row(output / "final_frame_summary.csv")
    fieldnames = [
        "image_name", "width", "height", "frame_area_pixels", "component_count_total",
        "accepted_object_count", "accepted_object_pixels", "accepted_area_percent_of_frame",
        "accepted_blue_pixels", "blue_pixel_percent_all_accepted", "accepted_R_mean",
        "accepted_G_mean", "accepted_B_mean", "accepted_R_std", "accepted_G_std",
        "accepted_B_std", "accepted_B_over_R_mean", "accepted_B_over_RGB_sum_mean",
        "accepted_gray_stddev", "qc_status",
    ]
    row = {
        "image_name": summary.get("image_name", ""),
        "width": summary.get("image_width", ""),
        "height": summary.get("image_height", ""),
        "frame_area_pixels": summary.get("frame_area_pixels", ""),
        "component_count_total": summary.get("component_count_total", ""),
        "accepted_object_count": summary.get("accepted_object_count", ""),
        "accepted_object_pixels": summary.get("accepted_object_pixels", ""),
        "accepted_area_percent_of_frame": summary.get("accepted_area_percent_of_frame", ""),
        "accepted_blue_pixels": summary.get("accepted_blue_pixels", ""),
        "blue_pixel_percent_all_accepted": summary.get("blue_pixel_percent_all_accepted", ""),
        "accepted_R_mean": summary.get("accepted_R_mean", ""),
        "accepted_G_mean": summary.get("accepted_G_mean", ""),
        "accepted_B_mean": summary.get("accepted_B_mean", ""),
        "accepted_R_std": summary.get("accepted_R_std", ""),
        "accepted_G_std": summary.get("accepted_G_std", ""),
        "accepted_B_std": summary.get("accepted_B_std", ""),
        "accepted_B_over_R_mean": summary.get("accepted_B_over_R_mean", ""),
        "accepted_B_over_RGB_sum_mean": summary.get("accepted_B_over_RGB_sum_mean", ""),
        "accepted_gray_stddev": summary.get("accepted_gray_stddev", ""),
        "qc_status": summary.get("qc_status", ""),
    }
    with (output / "frame_features_for_pca.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def postprocess_outputs(output: Path) -> None:
    write_blue_pixels_xlsx(output)
    write_frame_features_for_pca(output)

def append_runner_parameters_to_macro_log(output: Path, params: dict[str, str]) -> None:
    with (output / "macro_log.txt").open("a", encoding="utf-8", errors="replace") as f:
        f.write("\nRunner parameter summary:\n")
        for key, value in params.items():
            f.write(f"{key}={value}\n")


def read_last_checkpoint(output: Path) -> str:
    log_path = output / "macro_log.txt"
    if not log_path.exists():
        return ""
    last = ""
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("CHECKPOINT "):
            last = line
    return last


def run_fiji(project: Path, image: Path, output: Path, fiji: Path, macro: Path, params: dict[str, str], timeout_seconds: int) -> dict[str, str]:
    started = datetime.now()
    macro_arg, _ = build_macro_arg(image, output, project, params)
    cmd = [str(fiji), "--headless", "-macro", str(macro), macro_arg]
    write_run_parameters(output, project, image, fiji, macro, macro_arg, params)

    timed_out = False
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

    if returncode == 0:
        postprocess_outputs(output)

    missing, stale, empty = validate_expected_outputs(output, params, started)

    last_checkpoint = read_last_checkpoint(output)
    ok = returncode == 0 and not missing and not stale and not empty
    return {
        "project": str(project),
        "image": str(image),
        "output": str(output),
        "returncode": str(returncode),
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "missing_outputs": ";".join(missing),
        "stale_outputs": ";".join(stale),
        "empty_outputs": ";".join(empty),
        "last_checkpoint": last_checkpoint,
        "timed_out": str(timed_out),
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
