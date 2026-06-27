"""Reusable batch orchestration for the IronCellQuant pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import build_censored_feature_matrices
import build_pca_visualizations
import censor_cell_candidates
import run_one_fiji_headless
import run_pca


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXTENSIONS = [".bmp", ".tif", ".tiff", ".png", ".jpg", ".jpeg"]
STATUS_PASS = "PASS"
STATUS_SKIPPED = "SKIPPED"
STATUS_FAIL_REFERENCE_ROI = "FAIL_REFERENCE_ROI_NOT_FOUND"
STATUS_FAIL_RAW = "FAIL_RAW_STAGE"
STATUS_FAIL_CENSORING = "FAIL_CENSORING_STAGE"
STATUS_FAIL_MATRIX = "FAIL_MATRIX_STAGE"
STATUS_FAIL_PCA = "FAIL_PCA_STAGE"
STATUS_FAIL_VISUALIZATION = "FAIL_VISUALIZATION_STAGE"
STATUS_SKIPPED_INSUFFICIENT = "SKIPPED_INSUFFICIENT_DATA"

RAW_PATTERNS = [
    "reference_roi_manifest_raw_{stem}.csv",
    "reference_roi_manifest_overlay_{stem}.jpg",
    "cell_features_raw_{stem}.csv",
    "frame_features_raw_{stem}.csv",
    "cell_objects_overlay_raw_{stem}.jpg",
    "cell_objects_legend_raw_{stem}.csv",
    "blue_table_raw_{stem}.xlsx",
]

CENSORED_PATTERNS = [
    "cell_features_censored_{stem}.csv",
    "cell_censoring_audit_{stem}.csv",
    "cell_objects_overlay_censored_{stem}.jpg",
    "frame_features_censored_{stem}.csv",
    "blue_table_censored_{stem}.xlsx",
]

BATCH_REPORT_COLUMNS = [
    "image_name",
    "image_path",
    "frame_id",
    "frame_output_dir",
    "raw_output_dir",
    "censored_output_dir",
    "raw_status",
    "roi_count",
    "raw_accepted_count",
    "censored_selected_count",
    "outside_cell",
    "outside_blue",
    "cell_matrix_included",
    "frame_matrix_included",
    "pca_cells_status",
    "pca_frames_status",
    "visualization_status",
    "reference_roi_manifest",
    "cell_features_raw",
    "cell_features_censored",
    "frame_features_censored",
    "processing_started_at",
    "processing_finished_at",
    "duration_seconds",
    "warnings",
    "error_message",
]


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class StageResult:
    status: str
    detail: str = ""


@dataclass
class FrameRun:
    index: int
    image_path: Path
    frame_id: str
    output_dir: Path
    input_dir: Path
    raw_dir: Path
    censored_dir: Path
    logs_dir: Path
    staged_input: Path
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    raw_status: str = STATUS_SKIPPED
    censoring_status: str = STATUS_SKIPPED
    pca_cells_status: str = STATUS_SKIPPED
    pca_frames_status: str = STATUS_SKIPPED
    visualization_status: str = STATUS_SKIPPED
    raw_row: dict[str, str] = field(default_factory=dict)
    censoring_summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error_message: str = ""

    @property
    def image_stem(self) -> str:
        return self.image_path.stem


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IronCellQuant processing for a directory of image frames.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fiji", required=True, type=Path)
    parser.add_argument("--weka-model", required=True, type=Path)
    parser.add_argument("--censoring-config", required=True, type=Path)
    parser.add_argument("--pca-config-cells", type=Path)
    parser.add_argument("--pca-config-frames", type=Path)
    parser.add_argument("--visualization-config", type=Path)
    parser.add_argument("--run-pca", action="store_true")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--image-ext", default=",".join(DEFAULT_EXTENSIONS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-fatal", action="store_true")
    parser.add_argument("--preferred-censoring-config", default="")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--task-report", type=Path, help="Optional mirror copy of batch_pipeline_report.md.")
    return parser.parse_args(argv)


def configure_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def normalized_extensions(value: str) -> set[str]:
    extensions = set()
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        extensions.add(item if item.startswith(".") else "." + item)
    return extensions or set(DEFAULT_EXTENSIONS)


def discover_images(input_dir: Path, extensions: set[str], recursive: bool, limit: int | None) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    images = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in extensions)
    if limit is not None:
        images = images[: max(0, limit)]
    return images


def safe_frame_id(index: int, image: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image.stem).strip("._")
    if not stem:
        stem = f"frame_{index:04d}"
    return f"frame_{index:04d}_{stem[:80]}"


def sha256(path: Path | None) -> str:
    if path is None or not path.exists() or path.is_dir():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(command: list[str]) -> str:
    completed = subprocess.run(command, cwd=SCRIPT_DIR, text=True, capture_output=True, check=False)
    return ((completed.stdout or "") + (completed.stderr or "")).strip()


def git_info() -> dict[str, str]:
    return {
        "branch": run_git(["git", "branch", "--show-current"]),
        "commit": run_git(["git", "rev-parse", "HEAD"]),
        "status_short": run_git(["git", "status", "--short"]),
        "diff_stat": run_git(["git", "diff", "--stat"]),
    }


def fiji_version(path: Path) -> str:
    try:
        completed = subprocess.run([str(path), "--version"], text=True, capture_output=True, timeout=30, check=False)
        text = ((completed.stdout or "") + (completed.stderr or "")).strip()
        return text.splitlines()[0] if text else "unknown"
    except Exception:
        return "unknown"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def append_message(current: str, message: str) -> str:
    return message if not current else current + "; " + message


def call_noarg_main_with_argv(main_func: Any, script_name: str, argv: list[str]) -> int:
    old_argv = sys.argv[:]
    try:
        sys.argv = [script_name, *argv]
        result = main_func()
        return int(result or 0)
    finally:
        sys.argv = old_argv


def frame_path(frame: FrameRun, area: str, pattern: str) -> Path:
    root = frame.raw_dir if area == "raw" else frame.censored_dir
    return root / pattern.format(stem=frame.image_stem)


def existing_paths(frame: FrameRun, area: str, patterns: list[str]) -> list[Path]:
    return [frame_path(frame, area, pattern) for pattern in patterns if frame_path(frame, area, pattern).exists()]


def missing_paths(frame: FrameRun, area: str, patterns: list[str]) -> list[Path]:
    return [frame_path(frame, area, pattern) for pattern in patterns if not frame_path(frame, area, pattern).exists()]


def create_frame(image: Path, index: int, output_root: Path) -> FrameRun:
    frame_id = safe_frame_id(index, image)
    frame_dir = output_root / "frames" / frame_id
    input_dir = frame_dir / "input"
    raw_dir = frame_dir / "raw"
    censored_dir = frame_dir / "censored"
    logs_dir = frame_dir / "logs"
    for directory in [input_dir, raw_dir, censored_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    staged_input = input_dir / f"frame_{index:04d}{image.suffix.lower()}"
    if not staged_input.exists() or staged_input.resolve() != image.resolve():
        shutil.copy2(image, staged_input)
    return FrameRun(index, image, frame_id, frame_dir, input_dir, raw_dir, censored_dir, logs_dir, staged_input)


def write_frame_metadata(frame: FrameRun) -> None:
    data = {
        "frame_index": frame.index,
        "frame_id": frame.frame_id,
        "original_path": str(frame.image_path),
        "original_file_name": frame.image_path.name,
        "original_stem": frame.image_stem,
        "staged_path": str(frame.staged_input),
        "raw_dir": str(frame.raw_dir),
        "censored_dir": str(frame.censored_dir),
    }
    (frame.logs_dir / "frame_metadata.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def count_rows(path: Path) -> int:
    return len(read_csv_rows(path))


def count_roi(frame: FrameRun) -> int:
    return count_rows(frame_path(frame, "raw", "reference_roi_manifest_raw_{stem}.csv"))


def count_raw_accepted(frame: FrameRun) -> int:
    rows = read_csv_rows(frame_path(frame, "raw", "cell_features_raw_{stem}.csv"))
    if rows and "accepted_status" in rows[0]:
        return sum(1 for row in rows if str(row.get("accepted_status", "")).strip() == "accepted_cell_candidate")
    return len(rows)


def first_non_empty_csv_value(paths: list[Path], column: str) -> str:
    for path in paths:
        for row in read_csv_rows(path):
            value = str(row.get(column, "")).strip()
            if value:
                return value
    return ""


def count_censored_selected(frame: FrameRun) -> int:
    if "selected_rows" in frame.censoring_summary:
        return int(frame.censoring_summary.get("selected_rows") or 0)
    return count_rows(frame_path(frame, "censored", "cell_features_censored_{stem}.csv"))


def run_raw(args: argparse.Namespace, frame: FrameRun) -> None:
    if args.skip_existing and not missing_paths(frame, "raw", RAW_PATTERNS):
        frame.raw_status = STATUS_PASS
        frame.warnings.append("raw stage reused existing outputs")
        return
    params = dict(run_one_fiji_headless.DEFAULT_PARAMS)
    params["reference_roi_only"] = "true"
    row = run_one_fiji_headless.run_fiji(
        SCRIPT_DIR,
        frame.image_path.resolve(),
        frame.raw_dir,
        args.fiji.resolve(),
        (SCRIPT_DIR / "macros" / "Main_IronCells_headless.ijm").resolve(),
        params,
        args.timeout_seconds,
        args.weka_model.resolve(),
    )
    frame.raw_row = row
    detected_frame_id = first_non_empty_csv_value(
        [
            frame_path(frame, "raw", "frame_features_raw_{stem}.csv"),
            frame_path(frame, "raw", "cell_features_raw_{stem}.csv"),
            frame_path(frame, "raw", "cell_objects_legend_raw_{stem}.csv"),
        ],
        "frame_id",
    )
    if detected_frame_id:
        frame.frame_id = detected_frame_id
        write_frame_metadata(frame)
    if count_roi(frame) == 0:
        frame.raw_status = STATUS_FAIL_REFERENCE_ROI
        frame.error_message = append_message(frame.error_message, "reference ROI was not found; downstream frame stages skipped")
    elif row.get("ok") == "True" and not missing_paths(frame, "raw", RAW_PATTERNS):
        frame.raw_status = STATUS_PASS
    else:
        frame.raw_status = row.get("postprocess_error") or STATUS_FAIL_RAW
        frame.error_message = append_message(
            frame.error_message,
            row.get("postprocess_error") or row.get("missing_outputs") or "raw Fiji/Weka stage failed",
        )


def run_censoring(args: argparse.Namespace, frame: FrameRun) -> None:
    if frame.raw_status != STATUS_PASS:
        frame.censoring_status = STATUS_SKIPPED
        return
    if args.skip_existing and not missing_paths(frame, "censored", CENSORED_PATTERNS):
        frame.censoring_status = STATUS_PASS
        frame.warnings.append("censoring stage reused existing outputs")
        return
    argv = [
        "--cell-features-raw",
        str(frame_path(frame, "raw", "cell_features_raw_{stem}.csv")),
        "--frame-features-raw",
        str(frame_path(frame, "raw", "frame_features_raw_{stem}.csv")),
        "--raw-legend",
        str(frame_path(frame, "raw", "cell_objects_legend_raw_{stem}.csv")),
        "--raw-overlay",
        str(frame_path(frame, "raw", "cell_objects_overlay_raw_{stem}.jpg")),
        "--config",
        str(args.censoring_config),
        "--output-dir",
        str(frame.censored_dir),
    ]
    code = call_noarg_main_with_argv(censor_cell_candidates.main, "censor_cell_candidates.py", argv)
    summary_path = frame.censored_dir / "censoring_summary.json"
    if summary_path.exists():
        frame.censoring_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    missing = missing_paths(frame, "censored", CENSORED_PATTERNS)
    if code == 0 and not missing:
        frame.censoring_status = STATUS_PASS
    else:
        frame.censoring_status = STATUS_FAIL_CENSORING
        detail = f"censoring returned {code}; missing={'; '.join(str(path) for path in missing)}"
        frame.error_message = append_message(frame.error_message, detail)


def run_frame(args: argparse.Namespace, frame: FrameRun) -> None:
    started = perf_counter()
    frame.started_at = datetime.now().isoformat(timespec="seconds")
    try:
        write_frame_metadata(frame)
        if args.dry_run:
            frame.raw_status = STATUS_SKIPPED
            frame.censoring_status = STATUS_SKIPPED
            frame.warnings.append("dry-run: frame stages not executed")
            return
        run_raw(args, frame)
        run_censoring(args, frame)
    except Exception as exc:
        if frame.raw_status == STATUS_PASS:
            frame.censoring_status = STATUS_FAIL_CENSORING
        elif frame.raw_status == STATUS_SKIPPED:
            frame.raw_status = STATUS_FAIL_RAW
        frame.error_message = append_message(frame.error_message, repr(exc))
        if args.stop_on_fatal:
            raise
    finally:
        frame.finished_at = datetime.now().isoformat(timespec="seconds")
        frame.duration_seconds = round(perf_counter() - started, 3)


def run_matrix_export(output_root: Path, preferred: str) -> StageResult:
    try:
        code = build_censored_feature_matrices.main(
            [
                "--input-root",
                str(output_root / "frames"),
                "--output-dir",
                str(output_root),
                "--preferred-censoring-config",
                preferred,
            ]
        )
        if code == 0:
            return StageResult(STATUS_PASS, "matrix export completed")
        return StageResult(STATUS_FAIL_MATRIX, f"build_censored_feature_matrices returned {code}")
    except Exception as exc:
        return StageResult(STATUS_FAIL_MATRIX, repr(exc))


def matrix_has_enough_rows(path: Path) -> tuple[bool, str]:
    rows = read_csv_rows(path)
    if len(rows) < 2:
        return False, f"rows={len(rows)}"
    return True, f"rows={len(rows)}"


def run_pca_stage(matrix: Path, output_dir: Path, config: Path | None, mode: str) -> StageResult:
    if config is None:
        return StageResult(STATUS_FAIL_PCA, f"missing PCA config for {mode}")
    enough, detail = matrix_has_enough_rows(matrix)
    if not enough:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"PCA_{mode}_SKIPPED.txt").write_text(f"{STATUS_SKIPPED_INSUFFICIENT}: {detail}\n", encoding="utf-8")
        return StageResult(STATUS_SKIPPED_INSUFFICIENT, detail)
    try:
        code = run_pca.main(["--input", str(matrix), "--output", str(output_dir), "--config", str(config), "--mode", mode])
        if code == 0:
            return StageResult(STATUS_PASS, "PCA completed")
        return StageResult(STATUS_FAIL_PCA, f"run_pca returned {code}")
    except Exception as exc:
        return StageResult(STATUS_FAIL_PCA, repr(exc))


def make_visualization_config(args: argparse.Namespace, output_root: Path) -> Path | None:
    if args.visualization_config is None:
        return None
    with args.visualization_config.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    config.update(
        {
            "cell_feature_matrix_censored": str(output_root / "cell_feature_matrix_censored.csv"),
            "frame_feature_matrix_censored": str(output_root / "frame_feature_matrix_censored.csv"),
            "cell_features_censored_root": str(output_root / "frames"),
            "raw_legend_root": str(output_root / "frames"),
            "batch_run_report": str(output_root / "batch_run_report.csv"),
            "cli_command": "",
        }
    )
    path = output_root / "visualization_config_effective.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_visualization_stage(args: argparse.Namespace, output_root: Path, cells: StageResult, frames: StageResult) -> StageResult:
    if cells.status != STATUS_PASS or frames.status != STATUS_PASS:
        return StageResult(STATUS_SKIPPED, "visualization skipped because one or both PCA modes did not complete")
    if args.visualization_config is None:
        return StageResult(STATUS_SKIPPED, "visualization config not provided")
    config = make_visualization_config(args, output_root)
    try:
        code = build_pca_visualizations.main(
            [
                "--pca-cells-dir",
                str(output_root / "PCA_cells"),
                "--pca-frames-dir",
                str(output_root / "PCA_frames"),
                "--output-dir",
                str(output_root / "PCA_visualizations"),
                "--config",
                str(config),
            ]
        )
        if code == 0:
            return StageResult(STATUS_PASS, "visualization completed")
        return StageResult(STATUS_FAIL_VISUALIZATION, f"build_pca_visualizations returned {code}")
    except Exception as exc:
        return StageResult(STATUS_FAIL_VISUALIZATION, repr(exc))


def matrix_contains(path: Path, column: str, value: str) -> bool:
    return any(str(row.get(column, "")).strip() == value for row in read_csv_rows(path))


def batch_row(frame: FrameRun) -> dict[str, Any]:
    audit_rows = read_csv_rows(frame_path(frame, "censored", "cell_censoring_audit_{stem}.csv"))
    outside_cell = sum(1 for row in audit_rows if str(row.get("inside_reference_roi", "")).strip().lower() != "true")
    outside_blue = sum(1 for row in audit_rows if str(row.get("blue_pixels", "")).strip() in {"", "0", "0.0"})
    output_root = frame.output_dir.parent.parent
    return {
        "image_name": frame.image_path.name,
        "image_path": str(frame.image_path),
        "frame_id": frame.frame_id,
        "frame_output_dir": str(frame.output_dir),
        "raw_output_dir": str(frame.raw_dir),
        "censored_output_dir": str(frame.censored_dir),
        "raw_status": frame.raw_status,
        "roi_count": count_roi(frame),
        "raw_accepted_count": count_raw_accepted(frame),
        "censored_selected_count": count_censored_selected(frame),
        "outside_cell": outside_cell,
        "outside_blue": outside_blue,
        "cell_matrix_included": str(matrix_contains(output_root / "cell_feature_matrix_censored.csv", "frame_id", frame.frame_id)).lower(),
        "frame_matrix_included": str(matrix_contains(output_root / "frame_feature_matrix_censored.csv", "frame_id", frame.frame_id)).lower(),
        "pca_cells_status": frame.pca_cells_status,
        "pca_frames_status": frame.pca_frames_status,
        "visualization_status": frame.visualization_status,
        "reference_roi_manifest": str(frame_path(frame, "raw", "reference_roi_manifest_raw_{stem}.csv")),
        "cell_features_raw": str(frame_path(frame, "raw", "cell_features_raw_{stem}.csv")),
        "cell_features_censored": str(frame_path(frame, "censored", "cell_features_censored_{stem}.csv")),
        "frame_features_censored": str(frame_path(frame, "censored", "frame_features_censored_{stem}.csv")),
        "processing_started_at": frame.started_at,
        "processing_finished_at": frame.finished_at,
        "duration_seconds": frame.duration_seconds,
        "warnings": "; ".join(frame.warnings),
        "error_message": frame.error_message,
    }


def apply_batch_stage_statuses(frames: list[FrameRun], cells: StageResult, frame_pca: StageResult, viz: StageResult) -> None:
    for frame in frames:
        frame.pca_cells_status = cells.status
        frame.pca_frames_status = frame_pca.status
        frame.visualization_status = viz.status
        for detail in [cells.detail, frame_pca.detail, viz.detail]:
            if detail and detail not in frame.warnings:
                frame.warnings.append(detail)


def validate_matrices(output_root: Path, frames: list[FrameRun]) -> list[Check]:
    cell_matrix = output_root / "cell_feature_matrix_censored.csv"
    frame_matrix = output_root / "frame_feature_matrix_censored.csv"
    checks = [
        Check("cell_feature_matrix_censored.csv exists", cell_matrix.exists(), str(cell_matrix)),
        Check("frame_feature_matrix_censored.csv exists", frame_matrix.exists(), str(frame_matrix)),
    ]
    cell_rows = read_csv_rows(cell_matrix)
    frame_rows = read_csv_rows(frame_matrix)
    valid_frames = [frame for frame in frames if frame.censoring_status == STATUS_PASS and count_censored_selected(frame) > 0]
    checks.append(Check("cell matrix has rows for selected valid frames", len(cell_rows) > 0 if valid_frames else True, f"rows={len(cell_rows)}; valid_selected_frames={len(valid_frames)}"))
    checks.append(Check("frame matrix has rows for selected valid frames", len(frame_rows) > 0 if valid_frames else True, f"rows={len(frame_rows)}; valid_selected_frames={len(valid_frames)}"))
    ids = [str(row.get("feature_row_id", "")).strip() for row in cell_rows]
    checks.append(Check("feature_row_id globally unique in cell matrix", len(ids) == len(set(ids)) and all(ids), f"rows={len(ids)}"))
    frame_ids = [str(row.get("frame_id", "")).strip() for row in frame_rows]
    checks.append(Check("one row per frame_id in frame matrix", len(frame_ids) == len(set(frame_ids)) and all(frame_ids), f"rows={len(frame_ids)}"))
    legacy = [column for column in (frame_rows[0].keys() if frame_rows else []) if column.startswith("selected_")]
    checks.append(Check("legacy selected_* public measurement columns absent", not legacy, f"legacy={legacy}"))
    report = output_root / "feature_matrix_export_report.md"
    if report.exists():
        text = report.read_text(encoding="utf-8", errors="replace")
        checks.append(Check("frame aggregate validation PASS if report exists", "frame aggregate matches selected censored cell rows | PASS" in text, str(report)))
    return checks


def ids_subset(left: Path, right: Path, column: str) -> bool:
    left_values = {str(row.get(column, "")).strip() for row in read_csv_rows(left) if str(row.get(column, "")).strip()}
    right_values = {str(row.get(column, "")).strip() for row in read_csv_rows(right) if str(row.get(column, "")).strip()}
    return bool(left_values) and left_values.issubset(right_values)


def validate_outputs(output_root: Path, frames: list[FrameRun], matrix: StageResult, cells: StageResult, frame_pca: StageResult, viz: StageResult) -> list[Check]:
    checks = [
        Check("output root created", output_root.exists(), str(output_root)),
        Check("per-frame output folders created", all(frame.output_dir.exists() for frame in frames), f"frames={len(frames)}"),
        Check("batch_run_report.csv exists", (output_root / "batch_run_report.csv").exists(), str(output_root / "batch_run_report.csv")),
        Check("batch_run_report.csv has one row per discovered image", count_rows(output_root / "batch_run_report.csv") == len(frames), f"rows={count_rows(output_root / 'batch_run_report.csv')}; images={len(frames)}"),
        Check("pipeline_manifest.json exists", (output_root / "pipeline_manifest.json").exists(), str(output_root / "pipeline_manifest.json")),
    ]
    for frame in frames:
        if frame.raw_status == STATUS_PASS:
            checks.append(Check(f"{frame.frame_id}: raw outputs exist", not missing_paths(frame, "raw", RAW_PATTERNS), "; ".join(str(path) for path in missing_paths(frame, "raw", RAW_PATTERNS))))
            checks.append(Check(f"{frame.frame_id}: raw legend exists", frame_path(frame, "raw", "cell_objects_legend_raw_{stem}.csv").exists(), str(frame.raw_dir)))
            checks.append(Check(f"{frame.frame_id}: ROI manifest exists", frame_path(frame, "raw", "reference_roi_manifest_raw_{stem}.csv").exists(), str(frame.raw_dir)))
        if frame.censoring_status == STATUS_PASS:
            checks.append(Check(f"{frame.frame_id}: censored outputs exist", not missing_paths(frame, "censored", CENSORED_PATTERNS), "; ".join(str(path) for path in missing_paths(frame, "censored", CENSORED_PATTERNS))))
            checks.append(Check(f"{frame.frame_id}: censoring audit exists", frame_path(frame, "censored", "cell_censoring_audit_{stem}.csv").exists(), str(frame.censored_dir)))
        if frame.raw_status == STATUS_FAIL_REFERENCE_ROI:
            checks.append(Check(f"{frame.frame_id}: reference ROI failure controlled", bool(frame.error_message), frame.error_message))
    checks.extend(validate_matrices(output_root, frames))
    if cells.status == STATUS_PASS:
        checks.append(Check("feature_row_id preserved from raw/censored to PCA_Scores_cells", validate_cells_roundtrip(output_root, frames), str(output_root / "PCA_cells" / "PCA_Scores_cells.csv")))
        checks.append(Check("display_label preserved from raw/censored to PCA_Scores_cells", validate_cells_roundtrip(output_root, frames, "display_label"), str(output_root / "PCA_cells" / "PCA_Scores_cells.csv")))
    if frame_pca.status == STATUS_PASS:
        checks.append(Check("frame_id preserved to PCA_Scores_frames", validate_frame_roundtrip(output_root), str(output_root / "PCA_frames" / "PCA_Scores_frames.csv")))
    if viz.status == STATUS_PASS:
        viz_root = output_root / "PCA_visualizations"
        required = [
            "PCA_Legend_cells_PC1_PC2.csv",
            "PCA_Legend_frames_PC1_PC2.csv",
            "PCA_Scatter_PC1_PC2_cells.png",
            "PCA_Scatter_PC1_PC2_cells_numbered.png",
            "PCA_Scatter_PC1_PC2_frames.png",
            "PCA_Scatter_PC1_PC2_frames_numbered.png",
        ]
        missing = [name for name in required if not (viz_root / name).exists()]
        checks.append(Check("PCA visualization outputs exist", not missing, f"missing={missing}"))
    checks.append(Check("matrix stage completed when selected data exists", matrix.status == STATUS_PASS or all(frame.censoring_status != STATUS_PASS for frame in frames), matrix.detail))
    checks.append(Check("PCA cells executed or controlled skipped", cells.status in {STATUS_PASS, STATUS_SKIPPED_INSUFFICIENT, STATUS_SKIPPED, STATUS_FAIL_PCA}, f"{cells.status}: {cells.detail}"))
    checks.append(Check("PCA frames executed or controlled skipped", frame_pca.status in {STATUS_PASS, STATUS_SKIPPED_INSUFFICIENT, STATUS_SKIPPED, STATUS_FAIL_PCA}, f"{frame_pca.status}: {frame_pca.detail}"))
    checks.append(Check("visualization executed if PCA executed", (cells.status != STATUS_PASS or frame_pca.status != STATUS_PASS) or viz.status == STATUS_PASS, f"{viz.status}: {viz.detail}"))
    checks.append(Check("failed frame does not silently disappear from batch report", count_rows(output_root / "batch_run_report.csv") == len(frames), f"frames={len(frames)}"))
    report_text = (output_root / "batch_pipeline_report.md").read_text(encoding="utf-8", errors="replace") if (output_root / "batch_pipeline_report.md").exists() else ""
    checks.append(Check("no report claims actual chemical iron quantification", "concentration" not in report_text.lower(), "report uses proxy-feature wording"))
    return checks


def validate_cells_roundtrip(output_root: Path, frames: list[FrameRun], column: str = "feature_row_id") -> bool:
    scores = read_csv_rows(output_root / "PCA_cells" / "PCA_Scores_cells.csv")
    if not scores:
        return False
    censored_values = set()
    raw_values = set()
    for frame in frames:
        censored_values.update(str(row.get(column, "")).strip() for row in read_csv_rows(frame_path(frame, "censored", "cell_features_censored_{stem}.csv")))
        raw_values.update(str(row.get(column, "")).strip() for row in read_csv_rows(frame_path(frame, "raw", "cell_features_raw_{stem}.csv")))
    score_values = {str(row.get(column, "")).strip() for row in scores if str(row.get(column, "")).strip()}
    return bool(score_values) and score_values.issubset(censored_values) and score_values.issubset(raw_values)


def validate_frame_roundtrip(output_root: Path) -> bool:
    scores = {str(row.get("frame_id", "")).strip() for row in read_csv_rows(output_root / "PCA_frames" / "PCA_Scores_frames.csv")}
    matrix = {str(row.get("frame_id", "")).strip() for row in read_csv_rows(output_root / "frame_feature_matrix_censored.csv")}
    scores.discard("")
    return bool(scores) and scores.issubset(matrix)


def write_manifest(args: argparse.Namespace, output_root: Path, frames: list[FrameRun], git: dict[str, str]) -> None:
    manifest = {
        "git_commit": git["commit"],
        "git_branch": git["branch"],
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(args.input_dir),
        "input_files": [str(frame.image_path) for frame in frames],
        "image_count": len(frames),
        "output_root": str(output_root),
        "fiji_path": str(args.fiji),
        "fiji_version": fiji_version(args.fiji),
        "weka_model_path": str(args.weka_model),
        "weka_model_hash": sha256(args.weka_model),
        "censoring_config_path": str(args.censoring_config),
        "censoring_config_hash": sha256(args.censoring_config),
        "pca_config_cells_path": str(args.pca_config_cells or ""),
        "pca_config_cells_hash": sha256(args.pca_config_cells),
        "pca_config_frames_path": str(args.pca_config_frames or ""),
        "pca_config_frames_hash": sha256(args.pca_config_frames),
        "visualization_config_path": str(args.visualization_config or ""),
        "visualization_config_hash": sha256(args.visualization_config),
        "scripts": {
            "raw_entrypoint": str(SCRIPT_DIR / "run_one_fiji_headless.py"),
            "censor_cell_candidates": str(SCRIPT_DIR / "censor_cell_candidates.py"),
            "build_censored_feature_matrices": str(SCRIPT_DIR / "build_censored_feature_matrices.py"),
            "run_pca": str(SCRIPT_DIR / "run_pca.py"),
            "build_pca_visualizations": str(SCRIPT_DIR / "build_pca_visualizations.py"),
        },
        "staging_map": [
            {
                "frame_index": frame.index,
                "frame_id": frame.frame_id,
                "original_path": str(frame.image_path),
                "original_file_name": frame.image_path.name,
                "original_stem": frame.image_stem,
                "staged_path": str(frame.staged_input),
                "frame_output_dir": str(frame.output_dir),
            }
            for frame in frames
        ],
    }
    (output_root / "pipeline_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def status_summary(frames: list[FrameRun], attr: str) -> dict[str, int]:
    summary: dict[str, int] = {}
    for frame in frames:
        value = str(getattr(frame, attr))
        summary[value] = summary.get(value, 0) + 1
    return summary


def write_report(
    output_root: Path,
    args: argparse.Namespace,
    frames: list[FrameRun],
    checks: list[Check],
    matrix: StageResult,
    cells: StageResult,
    frame_pca: StageResult,
    viz: StageResult,
    git: dict[str, str],
    cli: str,
) -> None:
    failed = [frame for frame in frames if frame.raw_status != STATUS_PASS or frame.censoring_status not in {STATUS_PASS, STATUS_SKIPPED}]
    passed = [frame for frame in frames if frame.raw_status == STATUS_PASS and frame.censoring_status == STATUS_PASS]
    final = "PASS" if all(check.passed for check in checks) else ("PARTIAL" if passed else "FAIL")
    lines = [
        "# Batch Pipeline Report",
        "",
        "## 1. Task name",
        "",
        "Production batch orchestration for IronCellQuant.",
        "",
        "## 2. Branch",
        "",
        f"`{git['branch']}`",
        "",
        "## 3. Git commit",
        "",
        f"`{git['commit']}`",
        "",
        "## 4. CLI command",
        "",
        "```text",
        cli,
        "```",
        "",
        "## 5. Input dir",
        "",
        f"`{args.input_dir}`",
        "",
        "## 6. Output root",
        "",
        f"`{output_root}`",
        "",
        "## 7. Number of discovered images",
        "",
        str(len(frames)),
        "",
        "## 8. Number of processed/PASS images",
        "",
        str(len(passed)),
        "",
        "## 9. Number of failed/skipped images",
        "",
        str(len(frames) - len(passed)),
        "",
        "## 10. Per-stage summary",
        "",
        f"- raw: `{status_summary(frames, 'raw_status')}`",
        f"- censoring: `{status_summary(frames, 'censoring_status')}`",
        f"- matrix export: `{matrix.status}` - {matrix.detail}",
        f"- PCA cells: `{cells.status}` - {cells.detail}",
        f"- PCA frames: `{frame_pca.status}` - {frame_pca.detail}",
        f"- visualization: `{viz.status}` - {viz.detail}",
        "",
        "## 11. Root outputs",
        "",
        f"- `{output_root / 'batch_run_report.csv'}`",
        f"- `{output_root / 'batch_pipeline_report.md'}`",
        f"- `{output_root / 'pipeline_manifest.json'}`",
        f"- `{output_root / 'cell_feature_matrix_censored.csv'}`",
        f"- `{output_root / 'frame_feature_matrix_censored.csv'}`",
        f"- `{output_root / 'PCA_cells'}`",
        f"- `{output_root / 'PCA_frames'}`",
        f"- `{output_root / 'PCA_visualizations'}`",
        "",
        "## 12. Batch report columns",
        "",
        ", ".join(f"`{column}`" for column in BATCH_REPORT_COLUMNS),
        "",
        "## 13. Validation table",
        "",
        "| Check | Result | Detail |",
        "|---|---:|---|",
    ]
    for check in checks:
        detail = check.detail.replace("|", "\\|")
        lines.append(f"| {check.name} | {'PASS' if check.passed else 'FAIL'} | {detail} |")
    warnings = [warning for frame in frames for warning in frame.warnings]
    errors = [f"{frame.frame_id}: {frame.error_message}" for frame in frames if frame.error_message]
    lines.extend(
        [
            "",
            "## 14. Warnings",
            "",
            "- none" if not warnings else "\n".join(f"- {warning}" for warning in warnings),
            "",
            "## 15. Errors",
            "",
            "- none" if not errors else "\n".join(f"- {error}" for error in errors),
            "",
            "## 16. Interpretation statement",
            "",
            "Results describe statistical associations and preliminary blue-pixel optical proxy features only; they are not chemical quantification of iron.",
            "",
            "## 17. Git status",
            "",
            "```text",
            git["status_short"],
            "```",
            "",
            "## 18. Git diff --stat",
            "",
            "```text",
            git["diff_stat"],
            "```",
            "",
            "## 19. Final status",
            "",
            final,
            "",
            "## Failure handling",
            "",
            "Frame-level raw/censoring failures are recorded in `batch_run_report.csv`; downstream stages for that frame are skipped and remaining frames continue. Fatal configuration errors stop the batch after writing a fatal report when possible.",
            "",
        ]
    )
    text = "\n".join(lines)
    (output_root / "batch_pipeline_report.md").write_text(text, encoding="utf-8")
    if args.task_report:
        args.task_report.parent.mkdir(parents=True, exist_ok=True)
        args.task_report.write_text(text, encoding="utf-8")


def fatal_report(args: argparse.Namespace, messages: list[str], cli: str) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    git = git_info()
    text = "\n".join(
        [
            "# Batch Pipeline Fatal Report",
            "",
            "Status: FAIL",
            "",
            "## CLI",
            "",
            "```text",
            cli,
            "```",
            "",
            "## Errors",
            "",
            *[f"- {message}" for message in messages],
            "",
            "## Git status",
            "",
            "```text",
            git["status_short"],
            "```",
            "",
            "## Git diff --stat",
            "",
            "```text",
            git["diff_stat"],
            "```",
            "",
        ]
    )
    (args.output_dir / "batch_pipeline_report.md").write_text(text, encoding="utf-8")
    if args.task_report:
        args.task_report.parent.mkdir(parents=True, exist_ok=True)
        args.task_report.write_text(text, encoding="utf-8")
    print(text)
    return 2


def fatal_errors(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    required = {
        "input-dir": args.input_dir,
        "fiji": args.fiji,
        "weka-model": args.weka_model,
        "censoring-config": args.censoring_config,
        "run_one_fiji_headless.py": SCRIPT_DIR / "run_one_fiji_headless.py",
        "censor_cell_candidates.py": SCRIPT_DIR / "censor_cell_candidates.py",
        "build_censored_feature_matrices.py": SCRIPT_DIR / "build_censored_feature_matrices.py",
    }
    for name, path in required.items():
        if not path.exists():
            errors.append(f"missing {name}: {path}")
    if args.run_pca:
        for name, path in {"run_pca.py": SCRIPT_DIR / "run_pca.py", "build_pca_visualizations.py": SCRIPT_DIR / "build_pca_visualizations.py"}.items():
            if not path.exists():
                errors.append(f"missing {name}: {path}")
    if args.limit is not None and args.limit < 0:
        errors.append("--limit must be >= 0")
    return errors


def prepare_output_root(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    configure_stdio()
    args = parse_args(argv)
    cli = "python .\\run_batch_pipeline.py " + " ".join(sys.argv[1:] if argv is None else argv)
    errors = fatal_errors(args)
    if errors:
        return fatal_report(args, errors, cli)
    prepare_output_root(args)
    extensions = normalized_extensions(args.image_ext)
    images = discover_images(args.input_dir, extensions, args.recursive, args.limit)
    if not images:
        return fatal_report(args, [f"no supported image files discovered in {args.input_dir}"], cli)

    frames = [create_frame(image, index + 1, args.output_dir) for index, image in enumerate(images)]
    if args.dry_run:
        for frame in frames:
            frame.started_at = frame.finished_at = datetime.now().isoformat(timespec="seconds")
            frame.warnings.append("dry-run: discovered only")
    else:
        for frame in frames:
            run_frame(args, frame)

    matrix = StageResult(STATUS_SKIPPED, "dry-run") if args.dry_run else run_matrix_export(args.output_dir, args.preferred_censoring_config)
    cells = StageResult(STATUS_SKIPPED, "PCA not requested")
    frame_pca = StageResult(STATUS_SKIPPED, "PCA not requested")
    if args.run_pca and not args.dry_run and matrix.status == STATUS_PASS:
        cells = run_pca_stage(args.output_dir / "cell_feature_matrix_censored.csv", args.output_dir / "PCA_cells", args.pca_config_cells, "cells")
        frame_pca = run_pca_stage(args.output_dir / "frame_feature_matrix_censored.csv", args.output_dir / "PCA_frames", args.pca_config_frames, "frames")
    elif args.run_pca and matrix.status != STATUS_PASS:
        cells = StageResult(STATUS_FAIL_PCA, "matrix stage did not complete")
        frame_pca = StageResult(STATUS_FAIL_PCA, "matrix stage did not complete")

    pre_viz = StageResult(STATUS_SKIPPED, "visualization pending")
    apply_batch_stage_statuses(frames, cells, frame_pca, pre_viz)
    write_csv(args.output_dir / "batch_run_report.csv", BATCH_REPORT_COLUMNS, [batch_row(frame) for frame in frames])
    viz = StageResult(STATUS_SKIPPED, "dry-run") if args.dry_run else run_visualization_stage(args, args.output_dir, cells, frame_pca)
    apply_batch_stage_statuses(frames, cells, frame_pca, viz)
    write_csv(args.output_dir / "batch_run_report.csv", BATCH_REPORT_COLUMNS, [batch_row(frame) for frame in frames])
    git = git_info()
    write_manifest(args, args.output_dir, frames, git)
    checks = validate_outputs(args.output_dir, frames, matrix, cells, frame_pca, viz)
    write_report(args.output_dir, args, frames, checks, matrix, cells, frame_pca, viz, git_info(), cli)
    status = "PASS" if all(check.passed for check in checks) else "FAIL"
    print(json.dumps({"status": status, "image_count": len(frames), "output_root": str(args.output_dir)}, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
