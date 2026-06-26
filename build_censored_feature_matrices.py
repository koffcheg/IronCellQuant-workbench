"""Build censored cell-level and frame-level feature matrices for PCA input."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CELL_SERVICE_COLUMNS = [
    "image_name",
    "frame_id",
    "object_id",
    "feature_row_id",
    "display_label",
    "object_type",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "reference_roi_id",
    "inside_reference_roi",
    "reference_roi_overlap_fraction",
]

FRAME_SERVICE_COLUMNS = [
    "image_name",
    "frame_id",
    "feature_row_id",
    "object_type",
    "object_id",
    "cell_count",
    "object_pixels",
    "blue_pixels",
    "blue_pixel_fraction",
    "blue_pixel_percent",
    "aggregation_scope",
    "source_object_set",
]

LEGACY_SELECTED_MEASUREMENT_COLUMNS = {
    "selected_cell_count",
    "selected_object_pixels",
    "selected_blue_pixels",
    "selected_blue_pixel_fraction",
    "selected_blue_pixel_percent",
}

CELL_PREFIX = "cell_features_censored_"
FRAME_PREFIX = "frame_features_censored_"
AUDIT_PREFIX = "cell_censoring_audit_"
CSV_SUFFIX = ".csv"

AUDIT_STATUS_COLUMNS = {
    "accepted_status",
    "candidate_status",
    "censoring_frame_status",
    "censoring_reason",
    "censoring_status",
    "failed_stage",
    "last_passed_stage",
    "reference_roi_selection_note",
    "reject_reason",
    "roi_reconstruction_status",
    "selected_for_censored_frame",
    "selected_for_frame_summary",
    "selection_penalty_full_blue",
    "selection_rank_blue",
    "selection_rank_size",
    "selection_review_note",
    "selection_score",
    "stage_trace",
    "warn_full_blue_candidate",
}

CELL_AUDIT_ONLY_COLUMNS = {
    "censoring_status",
    "censoring_reason",
    "failed_stage",
    "last_passed_stage",
    "stage_trace",
    "selected_for_censored_frame",
}

STATUS_VALUES = {"PASS", "FAIL_NO_CENSORED_OBJECTS"}


@dataclass
class Table:
    path: Path
    fieldnames: list[str]
    rows: list[dict[str, str]]


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    detail: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect per-frame censored outputs into cell-level and frame-level feature matrices."
    )
    parser.add_argument("--input-root", required=True, type=Path, help="Directory containing task 3 censored outputs.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for exported matrices and report.")
    parser.add_argument(
        "--preferred-censoring-config",
        default="config_b_standard",
        help=(
            "When input-root contains multiple task-3 censoring scenarios, use files under this directory name. "
            "Set to an empty string to disable the preference."
        ),
    )
    return parser.parse_args(argv)


def read_csv(path: Path) -> Table:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return Table(path=path, fieldnames=list(reader.fieldnames or []), rows=list(reader))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def is_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "y"}


def as_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        result = float(value)
        if math.isnan(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def is_numeric_column(rows: list[dict[str, str]], column: str) -> bool:
    values = [row.get(column, "") for row in rows if str(row.get(column, "")).strip() != ""]
    if not values:
        return False
    return all(as_float(value) is not None for value in values)


def suffix_key(path: Path, prefix: str) -> str:
    name = path.name
    if name.startswith(prefix) and name.endswith(CSV_SUFFIX):
        return name[len(prefix) : -len(CSV_SUFFIX)]
    return path.stem


def preferred_paths(paths: list[Path], preferred_config: str) -> tuple[list[Path], str]:
    if not preferred_config:
        return paths, "no preferred censoring config requested"
    preferred = [path for path in paths if preferred_config in path.parts]
    if preferred:
        return preferred, f"using files under `{preferred_config}`"
    return paths, f"`{preferred_config}` not found; using all matching files"


def tables_by_image(paths: list[Path], prefix: str) -> dict[str, Table]:
    tables: dict[str, Table] = {}
    duplicates: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(paths):
        key = suffix_key(path, prefix)
        duplicates[key].append(path)
        tables[key] = read_csv(path)
    repeated = {key: values for key, values in duplicates.items() if len(values) > 1}
    if repeated:
        details = "; ".join(f"{key}: {[str(path) for path in values]}" for key, values in sorted(repeated.items()))
        raise ValueError(f"multiple {prefix} files for the same image key after filtering: {details}")
    return tables


def discover_tables(
    input_root: Path,
    preferred_config: str,
) -> tuple[dict[str, Table], dict[str, Table], dict[str, Table], str]:
    cell_paths, cell_filter_note = preferred_paths(
        sorted(input_root.rglob(f"{CELL_PREFIX}*{CSV_SUFFIX}")),
        preferred_config,
    )
    frame_paths, frame_filter_note = preferred_paths(
        sorted(input_root.rglob(f"{FRAME_PREFIX}*{CSV_SUFFIX}")),
        preferred_config,
    )
    audit_paths, audit_filter_note = preferred_paths(
        sorted(input_root.rglob(f"{AUDIT_PREFIX}*{CSV_SUFFIX}")),
        preferred_config,
    )
    filter_note = f"cell: {cell_filter_note}; frame: {frame_filter_note}; audit: {audit_filter_note}"
    return (
        tables_by_image(cell_paths, CELL_PREFIX),
        tables_by_image(frame_paths, FRAME_PREFIX),
        tables_by_image(audit_paths, AUDIT_PREFIX),
        filter_note,
    )


def ordered_union(tables: list[Table]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for table in tables:
        for column in table.fieldnames:
            if column not in seen:
                seen.add(column)
                columns.append(column)
    return columns


def classify_columns(
    rows: list[dict[str, str]],
    source_columns: list[str],
    service_columns: list[str],
) -> tuple[list[str], list[str], list[str]]:
    service = [column for column in service_columns if column in source_columns or column in service_columns]
    excluded = [
        column
        for column in source_columns
        if column not in service and (column in AUDIT_STATUS_COLUMNS or column in CELL_AUDIT_ONLY_COLUMNS)
    ]
    measurement = [
        column
        for column in source_columns
        if column not in service and column not in excluded and is_numeric_column(rows, column)
    ]
    non_numeric_context = [
        column
        for column in source_columns
        if column not in service and column not in measurement and column not in excluded
    ]
    excluded.extend(non_numeric_context)
    return service, measurement, excluded


def normalize_cell_rows(cell_tables: dict[str, Table]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key in sorted(cell_tables):
        rows.extend(dict(row) for row in cell_tables[key].rows)
    return rows


def normalize_frame_rows(frame_tables: dict[str, Table]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key in sorted(frame_tables):
        for source_row in frame_tables[key].rows:
            row = dict(source_row)
            frame_id = str(row.get("frame_id", "")).strip()
            row["feature_row_id"] = f"{frame_id}_censored_frame" if frame_id else str(row.get("feature_row_id", "")).strip()
            row["object_type"] = "censored_frame_cell_material"
            row["object_id"] = "frame"
            rows.append(row)
    return rows


def selected_audit_ids(audit_tables: dict[str, Table]) -> tuple[set[str], set[str]]:
    selected: set[str] = set()
    rejected: set[str] = set()
    for table in audit_tables.values():
        for row in table.rows:
            feature_row_id = str(row.get("feature_row_id", "")).strip()
            if not feature_row_id:
                continue
            if is_truthy(row.get("selected_for_censored_frame")):
                selected.add(feature_row_id)
            else:
                rejected.add(feature_row_id)
    return selected, rejected


def check_unique(rows: list[dict[str, str]], column: str) -> tuple[bool, str]:
    values = [str(row.get(column, "")).strip() for row in rows]
    missing = sum(1 for value in values if not value)
    duplicates = sorted(value for value, count in Counter(values).items() if value and count > 1)
    if missing or duplicates:
        return False, f"missing={missing}; duplicates={duplicates[:10]}"
    return True, f"{len(values)} non-empty unique values"


def check_display_labels(rows: list[dict[str, str]]) -> tuple[bool, str]:
    duplicates: list[str] = []
    by_frame: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        frame_id = str(row.get("frame_id", "")).strip()
        label = str(row.get("display_label", "")).strip()
        if label:
            by_frame[frame_id][label] += 1
    for frame_id, counter in by_frame.items():
        for label, count in counter.items():
            if count > 1:
                duplicates.append(f"{frame_id}:{label}")
    if duplicates:
        return False, "duplicates=" + ", ".join(duplicates[:10])
    return True, "no duplicated display_label within frame_id"


def check_frame_aggregates(cell_rows: list[dict[str, str]], frame_rows: list[dict[str, str]]) -> tuple[bool, str]:
    tolerance = 1e-6
    cells_by_frame: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in cell_rows:
        cells_by_frame[str(row.get("frame_id", "")).strip()].append(row)

    mismatches: list[str] = []
    for frame_row in frame_rows:
        frame_id = str(frame_row.get("frame_id", "")).strip()
        selected_cells = cells_by_frame.get(frame_id, [])
        expected_object_pixels = sum(as_float(row.get("object_pixels")) or 0.0 for row in selected_cells)
        expected_blue_pixels = sum(as_float(row.get("blue_pixels")) or 0.0 for row in selected_cells)
        expected_fraction = expected_blue_pixels / expected_object_pixels if expected_object_pixels else 0.0
        expected = {
            "cell_count": float(len(selected_cells)),
            "object_pixels": expected_object_pixels,
            "blue_pixels": expected_blue_pixels,
            "blue_pixel_fraction": expected_fraction,
            "blue_pixel_percent": 100.0 * expected_fraction,
        }
        for column, expected_value in expected.items():
            actual_value = as_float(frame_row.get(column))
            if actual_value is None:
                mismatches.append(
                    f"frame_id={frame_id}; column={column}; expected={expected_value}; actual=<missing>; abs_diff=<missing>"
                )
                continue
            diff = abs(actual_value - expected_value)
            if diff > tolerance:
                mismatches.append(
                    f"frame_id={frame_id}; column={column}; expected={expected_value}; actual={actual_value}; abs_diff={diff}"
                )

    if mismatches:
        return False, "; ".join(mismatches[:20])
    return True, f"checked {len(frame_rows)} frame row(s) against selected censored cell rows"


def validate_export(
    cell_rows: list[dict[str, str]],
    frame_rows: list[dict[str, str]],
    cell_fieldnames: list[str],
    frame_fieldnames: list[str],
    cell_measurement: list[str],
    frame_measurement: list[str],
    cell_excluded: list[str],
    frame_excluded: list[str],
    audit_tables: dict[str, Table],
    output_dir: Path,
) -> list[ValidationCheck]:
    checks: list[ValidationCheck] = []
    checks.append(
        ValidationCheck(
            "cell_feature_matrix_censored.csv created",
            (output_dir / "cell_feature_matrix_censored.csv").exists(),
            str(output_dir / "cell_feature_matrix_censored.csv"),
        )
    )
    checks.append(
        ValidationCheck(
            "frame_feature_matrix_censored.csv created",
            (output_dir / "frame_feature_matrix_censored.csv").exists(),
            str(output_dir / "frame_feature_matrix_censored.csv"),
        )
    )

    selected_ids, rejected_ids = selected_audit_ids(audit_tables)
    cell_ids = {str(row.get("feature_row_id", "")).strip() for row in cell_rows if str(row.get("feature_row_id", "")).strip()}
    if audit_tables:
        not_selected = sorted((cell_ids - selected_ids) | (cell_ids & rejected_ids))
        checks.append(
            ValidationCheck(
                "cell matrix contains only selected censored rows",
                not not_selected,
                f"audit_checked={len(selected_ids)} selected ids; non_selected_in_matrix={not_selected[:10]}",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                "cell matrix contains only selected censored rows",
                True,
                "no audit files found; cell_features_censored inputs are treated as selected-only outputs",
            )
        )

    frame_counts = Counter(str(row.get("frame_id", "")).strip() for row in frame_rows)
    duplicated_frames = sorted(frame_id for frame_id, count in frame_counts.items() if frame_id and count > 1)
    missing_frame_ids = sum(1 for row in frame_rows if not str(row.get("frame_id", "")).strip())
    checks.append(
        ValidationCheck(
            "frame matrix has one row per frame_id",
            not duplicated_frames and not missing_frame_ids,
            f"rows={len(frame_rows)}; duplicated={duplicated_frames[:10]}; missing_frame_id={missing_frame_ids}",
        )
    )

    passed, detail = check_unique(cell_rows, "feature_row_id")
    checks.append(ValidationCheck("feature_row_id non-empty and globally unique in cell matrix", passed, detail))

    passed, detail = check_unique(frame_rows, "feature_row_id")
    checks.append(ValidationCheck("frame-level feature_row_id globally unique in frame matrix", passed, detail))

    passed, detail = check_unique(frame_rows, "frame_id")
    checks.append(ValidationCheck("frame_id unique in frame matrix", passed, detail))

    missing_cell_service = [column for column in CELL_SERVICE_COLUMNS if column not in cell_fieldnames]
    missing_frame_service = [column for column in FRAME_SERVICE_COLUMNS if column not in frame_fieldnames]
    checks.append(
        ValidationCheck(
            "required service columns present",
            not missing_cell_service and not missing_frame_service,
            f"missing_cell={missing_cell_service}; missing_frame={missing_frame_service}",
        )
    )

    passed, detail = check_display_labels(cell_rows)
    checks.append(ValidationCheck("display_label not duplicated within frame_id", passed, detail))

    legacy_frame_measurements = sorted(LEGACY_SELECTED_MEASUREMENT_COLUMNS & set(frame_fieldnames))
    checks.append(
        ValidationCheck(
            "legacy selected_* frame measurement columns absent",
            not legacy_frame_measurements,
            f"legacy_columns={legacy_frame_measurements}",
        )
    )

    passed, detail = check_frame_aggregates(cell_rows, frame_rows)
    checks.append(ValidationCheck("frame aggregate matches selected censored cell rows", passed, detail))

    false_rows = [
        str(row.get("feature_row_id", "")).strip()
        for row in cell_rows
        if str(row.get("selected_for_censored_frame", "")).strip().lower() == "false"
    ]
    checks.append(
        ValidationCheck(
            "no selected_for_censored_frame=false rows in cell matrix",
            not false_rows,
            f"false_rows={false_rows[:10]}",
        )
    )

    audit_as_measurement = sorted(
        (set(cell_measurement) | set(frame_measurement)) & (AUDIT_STATUS_COLUMNS | CELL_AUDIT_ONLY_COLUMNS)
    )
    checks.append(
        ValidationCheck(
            "audit/status columns are not measurement features",
            not audit_as_measurement,
            f"audit_as_measurement={audit_as_measurement}; excluded={sorted((set(cell_excluded) | set(frame_excluded)) & (AUDIT_STATUS_COLUMNS | CELL_AUDIT_ONLY_COLUMNS))}",
        )
    )

    unknown_status = sorted(
        {
            str(row.get("censoring_frame_status", "")).strip()
            for row in frame_rows
            if str(row.get("censoring_frame_status", "")).strip()
            and str(row.get("censoring_frame_status", "")).strip() not in STATUS_VALUES
        }
    )
    fail_rows = [
        str(row.get("frame_id", "")).strip()
        for row in frame_rows
        if str(row.get("censoring_frame_status", "")).strip() == "FAIL_NO_CENSORED_OBJECTS"
    ]
    checks.append(
        ValidationCheck(
            "FAIL_NO_CENSORED_OBJECTS frames handled explicitly",
            not unknown_status,
            f"fail_frames={fail_rows}; unknown_status={unknown_status}",
        )
    )
    return checks


def markdown_list(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- `{item}`" for item in items)


def write_report(
    path: Path,
    input_root: Path,
    filter_note: str,
    cell_tables: dict[str, Table],
    frame_tables: dict[str, Table],
    audit_tables: dict[str, Table],
    cell_rows: list[dict[str, str]],
    frame_rows: list[dict[str, str]],
    cell_service: list[str],
    frame_service: list[str],
    cell_measurement: list[str],
    frame_measurement: list[str],
    cell_excluded: list[str],
    frame_excluded: list[str],
    checks: list[ValidationCheck],
) -> None:
    status = "PASS" if all(check.passed for check in checks) else "FAIL"
    lines = [
        "# Feature Matrix Export Report",
        "",
        f"Status: **{status}**",
        "",
        "## Inputs",
        "",
        f"- input_root: `{input_root}`",
        f"- censoring config selection: {filter_note}",
        f"- cell feature files: {len(cell_tables)}",
        f"- frame feature files: {len(frame_tables)}",
        f"- audit files: {len(audit_tables)}",
        "",
        "## Outputs",
        "",
        f"- `cell_feature_matrix_censored.csv`: {len(cell_rows)} rows",
        f"- `frame_feature_matrix_censored.csv`: {len(frame_rows)} rows",
        f"- `feature_matrix_export_report.md`: this report",
        "",
        "## Cell Service Columns",
        "",
        markdown_list(cell_service),
        "",
        "## Frame Service Columns",
        "",
        markdown_list(frame_service),
        "",
        "## Cell Numeric Measurement Feature Columns",
        "",
        markdown_list(cell_measurement),
        "",
        "## Frame Numeric Measurement Feature Columns",
        "",
        markdown_list(frame_measurement),
        "",
        "## Public Measurement Schema Synchronization",
        "",
        "Public measurement column names were synchronized between cell-level and frame-level outputs.",
        "",
        "The aggregation formulas were not changed.",
        "",
        "For frame-level rows, `object_pixels`, `blue_pixels`, `blue_pixel_fraction`, and `blue_pixel_percent` describe censored selected cell material of the frame.",
        "",
        "The selected/censored semantics are represented by `object_type = censored_frame_cell_material`, `aggregation_scope = censored_selected_objects`, and `source_object_set = selected_censored_objects`.",
        "",
        "Audit/status fields still keep selected terminology: `selected_for_censored_frame` and `censoring_status`.",
        "",
        "## Excluded Audit/Status/Context Columns",
        "",
        "### Cell",
        "",
        markdown_list(cell_excluded),
        "",
        "### Frame",
        "",
        markdown_list(frame_excluded),
        "",
        "## Validation Checks",
        "",
        "| Check | Result | Detail |",
        "|---|---:|---|",
    ]
    for check in checks:
        result = "PASS" if check.passed else "FAIL"
        detail = check.detail.replace("|", "\\|")
        lines.append(f"| {check.name} | {result} | {detail} |")
    lines.extend(
        [
            "",
            "## PCA Execution",
            "",
            "PCA was not executed by this export step.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    if not args.input_root.exists():
        print(f"Input root not found: {args.input_root}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cell_tables, frame_tables, audit_tables, filter_note = discover_tables(
            args.input_root,
            str(args.preferred_censoring_config or ""),
        )
    except ValueError as exc:
        print(f"Input discovery error: {exc}", file=sys.stderr)
        return 1
    if not cell_tables:
        print(f"No {CELL_PREFIX}*.csv files found under {args.input_root}", file=sys.stderr)
        return 1
    if not frame_tables:
        print(f"No {FRAME_PREFIX}*.csv files found under {args.input_root}", file=sys.stderr)
        return 1

    cell_rows = normalize_cell_rows(cell_tables)
    frame_rows = normalize_frame_rows(frame_tables)
    cell_source_columns = ordered_union(list(cell_tables.values()))
    frame_source_columns = ordered_union(list(frame_tables.values()))

    cell_service, cell_measurement, cell_excluded = classify_columns(cell_rows, cell_source_columns, CELL_SERVICE_COLUMNS)
    frame_service, frame_measurement, frame_excluded = classify_columns(frame_rows, frame_source_columns, FRAME_SERVICE_COLUMNS)
    cell_output_columns = cell_service + cell_measurement
    frame_output_columns = frame_service + frame_measurement
    if "censoring_frame_status" in frame_source_columns:
        frame_output_columns.append("censoring_frame_status")
        if "censoring_frame_status" not in frame_excluded:
            frame_excluded.append("censoring_frame_status")

    write_csv(args.output_dir / "cell_feature_matrix_censored.csv", cell_output_columns, cell_rows)
    write_csv(args.output_dir / "frame_feature_matrix_censored.csv", frame_output_columns, frame_rows)

    checks = validate_export(
        cell_rows,
        frame_rows,
        cell_output_columns,
        frame_output_columns,
        cell_measurement,
        frame_measurement,
        cell_excluded,
        frame_excluded,
        audit_tables,
        args.output_dir,
    )
    write_report(
        args.output_dir / "feature_matrix_export_report.md",
        args.input_root,
        filter_note,
        cell_tables,
        frame_tables,
        audit_tables,
        cell_rows,
        frame_rows,
        cell_service,
        frame_service,
        cell_measurement,
        frame_measurement,
        cell_excluded,
        frame_excluded,
        checks,
    )

    summary = {
        "status": "PASS" if all(check.passed for check in checks) else "FAIL",
        "cell_rows": len(cell_rows),
        "frame_rows": len(frame_rows),
        "cell_measurement_columns": cell_measurement,
        "frame_measurement_columns": frame_measurement,
        "outputs": {
            "cell_feature_matrix_censored": str(args.output_dir / "cell_feature_matrix_censored.csv"),
            "frame_feature_matrix_censored": str(args.output_dir / "frame_feature_matrix_censored.csv"),
            "feature_matrix_export_report": str(args.output_dir / "feature_matrix_export_report.md"),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
