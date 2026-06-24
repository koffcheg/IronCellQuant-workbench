"""Input FeatureMatrix validation and data check report generation."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import PCAConfig
from .data_io import load_feature_matrix, read_csv_header, write_text_report


@dataclass
class ValidationResult:
    dataframe: pd.DataFrame | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    missing_value_count: int = 0
    numeric_columns: list[str] = field(default_factory=list)
    non_numeric_columns: list[str] = field(default_factory=list)
    empty_columns: list[str] = field(default_factory=list)
    zero_variance_numeric_columns: list[str] = field(default_factory=list)
    field_count_errors: list[str] = field(default_factory=list)
    numeric_candidate_columns: list[str] = field(default_factory=list)
    non_numeric_excluded_columns: list[str] = field(default_factory=list)
    invalid_numeric_counts: dict[str, int] = field(default_factory=dict)
    missing_numeric_counts: dict[str, int] = field(default_factory=dict)

    @property
    def can_continue(self) -> bool:
        return not self.errors


def validate_input(input_path: str | Path, output_dir: str | Path, config: PCAConfig) -> ValidationResult:
    result = ValidationResult()
    path = Path(input_path)

    if not path.exists():
        result.errors.append(f"Input file does not exist: {path}")
        write_data_check_report(output_dir, result)
        return result
    if not path.is_file():
        result.errors.append(f"Input path is not a file: {path}")
        write_data_check_report(output_dir, result)
        return result

    try:
        header = read_csv_header(path, config.delimiter)
    except Exception as exc:
        result.errors.append(f"Failed to read CSV header: {exc}")
        write_data_check_report(output_dir, result)
        return result

    if not header:
        result.errors.append("CSV header is missing or empty.")
        write_data_check_report(output_dir, result)
        return result

    result.field_count_errors = _check_csv_field_counts(path, config.delimiter, len(header))
    if result.field_count_errors:
        result.errors.extend(result.field_count_errors)
        write_data_check_report(output_dir, result)
        return result

    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        result.errors.append(f"Duplicate column name(s): {', '.join(duplicates)}")

    try:
        dataframe = load_feature_matrix(path, config.delimiter)
    except Exception as exc:
        result.errors.append(f"Failed to read CSV file: {exc}")
        write_data_check_report(output_dir, result)
        return result

    result.dataframe = dataframe
    result.row_count = len(dataframe)
    result.column_count = len(dataframe.columns)

    if dataframe.empty:
        result.errors.append("CSV file is empty.")

    missing_service_columns = [column for column in config.service_columns or [] if column not in dataframe.columns]
    if missing_service_columns:
        result.errors.append(f"Missing required service column(s): {', '.join(missing_service_columns)}")

    result.missing_value_count = int(dataframe.isna().sum().sum())
    result.empty_columns = [column for column in dataframe.columns if dataframe[column].isna().all()]
    result.numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    result.non_numeric_columns = [column for column in dataframe.columns if column not in result.numeric_columns]
    result.zero_variance_numeric_columns = [
        column for column in result.numeric_columns if dataframe[column].nunique(dropna=True) <= 1
    ]

    candidate_columns, pca_candidate_columns = _candidate_columns(dataframe, config, result.errors)
    (
        result.numeric_candidate_columns,
        result.non_numeric_excluded_columns,
        result.invalid_numeric_counts,
        result.missing_numeric_counts,
    ) = _profile_candidate_columns(dataframe, candidate_columns, config.include_columns)

    pca_numeric_feature_columns = [
        column for column in pca_candidate_columns if column in result.numeric_candidate_columns
    ]
    if result.row_count < 2:
        result.errors.append("At least 2 rows/objects are required.")
    if len(pca_numeric_feature_columns) < 2:
        result.errors.append(
            "At least 2 numeric feature columns are required after service/excluded columns are removed."
        )

    if result.empty_columns:
        result.warnings.append(f"Fully empty column(s): {', '.join(result.empty_columns)}")
    if result.zero_variance_numeric_columns:
        result.warnings.append(
            f"Zero-variance numeric column(s): {', '.join(result.zero_variance_numeric_columns)}"
        )
    if result.invalid_numeric_counts:
        joined = ", ".join(
            f"{column}={count}" for column, count in sorted(result.invalid_numeric_counts.items())
        )
        result.warnings.append(f"Invalid numeric value(s) detected in candidate column(s): {joined}")

    write_data_check_report(output_dir, result)
    return result


def _check_csv_field_counts(path: Path, delimiter: str, expected_count: int) -> list[str]:
    errors: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        next(reader, None)
        for line_number, row in enumerate(reader, start=2):
            actual_count = len(row)
            if actual_count != expected_count:
                errors.append(
                    "CSV row has inconsistent field count: "
                    f"line {line_number}, expected {expected_count}, actual {actual_count}."
                )
    return errors


def _candidate_columns(
    dataframe: pd.DataFrame,
    config: PCAConfig,
    errors: list[str],
) -> tuple[list[str], list[str]]:
    service_columns = set(config.service_columns or [])
    excluded = service_columns | set(config.exclude_columns or [])

    if config.include_columns is not None:
        missing_include_columns = [column for column in config.include_columns if column not in dataframe.columns]
        if missing_include_columns:
            errors.append(f"include_columns contains missing column(s): {', '.join(missing_include_columns)}")
        candidate_columns = [
            column for column in config.include_columns if column in dataframe.columns and column not in excluded
        ]
    else:
        candidate_columns = [column for column in dataframe.columns if column not in excluded]

    pca_excluded = set(excluded)
    if config.exclude_target_from_pca and config.target_feature:
        pca_excluded.add(config.target_feature)
    pca_candidate_columns = [column for column in candidate_columns if column not in pca_excluded]

    for extra_column in [config.target_feature, config.color_feature]:
        if extra_column and extra_column in dataframe.columns and extra_column not in service_columns:
            if extra_column not in candidate_columns and extra_column not in set(config.exclude_columns or []):
                candidate_columns.append(extra_column)

    return candidate_columns, pca_candidate_columns


def _profile_candidate_columns(
    dataframe: pd.DataFrame,
    candidate_columns: list[str],
    include_columns: list[str] | None,
) -> tuple[list[str], list[str], dict[str, int], dict[str, int]]:
    numeric_candidate_columns: list[str] = []
    non_numeric_excluded_columns: list[str] = []
    invalid_numeric_counts: dict[str, int] = {}
    missing_numeric_counts: dict[str, int] = {}
    include_set = set(include_columns or [])

    for column in candidate_columns:
        series = dataframe[column]
        non_empty_mask = _non_empty_mask(series)
        converted = pd.to_numeric(series, errors="coerce")
        invalid_mask = non_empty_mask & converted.isna()
        valid_count = int(converted.notna().sum())
        invalid_count = int(invalid_mask.sum())
        missing_count = int((~non_empty_mask).sum())

        if valid_count == 0 and invalid_count > 0 and column not in include_set:
            non_numeric_excluded_columns.append(column)
            continue

        numeric_candidate_columns.append(column)
        if invalid_count:
            invalid_numeric_counts[column] = invalid_count
        if missing_count:
            missing_numeric_counts[column] = missing_count

    return numeric_candidate_columns, non_numeric_excluded_columns, invalid_numeric_counts, missing_numeric_counts


def _non_empty_mask(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        stripped = series.astype("string").str.strip()
        return series.notna() & stripped.ne("")
    return series.notna()


def write_data_check_report(output_dir: str | Path, result: ValidationResult) -> None:
    lines = [
        "Data Check Report",
        "=================",
        f"Objects/rows: {result.row_count}",
        f"Columns: {result.column_count}",
        f"Missing values: {result.missing_value_count}",
        f"Numeric columns ({len(result.numeric_columns)}): {', '.join(result.numeric_columns) or 'none'}",
        f"Non-numeric columns ({len(result.non_numeric_columns)}): {', '.join(result.non_numeric_columns) or 'none'}",
        (
            f"Numeric candidate columns ({len(result.numeric_candidate_columns)}): "
            f"{', '.join(result.numeric_candidate_columns) or 'none'}"
        ),
        (
            f"Non-numeric excluded candidate columns ({len(result.non_numeric_excluded_columns)}): "
            f"{', '.join(result.non_numeric_excluded_columns) or 'none'}"
        ),
        (
            "Invalid numeric values per candidate column: "
            f"{_format_counts(result.invalid_numeric_counts)}"
        ),
        (
            "Missing numeric values per candidate column: "
            f"{_format_counts(result.missing_numeric_counts)}"
        ),
        f"Fully empty columns ({len(result.empty_columns)}): {', '.join(result.empty_columns) or 'none'}",
        (
            f"Zero-variance numeric columns ({len(result.zero_variance_numeric_columns)}): "
            f"{', '.join(result.zero_variance_numeric_columns) or 'none'}"
        ),
        "",
        "Errors:",
    ]
    lines.extend([f"- {error}" for error in result.errors] or ["- none"])
    lines.append("")
    lines.append("Warnings:")
    lines.extend([f"- {warning}" for warning in result.warnings] or ["- none"])
    lines.append("")
    lines.append(f"Conclusion: {'PCA preprocessing can continue.' if result.can_continue else 'Critical errors found. Stop.'}")

    write_text_report(Path(output_dir) / "Data_Check_Report.txt", "\n".join(lines) + "\n")


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{column}={count}" for column, count in sorted(counts.items()))
