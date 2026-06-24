"""Input FeatureMatrix validation and data check report generation."""

from __future__ import annotations

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

    excluded = set(config.service_columns or []) | set(config.exclude_columns or [])
    if config.exclude_target_from_pca and config.target_feature:
        excluded.add(config.target_feature)

    candidate_columns = [column for column in dataframe.columns if column not in excluded]
    if config.include_columns is not None:
        missing_include_columns = [column for column in config.include_columns if column not in dataframe.columns]
        if missing_include_columns:
            result.errors.append(f"include_columns contains missing column(s): {', '.join(missing_include_columns)}")
        candidate_columns = [column for column in config.include_columns if column in dataframe.columns and column not in excluded]

    numeric_feature_columns = [column for column in candidate_columns if column in result.numeric_columns]
    if result.row_count < 2:
        result.errors.append("At least 2 rows/objects are required.")
    if len(numeric_feature_columns) < 2:
        result.errors.append("At least 2 numeric feature columns are required after service/excluded columns are removed.")

    if result.empty_columns:
        result.warnings.append(f"Fully empty column(s): {', '.join(result.empty_columns)}")
    if result.zero_variance_numeric_columns:
        result.warnings.append(
            f"Zero-variance numeric column(s): {', '.join(result.zero_variance_numeric_columns)}"
        )

    write_data_check_report(output_dir, result)
    return result


def write_data_check_report(output_dir: str | Path, result: ValidationResult) -> None:
    lines = [
        "Data Check Report",
        "=================",
        f"Objects/rows: {result.row_count}",
        f"Columns: {result.column_count}",
        f"Missing values: {result.missing_value_count}",
        f"Numeric columns ({len(result.numeric_columns)}): {', '.join(result.numeric_columns) or 'none'}",
        f"Non-numeric columns ({len(result.non_numeric_columns)}): {', '.join(result.non_numeric_columns) or 'none'}",
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
