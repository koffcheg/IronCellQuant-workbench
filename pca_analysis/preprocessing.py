"""Feature selection, numeric coercion, missing value handling, and standardization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import PCAConfig
from .data_io import write_text_report


@dataclass
class PreprocessingResult:
    standardized_features: pd.DataFrame | None = None
    raw_numeric_features: pd.DataFrame | None = None
    scaler: StandardScaler | None = None
    feature_columns_before: list[str] = field(default_factory=list)
    feature_columns_after: list[str] = field(default_factory=list)
    raw_numeric_feature_columns: list[str] = field(default_factory=list)
    non_numeric_excluded_columns: list[str] = field(default_factory=list)
    excluded_columns: list[str] = field(default_factory=list)
    removed_features_reasons: dict[str, str] = field(default_factory=dict)
    invalid_numeric_counts: dict[str, int] = field(default_factory=dict)
    missing_numeric_counts: dict[str, int] = field(default_factory=dict)
    removed_rows: int = 0
    raw_target_available: bool = False
    raw_target_status: str = "not checked"
    raw_color_available: bool = False
    raw_color_status: str = "not checked"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return not self.errors


def preprocess_features(dataframe: pd.DataFrame, output_dir: str | Path, config: PCAConfig) -> PreprocessingResult:
    result = PreprocessingResult()
    output_path = Path(output_dir)
    service_columns = config.service_columns or []

    missing_service_columns = [column for column in service_columns if column not in dataframe.columns]
    if missing_service_columns:
        result.errors.append(f"Missing service column(s) during preprocessing: {', '.join(missing_service_columns)}")
        write_preprocessing_report(output_path, result, config)
        return result

    explicit_excluded = set(config.exclude_columns or [])
    excluded = set(service_columns) | explicit_excluded
    for column in sorted(excluded):
        result.removed_features_reasons[column] = "service/excluded column"

    if config.include_columns is not None:
        missing_include_columns = [column for column in config.include_columns if column not in dataframe.columns]
        if missing_include_columns:
            result.errors.append(f"include_columns contains missing column(s): {', '.join(missing_include_columns)}")
            write_preprocessing_report(output_path, result, config)
            return result
        candidate_columns = [
            column for column in config.include_columns if column in dataframe.columns and column not in excluded
        ]
    else:
        candidate_columns = [column for column in dataframe.columns if column not in excluded]

    analysis_columns = candidate_columns.copy()
    for extra_column in [config.target_feature, config.color_feature]:
        if extra_column and extra_column in dataframe.columns and extra_column not in excluded:
            if extra_column not in analysis_columns:
                analysis_columns.append(extra_column)

    raw_numeric_data, numeric_candidate_columns = _coerce_numeric_candidates(
        dataframe,
        analysis_columns,
        set(config.include_columns or []),
        result,
    )

    pca_candidate_columns = candidate_columns.copy()
    if config.exclude_target_from_pca and config.target_feature in pca_candidate_columns:
        pca_candidate_columns.remove(config.target_feature)
        result.removed_features_reasons[config.target_feature] = "excluded from PCA by exclude_target_from_pca"

    pca_feature_columns = [column for column in pca_candidate_columns if column in numeric_candidate_columns]
    result.feature_columns_before = pca_feature_columns.copy()
    result.raw_numeric_feature_columns = numeric_candidate_columns.copy()
    result.excluded_columns = sorted(excluded) + result.non_numeric_excluded_columns.copy()

    features = raw_numeric_data[pca_feature_columns].copy()
    identifiers = dataframe[service_columns].copy()

    empty_features = [
        column
        for column in features.columns
        if features[column].isna().all() and column not in result.invalid_numeric_counts
    ]
    if empty_features:
        features = features.drop(columns=empty_features)
        for column in empty_features:
            result.excluded_columns.append(column)
            result.removed_features_reasons[column] = "fully empty feature"
        result.warnings.append(f"Removed fully empty feature(s): {', '.join(empty_features)}")

    strategy = config.missing_value_strategy
    missing_count = int(features.isna().sum().sum())
    if missing_count:
        if strategy == "error":
            result.errors.append(f"Invalid or missing numeric feature values found: {missing_count}")
            write_preprocessing_report(output_path, result, config)
            return result
        if strategy == "remove_row":
            before_rows = len(features)
            keep_mask = ~features.isna().any(axis=1)
            features = features.loc[keep_mask].copy()
            identifiers = identifiers.loc[keep_mask].copy()
            raw_numeric_data = raw_numeric_data.loc[keep_mask].copy()
            result.removed_rows = before_rows - len(features)
        elif strategy == "remove_feature":
            columns_with_missing = features.columns[features.isna().any(axis=0)].tolist()
            features = features.drop(columns=columns_with_missing)
            for column in columns_with_missing:
                result.excluded_columns.append(column)
                result.removed_features_reasons[column] = "invalid or missing numeric values"
            result.warnings.append(
                f"Removed feature(s) with invalid or missing numeric values: {', '.join(columns_with_missing)}"
            )
        elif strategy == "replace_mean":
            means = features.mean(numeric_only=True)
            features = features.fillna(means)
            for column in features.columns:
                if column in raw_numeric_data.columns:
                    raw_numeric_data[column] = raw_numeric_data[column].fillna(means[column])

    if config.remove_zero_variance_features:
        zero_variance_features = [column for column in features.columns if features[column].nunique(dropna=True) <= 1]
        if zero_variance_features:
            features = features.drop(columns=zero_variance_features)
            for column in zero_variance_features:
                result.excluded_columns.append(column)
                result.removed_features_reasons[column] = "zero variance"
            result.warnings.append(f"Removed zero-variance feature(s): {', '.join(zero_variance_features)}")

    if len(identifiers) < 2:
        result.errors.append("At least 2 rows/objects are required after preprocessing.")
    if len(features.columns) < 2:
        result.errors.append("At least 2 numeric feature columns are required after preprocessing.")
    if result.errors:
        write_preprocessing_report(output_path, result, config)
        return result

    if config.standardization_enabled:
        scaler = StandardScaler()
        transformed = scaler.fit_transform(features)
        transformed_features = pd.DataFrame(transformed, columns=features.columns, index=features.index)
        result.scaler = scaler
    else:
        transformed_features = features.copy()

    result.feature_columns_after = transformed_features.columns.tolist()
    result.raw_numeric_features = pd.concat(
        [identifiers.reset_index(drop=True), raw_numeric_data.loc[features.index].reset_index(drop=True)],
        axis=1,
    )
    _update_raw_availability(result, config)

    standardized_features = pd.concat(
        [identifiers.reset_index(drop=True), transformed_features.reset_index(drop=True)],
        axis=1,
    )
    result.standardized_features = standardized_features
    standardized_features.to_csv(output_path / "Standardized_Features.csv", index=False)
    write_preprocessing_report(output_path, result, config)
    return result


def _coerce_numeric_candidates(
    dataframe: pd.DataFrame,
    candidate_columns: list[str],
    include_columns: set[str],
    result: PreprocessingResult,
) -> tuple[pd.DataFrame, list[str]]:
    raw_numeric_data = pd.DataFrame(index=dataframe.index)
    numeric_candidate_columns: list[str] = []

    for column in candidate_columns:
        series = dataframe[column]
        non_empty_mask = _non_empty_mask(series)
        converted = pd.to_numeric(series, errors="coerce")
        invalid_mask = non_empty_mask & converted.isna()
        valid_count = int(converted.notna().sum())
        invalid_count = int(invalid_mask.sum())
        missing_count = int((~non_empty_mask).sum())

        if valid_count == 0 and invalid_count > 0 and column not in include_columns:
            result.non_numeric_excluded_columns.append(column)
            result.removed_features_reasons[column] = "non-numeric column"
            continue

        raw_numeric_data[column] = converted
        numeric_candidate_columns.append(column)
        if invalid_count:
            result.invalid_numeric_counts[column] = invalid_count
        if missing_count:
            result.missing_numeric_counts[column] = missing_count

    return raw_numeric_data, numeric_candidate_columns


def _non_empty_mask(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        stripped = series.astype("string").str.strip()
        return series.notna() & stripped.ne("")
    return series.notna()


def _update_raw_availability(result: PreprocessingResult, config: PCAConfig) -> None:
    raw = result.raw_numeric_features
    if raw is None:
        return

    target = config.target_feature
    if target and target in raw.columns and raw[target].notna().any():
        result.raw_target_available = True
        result.raw_target_status = "available"
    elif target:
        result.raw_target_status = "missing or non-numeric"

    color = config.color_feature
    if color and color in raw.columns and raw[color].notna().all():
        result.raw_color_available = True
        result.raw_color_status = "available"
    elif color and color in raw.columns and raw[color].notna().any():
        result.raw_color_status = "available with missing values"
    elif color:
        result.raw_color_status = "missing or non-numeric"


def write_preprocessing_report(output_dir: str | Path, result: PreprocessingResult, config: PCAConfig) -> None:
    lines = [
        "Feature Preprocessing Report",
        "============================",
        f"Objects after preprocessing: {0 if result.standardized_features is None else len(result.standardized_features)}",
        f"Numeric candidate columns ({len(result.raw_numeric_feature_columns)}): "
        f"{', '.join(result.raw_numeric_feature_columns) or 'none'}",
        f"Features before preprocessing: {len(result.feature_columns_before)}",
        f"Features after preprocessing: {len(result.feature_columns_after)}",
        f"Removed rows: {result.removed_rows}",
        f"Missing value strategy: {config.missing_value_strategy}",
        f"Standardization enabled: {config.standardization_enabled}",
        f"Zero-variance removal enabled: {config.remove_zero_variance_features}",
        f"Invalid numeric values per column: {_format_counts(result.invalid_numeric_counts)}",
        f"Missing numeric values per column: {_format_counts(result.missing_numeric_counts)}",
        f"Non-numeric excluded columns ({len(result.non_numeric_excluded_columns)}): "
        f"{', '.join(result.non_numeric_excluded_columns) or 'none'}",
        f"Raw target status: {result.raw_target_status}",
        f"Raw color status: {result.raw_color_status}",
        f"Excluded/removed columns ({len(result.excluded_columns)}): {', '.join(result.excluded_columns) or 'none'}",
        "",
        "Removed features and reasons:",
    ]
    if result.removed_features_reasons:
        lines.extend(
            f"- {feature}: {reason}" for feature, reason in sorted(result.removed_features_reasons.items())
        )
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Errors:")
    lines.extend([f"- {error}" for error in result.errors] or ["- none"])
    lines.append("")
    lines.append("Warnings:")
    lines.extend([f"- {warning}" for warning in result.warnings] or ["- none"])
    lines.append("")
    lines.append(f"Conclusion: {'Preprocessing completed.' if result.succeeded else 'Preprocessing stopped.'}")

    write_text_report(Path(output_dir) / "Feature_Preprocessing_Report.txt", "\n".join(lines) + "\n")


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{column}={count}" for column, count in sorted(counts.items()))
