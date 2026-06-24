"""Feature selection, missing value handling, and standardization."""

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
    scaler: StandardScaler | None = None
    feature_columns_before: list[str] = field(default_factory=list)
    feature_columns_after: list[str] = field(default_factory=list)
    excluded_columns: list[str] = field(default_factory=list)
    removed_features_reasons: dict[str, str] = field(default_factory=dict)
    removed_rows: int = 0
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

    excluded = set(service_columns)
    excluded.update(config.exclude_columns or [])
    if config.exclude_target_from_pca and config.target_feature:
        excluded.add(config.target_feature)

    if config.include_columns is not None:
        missing_include_columns = [column for column in config.include_columns if column not in dataframe.columns]
        if missing_include_columns:
            result.errors.append(f"include_columns contains missing column(s): {', '.join(missing_include_columns)}")
            write_preprocessing_report(output_path, result, config)
            return result
        candidate_columns = [column for column in config.include_columns if column not in excluded]
    else:
        candidate_columns = [column for column in dataframe.columns if column not in excluded]

    numeric_features = dataframe[candidate_columns].select_dtypes(include="number").columns.tolist()
    non_numeric_excluded = [column for column in candidate_columns if column not in numeric_features]
    result.feature_columns_before = numeric_features.copy()
    result.excluded_columns = sorted(excluded) + non_numeric_excluded
    for column in sorted(excluded):
        result.removed_features_reasons[column] = "service/excluded column"
    for column in non_numeric_excluded:
        result.removed_features_reasons[column] = "non-numeric column"

    features = dataframe[numeric_features].copy()
    identifiers = dataframe[service_columns].copy()

    empty_features = [column for column in features.columns if features[column].isna().all()]
    if empty_features:
        features = features.drop(columns=empty_features)
        result.excluded_columns.extend(empty_features)
        for column in empty_features:
            result.removed_features_reasons[column] = "fully empty feature"
        result.warnings.append(f"Removed fully empty feature(s): {', '.join(empty_features)}")

    strategy = config.missing_value_strategy
    allowed_strategies = {"error", "remove_row", "remove_feature", "replace_mean"}
    if strategy not in allowed_strategies:
        result.errors.append(
            f"Unsupported missing_value_strategy '{strategy}'. Expected one of: {', '.join(sorted(allowed_strategies))}"
        )
        write_preprocessing_report(output_path, result, config)
        return result

    missing_count = int(features.isna().sum().sum())
    if missing_count:
        if strategy == "error":
            result.errors.append(f"Missing numeric feature values found: {missing_count}")
            write_preprocessing_report(output_path, result, config)
            return result
        if strategy == "remove_row":
            before_rows = len(features)
            keep_mask = ~features.isna().any(axis=1)
            features = features.loc[keep_mask].copy()
            identifiers = identifiers.loc[keep_mask].copy()
            result.removed_rows = before_rows - len(features)
        elif strategy == "remove_feature":
            columns_with_missing = features.columns[features.isna().any(axis=0)].tolist()
            features = features.drop(columns=columns_with_missing)
            result.excluded_columns.extend(columns_with_missing)
            for column in columns_with_missing:
                result.removed_features_reasons[column] = "missing values"
            result.warnings.append(f"Removed feature(s) with missing values: {', '.join(columns_with_missing)}")
        elif strategy == "replace_mean":
            features = features.fillna(features.mean(numeric_only=True))

    if config.remove_zero_variance_features:
        zero_variance_features = [column for column in features.columns if features[column].nunique(dropna=True) <= 1]
        if zero_variance_features:
            features = features.drop(columns=zero_variance_features)
            result.excluded_columns.extend(zero_variance_features)
            for column in zero_variance_features:
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
    standardized_features = pd.concat(
        [identifiers.reset_index(drop=True), transformed_features.reset_index(drop=True)],
        axis=1,
    )
    result.standardized_features = standardized_features
    standardized_features.to_csv(output_path / "Standardized_Features.csv", index=False)
    write_preprocessing_report(output_path, result, config)
    return result


def write_preprocessing_report(output_dir: str | Path, result: PreprocessingResult, config: PCAConfig) -> None:
    lines = [
        "Feature Preprocessing Report",
        "============================",
        f"Objects after preprocessing: {0 if result.standardized_features is None else len(result.standardized_features)}",
        f"Features before preprocessing: {len(result.feature_columns_before)}",
        f"Features after preprocessing: {len(result.feature_columns_after)}",
        f"Removed rows: {result.removed_rows}",
        f"Missing value strategy: {config.missing_value_strategy}",
        f"Standardization enabled: {config.standardization_enabled}",
        f"Zero-variance removal enabled: {config.remove_zero_variance_features}",
        f"Excluded/removed columns ({len(result.excluded_columns)}): {', '.join(result.excluded_columns) or 'none'}",
        "",
        "Errors:",
    ]
    lines.extend([f"- {error}" for error in result.errors] or ["- none"])
    lines.append("")
    lines.append("Warnings:")
    lines.extend([f"- {warning}" for warning in result.warnings] or ["- none"])
    lines.append("")
    lines.append(f"Conclusion: {'Preprocessing completed.' if result.succeeded else 'Preprocessing stopped.'}")

    write_text_report(Path(output_dir) / "Feature_Preprocessing_Report.txt", "\n".join(lines) + "\n")
