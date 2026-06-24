"""PCA calculation, correlation analysis, and result export."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import PCAConfig
from .model_io import write_pca_model, write_run_metadata
from .preprocessing import PreprocessingResult
from .reporting import write_pca_report
from .validation import ValidationResult
from .visualization import write_pca_visualizations


@dataclass
class PCAAnalysisResult:
    summary: pd.DataFrame | None = None
    scores: pd.DataFrame | None = None
    loadings: pd.DataFrame | None = None
    top_features: pd.DataFrame | None = None
    correlations: pd.DataFrame | None = None
    pca_model: PCA | None = None
    selected_features: list[str] = field(default_factory=list)
    component_names: list[str] = field(default_factory=list)
    recommended_component_count: int | None = None
    generated_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return not self.errors


def run_pca_analysis(
    input_path: str | Path,
    output_dir: str | Path,
    config: PCAConfig,
    validation_result: ValidationResult,
    preprocessing_result: PreprocessingResult,
) -> PCAAnalysisResult:
    """Run PCA using preprocessed features and write the required artifacts."""
    result = PCAAnalysisResult()
    output_path = Path(output_dir)
    preprocessed = preprocessing_result.standardized_features

    if preprocessed is None:
        result.errors.append("Preprocessed feature matrix is missing.")
        return result

    service_columns = config.service_columns or []
    missing_service_columns = [column for column in service_columns if column not in preprocessed.columns]
    if missing_service_columns:
        result.errors.append(
            f"Missing service column(s) in preprocessed matrix: {', '.join(missing_service_columns)}"
        )
        return result

    feature_columns = [column for column in preprocessing_result.feature_columns_after if column in preprocessed.columns]
    if not feature_columns:
        result.errors.append("No preprocessed numeric features available for PCA.")
        return result

    feature_matrix = preprocessed[feature_columns].copy()
    finite_mask = np.isfinite(feature_matrix.to_numpy(dtype=float))
    if not finite_mask.all():
        result.errors.append("Preprocessed feature matrix contains NaN or infinite values.")
        return result

    n_rows, n_features = feature_matrix.shape
    max_components = min(n_rows, n_features)
    requested_components = int(config.pca_component_count)
    if requested_components < 0:
        result.errors.append("pca_component_count must be >= 0.")
        return result
    if requested_components == 0:
        component_count = max_components
    else:
        component_count = min(requested_components, max_components)
        if requested_components > max_components:
            result.warnings.append(
                f"Requested {requested_components} PCA components, capped to {max_components}."
            )

    pca_model = PCA(n_components=component_count)
    score_values = pca_model.fit_transform(feature_matrix)
    component_names = [f"PC{index}" for index in range(1, component_count + 1)]

    if not np.isfinite(score_values).all():
        result.errors.append("PCA scores contain NaN or infinite values.")
        return result

    loadings_values = pca_model.components_.T
    if not np.isfinite(loadings_values).all():
        result.errors.append("PCA loadings contain NaN or infinite values.")
        return result

    explained_ratio_sum = float(np.sum(pca_model.explained_variance_ratio_))
    if component_count == max_components and not np.isclose(explained_ratio_sum, 1.0, rtol=1e-6, atol=1e-6):
        result.warnings.append(
            f"Explained variance ratio sum is {explained_ratio_sum:.10f}; expected close to 1.0 for all components."
        )

    cumulative_percent = np.cumsum(pca_model.explained_variance_ratio_) * 100.0
    summary = pd.DataFrame(
        {
            "PC": component_names,
            "Eigenvalue": pca_model.explained_variance_,
            "ExplainedVariance": pca_model.explained_variance_ratio_,
            "ExplainedVariancePercent": pca_model.explained_variance_ratio_ * 100.0,
            "CumulativePercent": cumulative_percent,
        }
    )

    scores = pd.concat(
        [
            preprocessed[service_columns].reset_index(drop=True),
            pd.DataFrame(score_values, columns=component_names),
        ],
        axis=1,
    )
    loadings = pd.concat(
        [
            pd.DataFrame({"Feature": feature_columns}),
            pd.DataFrame(loadings_values, columns=component_names),
        ],
        axis=1,
    )
    top_features = _calculate_top_features(loadings, component_names, config.top_feature_count)
    raw_numeric = preprocessing_result.raw_numeric_features
    correlations = _calculate_target_correlations(raw_numeric, config, service_columns, result.warnings)

    result.summary = summary
    result.scores = scores
    result.loadings = loadings
    result.top_features = top_features
    result.correlations = correlations
    result.pca_model = pca_model
    result.selected_features = feature_columns
    result.component_names = component_names
    result.recommended_component_count = _recommended_component_count(
        pca_model.explained_variance_ratio_,
        config.min_explained_variance,
    )

    _write_csv(output_path, "PCA_Summary.csv", summary, result)
    _write_csv(output_path, "PCA_Scores.csv", scores, result)
    _write_csv(output_path, "PCA_Loadings.csv", loadings, result)
    _write_csv(output_path, "PCA_TopFeatures.csv", top_features, result)
    _write_csv(output_path, "PCA_Correlation_With_BluePixel.csv", correlations, result)
    result.generated_files.extend(
        write_pca_visualizations(
            output_path,
            config,
            summary,
            scores,
            loadings,
            raw_numeric,
            component_names,
            result.warnings,
        )
    )
    write_pca_report(input_path, output_path, config, validation_result, preprocessing_result, result)
    write_run_metadata(input_path, output_path, config, validation_result, preprocessing_result, result)
    write_pca_model(output_path, config, preprocessing_result, result)

    return result


def _calculate_top_features(loadings: pd.DataFrame, component_names: list[str], top_count: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    limit = max(0, int(top_count))
    for component in component_names:
        ranked = loadings[["Feature", component]].copy()
        ranked["AbsLoading"] = ranked[component].abs()
        ranked = ranked.sort_values("AbsLoading", ascending=False).head(limit)
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            rows.append(
                {
                    "PC": component,
                    "Rank": rank,
                    "Feature": row.Feature,
                    "Loading": getattr(row, component),
                    "AbsLoading": row.AbsLoading,
                }
            )
    return pd.DataFrame(rows, columns=["PC", "Rank", "Feature", "Loading", "AbsLoading"])


def _calculate_target_correlations(
    dataframe: pd.DataFrame | None,
    config: PCAConfig,
    service_columns: list[str],
    warnings: list[str],
) -> pd.DataFrame:
    columns = ["Rank", "Feature", "CorrelationWithTarget", "AbsCorrelation", "Direction", "Strength"]
    target = config.target_feature
    if dataframe is None:
        warnings.append("Raw numeric feature matrix is absent; correlation analysis was skipped.")
        return pd.DataFrame(columns=columns)
    if not target or target not in dataframe.columns:
        warnings.append(f"Target feature '{target}' is absent; correlation analysis was skipped.")
        return pd.DataFrame(columns=columns)

    target_values = pd.to_numeric(dataframe[target], errors="coerce")
    if target_values.notna().sum() < 2:
        warnings.append(f"Target feature '{target}' is missing or non-numeric; correlation analysis was skipped.")
        return pd.DataFrame(columns=columns)

    method = config.correlation_method.lower()
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    feature_columns = [column for column in numeric_columns if column != target and column not in set(service_columns)]
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        feature_values = pd.to_numeric(dataframe[feature], errors="coerce")
        if feature_values.notna().sum() < 2:
            continue
        correlation = target_values.corr(feature_values, method=method)
        if pd.isna(correlation):
            continue
        rows.append(
            {
                "Feature": feature,
                "CorrelationWithTarget": float(correlation),
                "AbsCorrelation": abs(float(correlation)),
                "Direction": _direction(float(correlation)),
                "Strength": _strength(float(correlation)),
            }
        )

    rows.sort(key=lambda item: item["AbsCorrelation"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["Rank"] = rank
    return pd.DataFrame(rows, columns=columns)


def _direction(value: float) -> str:
    if np.isclose(value, 0.0, atol=1e-12):
        return "none"
    return "positive" if value > 0 else "negative"


def _strength(value: float) -> str:
    absolute = abs(value)
    if np.isclose(absolute, 0.0, atol=1e-12):
        return "very_low"
    if absolute < 0.2:
        return "very_low"
    if absolute < 0.4:
        return "low"
    if absolute < 0.6:
        return "moderate"
    if absolute < 0.8:
        return "high"
    return "very_high"


def _recommended_component_count(explained_variance_ratio: np.ndarray, minimum: float) -> int:
    threshold = float(minimum)
    if threshold <= 0:
        return 1
    if threshold > 1:
        threshold = threshold / 100.0
    cumulative = np.cumsum(explained_variance_ratio)
    index = np.searchsorted(cumulative, threshold, side="left")
    return int(min(index + 1, len(explained_variance_ratio)))


def _write_csv(output_dir: Path, filename: str, dataframe: pd.DataFrame, result: PCAAnalysisResult) -> None:
    path = output_dir / filename
    dataframe.to_csv(path, index=False)
    result.generated_files.append(filename)
