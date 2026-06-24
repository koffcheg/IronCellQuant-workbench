"""PCA calculation, correlation analysis, and result export."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA

from .config import PCAConfig
from .data_io import write_text_report
from .preprocessing import PreprocessingResult
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
    correlations = _calculate_target_correlations(preprocessed, config, service_columns, result.warnings)

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
            preprocessed,
            component_names,
            result.warnings,
        )
    )
    _write_report(input_path, output_path, config, validation_result, preprocessing_result, result)
    _write_metadata(input_path, output_path, config, validation_result, preprocessing_result, result)
    _write_model(output_path, config, preprocessing_result, result)

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
    dataframe: pd.DataFrame,
    config: PCAConfig,
    service_columns: list[str],
    warnings: list[str],
) -> pd.DataFrame:
    columns = ["Rank", "Feature", "CorrelationWithTarget", "AbsCorrelation", "Direction", "Strength"]
    target = config.target_feature
    if not target or target not in dataframe.columns:
        warnings.append(f"Target feature '{target}' is absent; correlation analysis was skipped.")
        return pd.DataFrame(columns=columns)

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    if target not in numeric_columns:
        warnings.append(f"Target feature '{target}' is not numeric; correlation analysis was skipped.")
        return pd.DataFrame(columns=columns)

    method = config.correlation_method.lower()
    if method not in {"pearson", "spearman"}:
        warnings.append(
            f"Unsupported correlation_method '{config.correlation_method}'; using pearson instead."
        )
        method = "pearson"

    feature_columns = [
        column for column in numeric_columns if column != target and column not in set(service_columns)
    ]
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        correlation = dataframe[target].corr(dataframe[feature], method=method)
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


def _write_report(
    input_path: str | Path,
    output_dir: Path,
    config: PCAConfig,
    validation_result: ValidationResult,
    preprocessing_result: PreprocessingResult,
    result: PCAAnalysisResult,
) -> None:
    summary = result.summary if result.summary is not None else pd.DataFrame()
    top_features = result.top_features if result.top_features is not None else pd.DataFrame()
    correlations = result.correlations if result.correlations is not None else pd.DataFrame()

    lines = [
        "PCA Report",
        "==========",
        f"Input file: {Path(input_path)}",
        f"Total objects analysed: {0 if result.scores is None else len(result.scores)}",
        f"Original columns count: {validation_result.column_count}",
        f"Numeric features before preprocessing: {len(preprocessing_result.feature_columns_before)}",
        f"Features after preprocessing: {len(preprocessing_result.feature_columns_after)}",
        "Removed features and reasons:",
    ]
    if preprocessing_result.removed_features_reasons:
        for feature, reason in sorted(preprocessing_result.removed_features_reasons.items()):
            lines.append(f"- {feature}: {reason}")
    else:
        lines.append("- none")

    lines.extend(
        [
            f"Missing value strategy: {config.missing_value_strategy}",
            f"Standardization enabled: {config.standardization_enabled}",
            f"Number of PCA components: {len(result.component_names)}",
            "Loadings method: sklearn PCA components_.T (MVP implementation).",
        ]
    )

    for pc_index in range(1, 4):
        component = f"PC{pc_index}"
        row = summary[summary["PC"] == component]
        if not row.empty:
            lines.append(
                f"{component} explained variance: {float(row['ExplainedVariancePercent'].iloc[0]):.6f}%"
            )

    if not summary.empty:
        lines.append(
            f"Cumulative explained variance: {float(summary['CumulativePercent'].iloc[-1]):.6f}%"
        )
    if result.recommended_component_count is not None:
        lines.append(
            "Recommended number of components for "
            f"min_explained_variance={config.min_explained_variance}: {result.recommended_component_count}"
        )

    lines.append("")
    lines.append("Top features for PC1/PC2/PC3:")
    for component in ["PC1", "PC2", "PC3"]:
        component_rows = top_features[top_features["PC"] == component]
        if component_rows.empty:
            lines.append(f"{component}: not available")
        else:
            joined = ", ".join(
                f"{row.Feature} ({row.Loading:.6f})" for row in component_rows.itertuples(index=False)
            )
            lines.append(f"{component}: {joined}")

    lines.append("")
    lines.append(f"Features most correlated with {config.target_feature}:")
    if correlations.empty:
        lines.append("- none")
    else:
        for row in correlations.head(config.top_feature_count).itertuples(index=False):
            lines.append(
                f"- {row.Feature}: {row.CorrelationWithTarget:.6f} "
                f"({row.Direction}, {row.Strength})"
            )

    lines.append("")
    lines.append("Warnings:")
    all_warnings = validation_result.warnings + preprocessing_result.warnings + result.warnings
    lines.extend([f"- {warning}" for warning in all_warnings] or ["- none"])
    lines.append("")
    lines.append(
        "Conclusion: Results describe statistical associations and candidate features only; "
        "they are not evidence of actual iron concentration."
    )

    write_text_report(output_dir / "PCA_Report.txt", "\n".join(lines) + "\n")
    result.generated_files.append("PCA_Report.txt")


def _write_metadata(
    input_path: str | Path,
    output_dir: Path,
    config: PCAConfig,
    validation_result: ValidationResult,
    preprocessing_result: PreprocessingResult,
    result: PCAAnalysisResult,
) -> None:
    metadata = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_path": str(Path(input_path)),
        "output_path": str(output_dir),
        "config": asdict(config),
        "selected_features": result.selected_features,
        "removed_features": preprocessing_result.removed_features_reasons,
        "warnings": validation_result.warnings + preprocessing_result.warnings + result.warnings,
        "generated_files": result.generated_files + ["PCA_Run_Metadata.json", "PCA_Model.joblib"],
        "library_versions": {
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    path = output_dir / "PCA_Run_Metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    result.generated_files.append("PCA_Run_Metadata.json")


def _write_model(
    output_dir: Path,
    config: PCAConfig,
    preprocessing_result: PreprocessingResult,
    result: PCAAnalysisResult,
) -> None:
    model_payload = {
        "scaler": preprocessing_result.scaler,
        "pca_model": result.pca_model,
        "selected_feature_names": result.selected_features,
        "service_columns": config.service_columns or [],
        "config": asdict(config),
    }
    joblib.dump(model_payload, output_dir / "PCA_Model.joblib")
    result.generated_files.append("PCA_Model.joblib")
