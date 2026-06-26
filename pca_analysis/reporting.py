"""Final PCA text report generation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .config import PCAConfig
from .data_io import write_text_report
from .preprocessing import PreprocessingResult
from .validation import ValidationResult

if TYPE_CHECKING:
    from .analysis import PCAAnalysisResult


def write_pca_report(
    input_path: str | Path,
    output_dir: Path,
    config: PCAConfig,
    validation_result: ValidationResult,
    preprocessing_result: PreprocessingResult,
    result: "PCAAnalysisResult",
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
        f"Numeric candidate columns: {len(preprocessing_result.raw_numeric_feature_columns)}",
        f"Numeric PCA features before preprocessing: {len(preprocessing_result.feature_columns_before)}",
        f"Features after preprocessing: {len(preprocessing_result.feature_columns_after)}",
        f"Raw target status: {preprocessing_result.raw_target_status}",
        f"Raw color status: {preprocessing_result.raw_color_status}",
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
            "Loadings method: sklearn PCA components_.T.",
        ]
    )

    for pc_index in range(1, 4):
        component = f"PC{pc_index}"
        row = summary[summary["PC"] == component]
        if not row.empty:
            lines.append(f"{component} explained variance: {float(row['ExplainedVariancePercent'].iloc[0]):.6f}%")

    if not summary.empty:
        lines.append(f"Cumulative explained variance: {float(summary['CumulativePercent'].iloc[-1]):.6f}%")
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
    delivery = result.delivery_export
    lines.append(f"Delivery named copies: {'enabled' if delivery.get('enabled') else 'disabled'}")
    if delivery.get("enabled"):
        lines.append(f"Delivery export path: {delivery.get('directory')}")
        lines.append(f"Delivery filename mode: {delivery.get('filename_mode')}")
    lines.append("")
    lines.append("Generated files:")
    generated_files = _unique_filenames(result.generated_files + ["PCA_Report.txt", "PCA_Run_Metadata.json"])
    lines.extend([f"- {filename}" for filename in generated_files] or ["- none"])
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
    if "PCA_Report.txt" not in result.generated_files:
        result.generated_files.append("PCA_Report.txt")


def _unique_filenames(filenames: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for filename in filenames:
        if filename in seen:
            continue
        seen.add(filename)
        unique.append(filename)
    return unique
