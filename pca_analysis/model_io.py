"""PCA metadata and model persistence."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import joblib

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ironcellquant_matplotlib"))

import matplotlib
import numpy as np
import pandas as pd
import sklearn

from .config import PCAConfig
from .preprocessing import PreprocessingResult
from .validation import ValidationResult

if TYPE_CHECKING:
    from .analysis import PCAAnalysisResult


def write_run_metadata(
    input_path: str | Path,
    output_dir: Path,
    config: PCAConfig,
    validation_result: ValidationResult,
    preprocessing_result: PreprocessingResult,
    result: "PCAAnalysisResult",
) -> None:
    scatter_files = [filename for filename in result.generated_files if filename.startswith("PCA_Scatter_")]
    correlation_features: list[str] = []
    if result.correlations is not None and "Feature" in result.correlations.columns:
        correlation_features = result.correlations["Feature"].tolist()

    metadata = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_path": str(Path(input_path)),
        "output_path": str(output_dir),
        "config": asdict(config),
        "config_validation": {
            "status": "valid",
            "warnings": [],
        },
        "selected_pca_features": result.selected_features,
        "raw_numeric_features": preprocessing_result.raw_numeric_feature_columns,
        "raw_numeric_features_used_for_correlation": correlation_features,
        "raw_target_availability": {
            "feature": config.target_feature,
            "available": preprocessing_result.raw_target_available,
            "status": preprocessing_result.raw_target_status,
        },
        "raw_color_availability": {
            "feature": config.color_feature,
            "available": preprocessing_result.raw_color_available,
            "status": preprocessing_result.raw_color_status,
        },
        "invalid_numeric_counts": preprocessing_result.invalid_numeric_counts,
        "missing_numeric_counts": preprocessing_result.missing_numeric_counts,
        "removed_features_and_reasons": preprocessing_result.removed_features_reasons,
        "actual_scatter_filenames": scatter_files,
        "delivery_export": result.delivery_export,
        "warnings": validation_result.warnings + preprocessing_result.warnings + result.warnings,
        "generated_files": _unique_filenames(result.generated_files + ["PCA_Run_Metadata.json", "PCA_Model.joblib"]),
        "library_versions": {
            "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": _python_version(),
            "scikit_learn": sklearn.__version__,
        },
    }
    path = output_dir / "PCA_Run_Metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if "PCA_Run_Metadata.json" not in result.generated_files:
        result.generated_files.append("PCA_Run_Metadata.json")


def write_pca_model(
    output_dir: Path,
    config: PCAConfig,
    preprocessing_result: PreprocessingResult,
    result: "PCAAnalysisResult",
) -> None:
    model_payload = {
        "scaler": preprocessing_result.scaler,
        "pca_model": result.pca_model,
        "selected_feature_names": result.selected_features,
        "raw_numeric_feature_names": preprocessing_result.raw_numeric_feature_columns,
        "service_columns": config.service_columns or [],
        "config": asdict(config),
    }
    joblib.dump(model_payload, output_dir / "PCA_Model.joblib")
    if "PCA_Model.joblib" not in result.generated_files:
        result.generated_files.append("PCA_Model.joblib")


def _python_version() -> str:
    import sys

    return sys.version


def _unique_filenames(filenames: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for filename in filenames:
        if filename in seen:
            continue
        seen.add(filename)
        unique.append(filename)
    return unique
