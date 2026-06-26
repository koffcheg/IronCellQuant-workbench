"""Configuration for the independent PCA preprocessing workflow."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
from pathlib import Path
from typing import Any


@dataclass
class PCAConfig:
    delimiter: str = ","
    missing_value_strategy: str = "error"
    remove_zero_variance_features: bool = True
    standardization_enabled: bool = True
    pca_component_count: int = 0
    min_explained_variance: float = 0.95
    top_feature_count: int = 10
    correlation_method: str = "pearson"
    target_feature: str = "blue_pixel_percent"
    color_feature: str = "blue_pixel_percent"
    scatter_component_x: str = "PC1"
    scatter_component_y: str = "PC2"
    explained_variance_plot_max_components: int = 15
    biplot_label_mode: str = "numbered_legend"
    biplot_top_feature_count: int = 15
    biplot_legend_include_loadings: bool = True
    biplot_save_direct_label_debug_plot: bool = False
    allowed_object_types: list[str] | None = field(
        default_factory=lambda: [
            "single_cell",
            "aggregate",
            "all_cell_material",
            "all_cells",
            "frame",
            "frame_level",
        ]
    )
    delivery_export_enabled: bool = False
    delivery_export_dir: str | None = None
    delivery_filename_mode: str = "prefix_input_name"
    service_columns: list[str] | None = None
    exclude_columns: list[str] | None = None
    include_columns: list[str] | None = None
    exclude_target_from_pca: bool = False

    def __post_init__(self) -> None:
        if self.service_columns is None:
            self.service_columns = ["image_name", "object_type", "object_id"]
        if self.exclude_columns is None:
            self.exclude_columns = []


def load_config(config_path: str | Path | None = None) -> PCAConfig:
    """Load JSON config over defaults."""
    config = PCAConfig()
    if config_path is None:
        return config

    path = Path(config_path)
    with path.open("r", encoding="utf-8-sig") as handle:
        raw_config: dict[str, Any] = json.load(handle)

    allowed = {field.name for field in fields(PCAConfig)}
    unknown = sorted(set(raw_config) - allowed)
    if unknown:
        raise ValueError(f"Unknown config option(s): {', '.join(unknown)}")

    values = {field.name: getattr(config, field.name) for field in fields(PCAConfig)}
    values.update(raw_config)
    return PCAConfig(**values)


def validate_config(config: PCAConfig) -> list[str]:
    """Return human-readable configuration validation errors."""
    errors: list[str] = []

    if not isinstance(config.delimiter, str) or config.delimiter == "":
        errors.append("delimiter must be a non-empty string.")

    allowed_strategies = {"error", "remove_row", "remove_feature", "replace_mean"}
    if config.missing_value_strategy not in allowed_strategies:
        errors.append(
            "missing_value_strategy must be one of: "
            f"{', '.join(sorted(allowed_strategies))}."
        )

    allowed_correlations = {"pearson", "spearman"}
    if not isinstance(config.correlation_method, str) or config.correlation_method.lower() not in allowed_correlations:
        errors.append("correlation_method must be one of: pearson, spearman.")

    _validate_non_negative_int("pca_component_count", config.pca_component_count, errors)
    _validate_non_negative_int("top_feature_count", config.top_feature_count, errors)
    _validate_non_negative_int("biplot_top_feature_count", config.biplot_top_feature_count, errors)
    _validate_positive_int(
        "explained_variance_plot_max_components",
        config.explained_variance_plot_max_components,
        errors,
    )

    allowed_biplot_modes = {"numbered_legend", "direct_labels", "none"}
    if config.biplot_label_mode not in allowed_biplot_modes:
        errors.append(
            "biplot_label_mode must be one of: "
            f"{', '.join(sorted(allowed_biplot_modes))}."
        )

    allowed_delivery_modes = {"prefix_input_name", "standard"}
    if config.delivery_filename_mode not in allowed_delivery_modes:
        errors.append(
            "delivery_filename_mode must be one of: "
            f"{', '.join(sorted(allowed_delivery_modes))}."
        )

    if not isinstance(config.min_explained_variance, (int, float)) or isinstance(config.min_explained_variance, bool):
        errors.append("min_explained_variance must be numeric and > 0.")
    elif float(config.min_explained_variance) <= 0:
        errors.append("min_explained_variance must be > 0; values in 0..1 or 0..100 are accepted.")
    elif float(config.min_explained_variance) > 100:
        errors.append("min_explained_variance must be in 0..1 or 0..100.")

    _validate_string_list("service_columns", config.service_columns, errors, non_empty=True)
    _validate_string_list("exclude_columns", config.exclude_columns, errors, non_empty=False)
    if config.include_columns is not None:
        _validate_string_list("include_columns", config.include_columns, errors, non_empty=False)
    if config.allowed_object_types is not None:
        _validate_string_list("allowed_object_types", config.allowed_object_types, errors, non_empty=False)
    if config.delivery_export_dir is not None and not isinstance(config.delivery_export_dir, str):
        errors.append("delivery_export_dir must be a string or null.")

    _validate_bool("exclude_target_from_pca", config.exclude_target_from_pca, errors)
    _validate_bool("standardization_enabled", config.standardization_enabled, errors)
    _validate_bool("remove_zero_variance_features", config.remove_zero_variance_features, errors)
    _validate_bool("biplot_legend_include_loadings", config.biplot_legend_include_loadings, errors)
    _validate_bool("biplot_save_direct_label_debug_plot", config.biplot_save_direct_label_debug_plot, errors)
    _validate_bool("delivery_export_enabled", config.delivery_export_enabled, errors)

    for name in ["target_feature", "color_feature", "scatter_component_x", "scatter_component_y"]:
        value = getattr(config, name)
        if value is not None and not isinstance(value, str):
            errors.append(f"{name} must be a string or null.")

    return errors


def _validate_non_negative_int(name: str, value: Any, errors: list[str]) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        errors.append(f"{name} must be an integer >= 0.")


def _validate_positive_int(name: str, value: Any, errors: list[str]) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        errors.append(f"{name} must be an integer >= 1.")


def _validate_bool(name: str, value: Any, errors: list[str]) -> None:
    if not isinstance(value, bool):
        errors.append(f"{name} must be a boolean.")


def _validate_string_list(
    name: str,
    value: Any,
    errors: list[str],
    *,
    non_empty: bool,
) -> None:
    if not isinstance(value, list):
        errors.append(f"{name} must be a list of strings.")
        return
    if non_empty and not value:
        errors.append(f"{name} must be a non-empty list of strings.")
    invalid_items = [item for item in value if not isinstance(item, str)]
    if invalid_items:
        errors.append(f"{name} must contain only strings.")
