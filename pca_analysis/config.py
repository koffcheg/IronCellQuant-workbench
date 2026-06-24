"""Configuration for the independent PCA preprocessing workflow."""

from __future__ import annotations

from dataclasses import dataclass, fields
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
    biplot_top_feature_count: int = 15
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
