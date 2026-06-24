"""PCA visualization helpers."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ironcellquant_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PCAConfig


def write_pca_visualizations(
    output_dir: str | Path,
    config: PCAConfig,
    summary: pd.DataFrame,
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    raw_numeric_features: pd.DataFrame | None,
    component_names: list[str],
    warnings: list[str],
) -> list[str]:
    """Write PCA visualization PNG files and return generated filenames."""
    generated: list[str] = []
    output_path = Path(output_dir)

    generated.append(_write_explained_variance(output_path, summary))

    scatter_x = config.scatter_component_x or "PC1"
    scatter_y = config.scatter_component_y or "PC2"
    if _components_available([scatter_x, scatter_y], component_names):
        generated.append(
            _write_scatter(
                output_path,
                config,
                summary,
                scores,
                raw_numeric_features,
                scatter_x,
                scatter_y,
                warnings,
            )
        )
    else:
        warnings.append(
            "Scatter plot was skipped because the requested PCA components "
            f"are unavailable: {scatter_x}, {scatter_y}."
        )

    if _components_available(["PC1", "PC2"], component_names):
        generated.append(_write_biplot(output_path, config, summary, scores, loadings))
    else:
        warnings.append("Biplot was skipped because PC1 and PC2 are not both available.")

    return generated


def _components_available(components: list[str], component_names: list[str]) -> bool:
    return all(component in component_names for component in components)


def _write_explained_variance(output_dir: Path, summary: pd.DataFrame) -> str:
    filename = "PCA_ExplainedVariance.png"
    path = output_dir / filename

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    x = np.arange(len(summary)) + 1
    labels = summary["PC"].tolist()
    explained = summary["ExplainedVariancePercent"].to_numpy(dtype=float)
    cumulative = summary["CumulativePercent"].to_numpy(dtype=float)

    ax.bar(x, explained, color="#4c78a8", label="Explained variance")
    ax.plot(x, cumulative, color="#f58518", marker="o", linewidth=2, label="Cumulative variance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("PCA Explained Variance")
    ax.set_ylim(0, max(100.0, float(cumulative.max(initial=0.0)) * 1.05))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return filename


def _write_scatter(
    output_dir: Path,
    config: PCAConfig,
    summary: pd.DataFrame,
    scores: pd.DataFrame,
    raw_numeric_features: pd.DataFrame | None,
    component_x: str,
    component_y: str,
    warnings: list[str],
) -> str:
    filename = f"PCA_Scatter_{component_x}_{component_y}.png"
    path = output_dir / filename

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    color_values = _numeric_color_values(raw_numeric_features, config.color_feature)
    if color_values is None:
        if config.color_feature:
            warnings.append(
                f"Scatter color feature '{config.color_feature}' is unavailable as complete raw numeric data; "
                "scatter was written without color encoding."
            )
        ax.scatter(scores[component_x], scores[component_y], s=34, alpha=0.82, edgecolor="none")
    else:
        scatter = ax.scatter(
            scores[component_x],
            scores[component_y],
            c=color_values,
            cmap="viridis",
            s=34,
            alpha=0.86,
            edgecolor="none",
        )
        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label(f"{config.color_feature} (raw)")

    ax.axhline(0, color="#666666", linewidth=0.8, alpha=0.45)
    ax.axvline(0, color="#666666", linewidth=0.8, alpha=0.45)
    ax.set_xlabel(_axis_label(component_x, summary))
    ax.set_ylabel(_axis_label(component_y, summary))
    ax.set_title(f"PCA Scatter: {component_x} vs {component_y}")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return filename


def _write_biplot(
    output_dir: Path,
    config: PCAConfig,
    summary: pd.DataFrame,
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
) -> str:
    filename = "PCA_Biplot_PC1_PC2.png"
    path = output_dir / filename
    selected_loadings = _select_biplot_loadings(loadings, config.biplot_top_feature_count)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ax.scatter(scores["PC1"], scores["PC2"], s=28, alpha=0.58, color="#4c78a8", edgecolor="none")
    ax.axhline(0, color="#666666", linewidth=0.8, alpha=0.45)
    ax.axvline(0, color="#666666", linewidth=0.8, alpha=0.45)

    score_scale = _score_scale(scores["PC1"], scores["PC2"])
    loading_scale = _loading_scale(selected_loadings["PC1"], selected_loadings["PC2"])
    arrow_scale = 0.75 * score_scale / loading_scale if loading_scale > 0 else 1.0

    for row in selected_loadings.itertuples(index=False):
        x = float(row.PC1) * arrow_scale
        y = float(row.PC2) * arrow_scale
        ax.arrow(
            0,
            0,
            x,
            y,
            color="#e45756",
            alpha=0.85,
            length_includes_head=True,
            head_width=max(score_scale * 0.025, 0.05),
            linewidth=1.0,
        )
        ax.text(x * 1.06, y * 1.06, str(row.Feature), color="#b23b3b", fontsize=8)

    ax.set_xlabel(_axis_label("PC1", summary))
    ax.set_ylabel(_axis_label("PC2", summary))
    ax.set_title("PCA Biplot: PC1 vs PC2")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return filename


def _numeric_color_values(dataframe: pd.DataFrame | None, color_feature: str | None) -> pd.Series | None:
    if dataframe is None or not color_feature or color_feature not in dataframe.columns:
        return None
    values = pd.to_numeric(dataframe[color_feature], errors="coerce")
    if values.isna().any():
        return None
    return values


def _axis_label(component: str, summary: pd.DataFrame) -> str:
    row = summary[summary["PC"] == component]
    if row.empty:
        return component
    return f"{component} ({float(row['ExplainedVariancePercent'].iloc[0]):.2f}%)"


def _select_biplot_loadings(loadings: pd.DataFrame, top_count: int) -> pd.DataFrame:
    limit = max(0, int(top_count))
    ranked = loadings[["Feature", "PC1", "PC2"]].copy()
    ranked["MaxAbsLoading"] = ranked[["PC1", "PC2"]].abs().max(axis=1)
    return ranked.sort_values("MaxAbsLoading", ascending=False).head(limit)


def _score_scale(pc1: pd.Series, pc2: pd.Series) -> float:
    maximum = max(float(pc1.abs().max()), float(pc2.abs().max()), 1.0)
    return maximum


def _loading_scale(pc1: pd.Series, pc2: pd.Series) -> float:
    if pc1.empty or pc2.empty:
        return 1.0
    return max(float(pc1.abs().max()), float(pc2.abs().max()), 1.0e-12)
