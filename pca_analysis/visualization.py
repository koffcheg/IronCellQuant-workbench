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

    generated.append(_write_explained_variance(output_path, config, summary))

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
        generated.extend(_write_biplots(output_path, config, summary, scores, loadings))
    else:
        warnings.append("Biplot was skipped because PC1 and PC2 are not both available.")

    generated.extend(_write_loading_bar_plots(output_path, config, loadings, component_names, warnings))

    return generated


def _components_available(components: list[str], component_names: list[str]) -> bool:
    return all(component in component_names for component in components)


def _write_explained_variance(output_dir: Path, config: PCAConfig, summary: pd.DataFrame) -> str:
    filename = "PCA_ExplainedVariance.png"
    path = output_dir / filename

    max_components = max(1, int(config.explained_variance_plot_max_components))
    total_components = len(summary)
    visible_summary = summary.head(max_components)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    x = np.arange(len(visible_summary)) + 1
    labels = visible_summary["PC"].tolist()
    explained = visible_summary["ExplainedVariancePercent"].to_numpy(dtype=float)
    cumulative = visible_summary["CumulativePercent"].to_numpy(dtype=float)

    ax.bar(x, explained, color="#4c78a8", label="Explained variance")
    ax.plot(x, cumulative, color="#f58518", marker="o", linewidth=2, label="Cumulative variance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    if total_components > max_components:
        ax.set_title(f"PCA Explained Variance - Showing first {max_components} of {total_components} components")
    else:
        ax.set_title("PCA Explained Variance")
    ax.set_ylim(0, max(100.0, float(cumulative.max(initial=0.0)) * 1.05))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
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


def _write_biplots(
    output_dir: Path,
    config: PCAConfig,
    summary: pd.DataFrame,
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
) -> list[str]:
    generated = [
        _write_biplot(
            output_dir,
            config,
            summary,
            scores,
            loadings,
            "PCA_Biplot_PC1_PC2.png",
            config.biplot_label_mode,
        )
    ]
    if config.biplot_save_direct_label_debug_plot and config.biplot_label_mode != "direct_labels":
        generated.append(
            _write_biplot(
                output_dir,
                config,
                summary,
                scores,
                loadings,
                "PCA_Biplot_PC1_PC2_labeled.png",
                "direct_labels",
            )
        )
    elif config.biplot_save_direct_label_debug_plot:
        generated.append("PCA_Biplot_PC1_PC2_labeled.png")
        (output_dir / "PCA_Biplot_PC1_PC2_labeled.png").write_bytes(
            (output_dir / "PCA_Biplot_PC1_PC2.png").read_bytes()
        )
    return generated


def _write_biplot(
    output_dir: Path,
    config: PCAConfig,
    summary: pd.DataFrame,
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    filename: str,
    label_mode: str,
) -> str:
    path = output_dir / filename
    selected_loadings = _select_biplot_loadings(loadings, config.biplot_top_feature_count)

    figsize = (13, 8) if label_mode == "numbered_legend" else (11, 8)
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.scatter(scores["PC1"], scores["PC2"], s=24, alpha=0.34, color="#4c78a8", edgecolor="none")
    ax.axhline(0, color="#666666", linewidth=0.8, alpha=0.45)
    ax.axvline(0, color="#666666", linewidth=0.8, alpha=0.45)

    score_scale = _score_scale(scores["PC1"], scores["PC2"])
    loading_scale = _loading_scale(selected_loadings["PC1"], selected_loadings["PC2"])
    arrow_scale = 0.75 * score_scale / loading_scale if loading_scale > 0 else 1.0

    legend_lines: list[str] = []
    for index, row in enumerate(selected_loadings.itertuples(index=False), start=1):
        x = float(row.PC1) * arrow_scale
        y = float(row.PC2) * arrow_scale
        ax.arrow(
            0,
            0,
            x,
            y,
            color="#d64f4f",
            alpha=0.9,
            length_includes_head=True,
            head_width=max(score_scale * 0.03, 0.05),
            linewidth=1.15,
        )
        if label_mode == "numbered_legend":
            ax.text(
                x * 1.07,
                y * 1.07,
                str(index),
                color="#9d2d2d",
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#d64f4f", "alpha": 0.82},
            )
            legend_lines.append(_biplot_legend_line(index, row, config.biplot_legend_include_loadings))
        elif label_mode == "direct_labels":
            ax.text(x * 1.07, y * 1.07, str(row.Feature), color="#9d2d2d", fontsize=7)

    ax.set_xlabel(_axis_label("PC1", summary))
    ax.set_ylabel(_axis_label("PC2", summary))
    ax.set_title("PCA Biplot: PC1 vs PC2")
    ax.grid(alpha=0.22)
    if label_mode == "numbered_legend":
        fig.subplots_adjust(right=0.58)
        fig.text(
            0.61,
            0.92,
            "Feature loadings",
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="top",
        )
        fig.text(
            0.61,
            0.88,
            "\n".join(legend_lines) if legend_lines else "No loading arrows selected.",
            fontsize=8,
            ha="left",
            va="top",
            family="monospace",
            linespacing=1.35,
        )
    else:
        fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return filename


def _write_loading_bar_plots(
    output_dir: Path,
    config: PCAConfig,
    loadings: pd.DataFrame,
    component_names: list[str],
    warnings: list[str],
) -> list[str]:
    generated: list[str] = []
    for component in ["PC1", "PC2", "PC3"]:
        if component not in component_names:
            warnings.append(f"Loading bar plot was skipped because {component} is unavailable.")
            continue
        generated.append(_write_loading_bar_plot(output_dir, config, loadings, component))
    return generated


def _write_loading_bar_plot(
    output_dir: Path,
    config: PCAConfig,
    loadings: pd.DataFrame,
    component: str,
) -> str:
    filename = f"PCA_Loadings_{component}.png"
    path = output_dir / filename
    limit = max(0, int(config.top_feature_count))
    ranked = loadings[["Feature", component]].copy()
    ranked["AbsLoading"] = ranked[component].abs()
    ranked = ranked.sort_values("AbsLoading", ascending=False).head(limit).sort_values("AbsLoading")

    height = max(4.5, 0.34 * max(len(ranked), 1) + 1.8)
    fig, ax = plt.subplots(figsize=(10, height), dpi=150)
    colors = np.where(ranked[component] >= 0, "#4c78a8", "#e45756")
    ax.barh(ranked["Feature"], ranked[component], color=colors, alpha=0.88)
    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.set_xlabel("Loading")
    ax.set_title(f"Top loadings for {component}")
    ax.grid(axis="x", alpha=0.22)
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


def _biplot_legend_line(index: int, row: tuple, include_loadings: bool) -> str:
    feature = str(row.Feature)
    if include_loadings:
        return f"{index:>2} = {feature} (PC1={float(row.PC1):.3f}, PC2={float(row.PC2):.3f})"
    return f"{index:>2} = {feature}"


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
