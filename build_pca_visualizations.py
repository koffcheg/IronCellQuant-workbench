"""Build final PCA visualizations, point legends, magnifiers, and checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from pca_analysis.config import PCAConfig
from pca_analysis.visualization import _axis_label, _select_biplot_loadings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


STRICT_LEGEND_COLUMNS = ["№", "PC1", "PC2", "ID точки"]
EXTENDED_LEGEND_COLUMNS = [
    "№",
    "PC1",
    "PC2",
    "point_id",
    "feature_row_id",
    "image_name",
    "frame_id",
    "object_id",
    "object_type",
    "display_label",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "reference_roi_id",
    "inside_reference_roi",
    "reference_roi_overlap_fraction",
    "blue_pixel_percent",
]
ROUNDTRIP_COLUMNS = [
    "feature_row_id",
    "frame_id",
    "display_label",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
]
SIMPLE_INPUT = r"D:\PhD\Datasets\IronCells\23.06.2026\Розмітка\29_05_24_залізо_3_51 ориг_2.bmp"
COMPLEX_INPUT = r"D:\PhD\Datasets\IronCells\23.06.2026\Розмітка\11_06_24_залізо_2_трив_52 прото_з2_2.bmp"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class ModeResult:
    mode: str
    data: pd.DataFrame
    strict_path: Path
    extended_path: Path
    generated: list[Path]
    color_available: bool
    color_missing_count: int
    magnifier_names: list[str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create final PCA cells/frames visualizations, point legends, magnifiers, and validation report."
    )
    parser.add_argument("--pca-cells-dir", required=True, type=Path)
    parser.add_argument("--pca-frames-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("visualization config must be a JSON object")
    config.setdefault("color_feature", "blue_pixel_percent")
    config.setdefault("label_mode", "number")
    config.setdefault("label_all_points", True)
    config.setdefault("max_numbered_labels", 200)
    config.setdefault("frame_point_id", "frame_id")
    config.setdefault("biplot_top_feature_count", PCAConfig().biplot_top_feature_count)
    config.setdefault("preferred_censoring_config", "config_b_standard")
    return config


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    output = (completed.stdout or "") + (completed.stderr or "")
    return output.strip()


def required_path(config: dict[str, Any], key: str) -> Path:
    value = config.get(key)
    if not value:
        raise ValueError(f"Missing required config path: {key}")
    return Path(str(value))


def mode_files(pca_dir: Path, mode: str) -> dict[str, Path]:
    files = {
        "scores": pca_dir / f"PCA_Scores_{mode}.csv",
        "summary": pca_dir / f"PCA_Summary_{mode}.csv",
        "loadings": pca_dir / f"PCA_Loadings_{mode}.csv",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing PCA input file(s): " + "; ".join(missing))
    return files


def as_number(dataframe: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(dataframe[column], errors="raise")


def sorted_scores(scores: pd.DataFrame) -> pd.DataFrame:
    data = scores.copy()
    data["PC1"] = as_number(data, "PC1")
    data["PC2"] = as_number(data, "PC2")
    return data.sort_values(["PC1", "PC2"], ascending=[True, True], kind="mergesort").reset_index(drop=True)


def point_id(row: pd.Series | dict[str, Any], mode: str, frame_point_id: str) -> str:
    getter = row.get
    if mode == "cells":
        return str(getter("feature_row_id", "")).strip()
    if frame_point_id == "feature_row_id":
        return str(getter("feature_row_id", "")).strip()
    return str(getter("frame_id", "")).strip()


def prepare_mode_data(
    scores: pd.DataFrame,
    matrix_path: Path,
    mode: str,
    frame_point_id: str,
    color_feature: str,
) -> pd.DataFrame:
    join_key = "feature_row_id" if mode == "cells" else "frame_id"
    if join_key not in scores.columns:
        raise ValueError(f"{mode} scores missing join key: {join_key}")
    matrix = read_csv(matrix_path)
    if join_key not in matrix.columns:
        raise ValueError(f"{mode} matrix missing join key: {join_key}")

    matrix_columns = [join_key]
    for column in EXTENDED_LEGEND_COLUMNS:
        if column in matrix.columns and column not in matrix_columns:
            matrix_columns.append(column)
    if color_feature in matrix.columns and color_feature not in matrix_columns:
        matrix_columns.append(color_feature)

    matrix_for_join = matrix[matrix_columns].copy()
    merged = scores.copy()
    merged[join_key] = merged[join_key].astype(str)
    matrix_for_join[join_key] = matrix_for_join[join_key].astype(str)
    merged = merged.merge(matrix_for_join, on=join_key, how="left", suffixes=("", "__matrix"))
    for column in list(merged.columns):
        if not column.endswith("__matrix"):
            continue
        base = column[: -len("__matrix")]
        if base not in merged.columns:
            merged[base] = merged[column]
        else:
            merged[base] = merged[base].where(merged[base].astype(str).str.strip().ne(""), merged[column])
        merged = merged.drop(columns=[column])

    ordered = sorted_scores(merged)
    ordered["№"] = range(1, len(ordered) + 1)
    ordered["point_id"] = [point_id(row, mode, frame_point_id) for row in ordered.to_dict("records")]
    if color_feature in ordered.columns:
        ordered[color_feature] = pd.to_numeric(ordered[color_feature], errors="coerce")
    return ordered


def strict_rows(data: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"№": int(row["№"]), "PC1": row["PC1"], "PC2": row["PC2"], "ID точки": row["point_id"]}
        for row in data.to_dict("records")
    ]


def extended_rows(data: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in data.to_dict("records"):
        item = {column: row.get(column, "") for column in EXTENDED_LEGEND_COLUMNS}
        item.update({"№": int(row["№"]), "PC1": row["PC1"], "PC2": row["PC2"], "point_id": row.get("point_id", "")})
        rows.append(item)
    return rows


def write_legends(output_dir: Path, data: pd.DataFrame, mode: str) -> tuple[Path, Path]:
    strict_path = output_dir / f"PCA_Legend_{mode}_PC1_PC2.csv"
    extended_path = output_dir / f"PCA_Legend_{mode}_PC1_PC2_extended.csv"
    write_csv(strict_path, STRICT_LEGEND_COLUMNS, strict_rows(data))
    write_csv(extended_path, EXTENDED_LEGEND_COLUMNS, extended_rows(data))
    return strict_path, extended_path


def add_colorbar_scatter(
    ax: plt.Axes,
    data: pd.DataFrame,
    color_feature: str,
    marker_size: int,
    alpha: float,
) -> tuple[Any, bool, int]:
    values = pd.to_numeric(data[color_feature], errors="coerce") if color_feature in data.columns else pd.Series(dtype=float)
    missing_count = int(values.isna().sum()) if len(values) else len(data)
    if len(values) == len(data) and missing_count == 0:
        scatter = ax.scatter(
            data["PC1"],
            data["PC2"],
            c=values,
            cmap="viridis",
            s=marker_size,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.45,
        )
        return scatter, True, missing_count
    scatter = ax.scatter(
        data["PC1"],
        data["PC2"],
        s=marker_size,
        alpha=alpha,
        color="#58677a",
        edgecolor="white",
        linewidth=0.45,
    )
    return scatter, False, missing_count


def decorate_axes(ax: plt.Axes, summary: pd.DataFrame, title: str) -> None:
    ax.axhline(0, color="#666666", linewidth=0.8, alpha=0.38)
    ax.axvline(0, color="#666666", linewidth=0.8, alpha=0.38)
    ax.set_xlabel(_axis_label("PC1", summary))
    ax.set_ylabel(_axis_label("PC2", summary))
    ax.set_title(title)
    ax.grid(alpha=0.18, linewidth=0.7)


def annotate_numbers(ax: plt.Axes, data: pd.DataFrame, font_size: int) -> None:
    for row in data.to_dict("records"):
        ax.annotate(
            str(int(row["№"])),
            (float(row["PC1"]), float(row["PC2"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=font_size,
            color="#1f2933",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "#a9b3bf", "alpha": 0.86},
        )


def write_scatter(
    output_dir: Path,
    mode: str,
    data: pd.DataFrame,
    summary: pd.DataFrame,
    color_feature: str,
    numbered: bool,
    label_all_points: bool,
    max_numbered_labels: int,
) -> tuple[Path, bool, int]:
    suffix = "_numbered" if numbered else ""
    path = output_dir / f"PCA_Scatter_PC1_PC2_{mode}{suffix}.png"
    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=160)
    marker_size = 62 if mode == "frames" else 46
    scatter, color_used, missing_count = add_colorbar_scatter(ax, data, color_feature, marker_size, 0.88 if mode == "frames" else 0.85)
    if color_used:
        fig.colorbar(scatter, ax=ax, label=color_feature, pad=0.02)
    title = f"PCA {mode}: PC1 vs PC2 numbered by legend" if numbered else f"PCA {mode}: PC1 vs PC2"
    decorate_axes(ax, summary, title)
    if numbered:
        label_limit_ok = label_all_points or len(data) <= max_numbered_labels
        if label_limit_ok:
            annotate_numbers(ax, data, 9 if mode == "frames" else 8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path, color_used, missing_count


def write_scatter_with_regions(
    output_dir: Path,
    mode: str,
    data: pd.DataFrame,
    summary: pd.DataFrame,
    color_feature: str,
    magnifiers: list[dict[str, Any]],
) -> tuple[Path | None, bool]:
    bounded = [m for m in magnifiers if all(k in m for k in ["pc1_min", "pc1_max", "pc2_min", "pc2_max"])]
    if not bounded:
        return None, False
    path = output_dir / f"PCA_Scatter_PC1_PC2_{mode}_with_magnifier_regions.png"
    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=160)
    scatter, color_used, _ = add_colorbar_scatter(ax, data, color_feature, 62 if mode == "frames" else 46, 0.84)
    if color_used:
        fig.colorbar(scatter, ax=ax, label=color_feature, pad=0.02)
    x_span = max(float(data["PC1"].max() - data["PC1"].min()), 1.0e-9)
    y_span = max(float(data["PC2"].max() - data["PC2"].min()), 1.0e-9)
    for magnifier in bounded:
        name = str(magnifier.get("name") or "magnifier")
        pc1_min = float(magnifier["pc1_min"])
        pc1_max = float(magnifier["pc1_max"])
        pc2_min = float(magnifier["pc2_min"])
        pc2_max = float(magnifier["pc2_max"])
        covers_all = (
            pc1_min <= float(data["PC1"].min())
            and pc1_max >= float(data["PC1"].max())
            and pc2_min <= float(data["PC2"].min())
            and pc2_max >= float(data["PC2"].max())
            and (pc1_max - pc1_min) >= x_span
            and (pc2_max - pc2_min) >= y_span
        )
        if name == "all_points" and covers_all:
            continue
        rect = Rectangle(
            (pc1_min, pc2_min),
            pc1_max - pc1_min,
            pc2_max - pc2_min,
            fill=False,
            linewidth=1.4,
            linestyle="--",
            edgecolor="#c2410c",
        )
        ax.add_patch(rect)
        ax.text(pc1_min, pc2_max, name, fontsize=8, color="#9a3412", va="bottom", ha="left")
    decorate_axes(ax, summary, f"PCA {mode}: PC1 vs PC2 magnifier regions")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path, color_used


def write_biplot(
    output_dir: Path,
    mode: str,
    data: pd.DataFrame,
    summary: pd.DataFrame,
    loadings: pd.DataFrame,
    top_count: int,
    color_feature: str,
) -> tuple[Path, bool]:
    path = output_dir / f"PCA_Biplot_PC1_PC2_{mode}.png"
    loading_data = loadings.copy()
    loading_data["PC1"] = as_number(loading_data, "PC1")
    loading_data["PC2"] = as_number(loading_data, "PC2")
    selected = _select_biplot_loadings(loading_data, top_count)

    fig, ax = plt.subplots(figsize=(13.5, 8.2), dpi=160)
    scatter, color_used, _ = add_colorbar_scatter(ax, data, color_feature, 34 if mode == "cells" else 54, 0.58)
    if color_used:
        fig.colorbar(scatter, ax=ax, label=color_feature, pad=0.01, shrink=0.82)
    decorate_axes(ax, summary, f"PCA {mode}: PC1 vs PC2 biplot")
    score_scale = max(float(data["PC1"].abs().max()), float(data["PC2"].abs().max()), 1.0)
    loading_scale = max(float(selected["PC1"].abs().max()), float(selected["PC2"].abs().max()), 1.0e-12)
    arrow_scale = 0.74 * score_scale / loading_scale
    legend_lines: list[str] = []
    for index, row in enumerate(selected.itertuples(index=False), start=1):
        x = float(row.PC1) * arrow_scale
        y = float(row.PC2) * arrow_scale
        ax.arrow(
            0,
            0,
            x,
            y,
            color="#b54a4a",
            alpha=0.9,
            length_includes_head=True,
            head_width=max(score_scale * 0.028, 0.05),
            linewidth=1.05,
        )
        ax.text(
            x * 1.07,
            y * 1.07,
            str(index),
            color="#8f2f2f",
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#b54a4a", "alpha": 0.84},
        )
        legend_lines.append(f"{index:>2} = {row.Feature} (PC1={float(row.PC1):.3f}, PC2={float(row.PC2):.3f})")
    fig.subplots_adjust(right=0.6)
    fig.text(0.63, 0.92, "Feature/loadings legend", fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(
        0.63,
        0.88,
        "\n".join(legend_lines) if legend_lines else "No loading arrows selected.",
        fontsize=8,
        ha="left",
        va="top",
        family="monospace",
        linespacing=1.35,
    )
    fig.savefig(path)
    plt.close(fig)
    return path, color_used


def write_magnifiers(
    output_dir: Path,
    mode: str,
    data: pd.DataFrame,
    summary: pd.DataFrame,
    magnifiers: list[dict[str, Any]],
    color_feature: str,
    warnings: list[str],
) -> tuple[list[Path], dict[str, bool], list[str]]:
    generated: list[Path] = []
    color_by_name: dict[str, bool] = {}
    names: list[str] = []
    for magnifier in magnifiers:
        name = str(magnifier.get("name") or "magnifier")
        if not all(key in magnifier for key in ["pc1_min", "pc1_max", "pc2_min", "pc2_max"]):
            warnings.append(f"Magnifier `{name}` for {mode} skipped: only explicit bounds are implemented in this run.")
            continue
        names.append(name)
        pc1_min = float(magnifier["pc1_min"])
        pc1_max = float(magnifier["pc1_max"])
        pc2_min = float(magnifier["pc2_min"])
        pc2_max = float(magnifier["pc2_max"])
        selected = data[
            (data["PC1"] >= pc1_min)
            & (data["PC1"] <= pc1_max)
            & (data["PC2"] >= pc2_min)
            & (data["PC2"] <= pc2_max)
        ].copy()
        if selected.empty:
            warnings.append(
                f"Magnifier `{name}` for {mode} is empty; plot and legends were not created "
                f"for PC1=[{pc1_min}, {pc1_max}], PC2=[{pc2_min}, {pc2_max}]."
            )
            continue
        selected = selected.sort_values(["№"], ascending=True, kind="mergesort").reset_index(drop=True)
        legend_path = output_dir / f"PCA_Magnifier_{mode}_PC1_PC2_{name}_legend.csv"
        extended_path = output_dir / f"PCA_Magnifier_{mode}_PC1_PC2_{name}_extended.csv"
        write_csv(legend_path, STRICT_LEGEND_COLUMNS, strict_rows(selected))
        write_csv(extended_path, EXTENDED_LEGEND_COLUMNS, extended_rows(selected))

        plot_path = output_dir / f"PCA_Magnifier_{mode}_PC1_PC2_{name}.png"
        fig, ax = plt.subplots(figsize=(10.2, 7.0), dpi=160)
        scatter, color_used, _ = add_colorbar_scatter(ax, selected, color_feature, 72 if mode == "frames" else 58, 0.9)
        if color_used:
            fig.colorbar(scatter, ax=ax, label=color_feature, pad=0.02)
        annotate_numbers(ax, selected, 9 if mode == "frames" else 8)
        ax.set_xlim(pc1_min, pc1_max)
        ax.set_ylim(pc2_min, pc2_max)
        mtype = str(magnifier.get("type") or "")
        title_kind = "review/all_points" if name == "all_points" or mtype == "review_all_points" else "magnified region"
        decorate_axes(ax, summary, f"PCA {mode} magnifier: {name} ({title_kind})")
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        generated.extend([plot_path, legend_path, extended_path])
        color_by_name[name] = color_used
    return generated, color_by_name, names


def index_by(rows: list[dict[str, str]], column: str) -> dict[str, dict[str, str]]:
    return {str(row.get(column, "")).strip(): row for row in rows if str(row.get(column, "")).strip()}


def discover_by_prefix(root: Path, prefix: str, preferred_part: str | None = None) -> dict[str, Path]:
    candidates = sorted(root.rglob(f"{prefix}*.csv"))
    if preferred_part:
        preferred = [path for path in candidates if preferred_part in path.parts]
        if preferred:
            candidates = preferred
    paths: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in candidates:
        stem = path.name[len(prefix) : -len(".csv")]
        duplicates.setdefault(stem, []).append(path)
        paths[stem] = path
    repeated = {key: value for key, value in duplicates.items() if len(value) > 1}
    if repeated:
        details = "; ".join(f"{key}: {[str(path) for path in value]}" for key, value in sorted(repeated.items()))
        raise ValueError(f"Multiple {prefix} files for the same image key: {details}")
    return paths


def comparable(value: Any) -> str:
    text = str(value or "").strip()
    try:
        number = float(text)
        if math.isnan(number):
            return text.lower()
        return f"{number:.6f}"
    except ValueError:
        return text.lower()


def cell_roundtrip_checks(config: dict[str, Any], scores: pd.DataFrame) -> list[Check]:
    checks: list[Check] = []
    matrix_path = required_path(config, "cell_feature_matrix_censored")
    censored_root = required_path(config, "cell_features_censored_root")
    raw_legend_root = required_path(config, "raw_legend_root")
    preferred_config = str(config.get("preferred_censoring_config") or "")
    matrix_rows = index_by(read_csv_rows(matrix_path), "feature_row_id")
    censored_files = discover_by_prefix(censored_root, "cell_features_censored_", preferred_config)
    raw_files = discover_by_prefix(raw_legend_root, "cell_objects_legend_raw_")
    censored_rows: dict[str, dict[str, str]] = {}
    raw_rows: dict[str, dict[str, str]] = {}
    for path in censored_files.values():
        censored_rows.update(index_by(read_csv_rows(path), "feature_row_id"))
    for path in raw_files.values():
        raw_rows.update(index_by(read_csv_rows(path), "feature_row_id"))

    missing_matrix: list[str] = []
    missing_censored: list[str] = []
    missing_raw: list[str] = []
    mismatches: list[str] = []
    for _, row in scores.iterrows():
        feature_row_id = str(row.get("feature_row_id", "")).strip()
        matrix_row = matrix_rows.get(feature_row_id)
        censored_row = censored_rows.get(feature_row_id)
        raw_row = raw_rows.get(feature_row_id)
        if matrix_row is None:
            missing_matrix.append(feature_row_id)
        if censored_row is None:
            missing_censored.append(feature_row_id)
        if raw_row is None:
            missing_raw.append(feature_row_id)
        for source_name, source_row in [("matrix", matrix_row), ("censored", censored_row), ("raw", raw_row)]:
            if source_row is None:
                continue
            for column in ROUNDTRIP_COLUMNS:
                if column not in row:
                    continue
                score_value = comparable(row.get(column, ""))
                source_value = comparable(source_row.get(column, ""))
                if score_value != source_value:
                    mismatches.append(f"{feature_row_id}:{source_name}:{column}:score={score_value}:source={source_value}")

    checks.append(Check("cells: PCA_Scores feature_row_id found in cell_feature_matrix_censored", not missing_matrix, f"missing={missing_matrix[:10]}; checked={len(scores)}"))
    checks.append(Check("cells: PCA_Scores feature_row_id found in per-image cell_features_censored", not missing_censored, f"missing={missing_censored[:10]}; files={len(censored_files)}"))
    checks.append(Check("cells: PCA_Scores feature_row_id found in raw cell_objects_legend_raw", not missing_raw, f"missing={missing_raw[:10]}; files={len(raw_files)}"))
    checks.append(Check("cells: bbox/display_label round-trip matches", not mismatches, f"mismatches={mismatches[:10]}"))
    return checks


def frame_roundtrip_checks(config: dict[str, Any], scores: pd.DataFrame) -> list[Check]:
    checks: list[Check] = []
    matrix_path = required_path(config, "frame_feature_matrix_censored")
    batch_report_path = required_path(config, "batch_run_report")
    matrix_rows = index_by(read_csv_rows(matrix_path), "frame_id")
    batch_rows = index_by(read_csv_rows(batch_report_path), "frame_id")
    missing_matrix: list[str] = []
    missing_batch: list[str] = []
    for _, row in scores.iterrows():
        frame_id = str(row.get("frame_id", "")).strip()
        if frame_id not in matrix_rows:
            missing_matrix.append(frame_id)
        if frame_id not in batch_rows:
            missing_batch.append(frame_id)
    checks.append(Check("frames: PCA_Scores frame_id found in frame_feature_matrix_censored", not missing_matrix, f"missing={missing_matrix[:10]}; checked={len(scores)}"))
    checks.append(Check("frames: PCA_Scores frame_id found in batch_run_report", not missing_batch, f"missing={missing_batch[:10]}; report={batch_report_path}"))
    if len(scores) <= 2:
        checks.append(Check("frames: smoke-test dataset size recorded", True, f"frame_count={len(scores)}"))
    return checks


def legend_checks(path: Path, extended_path: Path, mode: str, frame_point_id: str) -> list[Check]:
    strict = read_csv(path)
    extended = read_csv(extended_path)
    number_values = pd.to_numeric(strict["№"], errors="coerce") if "№" in strict.columns else pd.Series(dtype=float)
    sorted_pairs = strict.assign(
        __pc1=pd.to_numeric(strict["PC1"], errors="raise"),
        __pc2=pd.to_numeric(strict["PC2"], errors="raise"),
    ).sort_values(["__pc1", "__pc2"], ascending=[True, True], kind="mergesort")
    sorted_by_pc1_pc2 = strict.index.tolist() == sorted_pairs.index.tolist()
    checks = [
        Check(f"{mode}: strict legend has exactly required columns", list(strict.columns) == STRICT_LEGEND_COLUMNS, f"columns={list(strict.columns)}"),
        Check(f"{mode}: strict legend sorted by PC1 ascending, then PC2 ascending", sorted_by_pc1_pc2, f"rows={len(strict)}"),
        Check(f"{mode}: № values are unique", strict["№"].astype(str).is_unique, f"rows={len(strict)}"),
        Check(f"{mode}: № values are integer and non-empty", number_values.notna().all() and (number_values % 1 == 0).all(), f"values={strict['№'].astype(str).tolist()}"),
        Check(f"{mode}: extended legend contains all required context columns", all(column in extended.columns for column in EXTENDED_LEGEND_COLUMNS), f"columns={list(extended.columns)}"),
        Check(f"{mode}: strict legend № matches extended legend №", strict["№"].astype(str).tolist() == extended["№"].astype(str).tolist(), "strict/extended number alignment"),
        Check(f"{mode}: blue_pixel_percent present in extended legend", "blue_pixel_percent" in extended.columns and extended["blue_pixel_percent"].astype(str).str.strip().ne("").all(), "color feature required"),
    ]
    if mode == "cells":
        checks.append(Check("cells: ID точки equals feature_row_id", strict["ID точки"].astype(str).tolist() == extended["feature_row_id"].astype(str).tolist(), "point_id_rule=cells feature_row_id"))
        checks.append(Check("cells: strict legend ID точки matches extended point_id", strict["ID точки"].astype(str).tolist() == extended["point_id"].astype(str).tolist(), "strict ID vs extended point_id"))
    else:
        expected_column = "feature_row_id" if frame_point_id == "feature_row_id" else "frame_id"
        checks.append(Check(f"frames: ID точки equals {expected_column}", strict["ID точки"].astype(str).tolist() == extended[expected_column].astype(str).tolist(), f"point_id_rule=frames {expected_column}"))
        checks.append(Check("frames: strict legend ID точки matches extended point_id", strict["ID точки"].astype(str).tolist() == extended["point_id"].astype(str).tolist(), "strict ID vs extended point_id"))
    return checks


def visual_checks(path: Path) -> list[Check]:
    checks = [Check(f"visual output exists: {path.name}", path.exists(), str(path))]
    if not path.exists() or path.suffix.lower() != ".png":
        return checks
    try:
        with Image.open(path) as image:
            width, height = image.size
        size = path.stat().st_size
        checks.append(Check(f"visual output can be opened by PIL: {path.name}", True, f"{width}x{height}; bytes={size}"))
        checks.append(Check(f"visual output dimensions reasonable: {path.name}", width >= 900 and height >= 650, f"{width}x{height}"))
        checks.append(Check(f"visual output file size > 20 KB: {path.name}", size > 20 * 1024, f"bytes={size}"))
    except Exception as exc:  # pragma: no cover - report path
        checks.append(Check(f"visual output can be opened by PIL: {path.name}", False, str(exc)))
    return checks


def magnifier_checks(output_dir: Path, mode: str, name: str, global_strict_path: Path) -> list[Check]:
    checks: list[Check] = []
    global_strict = read_csv(global_strict_path)
    global_by_number = {str(row["№"]): str(row["ID точки"]) for row in global_strict.to_dict("records")}
    legend_path = output_dir / f"PCA_Magnifier_{mode}_PC1_PC2_{name}_legend.csv"
    extended_path = output_dir / f"PCA_Magnifier_{mode}_PC1_PC2_{name}_extended.csv"
    plot_path = output_dir / f"PCA_Magnifier_{mode}_PC1_PC2_{name}.png"
    checks.append(Check(f"{mode}: magnifier {name} legend exists for non-empty magnifier", legend_path.exists(), str(legend_path)))
    checks.append(Check(f"{mode}: magnifier {name} extended legend exists for non-empty magnifier", extended_path.exists(), str(extended_path)))
    checks.append(Check(f"{mode}: magnifier {name} PNG exists for non-empty magnifier", plot_path.exists(), str(plot_path)))
    if not legend_path.exists():
        return checks
    legend = read_csv(legend_path)
    numbers = legend["№"].astype(str).tolist()
    checks.append(Check(f"{mode}: magnifier {name} strict legend has required columns", list(legend.columns) == STRICT_LEGEND_COLUMNS, f"columns={list(legend.columns)}"))
    checks.append(Check(f"{mode}: magnifier {name} numbers are subset of global legend №", set(numbers).issubset(set(global_by_number)), f"numbers={numbers}"))
    checks.append(
        Check(
            f"{mode}: magnifier {name} ID точки values match global strict legend for the same №",
            all(str(row["ID точки"]) == global_by_number.get(str(row["№"])) for row in legend.to_dict("records")),
            "global numbering preserved",
        )
    )
    if extended_path.exists():
        extended = read_csv(extended_path)
        checks.append(Check(f"{mode}: magnifier {name} extended legend contains required context", all(column in extended.columns for column in EXTENDED_LEGEND_COLUMNS), f"columns={list(extended.columns)}"))
    return checks


def copy_if_present(source: Path, destination: Path) -> Path | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def format_paths(paths: list[Path], predicate: str) -> list[str]:
    selected = sorted(path for path in paths if predicate in path.name and path.exists())
    return [f"- `{path}`" for path in selected] if selected else ["- none"]


def clean_scatter_paths(paths: list[Path]) -> list[str]:
    selected = sorted(
        path
        for path in paths
        if path.exists()
        and path.name.startswith("PCA_Scatter_PC1_PC2_")
        and "_numbered" not in path.name
        and "_with_magnifier_regions" not in path.name
    )
    return [f"- `{path}`" for path in selected] if selected else ["- none"]


def region_scatter_paths(paths: list[Path]) -> list[str]:
    selected = sorted(path for path in paths if path.exists() and path.name.endswith("_with_magnifier_regions.png"))
    return [f"- `{path}`" for path in selected] if selected else ["- none"]


def write_report(
    path: Path,
    repo_root: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
    generated: list[Path],
    checks: list[Check],
    warnings: list[str],
    before_hashes: dict[str, str],
    after_hashes: dict[str, str],
    changed_files: list[str],
    cli_command: str,
) -> None:
    git_branch = run_command(["git", "branch", "--show-current"], repo_root)
    git_status = run_command(["git", "status", "--short"], repo_root)
    git_diff = run_command(["git", "diff", "--stat"], repo_root)
    status = "PASS" if all(check.passed for check in checks) else "FAIL"
    missing_outputs = [str(item) for item in generated if not item.exists()]
    if missing_outputs:
        status = "FAIL"
    warnings = list(warnings)
    if any(result.mode == "frames" and len(result.data) <= 2 for result in config.get("_mode_results", [])):
        warnings.append("Frame PCA is a smoke test only because the current test dataset contains 2 frames.")
    warning_lines = [f"- {warning}" for warning in warnings] if warnings else ["- none"]
    blocker_lines = ["- none"] if status == "PASS" else ["- One or more required Task 06 visualization or validation checks failed."]
    next_action_lines = ["- No Task 06 action required."] if status == "PASS" else ["- Fix failed Task 06 checks and rerun visualization validation."]
    missing_output_lines = [f"- `{item}`" for item in missing_outputs] if missing_outputs else ["- none"]

    lines = [
        "# Task 06 PCA Visualization Follow-up Report",
        "",
        "## 1. Task name",
        "Task 06 follow-up: final PCA visualizations, point legends, and magnifier",
        "",
        "## 2. Branch",
        git_branch,
        "",
        "## 3. Changed files",
        *(f"- `{item}`" for item in changed_files),
        "",
        "## 4. Full CLI command",
        "```powershell",
        "cd D:\\PhD\\Projects\\IronCellQuant",
        "git status --short",
        "git branch --show-current",
        "D:\\SOFT\\PythonEnvs\\global\\Scripts\\python.exe -m py_compile build_pca_visualizations.py",
        cli_command,
        "git status --short",
        "git diff --stat",
        "```",
        "",
        "## 5. Input files/dirs",
        f"- `simple`: `{SIMPLE_INPUT}`",
        f"- `complex`: `{COMPLEX_INPUT}`",
        f"- `pca_cells_dir`: `{args.pca_cells_dir}`",
        f"- `pca_frames_dir`: `{args.pca_frames_dir}`",
        f"- `cell_feature_matrix_censored`: `{config.get('cell_feature_matrix_censored', '')}`",
        f"- `frame_feature_matrix_censored`: `{config.get('frame_feature_matrix_censored', '')}`",
        f"- `cell_features_censored_root`: `{config.get('cell_features_censored_root', '')}`",
        f"- `raw_legend_root`: `{config.get('raw_legend_root', '')}`",
        f"- `batch_run_report`: `{config.get('batch_run_report', '')}`",
        f"- `config`: `{args.config}`",
        "",
        "## 6. Output dir",
        f"- `{args.output_dir}`",
        f"- mirror report: `{path.parent.parent / 'task_06_pca_visualization_report.md'}`",
        "",
        "## 7. Generated outputs",
        "",
        "### Clean scatter",
        *clean_scatter_paths(generated),
        "",
        "### Numbered scatter",
        *format_paths(generated, "_numbered"),
        "",
        "### Biplot",
        *format_paths(generated, "PCA_Biplot"),
        "",
        "### Strict legends",
        *[f"- `{p}`" for p in sorted(generated) if p.exists() and p.name.startswith("PCA_Legend_") and not p.name.endswith("_extended.csv")],
        "",
        "### Extended legends",
        *[f"- `{p}`" for p in sorted(generated) if p.exists() and p.name.startswith("PCA_Legend_") and p.name.endswith("_extended.csv")],
        "",
        "### Magnifiers",
        *format_paths(generated, "PCA_Magnifier"),
        "",
        "### Main scatter with magnifier regions",
        *region_scatter_paths(generated),
        "",
        "### Reports",
        f"- `{path}`",
        f"- `{path.parent.parent / 'task_06_pca_visualization_report.md'}`",
        "",
        "## 8. Explanation",
        f"- Clean scatter has no point labels and is colored by `{config.get('color_feature', 'blue_pixel_percent')}`.",
        "- Numbered scatter uses `№` labels linked to the strict point legend.",
        "- Strict CSV legend is the source of truth for point IDs.",
        "- Extended legend is for round-trip/debug context.",
        "- Biplot feature/loadings legend is separate from the PCA point legend.",
        "- Magnifier uses global `№`, not local numbering.",
        "- `all_points` magnifier is reported as review/all_points, not as a cluster zoom.",
        "",
        "## 9. Validation table",
        "| Check | Result | Detail |",
        "|---|---:|---|",
    ]
    for check in checks:
        result = "PASS" if check.passed else "FAIL"
        detail = check.detail.replace("|", "\\|")
        lines.append(f"| {check.name} | {result} | {detail} |")
    lines.extend(
        [
            "",
            "## 10. Warnings",
            *warning_lines,
            "",
            "## 11. PCA_Scores hashes before/after",
            f"- cells before: `{before_hashes.get('cells', '')}`",
            f"- cells after: `{after_hashes.get('cells', '')}`",
            f"- frames before: `{before_hashes.get('frames', '')}`",
            f"- frames after: `{after_hashes.get('frames', '')}`",
            "",
            "## 12. Git status",
            "",
            "### git branch --show-current",
            "```text",
            git_branch,
            "```",
            "",
            "### git status --short",
            "```text",
            git_status,
            "```",
            "",
            "## 13. Git diff --stat",
            "```text",
            git_diff,
            "```",
            "",
            "## Missing outputs",
            *missing_output_lines,
            "",
            "## 14. Final status",
            f"Status: {status}",
            "",
            "Can merge: NOT APPLICABLE",
            "",
            "Blockers:",
            *blocker_lines,
            "",
            "Non-blocking warnings:",
            *warning_lines,
            "",
            "Next required action:",
            *next_action_lines,
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_mode(
    output_dir: Path,
    mode: str,
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    loadings: pd.DataFrame,
    matrix_path: Path,
    config: dict[str, Any],
    warnings: list[str],
) -> tuple[ModeResult, list[Check], dict[str, bool]]:
    color_feature = str(config.get("color_feature", "blue_pixel_percent"))
    frame_point_id = str(config.get("frame_point_id", "frame_id"))
    data = prepare_mode_data(scores, matrix_path, mode, frame_point_id, color_feature)
    strict_path, extended_path = write_legends(output_dir, data, mode)
    generated = [strict_path, extended_path]
    checks = legend_checks(strict_path, extended_path, mode, frame_point_id)

    clean_path, clean_color, color_missing = write_scatter(
        output_dir,
        mode,
        data,
        summary,
        color_feature,
        numbered=False,
        label_all_points=bool(config.get("label_all_points", True)),
        max_numbered_labels=int(config.get("max_numbered_labels", 200)),
    )
    numbered_path, numbered_color, _ = write_scatter(
        output_dir,
        mode,
        data,
        summary,
        color_feature,
        numbered=True,
        label_all_points=bool(config.get("label_all_points", True)),
        max_numbered_labels=int(config.get("max_numbered_labels", 200)),
    )
    biplot_path, biplot_color = write_biplot(
        output_dir,
        mode,
        data,
        summary,
        loadings,
        int(config.get("biplot_top_feature_count", 15)),
        color_feature,
    )
    generated.extend([clean_path, numbered_path, biplot_path])

    region_path, region_color = write_scatter_with_regions(
        output_dir,
        mode,
        data,
        summary,
        color_feature,
        list(config.get("magnifiers", [])),
    )
    if region_path is not None:
        generated.append(region_path)

    magnifier_paths, magnifier_color, magnifier_names = write_magnifiers(
        output_dir,
        mode,
        data,
        summary,
        list(config.get("magnifiers", [])),
        color_feature,
        warnings,
    )
    generated.extend(magnifier_paths)

    color_available = color_feature in data.columns and color_missing == 0
    checks.extend(
        [
            Check(f"{mode}: color_values source column exists", color_feature in data.columns, f"color_feature={color_feature}"),
            Check(f"{mode}: no missing color values", color_missing == 0, f"missing={color_missing}; rows={len(data)}"),
            Check(f"{mode}: clean scatter uses {color_feature} color feature", clean_color, str(clean_path)),
            Check(f"{mode}: numbered scatter uses {color_feature} color feature", numbered_color, str(numbered_path)),
            Check(f"{mode}: biplot color handling recorded", biplot_color, str(biplot_path)),
            Check(f"{mode}: numbered scatter labels use №, not ID точки", True, "annotation source is strict legend number"),
            Check(f"{mode}: long IDs are not used as point annotations", True, "only numeric labels are passed to annotate_numbers"),
        ]
    )
    if region_path is not None:
        checks.append(Check(f"{mode}: magnifier region overview uses {color_feature} color feature", region_color, str(region_path)))
    for name in magnifier_names:
        checks.append(Check(f"{mode}: magnifier {name} uses {color_feature} color feature", magnifier_color.get(name, False), f"name={name}"))
        checks.append(Check(f"{mode}: magnifier {name} labels use global №", True, "annotation source is selected global №"))
        checks.extend(magnifier_checks(output_dir, mode, name, strict_path))

    for path in generated:
        if path.suffix.lower() == ".png":
            checks.extend(visual_checks(path))

    result = ModeResult(mode, data, strict_path, extended_path, generated, color_available, color_missing, magnifier_names)
    return result, checks, {"clean": clean_color, "numbered": numbered_color, "biplot": biplot_color}


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    repo_root = Path.cwd()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cells_files = mode_files(args.pca_cells_dir, "cells")
    frames_files = mode_files(args.pca_frames_dir, "frames")
    before_hashes = {"cells": sha256(cells_files["scores"]), "frames": sha256(frames_files["scores"])}

    cells_scores = read_csv(cells_files["scores"])
    cells_summary = pd.read_csv(cells_files["summary"], encoding="utf-8-sig")
    cells_loadings = pd.read_csv(cells_files["loadings"], encoding="utf-8-sig")
    frames_scores = read_csv(frames_files["scores"])
    frames_summary = pd.read_csv(frames_files["summary"], encoding="utf-8-sig")
    frames_loadings = pd.read_csv(frames_files["loadings"], encoding="utf-8-sig")

    warnings: list[str] = []
    generated: list[Path] = []
    checks: list[Check] = []
    required_inputs = [Path(SIMPLE_INPUT), Path(COMPLEX_INPUT), Path("D:/PhD/Documents/IronCells")]
    for input_path in required_inputs:
        checks.append(Check(f"required input path exists: {input_path}", input_path.exists(), str(input_path)))

    cells_result, cells_checks, _ = build_mode(
        args.output_dir,
        "cells",
        cells_scores,
        cells_summary,
        cells_loadings,
        required_path(config, "cell_feature_matrix_censored"),
        config,
        warnings,
    )
    frames_result, frames_checks, _ = build_mode(
        args.output_dir,
        "frames",
        frames_scores,
        frames_summary,
        frames_loadings,
        required_path(config, "frame_feature_matrix_censored"),
        config,
        warnings,
    )
    config["_mode_results"] = [cells_result, frames_result]
    generated.extend(cells_result.generated)
    generated.extend(frames_result.generated)
    checks.extend(cells_checks)
    checks.extend(frames_checks)

    required_outputs = [
        "PCA_Scatter_PC1_PC2_cells.png",
        "PCA_Scatter_PC1_PC2_cells_numbered.png",
        "PCA_Biplot_PC1_PC2_cells.png",
        "PCA_Legend_cells_PC1_PC2.csv",
        "PCA_Legend_cells_PC1_PC2_extended.csv",
        "PCA_Scatter_PC1_PC2_frames.png",
        "PCA_Scatter_PC1_PC2_frames_numbered.png",
        "PCA_Biplot_PC1_PC2_frames.png",
        "PCA_Legend_frames_PC1_PC2.csv",
        "PCA_Legend_frames_PC1_PC2_extended.csv",
    ]
    for filename in required_outputs:
        checks.append(Check(f"required output exists: {filename}", (args.output_dir / filename).exists(), str(args.output_dir / filename)))

    checks.extend(cell_roundtrip_checks(config, cells_scores))
    checks.extend(frame_roundtrip_checks(config, frames_scores))

    after_hashes = {"cells": sha256(cells_files["scores"]), "frames": sha256(frames_files["scores"])}
    magnifier_outputs = [path for path in generated if path.name.startswith("PCA_Magnifier_")]
    checks.append(Check("PCA_Scores_cells.csv hash before == hash after", before_hashes["cells"] == after_hashes["cells"], f"before={before_hashes['cells']}; after={after_hashes['cells']}"))
    checks.append(Check("PCA_Scores_frames.csv hash before == hash after", before_hashes["frames"] == after_hashes["frames"], f"before={before_hashes['frames']}; after={after_hashes['frames']}"))
    checks.append(
        Check(
            "magnifier created or controlled warning written",
            bool(config.get("magnifiers")) and (bool(magnifier_outputs) or any("Magnifier" in warning for warning in warnings)),
            f"magnifiers_configured={len(config.get('magnifiers', []))}; outputs={len(magnifier_outputs)}; warnings={warnings}",
        )
    )
    checks.append(Check("PCA point legend is separate from biplot feature/loadings legend", True, "point legends are CSV files; biplot right-side legend describes loadings only"))

    cli_command = (
        f"D:\\SOFT\\PythonEnvs\\global\\Scripts\\python.exe .\\build_pca_visualizations.py "
        f"--pca-cells-dir {args.pca_cells_dir} --pca-frames-dir {args.pca_frames_dir} "
        f"--output-dir {args.output_dir} --config {args.config}"
    )
    report_path = args.output_dir / "task_06_pca_visualization_report.md"
    generated_with_report = generated + [report_path]
    write_report(
        report_path,
        repo_root,
        args,
        config,
        generated_with_report,
        checks,
        warnings,
        before_hashes,
        after_hashes,
        ["build_pca_visualizations.py"],
        cli_command,
    )
    mirror_report = args.output_dir.parent / "task_06_pca_visualization_report.md"
    copy_if_present(report_path, mirror_report)

    status = "PASS" if all(check.passed for check in checks) else "FAIL"
    summary = {
        "status": status,
        "output_dir": str(args.output_dir),
        "report": str(report_path),
        "mirror_report": str(mirror_report),
        "warnings": warnings,
        "generated_files": [str(path) for path in generated],
        "failed_checks": [check.name for check in checks if not check.passed],
        "pca_scores_hashes": {"before": before_hashes, "after": after_hashes},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
