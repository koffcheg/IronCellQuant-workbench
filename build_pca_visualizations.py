"""Build PCA point legends, mode-named plots, magnifiers, and round-trip checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pca_analysis.config import PCAConfig
from pca_analysis.visualization import _axis_label, _select_biplot_loadings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PCA cells/frames visualizations, point legends, magnifiers, and validation report."
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


def point_id(row: pd.Series, mode: str, frame_point_id: str) -> str:
    if mode == "cells":
        return str(row.get("feature_row_id", "")).strip()
    if frame_point_id == "feature_row_id":
        return str(row.get("feature_row_id", "")).strip()
    return str(row.get("frame_id", "")).strip()


def write_legends(
    output_dir: Path,
    scores: pd.DataFrame,
    mode: str,
    frame_point_id: str,
) -> tuple[Path, Path]:
    ordered = sorted_scores(scores)
    strict_rows: list[dict[str, Any]] = []
    extended_rows: list[dict[str, Any]] = []
    for index, row in enumerate(ordered.to_dict("records"), start=1):
        pid = point_id(pd.Series(row), mode, frame_point_id)
        strict_rows.append({"№": index, "PC1": row["PC1"], "PC2": row["PC2"], "ID точки": pid})
        extended = {column: row.get(column, "") for column in EXTENDED_LEGEND_COLUMNS}
        extended.update({"№": index, "PC1": row["PC1"], "PC2": row["PC2"], "point_id": pid})
        extended_rows.append(extended)

    strict_path = output_dir / f"PCA_Legend_{mode}_PC1_PC2.csv"
    extended_path = output_dir / f"PCA_Legend_{mode}_PC1_PC2_extended.csv"
    write_csv(strict_path, STRICT_LEGEND_COLUMNS, strict_rows)
    write_csv(extended_path, EXTENDED_LEGEND_COLUMNS, extended_rows)
    return strict_path, extended_path


def write_scatter(output_dir: Path, mode: str, scores: pd.DataFrame, summary: pd.DataFrame) -> Path:
    path = output_dir / f"PCA_Scatter_PC1_PC2_{mode}.png"
    data = scores.copy()
    data["PC1"] = as_number(data, "PC1")
    data["PC2"] = as_number(data, "PC2")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.scatter(data["PC1"], data["PC2"], s=38, alpha=0.82, color="#2f6f8f", edgecolor="white", linewidth=0.35)
    ax.axhline(0, color="#666666", linewidth=0.8, alpha=0.45)
    ax.axvline(0, color="#666666", linewidth=0.8, alpha=0.45)
    ax.set_xlabel(_axis_label("PC1", summary))
    ax.set_ylabel(_axis_label("PC2", summary))
    ax.set_title(f"PCA Scatter {mode}: PC1 vs PC2")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_biplot(
    output_dir: Path,
    mode: str,
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    loadings: pd.DataFrame,
    top_count: int,
) -> Path:
    path = output_dir / f"PCA_Biplot_PC1_PC2_{mode}.png"
    data = scores.copy()
    data["PC1"] = as_number(data, "PC1")
    data["PC2"] = as_number(data, "PC2")
    loading_data = loadings.copy()
    loading_data["PC1"] = as_number(loading_data, "PC1")
    loading_data["PC2"] = as_number(loading_data, "PC2")
    selected = _select_biplot_loadings(loading_data, top_count)

    fig, ax = plt.subplots(figsize=(13, 8), dpi=150)
    ax.scatter(data["PC1"], data["PC2"], s=24, alpha=0.32, color="#2f6f8f", edgecolor="none")
    ax.axhline(0, color="#666666", linewidth=0.8, alpha=0.45)
    ax.axvline(0, color="#666666", linewidth=0.8, alpha=0.45)
    score_scale = max(float(data["PC1"].abs().max()), float(data["PC2"].abs().max()), 1.0)
    loading_scale = max(float(selected["PC1"].abs().max()), float(selected["PC2"].abs().max()), 1.0e-12)
    arrow_scale = 0.75 * score_scale / loading_scale
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
            head_width=max(score_scale * 0.03, 0.05),
            linewidth=1.15,
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
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#b54a4a", "alpha": 0.82},
        )
        legend_lines.append(f"{index:>2} = {row.Feature} (PC1={float(row.PC1):.3f}, PC2={float(row.PC2):.3f})")

    ax.set_xlabel(_axis_label("PC1", summary))
    ax.set_ylabel(_axis_label("PC2", summary))
    ax.set_title(f"PCA Biplot {mode}: PC1 vs PC2")
    ax.grid(alpha=0.22)
    fig.subplots_adjust(right=0.58)
    fig.text(0.61, 0.92, "Feature loadings", fontsize=11, fontweight="bold", ha="left", va="top")
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
    fig.savefig(path)
    plt.close(fig)
    return path


def write_magnifiers(
    output_dir: Path,
    mode: str,
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    magnifiers: list[dict[str, Any]],
    frame_point_id: str,
    warnings: list[str],
) -> list[Path]:
    generated: list[Path] = []
    data = scores.copy()
    data["PC1"] = as_number(data, "PC1")
    data["PC2"] = as_number(data, "PC2")
    for magnifier in magnifiers:
        name = str(magnifier.get("name") or "magnifier")
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
                f"Magnifier `{name}` for {mode} is empty; plot and legend were not created "
                f"for PC1=[{pc1_min}, {pc1_max}], PC2=[{pc2_min}, {pc2_max}]."
            )
            continue
        selected = sorted_scores(selected)
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(selected.to_dict("records"), start=1):
            pid = point_id(pd.Series(row), mode, frame_point_id)
            rows.append({"№": index, "PC1": row["PC1"], "PC2": row["PC2"], "ID точки": pid})
        legend_path = output_dir / f"PCA_Magnifier_{mode}_PC1_PC2_{name}_legend.csv"
        write_csv(legend_path, STRICT_LEGEND_COLUMNS, rows)

        plot_path = output_dir / f"PCA_Magnifier_{mode}_PC1_PC2_{name}.png"
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        ax.scatter(selected["PC1"], selected["PC2"], s=56, alpha=0.9, color="#2f6f8f", edgecolor="white", linewidth=0.45)
        for _, row in selected.iterrows():
            ax.annotate(
                point_id(row, mode, frame_point_id),
                (float(row["PC1"]), float(row["PC2"])),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_xlim(pc1_min, pc1_max)
        ax.set_ylim(pc2_min, pc2_max)
        ax.axhline(0, color="#666666", linewidth=0.8, alpha=0.35)
        ax.axvline(0, color="#666666", linewidth=0.8, alpha=0.35)
        ax.set_xlabel(_axis_label("PC1", summary))
        ax.set_ylabel(_axis_label("PC2", summary))
        ax.set_title(f"PCA Magnifier {mode}: {name}")
        ax.grid(alpha=0.22)
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        generated.extend([plot_path, legend_path])
    return generated


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
                score_value = comparable(row.get(column, ""))
                source_value = comparable(source_row.get(column, ""))
                if score_value != source_value:
                    mismatches.append(f"{feature_row_id}:{source_name}:{column}:score={score_value}:source={source_value}")

    checks.append(
        Check(
            "cells: PCA_Scores feature_row_id found in cell_feature_matrix_censored",
            not missing_matrix,
            f"missing={missing_matrix[:10]}; checked={len(scores)}",
        )
    )
    checks.append(
        Check(
            "cells: PCA_Scores feature_row_id found in per-image cell_features_censored",
            not missing_censored,
            f"missing={missing_censored[:10]}; files={len(censored_files)}",
        )
    )
    checks.append(
        Check(
            "cells: PCA_Scores feature_row_id found in raw cell_objects_legend_raw",
            not missing_raw,
            f"missing={missing_raw[:10]}; files={len(raw_files)}",
        )
    )
    checks.append(
        Check(
            "cells: bbox/display_label round-trip matches",
            not mismatches,
            f"mismatches={mismatches[:10]}",
        )
    )
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
    checks.append(
        Check(
            "frames: PCA_Scores frame_id found in frame_feature_matrix_censored",
            not missing_matrix,
            f"missing={missing_matrix[:10]}; checked={len(scores)}",
        )
    )
    checks.append(
        Check(
            "frames: PCA_Scores frame_id found in batch_run_report",
            not missing_batch,
            f"missing={missing_batch[:10]}; report={batch_report_path}",
        )
    )
    return checks


def legend_checks(path: Path, extended_path: Path, mode: str, frame_point_id: str) -> list[Check]:
    strict = read_csv(path)
    extended = read_csv(extended_path)
    checks = [
        Check(
            f"{mode}: strict legend has exactly required columns",
            list(strict.columns) == STRICT_LEGEND_COLUMNS,
            f"columns={list(strict.columns)}",
        ),
        Check(
            f"{mode}: strict legend sorted by PC1 ascending",
            pd.to_numeric(strict["PC1"], errors="raise").is_monotonic_increasing,
            f"rows={len(strict)}",
        ),
        Check(
            f"{mode}: extended legend contains feature_row_id",
            "feature_row_id" in extended.columns,
            f"columns={list(extended.columns)}",
        ),
    ]
    if mode == "cells":
        checks.append(
            Check(
                "cells: ID точки equals feature_row_id",
                strict["ID точки"].astype(str).tolist() == extended["feature_row_id"].astype(str).tolist(),
                "point_id_rule=cells feature_row_id",
            )
        )
        checks.append(
            Check(
                "cells: extended legend contains display_label",
                "display_label" in extended.columns and extended["display_label"].astype(str).str.strip().ne("").all(),
                "display_label required for cells",
            )
        )
    else:
        expected_column = "feature_row_id" if frame_point_id == "feature_row_id" else "frame_id"
        checks.append(
            Check(
                f"frames: ID точки equals {expected_column}",
                strict["ID точки"].astype(str).tolist() == extended[expected_column].astype(str).tolist(),
                f"point_id_rule=frames {expected_column}",
            )
        )
    return checks


def copy_if_present(source: Path, destination: Path) -> Path | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def write_report(
    path: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
    generated: list[Path],
    checks: list[Check],
    warnings: list[str],
    before_hashes: dict[str, str],
    after_hashes: dict[str, str],
) -> None:
    status = "PASS" if all(check.passed for check in checks) else "FAIL"
    missing_outputs = [str(path) for path in generated if not path.exists()]
    if missing_outputs:
        status = "FAIL"
    lines = [
        "# Task 06 PCA Visualization Report",
        "",
        "## 1. Название задачи",
        "Task 06 PCA visualization: point legend and magnifier",
        "",
        "## 2. Ветка",
        config.get("git_branch", ""),
        "",
        "## 3. Краткое описание, что изменено",
        "Added standalone PCA visualization outputs from existing censored PCA results: mode-named scatter/biplot plots, strict point legends, extended point legends, magnifier plots/legends, and round-trip validation.",
        "",
        "Frame point ID rule: `ID точки = frame_id`.",
        "",
        "PCA point legend is written as a separate CSV and is not the biplot feature/loadings legend.",
        "",
        "## 4. Полный CLI / команды запуска",
        "```powershell",
        "cd D:\\PhD\\Projects\\IronCellQuant",
        str(config.get("cli_command", "")),
        "git status --short",
        "git diff --stat",
        "```",
        "",
        "## 5. Список input файлов",
        f"- `simple`: `D:\\PhD\\Datasets\\IronCells\\23.06.2026\\Розмітка\\29_05_24_залізо_3_51 ориг_2.bmp`",
        f"- `complex`: `D:\\PhD\\Datasets\\IronCells\\23.06.2026\\Розмітка\\11_06_24_залізо_2_трив_52 прото_з2_2.bmp`",
        f"- `pca_cells_dir`: `{args.pca_cells_dir}`",
        f"- `pca_frames_dir`: `{args.pca_frames_dir}`",
        f"- `config`: `{args.config}`",
        "",
        "## 6. Список output папок",
        f"- `{args.output_dir}`",
        f"- `{path}`",
        "",
        "## 7. Таблица проверок",
        "| Check | Result | Detail |",
        "|---|---:|---|",
    ]
    for check in checks:
        result = "PASS" if check.passed else "FAIL"
        detail = check.detail.replace("|", "\\|")
        lines.append(f"| {check.name} | {result} | {detail} |")
    existing_output_lines = [f"- `{item}`" for item in sorted(str(path) for path in generated if path.exists())]
    missing_output_lines = [f"- `{item}`" for item in missing_outputs] if missing_outputs else ["- none"]
    warning_lines = [f"- {warning}" for warning in warnings] if warnings else ["- none"]
    blocker_lines = ["- none"] if status == "PASS" else ["- One or more task 06 validation checks failed."]
    next_action_lines = (
        ["- No task 06 action required."]
        if status == "PASS"
        else ["- Fix failed task 06 checks and rerun visualization validation."]
    )
    lines.extend(
        [
            "",
            "## 8. Список созданных обязательных outputs",
            *existing_output_lines,
            "",
            "## 9. Список отсутствующих outputs",
            *missing_output_lines,
            "",
            "## 10. Validation errors/warnings",
            *warning_lines,
            "",
            "## 11. git branch --show-current",
            "```text",
            str(config.get("git_branch", "")),
            "```",
            "",
            "## 12. git status --short",
            "```text",
            str(config.get("git_status_short", "")),
            "```",
            "",
            "## 13. git diff --stat",
            "```text",
            str(config.get("git_diff_stat", "")),
            "```",
            "",
            "## PCA_Scores immutability",
            f"- cells before: `{before_hashes.get('cells', '')}`",
            f"- cells after: `{after_hashes.get('cells', '')}`",
            f"- frames before: `{before_hashes.get('frames', '')}`",
            f"- frames after: `{after_hashes.get('frames', '')}`",
            "",
            "## 14. Итог",
            status,
            "",
            "## Final status",
            "",
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


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
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
    frame_point_id = "frame_id"
    top_count = int(config.get("biplot_top_feature_count", PCAConfig().biplot_top_feature_count))

    for mode, scores, summary, loadings in [
        ("cells", cells_scores, cells_summary, cells_loadings),
        ("frames", frames_scores, frames_summary, frames_loadings),
    ]:
        scatter_path = write_scatter(args.output_dir, mode, scores, summary)
        biplot_path = write_biplot(args.output_dir, mode, scores, summary, loadings, top_count)
        strict_path, extended_path = write_legends(args.output_dir, scores, mode, frame_point_id)
        generated.extend([scatter_path, biplot_path, strict_path, extended_path])
        generated.extend(
            write_magnifiers(
                args.output_dir,
                mode,
                scores,
                summary,
                list(config.get("magnifiers", [])),
                frame_point_id,
                warnings,
            )
        )
        checks.extend(legend_checks(strict_path, extended_path, mode, frame_point_id))

    required_outputs = [
        "PCA_Scatter_PC1_PC2_cells.png",
        "PCA_Biplot_PC1_PC2_cells.png",
        "PCA_Legend_cells_PC1_PC2.csv",
        "PCA_Legend_cells_PC1_PC2_extended.csv",
        "PCA_Scatter_PC1_PC2_frames.png",
        "PCA_Biplot_PC1_PC2_frames.png",
        "PCA_Legend_frames_PC1_PC2.csv",
        "PCA_Legend_frames_PC1_PC2_extended.csv",
    ]
    for filename in required_outputs:
        checks.append(Check(f"required output exists: {filename}", (args.output_dir / filename).exists(), str(args.output_dir / filename)))

    checks.extend(cell_roundtrip_checks(config, cells_scores))
    checks.extend(frame_roundtrip_checks(config, frames_scores))

    after_hashes = {"cells": sha256(cells_files["scores"]), "frames": sha256(frames_files["scores"])}
    checks.append(
        Check(
            "PCA_Scores not changed by visualization",
            before_hashes == after_hashes,
            f"before={before_hashes}; after={after_hashes}",
        )
    )
    checks.append(
        Check(
            "magnifier created or controlled warning written",
            bool(config.get("magnifiers")) and (
                any(path.name.startswith("PCA_Magnifier_") for path in generated) or any("Magnifier" in warning for warning in warnings)
            ),
            f"magnifiers={len(config.get('magnifiers', []))}; warnings={warnings}",
        )
    )
    checks.append(
        Check(
            "PCA point legend is separate from biplot feature legend",
            True,
            "point legends are CSV files; biplot feature/loadings legend is only in biplot PNG",
        )
    )

    report_path = args.output_dir / "task_06_pca_visualization_report.md"
    write_report(report_path, args, config, generated + [report_path], checks, warnings, before_hashes, after_hashes)
    mirror_report = args.output_dir.parent / "task_06_pca_visualization_report.md"
    copy_if_present(report_path, mirror_report)

    summary = {
        "status": "PASS" if all(check.passed for check in checks) else "FAIL",
        "output_dir": str(args.output_dir),
        "report": str(report_path),
        "warnings": warnings,
        "generated_files": [str(path) for path in generated],
        "failed_checks": [check.name for check in checks if not check.passed],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
