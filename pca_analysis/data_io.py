"""CSV loading and text report writing helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


def read_csv_header(input_path: str | Path, delimiter: str) -> list[str]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        return next(reader, [])


def load_feature_matrix(input_path: str | Path, delimiter: str) -> pd.DataFrame:
    return pd.read_csv(input_path, delimiter=delimiter)


def ensure_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_text_report(path: str | Path, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")
