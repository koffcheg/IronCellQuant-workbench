"""CLI entry point for independent PCA preprocessing and analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pca_analysis.config import load_config, validate_config
from pca_analysis.analysis import run_pca_analysis
from pca_analysis.data_io import ensure_output_dir
from pca_analysis.preprocessing import preprocess_features
from pca_analysis.validation import validate_input


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate, preprocess, and run PCA for a FeatureMatrix CSV.")
    parser.add_argument("--input", required=True, help="Path to FeatureMatrix CSV.")
    parser.add_argument("--output", required=True, help="Output directory for reports and standardized features.")
    parser.add_argument("--config", help="Optional JSON config path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = ensure_output_dir(args.output)

    try:
        config = load_config(args.config)
        config_errors = validate_config(config)
        if config_errors:
            print("Configuration error:", file=sys.stderr)
            for error in config_errors:
                print(f"- {error}", file=sys.stderr)
            return 2
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    validation_result = validate_input(Path(args.input), output_dir, config)
    if not validation_result.can_continue or validation_result.dataframe is None:
        print(f"Validation failed. See: {output_dir / 'Data_Check_Report.txt'}", file=sys.stderr)
        return 1

    preprocessing_result = preprocess_features(validation_result.dataframe, output_dir, config)
    if not preprocessing_result.succeeded:
        print(f"Preprocessing failed. See: {output_dir / 'Feature_Preprocessing_Report.txt'}", file=sys.stderr)
        return 1

    analysis_result = run_pca_analysis(
        Path(args.input),
        output_dir,
        config,
        validation_result,
        preprocessing_result,
    )
    if not analysis_result.succeeded:
        for error in analysis_result.errors:
            print(f"PCA analysis error: {error}", file=sys.stderr)
        return 1

    print(f"PCA analysis completed. Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
