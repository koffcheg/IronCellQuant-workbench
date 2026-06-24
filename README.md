# IronCellQuant

Fiji/ImageJ headless MVP for preliminary quantification of iron-staining in microscopy images.

This project is intentionally small and conservative:

- Fiji/ImageJ macro code performs the image processing.
- Python is only a runner/wrapper for launching Fiji, passing parameters, creating clean output folders, checking outputs, and writing run metadata.
- The project does **not** estimate calibrated material concentration.
- The current target feature is a preliminary blue-pixel optical metric:

```text
blue_pixel_fraction = blue_pixels / object_pixels
blue_pixel_percent = 100 * blue_pixel_fraction
```

## Current Status

The active pipeline is still an MVP/work in progress.

Current code has been cleaned so the normal run saves only essential artifacts:

- one final visual overlay: `final_analysis_overlay.tif`
- one review bounding-box overlay: `review_detection_overlay.tif`
- component/object CSV files
- image summary CSV and PCA-ready frame feature CSV
- blue-pixel CSV and review XLSX
- run parameters
- Fiji stdout/stderr
- macro log

Intermediate masks and debug overlays are intentionally not saved by default because the source frames are large, e.g. 4000x3000 pixels.

Current validation status should be checked on each workstation with the smoke-test commands below because raw microscopy inputs and Fiji are local-only and not committed.

## Output Contract

A successful single-frame run writes the existing baseline artifacts and these point-4 review/export artifacts:

- `review_detection_overlay.tif`
- `review_detection_overlay_preview.jpg`
- `blue_pixels_features.xlsx`
- `frame_features_for_pca.csv`

The XLSX is a human-facing sorted copy of `blue_pixels_features.csv` with a final SUM row. The PCA-ready CSV is a one-row-per-run feature table; it does not perform PCA.

## Repository Layout

```text
IronCells_MVP/
  macros/
    Main_IronCells_headless.ijm
  run_one_fiji_headless.py
  README.md
  .gitignore
```

Local-only data folders are ignored by Git:

```text
input/
output/
```

## Requirements

- Windows
- Fiji/ImageJ installed locally
- Python 3.10+
- Python package: `openpyxl`

Pass the Fiji launcher path with `--fiji` or set `FIJI_PATH`. The runner also checks for `Fiji` / `Fiji.app` adjacent to or inside the project folder.

## Running One Image

From the project folder:

```powershell
cd <repo>
python run_one_fiji_headless.py --fiji <path-to-fiji-launcher>
```

Or explicitly:

```powershell
python .\run_one_fiji_headless.py --timeout-seconds 300
```

The runner automatically finds a default test input in `input\52*.bmp`.

You can also pass an explicit image:

```powershell
python .\run_one_fiji_headless.py --input "C:\path\to\image.bmp" --timeout-seconds 300
```

Each run writes to a fresh folder:

```text
output\single_test_YYYYMMDD_HHMMSS
```

The runner considers a run successful only if:

- Fiji returns exit code `0`;
- every required output exists;
- required outputs were written during the current run, not left over from an older run.

## Main Parameters

Segmentation:

- `threshold_method`
- `threshold_mode`
- `background_rolling`
- `median_radius`
- `contrast_saturated`
- `morph_open_iterations`
- `morph_close_iterations`
- `fill_holes`
- `metadata_bar_height`

Particle extraction and classification:

- `particle_extract_min_area`
- `particle_extract_max_area`
- `min_noise_area`
- `min_single_cell_area`
- `max_single_cell_area`
- `min_aggregate_area`
- `max_aggregate_area`
- `max_single_cell_aspect`
- `max_aggregate_aspect`
- `exclude_border_objects`
- `border_margin_px`

Blue-pixel rule:

- `blue_min`
- `blue_over_red`
- `blue_over_green`

Default rule:

```text
B > blue_min
B > R + blue_over_red
B > G + blue_over_green
```

## Macro Method

The Fiji macro is structured as:

1. Open original RGB image.
2. Split original RGB channels for color measurements.
3. Build grayscale segmentation image.
4. Threshold and morphologically clean a cell-material mask.
5. Run connected-component extraction with `Analyze Particles`.
6. Classify components by area, aspect ratio, and border/metadata contact.
7. Reconstruct each component ROI with `doWand()`.
8. Compare reconstructed ROI area against `Analyze Particles` area.
9. Count blue pixels only inside the real ROI using `selectionContains(x, y)`.
10. Write object-level and image-level CSV files.
11. Save one final overlay for visual inspection.

Bounding boxes are used only to limit loops for speed. They are not used as measurement regions.

## Notes For Future Work

Immediate next debugging target:

- avoid or optimize the slow Fiji morphology/particle stage on full 4000x3000 frames;
- consider a Fiji-native ROI/particle strategy that does not require full-frame expensive operations;
- keep output minimal until the segmentation profile is stable.

Do not run full-directory batch processing until the single-image pipeline produces a visually acceptable `final_analysis_overlay.tif`.

## PCA Analysis

The PCA module is independent from Fiji/ImageJ and from the current image-processing macro. It consumes an already prepared FeatureMatrix CSV and writes PCA tables, plots, reports, metadata, and a serialized model.

Input format:

- one row per analysed object (`object x features`);
- service columns are required: `image_name`, `object_type`, `object_id`;
- all non-service numeric columns are treated as candidate features unless excluded by config;
- the module does not estimate or report actual iron concentration.

Run with defaults:

```powershell
python .\run_pca.py --input .\output\frame_features_for_pca.csv --output .\output\pca_run
```

Run with a JSON config:

```powershell
python .\run_pca.py --input .\output\frame_features_for_pca.csv --output .\output\pca_run --config .\pca_config.json
```

Common config fields include:

```json
{
  "target_feature": "blue_pixel_percent",
  "color_feature": "blue_pixel_percent",
  "scatter_component_x": "PC1",
  "scatter_component_y": "PC2",
  "top_feature_count": 10,
  "biplot_top_feature_count": 15,
  "pca_component_count": 0
}
```

Primary output files:

- `Data_Check_Report.txt`
- `Standardized_Features.csv`
- `Feature_Preprocessing_Report.txt`
- `PCA_Summary.csv`
- `PCA_Loadings.csv`
- `PCA_Scores.csv`
- `PCA_TopFeatures.csv`
- `PCA_Correlation_With_BluePixel.csv`
- `PCA_Scatter_PC1_PC2.png`
- `PCA_Biplot_PC1_PC2.png`
- `PCA_ExplainedVariance.png`
- `PCA_Report.txt`
- `PCA_Run_Metadata.json`
- `PCA_Model.joblib`

`PCA_Scores.csv` stores object coordinates in principal-component space. `PCA_Loadings.csv` stores feature loadings used to interpret components. `PCA_TopFeatures.csv` ranks the strongest contributors per component. `PCA_Correlation_With_BluePixel.csv` summarizes statistical association with `blue_pixel_percent`; it is not a calibrated measurement of iron concentration.
