# IronCellQuant

Fiji/ImageJ headless MVP for preliminary quantification of iron-staining in microscopy images.

This project is intentionally small and conservative:

- Fiji/ImageJ macro code performs the image processing.
- Python is only a runner/wrapper for launching Fiji, passing parameters, creating clean output folders, checking outputs, and writing run metadata.
- The project does **not** estimate absolute iron concentration.
- The current target feature is a preliminary relative optical metric:

```text
blue_pixel_fraction = blue_pixels / object_pixels
blue_pixel_percent = 100 * blue_pixel_fraction
```

## Current Status

The active pipeline is still an MVP/work in progress.

Current code has been cleaned so the normal run saves only essential artifacts:

- one final visual overlay: `final_analysis_overlay.tif`
- component/object CSV files
- image summary CSV
- run parameters
- Fiji stdout/stderr
- macro log

Intermediate masks and debug overlays are intentionally not saved by default because the source frames are large, e.g. 4000x3000 pixels.

Latest local test status:

- Fiji launches correctly through `fiji.bat`.
- Python creates a fresh output directory and rejects stale/incomplete outputs.
- The macro opens the test image, builds the segmentation grayscale image, thresholds it, and reaches morphology.
- The last test timed out after `morph_close`; final CSV/overlay output was not yet produced.

So this repository currently captures the cleaned MVP codebase and the next debugging point, not a validated final analysis method.

## Repository Layout

```text
IronCells_MVP/
  macros/
    Main_IronCells_headless.ijm
  run_one_fiji_headless.py
  run_one_fiji_headless.bat
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

Current local Fiji runner path:

```text
C:\PERSONAL\ImageJ\Fiji\fiji.bat
```

If Fiji is installed elsewhere, pass `--fiji` to the runner.

## Running One Image

From the project folder:

```powershell
cd C:\PERSONAL\ImageJ\IronCells_MVP
.\run_one_fiji_headless.bat
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
