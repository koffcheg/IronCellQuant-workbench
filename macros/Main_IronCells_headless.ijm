// Main_IronCells_headless.ijm
// Fiji/ImageJ headless MVP for preliminary relative iron-staining quantification.
// Python only launches Fiji, passes parameters, validates outputs, and stores run metadata.

requires("1.53");

arg = getArgument();
inputPath = getArgString(arg, "input", "");
outputDir = getArgString(arg, "output", "");
if (inputPath == "") exit("Missing input=...");
if (outputDir == "") exit("Missing output=...");

originalLongPath = getArgString(arg, "original_long_path", inputPath);
originalFileName = getArgString(arg, "original_file_name", "");
groupName = getArgString(arg, "group_name", "unknown");
shortPathUsed = getArgString(arg, "short_path_used", "");

thresholdMethod = getArgString(arg, "threshold_method", "Li");
thresholdMode = getArgString(arg, "threshold_mode", "dark");
backgroundRolling = parseFloat(getArgString(arg, "background_rolling", "80"));
medianRadius = parseFloat(getArgString(arg, "median_radius", "2"));
contrastSaturated = parseFloat(getArgString(arg, "contrast_saturated", "0.35"));
morphOpenIterations = parseFloat(getArgString(arg, "morph_open_iterations", "0"));
morphCloseIterations = parseFloat(getArgString(arg, "morph_close_iterations", "1"));
fillHoles = getArgBool(arg, "fill_holes", 1);
metadataBarHeight = parseFloat(getArgString(arg, "metadata_bar_height", "120"));

particleExtractMinArea = parseFloat(getArgString(arg, "particle_extract_min_area", "10"));
particleExtractMaxArea = parseFloat(getArgString(arg, "particle_extract_max_area", "2000000"));
minNoiseArea = parseFloat(getArgString(arg, "min_noise_area", "10"));
minSingleCellArea = parseFloat(getArgString(arg, "min_single_cell_area", "40"));
maxSingleCellArea = parseFloat(getArgString(arg, "max_single_cell_area", "50000"));
minAggregateArea = parseFloat(getArgString(arg, "min_aggregate_area", "50000"));
maxAggregateArea = parseFloat(getArgString(arg, "max_aggregate_area", "2000000"));
maxSingleCellAspect = parseFloat(getArgString(arg, "max_single_cell_aspect", "10"));
maxAggregateAspect = parseFloat(getArgString(arg, "max_aggregate_aspect", "30"));
excludeBorderObjects = getArgBool(arg, "exclude_border_objects", 1);
borderMarginPx = parseFloat(getArgString(arg, "border_margin_px", "2"));

blueMin = parseFloat(getArgString(arg, "blue_min", "120"));
blueOverRed = parseFloat(getArgString(arg, "blue_over_red", "20"));
blueOverGreen = parseFloat(getArgString(arg, "blue_over_green", "10"));

saveOverlays = getArgBool(arg, "save_overlays", 1);
labelObjects = getArgBool(arg, "label_objects", 1);
drawRejectedObjects = getArgBool(arg, "draw_rejected_objects", 0);
contourWidth = parseFloat(getArgString(arg, "contour_width", "6"));
previewMaxSize = parseFloat(getArgString(arg, "final_overlay_preview_max_size", "1600"));
minExpectedAcceptedObjects = parseFloat(getArgString(arg, "min_expected_accepted_objects", "1"));
minStableAcceptedObjects = parseFloat(getArgString(arg, "min_stable_accepted_objects", "3"));
minStableAcceptedPixels = parseFloat(getArgString(arg, "min_stable_accepted_pixels", "500"));

File.makeDirectory(outputDir);
logPath = outputDir + "/macro_log.txt";
File.saveString("IronCells Fiji headless macro\n", logPath);
checkpoint("start");
logLine("Input(short/Fiji): " + inputPath);
logLine("Input(original): " + originalLongPath);
logLine("Output: " + outputDir);
logLine("Group: " + groupName);
logLine("Feature definition: blue_pixel_fraction = blue_pixels / object_pixels. This is a preliminary blue-pixel optical feature, not a calibrated concentration measurement.");

setBatchMode(true);
run("Clear Results");

checkpoint("before_open_input");
open(inputPath);
checkpoint("after_open_input");
originalTitle = getTitle();
if (originalFileName == "") originalFileName = originalTitle;
width = getWidth();
height = getHeight();
framePixels = width * height;
logLine("Opened image: " + originalTitle + " width=" + width + " height=" + height + " bitDepth=" + bitDepth());

selectWindow(originalTitle);
rename("Original_RGB");
originalTitle = "Original_RGB";
checkpoint("prepared_original_rgb_window");

checkpoint("before_split_rgb_channels");
selectWindow("Original_RGB");
run("Duplicate...", "title=Gray_Channel");
run("8-bit");
selectWindow("Original_RGB");
run("Duplicate...", "title=RGB_For_Channels");
run("Split Channels");
titles = getList("image.titles");
redTitle = ""; greenTitle = ""; blueTitle = "";
for (ti = 0; ti < titles.length; ti++) {
    t = titles[ti];
    if (startsWith(t, "C1-")) redTitle = t;
    if (redTitle == "") {
        if (indexOf(t, "red") >= 0) redTitle = t;
        if (indexOf(t, "Red") >= 0) redTitle = t;
    }
    if (startsWith(t, "C2-")) greenTitle = t;
    if (greenTitle == "") {
        if (indexOf(t, "green") >= 0) greenTitle = t;
        if (indexOf(t, "Green") >= 0) greenTitle = t;
    }
    if (startsWith(t, "C3-")) blueTitle = t;
    if (blueTitle == "") {
        if (indexOf(t, "blue") >= 0) blueTitle = t;
        if (indexOf(t, "Blue") >= 0) blueTitle = t;
    }
}
if (redTitle == "") {
    if (titles.length >= 3) redTitle = titles[titles.length - 3];
}
if (greenTitle == "") {
    if (titles.length >= 2) greenTitle = titles[titles.length - 2];
}
if (blueTitle == "") {
    if (titles.length >= 1) blueTitle = titles[titles.length - 1];
}
selectWindow(redTitle); rename("Red_Channel"); redTitle = "Red_Channel";
selectWindow(greenTitle); rename("Green_Channel"); greenTitle = "Green_Channel";
selectWindow(blueTitle); rename("Blue_Channel"); blueTitle = "Blue_Channel";
checkpoint("after_split_rgb_channels");

checkpoint("before_prepare_segmentation_gray");
selectWindow("Original_RGB");
run("Duplicate...", "title=Segmentation_Gray");
run("8-bit");
if (backgroundRolling > 0) run("Subtract Background...", "rolling=" + backgroundRolling);
if (medianRadius > 0) run("Median...", "radius=" + medianRadius);
if (contrastSaturated >= 0) run("Enhance Contrast...", "saturated=" + contrastSaturated + " normalize");
checkpoint("after_prepare_segmentation_gray");

selectWindow("Segmentation_Gray");
checkpoint("before_set_auto_threshold");
setAutoThreshold(thresholdMethod + " " + thresholdMode);
checkpoint("after_set_auto_threshold");
run("Convert to Mask");
checkpoint("after_convert_segmentation_to_mask");
cellMaskTitle = "Segmentation_Gray";

selectWindow(cellMaskTitle);
if (morphOpenIterations > 0) {
    checkpoint("before_morph_open");
    run("Options...", "iterations=" + morphOpenIterations + " count=1 black do=Open");
    checkpoint("after_morph_open");
}
if (morphCloseIterations > 0) {
    checkpoint("before_morph_close");
    run("Options...", "iterations=" + morphCloseIterations + " count=1 black do=Close");
    checkpoint("after_morph_close");
}
if (fillHoles == 1) {
    checkpoint("before_fill_holes");
    run("Fill Holes");
    checkpoint("after_fill_holes");
}
checkpoint("after_morphology_block");

checkpoint("before_blue_mask_creation");
selectWindow(blueTitle);
run("Duplicate...", "title=Blue_Min_Mask");
setThreshold(blueMin + 1, 255);
run("Convert to Mask");
imageCalculator("Subtract create", blueTitle, redTitle);
rename("Blue_Minus_R");
setThreshold(blueOverRed + 1, 255);
run("Convert to Mask");
imageCalculator("Subtract create", blueTitle, greenTitle);
rename("Blue_Minus_G");
setThreshold(blueOverGreen + 1, 255);
run("Convert to Mask");
imageCalculator("AND create", "Blue_Min_Mask", "Blue_Minus_R");
rename("Blue_Tmp_1");
imageCalculator("AND create", "Blue_Tmp_1", "Blue_Minus_G");
rename("Blue_Candidate_Mask");
imageCalculator("AND create", "Blue_Candidate_Mask", cellMaskTitle);
rename("Blue_Pixels_Mask");
totalBluePixelsInMask = whiteCount("Blue_Pixels_Mask");
checkpoint("after_blue_mask_creation total_blue_pixels_in_cleaned_mask=" + totalBluePixelsInMask);

checkpoint("before_analyze_particles");
selectWindow(cellMaskTitle);
run("Clear Results");
run("Set Measurements...", "area centroid bounding fit shape mean redirect=None decimal=3");
run("Analyze Particles...", "size=1-" + particleExtractMaxArea + " circularity=0.00-1.00 display clear");
nObjects = nResults;
checkpoint("after_analyze_particles component_count=" + nObjects);

checkpoint("before_results_table_read");
areas = newArray(nObjects); xs = newArray(nObjects); ys = newArray(nObjects);
bxs = newArray(nObjects); bys = newArray(nObjects); bws = newArray(nObjects); bhs = newArray(nObjects);
for (i = 0; i < nObjects; i++) {
    areas[i] = getResult("Area", i);
    xs[i] = getResult("X", i);
    ys[i] = getResult("Y", i);
    bxs[i] = getResult("BX", i);
    bys[i] = getResult("BY", i);
    bws[i] = getResult("Width", i);
    bhs[i] = getResult("Height", i);
}
checkpoint("after_results_table_read");

whiteAfterMorphology = 0;
for (i = 0; i < nObjects; i++) whiteAfterMorphology += areas[i];
if (whiteAfterMorphology > 0) maskBlueFraction = totalBluePixelsInMask / whiteAfterMorphology; else maskBlueFraction = 0;

componentsGe1 = 0; componentsGe5 = 0; componentsGe10 = 0; componentsGe20 = 0;
componentsGe50 = 0; componentsGe100 = 0; componentsGe200 = 0; componentsGe500 = 0;
singleCount = 0; aggregateCount = 0; tooSmallCount = 0; smallFragmentCount = 0;
tooLargeCount = 0; tooLongCount = 0; borderCount = 0; roiWarningCount = 0; acceptedCount = 0;
totalAcceptedObjectPixels = 0; totalAcceptedBluePixels = 0;
acceptedRSum = 0; acceptedGSum = 0; acceptedBSum = 0;
acceptedRSqSum = 0; acceptedGSqSum = 0; acceptedBSqSum = 0;
acceptedBOverRSum = 0; acceptedBOverRGBSumSum = 0; acceptedGraySum = 0; acceptedGraySqSum = 0;

allCsv = outputDir + "/all_components_before_filter.csv";
File.saveString("component_id,area_px,centroid_x,centroid_y,bbox_x,bbox_y,bbox_width,bbox_height,aspect_ratio,touches_border,classification,accepted_for_summary,reject_reason\n", allCsv);
rejectedCsv = outputDir + "/rejected_objects.csv";
File.saveString("image_name,group_name,object_id,classification,reject_reason,area_px,bbox_x,bbox_y,bbox_width,bbox_height,centroid_x,centroid_y,aspect_ratio,touches_border,roi_reconstruction_status\n", rejectedCsv);
objectCsv = outputDir + "/final_object_report.csv";
File.saveString("image_name,group_name,original_long_path,short_path_used,object_id,object_type,object_pixels,blue_pixels,blue_pixel_fraction,blue_pixel_percent,area_px,bbox_x,bbox_y,bbox_width,bbox_height,centroid_x,centroid_y,aspect_ratio,mean_R,mean_G,mean_B,R_mean,G_mean,B_mean,R_std,G_std,B_std,R_min,G_min,B_min,R_max,G_max,B_max,R_over_G,B_over_R,B_over_RGB_sum,gray_mean,gray_stddev,roi_area_from_particles,roi_area_reconstructed,roi_area_delta_percent,roi_reconstruction_status\n", objectCsv);
blueCsv = outputDir + "/blue_pixels_features.csv";
File.saveString("image_name,group_name,object_type,object_id,object_pixels,blue_pixels,blue_pixel_fraction,blue_pixel_percent\n", blueCsv);

if (saveOverlays == 1) {
    checkpoint("before_overlay_setup");
    selectWindow("Original_RGB");
    run("Duplicate...", "title=Overlay_Combined");
    selectWindow("Original_RGB");
    run("Duplicate...", "title=Review_Detection_Overlay");
    newImage("Accepted_Objects_Mask", "8-bit black", width, height, 1);
    checkpoint("after_overlay_setup");
}

checkpoint("before_per_object_loop");
for (i = 0; i < nObjects; i++) {
    area = areas[i];
    aspect = maxOf(bws[i], bhs[i]) / maxOf(1, minOf(bws[i], bhs[i]));
    touchesBorder = objectTouchesBorder(bxs[i], bys[i], bws[i], bhs[i]);
    classification = classifyObject(area, aspect, touchesBorder);
    rejectReason = rejectReasonFor(classification, area);
    accepted = isAcceptedClass(classification);

    if (area >= 1) componentsGe1++;
    if (area >= 5) componentsGe5++;
    if (area >= 10) componentsGe10++;
    if (area >= 20) componentsGe20++;
    if (area >= 50) componentsGe50++;
    if (area >= 100) componentsGe100++;
    if (area >= 200) componentsGe200++;
    if (area >= 500) componentsGe500++;
    if (classification == "too_small_noise") tooSmallCount++;
    if (classification == "small_cell_or_fragment") smallFragmentCount++;
    if (classification == "single_cell_candidate") singleCount++;
    if (classification == "aggregate_candidate") aggregateCount++;
    if (classification == "too_large_artifact") tooLargeCount++;
    if (classification == "too_long_artifact") tooLongCount++;
    if (classification == "border_object") borderCount++;

    File.append((i+1) + "," + d2s(area,0) + "," + d2s(xs[i],2) + "," + d2s(ys[i],2) + "," + d2s(bxs[i],0) + "," + d2s(bys[i],0) + "," + d2s(bws[i],0) + "," + d2s(bhs[i],0) + "," + d2s(aspect,4) + "," + boolText(touchesBorder) + "," + classification + "," + boolText(accepted) + "," + rejectReason + "\n", allCsv);

    roiStatus = "not_attempted"; roiArea = 0; roiDelta = 100; meanR = 0; meanG = 0; meanB = 0; stdR = 0; stdG = 0; stdB = 0; minR = 0; minG = 0; minB = 0; maxR = 0; maxG = 0; maxB = 0; grayMean = 0; grayStd = 0; objectPixels = 0; bluePixels = 0;
    selectWindow(cellMaskTitle);
    recoverObjectRoi(xs[i], ys[i], bxs[i], bys[i], bws[i], bhs[i]);
    if (selectionType() == -1) {
        roiStatus = "failed_no_selection";
        roiWarningCount++;
        accepted = 0;
        if (isAcceptedClass(classification) == 1) classification = "roi_reconstruction_warning";
        rejectReason = "roi_reconstruction_failed";
        logLine("WARNING object_id=" + (i+1) + " doWand ROI reconstruction failed");
    } else {
        getStatistics(roiArea, roiMean);
        roiDelta = abs(roiArea - area) / maxOf(1, area) * 100;
        if (roiDelta > 10) {
            roiStatus = "warning_area_delta_gt_10pct";
            roiWarningCount++;
            accepted = 0;
            if (isAcceptedClass(classification) == 1) classification = "roi_reconstruction_warning";
            rejectReason = "roi_area_delta_gt_10pct";
        } else {
            roiStatus = "ok";
        }
        selectWindow(redTitle); run("Restore Selection"); getStatistics(areaR, meanR, minR, maxR, stdR);
        selectWindow(greenTitle); run("Restore Selection"); getStatistics(areaG, meanG, minG, maxG, stdG);
        selectWindow(blueTitle); run("Restore Selection"); getStatistics(areaB, meanB, minB, maxB, stdB);
        selectWindow("Gray_Channel"); run("Restore Selection"); getStatistics(areaGray, grayMean, grayMin, grayMax, grayStd);
        selectWindow("Blue_Pixels_Mask"); run("Restore Selection");
        counts = countRoiPixelsInBbox(bxs[i], bys[i], bws[i], bhs[i]);
        objectPixels = counts[0];
        bluePixels = counts[1];
    }

    if (objectPixels > 0) fraction = bluePixels / objectPixels; else fraction = 0;
    epsilon = 0.000001;
    rOverG = meanR / maxOf(epsilon, meanG);
    bOverR = meanB / maxOf(epsilon, meanR);
    bOverRgbSum = meanB / maxOf(epsilon, meanR + meanG + meanB);

    if (accepted == 1) {
        acceptedCount++;
        totalAcceptedObjectPixels += objectPixels;
        totalAcceptedBluePixels += bluePixels;
        acceptedRSum += meanR * objectPixels; acceptedGSum += meanG * objectPixels; acceptedBSum += meanB * objectPixels;
        acceptedRSqSum += (stdR * stdR + meanR * meanR) * objectPixels;
        acceptedGSqSum += (stdG * stdG + meanG * meanG) * objectPixels;
        acceptedBSqSum += (stdB * stdB + meanB * meanB) * objectPixels;
        acceptedBOverRSum += bOverR * objectPixels;
        acceptedBOverRGBSumSum += bOverRgbSum * objectPixels;
        acceptedGraySum += grayMean * objectPixels;
        acceptedGraySqSum += (grayStd * grayStd + grayMean * grayMean) * objectPixels;
        objectType = classification;
        File.append(originalFileName + "," + groupName + "," + originalLongPath + "," + shortPathUsed + "," + (i+1) + "," + objectType + "," + d2s(objectPixels,0) + "," + d2s(bluePixels,0) + "," + d2s(fraction,8) + "," + d2s(100*fraction,4) + "," + d2s(area,2) + "," + d2s(bxs[i],0) + "," + d2s(bys[i],0) + "," + d2s(bws[i],0) + "," + d2s(bhs[i],0) + "," + d2s(xs[i],2) + "," + d2s(ys[i],2) + "," + d2s(aspect,4) + "," + d2s(meanR,3) + "," + d2s(meanG,3) + "," + d2s(meanB,3) + "," + d2s(meanR,3) + "," + d2s(meanG,3) + "," + d2s(meanB,3) + "," + d2s(stdR,3) + "," + d2s(stdG,3) + "," + d2s(stdB,3) + "," + d2s(minR,0) + "," + d2s(minG,0) + "," + d2s(minB,0) + "," + d2s(maxR,0) + "," + d2s(maxG,0) + "," + d2s(maxB,0) + "," + d2s(rOverG,6) + "," + d2s(bOverR,6) + "," + d2s(bOverRgbSum,6) + "," + d2s(grayMean,3) + "," + d2s(grayStd,3) + "," + d2s(area,2) + "," + d2s(roiArea,2) + "," + d2s(roiDelta,3) + "," + roiStatus + "\n", objectCsv);
        File.append(originalFileName + "," + groupName + "," + objectType + "," + (i+1) + "," + d2s(objectPixels,0) + "," + d2s(bluePixels,0) + "," + d2s(fraction,8) + "," + d2s(100*fraction,4) + "\n", blueCsv);
        if (saveOverlays == 1) {
            if (selectionType() != -1) {
                selectWindow("Accepted_Objects_Mask");
                run("Restore Selection");
                setForegroundColor(255,255,255);
                run("Fill", "slice");
            }
        }
    } else {
        File.append(originalFileName + "," + groupName + "," + (i+1) + "," + classification + "," + rejectReason + "," + d2s(area,2) + "," + d2s(bxs[i],0) + "," + d2s(bys[i],0) + "," + d2s(bws[i],0) + "," + d2s(bhs[i],0) + "," + d2s(xs[i],2) + "," + d2s(ys[i],2) + "," + d2s(aspect,4) + "," + boolText(touchesBorder) + "," + roiStatus + "\n", rejectedCsv);
    }

    if (saveOverlays == 1) {
        if (selectionType() != -1) {
            shouldDrawObject = accepted;
            if (drawRejectedObjects == 1) shouldDrawObject = 1;
            if (shouldDrawObject == 1) {
                drawObjectOverlay(i+1, classification, area, accepted, fraction, bxs[i], bys[i]);
                if (accepted == 1) drawReviewObjectOverlay(i+1, classification, fraction, bxs[i], bys[i], bws[i], bhs[i]);
            }
        }
    }
}
checkpoint("after_per_object_loop");

if (totalAcceptedObjectPixels > 0) {
    acceptedFraction = totalAcceptedBluePixels / totalAcceptedObjectPixels;
    acceptedRMean = acceptedRSum / totalAcceptedObjectPixels; acceptedGMean = acceptedGSum / totalAcceptedObjectPixels; acceptedBMean = acceptedBSum / totalAcceptedObjectPixels;
    acceptedRStd = sqrt(maxOf(0, acceptedRSqSum / totalAcceptedObjectPixels - acceptedRMean * acceptedRMean));
    acceptedGStd = sqrt(maxOf(0, acceptedGSqSum / totalAcceptedObjectPixels - acceptedGMean * acceptedGMean));
    acceptedBStd = sqrt(maxOf(0, acceptedBSqSum / totalAcceptedObjectPixels - acceptedBMean * acceptedBMean));
    acceptedBOverRMean = acceptedBOverRSum / totalAcceptedObjectPixels;
    acceptedBOverRGBSumMean = acceptedBOverRGBSumSum / totalAcceptedObjectPixels;
    acceptedGrayMean = acceptedGraySum / totalAcceptedObjectPixels;
    acceptedGrayVariance = acceptedGraySqSum / totalAcceptedObjectPixels - acceptedGrayMean * acceptedGrayMean;
    acceptedGrayStddev = sqrt(maxOf(0, acceptedGrayVariance));
} else {
    acceptedFraction = 0; acceptedRMean = 0; acceptedGMean = 0; acceptedBMean = 0; acceptedRStd = 0; acceptedGStd = 0; acceptedBStd = 0; acceptedBOverRMean = 0; acceptedBOverRGBSumMean = 0; acceptedGrayMean = 0; acceptedGrayStddev = 0;
}
acceptedAreaFractionOfFrame = totalAcceptedObjectPixels / framePixels;
acceptedAreaPercentOfFrame = 100 * acceptedAreaFractionOfFrame;
File.append(originalFileName + "," + groupName + ",all_accepted_cell_material,frame," + d2s(totalAcceptedObjectPixels,0) + "," + d2s(totalAcceptedBluePixels,0) + "," + d2s(acceptedFraction,8) + "," + d2s(100*acceptedFraction,4) + "\n", blueCsv);
File.append(originalFileName + "," + groupName + ",all_cleaned_cell_material,frame," + d2s(whiteAfterMorphology,0) + "," + d2s(totalBluePixelsInMask,0) + "," + d2s(maskBlueFraction,8) + "," + d2s(100*maskBlueFraction,4) + "\n", blueCsv);

if (saveOverlays == 1) {
    checkpoint("before_final_overlay_blue_layer");
    imageCalculator("AND create", "Blue_Pixels_Mask", "Accepted_Objects_Mask");
    rename("Blue_Accepted_Mask");
    selectWindow("Blue_Accepted_Mask");
    run("Create Selection");
    if (selectionType() != -1) {
        selectWindow("Overlay_Combined");
        run("Restore Selection");
        setForegroundColor(0,80,255);
        run("Fill", "slice");
        run("Select None");
    }
    // Create the lightweight preview before saveAs(Tiff).
    // In Fiji headless, saveAs can retitle the active image to the saved filename,
    // so selecting the old window title after saveAs is not reliable.
    checkpoint("before_save_final_overlay_preview");
    selectWindow("Overlay_Combined");
    savePreviewImage();
    checkpoint("after_save_final_overlay_preview");

    checkpoint("before_save_final_overlay");
    selectWindow("Overlay_Combined");
    saveAs("Tiff", outputDir + "/final_analysis_overlay.tif");
    checkpoint("after_save_final_overlay");

    checkpoint("before_save_review_overlay_preview");
    saveReviewPreviewImage();
    checkpoint("after_save_review_overlay_preview");
    selectWindow("Review_Detection_Overlay");
    saveAs("Tiff", outputDir + "/review_detection_overlay.tif");
}

lowAcceptedAreaWarning = 0;
if (acceptedCount < minStableAcceptedObjects) lowAcceptedAreaWarning = 1;
if (totalAcceptedObjectPixels < minStableAcceptedPixels) lowAcceptedAreaWarning = 1;

qcStatus = "PASS";
if (acceptedCount < minExpectedAcceptedObjects) qcStatus = appendQcStatus(qcStatus, "WARN_LOW_ACCEPTED_OBJECT_COUNT");
if (roiWarningCount > 0) qcStatus = appendQcStatus(qcStatus, "WARN_ROI_RECONSTRUCTION");
if (lowAcceptedAreaWarning == 1) qcStatus = appendQcStatus(qcStatus, "WARN_LOW_ACCEPTED_AREA");

summaryCsv = outputDir + "/final_frame_summary.csv";
File.saveString("image_name,group_name,original_long_path,short_path_used,image_width,image_height,threshold_method,threshold_mode,background_rolling,median_radius,contrast_saturated,morph_open_iterations,morph_close_iterations,fill_holes,metadata_bar_height,frame_area_pixels,accepted_area_fraction_of_frame,accepted_area_percent_of_frame,particle_extract_min_area,particle_extract_max_area,min_noise_area,min_single_cell_area,max_single_cell_area,min_aggregate_area,max_aggregate_area,max_single_cell_aspect,max_aggregate_aspect,exclude_border_objects,border_margin_px,blue_min,blue_over_red,blue_over_green,min_stable_accepted_objects,min_stable_accepted_pixels,total_cell_material_pixels,cell_material_area_fraction,total_blue_pixels_in_cell_material,blue_pixel_fraction_all_cell_material,blue_pixel_percent_all_cell_material,component_count_total,accepted_object_count,single_cell_candidate_count,aggregate_candidate_count,too_small_noise_count,small_cell_or_fragment_count,too_large_artifact_count,too_long_artifact_count,border_object_count,roi_reconstruction_warning_count,accepted_R_mean,accepted_G_mean,accepted_B_mean,accepted_R_std,accepted_G_std,accepted_B_std,accepted_B_over_R_mean,accepted_B_over_RGB_sum_mean,accepted_gray_stddev,accepted_object_pixels,accepted_blue_pixels,blue_pixel_fraction_all_accepted,blue_pixel_percent_all_accepted,components_area_ge_1,components_area_ge_5,components_area_ge_10,components_area_ge_20,components_area_ge_50,components_area_ge_100,components_area_ge_200,components_area_ge_500,qc_status\n", summaryCsv);
summaryLine = originalFileName + "," + groupName + "," + originalLongPath + "," + shortPathUsed + "," + width + "," + height + "," + thresholdMethod + "," + thresholdMode + "," + backgroundRolling + "," + medianRadius + "," + contrastSaturated + "," + morphOpenIterations + "," + morphCloseIterations + "," + boolText(fillHoles) + "," + metadataBarHeight + "," + d2s(framePixels,0) + "," + d2s(acceptedAreaFractionOfFrame,8) + "," + d2s(acceptedAreaPercentOfFrame,4) + "," + particleExtractMinArea + "," + particleExtractMaxArea + "," + minNoiseArea + "," + minSingleCellArea + "," + maxSingleCellArea + "," + minAggregateArea + "," + maxAggregateArea + "," + maxSingleCellAspect + "," + maxAggregateAspect + "," + boolText(excludeBorderObjects) + "," + borderMarginPx + "," + blueMin + "," + blueOverRed + "," + blueOverGreen + "," + minStableAcceptedObjects + "," + minStableAcceptedPixels + "," + d2s(whiteAfterMorphology,0) + "," + d2s(whiteAfterMorphology/framePixels,8) + "," + d2s(totalBluePixelsInMask,0) + "," + d2s(maskBlueFraction,8) + "," + d2s(100*maskBlueFraction,4) + "," + nObjects + "," + acceptedCount + "," + singleCount + "," + aggregateCount + "," + tooSmallCount + "," + smallFragmentCount + "," + tooLargeCount + "," + tooLongCount + "," + borderCount + "," + roiWarningCount + "," + d2s(acceptedRMean,3) + "," + d2s(acceptedGMean,3) + "," + d2s(acceptedBMean,3) + "," + d2s(acceptedRStd,3) + "," + d2s(acceptedGStd,3) + "," + d2s(acceptedBStd,3) + "," + d2s(acceptedBOverRMean,6) + "," + d2s(acceptedBOverRGBSumMean,6) + "," + d2s(acceptedGrayStddev,3) + "," + d2s(totalAcceptedObjectPixels,0) + "," + d2s(totalAcceptedBluePixels,0) + "," + d2s(acceptedFraction,8) + "," + d2s(100*acceptedFraction,4) + "," + componentsGe1 + "," + componentsGe5 + "," + componentsGe10 + "," + componentsGe20 + "," + componentsGe50 + "," + componentsGe100 + "," + componentsGe200 + "," + componentsGe500 + "," + qcStatus + "\n";
File.append(summaryLine, summaryCsv);

writeQcReport(qcStatus);
checkpoint("after_write_summary_and_qc");
logLine("component_count_total=" + nObjects);
logLine("accepted_object_count=" + acceptedCount);
logLine("roi_reconstruction_warning_count=" + roiWarningCount);
logLine("blue_pixel_percent_all_accepted=" + d2s(100*acceptedFraction,4));
logLine("Finished successfully.");

setBatchMode(false);
run("Close All");
eval("script", "java.lang.System.exit(0);");

function recoverObjectRoi(cx, cy, bx, by, bw, bh) {
    selectWindow(cellMaskTitle);
    run("Select None");
    doWand(round(cx), round(cy));
    if (selectionType() != -1) return;
    x0 = clampFloor(bx, 0, width - 1); y0 = clampFloor(by, 0, height - 1);
    x1 = clampCeil(bx + bw, 0, width - 1); y1 = clampCeil(by + bh, 0, height - 1);
    for (yy = y0; yy <= y1; yy++) {
        for (xx = x0; xx <= x1; xx++) {
            if (getPixel(xx, yy) == 255) {
                doWand(xx, yy);
                if (selectionType() != -1) return;
            }
        }
    }
}

function countRoiPixelsInBbox(bx, by, bw, bh) {
    objectCount = 0; blueCount = 0;
    x0 = clampFloor(bx, 0, width - 1); y0 = clampFloor(by, 0, height - 1);
    x1 = clampCeil(bx + bw, 0, width - 1); y1 = clampCeil(by + bh, 0, height - 1);
    for (yy = y0; yy <= y1; yy++) {
        for (xx = x0; xx <= x1; xx++) {
            if (selectionContains(xx, yy)) {
                objectCount++;
                if (getPixel(xx, yy) == 255) blueCount++;
            }
        }
    }
    return newArray(objectCount, blueCount);
}

function drawObjectOverlay(id, classification, area, accepted, fraction, bx, by) {
    if (accepted == 1) { rr=0; gg=255; bb=0; } else { rr=255; gg=120; bb=0; }
    selectWindow("Overlay_Combined");
    run("Restore Selection");
    setLineWidth(contourWidth); setForegroundColor(rr,gg,bb); run("Draw", "slice");
    if (labelObjects == 1) {
        setFont("SansSerif", 18, "bold");
        drawString("#" + id + " " + classification + " blue=" + d2s(100*fraction,2) + "%", bx, maxOf(20, by-6));
    }
    run("Select None");
}

function drawReviewObjectOverlay(id, classification, fraction, bx, by, bw, bh) {
    selectWindow("Review_Detection_Overlay");
    rectX = clampFloor(bx, 0, width - 1);
    rectY = clampFloor(by, 0, height - 1);
    rectW = maxOf(1, round(bw));
    rectH = maxOf(1, round(bh));
    if (rectX + rectW > width) rectW = width - rectX;
    if (rectY + rectH > height) rectH = height - rectY;
    makeRectangle(rectX, rectY, rectW, rectH);
    setLineWidth(contourWidth);
    setForegroundColor(255,255,0);
    run("Draw", "slice");
    if (labelObjects == 1) {
        setFont("SansSerif", 18, "bold");
        setColor(255,255,0);
        drawString("#" + id, rectX, maxOf(20, rectY-6));
    }
    run("Select None");
}

function saveReviewPreviewImage() {
    selectWindow("Review_Detection_Overlay");
    run("Duplicate...", "title=Review_Overlay_Preview");
    needsResize = 0;
    if (previewMaxSize > 0) {
        if (width > previewMaxSize) needsResize = 1;
        if (height > previewMaxSize) needsResize = 1;
    }
    if (needsResize == 1) {
        if (width >= height) {
            newW = previewMaxSize;
            newH = round(height * previewMaxSize / width);
        } else {
            newH = previewMaxSize;
            newW = round(width * previewMaxSize / height);
        }
        run("Size...", "width=" + newW + " height=" + newH + " interpolation=Bilinear average");
    }
    saveAs("Jpeg", outputDir + "/review_detection_overlay_preview.jpg");
    close();
}

function classifyObject(area, aspect, touchesBorder) {
    if (excludeBorderObjects == 1) {
        if (touchesBorder == 1) return "border_object";
    }
    if (area < particleExtractMinArea) return "too_small_noise";
    if (area < minNoiseArea) return "too_small_noise";
    if (area < minSingleCellArea) return "small_cell_or_fragment";
    if (area > maxAggregateArea) return "too_large_artifact";
    if (area <= maxSingleCellArea) {
        if (aspect > maxSingleCellAspect) return "too_long_artifact";
        return "single_cell_candidate";
    }
    if (area >= minAggregateArea) {
        if (area <= maxAggregateArea) {
            if (aspect > maxAggregateAspect) return "too_long_artifact";
            return "aggregate_candidate";
        }
    }
    return "small_cell_or_fragment";
}

function rejectReasonFor(classification, area) {
    if (isAcceptedClass(classification) == 1) return "";
    if (classification == "too_small_noise") {
        if (area < particleExtractMinArea) return "below_particle_extract_min_area";
    }
    return classification;
}

function isAcceptedClass(classification) {
    if (classification == "single_cell_candidate") return 1;
    if (classification == "aggregate_candidate") return 1;
    return 0;
}

function objectTouchesBorder(bx, by, bw, bh) {
    if (bx <= borderMarginPx) return 1;
    if (by <= borderMarginPx) return 1;
    if (bx + bw >= width - borderMarginPx) return 1;
    if (by + bh >= height - metadataBarHeight - borderMarginPx) return 1;
    return 0;
}

function clampFloor(value, minValue, maxValue) {
    roundedValue = floor(value);
    if (roundedValue < minValue) return minValue;
    if (roundedValue > maxValue) return maxValue;
    return roundedValue;
}

function clampCeil(value, minValue, maxValue) {
    roundedValue = -floor(-value);
    if (roundedValue < minValue) return minValue;
    if (roundedValue > maxValue) return maxValue;
    return roundedValue;
}

function whiteCount(title) {
    selectWindow(title);
    run("Select None");
    getStatistics(areaValue, meanValue);
    return round(areaValue * meanValue / 255);
}

function savePreviewImage() {
    selectWindow("Overlay_Combined");
    run("Duplicate...", "title=Overlay_Preview");
    needsResize = 0;
    if (previewMaxSize > 0) {
        if (width > previewMaxSize) needsResize = 1;
        if (height > previewMaxSize) needsResize = 1;
    }
    if (needsResize == 1) {
        if (width >= height) {
            newW = previewMaxSize;
            newH = round(height * previewMaxSize / width);
        } else {
            newH = previewMaxSize;
            newW = round(width * previewMaxSize / height);
        }
        run("Size...", "width=" + newW + " height=" + newH + " interpolation=Bilinear average");
    }
    saveAs("Jpeg", outputDir + "/final_analysis_overlay_preview.jpg");
    close();
}

function writeQcReport(qcStatus) {
    report = "# IronCellQuant single-frame QC report\n\n";
    report += "## Status\n\n" + qcStatus + "\n\n";
    report += "## Input\n\n";
    report += "- image_name: " + originalFileName + "\n";
    report += "- group_name: " + groupName + "\n";
    report += "- original_long_path: " + originalLongPath + "\n";
    report += "- width: " + width + "\n";
    report += "- height: " + height + "\n\n";
    report += "## Core result\n\n";
    report += "- component_count_total: " + nObjects + "\n";
    report += "- accepted_object_count: " + acceptedCount + "\n";
    report += "- single_cell_candidate_count: " + singleCount + "\n";
    report += "- aggregate_candidate_count: " + aggregateCount + "\n";
    report += "- roi_reconstruction_warning_count: " + roiWarningCount + "\n";
    report += "- frame_area_pixels: " + d2s(framePixels,0) + "\n";
    report += "- accepted_area_percent_of_frame: " + d2s(acceptedAreaPercentOfFrame,4) + "\n";
    report += "- accepted_object_pixels: " + d2s(totalAcceptedObjectPixels,0) + "\n";
    report += "- accepted_blue_pixels: " + d2s(totalAcceptedBluePixels,0) + "\n";
    report += "- blue_pixel_percent_all_accepted: " + d2s(100*acceptedFraction,4) + "\n";
    report += "- low_accepted_area_warning: " + boolText(lowAcceptedAreaWarning) + "\n";
    report += "- min_stable_accepted_objects: " + minStableAcceptedObjects + "\n";
    report += "- min_stable_accepted_pixels: " + minStableAcceptedPixels + "\n\n";
    if (qcStatus != "PASS") {
        report += "## Warnings\n\n";
        if (acceptedCount < minExpectedAcceptedObjects) report += "- WARN_LOW_ACCEPTED_OBJECT_COUNT: fewer accepted objects than the minimum expected count.\n";
        if (roiWarningCount > 0) report += "- WARN_ROI_RECONSTRUCTION: one or more objects had ROI reconstruction warnings and were excluded from final accepted summary.\n";
        if (lowAcceptedAreaWarning == 1) report += "- WARN_LOW_ACCEPTED_AREA: accepted object count or accepted object pixels are low; frame-level blue percent can be unstable and should be interpreted cautiously.\n";
        report += "\n";
    }
    report += "## Notes\n\n";
    report += "The measured feature is preliminary relative optical blue_pixel_fraction, not a calibrated concentration measurement.\n";
    report += "Bounding boxes are loop limits only; pixel membership is checked through selectionContains(x,y).\n";
    File.saveString(report, outputDir + "/extended_qc_report.md");
}


function appendQcStatus(currentStatus, warningStatus) {
    if (currentStatus == "PASS") return warningStatus;
    if (indexOf(currentStatus, warningStatus) >= 0) return currentStatus;
    return currentStatus + ";" + warningStatus;
}

function getArgString(arg, key, defaultValue) {
    parts = split(arg, ";");
    prefix = key + "=";
    for (j = 0; j < parts.length; j++) {
        item = parts[j];
        if (startsWith(item, prefix)) return substring(item, lengthOf(prefix), lengthOf(item));
    }
    return defaultValue;
}

function getArgBool(arg, key, defaultValue) {
    v = getArgString(arg, key, "__missing__");
    if (v == "__missing__") return defaultValue;
    if (v == "true") return 1;
    if (v == "True") return 1;
    if (v == "TRUE") return 1;
    if (v == "1") return 1;
    if (v == "yes") return 1;
    if (v == "Yes") return 1;
    if (v == "false") return 0;
    if (v == "False") return 0;
    if (v == "FALSE") return 0;
    if (v == "0") return 0;
    if (v == "no") return 0;
    if (v == "No") return 0;
    return defaultValue;
}

function boolText(value) {
    if (value == 1) return "true";
    return "false";
}

function logLine(text) {
    File.append(text + "\n", logPath);
    print(text);
}

function checkpoint(name) {
    logLine("CHECKPOINT " + name);
}
