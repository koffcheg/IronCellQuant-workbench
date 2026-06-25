// Main_IronCells_headless.ijm
// Fiji/ImageJ headless Stage 1 single-frame cell-material / ROI-like region analysis.
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
wekaModelPath = getArgString(arg, "weka_model", "");
wekaFailureStatus = "";
wekaTileSize = parseFloat(getArgString(arg, "weka_tile_size", "768"));
wekaTileOverlap = parseFloat(getArgString(arg, "weka_tile_overlap", "64"));

thresholdMethod = getArgString(arg, "threshold_method", "Li");
thresholdMode = getArgString(arg, "threshold_mode", "bright");
backgroundRolling = parseFloat(getArgString(arg, "background_rolling", "80"));
medianRadius = parseFloat(getArgString(arg, "median_radius", "2"));
contrastSaturated = parseFloat(getArgString(arg, "contrast_saturated", "0.35"));
morphOpenIterations = parseFloat(getArgString(arg, "morph_open_iterations", "0"));
morphCloseIterations = parseFloat(getArgString(arg, "morph_close_iterations", "2"));
fillHoles = getArgBool(arg, "fill_holes", 1);
metadataBarHeight = parseFloat(getArgString(arg, "metadata_bar_height", "120"));

particleExtractMinArea = parseFloat(getArgString(arg, "particle_extract_min_area", "100"));
particleExtractMaxArea = parseFloat(getArgString(arg, "particle_extract_max_area", "2000000"));
minNoiseArea = parseFloat(getArgString(arg, "min_noise_area", "100"));
minSingleCellArea = parseFloat(getArgString(arg, "min_single_cell_area", "200"));
maxSingleCellArea = parseFloat(getArgString(arg, "max_single_cell_area", "3000"));
minAggregateArea = parseFloat(getArgString(arg, "min_aggregate_area", "3000"));
maxAggregateArea = parseFloat(getArgString(arg, "max_aggregate_area", "2000000"));
maxSingleCellAspect = parseFloat(getArgString(arg, "max_single_cell_aspect", "4"));
maxAggregateAspect = parseFloat(getArgString(arg, "max_aggregate_aspect", "8"));
excludeBorderObjects = getArgBool(arg, "exclude_border_objects", 1);
borderMarginPx = parseFloat(getArgString(arg, "border_margin_px", "20"));

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
maxStableAcceptedObjects = parseFloat(getArgString(arg, "max_stable_accepted_objects", "40"));
minStableAcceptedPixels = parseFloat(getArgString(arg, "min_stable_accepted_pixels", "500"));
frameSelectTopSize = parseFloat(getArgString(arg, "frame_select_top_size", "40"));
frameSelectTopBlue = parseFloat(getArgString(arg, "frame_select_top_blue", "20"));

File.makeDirectory(outputDir);
logPath = outputDir + "/macro_log.txt";
File.saveString("IronCells Fiji headless macro\n", logPath);
checkpoint("start");
logLine("Input(short/Fiji): " + inputPath);
logLine("Input(original): " + originalLongPath);
logLine("Output: " + outputDir);
logLine("Group: " + groupName);
if (wekaModelPath != "") {
    logLine("Weka model: " + wekaModelPath);
    logLine("Weka tile size: " + wekaTileSize + " overlap=" + wekaTileOverlap);
}
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
if (wekaModelPath != "") {
    checkpoint("before_weka_prediction");
    wekaFailureStatus = runWekaPrediction(wekaModelPath);
    if (wekaFailureStatus == "") wekaFailureStatus = ensureWekaCellMaskWindow();
    if (wekaFailureStatus != "") {
        logLine(wekaFailureStatus + ": Weka tile inference did not produce a usable stitched cell-material mask.");
        newImage("CellMaterialMask", "8-bit black", width, height, 1);
        cellMaskTitle = "CellMaterialMask";
    } else {
        requireWindow("WekaCellMaskRaw");
        rename("CellMaterialMask");
        cellMaskTitle = "CellMaterialMask";
    }
    requireWindow(cellMaskTitle);
    if (saveOverlays == 1) saveDebugImage(cellMaskTitle, "debug_candidate_mask_raw.tif");
    checkpoint("after_weka_prediction");
} else {
checkpoint("before_prepare_segmentation_gray");
selectWindow("Original_RGB");
run("Duplicate...", "title=SegmentationBase");
run("8-bit");
if (backgroundRolling > 0) run("Subtract Background...", "rolling=" + backgroundRolling);
if (medianRadius > 0) run("Median...", "radius=" + medianRadius);
if (contrastSaturated >= 0) run("Enhance Contrast...", "saturated=" + contrastSaturated + " normalize");

requireWindow("SegmentationBase");
run("Duplicate...", "title=TextureEvidence");
run("Variance...", "radius=3");
if (contrastSaturated >= 0) run("Enhance Contrast...", "saturated=" + contrastSaturated + " normalize");
setAutoThreshold(thresholdMethod + " bright");
run("Convert to Mask");
rename("TextureEvidenceMask");

requireWindow("SegmentationBase");
run("Duplicate...", "title=EdgeEvidence");
run("Find Edges");
run("Gaussian Blur...", "sigma=1");
if (contrastSaturated >= 0) run("Enhance Contrast...", "saturated=" + contrastSaturated + " normalize");
setAutoThreshold(thresholdMethod + " bright");
run("Convert to Mask");
rename("EdgeEvidenceMask");

imageCalculator("OR create", "TextureEvidenceMask", "EdgeEvidenceMask");
rename("CellMaterialMask");
cellMaskTitle = "CellMaterialMask";
requireWindow(cellMaskTitle);
if (saveOverlays == 1) {
    saveDebugImage("TextureEvidenceMask", "debug_texture_evidence_mask.tif");
    saveDebugImage("EdgeEvidenceMask", "debug_edge_evidence_mask.tif");
    saveDebugImage(cellMaskTitle, "debug_candidate_mask_raw.tif");
}
checkpoint("after_prepare_stage1a_texture_contrast_candidate_mask");

}

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
checkpoint("before_clear_border_metadata_artifacts");
requireWindow(cellMaskTitle);
setBackgroundColor(0,0,0);
if (metadataBarHeight > 0) {
    makeRectangle(0, maxOf(0, height - metadataBarHeight), width, minOf(metadataBarHeight, height));
    run("Clear", "slice");
}
if (borderMarginPx > 0) {
    makeRectangle(0, 0, width, minOf(borderMarginPx, height)); run("Clear", "slice");
    makeRectangle(0, maxOf(0, height - borderMarginPx), width, minOf(borderMarginPx, height)); run("Clear", "slice");
    makeRectangle(0, 0, minOf(borderMarginPx, width), height); run("Clear", "slice");
    makeRectangle(maxOf(0, width - borderMarginPx), 0, minOf(borderMarginPx, width), height); run("Clear", "slice");
}
run("Select None");
if (saveOverlays == 1) saveDebugImage(cellMaskTitle, "debug_candidate_mask_cleaned.tif");
checkpoint("after_clear_border_metadata_artifacts");

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
rename("BlueInsideCells");
totalBluePixelsInMask = whiteCount("BlueInsideCells");
checkpoint("after_blue_mask_creation total_blue_pixels_in_cleaned_mask=" + totalBluePixelsInMask);

checkpoint("before_analyze_particles");
selectWindow(cellMaskTitle);
run("Clear Results");
run("Set Measurements...", "area centroid bounding fit shape mean redirect=None decimal=3");
run("Analyze Particles...", "size=" + particleExtractMinArea + "-" + particleExtractMaxArea + " circularity=0.00-1.00 display clear");
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
tooLargeCount = 0; tooLongCount = 0; borderCount = 0; roiWarningCount = 0; acceptedCount = 0; blueFullCount = 0;
totalAcceptedObjectPixels = 0; totalAcceptedBluePixels = 0;
acceptedRSum = 0; acceptedGSum = 0; acceptedBSum = 0;
acceptedRSqSum = 0; acceptedGSqSum = 0; acceptedBSqSum = 0;
acceptedBOverRSum = 0; acceptedBOverRGBSumSum = 0; acceptedGraySum = 0; acceptedGraySqSum = 0;

frameId = sanitizeId(originalFileName);
allCsv = outputDir + "/all_components_before_filter.csv";
File.saveString("component_id,area_px,centroid_x,centroid_y,bbox_x,bbox_y,bbox_width,bbox_height,aspect_ratio,touches_border,classification,accepted_for_summary,reject_reason\n", allCsv);
rejectedCsv = outputDir + "/rejected_objects.csv";
File.saveString("image_name,group_name,frame_id,object_id,feature_row_id,candidate_status,accepted_status,selected_for_frame_summary,classification,reject_reason,area_px,bbox_x,bbox_y,bbox_w,bbox_h,bbox_width,bbox_height,centroid_x,centroid_y,aspect_ratio,touches_border,roi_reconstruction_status\n", rejectedCsv);
objectCsv = outputDir + "/cell_features.csv";
cellFeatureHeader = "image_name,group_name,original_long_path,short_path_used,frame_id,object_id,feature_row_id,candidate_status,accepted_status,selected_for_frame_summary,reject_reason,selection_rank_size,selection_rank_blue,object_type,roi_area_pixels,object_pixels,blue_pixels,blue_pixel_fraction,blue_pixel_percent,bbox_x,bbox_y,bbox_w,bbox_h,bbox_width,bbox_height,centroid_x,centroid_y,aspect_ratio,R_mean,G_mean,B_mean,R_std,G_std,B_std,R_min,G_min,B_min,R_max,G_max,B_max,R_div_G,B_div_R,B_div_RGB_sum,intensity_mean,intensity_std,intensity_min,intensity_max,cell_material_area_px,roi_area_reconstructed,roi_area_delta_percent,roi_reconstruction_status\n";
File.saveString(cellFeatureHeader, objectCsv);
blueCsv = outputDir + "/blue_pixels_features.csv";
File.saveString("image_name,group_name,object_type,object_id,object_pixels,blue_pixels,blue_pixel_fraction,blue_pixel_percent\n", blueCsv);
acceptedFlags = newArray(nObjects); selectedFlags = newArray(nObjects);
objectPixelsForSelection = newArray(nObjects); bluePixelsForSelection = newArray(nObjects); blueFractionForSelection = newArray(nObjects);
sizeRankForSelection = newArray(nObjects); blueRankForSelection = newArray(nObjects); objectBaseLines = newArray(nObjects);

if (saveOverlays == 1) {
    checkpoint("before_overlay_setup");
    selectWindow("Original_RGB");
    run("Duplicate...", "title=RoiOverlay");
    selectWindow("Original_RGB");
    run("Duplicate...", "title=ReviewDetectionOverlay");
    newImage("Accepted_Objects_Mask", "8-bit black", width, height, 1);
    checkpoint("after_overlay_setup");
}

checkpoint("before_per_object_loop");
for (i = 0; i < nObjects; i++) {
    objectId = i + 1;
    featureRowId = frameId + "_object_" + objectId;
    area = areas[i];
    aspect = maxOf(bws[i], bhs[i]) / maxOf(1, minOf(bws[i], bhs[i]));
    touchesBorder = objectTouchesBorder(bxs[i], bys[i], bws[i], bhs[i]);
    fillRatio = area / maxOf(1, bws[i] * bhs[i]);
    classification = classifyObject(area, aspect, touchesBorder, fillRatio);
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
    if (classification == "too_small") tooSmallCount++;
    if (classification == "small_cell_or_fragment") smallFragmentCount++;
    if (classification == "cell_region_candidate") smallFragmentCount++;
    if (classification == "single_cell_candidate") singleCount++;
    if (classification == "aggregate_candidate") aggregateCount++;
    if (classification == "too_large_artifact") tooLargeCount++;
    if (classification == "too_long_artifact") tooLongCount++;
    if (classification == "rectangle_or_line_artifact") tooLongCount++;
    if (classification == "border_object") borderCount++;

    File.append((i+1) + "," + d2s(area,0) + "," + d2s(xs[i],2) + "," + d2s(ys[i],2) + "," + d2s(bxs[i],0) + "," + d2s(bys[i],0) + "," + d2s(bws[i],0) + "," + d2s(bhs[i],0) + "," + d2s(aspect,4) + "," + boolText(touchesBorder) + "," + classification + "," + boolText(accepted) + "," + rejectReason + "\n", allCsv);

    roiStatus = "not_attempted"; roiArea = 0; roiDelta = 100; meanR = 0; meanG = 0; meanB = 0; stdR = 0; stdG = 0; stdB = 0; minR = 0; minG = 0; minB = 0; maxR = 0; maxG = 0; maxB = 0; grayMean = 0; grayStd = 0; grayMin = 0; grayMax = 0; objectPixels = 0; bluePixels = 0; roiAreaPixels = round(bws[i]) * round(bhs[i]);
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
        selectWindow("BlueInsideCells"); run("Restore Selection");
        counts = countRoiPixelsInBbox(bxs[i], bys[i], bws[i], bhs[i]);
        objectPixels = counts[0];
        bluePixels = counts[1];
    }

    if (objectPixels > 0) fraction = bluePixels / objectPixels; else fraction = 0;
    if (accepted == 1) {
        postRejectReason = postFilterRejectReason(objectPixels, roiAreaPixels, bws[i], bhs[i], aspect, fillRatio, touchesBorder, fraction);
        if (postRejectReason != "") {
            accepted = 0;
            classification = postRejectReason;
            rejectReason = postRejectReason;
        }
    }
    epsilon = 0.000001;
    rOverG = meanR / maxOf(epsilon, meanG);
    bOverR = meanB / maxOf(epsilon, meanR);
    bOverRgbSum = meanB / maxOf(epsilon, meanR + meanG + meanB);

    if (accepted == 1) {
        acceptedCount++;
        totalAcceptedObjectPixels += objectPixels;
        totalAcceptedBluePixels += bluePixels;
        if (fraction >= 0.999) blueFullCount++;
        acceptedRSum += meanR * objectPixels; acceptedGSum += meanG * objectPixels; acceptedBSum += meanB * objectPixels;
        acceptedRSqSum += (stdR * stdR + meanR * meanR) * objectPixels;
        acceptedGSqSum += (stdG * stdG + meanG * meanG) * objectPixels;
        acceptedBSqSum += (stdB * stdB + meanB * meanB) * objectPixels;
        acceptedBOverRSum += bOverR * objectPixels;
        acceptedBOverRGBSumSum += bOverRgbSum * objectPixels;
        acceptedGraySum += grayMean * objectPixels;
        acceptedGraySqSum += (grayStd * grayStd + grayMean * grayMean) * objectPixels;
        objectType = outputObjectType(classification);
        acceptedFlags[i] = 1;
        objectPixelsForSelection[i] = objectPixels;
        bluePixelsForSelection[i] = bluePixels;
        blueFractionForSelection[i] = fraction;
        objectBaseLines[i] = objectType + "," + d2s(roiAreaPixels,0) + "," + d2s(objectPixels,0) + "," + d2s(bluePixels,0) + "," + d2s(fraction,8) + "," + d2s(100*fraction,4) + "," + d2s(bxs[i],0) + "," + d2s(bys[i],0) + "," + d2s(bws[i],0) + "," + d2s(bhs[i],0) + "," + d2s(bws[i],0) + "," + d2s(bhs[i],0) + "," + d2s(xs[i],2) + "," + d2s(ys[i],2) + "," + d2s(aspect,4) + "," + d2s(meanR,3) + "," + d2s(meanG,3) + "," + d2s(meanB,3) + "," + d2s(stdR,3) + "," + d2s(stdG,3) + "," + d2s(stdB,3) + "," + d2s(minR,0) + "," + d2s(minG,0) + "," + d2s(minB,0) + "," + d2s(maxR,0) + "," + d2s(maxG,0) + "," + d2s(maxB,0) + "," + d2s(rOverG,6) + "," + d2s(bOverR,6) + "," + d2s(bOverRgbSum,6) + "," + d2s(grayMean,3) + "," + d2s(grayStd,3) + "," + d2s(grayMin,0) + "," + d2s(grayMax,0) + "," + d2s(area,2) + "," + d2s(roiArea,2) + "," + d2s(roiDelta,3) + "," + roiStatus;
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
        File.append(originalFileName + "," + groupName + "," + frameId + "," + objectId + "," + featureRowId + ",weka_candidate,rejected,false," + classification + "," + rejectReason + "," + d2s(area,2) + "," + d2s(bxs[i],0) + "," + d2s(bys[i],0) + "," + d2s(bws[i],0) + "," + d2s(bhs[i],0) + "," + d2s(bws[i],0) + "," + d2s(bhs[i],0) + "," + d2s(xs[i],2) + "," + d2s(ys[i],2) + "," + d2s(aspect,4) + "," + boolText(touchesBorder) + "," + roiStatus + "\n", rejectedCsv);
    }

    if (saveOverlays == 1) {
        if (selectionType() != -1) {
            shouldDrawObject = accepted;
            if (drawRejectedObjects == 1) shouldDrawObject = 1;
            if (shouldDrawObject == 1) {
                drawObjectOverlay(objectId, classification, area, accepted, fraction, bxs[i], bys[i]);
                if (accepted == 1) drawReviewObjectOverlay(objectId, classification, fraction, bxs[i], bys[i], bws[i], bhs[i]);
            }
        }
    }
}

selectedCount = 0; selectedObjectPixels = 0; selectedBluePixels = 0; selectedObjectIds = "";
for (i = 0; i < nObjects; i++) {
    if (acceptedFlags[i] == 1) {
        rankSize = 1;
        for (j = 0; j < nObjects; j++) {
            if (acceptedFlags[j] == 1) {
                if (objectPixelsForSelection[j] > objectPixelsForSelection[i]) rankSize++;
                if (objectPixelsForSelection[j] == objectPixelsForSelection[i] && j < i) rankSize++;
            }
        }
        sizeRankForSelection[i] = rankSize;
    }
}
for (i = 0; i < nObjects; i++) {
    if (acceptedFlags[i] == 1) {
        rankBlue = 1;
        for (j = 0; j < nObjects; j++) {
            if (acceptedFlags[j] == 1 && sizeRankForSelection[j] <= frameSelectTopSize) {
                if (blueFractionForSelection[j] > blueFractionForSelection[i]) rankBlue++;
                if (blueFractionForSelection[j] == blueFractionForSelection[i] && j < i) rankBlue++;
            }
        }
        blueRankForSelection[i] = rankBlue;
        if (sizeRankForSelection[i] <= frameSelectTopSize && blueRankForSelection[i] <= frameSelectTopBlue) {
            selectedFlags[i] = 1;
            selectedCount++;
            selectedObjectPixels += objectPixelsForSelection[i];
            selectedBluePixels += bluePixelsForSelection[i];
            if (selectedObjectIds != "") selectedObjectIds += "|";
            selectedObjectIds += "" + (i + 1);
        }
    }
}
for (i = 0; i < nObjects; i++) {
    if (acceptedFlags[i] == 1) {
        File.append(originalFileName + "," + groupName + "," + originalLongPath + "," + shortPathUsed + "," + frameId + "," + (i+1) + "," + frameId + "_object_" + (i+1) + ",weka_candidate,accepted_cell_candidate," + boolText(selectedFlags[i]) + ",," + d2s(sizeRankForSelection[i],0) + "," + d2s(blueRankForSelection[i],0) + "," + objectBaseLines[i] + "\n", objectCsv);
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
if (selectedObjectPixels > 0) selectedFraction = selectedBluePixels / selectedObjectPixels; else selectedFraction = 0;
File.append(originalFileName + "," + groupName + ",all_accepted_cell_material,frame," + d2s(totalAcceptedObjectPixels,0) + "," + d2s(totalAcceptedBluePixels,0) + "," + d2s(acceptedFraction,8) + "," + d2s(100*acceptedFraction,4) + "\n", blueCsv);
File.append(originalFileName + "," + groupName + ",all_cleaned_cell_material,frame," + d2s(whiteAfterMorphology,0) + "," + d2s(totalBluePixelsInMask,0) + "," + d2s(maskBlueFraction,8) + "," + d2s(100*maskBlueFraction,4) + "\n", blueCsv);

if (saveOverlays == 1) {
    checkpoint("before_final_overlay_blue_layer");
    requireWindow("BlueInsideCells");
    requireWindow("Accepted_Objects_Mask");
    imageCalculator("AND create", "BlueInsideCells", "Accepted_Objects_Mask");
    rename("BlueAcceptedMask");
    requireWindow("BlueAcceptedMask");
    run("Create Selection");
    if (selectionType() != -1) {
        requireWindow("RoiOverlay");
        run("Restore Selection");
        setForegroundColor(0,80,255);
        run("Fill", "slice");
        run("Select None");
    }

    checkpoint("before_save_cellmask");
    requireWindow("Accepted_Objects_Mask");
    run("Select None");
    run("Duplicate...", "title=AcceptedCellMaterialMaskSave");
    requireWindow("AcceptedCellMaterialMaskSave");
    saveAs("Tiff", outputDir + "/cellmask.tif");
    close();
    requireWindow("Accepted_Objects_Mask");
    checkpoint("after_save_cellmask");

    checkpoint("before_save_vis_cellpixels");
    requireWindow("Original_RGB");
    run("Select None");
    run("Duplicate...", "title=VisCellPixels");
    requireWindow("Accepted_Objects_Mask");
    run("Create Selection");
    if (selectionType() != -1) {
        requireWindow("VisCellPixels");
        run("Restore Selection");
        setForegroundColor(255,0,255);
        run("Fill", "slice");
        run("Select None");
    }
    requireWindow("VisCellPixels");
    saveAs("Png", outputDir + "/vis_cellpixels.png");
    checkpoint("after_save_vis_cellpixels");

    checkpoint("before_save_blue_inside_cells");
    requireWindow("BlueAcceptedMask");
    run("Select None");
    run("Duplicate...", "title=BlueAcceptedMaskSave");
    requireWindow("BlueAcceptedMaskSave");
    saveAs("Tiff", outputDir + "/blue_inside_cells.tif");
    close();
    requireWindow("BlueAcceptedMask");
    checkpoint("after_save_blue_inside_cells");

    checkpoint("before_save_roi_overlay");
    requireWindow("RoiOverlay");
    run("Select None");
    saveAs("Jpeg", outputDir + "/roi_overlay.jpg");
    checkpoint("after_save_roi_overlay");
}

lowAcceptedAreaWarning = 0;
if (acceptedCount < minStableAcceptedObjects) lowAcceptedAreaWarning = 1;
if (totalAcceptedObjectPixels < minStableAcceptedPixels) lowAcceptedAreaWarning = 1;

qcStatus = "PASS";
if (wekaFailureStatus != "") qcStatus = appendQcStatus(qcStatus, wekaFailureStatus);
if (acceptedCount == 0) qcStatus = appendQcStatus(qcStatus, "FAIL_NO_ACCEPTED_OBJECTS");
if (acceptedCount < minExpectedAcceptedObjects) qcStatus = appendQcStatus(qcStatus, "WARN_LOW_ACCEPTED_OBJECT_COUNT");
if (selectedCount < minExpectedAcceptedObjects) qcStatus = appendQcStatus(qcStatus, "WARN_LOW_SELECTED_FRAME_OBJECT_COUNT");
if (roiWarningCount > 0) qcStatus = appendQcStatus(qcStatus, "WARN_ROI_RECONSTRUCTION");
if (lowAcceptedAreaWarning == 1) qcStatus = appendQcStatus(qcStatus, "WARN_LOW_ACCEPTED_AREA");
if (acceptedCount > maxStableAcceptedObjects) qcStatus = appendQcStatus(qcStatus, "WARN_HIGH_ACCEPTED_OBJECT_COUNT");
if (acceptedCount > 0) { if (blueFullCount / acceptedCount > 0.25) qcStatus = appendQcStatus(qcStatus, "WARN_MANY_FULL_BLUE_OBJECTS"); }
if (borderCount > 0) qcStatus = appendQcStatus(qcStatus, "WARN_BORDER_ARTIFACTS_REMOVED");
if (nObjects > 0) { if ((tooSmallCount + tooLongCount) / nObjects > 0.50) qcStatus = appendQcStatus(qcStatus, "WARN_MANY_REJECTED_ARTIFACTS"); }

summaryCsv = outputDir + "/final_frame_summary.csv";
summaryHeader = "image_name,group_name,original_long_path,short_path_used,image_width,image_height,threshold_method,threshold_mode,background_rolling,median_radius,contrast_saturated,morph_open_iterations,morph_close_iterations,fill_holes,metadata_bar_height,frame_area_pixels,accepted_area_fraction_of_frame,accepted_area_percent_of_frame,particle_extract_min_area,particle_extract_max_area,min_noise_area,min_single_cell_area,max_single_cell_area,min_aggregate_area,max_aggregate_area,max_single_cell_aspect,max_aggregate_aspect,exclude_border_objects,border_margin_px,blue_min,blue_over_red,blue_over_green,min_stable_accepted_objects,min_stable_accepted_pixels,total_cell_material_pixels,cell_material_area_fraction,total_blue_pixels_in_cell_material,blue_pixel_fraction_all_cell_material,blue_pixel_percent_all_cell_material,selected_frame_object_count,selected_frame_object_ids,selected_object_pixels,selected_blue_pixels,selected_blue_pixel_percent,component_count_total,accepted_object_count,single_cell_candidate_count,aggregate_candidate_count,too_small_noise_count,small_cell_or_fragment_count,too_large_artifact_count,too_long_artifact_count,border_object_count,roi_reconstruction_warning_count,accepted_R_mean,accepted_G_mean,accepted_B_mean,accepted_R_std,accepted_G_std,accepted_B_std,accepted_B_over_R_mean,accepted_B_over_RGB_sum_mean,accepted_gray_stddev,accepted_object_pixels,accepted_blue_pixels,blue_pixel_fraction_all_accepted,blue_pixel_percent_all_accepted,components_area_ge_1,components_area_ge_5,components_area_ge_10,components_area_ge_20,components_area_ge_50,components_area_ge_100,components_area_ge_200,components_area_ge_500,qc_status\n";
File.saveString(summaryHeader, summaryCsv);
summaryLine = originalFileName + "," + groupName + "," + originalLongPath + "," + shortPathUsed + "," + width + "," + height + "," + thresholdMethod + "," + thresholdMode + "," + backgroundRolling + "," + medianRadius + "," + contrastSaturated + "," + morphOpenIterations + "," + morphCloseIterations + "," + boolText(fillHoles) + "," + metadataBarHeight + "," + d2s(framePixels,0) + "," + d2s(acceptedAreaFractionOfFrame,8) + "," + d2s(acceptedAreaPercentOfFrame,4) + "," + particleExtractMinArea + "," + particleExtractMaxArea + "," + minNoiseArea + "," + minSingleCellArea + "," + maxSingleCellArea + "," + minAggregateArea + "," + maxAggregateArea + "," + maxSingleCellAspect + "," + maxAggregateAspect + "," + boolText(excludeBorderObjects) + "," + borderMarginPx + "," + blueMin + "," + blueOverRed + "," + blueOverGreen + "," + minStableAcceptedObjects + "," + minStableAcceptedPixels + "," + d2s(totalAcceptedObjectPixels,0) + "," + d2s(acceptedAreaFractionOfFrame,8) + "," + d2s(totalAcceptedBluePixels,0) + "," + d2s(acceptedFraction,8) + "," + d2s(100*acceptedFraction,4) + "," + selectedCount + "," + selectedObjectIds + "," + d2s(selectedObjectPixels,0) + "," + d2s(selectedBluePixels,0) + "," + d2s(100*selectedFraction,4) + "," + nObjects + "," + acceptedCount + "," + singleCount + "," + aggregateCount + "," + tooSmallCount + "," + smallFragmentCount + "," + tooLargeCount + "," + tooLongCount + "," + borderCount + "," + roiWarningCount + "," + d2s(acceptedRMean,3) + "," + d2s(acceptedGMean,3) + "," + d2s(acceptedBMean,3) + "," + d2s(acceptedRStd,3) + "," + d2s(acceptedGStd,3) + "," + d2s(acceptedBStd,3) + "," + d2s(acceptedBOverRMean,6) + "," + d2s(acceptedBOverRGBSumMean,6) + "," + d2s(acceptedGrayStddev,3) + "," + d2s(totalAcceptedObjectPixels,0) + "," + d2s(totalAcceptedBluePixels,0) + "," + d2s(acceptedFraction,8) + "," + d2s(100*acceptedFraction,4) + "," + componentsGe1 + "," + componentsGe5 + "," + componentsGe10 + "," + componentsGe20 + "," + componentsGe50 + "," + componentsGe100 + "," + componentsGe200 + "," + componentsGe500 + "," + qcStatus + "\n";
File.append(summaryLine, summaryCsv);

writeQcReport(qcStatus);
checkpoint("after_write_summary_and_qc");
logLine("component_count_total=" + nObjects);
logLine("accepted_object_count=" + acceptedCount);
logLine("selected_frame_object_count=" + selectedCount);
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
    requireWindow("RoiOverlay");
    run("Restore Selection");
    setLineWidth(contourWidth); setForegroundColor(rr,gg,bb); run("Draw", "slice");
    if (labelObjects == 1) {
        setFont("SansSerif", 18, "bold");
        drawString("#" + id, bx, maxOf(20, by-6));
    }
    run("Select None");
}

function drawReviewObjectOverlay(id, classification, fraction, bx, by, bw, bh) {
    requireWindow("ReviewDetectionOverlay");
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
    requireWindow("ReviewDetectionOverlay");
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

function outputObjectType(classification) {
    if (classification == "single_cell_candidate") return "single_cell";
    if (classification == "aggregate_candidate") return "aggregate";
    if (classification == "cell_region_candidate") return "cell_region";
    return classification;
}

function classifyObject(area, aspect, touchesBorder, fillRatio) {
    if (excludeBorderObjects == 1) {
        if (touchesBorder == 1) return "border_object";
    }
    if (area < particleExtractMinArea) return "too_small";
    if (area < minNoiseArea) return "too_small";
    if (area < minSingleCellArea) return "too_small";
    if (fillRatio < 0.12) return "rectangle_or_line_artifact";
    if (area > maxAggregateArea) return "background_texture";
    if (area <= maxSingleCellArea) {
        if (aspect > maxSingleCellAspect) return "rectangle_or_line_artifact";
        if (area < 500) return "cell_region_candidate";
        return "single_cell_candidate";
    }
    if (area >= minAggregateArea) {
        if (area <= maxAggregateArea) {
            if (aspect > maxAggregateAspect) return "rectangle_or_line_artifact";
            return "aggregate_candidate";
        }
    }
    return "uncertain_or_artifact";
}

function rejectReasonFor(classification, area) {
    if (isAcceptedClass(classification) == 1) return "";
    if (classification == "too_small") return "reject_too_small";
    if (classification == "border_object") return "reject_border_artifact";
    if (classification == "rectangle_or_line_artifact") return "reject_line_or_frame_artifact";
    if (classification == "too_large_artifact") return "reject_large_rectangular_artifact";
    return "reject_" + classification;
}

function postFilterRejectReason(objectPixels, roiAreaPixels, bw, bh, aspect, fillRatio, touchesBorder, fraction) {
    bboxArea = maxOf(1, bw * bh);
    objectFill = objectPixels / maxOf(1, roiAreaPixels);
    longSide = maxOf(bw, bh);
    shortSide = maxOf(1, minOf(bw, bh));
    if (touchesBorder == 1) return "reject_border_artifact";
    if (objectPixels < minSingleCellArea) return "reject_too_small";
    if (aspect > maxAggregateAspect) return "reject_line_or_frame_artifact";
    if (longSide > 0.40 * maxOf(width, height) && shortSide < 0.08 * minOf(width, height)) return "reject_line_or_frame_artifact";
    if (objectPixels > maxSingleCellArea && objectFill > 0.82 && fraction > 0.98) return "reject_large_full_blue_artifact";
    if (objectPixels > maxSingleCellArea && objectFill > 0.88 && aspect < 1.35 && fraction > 0.90) return "reject_large_rectangular_artifact";
    if (bboxArea > 0.015 * framePixels && objectFill > 0.75 && fraction > 0.95) return "reject_large_full_blue_artifact";
    return "";
}

function sanitizeId(value) {
    clean = replace(value, ",", "_");
    clean = replace(clean, " ", "_");
    clean = replace(clean, "\\", "_");
    clean = replace(clean, "/", "_");
    clean = replace(clean, ".", "_");
    return clean;
}

function isAcceptedClass(classification) {
    if (classification == "single_cell_candidate") return 1;
    if (classification == "aggregate_candidate") return 1;
    if (classification == "cell_region_candidate") return 1;
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
    requireWindow("RoiOverlay");
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
        if (wekaFailureStatus != "") report += "- " + wekaFailureStatus + ": Weka tile inference did not produce a usable stitched cell-material mask.\n";
        if (acceptedCount == 0) {
            report += "- FAIL_NO_ACCEPTED_OBJECTS: Stage 1A detector found no accepted biological ROI/cell-material regions.\n";
            if (borderCount > 0) report += "- Only rejected border/annotation artifact candidates were detected in this run; they are not reported as biological ROIs.\n";
            report += "- Stage 1B supervised detector investigation is required before biological interpretation of this frame.\n";
        }
        if (acceptedCount < minExpectedAcceptedObjects) report += "- WARN_LOW_ACCEPTED_OBJECT_COUNT: fewer accepted objects than the minimum expected count.\n";
        if (selectedCount < minExpectedAcceptedObjects) report += "- WARN_LOW_SELECTED_FRAME_OBJECT_COUNT: fewer selected frame-summary objects than the minimum expected count.\n";
        if (roiWarningCount > 0) report += "- WARN_ROI_RECONSTRUCTION: one or more objects had ROI reconstruction warnings and were excluded from final accepted summary.\n";
        if (lowAcceptedAreaWarning == 1) report += "- WARN_LOW_ACCEPTED_AREA: accepted object count or accepted object pixels are low; frame-level blue percent can be unstable and should be interpreted cautiously.\n";
        report += "\n";
    }
    report += "## Notes\n\n";
    report += "The measured feature is preliminary relative optical blue_pixel_fraction, not a calibrated concentration measurement.\n";
    report += "Bounding boxes are loop limits only; pixel membership is checked through selectionContains(x,y).\n";
    if (acceptedCount == 0) report += "This run is a detector/QC failure, not a successful biological Stage 1A result; use the generated files only for debugging and Stage 1B detector planning.\n";
    File.saveString(report, outputDir + "/extended_qc_report.md");
}


function runWekaPrediction(modelPath) {
    statusPath = outputDir + "/weka_status.txt";
    classMapPath = outputDir + "/debug_weka_class_map.tif";
    probabilityPath = outputDir + "/debug_weka_probability_map.tif";
    tileMaskPath = outputDir + "/debug_weka_tile_mask_raw.tif";
    script = "";
    script += "var IJ = Packages.ij.IJ;\n";
    script += "var WM = Packages.ij.WindowManager;\n";
    script += "var FileWriter = Packages.java.io.FileWriter;\n";
    script += "function ck(name) { var fw = new FileWriter('" + jsPath(logPath) + "', true); fw.write('CHECKPOINT ' + name + '\\n'); fw.close(); }\n";
    script += "try {\n";
    script += "  var WekaSegmentation = Packages.trainableSegmentation.WekaSegmentation;\n";
    script += "  var Roi = Packages.ij.gui.Roi;\n";
    script += "  var Duplicator = Packages.ij.plugin.Duplicator;\n";
    script += "  var ByteProcessor = Packages.ij.process.ByteProcessor;\n";
    script += "  var ImagePlus = Packages.ij.ImagePlus;\n";
    script += "  var Integer = Packages.java.lang.Integer;\n";
    script += "  var imp = WM.getImage('Original_RGB');\n";
    script += "  var w = imp.getWidth(); var h = imp.getHeight();\n";
    script += "  var tileSize = " + d2s(wekaTileSize, 0) + "; var overlap = " + d2s(wekaTileOverlap, 0) + ";\n";
    script += "  if (tileSize < 128) tileSize = 128;\n";
    script += "  if (overlap < 0) overlap = 0;\n";
    script += "  if (overlap * 2 >= tileSize) overlap = Math.floor(tileSize / 4);\n";
    script += "  var step = tileSize - 2 * overlap; if (step < 64) step = tileSize;\n";
    script += "  var full = new ByteProcessor(w, h);\n";
    script += "  var savedFirstProbability = false; var tileCount = 0;\n";
    script += "  ck('before_first_weka_tile');\n";
    script += "  for (var y = 0; y < h; y += step) {\n";
    script += "    for (var x = 0; x < w; x += step) {\n";
    script += "      var tw = Math.floor(Math.min(tileSize, w - x)); var th = Math.floor(Math.min(tileSize, h - y));\n";
    script += "      var xi = Integer.valueOf(String(Math.floor(x))); var yi = Integer.valueOf(String(Math.floor(y)));\n";
    script += "      var twi = Integer.valueOf(String(tw)); var thi = Integer.valueOf(String(th));\n";
    script += "      if (twi.intValue() <= 0 || thi.intValue() <= 0) continue;\n";
    script += "      imp.setRoi(new Roi(xi, yi, twi, thi));\n";
    script += "      if (tileCount == 0) ck('after_create_weka_tile_roi_1');\n";
    script += "      var tile = new Duplicator().run(imp);\n";
    script += "      var segmentator = new WekaSegmentation(tile);\n";
    script += "      segmentator.loadClassifier('" + jsPath(modelPath) + "');\n";
    script += "      var probability = segmentator.applyClassifier(tile, 0, true);\n";
    script += "      if (!savedFirstProbability) { probability.setTitle('WekaProbabilityTile0'); IJ.saveAs(probability, 'Tiff', '" + jsPath(probabilityPath) + "'); savedFirstProbability = true; }\n";
    script += "      probability.setSlice(1);\n";
    script += "      var proc = probability.getProcessor();\n";
    script += "      var threshold = (probability.getBitDepth() == 32) ? 0.5 : 128;\n";
    script += "      var left = (x == 0) ? 0 : overlap; var top = (y == 0) ? 0 : overlap;\n";
    script += "      var right = (x + tw >= w) ? 0 : overlap; var bottom = (y + th >= h) ? 0 : overlap;\n";
    script += "      for (var yy = top; yy < th - bottom; yy++) {\n";
    script += "        for (var xx = left; xx < tw - right; xx++) {\n";
    script += "          if (proc.getf(xx, yy) >= threshold) full.set(x + xx, y + yy, 255);\n";
    script += "        }\n";
    script += "      }\n";
    script += "      tileCount++; if (tileCount % 10 == 0) ck('after_weka_tile_' + tileCount);\n";
    script += "      tile.close(); probability.close();\n";
    script += "    }\n";
    script += "  }\n";
    script += "  ck('after_weka_tile_count_' + tileCount);\n";
    script += "  imp.killRoi();\n";
    script += "  ck('after_weka_stitching');\n";
    script += "  var mask = new ImagePlus('WekaCellMaskRaw', full);\n";
    script += "  mask.show();\n";
    script += "  ck('after_create_WekaCellMaskRaw');\n";
    script += "  IJ.saveAs(mask, 'Tiff', '" + jsPath(tileMaskPath) + "');\n";
    script += "  IJ.saveAs(mask, 'Tiff', '" + jsPath(classMapPath) + "');\n";
    script += "  var fw = new FileWriter('" + jsPath(statusPath) + "'); fw.write('OK'); fw.close();\n";
    script += "} catch (e) { var msg = String(e); var code = (msg.indexOf('ClassNotFound') >= 0 || msg.indexOf('NoClassDefFound') >= 0 || msg.indexOf('trainableSegmentation') >= 0) ? 'FAIL_WEKA_PLUGIN_UNAVAILABLE' : 'FAIL_WEKA_TILE_INFERENCE'; var fw = new FileWriter('" + jsPath(statusPath) + "'); fw.write(code + '\\n' + e); fw.close(); }\n";
    eval("script", script);
    status = File.openAsString(statusPath);
    if (startsWith(status, "OK")) return "";
    if (startsWith(status, "FAIL_WEKA_TILE_INFERENCE")) return "FAIL_WEKA_TILE_INFERENCE";
    return "FAIL_WEKA_PLUGIN_UNAVAILABLE";
}

function ensureWekaCellMaskWindow() {
    if (windowExists("WekaCellMaskRaw")) {
        selectWindow("WekaCellMaskRaw");
        if (getWidth() != width || getHeight() != height) {
            logLine("FAIL_WEKA_MASK_MISSING: WekaCellMaskRaw dimensions do not match the original image.");
            return "FAIL_WEKA_MASK_MISSING";
        }
        checkpoint("after_create_WekaCellMaskRaw");
        return "";
    }
    tileMaskPath = outputDir + "/debug_weka_tile_mask_raw.tif";
    if (File.exists(tileMaskPath)) {
        checkpoint("before_open_stitched_weka_mask");
        open(tileMaskPath);
        rename("WekaCellMaskRaw");
        if (getWidth() != width || getHeight() != height) {
            logLine("FAIL_WEKA_MASK_MISSING: restored stitched Weka mask dimensions do not match the original image.");
            return "FAIL_WEKA_MASK_MISSING";
        }
        checkpoint("after_create_WekaCellMaskRaw");
        return "";
    }
    logLine("FAIL_WEKA_MASK_MISSING: Weka status was OK, but no WekaCellMaskRaw window or stitched mask file was found.");
    return "FAIL_WEKA_MASK_MISSING";
}

function jsPath(pathValue) {
    fixed = replace(pathValue, "\\", "/");
    fixed = replace(fixed, "'", "\\'");
    return fixed;
}

function saveDebugImage(title, fileName) {
    requireWindow(title);
    run("Select None");
    run("Duplicate...", "title=DebugSaveWindow");
    requireWindow("DebugSaveWindow");
    saveAs("Tiff", outputDir + "/" + fileName);
    close();
}

function windowExists(title) {
    titles = getList("image.titles");
    for (wi = 0; wi < titles.length; wi++) {
        if (titles[wi] == title) return 1;
    }
    return 0;
}

function requireWindow(title) {
    if (windowExists(title) == 0) {
        message = "ERROR required ImageJ window missing: " + title;
        logLine(message);
        exit(message);
    }
    selectWindow(title);
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
