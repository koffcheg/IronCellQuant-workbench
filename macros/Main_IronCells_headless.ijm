// Main_IronCells_headless.ijm
// Головний Fiji/ImageJ headless macro для попереднього відносного показника iron-staining.
// Python runner тільки запускає Fiji, передає параметри, перевіряє outputs і збирає звіти.

requires("1.53");

arg = getArgument();
inputPath = getArgString(arg, "input", "");
outputDir = getArgString(arg, "output", "");
if (inputPath == "") exit("Missing input=...");
if (outputDir == "") exit("Missing output=...");

// Метадані шляху потрібні для звітів.
// Fiji отримує short path для стабільності, але CSV має зберігати зрозумілий original long path.
originalLongPath = getArgString(arg, "original_long_path", inputPath);
originalFileName = getArgString(arg, "original_file_name", "");
groupName = getArgString(arg, "group_name", "unknown");
shortPathUsed = getArgString(arg, "short_path_used", "");

// Усі параметри нижче приходять через macro argument string.
// Це дозволяє робити sweep без ручного редагування macro між експериментами.
// Блок segmentation відповідає тільки за побудову binary mask клітинного матеріалу.
thresholdMethod = getArgString(arg, "threshold_method", "Otsu");
thresholdMode = getArgString(arg, "threshold_mode", "dark");
backgroundRolling = parseFloat(getArgString(arg, "background_rolling", "80"));
medianRadius = parseFloat(getArgString(arg, "median_radius", "2"));
contrastSaturated = parseFloat(getArgString(arg, "contrast_saturated", "0.35"));
morphOpenIterations = parseFloat(getArgString(arg, "morph_open_iterations", "0"));
morphCloseIterations = parseFloat(getArgString(arg, "morph_close_iterations", "1"));
fillHoles = getArgBool(arg, "fill_holes", false);
metadataBarHeight = parseFloat(getArgString(arg, "metadata_bar_height", "180"));

// Блок extraction/classification навмисно розділений.
// Analyze Particles може витягнути дрібні компоненти для діагностики, а аналітична класифікація
// потім вирішує, що є noise, fragment, single-cell candidate або aggregate candidate.
particleExtractMinArea = parseFloat(getArgString(arg, "particle_extract_min_area", "10"));
particleExtractMaxArea = parseFloat(getArgString(arg, "particle_extract_max_area", "2000000"));
minNoiseArea = parseFloat(getArgString(arg, "min_noise_area", "10"));
minSingleCellArea = parseFloat(getArgString(arg, "min_single_cell_area", "80"));
maxSingleCellArea = parseFloat(getArgString(arg, "max_single_cell_area", "50000"));
minAggregateArea = parseFloat(getArgString(arg, "min_aggregate_area", "50000"));
maxAggregateArea = parseFloat(getArgString(arg, "max_aggregate_area", "2000000"));
maxSingleCellAspect = parseFloat(getArgString(arg, "max_single_cell_aspect", "10"));
maxAggregateAspect = parseFloat(getArgString(arg, "max_aggregate_aspect", "30"));
excludeBorderObjects = getArgBool(arg, "exclude_border_objects", true);
borderMarginPx = parseFloat(getArgString(arg, "border_margin_px", "2"));

// Blue-pixel rule рахується по оригінальному RGB, не по grayscale preprocessing.
// Це дає preliminary relative iron-staining feature, а не абсолютну концентрацію заліза.
blueMin = parseFloat(getArgString(arg, "blue_min", "120"));
blueOverRed = parseFloat(getArgString(arg, "blue_over_red", "20"));
blueOverGreen = parseFloat(getArgString(arg, "blue_over_green", "10"));

// У нормальному режимі зображення зберігається тільки одне: final_analysis_overlay.tif.
// CSV та macro_log несуть діагностику; проміжні маски не пишемо, щоб не витрачати I/O на 4000x3000 кадрах.
saveOverlays = getArgBool(arg, "save_overlays", true);
labelObjects = getArgBool(arg, "label_objects", true);
contourWidth = parseFloat(getArgString(arg, "contour_width", "6"));
makeSegmentationSweep = getArgBool(arg, "make_segmentation_sweep", false);

File.makeDirectory(outputDir);
logPath = outputDir + "/macro_log.txt";
File.saveString("IronCells Fiji headless macro\n", logPath);
logLine("Input(short/Fiji): " + inputPath);
logLine("Input(original): " + originalLongPath);
logLine("Output: " + outputDir);
logLine("Group: " + groupName);
logLine("Blue pixels are counted from original RGB-derived blue mask inside reconstructed object ROI.");
logLine("ROI membership is checked with selectionContains(x,y); bbox is only a loop limit.");
logLine("Run parameters are saved by the Python runner in run_parameters.txt and run_parameters.csv.");
logLine("Only final_analysis_overlay.tif is saved as image output in the normal path.");

// Batch mode зменшує GUI overhead у headless запуску.
// Results очищаються на старті, щоб старі таблиці ImageJ не змішалися з поточним кадром.
setBatchMode(true);
run("Clear Results");
open(inputPath);
originalTitle = getTitle();
if (originalFileName == "") originalFileName = originalTitle;
width = getWidth();
height = getHeight();
framePixels = width * height;
logLine("Opened image: " + originalTitle + " width=" + width + " height=" + height + " bitDepth=" + bitDepth());

// Відкритий кадр стає робочим Original_RGB без додаткового full-frame duplicate.
// Це економить пам'ять: великий RGB кадр 4000x3000 не копіюється зайвий раз.
selectWindow(originalTitle);
rename("Original_RGB");
originalTitle = "Original_RGB";
logLine("prepared_original_rgb_window");

// Канали RGB розділяються з оригінального зображення.
// Blue-pixel feature не рахується з grayscale або preprocess copy.
selectWindow("Original_RGB");
run("Duplicate...", "title=RGB_For_Channels");
run("Split Channels");
titles = getList("image.titles");
redTitle = ""; greenTitle = ""; blueTitle = "";
for (ti = 0; ti < titles.length; ti++) {
    t = titles[ti];
    if (startsWith(t, "C1-") || indexOf(t, "red") >= 0 || indexOf(t, "Red") >= 0) redTitle = t;
    if (startsWith(t, "C2-") || indexOf(t, "green") >= 0 || indexOf(t, "Green") >= 0) greenTitle = t;
    if (startsWith(t, "C3-") || indexOf(t, "blue") >= 0 || indexOf(t, "Blue") >= 0) blueTitle = t;
}
if (redTitle == "" && titles.length >= 3) redTitle = titles[titles.length - 3];
if (greenTitle == "" && titles.length >= 2) greenTitle = titles[titles.length - 2];
if (blueTitle == "" && titles.length >= 1) blueTitle = titles[titles.length - 1];
selectWindow(redTitle); rename("Red_Channel"); redTitle = "Red_Channel";
selectWindow(greenTitle); rename("Green_Channel"); greenTitle = "Green_Channel";
selectWindow(blueTitle); rename("Blue_Channel"); blueTitle = "Blue_Channel";

// Grayscale pipeline використовується тільки для пошуку cell-material mask.
// Ці операції можна тюнити через runner/sweep, не змінюючи macro code.
// Після threshold цей самий window перетворюється на binary mask in-place, щоб не робити зайві копії.
selectWindow("Original_RGB");
run("Duplicate...", "title=Segmentation_Gray");
run("8-bit");
if (backgroundRolling > 0) {
    run("Subtract Background...", "rolling=" + backgroundRolling);
}
if (medianRadius > 0) {
    run("Median...", "radius=" + medianRadius);
}
if (contrastSaturated >= 0) {
    run("Enhance Contrast...", "saturated=" + contrastSaturated + " normalize");
}
logLine("Prepared segmentation grayscale image");

selectWindow("Segmentation_Gray");
logLine("before_set_auto_threshold");
setAutoThreshold(thresholdMethod + " " + thresholdMode);
logLine("after_set_auto_threshold");
run("Convert to Mask");
logLine("after_convert_segmentation_to_mask");
cellMaskTitle = "Segmentation_Gray";
// Metadata bar не заливаємо фізично в binary mask: на великих кадрах Fiji headless може зависати
// на операціях Clear/Fill по великому selection. Замість цього нижня зона відсікається логічно:
// objectTouchesBorder() класифікує такі компоненти як border_object.
whiteAfterThreshold = -1;
logLine("skip_white_count_after_threshold=true");

selectWindow(cellMaskTitle);
if (morphOpenIterations > 0) {
    logLine("before_morph_open");
    run("Options...", "iterations=" + morphOpenIterations + " count=1 black do=Open");
    logLine("after_morph_open");
}
if (morphCloseIterations > 0) {
    logLine("before_morph_close");
    run("Options...", "iterations=" + morphCloseIterations + " count=1 black do=Close");
    logLine("after_morph_close");
}
if (fillHoles) {
    logLine("before_fill_holes");
    run("Fill Holes");
    logLine("after_fill_holes");
}
whiteAfterMorphology = -1;
logLine("white_pixels_mask_after_threshold=" + whiteAfterThreshold);
logLine("white_pixels_mask_after_morphology=" + whiteAfterMorphology);

// Blue mask будується з оригінальних RGB каналів.
// Потім вона перетинається з cleaned cell-material mask, але per-object count нижче
// додатково перевіряє membership у конкретному ROI.
// Тут створюється загальна blue mask у пам'яті; на диск її не пишемо.
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
totalBluePixelsInMask = -1;
logLine("total_blue_pixels_in_cleaned_mask=" + totalBluePixelsInMask);

selectWindow(cellMaskTitle);
run("Clear Results");
run("Set Measurements...", "area centroid bounding fit shape mean redirect=None decimal=3");
logLine("before_analyze_particles_display_all_components");
// Extract threshold = 1..max навмисно збирає всі компоненти для діагностики.
// particleExtractMinArea застосовується в класифікації, щоб не втратити дрібні компоненти в CSV.
run("Analyze Particles...", "size=1-" + particleExtractMaxArea + " circularity=0.00-1.00 display clear");
nObjects = nResults;
logLine("after_analyze_particles_display_all_components component_count=" + nObjects);

// Копіюємо Results table у масиви одразу після Analyze Particles.
// Далі macro може вільно перемикати image windows, не залежачи від активної таблиці.
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

// Після Analyze Particles сума Area всіх компонентів є стабільною оцінкою площі cleaned mask.
// Це замінює ранній whiteCount() по всьому 12MP mask, який у Fiji headless може бути дуже повільним.
whiteAfterMorphology = 0;
for (i = 0; i < nObjects; i++) {
    whiteAfterMorphology += areas[i];
}

componentsGe1 = 0; componentsGe5 = 0; componentsGe10 = 0; componentsGe20 = 0;
componentsGe50 = 0; componentsGe100 = 0; componentsGe200 = 0; componentsGe500 = 0;
singleCount = 0; aggregateCount = 0; tooSmallCount = 0; smallFragmentCount = 0;
tooLargeCount = 0; tooLongCount = 0; borderCount = 0; roiWarningCount = 0; acceptedCount = 0;
totalAcceptedObjectPixels = 0; totalAcceptedBluePixels = 0;

// all_components_before_filter.csv потрібен для tuning: він показує всі компоненти до фінального відсіву.
// per_object_features.csv містить тільки object-level measurements і ROI reconstruction статус.
allCsv = outputDir + "/all_components_before_filter.csv";
File.saveString("component_id,area_px,centroid_x,centroid_y,bbox_x,bbox_y,bbox_width,bbox_height,aspect_ratio,touches_border,classification,accepted_for_summary,reject_reason\n", allCsv);
perObjectCsv = outputDir + "/per_object_features.csv";
File.saveString("image_name,group_name,original_long_path,short_path_used,object_id,classification,accepted_for_summary,reject_reason,object_pixels,blue_pixels,blue_pixel_fraction,blue_pixel_percent,area_px,bbox_x,bbox_y,bbox_width,bbox_height,centroid_x,centroid_y,aspect_ratio,touches_border,mean_R,mean_G,mean_B,roi_area_from_particles,roi_area_reconstructed,roi_area_delta_percent,roi_reconstruction_status\n", perObjectCsv);

// Overlay_Combined — єдиний image output.
// Accepted_Objects_Mask — службова in-memory mask, через яку blue pixels обмежуються accepted ROI.
if (saveOverlays) {
    selectWindow("Original_RGB"); run("Duplicate...", "title=Overlay_Combined");
    newImage("Accepted_Objects_Mask", "8-bit black", width, height, 1);
}

// Основний object loop:
// 1) класифікує компонент;
// 2) відновлює ROI;
// 3) перевіряє площу ROI;
// 4) рахує blue pixels тільки всередині ROI;
// 5) пише CSV та малює контури на фінальному overlay.
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
    if (accepted) acceptedCount++;

    // Цей рядок пишеться для кожного компонента, включно з rejected noise.
    // Так можна бачити, які пороги відсікають об'єкти і чи не завеликий background capture.
    File.append((i+1) + "," + d2s(area,0) + "," + d2s(xs[i],2) + "," + d2s(ys[i],2) + "," + d2s(bxs[i],0) + "," + d2s(bys[i],0) + "," + d2s(bws[i],0) + "," + d2s(bhs[i],0) + "," + d2s(aspect,4) + "," + boolText(touchesBorder) + "," + classification + "," + boolText(accepted) + "," + rejectReason + "\n", allCsv);

    roiStatus = "not_attempted"; roiArea = 0; roiDelta = 100; meanR = 0; meanG = 0; meanB = 0; objectPixels = 0; bluePixels = 0;
    // doWand відновлює ROI компонента на binary mask.
    // Після цього площа ROI порівнюється з Area з Analyze Particles.
    selectWindow(cellMaskTitle);
    recoverObjectRoi(xs[i], ys[i], bxs[i], bys[i], bws[i], bhs[i]);
    if (selectionType() == -1) {
        roiStatus = "failed_no_selection";
        roiWarningCount++;
        accepted = false;
        if (classification == "single_cell_candidate" || classification == "aggregate_candidate") classification = "roi_reconstruction_warning";
        rejectReason = "roi_reconstruction_failed";
        logLine("WARNING object_id=" + (i+1) + " doWand ROI reconstruction failed");
    } else {
        getStatistics(roiArea, roiMean);
        roiDelta = abs(roiArea - area) / maxOf(1, area) * 100;
        if (roiDelta > 10) {
            roiStatus = "warning_area_delta_gt_10pct";
            roiWarningCount++;
            accepted = false;
            if (classification == "single_cell_candidate" || classification == "aggregate_candidate") classification = "roi_reconstruction_warning";
            rejectReason = "roi_area_delta_gt_10pct";
            logLine("WARNING object_id=" + (i+1) + " roi_area_from_particles=" + area + " roi_area_reconstructed=" + roiArea + " delta_percent=" + d2s(roiDelta,2));
        } else {
            roiStatus = "ok";
        }
        selectWindow(redTitle); run("Restore Selection"); getStatistics(areaR, meanR);
        selectWindow(greenTitle); run("Restore Selection"); getStatistics(areaG, meanG);
        selectWindow(blueTitle); run("Restore Selection"); getStatistics(areaB, meanB);
        // bbox потрібен тільки як межа циклу для швидкості.
        // selectionContains(xx, yy) гарантує, що blue pixels рахуються всередині ROI, а не в прямокутнику bbox.
        selectWindow("Blue_Pixels_Mask"); run("Restore Selection");
        counts = countRoiPixelsInBbox(bxs[i], bys[i], bws[i], bhs[i]);
        objectPixels = counts[0];
        bluePixels = counts[1];
    }

    // Summary totals включають тільки accepted object classes.
    // Noise, border objects, ROI warnings та artifacts лишаються в CSV, але не входять в підсумковий feature.
    if (accepted) {
        totalAcceptedObjectPixels += objectPixels;
        totalAcceptedBluePixels += bluePixels;
        if (saveOverlays && selectionType() != -1) {
            // Accepted_Objects_Mask потрібна, щоб blue overlay був тільки всередині прийнятих ROI.
            selectWindow("Accepted_Objects_Mask");
            run("Restore Selection");
            setForegroundColor(255, 255, 255);
            run("Fill", "slice");
        }
    }
    if (objectPixels > 0) fraction = bluePixels / objectPixels; else fraction = 0;

    // Object CSV зберігає і класифікацію, і ROI reconstruction diagnostics.
    // Якщо doWand дав ROI з площею, що відрізняється >10%, об'єкт позначається warning і не приймається.
    File.append(originalFileName + "," + groupName + "," + originalLongPath + "," + shortPathUsed + "," + (i+1) + "," + classification + "," + boolText(accepted) + "," + rejectReason + "," + d2s(objectPixels,0) + "," + d2s(bluePixels,0) + "," + d2s(fraction,8) + "," + d2s(100*fraction,4) + "," + d2s(area,2) + "," + d2s(bxs[i],0) + "," + d2s(bys[i],0) + "," + d2s(bws[i],0) + "," + d2s(bhs[i],0) + "," + d2s(xs[i],2) + "," + d2s(ys[i],2) + "," + d2s(aspect,4) + "," + boolText(touchesBorder) + "," + d2s(meanR,3) + "," + d2s(meanG,3) + "," + d2s(meanB,3) + "," + d2s(area,2) + "," + d2s(roiArea,2) + "," + d2s(roiDelta,3) + "," + roiStatus + "\n", perObjectCsv);

    if (saveOverlays && selectionType() != -1) drawObjectOverlays(i+1, classification, area, accepted, bxs[i], bys[i]);
}

if (saveOverlays) {
    // Синій шар: blue-rule pixels тільки всередині accepted object ROI, не в bbox і не в rejected noise.
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

    selectWindow("Overlay_Combined");
    saveAs("Tiff", outputDir + "/final_analysis_overlay.tif");
    rename("Overlay_Combined");
}

// Image-level summary пишеться після object loop.
// Тут є і площа всієї cleaned mask, і підсумки тільки accepted об'єктів.
if (totalAcceptedObjectPixels > 0) acceptedFraction = totalAcceptedBluePixels / totalAcceptedObjectPixels; else acceptedFraction = 0;
if (whiteAfterMorphology > 0 && totalBluePixelsInMask >= 0) maskBlueFraction = totalBluePixelsInMask / whiteAfterMorphology; else maskBlueFraction = 0;
summaryCsv = outputDir + "/per_image_summary.csv";
File.saveString("image_name,group_name,original_long_path,short_path_used,image_width,image_height,threshold_method,threshold_mode,background_rolling,median_radius,contrast_saturated,morph_open_iterations,morph_close_iterations,fill_holes,metadata_bar_height,particle_extract_min_area,particle_extract_max_area,min_noise_area,min_single_cell_area,max_single_cell_area,min_aggregate_area,max_aggregate_area,max_single_cell_aspect,max_aggregate_aspect,exclude_border_objects,border_margin_px,blue_min,blue_over_red,blue_over_green,total_cell_material_pixels,cell_material_area_fraction,total_blue_pixels_in_cell_material,blue_pixel_fraction_all_cell_material,blue_pixel_percent_all_cell_material,component_count_total,accepted_object_count,single_cell_candidate_count,aggregate_candidate_count,too_small_noise_count,small_cell_or_fragment_count,too_large_artifact_count,too_long_artifact_count,border_object_count,roi_reconstruction_warning_count,accepted_object_pixels,accepted_blue_pixels,blue_pixel_fraction_all_accepted,blue_pixel_percent_all_accepted,components_area_ge_1,components_area_ge_5,components_area_ge_10,components_area_ge_20,components_area_ge_50,components_area_ge_100,components_area_ge_200,components_area_ge_500\n", summaryCsv);
summaryLine = originalFileName + "," + groupName + "," + originalLongPath + "," + shortPathUsed + "," + width + "," + height + "," + thresholdMethod + "," + thresholdMode + "," + backgroundRolling + "," + medianRadius + "," + contrastSaturated + "," + morphOpenIterations + "," + morphCloseIterations + "," + boolText(fillHoles) + "," + metadataBarHeight + "," + particleExtractMinArea + "," + particleExtractMaxArea + "," + minNoiseArea + "," + minSingleCellArea + "," + maxSingleCellArea + "," + minAggregateArea + "," + maxAggregateArea + "," + maxSingleCellAspect + "," + maxAggregateAspect + "," + boolText(excludeBorderObjects) + "," + borderMarginPx + "," + blueMin + "," + blueOverRed + "," + blueOverGreen + "," + d2s(whiteAfterMorphology,0) + "," + d2s(whiteAfterMorphology/framePixels,8) + "," + d2s(totalBluePixelsInMask,0) + "," + d2s(maskBlueFraction,8) + "," + d2s(100*maskBlueFraction,4) + "," + nObjects + "," + acceptedCount + "," + singleCount + "," + aggregateCount + "," + tooSmallCount + "," + smallFragmentCount + "," + tooLargeCount + "," + tooLongCount + "," + borderCount + "," + roiWarningCount + "," + d2s(totalAcceptedObjectPixels,0) + "," + d2s(totalAcceptedBluePixels,0) + "," + d2s(acceptedFraction,8) + "," + d2s(100*acceptedFraction,4) + "," + componentsGe1 + "," + componentsGe5 + "," + componentsGe10 + "," + componentsGe20 + "," + componentsGe50 + "," + componentsGe100 + "," + componentsGe200 + "," + componentsGe500 + "\n";
File.append(summaryLine, summaryCsv);

logLine("component_count_total=" + nObjects);
logLine("accepted_object_count=" + acceptedCount);
logLine("single_cell_candidate_count=" + singleCount);
logLine("aggregate_candidate_count=" + aggregateCount);
logLine("too_small_noise_count=" + tooSmallCount);
logLine("small_cell_or_fragment_count=" + smallFragmentCount);
logLine("border_object_count=" + borderCount);
logLine("roi_reconstruction_warning_count=" + roiWarningCount);
logLine("total_cell_material_pixels=" + whiteAfterMorphology);
logLine("accepted_object_pixels=" + totalAcceptedObjectPixels);
logLine("blue_pixel_percent_all_accepted=" + d2s(100*acceptedFraction,4));
logLine("Finished successfully.");

setBatchMode(false);
run("Close All");
eval("script", "java.lang.System.exit(0);");

function recoverObjectRoi(cx, cy, bx, by, bw, bh) {
    // Спочатку пробуємо centroid з Analyze Particles.
    // Якщо centroid потрапив у дірку або межу, шукаємо перший білий pixel у bbox як fallback.
    selectWindow(cellMaskTitle);
    run("Select None");
    doWand(round(cx), round(cy));
    if (selectionType() != -1) return;
    x0 = maxOf(0, floor(bx)); y0 = maxOf(0, floor(by));
    x1 = minOf(width-1, ceil(bx+bw)); y1 = minOf(height-1, ceil(by+bh));
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
    // Важливо: bbox не є областю вимірювання.
    // Він лише обмежує цикл, а selectionContains перевіряє реальну ROI membership.
    objectCount = 0; blueCount = 0;
    x0 = maxOf(0, floor(bx)); y0 = maxOf(0, floor(by));
    x1 = minOf(width-1, ceil(bx+bw)); y1 = minOf(height-1, ceil(by+bh));
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

function drawObjectOverlays(id, classification, area, accepted, bx, by) {
    if (accepted) { rr=0; gg=255; bb=0; } else { rr=255; gg=120; bb=0; }
    selectWindow("Overlay_Combined");
    run("Restore Selection");
    setLineWidth(contourWidth); setForegroundColor(rr,gg,bb); run("Draw", "slice");
    if (labelObjects) { setFont("SansSerif", 18, "bold"); drawString("#" + id + " " + d2s(area,0) + " " + classification, bx, maxOf(20, by-6)); }
    run("Select None");
}

function classifyObject(area, aspect, touchesBorder) {
    // Класифікація відділена від extraction threshold.
    // Маленькі компоненти лишаються в all_components_before_filter.csv для діагностики порогів.
    if (excludeBorderObjects && touchesBorder) return "border_object";
    if (area < particleExtractMinArea) return "too_small_noise";
    if (area < minNoiseArea) return "too_small_noise";
    if (area < minSingleCellArea) return "small_cell_or_fragment";
    if (area > maxAggregateArea) return "too_large_artifact";
    if (area <= maxSingleCellArea) {
        if (aspect > maxSingleCellAspect) return "too_long_artifact";
        return "single_cell_candidate";
    }
    if (area >= minAggregateArea && area <= maxAggregateArea) {
        if (aspect > maxAggregateAspect) return "too_long_artifact";
        return "aggregate_candidate";
    }
    return "small_cell_or_fragment";
}

function rejectReasonFor(classification, area) {
    if (classification == "single_cell_candidate" || classification == "aggregate_candidate") return "";
    if (classification == "too_small_noise" && area < particleExtractMinArea) return "below_particle_extract_min_area";
    return classification;
}

function isAcceptedClass(classification) {
    return classification == "single_cell_candidate" || classification == "aggregate_candidate";
}

function objectTouchesBorder(bx, by, bw, bh) {
    // Metadata bar внизу кадру вважається забороненою областю, як і зовнішні межі frame.
    if (bx <= borderMarginPx) return true;
    if (by <= borderMarginPx) return true;
    if (bx + bw >= width - borderMarginPx) return true;
    if (by + bh >= height - metadataBarHeight - borderMarginPx) return true;
    return false;
}

function whiteCount(title) {
    // Для binary mask mean/255*area дає кількість white pixels.
    // Це стабільніше у headless, ніж getHistogram на великих 4000x3000 кадрах.
    selectWindow(title);
    run("Select None");
    getStatistics(areaValue, meanValue);
    return round(areaValue * meanValue / 255);
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
    return v == "true" || v == "True" || v == "1" || v == "yes";
}

function boolText(value) {
    if (value) return "true";
    return "false";
}

function logLine(text) {
    File.append(text + "\n", logPath);
    print(text);
}
