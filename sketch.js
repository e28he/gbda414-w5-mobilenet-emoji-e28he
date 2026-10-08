let video; // Live webcam video.
let classifier; // ml5.js MobileNet classifier.
let mappings = []; // Records loaded from the JSON file.
let currentLabel = "Waiting for classification..."; // Latest model label.
let currentConfidence = 0; // Confidence from 0 to 1.
let currentRecord = null; // Matching JSON record, if found.
let jsonLoaded = false; // Whether JSON loading succeeded.
let jsonError = false; // Whether JSON loading failed.

function preload() {
  // p5 waits for the dataset before setup().
  mappings = loadJSON("finalImageNetLabelsAndEmojis.json", jsonLoadedSuccessfully, jsonFailedToLoad);
}

function jsonLoadedSuccessfully(data) {
  // Store the loaded records.
  mappings = data;
  jsonLoaded = true;
}

function jsonFailedToLoad(error) {
  // An empty array allows the interface to show failure without crashing.
  console.error("Could not load JSON dataset:", error);
  jsonError = true;
  mappings = [];
}

function setup() {
  createCanvas(960, 720);
  video = createCapture(VIDEO, { flipped: true });
  video.size(width, height);
  video.hide();
  classifier = ml5.imageClassifier("MobileNet", { flipped: true });
  classifier.classifyStart(video, gotResults);
}

function normalizeLabel(labelText) {
  // Lowercase and trim a label so comparisons are less sensitive to formatting.
  return String(labelText).toLowerCase().trim();
}

function firstTerm(labelText) {
  // Keep the first term from a comma-separated synonym list.
  return normalizeLabel(labelText).split(",")[0].trim();
}

function labelTerms(labelText) {
  // Turn a comma-separated label into clean, searchable terms.
  return normalizeLabel(labelText)
    .split(",")
    .map(function (term) {
      return term.trim();
    })
    .filter(function (term) {
      return term.length > 0;
    });
}

function findMapping(modelLabel) {
  // Do not search until the JSON has loaded.
  if (!jsonLoaded || !Array.isArray(mappings)) return null;
  let modelFull = normalizeLabel(modelLabel);
  let modelFirst = firstTerm(modelLabel);
  let modelTerms = labelTerms(modelLabel);

  // PASS 1: exact full-label match.
  for (let item of mappings) {
    if (!item || !item.label) continue;
    if (modelFull === normalizeLabel(item.label)) return item;
  }

  // PASS 2: first-term match.
  for (let item of mappings) {
    if (!item || !item.label) continue;
    if (modelFirst === firstTerm(item.label)) return item;
  }

  // PASS 3: exact synonym-term match.
  for (let item of mappings) {
    if (!item || !item.label) continue;
    let datasetTerms = labelTerms(item.label);
    for (let modelTerm of modelTerms) {
      for (let datasetTerm of datasetTerms) {
        if (modelTerm === datasetTerm) return item;
      }
    }
  }

  // PASS 4: containment match, restricted to terms of four or more characters.
  for (let item of mappings) {
    if (!item || !item.label) continue;
    let datasetTerms = labelTerms(item.label);
    for (let modelTerm of modelTerms) {
      for (let datasetTerm of datasetTerms) {
        if (modelTerm.length < 4 || datasetTerm.length < 4) continue;
        if (modelTerm.includes(datasetTerm) || datasetTerm.includes(modelTerm)) return item;
      }
    }
  }
  console.log("No mapping found for:", modelLabel);
  return null;
}

function gotResults(results) {
  // Store the latest prediction and search the JSON for its translation.
  if (!results || results.length === 0) return;
  currentLabel = results[0].label;
  currentConfidence = results[0].confidence;
  currentRecord = findMapping(currentLabel);
  console.log("MobileNet label:", currentLabel);
  console.log("Confidence:", currentConfidence);
  console.log("Matching record:", currentRecord);
}

function draw() {
  // Draw the webcam and then the information panel.
  image(video, 0, 0, width, height);
  fill(0, 190);
  noStroke();
  rect(20, 20, 820, 300, 12);
  fill(255);
  textAlign(LEFT, TOP);
  textSize(18);
  text("STEP 4: MATCH THE MOBILENET LABEL", 40, 42);
  text("Model label: " + currentLabel, 40, 82);
  text("Confidence: " + nf(currentConfidence * 100, 2, 1) + "%", 40, 115);
  if (currentRecord && currentRecord.emoji) {
    push();
    // 1. 根据模型的置信度 (0 到 1)，动态计算 Emoji 的字体大小 (40px 到 180px)
    let dynamicSize = map(currentConfidence, 0, 1, 40, 180);
    textSize(dynamicSize);
    textAlign(CENTER, CENTER);
    
    // 2. 根据置信度计算透明度：把握越低越淡，把握越高越实
    let dynamicAlpha = map(currentConfidence, 0, 1, 100, 255);
    fill(255, dynamicAlpha);
    
    // 3. 在画面中央画出识别出的 Emoji
    text(currentRecord.emoji, width / 2, height / 2 + 100);

    // 4. 在 Emoji 下方补充显示 WordNet 分类
    textSize(18);
    fill(255);
    text("Category: " + (currentRecord.workshopCategory || currentRecord.wordnetParent || "General"), width / 2, height / 2 + 220);
    pop();
  } else {
    // 5. 兜底回退显示 (Fallback)：未识别或找不到对应 Emoji 时的问号
    push();
    textSize(80);
    textAlign(CENTER, CENTER);
    fill(255, 120);
    text("❓", width / 2, height / 2 + 100);
    pop();
  }
}
