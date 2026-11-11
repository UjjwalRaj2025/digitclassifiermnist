// DOM Elements
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const clearBtn = document.getElementById('clear');
const undoBtn = document.getElementById('undoBtn');
const brush = document.getElementById('brush');
const brushSizeVal = document.getElementById('brushSizeVal');
const brushDot = document.getElementById('brushDot');
const rt = document.getElementById('rt');
const predictBtn = document.getElementById('predictBtn');
const predEl = document.getElementById('pred');
const confidenceVal = document.getElementById('confidenceVal');
const confidenceFill = document.getElementById('confidenceFill');
const barsEl = document.getElementById('bars');
const inferenceStatus = document.getElementById('inferenceStatus');

// Canvas Drawing State
let drawing = false;
let lastPoint = null;
let historyStack = [];
const MAX_HISTORY = 20;

// Client-side Model State
let localModel = null;
let isModelLoading = false;

// Initialize Canvas
function initCanvas() {
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#000000';
  updateBrush();
}

function updateBrush() {
  const size = Number(brush.value);
  ctx.lineWidth = size;
  brushSizeVal.textContent = `${size}px`;
  
  // Update live preview dot (scale to fit 24px preview box)
  const displaySize = Math.max(4, Math.min(22, size * 0.6));
  brushDot.style.width = `${displaySize}px`;
  brushDot.style.height = `${displaySize}px`;
}

brush.addEventListener('input', updateBrush);

// Snapshot for Undo functionality
function saveSnapshot() {
  if (historyStack.length >= MAX_HISTORY) {
    historyStack.shift();
  }
  historyStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
}

function undo() {
  if (historyStack.length > 0) {
    const prevState = historyStack.pop();
    ctx.putImageData(prevState, 0, 0);
    debouncedPredict(50);
  } else {
    clearCanvas();
  }
}

undoBtn.addEventListener('click', undo);

// Coordinate Calculation
function posFromEvent(e) {
  const rect = canvas.getBoundingClientRect();
  const touch = e.touches && e.touches[0];
  const clientX = touch ? touch.clientX : e.clientX;
  const clientY = touch ? touch.clientY : e.clientY;
  return {
    x: (clientX - rect.left) * (canvas.width / rect.width),
    y: (clientY - rect.top) * (canvas.height / rect.height)
  };
}

function drawLine(p1, p2) {
  ctx.beginPath();
  ctx.moveTo(p1.x, p1.y);
  ctx.lineTo(p2.x, p2.y);
  ctx.stroke();
}

// Mouse & Touch Event Handlers
function onDown(e) {
  e.preventDefault();
  saveSnapshot();
  drawing = true;
  lastPoint = posFromEvent(e);
  
  // Draw single point for clicks/taps
  ctx.beginPath();
  ctx.arc(lastPoint.x, lastPoint.y, ctx.lineWidth / 2, 0, Math.PI * 2);
  ctx.fillStyle = '#000000';
  ctx.fill();
}

function onMove(e) {
  if (!drawing) return;
  e.preventDefault();
  const currentPoint = posFromEvent(e);
  drawLine(lastPoint, currentPoint);
  lastPoint = currentPoint;
  
  if (rt.checked) {
    debouncedPredict(150);
  }
}

function onUp(e) {
  if (!drawing) return;
  drawing = false;
  lastPoint = null;
  
  if (rt.checked) {
    predict();
  }
}

canvas.addEventListener('mousedown', onDown);
canvas.addEventListener('mousemove', onMove);
window.addEventListener('mouseup', onUp);

canvas.addEventListener('touchstart', onDown, { passive: false });
canvas.addEventListener('touchmove', onMove, { passive: false });
canvas.addEventListener('touchend', onUp);

// Clear Canvas
function clearCanvas() {
  saveSnapshot();
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  setPrediction(null);
  updateStatus('Cleared');
}

clearBtn.addEventListener('click', clearCanvas);
predictBtn.addEventListener('click', () => predict());

// Keyboard Shortcuts
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;

  if (e.key === 'c' || e.key === 'C') {
    clearCanvas();
  } else if ((e.key === 'z' || e.key === 'Z') && !e.shiftKey) {
    undo();
  } else if (e.key === 'Enter') {
    predict();
  }
});

// UI Probability & Prediction Rendering
function buildInitialBars() {
  barsEl.innerHTML = '';
  for (let i = 0; i < 10; i++) {
    const item = document.createElement('div');
    item.className = 'prob-item';
    item.id = `prob-row-${i}`;

    item.innerHTML = `
      <span class="prob-badge">${i}</span>
      <div class="prob-track">
        <div class="prob-bar" id="bar-${i}" style="width: 0%"></div>
      </div>
      <span class="prob-val" id="val-${i}">0.0%</span>
      <span class="prob-tag" id="tag-${i}">TOP</span>
    `;

    barsEl.appendChild(item);
  }
}

function setPrediction(res) {
  if (!res || res.prediction === '-' || res.prediction === null) {
    predEl.textContent = '—';
    predEl.style.color = '#ffffff';
    confidenceVal.textContent = '0.0%';
    confidenceFill.style.width = '0%';

    for (let i = 0; i < 10; i++) {
      const row = document.getElementById(`prob-row-${i}`);
      const bar = document.getElementById(`bar-${i}`);
      const val = document.getElementById(`val-${i}`);
      if (row) row.classList.remove('is-top');
      if (bar) bar.style.width = '0%';
      if (val) val.textContent = '0.0%';
    }
    return;
  }

  const predictedDigit = Number(res.prediction);
  predEl.textContent = String(predictedDigit);
  predEl.style.color = '#38bdf8';

  const confPercent = res.confidence ? (res.confidence * 100).toFixed(1) : ((res.probabilities[predictedDigit] || 0) * 100).toFixed(1);
  confidenceVal.textContent = `${confPercent}%`;
  confidenceFill.style.width = `${Math.min(100, Math.max(0, confPercent))}%`;

  if (Array.isArray(res.probabilities)) {
    res.probabilities.forEach((p, i) => {
      const row = document.getElementById(`prob-row-${i}`);
      const bar = document.getElementById(`bar-${i}`);
      const val = document.getElementById(`val-${i}`);
      const isTop = i === predictedDigit && p > 0.15;

      if (row) {
        if (isTop) {
          row.classList.add('is-top');
        } else {
          row.classList.remove('is-top');
        }
      }

      const percent = (p * 100).toFixed(1);
      if (bar) {
        bar.style.width = `${Math.min(100, Math.max(0, p * 100))}%`;
      }
      if (val) {
        val.textContent = `${percent}%`;
      }
    });
  }
}

function updateStatus(text, isActive = false) {
  inferenceStatus.textContent = text;
  if (isActive) {
    inferenceStatus.classList.add('active');
  } else {
    inferenceStatus.classList.remove('active');
  }
}

// Client-Side MNIST Preprocessor (Center of Mass + 20x20 aspect scaling)
function preprocessCanvasTo28x28(srcCanvas) {
  const sCtx = srcCanvas.getContext('2d');
  const imgData = sCtx.getImageData(0, 0, srcCanvas.width, srcCanvas.height);
  const data = imgData.data;

  let minX = srcCanvas.width, minY = srcCanvas.height, maxX = 0, maxY = 0;
  let hasInk = false;
  let inkCount = 0;

  for (let y = 0; y < srcCanvas.height; y++) {
    for (let x = 0; x < srcCanvas.width; x++) {
      const idx = (y * srcCanvas.width + x) * 4;
      // White canvas (255) with black strokes (0) -> invert
      const gray = 0.299 * data[idx] + 0.587 * data[idx+1] + 0.114 * data[idx+2];
      const intensity = 255 - gray;

      if (intensity > 35) {
        hasInk = true;
        inkCount++;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  // Reject noise / empty
  if (!hasInk || inkCount < 20) return null;

  const cropW = Math.max(1, maxX - minX + 1);
  const cropH = Math.max(1, maxY - minY + 1);
  const scale = 20.0 / Math.max(cropW, cropH);
  const newW = Math.max(1, Math.round(cropW * scale));
  const newH = Math.max(1, Math.round(cropH * scale));

  // Offscreen canvas for aspect-ratio scaling
  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = cropW;
  cropCanvas.height = cropH;
  const cCtx = cropCanvas.getContext('2d');
  cCtx.drawImage(srcCanvas, minX, minY, cropW, cropH, 0, 0, cropW, cropH);

  const scaledCanvas = document.createElement('canvas');
  scaledCanvas.width = newW;
  scaledCanvas.height = newH;
  const scCtx = scaledCanvas.getContext('2d');
  scCtx.imageSmoothingEnabled = true;
  scCtx.imageSmoothingQuality = 'high';
  scCtx.drawImage(cropCanvas, 0, 0, newW, newH);

  // 28x28 canvas with centered paste
  const finalCanvas = document.createElement('canvas');
  finalCanvas.width = 28;
  finalCanvas.height = 28;
  const fCtx = finalCanvas.getContext('2d');
  fCtx.fillStyle = '#000000';
  fCtx.fillRect(0, 0, 28, 28);

  const pasteX = Math.floor((28 - newW) / 2);
  const pasteY = Math.floor((28 - newH) / 2);

  const scaledImgData = scCtx.getImageData(0, 0, newW, newH).data;
  const finalImgData = fCtx.getImageData(0, 0, 28, 28);
  const finalPixels = finalImgData.data;

  let totalMass = 0;
  let cx = 0, cy = 0;

  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const sIdx = (y * newW + x) * 4;
      const gray = 0.299 * scaledImgData[sIdx] + 0.587 * scaledImgData[sIdx+1] + 0.114 * scaledImgData[sIdx+2];
      const inv = 255 - gray;

      const fx = pasteX + x;
      const fy = pasteY + y;
      if (fx >= 0 && fx < 28 && fy >= 0 && fy < 28) {
        const fIdx = (fy * 28 + fx) * 4;
        finalPixels[fIdx] = inv;
        finalPixels[fIdx+1] = inv;
        finalPixels[fIdx+2] = inv;
        finalPixels[fIdx+3] = 255;

        totalMass += inv;
        cx += fx * inv;
        cy += fy * inv;
      }
    }
  }

  // Calculate center of mass shift to (14, 14)
  let shiftX = 0, shiftY = 0;
  if (totalMass > 0) {
    cx /= totalMass;
    cy /= totalMass;
    shiftX = Math.round(14 - cx);
    shiftY = Math.round(14 - cy);
    shiftX = Math.max(-4, Math.min(4, shiftX));
    shiftY = Math.max(-4, Math.min(4, shiftY));
  }

  // Create Float32Array (1, 28, 28, 1) normalized [0, 1]
  const floatArr = new Float32Array(28 * 28);
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const srcX = x - shiftX;
      const srcY = y - shiftY;
      let val = 0;
      if (srcX >= 0 && srcX < 28 && srcY >= 0 && srcY < 28) {
        const idx = (srcY * 28 + srcX) * 4;
        val = finalPixels[idx] / 255.0;
      }
      floatArr[y * 28 + x] = val;
    }
  }

  return floatArr;
}

// Load TensorFlow.js Model directly into Browser
async function loadClientModel() {
  if (typeof tf === 'undefined') {
    console.warn('TensorFlow.js CDN not ready yet.');
    return;
  }
  isModelLoading = true;
  updateStatus('Loading AI...');

  try {
    localModel = await tf.loadLayersModel('/static/model/model.json');
    console.log('Client-side TensorFlow.js model loaded successfully!');
    updateStatus('AI Ready', true);
  } catch (err) {
    console.warn('Could not load client-side model from /static/model/model.json, using backend API:', err);
    updateStatus('Server Ready', true);
  } finally {
    isModelLoading = false;
  }
}

// Combined Inference (Client-Side First, Server Fallback)
let isPredicting = false;
async function predict() {
  if (isPredicting) return;
  isPredicting = true;
  const startTime = performance.now();

  try {
    const inputFloatArr = preprocessCanvasTo28x28(canvas);
    if (!inputFloatArr) {
      setPrediction(null);
      updateStatus('Empty Canvas', false);
      isPredicting = false;
      return;
    }

    // 1. Try Client-side inference via TensorFlow.js
    if (localModel && typeof tf !== 'undefined') {
      const tensor = tf.tensor4d(inputFloatArr, [1, 28, 28, 1]);
      const output = localModel.predict(tensor);
      const probabilities = Array.from(await output.data());
      tensor.dispose();
      output.dispose();

      let maxIdx = 0;
      let maxProb = probabilities[0];
      for (let i = 1; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
          maxProb = probabilities[i];
          maxIdx = i;
        }
      }

      const latency = Math.round(performance.now() - startTime);
      setPrediction({
        prediction: maxIdx,
        confidence: maxProb,
        probabilities: probabilities
      });
      updateStatus(`${latency}ms (Live)`, true);
      isPredicting = false;
      return;
    }

    // 2. Fallback to Server API (/predict or /api/predict)
    updateStatus('Inferring...', true);
    const dataURL = canvas.toDataURL('image/png');
    
    // Try /predict then /api/predict
    let res;
    try {
      res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataURL })
      });
    } catch {
      res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataURL })
      });
    }

    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const json = await res.json();
    const latency = Math.round(performance.now() - startTime);
    setPrediction(json);
    updateStatus(`${latency}ms`, true);
  } catch (err) {
    console.error('Prediction failed:', err);
    updateStatus('Error', false);
  } finally {
    isPredicting = false;
  }
}

// Debouncer
let debounceTimer = null;
function debouncedPredict(delay = 180) {
  if (!rt.checked) return;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(predict, delay);
}

// Bootstrap
buildInitialBars();
initCanvas();
setPrediction(null);

// Initialize TensorFlow.js model in background
if (document.readyState === 'complete') {
  loadClientModel();
} else {
  window.addEventListener('load', loadClientModel);
}
