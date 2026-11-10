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
    debouncedPredict(220);
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
  // Ignore inside inputs
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

// Inference Request with latency tracking
let isPredicting = false;
async function predict() {
  if (isPredicting) return;
  isPredicting = true;
  updateStatus('Analyzing...', true);
  const startTime = performance.now();

  try {
    const dataURL = canvas.toDataURL('image/png');
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataURL })
    });

    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const json = await res.json();
    
    const latency = Math.round(performance.now() - startTime);
    setPrediction(json);
    
    if (json.prediction === '-' || json.prediction === null) {
      updateStatus('Empty Canvas', false);
    } else {
      updateStatus(`${latency}ms`, true);
    }
  } catch (err) {
    console.error('Prediction failed:', err);
    updateStatus('Offline', false);
  } finally {
    isPredicting = false;
  }
}

// Debouncer
let debounceTimer = null;
function debouncedPredict(delay = 200) {
  if (!rt.checked) return;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(predict, delay);
}

// Bootstrap
buildInitialBars();
initCanvas();
setPrediction(null);
