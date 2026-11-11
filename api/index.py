import os
import io
import base64
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import numpy as np

app = FastAPI(title="MNIST Digit Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent

# Locate model files
WEIGHTS_NPZ = CURRENT_DIR / "mnist_weights.npz"
if not WEIGHTS_NPZ.exists():
    WEIGHTS_NPZ = BASE_DIR / "mnist_weights.npz"

STATIC_DIR = CURRENT_DIR / "static"
if not STATIC_DIR.exists():
    STATIC_DIR = BASE_DIR / "static"

INDEX_FILE = CURRENT_DIR / "index.html"
if not INDEX_FILE.exists():
    INDEX_FILE = BASE_DIR / "index.html"

# Mount static files if present
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Load weights for pure NumPy ultra-fast inference (no heavy TF or C++ deps)
w_conv1 = None
b_conv1 = None
w_conv2 = None
b_conv2 = None
w_dense1 = None
b_dense1 = None
w_dense2 = None
b_dense2 = None

def load_weights():
    global w_conv1, b_conv1, w_conv2, b_conv2, w_dense1, b_dense1, w_dense2, b_dense2
    if w_conv1 is not None:
        return
    data = np.load(str(WEIGHTS_NPZ))
    w_conv1 = data['layer_0_weights']
    b_conv1 = data['layer_0_biases']
    w_conv2 = data['layer_2_weights']
    b_conv2 = data['layer_2_biases']
    w_dense1 = data['layer_5_weights']
    b_dense1 = data['layer_5_biases']
    w_dense2 = data['layer_7_weights']
    b_dense2 = data['layer_7_biases']

def conv2d(x, w, b):
    H, W, Cin = x.shape
    kH, kW, _, Cout = w.shape
    outH, outW = H - kH + 1, W - kW + 1
    out = np.zeros((outH, outW, Cout), dtype=np.float32)
    for i in range(kH):
        for j in range(kW):
            x_slice = x[i:i+outH, j:j+outW, :]
            out += np.tensordot(x_slice, w[i, j], axes=([2], [0]))
    out += b
    return np.maximum(out, 0)

def maxpool2d(x):
    H, W, C = x.shape
    outH, outW = H // 2, W // 2
    x_even = x[:outH * 2, :outW * 2, :]
    return x_even.reshape(outH, 2, outW, 2, C).max(axis=(1, 3))

def predict_cnn_numpy(x_28_28_1):
    load_weights()
    c1 = conv2d(x_28_28_1, w_conv1, b_conv1)
    p1 = maxpool2d(c1)
    c2 = conv2d(p1, w_conv2, b_conv2)
    p2 = maxpool2d(c2)
    flat = p2.flatten()
    d1 = np.maximum(np.dot(flat, w_dense1) + b_dense1, 0)
    logits = np.dot(d1, w_dense2) + b_dense2
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def center_of_mass(arr):
    total = np.sum(arr)
    if total == 0:
        return 14.0, 14.0
    y_coords, x_coords = np.indices(arr.shape)
    cy = np.sum(y_coords * arr) / total
    cx = np.sum(x_coords * arr) / total
    return cy, cx

def shift_2d(arr, sy, sx):
    res = np.zeros_like(arr)
    H, W = arr.shape
    y1, y2 = max(0, sy), min(H, H + sy)
    x1, x2 = max(0, sx), min(W, W + sx)
    sy1, sy2 = max(0, -sy), min(H, H - sy)
    sx1, sx2 = max(0, -sx), min(W, W - sx)
    res[y1:y2, x1:x2] = arr[sy1:sy2, sx1:sx2]
    return res

def preprocess_mnist_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img_arr = np.array(img)

    # Invert if user drew black on white
    if np.mean(img_arr) > 127:
        img = ImageOps.invert(img)
        img_arr = np.array(img)

    # Guard: empty or noise only
    if np.max(img_arr) < 30 or np.sum(img_arr > 40) < 20:
        return None

    # Get bounding box of drawn digit
    thresh = img.point(lambda p: 255 if p > 30 else 0)
    bbox = thresh.getbbox()
    if not bbox:
        return None

    crop = img.crop(bbox)
    w, h = crop.size
    if w == 0 or h == 0:
        return None

    # Scale to 20x20 box preserving aspect ratio
    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Paste into 28x28 black canvas
    centered_img = Image.new('L', (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    centered_img.paste(resized, (paste_x, paste_y))

    # Center of mass alignment
    arr = np.array(centered_img, dtype=np.float32)
    cy, cx = center_of_mass(arr)
    shift_x = int(np.clip(np.round(14.0 - cx), -4, 4))
    shift_y = int(np.clip(np.round(14.0 - cy), -4, 4))
    arr = shift_2d(arr, shift_y, shift_x)

    # Normalize to [0, 1]
    arr_normalized = (np.clip(arr, 0.0, 255.0) / 255.0).astype(np.float32)
    return arr_normalized.reshape(28, 28, 1)

@app.get("/")
@app.get("/index.html")
async def serve_index():
    if INDEX_FILE.exists():
        content = INDEX_FILE.read_text(encoding="utf-8")
        return HTMLResponse(content=content)
    return HTMLResponse("<h1>MNIST Digit Classifier</h1>")

@app.get("/api/health")
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "mnist-digit-classifier"}

@app.post("/predict")
@app.post("/api/predict")
async def predict(request: Request):
    content_type = request.headers.get("content-type", "")
    image_bytes = None

    if "application/json" in content_type:
        body = await request.json()
        raw_image = body.get("image", "")
        if "," in raw_image:
            _, encoded = raw_image.split(",", 1)
        else:
            encoded = raw_image
        image_bytes = base64.b64decode(encoded)
    else:
        form = await request.form()
        file = form.get("file")
        if file:
            image_bytes = await file.read()

    if not image_bytes:
        return {"prediction": "-", "probabilities": [0.0] * 10, "confidence": 0.0}

    processed = preprocess_mnist_image(image_bytes)
    if processed is None:
        return {"prediction": "-", "probabilities": [0.0] * 10, "confidence": 0.0}

    probs = predict_cnn_numpy(processed)
    predicted_digit = int(np.argmax(probs))
    confidence = float(np.max(probs))
    probabilities = [float(p) for p in probs]

    return {
        "prediction": predicted_digit,
        "confidence": confidence,
        "probabilities": probabilities
    }
