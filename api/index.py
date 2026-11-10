import os
import io
import base64
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import scipy.ndimage
import numpy as np

app = FastAPI(title="MNIST Digit Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_TFLITE = BASE_DIR / "mnist_cnn.tflite"

interpreter = None

def get_interpreter():
    global interpreter
    if interpreter is not None:
        return interpreter

    # Try LiteRT / TFLite runtime first (for Vercel serverless)
    try:
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=str(MODEL_TFLITE))
    except Exception:
        try:
            from tflite_runtime.interpreter import Interpreter
            interpreter = Interpreter(model_path=str(MODEL_TFLITE))
        except Exception:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=str(MODEL_TFLITE))

    interpreter.allocate_tensors()
    return interpreter

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
    cy, cx = scipy.ndimage.center_of_mass(arr)
    if not (np.isnan(cy) or np.isnan(cx)):
        shift_x = int(np.clip(np.round(14.0 - cx), -4, 4))
        shift_y = int(np.clip(np.round(14.0 - cy), -4, 4))
        arr = scipy.ndimage.shift(arr, (shift_y, shift_x), mode='constant', cval=0.0)

    # Normalize to [0, 1]
    arr_normalized = (np.clip(arr, 0.0, 255.0) / 255.0).astype(np.float32)
    return arr_normalized.reshape(1, 28, 28, 1)

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

    interp = get_interpreter()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    interp.set_tensor(input_details[0]['index'], processed)
    interp.invoke()
    preds = interp.get_tensor(output_details[0]['index'])[0]

    predicted_digit = int(np.argmax(preds))
    confidence = float(np.max(preds))
    probabilities = [float(p) for p in preds]

    return {
        "prediction": predicted_digit,
        "confidence": confidence,
        "probabilities": probabilities
    }
