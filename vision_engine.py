import io
import time
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path

# Load TFLite
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# ── EfficientNet B0 TFLite Model ──
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_rice_model_B0.tflite"

print(f"Loading TFLite Vision Model ({MODEL_PATH.name})...")
try:
    interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite Model loaded successfully.")
    is_loaded = True
except Exception as e:
    import traceback
    err_str = traceback.format_exc()
    print(f"Error loading TFLite model: {err_str}")
    is_loaded = False
    load_err = err_str
    interpreter = None

CLASS_NAMES = [
    "bacterial_leaf_blight",
    "brown_spot",
    "healthy",
    "leaf_blast",
    "leaf_scald",
    "narrow_brown_spot",
    "neck_blast",
    "rice_hispa",
    "sheath_blight",
    "tungro"
]

VIETNAMESE_MAPPING = {
    "bacterial_leaf_blight": "bệnh bạc lá (bạc lá vi khuẩn)",
    "brown_spot":            "bệnh đốm nâu",
    "healthy":               "cây lúa khỏe mạnh",
    "leaf_blast":            "bệnh đạo ôn lá",
    "leaf_scald":            "bệnh cháy bìa lá",
    "narrow_brown_spot":     "bệnh đốm nâu hẹp",
    "neck_blast":            "bệnh đạo ôn cổ bông",
    "rice_hispa":            "bọ cánh cứng gai hại lúa (rice hispa)",
    "sheath_blight":         "bệnh khô vằn",
    "tungro":                "bệnh vàng lùn (tungro)"
}

LOW_CONFIDENCE_THRESHOLD = 0.80

def _enhance_image(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    return img

def predict_disease(image_bytes: bytes) -> dict:
    if not is_loaded:
        return {"error": f"Vision model is not loaded. Details: {load_err}"}

    try:
        IMG_SIZE = 224  # EfficientNet B0 standard size (or matching your training. TFLite converted model typically expects 224x224 or 380x380 depending on what was frozen. Let's dynamically read from input_details)
        _, height, width, _ = input_details[0]['shape']

        # Load + enhance image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = _enhance_image(img)
        img = img.resize((width, height))

        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        # EfficientNet V2 / B0 Keras expects raw pixels [0-255], no manual scaling needed.

        # Run inference
        t_start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        inference_time_s = round(time.perf_counter() - t_start, 3)

        probs = predictions[0]

        # Top-3
        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = [
            {
                "english_label":    CLASS_NAMES[i],
                "vietnamese_label": VIETNAMESE_MAPPING.get(CLASS_NAMES[i], CLASS_NAMES[i]),
                "confidence":       round(float(probs[i]), 4),
            }
            for i in top3_indices
        ]

        best = top3[0]
        low_conf = best["confidence"] < LOW_CONFIDENCE_THRESHOLD

        return {
            "english_label":    best["english_label"],
            "vietnamese_label": best["vietnamese_label"],
            "confidence":       best["confidence"],
            "low_confidence":   low_conf,
            "top3":             top3,
            "inference_time_s": inference_time_s,
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print("Classes mapped to Vietnamese:")
    for eng, vi in VIETNAMESE_MAPPING.items():
        print(f" - {eng} -> {vi}")
