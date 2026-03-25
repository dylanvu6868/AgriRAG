import io
import time
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf

# ── EfficientNet B0 model (trained at 380×380) ──
MODEL_PATH = r"d:\SP26\DSP\RAG_DSP\best_rice_model_B0.keras"

print("Loading Rice Disease Vision Model...")
_model = None
try:
    _model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

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

# ── Confidence threshold to trigger top-3 display ──
LOW_CONFIDENCE_THRESHOLD = 0.80


def _enhance_image(img: Image.Image) -> Image.Image:
    """
    Apply light preprocessing to improve classification of low-quality images
    (screenshots, dark photos, blurry captures):
     - Auto-level contrast
     - Mild sharpening
    Uses only Pillow — no OpenCV dependency needed.
    """
    # Auto-level contrast (like CLAHE lite — stretches histogram)
    img = ImageEnhance.Contrast(img).enhance(1.3)
    # Mild sharpening
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    return img


def predict_disease(image_bytes: bytes) -> dict:
    """
    Takes raw image bytes, runs through the Keras model.
    Returns:
      - english_label, vietnamese_label, confidence  (top-1)
      - top3: list of {english_label, vietnamese_label, confidence}  (always)
      - low_confidence: True if top-1 confidence < LOW_CONFIDENCE_THRESHOLD
      - inference_time_s
    """
    if _model is None:
        return {"error": "Vision model is not loaded."}

    try:
        IMG_SIZE = 380  # Confirmed via model.input_shape

        # Load + enhance image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = _enhance_image(img)
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        # Run inference
        t_start = time.time()
        predictions = _model.predict(img_array, verbose=0)
        inference_time_s = round(time.time() - t_start, 3)

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
