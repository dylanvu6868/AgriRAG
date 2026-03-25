import tensorflow as tf
import os

print("Đang nạp model Keras...")
model = tf.keras.models.load_model("best_rice_model.keras")

print("Đang biên dịch sang TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # Lượng tử hóa nhẹ (Quantization) để giảm RAM
tflite_model = converter.convert()

print("Đang lưu ra file best_rice_model_B0.tflite...")
with open("best_rice_model_B0.tflite", "wb") as f:
    f.write(tflite_model)

print("Xong! Đã tạo file TFLite thành công.")
