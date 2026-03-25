import tensorflow as tf

keras_path = r"D:\SP26\DSP\RAG_DSP\best_rice_model.keras"
model = tf.keras.models.load_model(keras_path)

print("\n--- METRICS/CONFIG ---")
config = model.get_config()
print("Model name:", config.get("name"))

# Sometimes metrics have names like "accuracy_Blight" etc.
for m in model.metrics:
    print("Metric:", m.name)

# Last layer config
last_layer = model.layers[-1]
print("\n--- LAST LAYER ---")
print(last_layer.name, last_layer.get_config())

# Maybe the classes are saved in the optimizer or loss?
print("\n--- LOSS ---")
try:
    print(model.loss)
except:
    pass