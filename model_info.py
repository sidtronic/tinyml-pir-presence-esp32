import tensorflow as tf

model = tf.keras.models.load_model("pir_model.keras")

total = 0
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    n = sum(w.size for w in weights)
    total += n
    print(f"Layer {i}: {layer.name:12s} weights shapes: {[w.shape for w in weights]}  params: {n}")

print(f"\nTotal parameters: {total}")
print(f"Float32 size: {total*4} bytes")
print(f"INT8 weights + float32 bias (as stored in EEPROM): see export_engine.py output")
