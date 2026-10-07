import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

SAMPLES = 50

# ── 1. Load data (same split as train_model.py) ───────────────────────────────
df = pd.read_csv("pir_data.csv")
X = df.drop("label", axis=1).values.astype(np.float32).reshape(-1, SAMPLES, 1)
y = df["label"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def representative_dataset():
    for i in range(len(X_train)):
        yield [X_train[i:i+1]]

# ── 2. Convert to TFLite with full int8 quantization ──────────────────────────
# Only used as the TFLite reference point (model size / accuracy) for the paper;
# the ESP32 runs the custom engine from export_engine.py.
model = tf.keras.models.load_model("pir_model.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

with open("pir_model.tflite", "wb") as f:
    f.write(tflite_model)
print(f"TFLite model size: {len(tflite_model)} bytes ({len(tflite_model)/1024:.2f} KB)")

# ── 3. Verify quantized accuracy on the held-out test set ─────────────────────
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
scale, zero_point = inp["quantization"]

correct = 0
for i in range(len(X_test)):
    q = np.clip(np.round(X_test[i:i+1] / scale + zero_point), -128, 127).astype(np.int8)
    interpreter.set_tensor(inp["index"], q)
    interpreter.invoke()
    if np.argmax(interpreter.get_tensor(out["index"])) == y_test[i]:
        correct += 1
print(f"TFLite int8 test accuracy: {correct/len(X_test)*100:.1f}%")
