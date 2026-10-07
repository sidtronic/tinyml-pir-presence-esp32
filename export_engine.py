import json
import struct
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

SAMPLES = 50
LABEL_NAMES = ["no_person", "person"]

model = tf.keras.models.load_model("pir_model.keras")

# Model is Flatten -> Dense(2, softmax); the Dense layer is the only one with weights
dense = [l for l in model.layers if l.get_weights()][0]
W, b = dense.get_weights()            # W: (50, 2), b: (2,)
n_in, n_out = W.shape

# ── INT8 weights: symmetric per-tensor quantization ───────────────────────────
#   w_q = round(w / scale),  scale = max|w| / 127,   w ≈ scale * w_q
# Bias stays float32 (2 values). This is the format that goes into the EEPROM.
w_scale = float(np.max(np.abs(W)) / 127.0)
W_q = np.clip(np.round(W / w_scale), -127, 127).astype(np.int8)

def predict_like_esp32(x):
    """Exactly the math the C engine does: acc = b + scale * sum(x * w_q)."""
    logits = b + w_scale * (x @ W_q.astype(np.float32))
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

# ── Verify that INT8 storage didn't cost accuracy ─────────────────────────────
df = pd.read_csv("pir_data.csv")
X = df.drop("label", axis=1).values.astype(np.float32)
y = df["label"].values
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
float_acc = (np.argmax(model.predict(X_test.reshape(-1, SAMPLES, 1), verbose=0), 1) == y_test).mean()
int8_acc  = (np.argmax(predict_like_esp32(X_test), 1) == y_test).mean()
print(f"Float Keras test accuracy:      {float_acc*100:.1f}%")
print(f"INT8-weight engine test accuracy: {int8_acc*100:.1f}%")

# ── 1. Weight blob for the EEPROM / descriptor packer ─────────────────────────
#   [ int8  W[n_in*n_out]  row-major (W[i][o] at i*n_out + o) ]
#   [ float32 b[n_out]     little-endian                      ]
blob = W_q.tobytes() + struct.pack(f"<{n_out}f", *b.tolist())
with open("pir_weights.bin", "wb") as f:
    f.write(blob)

meta = {
    "sensor": "HC-SR501",
    "interface": "digital",          # goes on AOUT; descriptor interface byte = analog/digital pin
    "sample_rate_hz": 10,
    "window_samples": SAMPLES,
    "channels": 1,
    "normalization": "none (raw 0/1)",
    "labels": LABEL_NAMES,
    "layers": [
        {"op": "Flatten", "in": [SAMPLES, 1], "out": [SAMPLES]},
        {"op": "Dense", "in": n_in, "out": n_out, "activation": "none",
         "weights": {"dtype": "int8", "quant": "symmetric_per_tensor",
                     "scale": w_scale, "zero_point": 0,
                     "offset": 0, "bytes": int(W_q.size)},
         "bias": {"dtype": "float32", "offset": int(W_q.size), "bytes": 4 * n_out}},
        {"op": "Softmax", "n": n_out},
    ],
    "total_weight_bytes": len(blob),
    "test_accuracy_float": round(float(float_acc), 4),
    "test_accuracy_int8": round(float(int8_acc), 4),
}
with open("pir_model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

# ── 2. Standalone C header (for testing on the ESP32 before the EEPROM path) ──
def c_array(name, arr, ctype, fmt):
    vals = ", ".join(fmt(v) for v in arr.flatten().tolist())
    return f"const {ctype} {name}[{arr.size}] = {{{vals}}};\n"

header = f"""#pragma once
#include <math.h>
#include <stdint.h>

// Auto-generated PIR presence engine: Flatten -> Dense({n_in}->{n_out}) -> Softmax
// Weights stored as INT8 (symmetric per-tensor), bias as float32.
// Input: {SAMPLES} raw PIR samples (0/1) @ 10 Hz. No normalization.

#define PIR_N_IN  {n_in}
#define PIR_N_OUT {n_out}
const float DENSE_W_SCALE = {w_scale:.9g}f;
"""
header += c_array("DENSE_W_Q", W_q, "int8_t", lambda v: str(v))
header += c_array("DENSE_B", b, "float", lambda v: f"{v:.6f}f")
header += """
const char* PIR_LABELS[PIR_N_OUT] = {"no_person", "person"};

void softmax(float* x, int n) {
  float m = x[0];
  for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
  float s = 0;
  for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
  for (int i = 0; i < n; i++) x[i] /= s;
}

// Dense with INT8 weights: out[o] = b[o] + scale * sum_i in[i] * w_q[i][o]
void dense_int8(const float* in, int n_in, const int8_t* w_q, float w_scale,
                const float* bias, float* out, int n_out) {
  for (int o = 0; o < n_out; o++) {
    float acc = 0;
    for (int i = 0; i < n_in; i++) acc += in[i] * (float)w_q[i * n_out + o];
    out[o] = bias[o] + w_scale * acc;
  }
}

// input: float[50] of 0/1 samples (Flatten is a no-op on a 1-channel window)
void predict(const float* input, float* scores) {
  dense_int8(input, PIR_N_IN, DENSE_W_Q, DENSE_W_SCALE, DENSE_B, scores, PIR_N_OUT);
  softmax(scores, PIR_N_OUT);
}
"""
with open("inference/pir_inference_engine.h", "w") as f:
    f.write(header)

print(f"\nWeight blob:  pir_weights.bin  ({len(blob)} bytes: {W_q.size} int8 W + {4*n_out} float32 bias)")
print(f"Weight scale: {w_scale:.9g}")
print("Metadata:     pir_model_meta.json")
print("C header:     inference/pir_inference_engine.h")
