import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras import layers, models

LABEL_NAMES = ["no_person", "person"]
SAMPLES = 50

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("pir_data.csv")
X = df.drop("label", axis=1).values.astype(np.float32)   # (N, 50), values 0/1
y = df["label"].values

# ── 2. Split ──────────────────────────────────────────────────────────────────
# No normalization: the PIR output is already 0/1, so the ESP32 can feed raw
# samples straight in (and the descriptor needs no mean/std fields).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train = X_train.reshape(-1, SAMPLES, 1)
X_test  = X_test.reshape(-1, SAMPLES, 1)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── 3. Build model ────────────────────────────────────────────────────────────
# Deliberately minimal: Flatten -> Dense(2) -> Softmax (102 parameters).
# Uses only ops the universal interpreter already supports.
model = models.Sequential([
    layers.Input(shape=(SAMPLES, 1)),
    layers.Flatten(),
    layers.Dense(2, activation="softmax"),
])
model.summary()

# ── 4. Train ──────────────────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)],
)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc*100:.1f}%")

# Rule-based baseline for the paper: "any HIGH sample in the window -> person".
# Reviewers will ask how the model compares to a simple threshold - report both.
rule_pred = (X_test.reshape(len(X_test), -1).max(axis=1) > 0).astype(int)
print(f"Rule baseline (any HIGH -> person) accuracy: {(rule_pred == y_test).mean()*100:.1f}%")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(cmap="Blues")
plt.title("Confusion Matrix (PIR presence)")
plt.savefig("pir_confusion_matrix.png", dpi=150)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history["accuracy"], label="train")
ax1.plot(history.history["val_accuracy"], label="val")
ax1.set_title("Accuracy"); ax1.legend()
ax2.plot(history.history["loss"], label="train")
ax2.plot(history.history["val_loss"], label="val")
ax2.set_title("Loss"); ax2.legend()
plt.savefig("pir_training_curves.png", dpi=150)
plt.show()

# ── 6. Save model ─────────────────────────────────────────────────────────────
model.save("pir_model.keras")
print("Model saved.")
