import pandas as pd
import numpy as np

LABELS = {0: "no_person", 1: "person"}

df = pd.read_csv("pir_data.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

print(f"Total windows: {len(df)}")
print(f"Shape: {df.shape}")
print(f"Any missing values: {df.isnull().sum().sum()}")

print("\nWindows per class:")
print(df["label"].map(LABELS).value_counts())

print("\nFraction of HIGH samples per window (mean / min / max):")
for k, name in LABELS.items():
    frac = X[y == k].mean(axis=1)
    print(f"  {name:10s}  {frac.mean():.2f} / {frac.min():.2f} / {frac.max():.2f}")

# Possible label noise: false triggers while empty, or no motion while "person"
noisy_empty = np.where((y == 0) & (X.sum(axis=1) > 0))[0]
quiet_person = np.where((y == 1) & (X.sum(axis=1) == 0))[0]
print(f"\nno_person windows with any HIGH sample: {len(noisy_empty)}  rows {noisy_empty.tolist()[:20]}")
print(f"person windows with no HIGH sample:     {len(quiet_person)}  rows {quiet_person.tolist()[:20]}")
print("(A few of each are normal. Many means the room wasn't really empty, or you were too still.)")
