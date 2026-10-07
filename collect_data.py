import serial
import csv
import time
import os
import pandas as pd

PORT = "COM5"          # Change this to your ESP32's port
BAUD = 115200
SAMPLES = 50           # 50 samples @ 10 Hz = 5 s window
SAMPLES_PER_CLASS = 150
SKIP_WARMUP = False    # True only if the PIR has already been powered > 60 s
OUTPUT_FILE = "pir_data.csv"

LABELS = {0: "no_person", 1: "person"}

INSTRUCTIONS = {
    "no_person": "Leave the sensor's field of view (or leave the room). "
                 "Recording starts after a 10 s countdown so the PIR's hold time can expire.",
    "person":    "Stay in view and move naturally: walk across, walk in/out, sit and fidget, wave. "
                 "Vary distance and direction between windows.",
}


def wait_for(ser, token, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        line = ser.readline().decode(errors="ignore").strip()
        if line.startswith("WARMUP"):
            print(f"    PIR warming up... {line.split()[1]} s left")
        if line == token:
            return True
    return False


def record_window(ser, label):
    ser.write(b"g")
    if not wait_for(ser, "START", 5):
        print("    No START received, retrying.")
        return None

    values = []
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line == "END" or line == "":
            break
        if line in ("0", "1"):
            values.append(int(line))

    if len(values) != SAMPLES:
        print(f"    Incomplete window ({len(values)}/{SAMPLES}), skipping.")
        return None
    return values + [label]


def existing_counts():
    if not os.path.exists(OUTPUT_FILE):
        return {k: 0 for k in LABELS}
    df = pd.read_csv(OUTPUT_FILE)
    return {k: int((df["label"] == k).sum()) for k in LABELS}


def main():
    counts = existing_counts()
    file_exists = os.path.exists(OUTPUT_FILE)

    ser = serial.Serial(PORT, BAUD, timeout=3)
    time.sleep(2)              # opening the port resets the ESP32
    ser.reset_input_buffer()
    if SKIP_WARMUP:
        ser.write(b"s")

    print("=== PIR Presence Data Collector ===")
    print("Waiting for the PIR to finish warming up (up to 60 s)...")
    if not wait_for(ser, "READY", 90):
        print("ESP32 never reported READY. Check the port and the sketch.")
        return

    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([f"s{i}" for i in range(SAMPLES)] + ["label"])

        for label, name in LABELS.items():
            collected = counts[label]
            if collected >= SAMPLES_PER_CLASS:
                print(f"\n{name}: already have {collected}, skipping.")
                continue

            print(f"\n{'=' * 50}")
            print(f"Class: {name.upper()}  ({collected}/{SAMPLES_PER_CLASS} already saved)")
            print(INSTRUCTIONS[name])
            print(f"{'=' * 50}")
            input(">>> Press Enter to start recording this class (Ctrl+C to stop any time)...")

            if name == "no_person":
                for s in range(10, 0, -1):
                    print(f"    Starting in {s} s - get out of view!", end="\r")
                    time.sleep(1)
                print()

            # Windows are recorded back-to-back; pause with Ctrl+C, re-run to resume.
            while collected < SAMPLES_PER_CLASS:
                row = record_window(ser, label)
                if row:
                    writer.writerow(row)
                    f.flush()
                    collected += 1
                    highs = sum(row[:-1])
                    print(f"    Saved {collected}/{SAMPLES_PER_CLASS}   HIGH samples in window: {highs:2d}/50")

    print(f"\nDone! {OUTPUT_FILE} is ready.")
    ser.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Everything recorded so far is saved; run again to resume.")
