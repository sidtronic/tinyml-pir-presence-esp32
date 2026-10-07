// HC-SR501 presence inference on ESP32 (standalone test, weights compiled in).
// Copy pir_inference_engine.h (from export_engine.py) into this folder first.
//
// Samples the PIR at 10 Hz into a 50-sample (5 s) sliding window and runs
// the model once per second. Prints label, confidence and inference time.

#include "pir_inference_engine.h"

#define PIR_PIN    34
#define LED_PIN    2
#define SAMPLES    50
#define SAMPLE_HZ  10
#define INFER_EVERY 10            // run every 10 samples = 1 s
const uint32_t PERIOD_US = 1000000UL / SAMPLE_HZ;
const uint32_t WARMUP_MS = 60000;

uint8_t ring[SAMPLES];
int head = 0;          // next write position
int filled = 0;
int sinceInfer = 0;
uint32_t nextSample;

void setup() {
  Serial.begin(115200);
  pinMode(PIR_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
  Serial.println("Warming up PIR (60 s)... send 's' to skip");
  while (millis() < WARMUP_MS) {
    if (Serial.available() && Serial.read() == 's') break;
    delay(10);
  }
  Serial.println("Running.");
  nextSample = micros();
}

void loop() {
  if ((int32_t)(micros() - nextSample) < 0) return;
  nextSample += PERIOD_US;

  ring[head] = digitalRead(PIR_PIN);
  head = (head + 1) % SAMPLES;
  if (filled < SAMPLES) filled++;
  if (filled < SAMPLES || ++sinceInfer < INFER_EVERY) return;
  sinceInfer = 0;

  // Unroll ring buffer oldest -> newest (same order as training data)
  float window[SAMPLES];
  for (int i = 0; i < SAMPLES; i++) window[i] = ring[(head + i) % SAMPLES];

  float scores[PIR_N_OUT];
  uint32_t t0 = micros();
  predict(window, scores);
  uint32_t dt = micros() - t0;

  int best = scores[1] > scores[0] ? 1 : 0;
  digitalWrite(LED_PIN, best == 1);
  Serial.printf("%-9s  conf=%.2f  infer=%lu us\n", PIR_LABELS[best], scores[best], (unsigned long)dt);
}
