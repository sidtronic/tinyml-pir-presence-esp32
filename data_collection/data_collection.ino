// HC-SR501 PIR data collection for the presence model.
// Protocol is the same as the MPU6050 collector:
//   PC sends 'g'  -> ESP32 prints "START", 50 lines of 0/1, then "END".
//   PC sends 's'  -> skip the warm-up wait (use only if the PIR has
//                    been powered for > 60 s already, e.g. after a reset).
//
// Window: 50 samples @ 10 Hz = 5.0 s

#define PIR_PIN    34      // GPIO34 = ADC1_CH6, input-only (this is "AOUT" on the module connector)
#define LED_PIN    2       // on-board LED, mirrors the PIR output
#define SAMPLES    50
#define SAMPLE_HZ  10
const uint32_t PERIOD_US = 1000000UL / SAMPLE_HZ;
const uint32_t WARMUP_MS = 60000;   // HC-SR501 needs ~30-60 s to settle after power-on

bool ready = false;

void setup() {
  Serial.begin(115200);
  pinMode(PIR_PIN, INPUT);          // HC-SR501 drives the line itself (no pull needed)
  pinMode(LED_PIN, OUTPUT);
  delay(200);
  Serial.println("BOOT");
}

void waitWarmup() {
  uint32_t lastPrint = 0;
  while (millis() < WARMUP_MS) {
    if (Serial.available() && Serial.read() == 's') break;
    if (millis() - lastPrint >= 5000) {
      lastPrint = millis();
      Serial.print("WARMUP ");
      Serial.println((WARMUP_MS - millis()) / 1000);
    }
    digitalWrite(LED_PIN, digitalRead(PIR_PIN));
    delay(10);
  }
  while (Serial.available()) Serial.read();   // drop anything sent during warm-up
  Serial.println("READY");
  ready = true;
}

void recordWindow() {
  Serial.println("START");
  uint32_t next = micros();
  for (int i = 0; i < SAMPLES; i++) {
    while ((int32_t)(micros() - next) < 0) { }   // fixed-rate sampling
    next += PERIOD_US;
    int v = digitalRead(PIR_PIN);
    digitalWrite(LED_PIN, v);
    Serial.println(v);
  }
  Serial.println("END");
}

void loop() {
  if (!ready) { waitWarmup(); return; }
  digitalWrite(LED_PIN, digitalRead(PIR_PIN));
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'g') recordWindow();
  }
}
