// 24LC512 wiring check for the ESP32 base board.
// 1) Scans the I2C bus and lists every address that answers.
// 2) Does a non-destructive write/read test on the LAST 64 bytes of the
//    EEPROM (0xFFC0-0xFFFF): saves the original bytes, writes a pattern,
//    verifies it, then restores the original. Your descriptor at 0x0000 is untouched.
//
// Expected: 0x50 (EEPROM). With the MPU6050 module also on the bus: 0x68 too.

#include <Wire.h>

#define SDA_PIN     21
#define SCL_PIN     22
#define EEPROM_ADDR 0x50
#define TEST_ADDR   0xFFC0     // inside one 128-byte page
#define TEST_LEN    64

bool waitWriteDone() {          // ACK polling: the chip NACKs while it's writing (~5 ms)
  for (int i = 0; i < 50; i++) {
    Wire.beginTransmission(EEPROM_ADDR);
    if (Wire.endTransmission() == 0) return true;
    delay(1);
  }
  return false;
}

bool eepromWrite(uint16_t addr, const uint8_t* data, int len) {
  Wire.beginTransmission(EEPROM_ADDR);
  Wire.write(addr >> 8);
  Wire.write(addr & 0xFF);
  Wire.write(data, len);
  if (Wire.endTransmission() != 0) return false;
  return waitWriteDone();
}

bool eepromRead(uint16_t addr, uint8_t* data, int len) {
  Wire.beginTransmission(EEPROM_ADDR);
  Wire.write(addr >> 8);
  Wire.write(addr & 0xFF);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(EEPROM_ADDR, len) != len) return false;
  for (int i = 0; i < len; i++) data[i] = Wire.read();
  return true;
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Wire.begin(SDA_PIN, SCL_PIN, 100000);

  Serial.println("\n--- I2C scan ---");
  int found = 0;
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      Serial.printf("  device at 0x%02X%s\n", a,
                    a == 0x50 ? "  (24LC512)" : a == 0x68 ? "  (MPU6050)" : "");
      found++;
    }
  }
  if (!found) {
    Serial.println("  nothing found -> check SDA/SCL swap, pull-ups, 3.3V/GND to the EEPROM");
    return;
  }

  Serial.println("\n--- EEPROM read/write test ---");
  uint8_t orig[TEST_LEN], pattern[TEST_LEN], back[TEST_LEN];
  for (int i = 0; i < TEST_LEN; i++) pattern[i] = (uint8_t)(i * 7 + 0xA5);

  if (!eepromRead(TEST_ADDR, orig, TEST_LEN)) { Serial.println("  read FAILED"); return; }
  if (!eepromWrite(TEST_ADDR, pattern, TEST_LEN)) {
    Serial.println("  write FAILED -> is WP (pin 7) tied to GND?");
    return;
  }
  eepromRead(TEST_ADDR, back, TEST_LEN);
  bool ok = memcmp(pattern, back, TEST_LEN) == 0;
  eepromWrite(TEST_ADDR, orig, TEST_LEN);     // restore
  Serial.println(ok ? "  PASS: wrote, verified and restored 64 bytes"
                    : "  FAIL: read-back mismatch (WP high? loose wire?)");
}

void loop() {}
