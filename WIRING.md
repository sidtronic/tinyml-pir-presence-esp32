# Wiring: ESP32 + HC-SR501 + 24LC512

## Setup A: bench setup for data collection and standalone inference

Do this first. No EEPROM is needed to collect data and train.

| HC-SR501 pin | ESP32 DevKit v1 | Notes |
|---|---|---|
| VCC | **VIN (5 V from USB)** | HC-SR501 needs 4.5–20 V. Its onboard regulator drops out at 3.3 V |
| OUT | **GPIO 34** | Output HIGH is 3.3 V, so it's safe for the ESP32. GPIO 34 is input-only (ADC1_CH6) and is the same pin used as AOUT later |
| GND | GND | |

The pin labels sit under the white dome. Pull the dome off to check them, because the pin order varies between clones.

**HC-SR501 settings (keep them fixed and write them in the paper):**

| Control | Set to | Why |
|---|---|---|
| Trigger jumper | **H** (repeatable trigger) | Output stays HIGH while motion continues instead of pulsing |
| Time-delay pot (Tx) | **Fully anticlockwise (~3 s)** | Shortest hold time, so windows react fastest |
| Sensitivity pot (Sx) | Middle (~3–4 m) | Pick one value and leave it there |
| Warm-up | 60 s after power-on | Output is unreliable while the sensor settles. The sketches wait for it |

```
            HC-SR501                    ESP32 DevKit v1
          ┌───────────┐               ┌──────────────┐
          │  VCC  ────┼───────────────┼── VIN (5V)   │
          │  OUT  ────┼───────────────┼── GPIO34     │
          │  GND  ────┼───────────────┼── GND        │
          └───────────┘               │   GPIO2 = LED│
                                      └──────────────┘
```

## Setup B: PIR module with its own EEPROM (the hot-swap module)

### 24LC512 pinout (DIP-8, notch on the left, pin 1 top-left)

| Pin | Name | Connect to | Notes |
|---|---|---|---|
| 1 | A0 | GND | A0–A2 all at GND → I²C address **0x50** |
| 2 | A1 | GND | |
| 3 | A2 | GND | |
| 4 | VSS | GND | |
| 5 | SDA | connector SDA → ESP32 **GPIO 21** | |
| 6 | SCL | connector SCL → ESP32 **GPIO 22** | |
| 7 | WP | **GND while programming**. Optionally switch it to VCC once the module is finalised | WP high = read-only. This is the most common reason writes fail |
| 8 | VCC | **3.3 V** | Add a **100 nF** capacitor between pins 8 and 4, close to the chip |

**I²C pull-ups:** fit 4.7 kΩ resistors from SDA and SCL to **3.3 V**, never 5 V, **once, on the base board**. Don't fit them on every module. The GY-521 MPU6050 board already has its own pull-ups. When it's plugged in, the combined value is about 2.35 kΩ, which is fine at 100–400 kHz.

There's no address conflict: EEPROM = 0x50, MPU6050 = 0x68 (with AD0 to GND).

### Module connector signals

```
 Base board (ESP32)                        PIR module
 ──────────────────                        ──────────────────────────
 3V3 ───────────────────── 3V3 ──────────── 24LC512 VCC (pin 8)
 5V ─[TPS22919]─[INA181 shunt]─ VMOD ───── HC-SR501 VCC
 GND ───────────────────── GND ──────────── 24LC512 VSS, A0-A2, WP; PIR GND
 GPIO21 ─┬──────────────── SDA ──────────── 24LC512 SDA (pin 5)
 GPIO22 ─┼──────────────── SCL ──────────── 24LC512 SCL (pin 6)
     4.7k pull-ups to 3V3
 GPIO34 ────────────────── AOUT ─────────── HC-SR501 OUT
 GPIO35 ──┬─────────────── ID ───────────── R_ID ── GND
       R_pullup to 3V3
```

### Suggested ESP32 base-board pin map

Change these to match your schematic if it already fixes the pins.

| Function | ESP32 pin | Notes |
|---|---|---|
| I²C SDA / SCL | GPIO 21 / 22 | Same as the MPU6050 setup |
| AOUT (analog or digital sensor output) | GPIO 34 | ADC1, so it still works with Wi-Fi on. ADC2 pins don't |
| R_ID insertion/identity sense | GPIO 35 | ADC1. Reads ~4095 with no module inserted |
| INA181 current output | GPIO 36 (VP) | ADC1, input-only |
| TPS22919 ON (load-switch enable) | GPIO 25 | Keep LOW at boot, then enable once R_ID is valid |

### PIR supply voltage

The paper flagged this as an open point.

The HC-SR501 needs **≥ 4.5 V** on VCC, so the module's switched rail must be 5 V for this sensor (the TPS22919 is rated up to 5.5 V, so that's fine). The EEPROM still runs from 3.3 V so the I²C lines stay at 3.3 V.

If your connector only carries one switched 3.3 V rail, there are two options:

- Run the HC-SR501 at 3.3 V by bypassing its onboard regulator. This is a common mod, but less robust.
- Use a 3.3 V-native PIR such as the AM312.

Whichever you choose, state the actual supply voltage in the paper.

## Bring-up order

1. Wire Setup A → flash `data_collection/` → run `collect_data.py`.
2. Wire the EEPROM → flash `eeprom_test/` → expect `device at 0x50` and `PASS`.
3. Train → `export_engine.py` → flash `inference/` → check live predictions and inference time.
4. Pack `pir_weights.bin` + `pir_model_meta.json` into the descriptor with `descriptor_packer.py` and write it to the module's EEPROM.
