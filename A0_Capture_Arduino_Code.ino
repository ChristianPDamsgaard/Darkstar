/*
 * UNO R4 — 200 Hz CSV stream for live_plot.py
 * Line format: 0,7,v_a0  (refs + A0 only, 0–5 V ADC scale)
 */
#include <Arduino.h>

const unsigned long SAMPLE_HZ = 200;
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_HZ;  // 5000 µs
const int ADC_BITS = 14;
const int ADC_MAX = (1 << ADC_BITS) - 1;
const float ADC_REF_VOLTS = 5.0f;

void setup() {
  Serial.begin(115200);
  analogReadResolution(ADC_BITS);
}

void loop() {
  static unsigned long nextTick = micros();

  unsigned long now = micros();
  if ((long)(now - nextTick) < 0) {
    return;
  }
  nextTick += SAMPLE_INTERVAL_US;

  int r0 = analogRead(A0);
  float v0 = (r0 / (float)ADC_MAX) * ADC_REF_VOLTS;

  Serial.print(0.0f, 4);
  Serial.print(',');
  Serial.print(7.0f, 4);
  Serial.print(',');
  Serial.println(v0, 4);
}
