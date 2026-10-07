#pragma once
#include <math.h>
#include <stdint.h>

// Auto-generated PIR presence engine: Flatten -> Dense(50->2) -> Softmax
// Weights stored as INT8 (symmetric per-tensor), bias as float32.
// Input: 50 raw PIR samples (0/1) @ 10 Hz. No normalization.

#define PIR_N_IN  50
#define PIR_N_OUT 2
const float DENSE_W_SCALE = 0.0193618815f;
const int8_t DENSE_W_Q[100] = {-52, 44, -64, 56, -81, 60, -24, 52, -10, 35, -25, 4, -36, 20, -13, 32, -2, 11, -26, 15, -12, 15, -26, 17, -10, 30, -33, 6, -30, 28, -28, 38, -31, 39, -10, 22, -13, 27, -9, 34, -7, 28, -9, 8, -22, 37, -35, 36, -32, 4, -18, 41, -13, 18, -22, 10, -30, 12, -10, 35, -19, 35, -8, 37, -22, 39, -26, 32, -22, 28, -23, 33, -24, 16, -40, 35, -17, 22, -18, 38, -24, 4, -26, 27, -25, 7, -17, 17, -11, 31, -14, 30, -35, 25, -7, 34, -58, 78, -127, 121};
const float DENSE_B[2] = {1.690141f, -1.690141f};

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
