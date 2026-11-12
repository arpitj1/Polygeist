// A shared fill must remain visible until the complete scan is considered.
// Large dimensions prevent constant folding from accidentally exposing it.
#ifndef SCAN_PRODUCT
#define SCAN_PRODUCT 1
#endif
void large_scan_seed(float input[1025][2049], float output[1025][2049]) {
  for (int r = 0; r < 1025; ++r) {
    float seed = 1.0f;
    for (int k = 0; k < 2049; ++k) {
#if SCAN_PRODUCT
      seed *= input[r][k];
#else
      seed += input[r][k];
#endif
      output[r][k] = seed;
    }
  }
}
