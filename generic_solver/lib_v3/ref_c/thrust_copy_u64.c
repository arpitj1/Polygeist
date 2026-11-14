/* Public Thrust copy's identity transform, specialized to uint64_t. */
#include <stdint.h>
void NETLIB_SPEC(int n, uint64_t input[n], uint64_t output[n]) {
  for (int i = 0; i < n; ++i) output[i] = input[i];
}
