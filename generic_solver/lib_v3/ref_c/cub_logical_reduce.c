/* CUB seeded ThreadReduceSequential, with cuda::std::logical_and<int> or
 * logical_or<int>. Inline the seeded prefix and specialize its operator.
 * The fixed C ABI represents the logical result as int, preserving 0/1. */
#ifndef RETURN_VALUE
#define RETURN_VALUE 1
#endif
#if RETURN_VALUE
int NETLIB_SPEC(int n, int seed, int input[]) {
#else
void NETLIB_SPEC(int n, int seed, int input[], int output[1]) {
#endif
  int value = seed;
  for (int i = 0; i < n; ++i) {
#if LOGICAL_AND
    value = value && input[i];
#else
    value = value || input[i];
#endif
  }
#if RETURN_VALUE
  return value;
#else
  output[0] = value;
#endif
}
