/* Public ATen where scalar function with an int32 condition specialization,
 * applied by Thrust for_each over zipped contiguous arrays. */
void NETLIB_SPEC(int n, int condition[n], float left[n], float right[n], float output[n]) {
  for (int i = 0; i < n; ++i)
    output[i] = condition[i] ? left[i] : right[i];
}
