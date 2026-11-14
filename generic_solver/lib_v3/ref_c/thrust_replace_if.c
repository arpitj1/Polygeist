/* Public Thrust replace_if with an integer stencil and identity predicate.
 * Unselected output elements retain their original values. */
void NETLIB_SPEC(int n, float output[n], int mask[n], float value) {
  for (int i = 0; i < n; ++i) if (mask[i]) output[i] = value;
}
