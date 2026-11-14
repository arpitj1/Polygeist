/* Composition of public ATen scalar comparisons, Boolean logic and where.
 * Keep both predicates explicit, including their unordered-NaN behavior.
 */
void NETLIB_SPEC(int n, float input[n], float value[n], float output[n],
                float lower, float upper, float selected) {
  for (int i = 0; i < n; ++i)
    output[i] = (input[i] LOWER_COMPARE lower) LOGICAL (input[i] UPPER_COMPARE upper)
                    ? selected : value[i];
}
