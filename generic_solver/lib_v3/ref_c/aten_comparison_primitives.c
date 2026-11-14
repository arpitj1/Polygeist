/* Public ATen comparison/where composition with explicit scalar broadcasts.
 * No min/max identities: comparison direction and selected values are kept.
 */
#if BOUND_ARRAY
#define BOUND_ARG float bound[n]
#define BOUND bound[i]
#else
#define BOUND_ARG float bound
#define BOUND bound
#endif
void NETLIB_SPEC(int n, float input[n], float output[n]
#if !SELF_BOUND
                , BOUND_ARG
#endif
#if HAS_VALUE
                , float value[n]
#endif
#if HAS_OTHER
                , float other[n]
#endif
#if HAS_TRUE
                , float true_value
#endif
#if HAS_FALSE
                , float false_value
#endif
) {
  for (int i = 0; i < n; ++i) {
#if SELF_BOUND
    int predicate = input[i] COMPARE input[i];
#else
    int predicate = input[i] COMPARE BOUND;
#endif
#if AS_FLOAT
    output[i] = (float)predicate;
#else
    output[i] = predicate ? TRUE_VALUE : FALSE_VALUE;
#endif
  }
}
