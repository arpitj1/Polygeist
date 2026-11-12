/* Fill complete leading rows through a flat view, then read all rows.
 * The untouched suffix must retain its original value. */
void NETLIB_SPEC(float seed, float input[5][2][3], float output[5][2][3]) {
  for (int i = 0; i < 18; ++i)
    ((float *)output)[i] = seed;
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 3; ++k)
        output[i][j][k] += input[i][j][k];
}
