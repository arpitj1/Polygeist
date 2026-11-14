/* Scalar expansion of public Thrust gather with typed index iterators.
 * Modes describe flat, row, column, row-local, and paired-row index views.
 * Dimensions are runtime parameters; all indices must address the input.
 */
#ifndef DATA_TYPE
#define DATA_TYPE float
#endif
#if MODE == 0
void NETLIB_SPEC(int n, int m, int index[n], DATA_TYPE input[m], DATA_TYPE output[n]) {
  for (int i = 0; i < n; ++i) output[i] = input[index[i]];
}
#elif MODE == 1
void NETLIB_SPEC(int rows, int cols, int source_rows, int index[rows],
                DATA_TYPE input[source_rows][cols], DATA_TYPE output[rows][cols]) {
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) output[r][c] = input[index[r]][c];
}
#elif MODE == 2
void NETLIB_SPEC(int rows, int cols, int source_cols, int index[cols],
                DATA_TYPE input[rows][source_cols], DATA_TYPE output[rows][cols]) {
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) output[r][c] = input[r][index[c]];
}
#elif MODE == 3
void NETLIB_SPEC(int rows, int cols, int source_cols, int index[rows][cols],
                DATA_TYPE input[rows][source_cols], DATA_TYPE output[rows][cols]) {
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) output[r][c] = input[r][index[r][c]];
}
#elif MODE == 4
void NETLIB_SPEC(int rows, int cols, int index[rows],
                DATA_TYPE input[rows][cols], DATA_TYPE output[rows]) {
  for (int r = 0; r < rows; ++r) output[r] = input[r][index[r]];
}
#else
void NETLIB_SPEC(int n,int rows,int cols,int row[n],int column[n],
                DATA_TYPE input[rows][cols],DATA_TYPE output[n]) {
  for(int i=0;i<n;++i) output[i]=input[row[i]][column[i]];
}
#endif
