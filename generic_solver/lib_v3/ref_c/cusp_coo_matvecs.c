// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
/* CUSP coo_matvec<int,float> repeated independently for each dense column.
 * Specialize alpha=1,beta=0 by zeroing the output. The column loop is
 * interchanged with the nonzero loop; each output's accumulation order is
 * unchanged. The public reference counterpart is not cuSPARSE's internals. */
void NETLIB_SPEC(int rows, int nnz, int row_index_count,
    const int row_indices[row_index_count], int column_index_count,
    const int column_indices[column_index_count], int value_count,
    const float values[value_count], int b_rows, int b_cols,
    const float b[b_rows][b_cols], int c_rows, int c_cols,
    float c[c_rows][c_cols]) {
  for (int i = 0; i < rows; ++i)
    for (int k = 0; k < b_cols; ++k)
      c[i][k] = 0.0f;
  for (int n = 0; n < nnz; ++n)
    for (int k = 0; k < b_cols; ++k)
      c[row_indices[n]][k] += values[n] * b[column_indices[n]][k];
}
