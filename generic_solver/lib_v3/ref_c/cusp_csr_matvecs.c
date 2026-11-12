// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
/* C instantiation of the public NVIDIA AMGX/CUSP host reference.
 * cuSPARSE is closed source; this is a public reference counterpart.
 * Instantiate I=int, T=float, expose capacities as VLA extents, and apply
 * the reference to a zero-filled output (alpha=1, beta=0).
 * Sources and all adaptations are recorded in the adjacent manifest. */
void NETLIB_SPEC(int rows, int row_offset_count,
    const int row_offsets[row_offset_count], int column_index_count,
    const int column_indices[column_index_count], int value_count,
    const float values[value_count], int b_rows, int b_cols,
    const float b[b_rows][b_cols], int c_rows, int c_cols,
    float c[c_rows][c_cols]) {
  for (int i = 0; i < rows; ++i)
    for (int k = 0; k < b_cols; ++k)
      c[i][k] = 0.0f;
  for (int i = 0; i < rows; ++i) {
    for (int jj = row_offsets[i]; jj < row_offsets[i + 1]; ++jj) {
      const int j = column_indices[jj];
      const float a = values[jj];
      /* Inline the public dense.h axpy; preserve row/nonzero/vector order. */
      for (int k = 0; k < b_cols; ++k)
        c[i][k] += a * b[j][k];
    }
  }
}
