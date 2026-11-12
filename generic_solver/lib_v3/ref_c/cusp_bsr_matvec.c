// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
/* C instantiation of public NVIDIA AMGX/CUSP bsr_matvec<int,float> and
 * its dense gemv helper. cuSPARSE is closed source; this is a public
 * reference counterpart. Select square blocks for cuSPARSE's BSR SpMM,
 * expose row-major blocks and vectors as VLA dimensions, and
 * apply the accumulating reference to a zero-filled output (beta=0).
 * The general source path also handles the 1x1 CSR case; no fixed block
 * size is selected. Sources and adaptations are recorded in the manifest.
 */
void NETLIB_SPEC(int block_rows, int block_dim,
    int row_offset_count, const int row_offsets[row_offset_count],
    int column_index_count, const int column_indices[column_index_count],
    int value_block_count,
    const float values[value_block_count][block_dim][block_dim],
    int x_blocks, const float x[x_blocks][block_dim],
    int y_blocks, float y[y_blocks][block_dim]) {
  for (int i = 0; i < block_rows; ++i)
    for (int r = 0; r < block_dim; ++r)
      y[i][r] = 0.0f;
  for (int i = 0; i < block_rows; ++i) {
    for (int jj = row_offsets[i]; jj < row_offsets[i + 1]; ++jj) {
      const int j = column_indices[jj];
      for (int r = 0; r < block_dim; ++r) {
        float dot = y[i][r];
        for (int c = 0; c < block_dim; ++c)
          dot += values[jj][r][c] * x[j][c];
        y[i][r] = dot;
      }
    }
  }
}
