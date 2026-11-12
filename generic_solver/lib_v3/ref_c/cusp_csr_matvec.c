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
    const float values[value_count], int x_count, const float x[x_count],
    int y_count, float y[y_count]) {
  for (int i = 0; i < rows; ++i) {
    float sum = 0.0f;
    for (int jj = row_offsets[i]; jj < row_offsets[i + 1]; ++jj)
      sum += values[jj] * x[column_indices[jj]];
    y[i] = sum;
  }
}
