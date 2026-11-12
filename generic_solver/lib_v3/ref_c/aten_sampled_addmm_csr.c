// SPDX-License-Identifier: BSD-3-Clause
/* FP32/int32, unbatched instantiation of public PyTorch SampledAddmmKernel.cpp.
 * The source library is public; cuSPARSE itself is closed source.
 * SparseBlas.cpp copies self into result and transposes mat2 before the CPU
 * kernel. Here the old values are a separate input, and b[k][n] composes that
 * caller transpose with the kernel's mat2_ptr[n*K+k] access. The public
 * vector map2_reduce_all is serialized under the declared relaxed model.
 * Dimensions/capacities remain parameters; no benchmark shape is embedded.
 */
void NETLIB_SPEC(int rows,
    int row_offset_count, const int row_offsets[row_offset_count],
    int column_index_count, const int column_indices[column_index_count],
    int self_value_count, const float self_values[self_value_count],
    int a_rows, int a_cols, const float a[a_rows][a_cols],
    int b_rows, int b_cols, const float b[b_rows][b_cols],
    float alpha, float beta, int out_value_count, float out_values[out_value_count]) {
  for (int m = 0; m < rows; ++m) {
    int row_start = row_offsets[m];
    int row_end = row_offsets[m + 1];
    for (int e = row_start; e < row_end; ++e) {
      int n = column_indices[e];
      float val = self_values[e];
      float dot = 0.0f;
      for (int k = 0; k < a_cols; ++k)
        dot += a[m][k] * b[k][n];
      val = alpha * dot + beta * val;
      out_values[e] = val;
    }
  }
}
