// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
/* C instantiation of NVIDIA AMGX's pinned CUSP host reference expandptr<int>.
 * The two capacity arguments expose array extents to the typed API binding;
 * they do not change the reference's iteration domain or stores.
 * cuSPARSE is closed source: this is an explicitly identified public reference
 * counterpart for cusparseXcsr2coo, not an extraction of cuSPARSE internals. */
void NETLIB_SPEC(int rows, int csr_count, const int row_offsets[csr_count],
                 int coo_capacity, int row_indices[coo_capacity]) {
  for (int i = 0; i < rows; i++) {
    for (int jj = row_offsets[i]; jj < row_offsets[i + 1]; jj++) {
      row_indices[jj] = i;
    }
  }
}
