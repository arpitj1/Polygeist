/* Mechanical ABI adapter for imported column-major FP64 contraction references.
 * Array pointers are device pointers, supplied by the typed-wrapper mapper.
 * Logical modes are i=0, j=1, k=2: C[i,j] = sum_k A[i,k] * B[k,j].
 * Transpose and accumulation flags are validated reference specializations.
 */
void polygeist_cutensornet_matrix_f64(
    int M, int N, int K, int lda, const double *A, int ldb, const double *B,
    int ldc, double *C, int transpose_a, int transpose_b, int accumulate) {
  if (M <= 0 || N <= 0 || K <= 0 ||
      (transpose_a != 0 && transpose_a != 1) ||
      (transpose_b != 0 && transpose_b != 1) ||
      (accumulate != 0 && accumulate != 1) ||
      lda < (transpose_a ? K : M) || ldb < (transpose_b ? N : K) ||
      ldc < M || !A || !B || !C) {
    fprintf(stderr, "polygeist runtime: invalid cuTensorNet matrix layout\n");
    abort();
  }
#if POLYGEIST_HAS_CUTENSORNET
  const int64_t metadata[] = {
      1, 2, accumulate, 2, 2, 2,
      transpose_a ? M : K, lda, transpose_a ? 0 : 2,
      transpose_a ? K : M, 1, transpose_a ? 2 : 0,
      transpose_b ? K : N, ldb, transpose_b ? 2 : 1,
      transpose_b ? N : K, 1, transpose_b ? 1 : 2,
      N, ldc, 1, M, 1, 0};
  const int64_t pointers[] = {(int64_t)(uintptr_t)A, (int64_t)(uintptr_t)B,
                              (int64_t)(uintptr_t)C};
  polygeist_cutensornet_network_f64_device(pointers, metadata);
#else
  fprintf(stderr, "polygeist runtime: cuTensorNet matrix binding requires "
                  "POLYGEIST_ENABLE_CUTENSORNET and the cuTensorNet library\n");
  abort();
#endif
}

/* A contiguous matrix-vector ABI is the N=1 form of the same contraction. */
void polygeist_cutensornet_matvec_f64(
    int m, int n, int lda, const double *A, const double *x, double *y,
    int transpose_a, int accumulate) {
  int output = transpose_a ? n : m;
  int reduction = transpose_a ? m : n;
  polygeist_cutensornet_matrix_f64(output, 1, reduction, lda, A, reduction, x,
                                  output, y, transpose_a, 0, accumulate);
}
