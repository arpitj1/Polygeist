/*****************************************************************************
Copyright (c) 2020, The OpenBLAS Project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
   3. Neither the name of the OpenBLAS project nor the names of 
      its contributors may be used to endorse or promote products 
      derived from this software without specific prior written 
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**********************************************************************************/

/* One group of equal-sized OpenBLAS GEMM_BATCH operations: FP32,
 * alpha=1, beta=0, column-major NN, dense leading dimensions.
 * The batch driver assigns each group's pointer arrays independently.
 * Replace these pointer arrays with contiguous batches and inline the
 * existing portable gemm_small_matrix_kernel_nn.c scalar loop unchanged.
 * SHARED_A uses a repeated read-only pointer in that same pointer array.
 * No application names or dimensions participate in this reference.
 */
#ifndef SHARED_A
#define SHARED_A 0
#endif
void NETLIB_SPEC(int batch, int M, int N, int K,
#if SHARED_A
                float A[K][M],
#else
                float A[batch][K][M],
#endif
                float B[batch][N][K], float C[batch][N][M]) {
  for (int b = 0; b < batch; ++b)
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) {
        float result = 0.0f;
        for (int k = 0; k < K; ++k)
#if SHARED_A
          result += A[k][i] * B[b][j][k];
#else
          result += A[b][k][i] * B[b][j][k];
#endif
        C[b][j][i] = result;
      }
}
