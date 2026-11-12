// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/* cuDNN conv_sample.cpp conv_cpu_ref<float,float>, one spatial axis,
 * unit stride/dilation, no padding, alpha=1, beta=0, cross-correlation.
 * Input/output strides describe TBC storage. The filter KIO storage is
 * packed by the runtime into cuDNN OIK storage without changing values.
 * Keep the sample's channel-then-spatial reduction order. The application
 * may need the explicitly relaxed arithmetic model to permute that order.
 * The input length is TO+K-1, checked against the actual tensor extent by
 * the typed ABI lowering. This avoids a semantically unused outer C extent.
 */
void NETLIB_SPEC(int B, int I, int O, int K, int TO,
                float input[TO+K-1][B][I], float filter[K][I][O],
                float output[TO][B][O]) {
  for (int b = 0; b < B; ++b)
    for (int o = 0; o < O; ++o)
      for (int t = 0; t < TO; ++t) {
        float tmp = 0.0f;
        for (int i = 0; i < I; ++i)
          for (int k = 0; k < K; ++k)
            tmp = filter[k][i][o] * input[t+k][b][i] + tmp;
        output[t][b][o] = tmp;
      }
}
