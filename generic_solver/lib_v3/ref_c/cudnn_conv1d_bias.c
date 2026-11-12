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

/* Public cuDNN conv_cpu_ref: one spatial axis, NCL/OIL storage,
 * cross-correlation, unit stride/dilation, alpha=1, beta=1. The old output
 * is a broadcast channel bias. The library adapter invokes convolution
 * followed by cudnnAddTensor. Moving bias from the source reduction seed
 * to this epilogue requires the explicitly relaxed numerical model. */
void NETLIB_SPEC(int B, int IC, int OC, int K, int OL,
                float input[B][IC][OL+K-1], float filter[OC][IC][K],
                float bias[OC], float output[B][OC][OL]) {
  for (int b=0; b<B; ++b)
    for (int o=0; o<OC; ++o)
      for (int t=0; t<OL; ++t) {
        float tmp=0.0f;
        for (int i=0; i<IC; ++i)
          for (int k=0; k<K; ++k)
            tmp=filter[o][i][k]*input[b][i][t+k]+tmp;
        output[b][o][t]=tmp+bias[o];
      }
}
