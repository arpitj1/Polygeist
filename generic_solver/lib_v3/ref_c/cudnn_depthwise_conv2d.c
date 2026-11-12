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

/* Public cuDNN conv_cpu_ref<float,float>, applied independently to each
 * one-channel group. Specialize cross-correlation, NCHW, unit stride and
 * dilation, alpha=1, beta=1 with broadcast channel bias as old output.
 * RH/RW are nonnegative filter radii: odd filters are (2*RH+1)x(2*RW+1),
 * and padding equals those radii. This guarantees output HxW for every
 * supported filter, rather than relying on an unchecked odd-size promise.
 * Expand lin2dim/dim2lin into spatial loops. Group decomposition removes
 * the singleton input-channel reduction. The cuDNN grouped convolution
 * adapter performs the same groups in one API call, then adds the bias.
 * Bias epilogue and parallel vendor accumulation require relaxed arithmetic.
 */
void NETLIB_SPEC(int B, int C, int H, int W, int RH, int RW,
                float input[B][C][H][W],
                float filter[C][2*RH+1][2*RW+1], float bias[C],
                float output[B][C][H][W]) {
  for (int n=0; n<B; ++n)
    for (int c=0; c<C; ++c)
      for (int y=0; y<H; ++y)
        for (int x=0; x<W; ++x) {
          float tmp=0.0f;
          for (int ky=0; ky<2*RH+1; ++ky)
            for (int kx=0; kx<2*RW+1; ++kx) {
              int iy=y+ky-RH, ix=x+kx-RW;
              if (iy>=0 && iy<H && ix>=0 && ix<W)
                tmp=filter[c][ky][kx]*input[n][c][iy][ix]+tmp;
            }
          output[n][c][y][x]=tmp+bias[c];
        }
}
