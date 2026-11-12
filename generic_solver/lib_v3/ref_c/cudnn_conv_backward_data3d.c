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

/* Derived from public cuDNN dataGrad_cpu_ref<float> in conv_sample.cpp.
 * Select cross-correlation, unit stride/dilation, no padding, alpha=1,
 * beta=0, and one batch. Extend the two spatial axes to three.
 * Invert the gather relation p = h-r, q = w-s into h = p+r, w = q+s;
 * iterate the valid gradient/filter Cartesian product and scatter-add.
 * This is an explicit manual reference adaptation. It changes the floating
 * reduction order and requires the relaxed numeric model; it is not a
 * byte-for-byte extraction of the sample. No input-dependent bounds checks
 * remain because the output extent is gradient extent + filter extent - 1.
 */
void NETLIB_SPEC(int IC, int OC, int D, int H, int W,
                 int KD, int KH, int KW,
                 float input[IC][D][H][W],
                 float filter[IC][OC][KD][KH][KW],
                 float output[OC][D+KD-1][H+KH-1][W+KW-1]) {
  for (int co=0; co<OC; ++co)
    for (int z=0; z<D+KD-1; ++z)
      for (int y=0; y<H+KH-1; ++y)
        for (int x=0; x<W+KW-1; ++x)
          output[co][z][y][x] = 0.0f;
  for (int ci=0; ci<IC; ++ci)
    for (int co=0; co<OC; ++co)
      for (int z=0; z<D; ++z)
        for (int y=0; y<H; ++y)
          for (int x=0; x<W; ++x)
            for (int kz=0; kz<KD; ++kz)
              for (int ky=0; ky<KH; ++ky)
                for (int kx=0; kx<KW; ++kx)
                  output[co][z+kz][y+ky][x+kx] +=
                      filter[ci][co][kz][ky][kx] * input[ci][z][y][x];
}
