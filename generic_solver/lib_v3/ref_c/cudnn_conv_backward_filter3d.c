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

/* Derived from the public cuDNN weightGrad_cpu_ref<float>.
 * Select its cross-correlation branch, alpha=1, beta=0, no padding,
 * unit stride/dilation, and one batch. Extend the two spatial loops to
 * three axes, as supported by the ND cuDNN API. The scalar accumulation
 * order remains depth/height/width for every filter element.
 * Require ID=OD+KD-1, IH=OH+KH-1, IW=OW+KW-1; no bounds mask is then needed.
 * This dimensional extension and mode selection are manual adaptations,
 * rather than an unchanged extraction of the two-dimensional sample.
 */
void NETLIB_SPEC(int IC, int OC, int ID, int IH, int IW,
                 int OD, int OH, int OW, int KD, int KH, int KW,
                 float input[IC][ID][IH][IW],
                 float grad_output[OC][OD][OH][OW],
                 float grad_filter[OC][IC][KD][KH][KW]) {
  for (int ci = 0; ci < IC; ++ci)
    for (int kz = 0; kz < KD; ++kz)
      for (int ky = 0; ky < KH; ++ky)
        for (int kx = 0; kx < KW; ++kx)
          for (int ko = 0; ko < OC; ++ko) {
            float value = 0.0f;
            for (int z = 0; z < OD; ++z)
              for (int y = 0; y < OH; ++y)
                for (int x = 0; x < OW; ++x)
                  value = grad_output[ko][z][y][x] *
                          input[ci][z+kz][y+ky][x+kx] + value;
            grad_filter[ko][ci][kz][ky][kx] = value;
          }
}
