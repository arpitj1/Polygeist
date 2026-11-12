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

/* Specialize public cuDNN dataGrad to cross-correlation, alpha=1, beta=0, unit stride/dilation and no padding. Invert its gather relation p=h-r, q=w-s into scatter destinations h=p+r, w=q+s. Traverse the valid gradient/filter product in gradient storage order. This manual adaptation changes floating-point reduction order. */
void NETLIB_SPEC(int B,int IC,int OC,int H,int W,int KH,int KW,
 float input[B][IC][H][W],float filter[IC][OC][KH][KW],
 float output[B][OC][H+KH-1][W+KW-1]) {
 for(int b=0;b<B;++b)for(int o=0;o<OC;++o)
  for(int y=0;y<H+KH-1;++y)for(int x=0;x<W+KW-1;++x)output[b][o][y][x]=0.0f;
 for(int b=0;b<B;++b)for(int c=0;c<IC;++c)
  for(int y=0;y<H;++y)for(int x=0;x<W;++x)
   for(int o=0;o<OC;++o)for(int ky=0;ky<KH;++ky)for(int kx=0;kx<KW;++kx)
    output[b][o][y+ky][x+kx]+=filter[c][o][ky][kx]*input[b][c][y][x];
}
