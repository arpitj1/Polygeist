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

/* Derived from NVIDIA cuDNN conv_sample.cpp conv_cpu_ref<float,float>.
 * Specialize NCHW, FP32, resizeFactor=1, alpha=1, beta=0, no padding,
 * unit stride. Retain its cross-correlation branch; the sample hardcodes
 * isConv=true despite carrying both branches, so this mode is explicit.
 * Expand lin2dim/dim2lin and delinearize outId and filId into lexicographic
 * spatial loops. This preserves each output's ordered accumulation.
 * DILATION is the library parameter specialization, independent of a caller.
 * Requires OH=(IH-DILATION*(KH-1)), OW=(IW-DILATION*(KW-1)).
 */
#ifndef DILATION
#define DILATION 1
#endif
#ifndef BATCHED
#define BATCHED 1
#endif
void NETLIB_SPEC(
#if BATCHED
int N,
#endif
int IC,int IH,int IW,int OC,int KH,int KW,int OH,int OW,
#if BATCHED
 float input[N][IC][IH][IW],float filter[OC][IC][KH][KW],float output[N][OC][OH][OW]) {
 for(int ni=0;ni<N;++ni)
#else
 float input[IC][IH][IW],float filter[OC][IC][KH][KW],float output[OC][OH][OW]) {
#endif
  for(int ko=0;ko<OC;++ko)
   for(int oy=0;oy<OH;++oy)
    for(int ox=0;ox<OW;++ox) {
     float tmp=0;
     for(int ci=0;ci<IC;++ci)
      for(int fy=0;fy<KH;++fy)
       for(int fx=0;fx<KW;++fx) {
        float fval=filter[ko][ci][fy][fx];
        #if BATCHED
        float ival=input[ni][ci][oy+DILATION*fy][ox+DILATION*fx];
#else
        float ival=input[ci][oy+DILATION*fy][ox+DILATION*fx];
#endif
        tmp=fval*ival+tmp;
       }
     #if BATCHED
     output[ni][ko][oy][ox]=tmp;
#else
     output[ko][oy][ox]=tmp;
#endif
    }
}
