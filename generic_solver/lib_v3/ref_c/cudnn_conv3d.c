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

/* Source-derived NCDHW FP32 specialization of the public
 * NVIDIA conv_cpu_ref, with the existing cross-correlation branch selected.
 * alpha=1, beta=0 or 1, resizeFactor=1, unit stride/dilation, valid output domain.
 * Lexicographic delinearization retains the original per-output fold order. */
void NETLIB_SPEC(
#ifdef NETLIB_BATCHED
int N,
#endif
int IC,int ID,int IH,int IW,int OC,int KD,int KH,int KW,int OD,int OH,int OW,
#ifdef NETLIB_BATCHED
float input[N][IC][ID][IH][IW],float filter[OC][IC][KD][KH][KW],float output[N][OC][OD][OH][OW]) {
 for(int b=0;b<N;++b) {
#else
float input[IC][ID][IH][IW],float filter[OC][IC][KD][KH][KW],float output[OC][OD][OH][OW]) {
#endif
 for(int ko=0;ko<OC;++ko)for(int oz=0;oz<OD;++oz)for(int oy=0;oy<OH;++oy)for(int ox=0;ox<OW;++ox){
  float tmp=0;
  for(int ci=0;ci<IC;++ci)for(int fz=0;fz<KD;++fz)for(int fy=0;fy<KH;++fy)for(int fx=0;fx<KW;++fx){
   float fval=filter[ko][ci][fz][fy][fx];
   float ival=
#ifdef NETLIB_BATCHED
       input[b][ci][oz+fz][oy+fy][ox+fx];
#else
       input[ci][oz+fz][oy+fy][ox+fx];
#endif
   tmp=fval*ival+tmp;
  }
#ifdef NETLIB_ACCUMULATE_OUTPUT
  // The public reference adds the old output after the zero-seeded fold.
#ifdef NETLIB_BATCHED
  output[b][ko][oz][oy][ox]=tmp+output[b][ko][oz][oy][ox];
#else
  output[ko][oz][oy][ox]=tmp+output[ko][oz][oy][ox];
#endif
#else
#ifdef NETLIB_BATCHED
  output[b][ko][oz][oy][ox]=tmp;
#else
  output[ko][oz][oy][ox]=tmp;
#endif
#endif
 }
#ifdef NETLIB_BATCHED
 }
#endif
}
