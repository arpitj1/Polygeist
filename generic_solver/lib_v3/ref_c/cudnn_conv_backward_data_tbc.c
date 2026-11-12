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

/* Specialize public cuDNN dataGrad to one spatial axis, cross-correlation, alpha=1, beta=0, unit stride/dilation and no padding. Use TBC/KIO storage and invert the gather relation into a scatter traversal following gradient TBC order. These manual dimension/layout/traversal adaptations change floating-point reduction order; the runtime packs the same values for the cuDNN API. */
void NETLIB_SPEC(int T,int B,int I,int O,int K,
 float grad[T][B][O],float filter[K][I][O],float output[T+K-1][B][I]) {
 for(int t=0;t<T+K-1;++t)for(int b=0;b<B;++b)for(int i=0;i<I;++i)output[t][b][i]=0.0f;
 for(int t=0;t<T;++t)for(int b=0;b<B;++b)for(int o=0;o<O;++o)
  for(int k=0;k<K;++k)for(int i=0;i<I;++i)
   output[t+k][b][i]+=filter[k][i][o]*grad[t][b][o];
}
