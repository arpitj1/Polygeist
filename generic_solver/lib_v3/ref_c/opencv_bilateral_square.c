/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Copyright (C) 2025, Advanced Micro Devices, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
/* OpenCV bilateral scalar sum/weight algorithm (bilateral.cl and dispatch).
 * Explicit reference variant: full 3x3 square offsets, evaluate the combined
 * spatial/range exponential directly, round to nearest integer. The target
 * PVA API uses circular support and quantized weights; this binding requires
 * explicit acceptance of backend approximation, never exact vendor equality.
 */
#include <stdint.h>
#include <math.h>
void NETLIB_SPEC(int h,int w,const uint8_t input[],uint8_t output[],float sigma_r,float sigma_s) {
  float ir=-0.5f/(sigma_r*sigma_r),is=-0.5f/(sigma_s*sigma_s);
  for(int y=1;y<h-1;++y) for(int x=1;x<w-1;++x) {
    float center=input[y*w+x],weights=0,values=0;
    for(int ky=-1;ky<=1;++ky) for(int kx=-1;kx<=1;++kx) {
      float sample=input[(y+ky)*w+x+kx],delta=sample-center;
      float weight=expf((float)(kx*kx+ky*ky)*is+delta*delta*ir);
      weights+=weight;values+=weight*sample;
    }
    int value=(int)(values/weights+0.5f);
    output[y*w+x]=(uint8_t)(value<0?0:(value>255?255:value));
  }
}
