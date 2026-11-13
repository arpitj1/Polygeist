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
/* Public OpenCV hlineSmooth3N121<uint8_t,ufixedpoint16> and
 * vlineSmooth3N121<uint8_t,ufixedpoint16>, scalar single-channel interior.
 * Fixed-point constructors and shifts are expanded to their public raw integer
 * operations. Each horizontal sum is at most 65280, so the saturating addition
 * in ufixedpoint16 never takes its overflow arm for these input types.
 * The vertical implementation is copied as raw unsigned integer arithmetic.
 * The driver retains the untouched output border.
 */
#include <stdint.h>
void NETLIB_SPEC(int h,int w,const uint8_t input[h][w],uint8_t output[h][w]) {
  uint16_t horizontal[h][w];
  for(int y=0;y<h;++y) for(int x=1;x<w-1;++x) {
    uint16_t left=(uint16_t)((uint16_t)input[y][x-1]<<8)>>H_SIDE_SHIFT;
    uint16_t right=(uint16_t)((uint16_t)input[y][x+1]<<8)>>H_SIDE_SHIFT;
    uint16_t middle=(uint16_t)((uint16_t)input[y][x]<<8)>>H_CENTER_SHIFT;
    horizontal[y][x]=(uint16_t)((uint16_t)(left+right)+middle);
  }
  for(int y=1;y<h-1;++y) for(int x=1;x<w-1;++x)
    output[y][x]=(uint8_t)(((uint32_t)horizontal[y-1][x]+(uint32_t)horizontal[y+1][x]+
                          ((uint32_t)horizontal[y][x]<<V_CENTER_SHIFT)+V_ROUND)>>V_OUTPUT_SHIFT);
}
