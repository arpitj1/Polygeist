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
/* Public OpenCV RowSum<uchar,ushort> and ColumnSum<ushort,uchar>,
 * scalar single-channel 3x3 interior. Recompute the three-row accumulator
 * instead of retaining ColumnSum's rolling scratch state. The public Q23
 * normalization for scale=1/9 gives divDelta=4, divScale=932068.
 * Row sums fit uint16_t; all vertical products fit signed int32_t.
 * Keep the caller's output border unchanged.
 */
#include <stdint.h>
void NETLIB_SPEC(int h,int w,const uint8_t input[h][w],uint8_t output[h][w]) {
  uint16_t horizontal[h][w];
  for(int y=0;y<h;++y) for(int x=1;x<w-1;++x)
    horizontal[y][x]=(uint16_t)(input[y][x-1]+input[y][x]+input[y][x+1]);
  for(int y=1;y<h-1;++y) for(int x=1;x<w-1;++x) {
    int s0=horizontal[y-1][x]+horizontal[y][x]+horizontal[y+1][x];
    output[y][x]=(uint8_t)(((s0+4)*932068)>>23);
  }
}
