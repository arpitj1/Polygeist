//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Xu Pang, pangxu010@163.com
//    Wenju He, wenju@multicorewareinc.com
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
// This software is provided by the copyright holders and contributors as is and
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
//

/* Scalar serialization of OpenCV calculate_histogram, kercn=1, one
 * work-group/work-item, contiguous one-channel image. Expose the existing row/column coordinates
 * of src_index in a row-major traversal. Atomic increment becomes
 * the same integer increment in serial order. The single partial histogram is
 * copied by the public merge stage. No library recognition rule is introduced.
 */
#include <stdint.h>
void NETLIB_SPEC(int h,int w,const uint8_t input[h][w],OUTPUT_TYPE output[256]) {
  int localhist[256];
  for(int i=0;i<256;++i) localhist[i]=0;
  for(int y=0;y<h;++y)
    for(int x=0;x<w;++x) localhist[(int)input[y][x]]++;
  for(int i=0;i<256;++i) output[i]=localhist[i];
}
