// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3
/* Public CUB BlockHistogram::InitHistogram and BlockHistogramAtomic::Composite.
 * Serialize the integer atomic increments, expose template extents as runtime
 * arguments, and specialize samples to direct integer bin indices. For valid
 * indices and count <= INT_MAX, every interleaving gives these exact counts.
 * DeviceHistogram::HistogramEven uses unit-width bins [0, bins). */
void NETLIB_SPEC(int count, const int samples[count], int bins, int output[bins]) {
  for (int bin = 0; bin < bins; ++bin) output[bin] = 0;
  for (int i = 0; i < count; ++i) output[samples[i]] += 1;
}
