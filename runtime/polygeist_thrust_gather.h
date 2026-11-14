#ifndef POLYGEIST_THRUST_GATHER_H
#define POLYGEIST_THRUST_GATHER_H
#include <cstdint>
#if defined(__CUDACC__)
#define POLYGEIST_GATHER_HD __host__ __device__
#else
#define POLYGEIST_GATHER_HD
#endif
// Index-iterator adapters for public Thrust gather. No output computation is
// implemented here: gather performs the indirect read and output assignment.
template <int Mode> struct PolygeistGatherIndex {
  const int32_t *indices;
  int64_t columns, sourceColumns;
  const int32_t *secondary = nullptr;
  POLYGEIST_GATHER_HD int64_t operator()(int64_t i) const {
    if constexpr (Mode == 0) return indices[i];
    if constexpr (Mode == 1) return int64_t(indices[i / columns]) * columns + i % columns;
    if constexpr (Mode == 2) return (i / columns) * sourceColumns + indices[i % columns];
    if constexpr (Mode == 3) return (i / columns) * sourceColumns + indices[i];
    if constexpr (Mode == 4) return i * columns + indices[i];
    if constexpr (Mode == 5) return int64_t(indices[i]) * columns + secondary[i];
  }
};
#undef POLYGEIST_GATHER_HD
#endif
