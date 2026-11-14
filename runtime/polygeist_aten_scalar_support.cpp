// Host implementations used to execute imported scalar reference calls.
#include <ATen/native/Math.h>
#define POLYGEIST_ATEN_SCALAR1(NAME, IMPL, ENTRY) \
  extern "C" float NAME(float x) { return IMPL<float>(x); }
#define POLYGEIST_ATEN_SCALAR2(NAME, IMPL, ENTRY) \
  extern "C" float NAME(float x, float y) { return IMPL<float>(x, y); }
#define POLYGEIST_ATEN_SCALARINT(NAME, IMPL, ENTRY) \
  extern "C" float NAME(int order, float x) { return IMPL<float>(x, order); }
#include "polygeist_aten_scalar_bindings.inc"

extern "C" float calc_trigammaf(float x) { return trigamma(x); }
