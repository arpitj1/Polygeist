// RUN: polygeist-opt %s --lower-supported-math-to-libm | FileCheck %s

// Accurate calls coexist with operations left for LLVM intrinsic lowering.
// CHECK-LABEL: func.func @mixed
// CHECK: call @expm1f
// CHECK: call @log1pf
// CHECK: math.exp
func.func @mixed(%x: f32) -> f32 {
  %a = math.expm1 %x : f32
  %b = math.log1p %a : f32
  %c = math.exp %b : f32
  return %c : f32
}

// CHECK-LABEL: func.func @double
// CHECK: call @expm1
// CHECK: call @log1p
func.func @double(%x: f64) -> f64 {
  %a = math.expm1 %x : f64
  %b = math.log1p %a : f64
  return %b : f64
}
