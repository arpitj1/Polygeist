// RUN: polygeist-opt %s --normalize-linalg-matching-views="split-pointwise-inputs=true" | FileCheck %s
// RUN: polygeist-opt %s --normalize-linalg-matching-views | FileCheck %s --check-prefix=DEFAULT

// Scalar captures remain typed, and multiplication precedes addition.
// CHECK-LABEL: func.func @scaled_shift
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: linalg.yield
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: linalg.yield
// DEFAULT-LABEL: func.func @scaled_shift
// DEFAULT-COUNT-1: linalg.generic
// DEFAULT: arith.mulf
// DEFAULT-NEXT: {{.*}}arith.addf
func.func @scaled_shift(%x: tensor<?xf32>, %y: tensor<?xf32>, %scale: f32, %bias: f32) -> tensor<?xf32> {
  %r = linalg.generic {indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>], iterator_types=["parallel"]} ins(%x:tensor<?xf32>) outs(%y:tensor<?xf32>) {
  ^bb0(%v:f32,%old:f32):
    %a = arith.mulf %scale,%v : f32
    %b = arith.addf %bias,%a : f32
    linalg.yield %b : f32
  } -> tensor<?xf32>
  return %r : tensor<?xf32>
}

// The optional pointwise transformation does not rewrite a reduction.
// CHECK-LABEL: func.func @seeded_reduction
// CHECK-COUNT-1: linalg.generic
// CHECK: arith.mulf
// CHECK-NEXT: {{.*}}arith.addf
func.func @seeded_reduction(%x:tensor<17xf32>,%seed:tensor<f32>,%factor:f32) -> tensor<f32> {
  %r = linalg.generic {indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->()>], iterator_types=["reduction"]} ins(%x:tensor<17xf32>) outs(%seed:tensor<f32>) {
  ^bb0(%v:f32,%old:f32):
    %a = arith.mulf %v,%factor : f32
    %b = arith.addf %old,%a : f32
    linalg.yield %b : f32
  } -> tensor<f32>
  return %r : tensor<f32>
}

// Effects and computations depending on output state cannot be hoisted.
// CHECK-LABEL: func.func @effectful
// CHECK-COUNT-1: linalg.generic
// CHECK: func.call @observe
// CHECK: arith.addf
func.func private @observe(f32) -> f32
func.func @effectful(%x:tensor<9xf32>,%y:tensor<9xf32>,%bias:f32) -> tensor<9xf32> {
  %r = linalg.generic {indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>], iterator_types=["parallel"]} ins(%x:tensor<9xf32>) outs(%y:tensor<9xf32>) {
  ^bb0(%v:f32,%old:f32):
    %a = func.call @observe(%v) : (f32)->f32
    %b = arith.addf %a,%bias : f32
    linalg.yield %b : f32
  } -> tensor<9xf32>
  return %r : tensor<9xf32>
}
