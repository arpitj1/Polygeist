// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-operand-rank=1 fuse-initializers=true" | FileCheck %s

// Consecutive reduction axes traverse their original row-major order. The
// scalar seed is unchanged when these input dimensions collapse.
// CHECK-LABEL: func.func @ordered_sum
// CHECK: tensor.collapse_shape
// CHECK-SAME: tensor<3x5xf32> into tensor<15xf32>
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["reduction"]
// CHECK: arith.addf
// CHECK: return
func.func @ordered_sum(%x:tensor<3x5xf32>,%seed:f32) -> tensor<f32> {
 %empty = tensor.empty() : tensor<f32>
 %initial = tensor.insert %seed into %empty[] : tensor<f32>
 %r = linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->()>],iterator_types=["reduction","reduction"]} ins(%x:tensor<3x5xf32>) outs(%initial:tensor<f32>) {
 ^bb0(%v:f32,%old:f32):
  %sum = arith.addf %old,%v : f32
  linalg.yield %sum : f32
 } -> tensor<f32>
 return %r : tensor<f32>
}

// A row-broadcast operand prevents collapsing only part of its indexing map.
// CHECK-LABEL: func.func @row_broadcast_reduction
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["reduction", "reduction"]
// CHECK: return
func.func @row_broadcast_reduction(%x:tensor<3x5xf32>,%rows:tensor<3xf32>,%seed:tensor<f32>) -> tensor<f32> {
 %r = linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i)>,affine_map<(i,j)->()>],iterator_types=["reduction","reduction"]} ins(%x,%rows:tensor<3x5xf32>,tensor<3xf32>) outs(%seed:tensor<f32>) {
 ^bb0(%v:f32,%weight:f32,%old:f32):
  %product = arith.mulf %v,%weight : f32
  %sum = arith.addf %old,%product : f32
  linalg.yield %sum : f32
 } -> tensor<f32>
 return %r : tensor<f32>
}
