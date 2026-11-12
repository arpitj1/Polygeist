// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-operand-rank=1 fuse-initializers=true" | FileCheck %s
// An invariant scalar tensor becomes a scalar capture; element type and the
// number of assignments remain unchanged.
// CHECK-LABEL: func.func @broadcast_value
// CHECK: %[[SCALAR:.*]] = tensor.extract %{{.*}}[] : tensor<f32>
// CHECK: linalg.generic
// CHECK-NOT: ins(
// CHECK: linalg.yield %[[SCALAR]] : f32
func.func @broadcast_value(%scalar:tensor<f32>,%out:tensor<17xf32>) -> tensor<17xf32> {
 %r = linalg.generic {indexing_maps=[affine_map<(i)->()>,affine_map<(i)->(i)>],iterator_types=["parallel"]} ins(%scalar:tensor<f32>) outs(%out:tensor<17xf32>) {
 ^bb0(%x:f32,%old:f32):
  linalg.yield %x : f32
 } -> tensor<17xf32>
 return %r : tensor<17xf32>
}
