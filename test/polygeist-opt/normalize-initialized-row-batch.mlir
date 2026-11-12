// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-operand-rank=2 fuse-initializers=true" | FileCheck %s
// Both parallel and reduction groups retain their original order. The same
// dynamic scalar seed must remain visible after the upstream rank collapse.
func.func @unrelated(%input: tensor<2x3x5x7xf32>, %output: tensor<2x3xf32>, %seed: f32) -> tensor<2x3xf32> {
  %filled = linalg.generic {indexing_maps=[affine_map<(a,b)->(a,b)>], iterator_types=["parallel","parallel"]} outs(%output:tensor<2x3xf32>) {
  ^bb0(%old:f32):
    linalg.yield %seed : f32
  } -> tensor<2x3xf32>
  %result = linalg.generic {indexing_maps=[affine_map<(a,b,c,d)->(a,b,c,d)>,affine_map<(a,b,c,d)->(a,b)>], iterator_types=["parallel","parallel","reduction","reduction"]} ins(%input:tensor<2x3x5x7xf32>) outs(%filled:tensor<2x3xf32>) {
  ^bb0(%value:f32,%acc:f32):
    %next = arith.addf %acc,%value : f32
    linalg.yield %next : f32
  } -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}
// CHECK-LABEL: func.func @unrelated
// CHECK-NOT: scf.for
// CHECK: linalg.generic
// CHECK: linalg.yield %arg2
// CHECK: tensor<6xf32>
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: tensor<6x35xf32>
// CHECK: arith.addf
// CHECK-NOT: scf.for
// CHECK: return
