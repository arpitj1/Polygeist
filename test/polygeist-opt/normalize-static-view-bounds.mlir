// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-operand-rank=1 fuse-initializers=true" | FileCheck %s

// Constant bounds define the logical view's shape even when its storage map
// interleaves coordinates. Preserve those bounds so upstream rank collapse
// does not introduce one loop/call per row merely to recover static dimensions.
// CHECK-LABEL: func.func @interleaved_view
// CHECK-NOT: scf.for
// CHECK: tensor.collapse_shape
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK-NOT: scf.for
// CHECK: return
func.func @interleaved_view(%x: tensor<3x4xf32>, %y: tensor<12xf32>) -> tensor<12xf32> {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %a = polygeist.submap(%x, %c3, %c4) {map=affine_map<(i,j)->(i,j)>} : (tensor<3x4xf32>,index,index) -> tensor<?x?xf32>
  %b = polygeist.submap(%y, %c3, %c4) {map=affine_map<(i,j)->(j*3+i)>} : (tensor<12xf32>,index,index) -> tensor<?x?xf32>
  %r = linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]} ins(%a:tensor<?x?xf32>) outs(%b:tensor<?x?xf32>) {
  ^bb0(%v:f32,%old:f32):
    linalg.yield %v : f32
  } -> tensor<?x?xf32>
  %out = polygeist.submapInverse(%y,%r,%c3,%c4) {map=affine_map<(i,j)->(j*3+i)>} : (tensor<12xf32>,tensor<?x?xf32>,index,index) -> tensor<12xf32>
  return %out : tensor<12xf32>
}
