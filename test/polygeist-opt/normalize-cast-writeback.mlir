// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-operand-rank=1 fuse-initializers=true" | FileCheck %s

// Shape refinement may cast an expanded result back to a dynamic shape. The
// cast retains the same coordinates, so the bounded writeback still collapses.
// CHECK-LABEL: func.func @cast_writeback
// CHECK: polygeist.submapInverse
// CHECK-SAME: (tensor<?xf32>, tensor<12xf32>, index) -> tensor<?xf32>
func.func @cast_writeback(%base: tensor<?x4xf32>, %new: tensor<12xf32>) -> tensor<?x4xf32> {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %expanded = tensor.expand_shape %new [[0, 1]] : tensor<12xf32> into tensor<3x4xf32>
  %cast = tensor.cast %expanded : tensor<3x4xf32> to tensor<?x?xf32>
  %out = polygeist.submapInverse(%base,%cast,%c3,%c4) {map=affine_map<(i,j)->(i,j)>} : (tensor<?x4xf32>,tensor<?x?xf32>,index,index) -> tensor<?x4xf32>
  return %out : tensor<?x4xf32>
}
