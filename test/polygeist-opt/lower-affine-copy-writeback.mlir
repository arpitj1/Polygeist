// RUN: polygeist-opt %s --lower-polygeist-submap="affine-copy-writeback=true" | FileCheck %s
// A bounded update uses only the specified prefix of its source. The remaining
// source elements must not become additional stores through shape inference.
// CHECK-LABEL: func.func @bounded_permuted_update
// CHECK: tensor.extract_slice {{.*}}[0, 0] [3, 4] [1, 1]
// CHECK-SAME: tensor<5x7xf32> to tensor<3x4xf32>
// CHECK: linalg.generic
// CHECK-NOT: scf.for
// CHECK: return
func.func @bounded_permuted_update(%base:tensor<12xf32>, %values:tensor<5x7xf32>) -> tensor<12xf32> {
 %c3 = arith.constant 3 : index
 %c4 = arith.constant 4 : index
 %r = polygeist.submapInverse(%base,%values,%c3,%c4) {map=affine_map<(i,j)->(j*3+i)>} : (tensor<12xf32>,tensor<5x7xf32>,index,index) -> tensor<12xf32>
 return %r : tensor<12xf32>
}
