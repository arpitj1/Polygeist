// RUN: polygeist-opt %s --normalize-linalg-matching-views | FileCheck %s
#blocked = affine_map<(i,j)->(i floordiv 3,j floordiv 5,i mod 3,j mod 5)>
func.func @read_blocks(%base:tensor<2x4x3x5xf32>) -> tensor<6x20xf32> {
 %n=arith.constant 6:index
 %m=arith.constant 20:index
 %v=polygeist.submap(%base,%n,%m) {map=#blocked} : (tensor<2x4x3x5xf32>,index,index)->tensor<6x20xf32>
 return %v:tensor<6x20xf32>
}
// CHECK-LABEL: func.func @read_blocks
// CHECK: polygeist.submap
// CHECK-SAME: tensor<2x3x4x5xf32>
// CHECK: tensor.collapse_shape
// CHECK-SAME: [0, 1], [2, 3]
// CHECK: return
func.func @write_blocks(%base:tensor<2x4x3x5xf32>,%v:tensor<6x20xf32>) -> tensor<2x4x3x5xf32> {
 %n=arith.constant 6:index
 %m=arith.constant 20:index
 %cast=tensor.cast %v:tensor<6x20xf32> to tensor<?x?xf32>
 %r=polygeist.submapInverse(%base,%cast,%n,%m) {map=#blocked} : (tensor<2x4x3x5xf32>,tensor<?x?xf32>,index,index)->tensor<2x4x3x5xf32>
 return %r:tensor<2x4x3x5xf32>
}
// CHECK-LABEL: func.func @write_blocks
// CHECK: tensor.expand_shape
// CHECK-SAME: [0, 1], [2, 3]
// CHECK: polygeist.submapInverse
// CHECK-SAME: tensor<2x3x4x5xf32>
// CHECK: return
func.func @incomplete_tile(%base:tensor<2x4x3x5xf32>) -> tensor<5x20xf32> {
 %n=arith.constant 5:index
 %m=arith.constant 20:index
 %v=polygeist.submap(%base,%n,%m) {map=#blocked} : (tensor<2x4x3x5xf32>,index,index)->tensor<5x20xf32>
 return %v:tensor<5x20xf32>
}
// CHECK-LABEL: func.func @incomplete_tile
// CHECK-NOT: tensor.collapse_shape
// CHECK: polygeist.submap
// CHECK-SAME: tensor<5x20xf32>
// CHECK: return
func.func @repeated_quotient(%base:tensor<2x2x3xf32>) -> tensor<6xf32> {
 %n=arith.constant 6:index
 %v=polygeist.submap(%base,%n) {map=affine_map<(i)->(i floordiv 3,i floordiv 3,i mod 3)>} : (tensor<2x2x3xf32>,index)->tensor<6xf32>
 return %v:tensor<6xf32>
}
// CHECK-LABEL: func.func @repeated_quotient
// CHECK-NOT: tensor.collapse_shape
// CHECK: polygeist.submap
// CHECK: return
