// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-operand-rank=1 fuse-initializers=true" | FileCheck %s
#id = affine_map<(i)->(i)>
func.func @nested(%parent:tensor<?xf32>,%old:tensor<6xf32>,%new:tensor<6xf32>) -> tensor<?xf32> {
 %n=arith.constant 6:index
 %inner=polygeist.submapInverse(%old,%new,%n) {map=#id} : (tensor<6xf32>,tensor<6xf32>,index)->tensor<6xf32>
 %outer=polygeist.submapInverse(%parent,%inner,%n) {map=#id} : (tensor<?xf32>,tensor<6xf32>,index)->tensor<?xf32>
 return %outer:tensor<?xf32>
}
// CHECK-LABEL: func.func @nested
// CHECK: polygeist.submapInverse(%arg0, %arg2,
// CHECK-NOT: polygeist.submapInverse
// CHECK: return
func.func @standalone(%old:tensor<6xf32>,%new:tensor<6xf32>) -> tensor<6xf32> {
 %n=arith.constant 6:index
 %inner=polygeist.submapInverse(%old,%new,%n) {map=#id} : (tensor<6xf32>,tensor<6xf32>,index)->tensor<6xf32>
 return %inner:tensor<6xf32>
}
// CHECK-LABEL: func.func @standalone
// CHECK: polygeist.submapInverse
// CHECK: return
func.func @partial(%parent:tensor<?xf32>,%old:tensor<6xf32>,%new:tensor<6xf32>) -> tensor<?xf32> {
 %three=arith.constant 3:index
 %six=arith.constant 6:index
 %inner=polygeist.submapInverse(%old,%new,%three) {map=#id} : (tensor<6xf32>,tensor<6xf32>,index)->tensor<6xf32>
 %outer=polygeist.submapInverse(%parent,%inner,%six) {map=#id} : (tensor<?xf32>,tensor<6xf32>,index)->tensor<?xf32>
 return %outer:tensor<?xf32>
}
// CHECK-LABEL: func.func @partial
// CHECK: polygeist.submapInverse
// CHECK: polygeist.submapInverse
// CHECK: return
