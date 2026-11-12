// RUN: polygeist-opt %s --lower-polygeist-submap | FileCheck %s

// Repeated window coordinates remain a read-only strided view. In particular,
// no gather loop may read a potentially device-resident source on the host.
// CHECK-LABEL: func.func @window
// CHECK: memref.reinterpret_cast
// CHECK-SAME: strides: [8, 1, 1]
// CHECK-NOT: linalg.generic
// CHECK: return
func.func @window(%a: memref<4x8xf32>) -> tensor<4x3x6xf32> {
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %t = bufferization.to_tensor %a : memref<4x8xf32>
  %v = polygeist.submap(%t, %c4, %c3, %c6) {map = affine_map<(i,j,k)->(i,j+k)>} : (tensor<4x8xf32>,index,index,index) -> tensor<4x3x6xf32>
  return %v : tensor<4x3x6xf32>
}

// Physical source stride and offset participate in the composed address.
// CHECK-LABEL: func.func @strided_source
// CHECK: memref.reinterpret_cast
// CHECK-SAME: offset: [5]
// CHECK-SAME: strides: [4]
// CHECK: return
func.func @strided_source(%a: memref<32xf32,strided<[2],offset:3>>) -> tensor<4xf32> {
  %c4 = arith.constant 4 : index
  %t = bufferization.to_tensor %a : memref<32xf32,strided<[2],offset:3>>
  %v = polygeist.submap(%t,%c4) {map=affine_map<(i)->(2*i+1)>} : (tensor<32xf32>,index) -> tensor<4xf32>
  return %v : tensor<4xf32>
}

// Writable tensor roots retain materialization and writeback semantics.
// CHECK-LABEL: func.func @writable_window
// CHECK-NOT: memref.reinterpret_cast
// CHECK: linalg.generic
// CHECK: return
func.func @writable_window(%a: memref<4x8xf32>) -> tensor<4x3x6xf32> {
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %c6 = arith.constant 6 : index
  %t = bufferization.to_tensor %a writable : memref<4x8xf32>
  %v = polygeist.submap(%t, %c4, %c3, %c6) {map = affine_map<(i,j,k)->(i,j+k)>} : (tensor<4x8xf32>,index,index,index) -> tensor<4x3x6xf32>
  return %v : tensor<4x3x6xf32>
}
