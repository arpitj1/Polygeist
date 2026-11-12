// RUN: polygeist-opt %s --lower-polygeist-submap=affine-copy-writeback=true | FileCheck %s

// Large domains must use a proof of uniqueness, not element enumeration.
// CHECK-LABEL: func.func @large_tensor_product
// CHECK-NOT: polygeist.submapInverse
// CHECK: linalg.generic
// CHECK-NOT: polygeist.submapInverse
// CHECK: return
func.func @large_tensor_product(%base: tensor<?x4096xf32>,
    %view: tensor<256x128x32x32xf32>) -> tensor<?x4096xf32> {
  %a = arith.constant 256 : index
  %b = arith.constant 128 : index
  %c = arith.constant 32 : index
  %r = polygeist.submapInverse(%base, %view, %a, %b, %c, %c)
      {map = affine_map<(d0,d1,d2,d3) -> (d0*32+d2,d1*32+d3)>} :
      (tensor<?x4096xf32>, tensor<256x128x32x32xf32>, index,index,index,index)
      -> tensor<?x4096xf32>
  return %r : tensor<?x4096xf32>
}

// CHECK-LABEL: func.func @large_permuted_split_dimension
// CHECK-NOT: polygeist.submapInverse
// CHECK: linalg.generic
// CHECK-NOT: polygeist.submapInverse
// CHECK: return
func.func @large_permuted_split_dimension(%base: tensor<?x1053x34x34xf32>,
    %view: tensor<4x13x34x34x9x9xf32>) -> tensor<?x1053x34x34xf32> {
  %b = arith.constant 4 : index
  %c = arith.constant 13 : index
  %h = arith.constant 34 : index
  %r = arith.constant 9 : index
  %v = polygeist.submapInverse(%base, %view, %b, %c, %h, %h, %r, %r)
      {map = affine_map<(d0,d1,d2,d3,d4,d5) -> (d0,d1*81+d4*9+d5,d2,d3)>} :
      (tensor<?x1053x34x34xf32>, tensor<4x13x34x34x9x9xf32>,
       index,index,index,index,index,index) -> tensor<?x1053x34x34xf32>
  return %v : tensor<?x1053x34x34xf32>
}

// A one-element overlap must be rejected even for the large domain.
// CHECK-LABEL: func.func @large_collision
// CHECK: polygeist.submapInverse
// CHECK: return
func.func @large_collision(%base: tensor<?x4096xf32>,
    %view: tensor<256x128x32x32xf32>) -> tensor<?x4096xf32> {
  %a = arith.constant 256 : index
  %b = arith.constant 128 : index
  %c = arith.constant 32 : index
  %r = polygeist.submapInverse(%base, %view, %a, %b, %c, %c)
      {map = affine_map<(d0,d1,d2,d3) -> (d0*31+d2,d1*32+d3)>} :
      (tensor<?x4096xf32>, tensor<256x128x32x32xf32>, index,index,index,index)
      -> tensor<?x4096xf32>
  return %r : tensor<?x4096xf32>
}

// Reversed dimensions and padding still have unique destinations.
// CHECK-LABEL: func.func @large_reversed_padded
// CHECK-NOT: polygeist.submapInverse
// CHECK: linalg.generic
// CHECK-NOT: polygeist.submapInverse
// CHECK: return
func.func @large_reversed_padded(%base: tensor<?xf32>,
    %view: tensor<131072x64xf32>) -> tensor<?xf32> {
  %rows = arith.constant 131072 : index
  %cols = arith.constant 64 : index
  %r = polygeist.submapInverse(%base, %view, %rows, %cols)
      {map = affine_map<(d0,d1) -> (d0*65-d1+63)>} :
      (tensor<?xf32>, tensor<131072x64xf32>, index,index) -> tensor<?xf32>
  return %r : tensor<?xf32>
}
