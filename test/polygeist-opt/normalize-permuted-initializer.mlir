// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-operand-rank=2 fuse-initializers=true" | FileCheck %s

// A captured scalar is unchanged by transposing a fully covered rectangle.
// CHECK-LABEL: func.func @permuted_fill
// CHECK-NOT: polygeist.submapInverse
// CHECK: linalg.generic
// CHECK: linalg.yield %arg3
// CHECK-NOT: polygeist.submapInverse
// CHECK: return
func.func @permuted_fill(%base: tensor<?x?xf32>, %m: index, %n: index, %seed: f32) -> tensor<?x?xf32> {
  %view = polygeist.submap(%base, %m, %n) {map = affine_map<(i,j)->(j,i)>} : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
  %fill = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>], iterator_types = ["parallel", "parallel"]} outs(%view: tensor<?x?xf32>) {
  ^bb0(%old: f32):
    linalg.yield %seed : f32
  } -> tensor<?x?xf32>
  %updated = polygeist.submapInverse(%base, %fill, %m, %n) {map = affine_map<(i,j)->(j,i)>} : (tensor<?x?xf32>, tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
  %read = polygeist.submap(%updated, %n, %m) {map = affine_map<(i,j)->(i,j)>} : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
  return %read : tensor<?x?xf32>
}

// A read of a different rectangle must preserve the writeback.
// CHECK-LABEL: func.func @partial_fill
// CHECK: polygeist.submapInverse
// CHECK: polygeist.submap(
// CHECK: return
func.func @partial_fill(%base: tensor<?x?xf32>, %m: index, %n: index, %k: index, %seed: f32) -> tensor<?x?xf32> {
  %view = polygeist.submap(%base, %m, %n) {map = affine_map<(i,j)->(j,i)>} : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
  %fill = linalg.generic {indexing_maps = [affine_map<(i,j)->(i,j)>], iterator_types = ["parallel", "parallel"]} outs(%view: tensor<?x?xf32>) {
  ^bb0(%old: f32):
    linalg.yield %seed : f32
  } -> tensor<?x?xf32>
  %updated = polygeist.submapInverse(%base, %fill, %m, %n) {map = affine_map<(i,j)->(j,i)>} : (tensor<?x?xf32>, tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
  %read = polygeist.submap(%updated, %n, %k) {map = affine_map<(i,j)->(i,j)>} : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
  return %read : tensor<?x?xf32>
}

// An arbitrary tensor is not a uniform fill, even with identical bounds.
// CHECK-LABEL: func.func @nonuniform
// CHECK: polygeist.submapInverse
// CHECK: polygeist.submap(
// CHECK: return
func.func @nonuniform(%base: tensor<?x?xf32>, %value: tensor<?x?xf32>, %m: index, %n: index) -> tensor<?x?xf32> {
  %updated = polygeist.submapInverse(%base, %value, %m, %n) {map = affine_map<(i,j)->(j,i)>} : (tensor<?x?xf32>, tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
  %read = polygeist.submap(%updated, %n, %m) {map = affine_map<(i,j)->(i,j)>} : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
  return %read : tensor<?x?xf32>
}

// A write to the same physical rectangle supersedes the previous write.
// CHECK-LABEL: func.func @permuted_overwrite
// CHECK-NOT: polygeist.submapInverse
// CHECK: %[[WRITE:.*]] = polygeist.submapInverse(%arg0, %arg2, %arg4, %arg3)
// CHECK-NOT: polygeist.submapInverse
// CHECK: return %[[WRITE]]
func.func @permuted_overwrite(%base: tensor<?x?xf32>, %first: tensor<?x?xf32>, %second: tensor<?x?xf32>, %m: index, %n: index) -> tensor<?x?xf32> {
  %updated = polygeist.submapInverse(%base, %first, %m, %n) {map = affine_map<(i,j)->(j,i)>} : (tensor<?x?xf32>, tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
  %final = polygeist.submapInverse(%updated, %second, %n, %m) {map = affine_map<(i,j)->(i,j)>} : (tensor<?x?xf32>, tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
  return %final : tensor<?x?xf32>
}

// A partial overwrite cannot discard the rest of the previous rectangle.
// CHECK-LABEL: func.func @partial_overwrite
// CHECK: polygeist.submapInverse
// CHECK: polygeist.submapInverse
// CHECK: return
func.func @partial_overwrite(%base: tensor<?x?xf32>, %first: tensor<?x?xf32>, %second: tensor<?x?xf32>, %m: index, %n: index, %k: index) -> tensor<?x?xf32> {
  %updated = polygeist.submapInverse(%base, %first, %m, %n) {map = affine_map<(i,j)->(j,i)>} : (tensor<?x?xf32>, tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
  %final = polygeist.submapInverse(%updated, %second, %n, %k) {map = affine_map<(i,j)->(i,j)>} : (tensor<?x?xf32>, tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
  return %final : tensor<?x?xf32>
}
