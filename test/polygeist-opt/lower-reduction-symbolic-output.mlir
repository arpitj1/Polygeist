// RUN: polygeist-opt %s --lower-polygeist-submap | FileCheck %s

// All reduction iterations update the scalar selected by a runtime index.
// Preserve that index while removing the aliased logical output dimension.
// CHECK-LABEL: func.func @scalar_output
// CHECK-SAME: %[[BASE:[a-zA-Z0-9_]+]]: tensor<?xf64>, %[[POS:[a-zA-Z0-9_]+]]: index,
// CHECK-NOT: polygeist.submap
// CHECK: tensor.extract_slice %[[BASE]][%[[POS]]]
// CHECK: linalg.generic
// CHECK: outs({{.*}} : tensor<f64>)
// CHECK: tensor.insert {{.*}} into %[[BASE]][%[[POS]]]
// CHECK-NOT: polygeist.submap
// CHECK: return
func.func @scalar_output(%input: tensor<?xf64>, %base: tensor<?xf64>,
                         %pos: index, %n: index) -> tensor<?xf64> {
  %view = polygeist.submap(%base, %pos, %n)
    {map = affine_map<(d0)[s0] -> (s0)>}
    : (tensor<?xf64>, index, index) -> tensor<?xf64>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["reduction"]
  } ins(%input : tensor<?xf64>) outs(%view : tensor<?xf64>) {
  ^bb0(%value: f64, %acc: f64):
    %next = arith.subf %acc, %value : f64
    linalg.yield %next : f64
  } -> tensor<?xf64>
  %updated = polygeist.submapInverse(%base, %result, %pos, %n)
    {map = affine_map<(d0)[s0] -> (s0)>}
    : (tensor<?xf64>, tensor<?xf64>, index, index) -> tensor<?xf64>
  return %updated : tensor<?xf64>
}

// Parallel rows retain their extent and a runtime translation; the reduction
// dimension disappears. Only that slice may be written back to the base.
// CHECK-LABEL: func.func @translated_output
// CHECK-SAME: %[[BASE:[a-zA-Z0-9_]+]]: tensor<?xf64>, %[[POS:[a-zA-Z0-9_]+]]: index,
// CHECK-NOT: polygeist.submap
// CHECK: linalg.generic
// CHECK: outs({{.*}} : tensor<?xf64>)
// CHECK: tensor.insert_slice {{.*}} into %[[BASE]][%[[POS]]]
// CHECK-NOT: polygeist.submap
// CHECK: return
func.func @translated_output(%input: tensor<?x?xf64>, %base: tensor<?xf64>,
                             %pos: index, %m: index, %n: index)
                             -> tensor<?xf64> {
  %view = polygeist.submap(%base, %pos, %m, %n)
    {map = affine_map<(d0, d1)[s0] -> (d0 + s0)>}
    : (tensor<?xf64>, index, index, index) -> tensor<?x?xf64>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%input : tensor<?x?xf64>) outs(%view : tensor<?x?xf64>) {
  ^bb0(%value: f64, %acc: f64):
    %next = arith.addf %acc, %value : f64
    linalg.yield %next : f64
  } -> tensor<?x?xf64>
  %updated = polygeist.submapInverse(%base, %result, %pos, %m, %n)
    {map = affine_map<(d0, d1)[s0] -> (d0 + s0)>}
    : (tensor<?xf64>, tensor<?x?xf64>, index, index, index) -> tensor<?xf64>
  return %updated : tensor<?xf64>
}
