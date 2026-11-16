// RUN: polygeist-opt %s --lower-polygeist-submap | FileCheck %s
// RUN: polygeist-opt %s --lower-polygeist-submap=affine-copy-writeback=true | FileCheck %s

// The diagonal is injective for any runtime extent. Use guarded scatter,
// including when a rectangular logical view reaches outside the base.
// CHECK-LABEL: func.func @dynamic_diagonal
// CHECK: scf.for
// CHECK: arith.cmpi ult
// CHECK: scf.if
// CHECK: tensor.insert
// CHECK-NOT: polygeist.submapInverse
// CHECK: return
func.func @dynamic_diagonal(%base: tensor<?x?xf64>, %view: tensor<?xf64>,
                            %n: index) -> tensor<?x?xf64> {
  %r = polygeist.submapInverse(%base, %view, %n)
      {map = affine_map<(d0) -> (d0, d0)>} :
      (tensor<?x?xf64>, tensor<?xf64>, index) -> tensor<?x?xf64>
  return %r : tensor<?x?xf64>
}

// Translations cancel in the uniqueness proof. The extent is deliberately
// independent of the source tensor shape: only the requested prefix is used.
// CHECK-LABEL: func.func @dynamic_shifted_triangle
// CHECK: scf.for {{.*}} to %arg3
// CHECK: scf.for {{.*}} to %arg4
// CHECK: arith.cmpi ult
// CHECK: scf.if
// CHECK: tensor.insert
// CHECK-NOT: polygeist.submapInverse
// CHECK: return
func.func @dynamic_shifted_triangle(%base: tensor<?x?xf64>,
    %view: tensor<?x?xf64>, %offset: index, %m: index, %n: index)
    -> tensor<?x?xf64> {
  %r = polygeist.submapInverse(%base, %view, %offset, %m, %n)
      {map = affine_map<(d0,d1)[s0] -> (d0,d0+d1+s0)>} :
      (tensor<?x?xf64>, tensor<?x?xf64>, index,index,index) -> tensor<?x?xf64>
  return %r : tensor<?x?xf64>
}

// Masked triangular loops must not speculatively read outside their base.
// CHECK-LABEL: func.func @dynamic_triangle_read
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arith.cmpi ult
// CHECK: scf.if
// CHECK: tensor.extract
// CHECK: else
// CHECK-NOT: polygeist.submap
// CHECK: return
func.func @dynamic_triangle_read(%base: tensor<?x?xf64>, %m: index, %n: index)
    -> tensor<?x?xf64> {
  %r = polygeist.submap(%base, %m, %n)
      {map = affine_map<(d0,d1) -> (d0,d0+d1+1)>} :
      (tensor<?x?xf64>, index,index) -> tensor<?x?xf64>
  return %r : tensor<?x?xf64>
}

// A linear map alone is insufficient: collisions remain illegal.
// CHECK-LABEL: func.func @dynamic_collision
// CHECK: polygeist.submapInverse
// CHECK: return
func.func @dynamic_collision(%base: tensor<?xf64>, %view: tensor<?x?xf64>,
                             %m: index, %n: index) -> tensor<?xf64> {
  %r = polygeist.submapInverse(%base, %view, %m, %n)
      {map = affine_map<(d0,d1) -> (d0+d1)>} :
      (tensor<?xf64>, tensor<?x?xf64>, index,index) -> tensor<?xf64>
  return %r : tensor<?xf64>
}

// Retain known bounds in a mixed domain: a four-wide row is injective even
// when the row count is unknown. A fifth column would overlap the next row.
// CHECK-LABEL: func.func @mixed_domain
// CHECK: scf.for
// CHECK: scf.for
// CHECK: tensor.insert
// CHECK-NOT: polygeist.submapInverse
// CHECK: return
func.func @mixed_domain(%base: tensor<?xf64>, %view: tensor<?x4xf64>,
                        %n: index) -> tensor<?xf64> {
  %c4 = arith.constant 4 : index
  %r = polygeist.submapInverse(%base, %view, %n, %c4)
      {map = affine_map<(d0,d1) -> (4*d0+d1)>} :
      (tensor<?xf64>, tensor<?x4xf64>, index,index) -> tensor<?xf64>
  return %r : tensor<?xf64>
}

// CHECK-LABEL: func.func @mixed_domain_collision
// CHECK: polygeist.submapInverse
// CHECK: return
func.func @mixed_domain_collision(%base: tensor<?xf64>, %view: tensor<?x5xf64>,
                                  %n: index) -> tensor<?xf64> {
  %c5 = arith.constant 5 : index
  %r = polygeist.submapInverse(%base, %view, %n, %c5)
      {map = affine_map<(d0,d1) -> (4*d0+d1)>} :
      (tensor<?xf64>, tensor<?x5xf64>, index,index) -> tensor<?xf64>
  return %r : tensor<?xf64>
}

// A symbol-dependent stride may be zero. Do not discard it as a translation.
// CHECK-LABEL: func.func @dynamic_stride_not_proven
// CHECK: polygeist.submapInverse
// CHECK: return
func.func @dynamic_stride_not_proven(%base: tensor<?xf64>, %view: tensor<?xf64>,
                                    %s: index, %n: index) -> tensor<?xf64> {
  %r = polygeist.submapInverse(%base, %view, %s, %n)
      {map = affine_map<(d0)[s0] -> (d0+d0*s0)>} :
      (tensor<?xf64>, tensor<?xf64>, index,index) -> tensor<?xf64>
  return %r : tensor<?xf64>
}

// Drop only the declared reduction dimension before scattering. The two
// retained dimensions have a dynamic, injective triangular output map.
// CHECK-LABEL: func.func @dynamic_triangle_reduction
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: outs({{.*}} : tensor<?x?xf64>)
// CHECK-NOT: polygeist.submapInverse
// CHECK: return
func.func @dynamic_triangle_reduction(%base: tensor<?x?xf64>,
    %input: tensor<?x?x?xf64>, %m: index, %n: index, %k: index)
    -> tensor<?x?xf64> {
  %v = polygeist.submap(%base, %m, %n, %k)
      {map = affine_map<(d0,d1,d2) -> (d0,d0+d1+1)>} :
      (tensor<?x?xf64>, index,index,index) -> tensor<?x?x?xf64>
  %r = linalg.generic {
      indexing_maps = [affine_map<(d0,d1,d2) -> (d0,d1,d2)>,
                       affine_map<(d0,d1,d2) -> (d0,d1,d2)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%input : tensor<?x?x?xf64>) outs(%v : tensor<?x?x?xf64>) {
  ^bb0(%in: f64, %acc: f64):
    %sum = arith.addf %in, %acc : f64
    linalg.yield %sum : f64
  } -> tensor<?x?x?xf64>
  %out = polygeist.submapInverse(%base, %r, %m, %n, %k)
      {map = affine_map<(d0,d1,d2) -> (d0,d0+d1+1)>} :
      (tensor<?x?xf64>, tensor<?x?x?xf64>, index,index,index) -> tensor<?x?xf64>
  return %out : tensor<?x?xf64>
}
