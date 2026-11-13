#map = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (d0, d0)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d1 + d0 + 1, d0)>
#map9 = affine_map<(d0, d1, d2) -> (d1 + d0 + 1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dsymm_ru(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>, %arg7: f64, %arg8: i32, %arg9: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.subi %0, %c1 : index
    %3 = affine.apply #map(%c0)[%0]
    %4 = polygeist.submap(%arg6, %0, %1) {map = #map1} : (memref<?x?xf64>, index, index) -> memref<?x?xf64>
    %5 = polygeist.submap(%arg9, %0, %1) {map = #map1} : (memref<?x?xf64>, index, index) -> memref<?x?xf64>
    %6 = polygeist.submap(%arg4, %0) {map = #map2} : (memref<?x?xf64>, index) -> memref<?xf64>
    %7 = polygeist.submap(%6, %0, %1) {map = #map3} : (memref<?xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%7, %4 : memref<?x?xf64>, memref<?x?xf64>) outs(%5 : memref<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %14 = arith.mulf %arg2, %in : f64
      %15 = arith.mulf %arg7, %out : f64
      %16 = arith.mulf %14, %in_0 : f64
      %17 = arith.addf %15, %16 : f64
      linalg.yield %17 : f64
    }
    %8 = polygeist.submap(%arg4, %0, %2, %1) {map = #map4} : (memref<?x?xf64>, index, index, index) -> memref<?x?x?xf64>
    %9 = polygeist.submap(%arg6, %0, %2, %1) {map = #map5} : (memref<?x?xf64>, index, index, index) -> memref<?x?x?xf64>
    %10 = polygeist.submap(%arg9, %0, %2, %1) {map = #map6} : (memref<?x?xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "reduction", "parallel"]} ins(%8, %9 : memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%10 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %14 = linalg.index 0 : index
      %15 = arith.mulf %arg2, %in : f64
      %16 = arith.mulf %15, %in_0 : f64
      %17 = arith.addf %out, %16 : f64
      %18 = linalg.index 1 : index
      %19 = arith.cmpi slt, %18, %14 : index
      %20 = arith.select %19, %17, %out : f64
      linalg.yield %20 : f64
    }
    %11 = polygeist.submap(%arg4, %0, %3, %1) {map = #map8} : (memref<?x?xf64>, index, index, index) -> memref<?x?x?xf64>
    %12 = polygeist.submap(%arg6, %0, %3, %1) {map = #map9} : (memref<?x?xf64>, index, index, index) -> memref<?x?x?xf64>
    %13 = polygeist.submap(%arg9, %0, %3, %1) {map = #map6} : (memref<?x?xf64>, index, index, index) -> memref<?x?x?xf64>
    linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "reduction", "parallel"]} ins(%11, %12 : memref<?x?x?xf64>, memref<?x?x?xf64>) outs(%13 : memref<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %14 = linalg.index 0 : index
      %15 = arith.mulf %arg2, %in : f64
      %16 = arith.mulf %15, %in_0 : f64
      %17 = arith.addf %out, %16 : f64
      %18 = linalg.index 1 : index
      %19 = affine.apply #map(%14)[%0]
      %20 = arith.cmpi slt, %18, %19 : index
      %21 = arith.select %20, %17, %out : f64
      linalg.yield %21 : f64
    }
    return
  }
}

