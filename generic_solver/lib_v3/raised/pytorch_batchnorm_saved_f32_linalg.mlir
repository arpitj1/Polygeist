#map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @pytorch_batchnorm_saved_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<?x?x?x?xf32>, %arg5: memref<?xf32>, %arg6: memref<?xf32>, %arg7: memref<?xf32>, %arg8: memref<?xf32>, %arg9: memref<?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.index_cast %arg2 : i32 to index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = polygeist.submap(%arg4, %0, %1, %2, %3) {map = #map} : (memref<?x?x?x?xf32>, index, index, index, index) -> memref<?x?x?x?xf32>
    %5 = polygeist.submap(%arg9, %0, %1, %2, %3) {map = #map} : (memref<?x?x?x?xf32>, index, index, index, index) -> memref<?x?x?x?xf32>
    %6 = polygeist.submap(%arg7, %0) {map = #map1} : (memref<?xf32>, index) -> memref<?xf32>
    %7 = polygeist.submap(%6, %0, %1, %2, %3) {map = #map2} : (memref<?xf32>, index, index, index, index) -> memref<?x?x?x?xf32>
    %8 = polygeist.submap(%arg8, %0) {map = #map1} : (memref<?xf32>, index) -> memref<?xf32>
    %9 = polygeist.submap(%8, %0, %1, %2, %3) {map = #map2} : (memref<?xf32>, index, index, index, index) -> memref<?x?x?x?xf32>
    %10 = polygeist.submap(%arg5, %0) {map = #map1} : (memref<?xf32>, index) -> memref<?xf32>
    %11 = polygeist.submap(%10, %0, %1, %2, %3) {map = #map2} : (memref<?xf32>, index, index, index, index) -> memref<?x?x?x?xf32>
    %12 = polygeist.submap(%arg6, %0) {map = #map1} : (memref<?xf32>, index) -> memref<?xf32>
    %13 = polygeist.submap(%12, %0, %1, %2, %3) {map = #map2} : (memref<?xf32>, index, index, index, index) -> memref<?x?x?x?xf32>
    linalg.generic {indexing_maps = [#map3, #map3, #map3, #map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7, %9, %11, %13, %4 : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>) outs(%5 : memref<?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %in_3: f32, %out: f32):
      %14 = arith.subf %in_3, %in_1 : f32
      %15 = arith.mulf %in, %14 : f32
      %16 = arith.mulf %15, %in_2 : f32
      %17 = arith.addf %16, %in_0 : f32
      linalg.yield %17 : f32
    }
    return
  }
}

