#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (d0, 0)>
#map3 = affine_map<(d0, d1) -> (d0, d1 + 1)>
#map4 = affine_map<(d0, d1) -> (d0)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cub_segmented_argmin_f32(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf32>, %arg3: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = affine.apply #map()[%1]
    %alloca = memref.alloca(%0) : memref<?xi32>
    %alloca_0 = memref.alloca(%0) : memref<?xf32>
    %3 = polygeist.submap(%alloca, %0) {map = #map1} : (memref<?xi32>, index) -> memref<?xi32>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%3 : memref<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    }
    %4 = polygeist.submap(%arg2, %0) {map = #map2} : (memref<?x?xf32>, index) -> memref<?xf32>
    %5 = polygeist.submap(%alloca_0, %0) {map = #map1} : (memref<?xf32>, index) -> memref<?xf32>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%4 : memref<?xf32>) outs(%5 : memref<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %6 = polygeist.submap(%arg2, %0, %2) {map = #map3} : (memref<?x?xf32>, index, index) -> memref<?x?xf32>
    %7 = polygeist.submap(%alloca, %0, %2) {map = #map4} : (memref<?xi32>, index, index) -> memref<?x?xi32>
    %8 = polygeist.submap(%alloca_0, %0, %2) {map = #map4} : (memref<?xf32>, index, index) -> memref<?x?xf32>
    linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "reduction"]} ins(%6 : memref<?x?xf32>) outs(%7, %8 : memref<?x?xi32>, memref<?x?xf32>) {
    ^bb0(%in: f32, %out: i32, %out_1: f32):
      %11 = linalg.index 1 : index
      %12 = affine.apply #map6(%11)
      %13 = arith.index_cast %12 : index to i32
      %14 = arith.cmpf olt, %in, %out_1 : f32
      %15 = arith.select %14, %13, %out : i32
      %16 = arith.select %14, %in, %out_1 : f32
      linalg.yield %15, %16 : i32, f32
    }
    %9 = polygeist.submap(%alloca, %0) {map = #map1} : (memref<?xi32>, index) -> memref<?xi32>
    %10 = polygeist.submap(%arg3, %0) {map = #map1} : (memref<?xi32>, index) -> memref<?xi32>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%9 : memref<?xi32>) outs(%10 : memref<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    }
    return
  }
}

