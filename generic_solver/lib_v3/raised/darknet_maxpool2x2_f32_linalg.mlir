#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4 + d2 * 2, d5 + d3 * 2)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @darknet_maxpool2x2_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<?x?x?x?xf32>, %arg5: memref<?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %cst = arith.constant -3.40282347E+38 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = arith.index_cast %arg2 : i32 to index
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = polygeist.submap(%arg5, %0, %1, %2, %3) {map = #map} : (memref<?x?x?x?xf32>, index, index, index, index) -> memref<?x?x?x?xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%4 : memref<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    %5 = polygeist.submap(%arg4, %0, %1, %2, %3, %c2, %c2) {map = #map1} : (memref<?x?x?x?xf32>, index, index, index, index, index, index) -> memref<?x?x?x?x?x?xf32>
    %6 = polygeist.submap(%arg5, %0, %1, %2, %3, %c2, %c2) {map = #map2} : (memref<?x?x?x?xf32>, index, index, index, index, index, index) -> memref<?x?x?x?x?x?xf32>
    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%5 : memref<?x?x?x?x?x?xf32>) outs(%6 : memref<?x?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %7 = arith.cmpf ogt, %in, %out : f32
      %8 = arith.select %7, %in, %out : f32
      linalg.yield %8 : f32
    }
    return
  }
}

