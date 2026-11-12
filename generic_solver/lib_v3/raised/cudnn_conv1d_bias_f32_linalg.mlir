#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2)[s0] -> (s0, d1, d2)>
#map2 = affine_map<(d0, d1, d2)[s0] -> (s0, d1, d2 + d0)>
#map3 = affine_map<(d0, d1, d2) -> (d0)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0)[s0] -> (s0)>
#map6 = affine_map<(d0)[s0, s1] -> (s0, s1, d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cudnn_conv1d_bias_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x?x?xf32>, %arg6: memref<?x?x?xf32>, %arg7: memref<?xf32>, %arg8: memref<?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.index_cast %arg4 : i32 to index
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.index_cast %arg3 : i32 to index
    affine.for %arg9 = 0 to %0 {
      affine.for %arg10 = 0 to %1 {
        %alloca = memref.alloca(%2) : memref<?xf32>
        %5 = polygeist.submap(%alloca, %2) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
        linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%5 : memref<?xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        }
        %6 = polygeist.submap(%arg6, %arg10, %2, %3, %4) {map = #map1} : (memref<?x?x?xf32>, index, index, index, index) -> memref<?x?x?xf32>
        %7 = polygeist.submap(%arg5, %arg9, %2, %3, %4) {map = #map2} : (memref<?x?x?xf32>, index, index, index, index) -> memref<?x?x?xf32>
        %8 = polygeist.submap(%alloca, %2, %3, %4) {map = #map3} : (memref<?xf32>, index, index, index) -> memref<?x?x?xf32>
        linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "reduction", "reduction"]} ins(%6, %7 : memref<?x?x?xf32>, memref<?x?x?xf32>) outs(%8 : memref<?x?x?xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %12 = arith.mulf %in, %in_0 : f32
          %13 = arith.addf %12, %out : f32
          linalg.yield %13 : f32
        }
        %9 = polygeist.submap(%alloca, %2) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
        %10 = polygeist.submap(%arg7, %arg10, %2) {map = #map5} : (memref<?xf32>, index, index) -> memref<?xf32>
        %11 = polygeist.submap(%arg8, %arg9, %arg10, %2) {map = #map6} : (memref<?x?x?xf32>, index, index, index) -> memref<?xf32>
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%9, %10 : memref<?xf32>, memref<?xf32>) outs(%11 : memref<?xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %12 = arith.addf %in, %in_0 : f32
          linalg.yield %12 : f32
        }
      } {polygeist.was_parallel}
    } {polygeist.was_parallel}
    return
  }
}

