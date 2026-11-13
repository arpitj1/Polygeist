#map = affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d5, d6, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d5 + d1, d6 + d2, d7 + d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cudnn_conv3d_backward_filter_valid_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: memref<?x?x?x?xf32>, %arg12: memref<?x?x?x?xf32>, %arg13: memref<?x?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg8 : i32 to index
    %2 = arith.index_cast %arg9 : i32 to index
    %3 = arith.index_cast %arg10 : i32 to index
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = arith.index_cast %arg5 : i32 to index
    %6 = arith.index_cast %arg6 : i32 to index
    %7 = arith.index_cast %arg7 : i32 to index
    %8 = polygeist.submap(%arg13, %0, %1, %2, %3, %4) {map = #map} : (memref<?x?x?x?x?xf32>, index, index, index, index, index) -> memref<?x?x?x?x?xf32>
    linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%8 : memref<?x?x?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    %9 = polygeist.submap(%arg12, %0, %1, %2, %3, %4, %5, %6, %7) {map = #map2} : (memref<?x?x?x?xf32>, index, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?x?xf32>
    %10 = polygeist.submap(%arg11, %0, %1, %2, %3, %4, %5, %6, %7) {map = #map3} : (memref<?x?x?x?xf32>, index, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?x?xf32>
    %11 = polygeist.submap(%arg13, %0, %1, %2, %3, %4, %5, %6, %7) {map = #map4} : (memref<?x?x?x?x?xf32>, index, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?x?xf32>
    linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%9, %10 : memref<?x?x?x?x?x?x?x?xf32>, memref<?x?x?x?x?x?x?x?xf32>) outs(%11 : memref<?x?x?x?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %12 = arith.mulf %in, %in_0 : f32
      %13 = arith.addf %12, %out : f32
      linalg.yield %13 : f32
    }
    return
  }
}

