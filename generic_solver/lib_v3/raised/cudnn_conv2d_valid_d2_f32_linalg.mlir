#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d5 * 2 + d2, d6 * 2 + d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cudnn_conv2d_valid_d2_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: memref<?x?x?x?xf32>, %arg10: memref<?x?x?x?xf32>, %arg11: memref<?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.index_cast %arg7 : i32 to index
    %3 = arith.index_cast %arg8 : i32 to index
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = arith.index_cast %arg5 : i32 to index
    %6 = arith.index_cast %arg6 : i32 to index
    %7 = polygeist.submap(%arg11, %0, %1, %2, %3) {map = #map} : (memref<?x?x?x?xf32>, index, index, index, index) -> memref<?x?x?x?xf32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%7 : memref<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    %8 = polygeist.submap(%arg10, %0, %1, %2, %3, %4, %5, %6) {map = #map1} : (memref<?x?x?x?xf32>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf32>
    %9 = polygeist.submap(%arg9, %0, %1, %2, %3, %4, %5, %6) {map = #map2} : (memref<?x?x?x?xf32>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf32>
    %10 = polygeist.submap(%arg11, %0, %1, %2, %3, %4, %5, %6) {map = #map3} : (memref<?x?x?x?xf32>, index, index, index, index, index, index, index) -> memref<?x?x?x?x?x?x?xf32>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%8, %9 : memref<?x?x?x?x?x?x?xf32>, memref<?x?x?x?x?x?x?xf32>) outs(%10 : memref<?x?x?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %11 = arith.mulf %in, %in_0 : f32
      %12 = arith.addf %11, %out : f32
      linalg.yield %12 : f32
    }
    return
  }
}

