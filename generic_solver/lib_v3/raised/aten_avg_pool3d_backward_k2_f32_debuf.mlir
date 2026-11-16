#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4 + d1 * 2, d5 + d2 * 2, d6 + d3 * 2)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_avg_pool3d_backward_k2_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: memref<?x?x?x?xf32>, %arg8: memref<?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 8.000000e+00 : f32
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg8 : memref<?x?x?x?xf32>
    %1 = bufferization.to_tensor %arg7 : memref<?x?x?x?xf32>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = arith.muli %arg1, %c2_i32 : i32
    %4 = arith.addi %3, %arg4 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6 = arith.muli %arg2, %c2_i32 : i32
    %7 = arith.addi %6, %arg5 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = arith.muli %arg3, %c2_i32 : i32
    %10 = arith.addi %9, %arg6 : i32
    %11 = arith.index_cast %10 : i32 to index
    %12 = arith.index_cast %arg1 : i32 to index
    %13 = arith.index_cast %arg2 : i32 to index
    %14 = arith.index_cast %arg3 : i32 to index
    %15 = polygeist.submap(%0, %2, %5, %8, %11) {map = #map} : (tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %16 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%15 : tensor<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?x?x?x?xf32>
    %17 = polygeist.submapInverse(%0, %16, %2, %5, %8, %11) {map = #map} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %18 = polygeist.submap(%1, %2, %12, %13, %14, %c2, %c2, %c2) {map = #map1} : (tensor<?x?x?x?xf32>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf32>
    %19 = polygeist.submap(%17, %2, %12, %13, %14, %c2, %c2, %c2) {map = #map2} : (tensor<?x?x?x?xf32>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf32>
    %20 = linalg.generic {doc = "", indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%18 : tensor<?x?x?x?x?x?x?xf32>) outs(%19 : tensor<?x?x?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %23 = arith.divf %in, %cst_0 : f32
      %24 = arith.addf %out, %23 : f32
      linalg.yield %24 : f32
    } -> tensor<?x?x?x?x?x?x?xf32>
    %21 = polygeist.submapInverse(%17, %20, %2, %12, %13, %14, %c2, %c2, %c2) {map = #map2} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?x?x?x?xf32>, index, index, index, index, index, index, index) -> tensor<?x?x?x?xf32>
    %22 = bufferization.to_memref %21 : memref<?x?x?x?xf32>
    memref.copy %22, %arg8 : memref<?x?x?x?xf32> to memref<?x?x?x?xf32>
    return
  }
}

