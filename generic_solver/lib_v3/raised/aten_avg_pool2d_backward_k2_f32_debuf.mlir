#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3 + d1 * 2, d4 + d2 * 2)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_avg_pool2d_backward_k2_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x?x?xf32>, %arg6: memref<?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 4.000000e+00 : f32
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg6 : memref<?x?x?xf32>
    %1 = bufferization.to_tensor %arg5 : memref<?x?x?xf32>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = arith.muli %arg1, %c2_i32 : i32
    %4 = arith.addi %3, %arg3 : i32
    %5 = arith.index_cast %4 : i32 to index
    %6 = arith.muli %arg2, %c2_i32 : i32
    %7 = arith.addi %6, %arg4 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = arith.index_cast %arg1 : i32 to index
    %10 = arith.index_cast %arg2 : i32 to index
    %11 = polygeist.submap(%0, %2, %5, %8) {map = #map} : (tensor<?x?x?xf32>, index, index, index) -> tensor<?x?x?xf32>
    %12 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%11 : tensor<?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?x?x?xf32>
    %13 = polygeist.submapInverse(%0, %12, %2, %5, %8) {map = #map} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, index, index, index) -> tensor<?x?x?xf32>
    %14 = polygeist.submap(%1, %2, %9, %10, %c2, %c2) {map = #map1} : (tensor<?x?x?xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %15 = polygeist.submap(%13, %2, %9, %10, %c2, %c2) {map = #map2} : (tensor<?x?x?xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %16 = linalg.generic {doc = "", indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%14 : tensor<?x?x?x?x?xf32>) outs(%15 : tensor<?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %19 = arith.divf %in, %cst_0 : f32
      %20 = arith.addf %out, %19 : f32
      linalg.yield %20 : f32
    } -> tensor<?x?x?x?x?xf32>
    %17 = polygeist.submapInverse(%13, %16, %2, %9, %10, %c2, %c2) {map = #map2} : (tensor<?x?x?xf32>, tensor<?x?x?x?x?xf32>, index, index, index, index, index) -> tensor<?x?x?xf32>
    %18 = bufferization.to_memref %17 : memref<?x?x?xf32>
    memref.copy %18, %arg6 : memref<?x?x?xf32> to memref<?x?x?xf32>
    return
  }
}

