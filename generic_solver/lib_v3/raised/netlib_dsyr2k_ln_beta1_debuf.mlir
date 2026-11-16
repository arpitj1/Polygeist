#map = affine_map<(d0)[s0] -> (-d0 + s0)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d0)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2 + d0)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2 + d0)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dsyr2k_ln_beta1(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>, %arg7: i32, %arg8: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg8 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg6 : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg4 : memref<?x?xf64>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = affine.apply #map(%c0)[%3]
    %6 = polygeist.submap(%1, %3, %4, %5) {map = #map1} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %7 = polygeist.submap(%2, %3, %4, %5) {map = #map1} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %8 = polygeist.submap(%2, %3, %4, %5) {map = #map2} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %9 = polygeist.submap(%1, %3, %4, %5) {map = #map2} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %10 = polygeist.submap(%0, %3, %4, %5) {map = #map3} : (tensor<?x?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %11 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "reduction", "parallel"], library_call = ""} ins(%6, %7, %8, %9 : tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%10 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %out: f64):
      %14 = linalg.index 0 : index
      %15 = arith.mulf %arg2, %in : f64
      %16 = arith.mulf %arg2, %in_0 : f64
      %17 = arith.mulf %in_1, %15 : f64
      %18 = arith.addf %out, %17 : f64
      %19 = arith.mulf %in_2, %16 : f64
      %20 = arith.addf %18, %19 : f64
      %21 = linalg.index 2 : index
      %22 = affine.apply #map(%14)[%3]
      %23 = arith.cmpi slt, %21, %22 : index
      %24 = arith.select %23, %20, %out : f64
      linalg.yield %24 : f64
    } -> tensor<?x?x?xf64>
    %12 = polygeist.submapInverse(%0, %11, %3, %4, %5) {map = #map3} : (tensor<?x?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?x?xf64>
    %13 = bufferization.to_memref %12 : memref<?x?xf64>
    memref.copy %13, %arg8 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

