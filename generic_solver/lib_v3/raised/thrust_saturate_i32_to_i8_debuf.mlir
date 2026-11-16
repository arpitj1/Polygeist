#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_saturate_i32_to_i8(%arg0: i32, %arg1: memref<?xi32>, %arg2: memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c127_i32 = arith.constant 127 : i32
    %c-128_i32 = arith.constant -128 : i32
    %false = arith.constant false
    %0 = bufferization.to_tensor %arg2 : memref<?xi8>
    %1 = bufferization.to_tensor %arg1 : memref<?xi32>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = polygeist.submap(%1, %2) {map = #map} : (tensor<?xi32>, index) -> tensor<?xi32>
    %4 = polygeist.submap(%0, %2) {map = #map} : (tensor<?xi8>, index) -> tensor<?xi8>
    %5 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%3 : tensor<?xi32>) outs(%4 : tensor<?xi8>) {
    ^bb0(%in: i32, %out: i8):
      %8 = arith.cmpi slt, %in, %c-128_i32 : i32
      %9 = arith.select %8, %c-128_i32, %in : i32
      %10 = arith.cmpi sgt, %in, %c127_i32 : i32
      %11 = arith.select %8, %false, %10 : i1
      %12 = arith.select %11, %c127_i32, %9 : i32
      %13 = arith.trunci %12 : i32 to i8
      linalg.yield %13 : i8
    } -> tensor<?xi8>
    %6 = polygeist.submapInverse(%0, %5, %2) {map = #map} : (tensor<?xi8>, tensor<?xi8>, index) -> tensor<?xi8>
    %7 = bufferization.to_memref %6 : memref<?xi8>
    memref.copy %7, %arg2 : memref<?xi8> to memref<?xi8>
    return
  }
}

