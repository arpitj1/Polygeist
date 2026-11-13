#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_histogram_u8_u32(%arg0: i32, %arg1: i32, %arg2: memref<?x?xi8>, %arg3: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c256 = arith.constant 256 : index
    %0 = bufferization.to_tensor %arg3 : memref<?xi32>
    %1 = bufferization.to_tensor %arg2 : memref<?x?xi8>
    %2 = tensor.empty() : tensor<256xi32>
    %3 = polygeist.submap(%2, %c256) {map = #map} : (tensor<256xi32>, index) -> tensor<?xi32>
    %4 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%3 : tensor<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    } -> tensor<?xi32>
    %5 = polygeist.submapInverse(%2, %4, %c256) {map = #map} : (tensor<256xi32>, tensor<?xi32>, index) -> tensor<256xi32>
    %6 = arith.index_cast %arg0 : i32 to index
    %7 = arith.index_cast %arg1 : i32 to index
    %8 = affine.for %arg4 = 0 to %6 iter_args(%arg5 = %5) -> (tensor<256xi32>) {
      %14 = affine.for %arg6 = 0 to %7 iter_args(%arg7 = %arg5) -> (tensor<256xi32>) {
        %extracted = tensor.extract %1[%arg4, %arg6] : tensor<?x?xi8>
        %15 = arith.extui %extracted : i8 to i32
        %16 = arith.index_cast %15 : i32 to index
        %extracted_0 = tensor.extract %arg7[%16] : tensor<256xi32>
        %17 = arith.addi %extracted_0, %c1_i32 : i32
        %inserted = tensor.insert %17 into %arg7[%16] : tensor<256xi32>
        affine.yield %inserted : tensor<256xi32>
      }
      affine.yield %14 : tensor<256xi32>
    }
    %9 = polygeist.submap(%8, %c256) {map = #map} : (tensor<256xi32>, index) -> tensor<?xi32>
    %10 = polygeist.submap(%0, %c256) {map = #map} : (tensor<?xi32>, index) -> tensor<?xi32>
    %11 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%9 : tensor<?xi32>) outs(%10 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<?xi32>
    %12 = polygeist.submapInverse(%0, %11, %c256) {map = #map} : (tensor<?xi32>, tensor<?xi32>, index) -> tensor<?xi32>
    %13 = bufferization.to_memref %12 : memref<?xi32>
    memref.copy %13, %arg3 : memref<?xi32> to memref<?xi32>
    return
  }
}

