#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#map2 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_equalize_integer_u8(%arg0: i32, %arg1: i32, %arg2: memref<?xi8>, %arg3: memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c255_i32 = arith.constant 255 : i32
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg3 : memref<?xi8>
    %1 = bufferization.to_tensor %arg2 : memref<?xi8>
    %2 = tensor.empty() : tensor<256xi32>
    %3 = tensor.empty() : tensor<256xi32>
    %4 = polygeist.submap(%3, %c256) {map = #map} : (tensor<256xi32>, index) -> tensor<?xi32>
    %5 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%4 : tensor<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    } -> tensor<?xi32>
    %6 = polygeist.submapInverse(%3, %5, %c256) {map = #map} : (tensor<256xi32>, tensor<?xi32>, index) -> tensor<256xi32>
    %7 = arith.muli %arg0, %arg1 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = affine.for %arg4 = 0 to %8 iter_args(%arg5 = %6) -> (tensor<256xi32>) {
      %extracted_2 = tensor.extract %1[%arg4] : tensor<?xi8>
      %21 = arith.index_castui %extracted_2 : i8 to index
      %extracted_3 = tensor.extract %arg5[%21] : tensor<256xi32>
      %22 = arith.addi %extracted_3, %c1_i32 : i32
      %inserted_4 = tensor.insert %22 into %arg5[%21] : tensor<256xi32>
      affine.yield %inserted_4 : tensor<256xi32>
    }
    %extracted = tensor.extract %9[%c0] : tensor<256xi32>
    %inserted = tensor.insert %extracted into %2[%c0] : tensor<256xi32>
    %10 = affine.for %arg4 = 0 to 255 iter_args(%arg5 = %inserted) -> (tensor<256xi32>) {
      %extracted_2 = tensor.extract %arg5[%arg4] : tensor<256xi32>
      %21 = affine.apply #map1(%arg4)
      %extracted_3 = tensor.extract %9[%21] : tensor<256xi32>
      %22 = arith.addi %extracted_2, %extracted_3 : i32
      %23 = affine.apply #map1(%arg4)
      %inserted_4 = tensor.insert %22 into %arg5[%23] : tensor<256xi32>
      affine.yield %inserted_4 : tensor<256xi32>
    }
    %11 = tensor.empty() : tensor<i32>
    %inserted_0 = tensor.insert %c0_i32 into %11[] : tensor<i32>
    %12 = polygeist.submap(%10, %c256) {map = #map} : (tensor<256xi32>, index) -> tensor<?xi32>
    %13 = linalg.generic {doc = "", indexing_maps = [#map, #map2], iterator_types = ["reduction"], library_call = ""} ins(%12 : tensor<?xi32>) outs(%inserted_0 : tensor<i32>) {
    ^bb0(%in: i32, %out: i32):
      %21 = arith.cmpi eq, %out, %c0_i32 : i32
      %22 = arith.cmpi ne, %in, %c0_i32 : i32
      %23 = arith.select %22, %in, %out : i32
      %24 = arith.select %21, %23, %out : i32
      linalg.yield %24 : i32
    } -> tensor<i32>
    %extracted_1 = tensor.extract %13[] : tensor<i32>
    %14 = arith.cmpi sgt, %7, %extracted_1 : i32
    %15 = arith.subi %7, %extracted_1 : i32
    %16 = arith.select %14, %15, %c1_i32 : i32
    %17 = polygeist.submap(%0, %8) {map = #map} : (tensor<?xi8>, index) -> tensor<?xi8>
    %18 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%17 : tensor<?xi8>) {
    ^bb0(%out: i8):
      %21 = linalg.index 0 : index
      %extracted_2 = tensor.extract %1[%21] : tensor<?xi8>
      %22 = arith.index_castui %extracted_2 : i8 to index
      %extracted_3 = tensor.extract %10[%22] : tensor<256xi32>
      %23 = arith.subi %extracted_3, %extracted_1 : i32
      %24 = arith.muli %23, %c255_i32 : i32
      %25 = arith.divsi %24, %16 : i32
      %26 = arith.cmpi slt, %25, %c0_i32 : i32
      %27 = arith.cmpi sgt, %25, %c255_i32 : i32
      %28 = arith.select %27, %c255_i32, %25 : i32
      %29 = arith.select %26, %c0_i32, %28 : i32
      %30 = arith.trunci %29 : i32 to i8
      linalg.yield %30 : i8
    } -> tensor<?xi8>
    %19 = polygeist.submapInverse(%0, %18, %8) {map = #map} : (tensor<?xi8>, tensor<?xi8>, index) -> tensor<?xi8>
    %20 = bufferization.to_memref %19 : memref<?xi8>
    memref.copy %20, %arg3 : memref<?xi8> to memref<?xi8>
    return
  }
}

