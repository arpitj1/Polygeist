#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_histogram_u16_u32(%arg0: i32, %arg1: i32, %arg2: memref<?x?xi16>, %arg3: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c256 = arith.constant 256 : index
    %0 = bufferization.to_tensor %arg3 : memref<?xi32>
    %1 = bufferization.to_tensor %arg2 : memref<?x?xi16>
    %2 = polygeist.submap(%0, %c256) {map = #map} : (tensor<?xi32>, index) -> tensor<?xi32>
    %3 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%2 : tensor<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    } -> tensor<?xi32>
    %4 = polygeist.submapInverse(%0, %3, %c256) {map = #map} : (tensor<?xi32>, tensor<?xi32>, index) -> tensor<?xi32>
    %5 = arith.index_cast %arg0 : i32 to index
    %6 = arith.index_cast %arg1 : i32 to index
    %7 = affine.for %arg4 = 0 to %5 iter_args(%arg5 = %4) -> (tensor<?xi32>) {
      %9 = affine.for %arg6 = 0 to %6 iter_args(%arg7 = %arg5) -> (tensor<?xi32>) {
        %extracted = tensor.extract %1[%arg4, %arg6] : tensor<?x?xi16>
        %10 = arith.extui %extracted : i16 to i32
        %11 = arith.shrsi %10, %c8_i32 : i32
        %12 = arith.index_castui %11 : i32 to index
        %extracted_0 = tensor.extract %arg7[%12] : tensor<?xi32>
        %13 = arith.addi %extracted_0, %c1_i32 : i32
        %inserted = tensor.insert %13 into %arg7[%12] : tensor<?xi32>
        affine.yield %inserted : tensor<?xi32>
      }
      affine.yield %9 : tensor<?xi32>
    }
    %8 = bufferization.to_memref %7 : memref<?xi32>
    memref.copy %8, %arg3 : memref<?xi32> to memref<?xi32>
    return
  }
}

