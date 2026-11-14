#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cusp_coo_matvecs_beta0_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?xi32>, %arg4: i32, %arg5: memref<?xi32>, %arg6: i32, %arg7: memref<?xf32>, %arg8: i32, %arg9: i32, %arg10: memref<?x?xf32>, %arg11: i32, %arg12: i32, %arg13: memref<?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg13 : memref<?x?xf32>
    %1 = bufferization.to_tensor %arg10 : memref<?x?xf32>
    %2 = bufferization.to_tensor %arg7 : memref<?xf32>
    %3 = bufferization.to_tensor %arg5 : memref<?xi32>
    %4 = bufferization.to_tensor %arg3 : memref<?xi32>
    %5 = arith.index_cast %arg0 : i32 to index
    %6 = arith.index_cast %arg9 : i32 to index
    %7 = polygeist.submap(%0, %5, %6) {map = #map} : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
    %8 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%7 : tensor<?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?x?xf32>
    %9 = polygeist.submapInverse(%0, %8, %5, %6) {map = #map} : (tensor<?x?xf32>, tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
    %10 = arith.index_cast %arg1 : i32 to index
    %11 = affine.for %arg14 = 0 to %10 iter_args(%arg15 = %9) -> (tensor<?x?xf32>) {
      %13 = affine.for %arg16 = 0 to %6 iter_args(%arg17 = %arg15) -> (tensor<?x?xf32>) {
        %extracted = tensor.extract %4[%arg14] : tensor<?xi32>
        %14 = arith.index_cast %extracted : i32 to index
        %extracted_0 = tensor.extract %2[%arg14] : tensor<?xf32>
        %extracted_1 = tensor.extract %3[%arg14] : tensor<?xi32>
        %15 = arith.index_cast %extracted_1 : i32 to index
        %extracted_2 = tensor.extract %1[%15, %arg16] : tensor<?x?xf32>
        %16 = arith.mulf %extracted_0, %extracted_2 : f32
        %extracted_3 = tensor.extract %arg17[%14, %arg16] : tensor<?x?xf32>
        %17 = arith.addf %extracted_3, %16 : f32
        %inserted = tensor.insert %17 into %arg17[%14, %arg16] : tensor<?x?xf32>
        affine.yield %inserted : tensor<?x?xf32>
      }
      affine.yield %13 : tensor<?x?xf32>
    }
    %12 = bufferization.to_memref %11 : memref<?x?xf32>
    memref.copy %12, %arg13 : memref<?x?xf32> to memref<?x?xf32>
    return
  }
}

