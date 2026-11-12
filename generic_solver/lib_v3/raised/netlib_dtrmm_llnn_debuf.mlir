#map = affine_map<(d0, d1)[s0] -> (-d1 + s0 - 1)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map2 = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1, d0 - s0 + s1)>
#map3 = affine_map<(d0)[s0, s1, s2] -> (s0, d0 - s1 + s2)>
#map4 = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dtrmm_llnn(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg6 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg4 : memref<?x?xf64>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = arith.subi %2, %c1 : index
    %5 = affine.for %arg7 = 0 to %3 iter_args(%arg8 = %0) -> (tensor<?x?xf64>) {
      %7 = affine.for %arg9 = 0 to %2 iter_args(%arg10 = %arg8) -> (tensor<?x?xf64>) {
        %8 = affine.apply #map(%arg7, %arg9)[%2]
        %extracted = tensor.extract %arg10[%arg7, %8] : tensor<?x?xf64>
        %9 = arith.mulf %arg2, %extracted : f64
        %10 = affine.apply #map(%arg7, %arg9)[%2]
        %inserted = tensor.insert %9 into %arg10[%arg7, %10] : tensor<?x?xf64>
        %11 = affine.apply #map1(%arg9)[%2]
        %12 = affine.apply #map1(%arg9)[%2]
        %extracted_0 = tensor.extract %1[%11, %12] : tensor<?x?xf64>
        %13 = arith.mulf %9, %extracted_0 : f64
        %14 = affine.apply #map(%arg7, %arg9)[%2]
        %inserted_1 = tensor.insert %13 into %inserted[%arg7, %14] : tensor<?x?xf64>
        %15 = polygeist.submap(%1, %arg9, %2, %4) {map = #map2} : (tensor<?x?xf64>, index, index, index) -> tensor<?xf64>
        %16 = polygeist.submap(%inserted_1, %arg7, %arg9, %2, %4) {map = #map3} : (tensor<?x?xf64>, index, index, index, index) -> tensor<?xf64>
        %17 = linalg.generic {doc = "", indexing_maps = [#map4, #map4], iterator_types = ["parallel"], library_call = ""} ins(%15 : tensor<?xf64>) outs(%16 : tensor<?xf64>) {
        ^bb0(%in: f64, %out: f64):
          %19 = arith.mulf %9, %in : f64
          %20 = arith.addf %out, %19 : f64
          %21 = linalg.index 0 : index
          %22 = arith.cmpi slt, %21, %arg9 : index
          %23 = arith.select %22, %20, %out : f64
          linalg.yield %23 : f64
        } -> tensor<?xf64>
        %18 = polygeist.submapInverse(%inserted_1, %17, %arg7, %arg9, %2, %4) {map = #map3} : (tensor<?x?xf64>, tensor<?xf64>, index, index, index, index) -> tensor<?x?xf64>
        affine.yield %18 : tensor<?x?xf64>
      }
      affine.yield %7 : tensor<?x?xf64>
    }
    %6 = bufferization.to_memref %5 : memref<?x?xf64>
    memref.copy %6, %arg6 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

