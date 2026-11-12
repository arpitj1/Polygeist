#map = affine_map<(d0, d1)[s0] -> (-d1 + s0 - 1)>
#map1 = affine_map<(d0)[s0, s1] -> (-s0 + s1 - 1, d0 - s0 + s1)>
#map2 = affine_map<(d0)[s0, s1, s2] -> (s0, d0 - s1 + s2)>
#map3 = affine_map<(d0)[s0] -> (s0)>
#map4 = affine_map<(d0) -> (d0)>
#map5 = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @netlib_dsymm_ll(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: i32, %arg4: memref<?x?xf64>, %arg5: i32, %arg6: memref<?x?xf64>, %arg7: f64, %arg8: i32, %arg9: memref<?x?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c1 = arith.constant 1 : index
    %0 = bufferization.to_tensor %arg9 : memref<?x?xf64>
    %1 = bufferization.to_tensor %arg6 : memref<?x?xf64>
    %2 = bufferization.to_tensor %arg4 : memref<?x?xf64>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = arith.subi %3, %c1 : index
    %6 = affine.for %arg10 = 0 to %4 iter_args(%arg11 = %0) -> (tensor<?x?xf64>) {
      %alloca = memref.alloca(%3) : memref<?xf64>
      %8 = bufferization.to_tensor %alloca : memref<?xf64>
      %9:2 = affine.for %arg12 = 0 to %3 iter_args(%arg13 = %8, %arg14 = %arg11) -> (tensor<?xf64>, tensor<?x?xf64>) {
        %10 = affine.apply #map(%arg10, %arg12)[%3]
        %extracted = tensor.extract %1[%arg10, %10] : tensor<?x?xf64>
        %11 = arith.mulf %arg2, %extracted : f64
        %inserted = tensor.insert %cst into %arg13[%arg12] : tensor<?xf64>
        %12 = polygeist.submap(%2, %arg12, %3, %5) {map = #map1} : (tensor<?x?xf64>, index, index, index) -> tensor<?xf64>
        %13 = polygeist.submap(%1, %arg10, %arg12, %3, %5) {map = #map2} : (tensor<?x?xf64>, index, index, index, index) -> tensor<?xf64>
        %14 = polygeist.submap(%2, %arg12, %3, %5) {map = #map1} : (tensor<?x?xf64>, index, index, index) -> tensor<?xf64>
        %15 = polygeist.submap(%arg14, %arg10, %arg12, %3, %5) {map = #map2} : (tensor<?x?xf64>, index, index, index, index) -> tensor<?xf64>
        %16 = polygeist.submap(%inserted, %arg12, %5) {map = #map3} : (tensor<?xf64>, index, index) -> tensor<?xf64>
        %17:2 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4, #map4, #map4], iterator_types = ["reduction"], library_call = ""} ins(%12, %13, %14 : tensor<?xf64>, tensor<?xf64>, tensor<?xf64>) outs(%15, %16 : tensor<?xf64>, tensor<?xf64>) {
        ^bb0(%in: f64, %in_4: f64, %in_5: f64, %out: f64, %out_6: f64):
          %29 = arith.mulf %11, %in : f64
          %30 = arith.addf %out, %29 : f64
          %31 = arith.mulf %in_4, %in_5 : f64
          %32 = arith.addf %out_6, %31 : f64
          %33 = linalg.index 0 : index
          %34 = arith.cmpi slt, %33, %arg12 : index
          %35 = arith.select %34, %30, %out : f64
          %36 = arith.select %34, %32, %out_6 : f64
          linalg.yield %35, %36 : f64, f64
        } -> (tensor<?xf64>, tensor<?xf64>)
        %18 = polygeist.submapInverse(%arg14, %17#0, %arg10, %arg12, %3, %5) {map = #map2} : (tensor<?x?xf64>, tensor<?xf64>, index, index, index, index) -> tensor<?x?xf64>
        %19 = polygeist.submapInverse(%inserted, %17#1, %arg12, %5) {map = #map3} : (tensor<?xf64>, tensor<?xf64>, index, index) -> tensor<?xf64>
        %extracted_0 = tensor.extract %19[%arg12] : tensor<?xf64>
        %20 = affine.apply #map(%arg10, %arg12)[%3]
        %extracted_1 = tensor.extract %18[%arg10, %20] : tensor<?x?xf64>
        %21 = arith.mulf %arg7, %extracted_1 : f64
        %22 = affine.apply #map5(%arg12)[%3]
        %23 = affine.apply #map5(%arg12)[%3]
        %extracted_2 = tensor.extract %2[%22, %23] : tensor<?x?xf64>
        %24 = arith.mulf %11, %extracted_2 : f64
        %25 = arith.addf %21, %24 : f64
        %26 = arith.mulf %arg2, %extracted_0 : f64
        %27 = arith.addf %25, %26 : f64
        %28 = affine.apply #map(%arg10, %arg12)[%3]
        %inserted_3 = tensor.insert %27 into %18[%arg10, %28] : tensor<?x?xf64>
        affine.yield %19, %inserted_3 : tensor<?xf64>, tensor<?x?xf64>
      }
      affine.yield %9#1 : tensor<?x?xf64>
    }
    %7 = bufferization.to_memref %6 : memref<?x?xf64>
    memref.copy %7, %arg9 : memref<?x?xf64> to memref<?x?xf64>
    return
  }
}

