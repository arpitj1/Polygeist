#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_trigamma_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 9.86960506 : f32
    %cst_0 = arith.constant 0.166666672 : f32
    %cst_1 = arith.constant 0.0333333351 : f32
    %cst_2 = arith.constant 0.0238095243 : f32
    %cst_3 = arith.constant 2.000000e+00 : f32
    %cst_4 = arith.constant -1.000000e+00 : f32
    %cst_5 = arith.constant 5.000000e-01 : f32
    %cst_6 = arith.constant 0.000000e+00 : f32
    %cst_7 = arith.constant 1.000000e+00 : f32
    %cst_8 = arith.constant 3.14159274 : f32
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = polygeist.submap(%1, %2) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %4 = polygeist.submap(%0, %2) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %5 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%3 : tensor<?xf32>) outs(%4 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.cmpf olt, %in, %cst_5 : f32
      %9 = arith.select %8, %cst_4, %cst_7 : f32
      %10 = arith.mulf %in, %cst_8 : f32
      %11 = math.sin %10 : f32
      %12 = arith.mulf %11, %11 : f32
      %13 = arith.divf %cst, %12 : f32
      %14 = arith.subf %cst_6, %13 : f32
      %15 = arith.subf %cst_7, %in : f32
      %16 = arith.select %8, %14, %cst_6 : f32
      %17 = arith.select %8, %15, %in : f32
      %18 = arith.mulf %17, %17 : f32
      %19 = arith.divf %cst_7, %18 : f32
      %20 = arith.addf %16, %19 : f32
      %21 = arith.addf %17, %cst_7 : f32
      %22 = arith.mulf %21, %21 : f32
      %23 = arith.divf %cst_7, %22 : f32
      %24 = arith.addf %20, %23 : f32
      %25 = arith.addf %21, %cst_7 : f32
      %26 = arith.mulf %25, %25 : f32
      %27 = arith.divf %cst_7, %26 : f32
      %28 = arith.addf %24, %27 : f32
      %29 = arith.addf %25, %cst_7 : f32
      %30 = arith.mulf %29, %29 : f32
      %31 = arith.divf %cst_7, %30 : f32
      %32 = arith.addf %28, %31 : f32
      %33 = arith.addf %29, %cst_7 : f32
      %34 = arith.mulf %33, %33 : f32
      %35 = arith.divf %cst_7, %34 : f32
      %36 = arith.addf %32, %35 : f32
      %37 = arith.addf %33, %cst_7 : f32
      %38 = arith.mulf %37, %37 : f32
      %39 = arith.divf %cst_7, %38 : f32
      %40 = arith.addf %36, %39 : f32
      %41 = arith.addf %37, %cst_7 : f32
      %42 = arith.mulf %41, %41 : f32
      %43 = arith.divf %cst_7, %42 : f32
      %44 = arith.mulf %41, %cst_3 : f32
      %45 = arith.divf %cst_7, %44 : f32
      %46 = arith.addf %45, %cst_7 : f32
      %47 = arith.mulf %43, %cst_2 : f32
      %48 = arith.subf %cst_1, %47 : f32
      %49 = arith.mulf %43, %48 : f32
      %50 = arith.subf %cst_0, %49 : f32
      %51 = arith.mulf %43, %50 : f32
      %52 = arith.addf %46, %51 : f32
      %53 = arith.divf %52, %41 : f32
      %54 = arith.addf %40, %53 : f32
      %55 = arith.mulf %9, %54 : f32
      linalg.yield %55 : f32
    } -> tensor<?xf32>
    %6 = polygeist.submapInverse(%0, %5, %2) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
    %7 = bufferization.to_memref %6 : memref<?xf32>
    memref.copy %7, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

