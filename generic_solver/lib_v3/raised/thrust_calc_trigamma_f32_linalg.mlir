#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_trigamma_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.14159274 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 5.000000e-01 : f32
    %cst_3 = arith.constant -1.000000e+00 : f32
    %cst_4 = arith.constant 2.000000e+00 : f32
    %cst_5 = arith.constant 0.0238095243 : f32
    %cst_6 = arith.constant 0.0333333351 : f32
    %cst_7 = arith.constant 0.166666672 : f32
    %cst_8 = arith.constant 9.86960506 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = polygeist.submap(%arg1, %0) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
    %2 = polygeist.submap(%arg2, %0) {map = #map} : (memref<?xf32>, index) -> memref<?xf32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%1 : memref<?xf32>) outs(%2 : memref<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.cmpf olt, %in, %cst_2 : f32
      %4 = arith.select %3, %cst_3, %cst_0 : f32
      %5 = arith.mulf %in, %cst : f32
      %6 = math.sin %5 : f32
      %7 = arith.mulf %6, %6 : f32
      %8 = arith.divf %cst_8, %7 : f32
      %9 = arith.subf %cst_1, %8 : f32
      %10 = arith.subf %cst_0, %in : f32
      %11 = arith.select %3, %9, %cst_1 : f32
      %12 = arith.select %3, %10, %in : f32
      %13 = arith.mulf %12, %12 : f32
      %14 = arith.divf %cst_0, %13 : f32
      %15 = arith.addf %11, %14 : f32
      %16 = arith.addf %12, %cst_0 : f32
      %17 = arith.mulf %16, %16 : f32
      %18 = arith.divf %cst_0, %17 : f32
      %19 = arith.addf %15, %18 : f32
      %20 = arith.addf %16, %cst_0 : f32
      %21 = arith.mulf %20, %20 : f32
      %22 = arith.divf %cst_0, %21 : f32
      %23 = arith.addf %19, %22 : f32
      %24 = arith.addf %20, %cst_0 : f32
      %25 = arith.mulf %24, %24 : f32
      %26 = arith.divf %cst_0, %25 : f32
      %27 = arith.addf %23, %26 : f32
      %28 = arith.addf %24, %cst_0 : f32
      %29 = arith.mulf %28, %28 : f32
      %30 = arith.divf %cst_0, %29 : f32
      %31 = arith.addf %27, %30 : f32
      %32 = arith.addf %28, %cst_0 : f32
      %33 = arith.mulf %32, %32 : f32
      %34 = arith.divf %cst_0, %33 : f32
      %35 = arith.addf %31, %34 : f32
      %36 = arith.addf %32, %cst_0 : f32
      %37 = arith.mulf %36, %36 : f32
      %38 = arith.divf %cst_0, %37 : f32
      %39 = arith.mulf %36, %cst_4 : f32
      %40 = arith.divf %cst_0, %39 : f32
      %41 = arith.addf %40, %cst_0 : f32
      %42 = arith.mulf %38, %cst_5 : f32
      %43 = arith.subf %cst_6, %42 : f32
      %44 = arith.mulf %38, %43 : f32
      %45 = arith.subf %cst_7, %44 : f32
      %46 = arith.mulf %38, %45 : f32
      %47 = arith.addf %41, %46 : f32
      %48 = arith.divf %47, %36 : f32
      %49 = arith.addf %35, %48 : f32
      %50 = arith.mulf %4, %49 : f32
      linalg.yield %50 : f32
    }
    return
  }
}

