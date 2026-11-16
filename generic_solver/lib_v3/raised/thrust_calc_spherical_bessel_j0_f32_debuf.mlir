#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_spherical_bessel_j0_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0x7F800000 : f32
    %cst_0 = arith.constant 1.60590444E-10 : f32
    %cst_1 = arith.constant -2.50521079E-8 : f32
    %cst_2 = arith.constant 2.75573188E-6 : f32
    %cst_3 = arith.constant -1.98412701E-4 : f32
    %cst_4 = arith.constant 0.00833333377 : f32
    %cst_5 = arith.constant -0.166666672 : f32
    %cst_6 = arith.constant 1.000000e+00 : f32
    %cst_7 = arith.constant 5.000000e-01 : f32
    %cst_8 = arith.constant 0.000000e+00 : f32
    %c0_i8 = arith.constant 0 : i8
    %0 = bufferization.to_tensor %arg2 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = polygeist.submap(%1, %2) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %4 = polygeist.submap(%0, %2) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %5 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel"], library_call = ""} ins(%3 : tensor<?xf32>) outs(%4 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = math.absf %in : f32
      %9 = arith.cmpf oeq, %8, %cst : f32
      %10 = arith.extui %9 : i1 to i8
      %11 = arith.cmpi eq, %10, %c0_i8 : i8
      %12 = arith.cmpf olt, %8, %cst_7 : f32
      %13 = arith.mulf %in, %in : f32
      %14 = arith.mulf %13, %cst_0 : f32
      %15 = arith.addf %14, %cst_1 : f32
      %16 = arith.mulf %13, %15 : f32
      %17 = arith.addf %16, %cst_2 : f32
      %18 = arith.mulf %13, %17 : f32
      %19 = arith.addf %18, %cst_3 : f32
      %20 = arith.mulf %13, %19 : f32
      %21 = arith.addf %20, %cst_4 : f32
      %22 = arith.mulf %13, %21 : f32
      %23 = arith.addf %22, %cst_5 : f32
      %24 = arith.mulf %13, %23 : f32
      %25 = arith.addf %24, %cst_6 : f32
      %26 = math.sin %in : f32
      %27 = arith.divf %26, %in : f32
      %28 = arith.select %12, %25, %27 : f32
      %29 = arith.select %11, %28, %cst_8 : f32
      linalg.yield %29 : f32
    } -> tensor<?xf32>
    %6 = polygeist.submapInverse(%0, %5, %2) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
    %7 = bufferization.to_memref %6 : memref<?xf32>
    memref.copy %7, %arg2 : memref<?xf32> to memref<?xf32>
    return
  }
}

