module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
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
    affine.for %arg3 = 0 to %0 {
      %1 = affine.load %arg1[%arg3] : memref<?xf32>
      %2 = arith.cmpf olt, %1, %cst_2 : f32
      %3 = arith.select %2, %cst_3, %cst_0 : f32
      %4:2 = scf.if %2 -> (f32, f32) {
        %20 = arith.mulf %1, %cst : f32
        %21 = math.sin %20 : f32
        %22 = arith.mulf %21, %21 : f32
        %23 = arith.divf %cst_8, %22 : f32
        %24 = arith.subf %cst_1, %23 : f32
        %25 = arith.subf %cst_0, %1 : f32
        scf.yield %24, %25 : f32, f32
      } else {
        scf.yield %cst_1, %1 : f32, f32
      }
      %5:2 = affine.for %arg4 = 0 to 6 iter_args(%arg5 = %4#0, %arg6 = %4#1) -> (f32, f32) {
        %20 = arith.mulf %arg6, %arg6 : f32
        %21 = arith.divf %cst_0, %20 : f32
        %22 = arith.addf %arg5, %21 : f32
        %23 = arith.addf %arg6, %cst_0 : f32
        affine.yield %22, %23 : f32, f32
      }
      %6 = arith.mulf %5#1, %5#1 : f32
      %7 = arith.divf %cst_0, %6 : f32
      %8 = arith.mulf %5#1, %cst_4 : f32
      %9 = arith.divf %cst_0, %8 : f32
      %10 = arith.addf %9, %cst_0 : f32
      %11 = arith.mulf %7, %cst_5 : f32
      %12 = arith.subf %cst_6, %11 : f32
      %13 = arith.mulf %7, %12 : f32
      %14 = arith.subf %cst_7, %13 : f32
      %15 = arith.mulf %7, %14 : f32
      %16 = arith.addf %10, %15 : f32
      %17 = arith.divf %16, %5#1 : f32
      %18 = arith.addf %5#0, %17 : f32
      %19 = arith.mulf %3, %18 : f32
      affine.store %19, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
}

