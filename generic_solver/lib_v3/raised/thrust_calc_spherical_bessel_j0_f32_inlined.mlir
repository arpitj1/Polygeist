module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_spherical_bessel_j0_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i8 = arith.constant 0 : i8
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant -0.166666672 : f32
    %cst_3 = arith.constant 0.00833333377 : f32
    %cst_4 = arith.constant -1.98412701E-4 : f32
    %cst_5 = arith.constant 2.75573188E-6 : f32
    %cst_6 = arith.constant -2.50521079E-8 : f32
    %cst_7 = arith.constant 1.60590444E-10 : f32
    %cst_8 = arith.constant 0x7F800000 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    affine.for %arg3 = 0 to %0 {
      %2 = affine.load %arg1[%arg3] : memref<?xf32>
      %3 = math.absf %2 : f32
      %4 = arith.cmpf oeq, %3, %cst_8 : f32
      %5 = arith.extui %4 : i1 to i8
      %6 = arith.cmpi eq, %5, %c0_i8 : i8
      %7 = scf.if %6 -> (f32) {
        %8 = arith.cmpf olt, %3, %cst_0 : f32
        %9 = scf.if %8 -> (f32) {
          %10 = arith.mulf %2, %2 : f32
          %11 = arith.mulf %10, %cst_7 : f32
          %12 = arith.addf %11, %cst_6 : f32
          %13 = arith.mulf %10, %12 : f32
          %14 = arith.addf %13, %cst_5 : f32
          %15 = arith.mulf %10, %14 : f32
          %16 = arith.addf %15, %cst_4 : f32
          %17 = arith.mulf %10, %16 : f32
          %18 = arith.addf %17, %cst_3 : f32
          %19 = arith.mulf %10, %18 : f32
          %20 = arith.addf %19, %cst_2 : f32
          %21 = arith.mulf %10, %20 : f32
          %22 = arith.addf %21, %cst_1 : f32
          scf.yield %22 : f32
        } else {
          %10 = math.sin %2 : f32
          %11 = arith.divf %10, %2 : f32
          scf.yield %11 : f32
        }
        scf.yield %9 : f32
      } else {
        %8 = arith.select %4, %cst, %1 : f32
        scf.yield %8 : f32
      }
      affine.store %7, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
}

