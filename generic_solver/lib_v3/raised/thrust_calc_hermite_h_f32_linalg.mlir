module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_hermite_h_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0x7F800000 : f32
    %c-1_i64 = arith.constant -1 : i64
    %c128_i64 = arith.constant 128 : i64
    %cst_0 = arith.constant 0x7FC00000 : f32
    %c2 = arith.constant 2 : index
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %true = arith.constant true
    %false = arith.constant false
    %c1_i64 = arith.constant 1 : i64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    affine.for %arg4 = 0 to %0 {
      %2 = affine.load %arg1[%arg4] : memref<?xf32>
      %3 = affine.load %arg2[%arg4] : memref<?xf32>
      %4 = math.absf %3 : f32
      %5 = arith.cmpf oeq, %4, %cst : f32
      %6 = arith.cmpf uno, %3, %3 : f32
      %7 = arith.fptosi %3 : f32 to i64
      %8 = arith.cmpi slt, %7, %c0_i64 : i64
      %9 = arith.cmpi sge, %7, %c0_i64 : i64
      %10 = arith.cmpi eq, %7, %c0_i64 : i64
      %11 = arith.cmpi ne, %7, %c0_i64 : i64
      %12 = arith.cmpi eq, %7, %c1_i64 : i64
      %13 = arith.cmpi sgt, %7, %c128_i64 : i64
      %14 = arith.cmpi sle, %7, %c128_i64 : i64
      %15 = arith.select %6, %c-1_i64, %7 : i64
      %16 = arith.select %6, %true, %8 : i1
      %17 = arith.select %6, %false, %9 : i1
      %18 = arith.select %6, %false, %10 : i1
      %19 = arith.select %6, %true, %11 : i1
      %20 = arith.select %6, %false, %12 : i1
      %21 = arith.select %6, %false, %13 : i1
      %22 = arith.select %6, %true, %14 : i1
      %23 = arith.select %5, %c-1_i64, %15 : i64
      %24 = arith.select %5, %true, %16 : i1
      %25 = arith.select %5, %false, %17 : i1
      %26 = arith.select %5, %false, %18 : i1
      %27 = arith.select %5, %true, %19 : i1
      %28 = arith.select %5, %false, %20 : i1
      %29 = arith.select %5, %false, %21 : i1
      %30 = arith.select %5, %true, %22 : i1
      %31 = arith.index_cast %23 : i64 to index
      %32 = arith.select %24, %cst_1, %1 : f32
      %33 = arith.andi %25, %27 : i1
      %34 = arith.select %26, %cst_2, %32 : f32
      %35 = arith.select %25, %34, %32 : f32
      %36 = arith.andi %33, %28 : i1
      %37 = arith.xori %36, %true : i1
      %38 = arith.andi %37, %33 : i1
      %39 = arith.addf %2, %2 : f32
      %40 = arith.select %36, %39, %35 : f32
      %41 = scf.if %38 -> (f32) {
        %42 = scf.if %30 -> (f32) {
          %43 = arith.addf %2, %2 : f32
          %44 = arith.muli %31, %c2 : index
          %45:3 = scf.for %arg5 = %c2 to %44 step %c2 iter_args(%arg6 = %43, %arg7 = %cst_2, %arg8 = %cst_1) -> (f32, f32, f32) {
            %46 = arith.index_cast %arg5 : index to i64
            %47 = arith.mulf %43, %arg6 : f32
            %48 = arith.sitofp %46 : i64 to f32
            %49 = arith.mulf %48, %arg7 : f32
            %50 = arith.subf %47, %49 : f32
            scf.yield %50, %arg6, %50 : f32, f32, f32
          }
          scf.yield %45#2 : f32
        } else {
          %43 = arith.select %29, %cst_0, %40 : f32
          scf.yield %43 : f32
        }
        scf.yield %42 : f32
      } else {
        scf.yield %40 : f32
      }
      affine.store %41, %arg3[%arg4] : memref<?xf32>
    } {polygeist.was_parallel}
    return
  }
}

