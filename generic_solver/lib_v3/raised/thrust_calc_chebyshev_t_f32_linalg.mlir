module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_chebyshev_t_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %false = arith.constant false
    %c0_i8 = arith.constant 0 : i8
    %c1_i64 = arith.constant 1 : i64
    %c6_i64 = arith.constant 6 : i64
    %cst = arith.constant -1.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    affine.for %arg4 = 0 to %0 {
      %2 = affine.load %arg1[%arg4] : memref<?xf32>
      %3 = affine.load %arg2[%arg4] : memref<?xf32>
      %4 = arith.fptosi %3 : f32 to i64
      %5 = arith.cmpi slt, %4, %c0_i64 : i64
      %6 = arith.cmpi sge, %4, %c0_i64 : i64
      %7 = arith.select %5, %cst_1, %1 : f32
      %8 = math.absf %2 : f32
      %9 = arith.cmpf oeq, %8, %cst_0 : f32
      %10 = arith.xori %9, %true : i1
      %11 = arith.andi %6, %9 : i1
      %12 = arith.andi %6, %10 : i1
      %13 = arith.cmpf ogt, %2, %cst_1 : f32
      %14 = arith.remsi %4, %c2_i64 : i64
      %15 = arith.cmpi eq, %14, %c0_i64 : i64
      %16 = arith.select %13, %true, %15 : i1
      %17 = arith.xori %16, %true : i1
      %18 = arith.select %17, %cst, %cst_0 : f32
      %19 = arith.select %11, %18, %7 : f32
      %20 = arith.cmpi sgt, %4, %c6_i64 : i64
      %21 = arith.andi %12, %20 : i1
      %22 = arith.cmpf olt, %8, %cst_0 : f32
      %23 = arith.andi %22, %12 : i1
      %24 = arith.xori %23, %true : i1
      %25 = arith.andi %24, %12 : i1
      %26 = arith.andi %21, %23 : i1
      %27 = arith.xori %21, %true : i1
      %28 = arith.andi %21, %25 : i1
      %29 = arith.andi %27, %12 : i1
      %30 = arith.ori %28, %29 : i1
      %31 = scf.if %26 -> (f32) {
        %41 = arith.sitofp %4 : i64 to f32
        %42 = func.call @acosf(%2) : (f32) -> f32
        %43 = arith.mulf %41, %42 : f32
        %44 = math.cos %43 : f32
        scf.yield %44 : f32
      } else {
        scf.yield %19 : f32
      }
      %32 = arith.cmpi eq, %4, %c0_i64 : i64
      %33 = arith.cmpi ne, %4, %c0_i64 : i64
      %34 = arith.select %32, %cst_0, %31 : f32
      %35 = arith.select %30, %33, %false : i1
      %36 = arith.select %30, %34, %31 : f32
      %37 = arith.cmpi eq, %4, %c1_i64 : i64
      %38 = arith.cmpi ne, %4, %c1_i64 : i64
      %39 = arith.select %37, %2, %36 : f32
      %40 = scf.if %35 -> (f32) {
        %41 = scf.if %38 -> (f32) {
          %42 = arith.addf %2, %2 : f32
          %43:4 = scf.while (%arg5 = %c2_i64, %arg6 = %2, %arg7 = %2, %arg8 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
            %44 = arith.cmpi sle, %arg5, %4 : i64
            %45 = arith.cmpf uno, %arg7, %arg7 : f32
            %46 = arith.extui %45 : i1 to i8
            %47 = arith.cmpi eq, %46, %c0_i8 : i8
            %48 = arith.andi %44, %47 : i1
            scf.condition(%48) %arg6, %arg5, %arg7, %arg8 : f32, i64, f32, f32
          } do {
          ^bb0(%arg5: f32, %arg6: i64, %arg7: f32, %arg8: f32):
            %44 = arith.cmpf uno, %arg7, %arg7 : f32
            %45 = arith.extui %44 : i1 to i8
            %46 = arith.cmpi eq, %45, %c0_i8 : i8
            %47 = arith.select %46, %arg7, %arg8 : f32
            %48 = arith.cmpi sle, %arg6, %4 : i64
            %49 = arith.select %48, %47, %arg8 : f32
            %50 = arith.mulf %42, %arg7 : f32
            %51 = arith.subf %50, %arg8 : f32
            %52 = arith.addi %arg6, %c1_i64 : i64
            scf.yield %52, %51, %51, %49 : i64, f32, f32, f32
          }
          scf.yield %43#0 : f32
        } else {
          scf.yield %39 : f32
        }
        scf.yield %41 : f32
      } else {
        scf.yield %36 : f32
      }
      affine.store %40, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
  func.func private @acosf(f32) -> f32
}

