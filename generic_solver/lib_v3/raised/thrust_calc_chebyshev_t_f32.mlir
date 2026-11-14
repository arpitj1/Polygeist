module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
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
      %2 = memref.load %arg1[%arg4] : memref<?xf32>
      %3 = memref.load %arg2[%arg4] : memref<?xf32>
      %4 = arith.fptosi %3 : f32 to i64
      %5 = arith.cmpi slt, %4, %c0_i64 : i64
      %6 = arith.cmpi sge, %4, %c0_i64 : i64
      %7 = arith.select %5, %cst_1, %1 : f32
      %8 = math.absf %2 : f32
      %9 = arith.cmpf oeq, %8, %cst_0 : f32
      %10 = arith.xori %9, %true : i1
      %11 = arith.andi %6, %9 : i1
      %12 = arith.andi %6, %10 : i1
      %13 = scf.if %11 -> (f32) {
        %31 = arith.cmpf ogt, %2, %cst_1 : f32
        %32 = scf.if %31 -> (i1) {
          scf.yield %true : i1
        } else {
          %35 = arith.remsi %4, %c2_i64 : i64
          %36 = arith.cmpi eq, %35, %c0_i64 : i64
          scf.yield %36 : i1
        }
        %33 = arith.xori %32, %true : i1
        %34 = arith.select %33, %cst, %cst_0 : f32
        scf.yield %34 : f32
      } else {
        scf.yield %7 : f32
      }
      %14 = arith.cmpi sgt, %4, %c6_i64 : i64
      %15 = arith.andi %12, %14 : i1
      %16 = arith.cmpf olt, %8, %cst_0 : f32
      %17 = arith.andi %16, %12 : i1
      %18 = arith.xori %17, %true : i1
      %19 = arith.andi %18, %12 : i1
      %20 = arith.andi %15, %17 : i1
      %21 = arith.xori %15, %true : i1
      %22 = arith.andi %15, %19 : i1
      %23 = arith.andi %21, %12 : i1
      %24 = arith.ori %22, %23 : i1
      %25 = scf.if %20 -> (f32) {
        %31 = arith.sitofp %4 : i64 to f32
        %32 = func.call @acosf(%2) : (f32) -> f32
        %33 = arith.mulf %31, %32 : f32
        %34 = math.cos %33 : f32
        scf.yield %34 : f32
      } else {
        scf.yield %13 : f32
      }
      %26:2 = scf.if %24 -> (i1, f32) {
        %31 = arith.cmpi eq, %4, %c0_i64 : i64
        %32 = arith.cmpi ne, %4, %c0_i64 : i64
        %33 = arith.select %31, %cst_0, %25 : f32
        scf.yield %32, %33 : i1, f32
      } else {
        scf.yield %false, %25 : i1, f32
      }
      %27 = arith.cmpi eq, %4, %c1_i64 : i64
      %28 = arith.cmpi ne, %4, %c1_i64 : i64
      %29 = arith.select %27, %2, %26#1 : f32
      %30 = scf.if %26#0 -> (f32) {
        %31 = scf.if %28 -> (f32) {
          %32 = arith.addf %2, %2 : f32
          %33:4 = scf.while (%arg5 = %c2_i64, %arg6 = %2, %arg7 = %2, %arg8 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
            %34 = arith.cmpi sle, %arg5, %4 : i64
            %35 = arith.cmpf uno, %arg7, %arg7 : f32
            %36 = arith.extui %35 : i1 to i8
            %37 = arith.cmpi eq, %36, %c0_i8 : i8
            %38 = arith.andi %34, %37 : i1
            scf.condition(%38) %arg6, %arg5, %arg7, %arg8 : f32, i64, f32, f32
          } do {
          ^bb0(%arg5: f32, %arg6: i64, %arg7: f32, %arg8: f32):
            %34 = arith.cmpf uno, %arg7, %arg7 : f32
            %35 = arith.extui %34 : i1 to i8
            %36 = arith.cmpi eq, %35, %c0_i8 : i8
            %37 = arith.select %36, %arg7, %arg8 : f32
            %38 = arith.cmpi sle, %arg6, %4 : i64
            %39 = arith.select %38, %37, %arg8 : f32
            %40 = arith.mulf %32, %arg7 : f32
            %41 = arith.subf %40, %arg8 : f32
            %42 = arith.addi %arg6, %c1_i64 : i64
            scf.yield %42, %41, %41, %39 : i64, f32, f32, f32
          }
          scf.yield %33#0 : f32
        } else {
          scf.yield %29 : f32
        }
        scf.yield %31 : f32
      } else {
        scf.yield %26#1 : f32
      }
      memref.store %30, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
  func.func private @acosf(f32) -> f32
}

