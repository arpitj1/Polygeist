module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_legendre_p_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i8 = arith.constant 0 : i8
    %c1_i64 = arith.constant 1 : i64
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
      %10:2 = scf.if %6 -> (i1, f32) {
        %15:2 = scf.if %9 -> (i1, f32) {
          %16 = arith.cmpf ogt, %2, %cst_1 : f32
          %17 = scf.if %16 -> (i1) {
            scf.yield %true : i1
          } else {
            %20 = arith.remsi %4, %c2_i64 : i64
            %21 = arith.cmpi eq, %20, %c0_i64 : i64
            scf.yield %21 : i1
          }
          %18 = arith.xori %17, %true : i1
          %19 = arith.select %18, %cst, %cst_0 : f32
          scf.yield %false, %19 : i1, f32
        } else {
          %16 = arith.cmpi eq, %4, %c0_i64 : i64
          %17 = arith.cmpi ne, %4, %c0_i64 : i64
          %18 = arith.select %16, %cst_0, %7 : f32
          scf.yield %17, %18 : i1, f32
        }
        scf.yield %15#0, %15#1 : i1, f32
      } else {
        scf.yield %false, %7 : i1, f32
      }
      %11 = arith.cmpi eq, %4, %c1_i64 : i64
      %12 = arith.cmpi ne, %4, %c1_i64 : i64
      %13 = arith.select %11, %2, %10#1 : f32
      %14 = scf.if %10#0 -> (f32) {
        %15 = scf.if %12 -> (f32) {
          %16:4 = scf.while (%arg5 = %c1_i64, %arg6 = %2, %arg7 = %2, %arg8 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
            %17 = arith.cmpi slt, %arg5, %4 : i64
            %18 = arith.cmpf uno, %arg7, %arg7 : f32
            %19 = arith.extui %18 : i1 to i8
            %20 = arith.cmpi eq, %19, %c0_i8 : i8
            %21 = arith.andi %17, %20 : i1
            scf.condition(%21) %arg6, %arg5, %arg7, %arg8 : f32, i64, f32, f32
          } do {
          ^bb0(%arg5: f32, %arg6: i64, %arg7: f32, %arg8: f32):
            %17 = arith.cmpf uno, %arg7, %arg7 : f32
            %18 = arith.extui %17 : i1 to i8
            %19 = arith.cmpi eq, %18, %c0_i8 : i8
            %20 = arith.select %19, %arg7, %arg8 : f32
            %21 = arith.cmpi slt, %arg6, %4 : i64
            %22 = arith.select %21, %20, %arg8 : f32
            %23 = arith.addi %arg6, %arg6 : i64
            %24 = arith.addi %23, %c1_i64 : i64
            %25 = arith.sitofp %24 : i64 to f32
            %26 = arith.mulf %25, %2 : f32
            %27 = arith.mulf %26, %arg7 : f32
            %28 = arith.sitofp %arg6 : i64 to f32
            %29 = arith.mulf %28, %arg8 : f32
            %30 = arith.subf %27, %29 : f32
            %31 = arith.addi %arg6, %c1_i64 : i64
            %32 = arith.sitofp %31 : i64 to f32
            %33 = arith.divf %30, %32 : f32
            scf.yield %31, %33, %33, %22 : i64, f32, f32, f32
          }
          scf.yield %16#0 : f32
        } else {
          scf.yield %13 : f32
        }
        scf.yield %15 : f32
      } else {
        scf.yield %10#1 : f32
      }
      memref.store %14, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
}

