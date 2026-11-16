module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global constant @"_Z4zetaIfLb0EET_S0_S0_@static@_ZZ4zetaIfLb0EET_S0_S0_E1A" : memref<12xf64> = dense<[1.200000e+01, -7.200000e+02, 3.024000e+04, -1.209600e+06, 0x4186D73000000000, -1892437580.3183792, 0x423165E990000000, -2950130727918.1641, 116467828143500.67, 0xC33055D6586F1431, 1.8152105401943546E+17, -7.1661652561756672E+18]>
  func.func @thrust_calc_zeta_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.1102230246251565E-16 : f64
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant 5.000000e-01 : f64
    %cst_2 = arith.constant 1.000000e+00 : f64
    %c0_i32 = arith.constant 0 : i32
    %c9_i32 = arith.constant 9 : i32
    %cst_3 = arith.constant 9.000000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %cst_4 = arith.constant -1.1102230246251565E-16 : f64
    %cst_5 = arith.constant 0x7F800000 : f32
    %cst_6 = arith.constant 0x7FC00000 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f64
    %2 = memref.get_global @"_Z4zetaIfLb0EET_S0_S0_@static@_ZZ4zetaIfLb0EET_S0_S0_E1A" : memref<12xf64>
    %3 = llvm.mlir.undef : f32
    affine.for %arg4 = 0 to %0 {
      %4 = affine.load %arg1[%arg4] : memref<?xf32>
      %5 = affine.load %arg2[%arg4] : memref<?xf32>
      %6 = arith.extf %4 : f32 to f64
      %7 = arith.cmpf oeq, %6, %cst_2 : f64
      %8 = arith.xori %7, %true : i1
      %9 = arith.select %7, %cst_5, %3 : f32
      %10 = arith.cmpf olt, %6, %cst_2 : f64
      %11 = arith.andi %10, %8 : i1
      %12 = arith.xori %11, %true : i1
      %13 = arith.andi %12, %8 : i1
      %14 = arith.select %11, %cst_6, %9 : f32
      %15 = arith.extf %5 : f32 to f64
      %16 = arith.cmpf ole, %15, %cst_0 : f64
      %17 = arith.andi %13, %16 : i1
      %18:2 = scf.if %17 -> (i1, f32) {
        %20:2 = scf.if %13 -> (i1, f32) {
          %21 = func.call @floorf(%5) : (f32) -> f32
          %22 = arith.cmpf oeq, %5, %21 : f32
          %23:2 = scf.if %22 -> (i1, f32) {
            scf.yield %false, %cst_5 : i1, f32
          } else {
            %24 = func.call @floorf(%4) : (f32) -> f32
            %25 = arith.cmpf une, %4, %24 : f32
            %26 = arith.xori %25, %true : i1
            %27 = arith.select %25, %cst_6, %14 : f32
            scf.yield %26, %27 : i1, f32
          }
          scf.yield %23#0, %23#1 : i1, f32
        } else {
          scf.yield %false, %14 : i1, f32
        }
        scf.yield %20#0, %20#1 : i1, f32
      } else {
        scf.yield %13, %14 : i1, f32
      }
      %19 = scf.if %18#0 -> (f32) {
        %20 = arith.negf %4 : f32
        %21 = math.powf %5, %20 : f32
        %22 = arith.extf %21 : f32 to f64
        %23 = arith.extf %20 : f32 to f64
        %24:6 = scf.while (%arg5 = %c0_i32, %arg6 = %22, %arg7 = %cst_0, %arg8 = %15, %arg9 = %true, %arg10 = %18#1, %arg11 = %true) : (i32, f64, f64, f64, i1, f32, i1) -> (f64, f64, f64, i1, f32, i32) {
          %26 = arith.cmpi slt, %arg5, %c9_i32 : i32
          %27 = scf.if %26 -> (i1) {
            scf.yield %true : i1
          } else {
            %29 = arith.cmpf ole, %arg8, %cst_3 : f64
            scf.yield %29 : i1
          }
          %28 = arith.andi %27, %arg11 : i1
          scf.condition(%28) %arg6, %arg7, %arg8, %arg9, %arg10, %arg5 : f64, f64, f64, i1, f32, i32
        } do {
        ^bb0(%arg5: f64, %arg6: f64, %arg7: f64, %arg8: i1, %arg9: f32, %arg10: i32):
          %26 = arith.addi %arg10, %c1_i32 : i32
          %27 = arith.addf %arg7, %cst_2 : f64
          %28 = math.powf %27, %23 : f64
          %29 = arith.addf %arg5, %28 : f64
          %30 = arith.mulf %29, %cst_4 : f64
          %31 = arith.cmpf olt, %30, %28 : f64
          %32 = arith.mulf %29, %cst : f64
          %33 = arith.cmpf olt, %28, %32 : f64
          %34 = arith.xori %33, %true : i1
          %35 = arith.andi %34, %arg8 : i1
          %36 = arith.andi %31, %33 : i1
          %37 = arith.xori %31, %true : i1
          %38 = arith.andi %31, %35 : i1
          %39 = arith.andi %37, %arg8 : i1
          %40 = arith.ori %38, %39 : i1
          %41 = arith.andi %31, %34 : i1
          %42 = arith.ori %41, %37 : i1
          %43 = scf.if %36 -> (f32) {
            %44 = arith.truncf %29 : f64 to f32
            scf.yield %44 : f32
          } else {
            scf.yield %arg9 : f32
          }
          scf.yield %26, %29, %28, %27, %40, %43, %42 : i32, f64, f64, f64, i1, f32, i1
        }
        %25 = scf.if %24#3 -> (f32) {
          %26 = arith.mulf %24#1, %24#2 : f64
          %27 = arith.subf %6, %cst_2 : f64
          %28 = arith.divf %26, %27 : f64
          %29 = arith.addf %24#0, %28 : f64
          %30 = arith.mulf %24#1, %cst_1 : f64
          %31 = arith.subf %29, %30 : f64
          %32:12 = affine.for %arg5 = 0 to 12 iter_args(%arg6 = %31, %arg7 = %true, %arg8 = %24#4, %arg9 = %1, %arg10 = %31, %arg11 = %cst_0, %arg12 = %24#1, %arg13 = %cst_2, %arg14 = %true, %arg15 = %24#4, %arg16 = %true, %arg17 = %true) -> (f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1) {
            %34:12 = scf.if %arg17 -> (f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1) {
              %35:4 = scf.if %arg16 -> (f64, f64, f64, f64) {
                %50 = arith.addf %6, %arg11 : f64
                %51 = arith.mulf %arg13, %50 : f64
                %52 = arith.divf %arg12, %24#2 : f64
                %53 = arith.mulf %51, %52 : f64
                %54 = affine.load %2[%arg5] : memref<12xf64>
                %55 = arith.divf %53, %54 : f64
                %56 = arith.addf %arg10, %55 : f64
                scf.yield %51, %52, %55, %56 : f64, f64, f64, f64
              } else {
                scf.yield %arg13, %arg12, %arg9, %arg10 : f64, f64, f64, f64
              }
              %36 = arith.divf %35#2, %35#3 : f64
              %37 = math.absf %36 : f64
              %38 = arith.select %arg16, %37, %35#2 : f64
              %39 = arith.cmpf olt, %37, %cst : f64
              %40 = arith.andi %39, %arg16 : i1
              %41 = arith.xori %40, %true : i1
              %42 = arith.andi %41, %arg14 : i1
              %43 = arith.andi %41, %arg16 : i1
              %44 = arith.xori %arg16, %true : i1
              %45 = arith.andi %arg16, %42 : i1
              %46 = arith.andi %44, %arg14 : i1
              %47 = arith.ori %45, %46 : i1
              %48 = scf.if %40 -> (f32) {
                %50 = arith.truncf %35#3 : f64 to f32
                scf.yield %50 : f32
              } else {
                scf.yield %arg15 : f32
              }
              %49:3 = scf.if %43 -> (f64, f64, f64) {
                %50 = arith.addf %arg11, %cst_2 : f64
                %51 = arith.addf %6, %50 : f64
                %52 = arith.mulf %35#0, %51 : f64
                %53 = arith.divf %35#1, %24#2 : f64
                %54 = arith.addf %50, %cst_2 : f64
                scf.yield %52, %53, %54 : f64, f64, f64
              } else {
                scf.yield %35#0, %35#1, %arg11 : f64, f64, f64
              }
              scf.yield %35#3, %47, %48, %38, %35#3, %49#2, %49#1, %49#0, %47, %48, %43, %43 : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
            } else {
              scf.yield %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %false : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
            }
            affine.yield %34#0, %34#1, %34#2, %34#3, %34#4, %34#5, %34#6, %34#7, %34#8, %34#9, %34#10, %34#11 : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
          }
          %33 = scf.if %32#1 -> (f32) {
            %34 = arith.truncf %32#0 : f64 to f32
            scf.yield %34 : f32
          } else {
            scf.yield %32#2 : f32
          }
          scf.yield %33 : f32
        } else {
          scf.yield %24#4 : f32
        }
        scf.yield %25 : f32
      } else {
        scf.yield %18#1 : f32
      }
      affine.store %19, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
  func.func private @floorf(f32) -> f32
}

