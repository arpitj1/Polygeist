module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global constant @"_Z4zetaIfLb0EET_S0_S0_@static@_ZZ4zetaIfLb0EET_S0_S0_E1A" : memref<12xf64> = dense<[1.200000e+01, -7.200000e+02, 3.024000e+04, -1.209600e+06, 0x4186D73000000000, -1892437580.3183792, 0x423165E990000000, -2950130727918.1641, 116467828143500.67, 0xC33055D6586F1431, 1.8152105401943546E+17, -7.1661652561756672E+18]>
  func.func @thrust_calc_polygamma_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: i32, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.000000e+00 : f32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant -1.000000e+00 : f32
    %cst_1 = arith.constant 0x7FC00000 : f32
    %cst_2 = arith.constant 0x7F800000 : f32
    %cst_3 = arith.constant -1.1102230246251565E-16 : f64
    %false = arith.constant false
    %true = arith.constant true
    %cst_4 = arith.constant 9.000000e+00 : f64
    %c9_i32 = arith.constant 9 : i32
    %cst_5 = arith.constant 1.000000e+00 : f64
    %cst_6 = arith.constant 5.000000e-01 : f64
    %cst_7 = arith.constant 0.000000e+00 : f64
    %cst_8 = arith.constant 1.1102230246251565E-16 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.remsi %arg2, %c2_i32 : i32
    %2 = arith.cmpi ne, %1, %c0_i32 : i32
    %3 = arith.select %2, %cst, %cst_0 : f32
    %4 = arith.sitofp %arg2 : i32 to f32
    %5 = arith.addf %4, %cst : f32
    %6 = arith.addi %arg2, %c1_i32 : i32
    %7 = arith.sitofp %6 : i32 to f32
    %8 = llvm.mlir.undef : f64
    %9 = memref.get_global @"_Z4zetaIfLb0EET_S0_S0_@static@_ZZ4zetaIfLb0EET_S0_S0_E1A" : memref<12xf64>
    %10 = llvm.mlir.undef : f32
    %11 = arith.extf %7 : f32 to f64
    %12 = arith.cmpf oeq, %11, %cst_5 : f64
    %13 = arith.xori %12, %true : i1
    %14 = arith.select %12, %cst_2, %10 : f32
    %15 = arith.cmpf olt, %11, %cst_5 : f64
    %16 = arith.andi %15, %13 : i1
    %17 = arith.xori %16, %true : i1
    %18 = arith.andi %17, %13 : i1
    %19 = arith.select %16, %cst_1, %14 : f32
    %20 = arith.negf %7 : f32
    %21 = arith.extf %20 : f32 to f64
    %22 = arith.subf %11, %cst_5 : f64
    affine.for %arg4 = 0 to %0 {
      %23 = affine.load %arg1[%arg4] : memref<?xf32>
      %24 = func.call @lgammaf(%5) : (f32) -> f32
      %25 = math.exp %24 : f32
      %26 = arith.mulf %3, %25 : f32
      %27 = arith.extf %23 : f32 to f64
      %28 = arith.cmpf ole, %27, %cst_7 : f64
      %29 = arith.andi %18, %28 : i1
      %30:2 = scf.if %29 -> (i1, f32) {
        %33:2 = scf.if %18 -> (i1, f32) {
          %34 = func.call @floorf(%23) : (f32) -> f32
          %35 = arith.cmpf oeq, %23, %34 : f32
          %36:2 = scf.if %35 -> (i1, f32) {
            scf.yield %false, %cst_2 : i1, f32
          } else {
            %37 = func.call @floorf(%7) : (f32) -> f32
            %38 = arith.cmpf une, %7, %37 : f32
            %39 = arith.xori %38, %true : i1
            %40 = arith.select %38, %cst_1, %19 : f32
            scf.yield %39, %40 : i1, f32
          }
          scf.yield %36#0, %36#1 : i1, f32
        } else {
          scf.yield %false, %19 : i1, f32
        }
        scf.yield %33#0, %33#1 : i1, f32
      } else {
        scf.yield %18, %19 : i1, f32
      }
      %31 = scf.if %30#0 -> (f32) {
        %33 = math.powf %23, %20 : f32
        %34 = arith.extf %33 : f32 to f64
        %35:6 = scf.while (%arg5 = %c0_i32, %arg6 = %34, %arg7 = %cst_7, %arg8 = %27, %arg9 = %true, %arg10 = %30#1, %arg11 = %true) : (i32, f64, f64, f64, i1, f32, i1) -> (f64, f64, f64, i1, f32, i32) {
          %37 = arith.cmpi slt, %arg5, %c9_i32 : i32
          %38 = scf.if %37 -> (i1) {
            scf.yield %true : i1
          } else {
            %40 = arith.cmpf ole, %arg8, %cst_4 : f64
            scf.yield %40 : i1
          }
          %39 = arith.andi %38, %arg11 : i1
          scf.condition(%39) %arg6, %arg7, %arg8, %arg9, %arg10, %arg5 : f64, f64, f64, i1, f32, i32
        } do {
        ^bb0(%arg5: f64, %arg6: f64, %arg7: f64, %arg8: i1, %arg9: f32, %arg10: i32):
          %37 = arith.addi %arg10, %c1_i32 : i32
          %38 = arith.addf %arg7, %cst_5 : f64
          %39 = math.powf %38, %21 : f64
          %40 = arith.addf %arg5, %39 : f64
          %41 = arith.mulf %40, %cst_3 : f64
          %42 = arith.cmpf olt, %41, %39 : f64
          %43 = arith.mulf %40, %cst_8 : f64
          %44 = arith.cmpf olt, %39, %43 : f64
          %45 = arith.xori %44, %true : i1
          %46 = arith.andi %45, %arg8 : i1
          %47 = arith.andi %42, %44 : i1
          %48 = arith.xori %42, %true : i1
          %49 = arith.andi %42, %46 : i1
          %50 = arith.andi %48, %arg8 : i1
          %51 = arith.ori %49, %50 : i1
          %52 = arith.andi %42, %45 : i1
          %53 = arith.ori %52, %48 : i1
          %54 = scf.if %47 -> (f32) {
            %55 = arith.truncf %40 : f64 to f32
            scf.yield %55 : f32
          } else {
            scf.yield %arg9 : f32
          }
          scf.yield %37, %40, %39, %38, %51, %54, %53 : i32, f64, f64, f64, i1, f32, i1
        }
        %36 = scf.if %35#3 -> (f32) {
          %37 = arith.mulf %35#1, %35#2 : f64
          %38 = arith.divf %37, %22 : f64
          %39 = arith.addf %35#0, %38 : f64
          %40 = arith.mulf %35#1, %cst_6 : f64
          %41 = arith.subf %39, %40 : f64
          %42:12 = affine.for %arg5 = 0 to 12 iter_args(%arg6 = %41, %arg7 = %true, %arg8 = %35#4, %arg9 = %8, %arg10 = %41, %arg11 = %cst_7, %arg12 = %35#1, %arg13 = %cst_5, %arg14 = %true, %arg15 = %35#4, %arg16 = %true, %arg17 = %true) -> (f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1) {
            %44:12 = scf.if %arg17 -> (f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1) {
              %45:4 = scf.if %arg16 -> (f64, f64, f64, f64) {
                %60 = arith.addf %11, %arg11 : f64
                %61 = arith.mulf %arg13, %60 : f64
                %62 = arith.divf %arg12, %35#2 : f64
                %63 = arith.mulf %61, %62 : f64
                %64 = affine.load %9[%arg5] : memref<12xf64>
                %65 = arith.divf %63, %64 : f64
                %66 = arith.addf %arg10, %65 : f64
                scf.yield %61, %62, %65, %66 : f64, f64, f64, f64
              } else {
                scf.yield %arg13, %arg12, %arg9, %arg10 : f64, f64, f64, f64
              }
              %46 = arith.divf %45#2, %45#3 : f64
              %47 = math.absf %46 : f64
              %48 = arith.select %arg16, %47, %45#2 : f64
              %49 = arith.cmpf olt, %47, %cst_8 : f64
              %50 = arith.andi %49, %arg16 : i1
              %51 = arith.xori %50, %true : i1
              %52 = arith.andi %51, %arg14 : i1
              %53 = arith.andi %51, %arg16 : i1
              %54 = arith.xori %arg16, %true : i1
              %55 = arith.andi %arg16, %52 : i1
              %56 = arith.andi %54, %arg14 : i1
              %57 = arith.ori %55, %56 : i1
              %58 = scf.if %50 -> (f32) {
                %60 = arith.truncf %45#3 : f64 to f32
                scf.yield %60 : f32
              } else {
                scf.yield %arg15 : f32
              }
              %59:3 = scf.if %53 -> (f64, f64, f64) {
                %60 = arith.addf %arg11, %cst_5 : f64
                %61 = arith.addf %11, %60 : f64
                %62 = arith.mulf %45#0, %61 : f64
                %63 = arith.divf %45#1, %35#2 : f64
                %64 = arith.addf %60, %cst_5 : f64
                scf.yield %62, %63, %64 : f64, f64, f64
              } else {
                scf.yield %45#0, %45#1, %arg11 : f64, f64, f64
              }
              scf.yield %45#3, %57, %58, %48, %45#3, %59#2, %59#1, %59#0, %57, %58, %53, %53 : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
            } else {
              scf.yield %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %false : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
            }
            affine.yield %44#0, %44#1, %44#2, %44#3, %44#4, %44#5, %44#6, %44#7, %44#8, %44#9, %44#10, %44#11 : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
          }
          %43 = scf.if %42#1 -> (f32) {
            %44 = arith.truncf %42#0 : f64 to f32
            scf.yield %44 : f32
          } else {
            scf.yield %42#2 : f32
          }
          scf.yield %43 : f32
        } else {
          scf.yield %35#4 : f32
        }
        scf.yield %36 : f32
      } else {
        scf.yield %30#1 : f32
      }
      %32 = arith.mulf %26, %31 : f32
      affine.store %32, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
  func.func private @lgammaf(f32) -> f32
  func.func private @floorf(f32) -> f32
}

