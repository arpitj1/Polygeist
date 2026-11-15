module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global constant @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AGD" : memref<10xf32> = dense<[9.30892944, 19.8352928, 15.5646629, 5.47686052, 0.954293608, 0.0864580795, 0.00412656507, 1.01259087E-4, 1.17166735E-6, 4.91834573E-9]>
  memref.global constant @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AGN" : memref<11xf32> = dense<[0.0197339933, 0.391103029, 1.065799, 0.939169228, 0.351465642, 0.0633888915, 0.00585804135, 2.82851601E-4, 6.98793655E-6, 8.11789249E-8, 3.41551787E-10]>
  memref.global constant @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AFD" : memref<9xf32> = dense<[13.3560419, 32.6825027, 26.7367039, 9.1870737, 1.47529149, 0.115687177, 0.00440291641, 7.54720313E-5, 4.51850099E-7]>
  memref.global constant @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AFN" : memref<9xf32> = dense<[-0.131696329, -0.626456558, -0.693158031, -0.279779971, -0.0491900146, -0.00406265911, -1.59276489E-4, -2.7764911E-6, -1.67787704E-8]>
  memref.global constant @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E2AD" : memref<8xf32> = dense<[0.567594528, 14.7562561, 8.451390e+01, 177.318085, 164.234695, 71.4778366, 14.0959139, 1.000000e+00]>
  memref.global constant @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E2AN" : memref<8xf32> = dense<[0.346538097, 12.0075951, 7.627960e+01, 168.089218, 159.756393, 70.536087, 14.0264692, 1.000000e+00]>
  func.func @thrust_calc_airy_ai_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i8 = arith.constant 0 : i8
    %cst_0 = arith.constant 1.038920e+02 : f32
    %cst_1 = arith.constant -2.090000e+00 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant -2.000000e+00 : f32
    %cst_4 = arith.constant 3.000000e+00 : f32
    %cst_5 = arith.constant 0.564189613 : f32
    %cst_6 = arith.constant 2.090000e+00 : f32
    %c5_i32 = arith.constant 5 : i32
    %cst_7 = arith.constant 2.000000e+00 : f32
    %cst_8 = arith.constant 8.32033538 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_9 = arith.constant 0.355028063 : f32
    %cst_10 = arith.constant 0.258819401 : f32
    %true = arith.constant true
    %false = arith.constant false
    %cst_11 = arith.constant 0x7F800000 : f32
    %cst_12 = arith.constant 0x7FC00000 : f32
    %cst_13 = arith.constant 1.1920929E-7 : f32
    %cst_14 = arith.constant 0.785398185 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    %2 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AGD" : memref<10xf32>
    %3 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AGN" : memref<11xf32>
    %4 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AFD" : memref<9xf32>
    %5 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AFN" : memref<9xf32>
    %6 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E2AD" : memref<8xf32>
    %7 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E2AN" : memref<8xf32>
    affine.for %arg3 = 0 to %0 {
      %8 = affine.load %arg1[%arg3] : memref<?xf32>
      %9 = math.absf %8 : f32
      %10 = arith.cmpf oeq, %9, %cst_11 : f32
      %11 = arith.extui %10 : i1 to i8
      %12 = arith.cmpi eq, %11, %c0_i8 : i8
      %13 = arith.select %10, %cst_12, %1 : f32
      %14:2 = scf.if %12 -> (i1, f32) {
        %27 = arith.cmpf ogt, %8, %cst_0 : f32
        %28 = arith.xori %27, %true : i1
        %29 = arith.select %27, %cst, %13 : f32
        scf.yield %28, %29 : i1, f32
      } else {
        scf.yield %false, %13 : i1, f32
      }
      %15 = arith.cmpf olt, %8, %cst_1 : f32
      %16 = arith.andi %15, %14#0 : i1
      %17 = arith.xori %15, %true : i1
      %18 = arith.andi %17, %14#0 : i1
      %19 = scf.if %16 -> (f32) {
        %27 = arith.mulf %8, %cst_3 : f32
        %28 = arith.negf %8 : f32
        %29 = math.sqrt %28 : f32
        %30 = arith.mulf %27, %29 : f32
        %31 = arith.divf %30, %cst_4 : f32
        %32 = arith.divf %cst_2, %31 : f32
        %33 = arith.mulf %32, %32 : f32
        %34 = affine.for %arg4 = 0 to 9 iter_args(%arg5 = %cst) -> (f32) {
          %52 = arith.index_cast %arg4 : index to i8
          %53 = arith.mulf %arg5, %33 : f32
          %54 = arith.index_castui %52 : i8 to index
          %55 = memref.load %5[%54] : memref<9xf32>
          %56 = arith.addf %53, %55 : f32
          affine.yield %56 : f32
        }
        %35 = affine.for %arg4 = 0 to 9 iter_args(%arg5 = %cst) -> (f32) {
          %52 = arith.index_cast %arg4 : index to i8
          %53 = arith.mulf %arg5, %33 : f32
          %54 = arith.index_castui %52 : i8 to index
          %55 = memref.load %4[%54] : memref<9xf32>
          %56 = arith.addf %53, %55 : f32
          affine.yield %56 : f32
        }
        %36 = affine.for %arg4 = 0 to 11 iter_args(%arg5 = %cst) -> (f32) {
          %52 = arith.index_cast %arg4 : index to i8
          %53 = arith.mulf %arg5, %33 : f32
          %54 = arith.index_castui %52 : i8 to index
          %55 = memref.load %3[%54] : memref<11xf32>
          %56 = arith.addf %53, %55 : f32
          affine.yield %56 : f32
        }
        %37 = affine.for %arg4 = 0 to 10 iter_args(%arg5 = %cst) -> (f32) {
          %52 = arith.index_cast %arg4 : index to i8
          %53 = arith.mulf %arg5, %33 : f32
          %54 = arith.index_castui %52 : i8 to index
          %55 = memref.load %2[%54] : memref<10xf32>
          %56 = arith.addf %53, %55 : f32
          affine.yield %56 : f32
        }
        %38 = arith.addf %31, %cst_14 : f32
        %39 = math.sqrt %29 : f32
        %40 = arith.divf %cst_5, %39 : f32
        %41 = math.sin %38 : f32
        %42 = arith.mulf %33, %34 : f32
        %43 = arith.divf %42, %35 : f32
        %44 = arith.addf %43, %cst_2 : f32
        %45 = arith.mulf %41, %44 : f32
        %46 = math.cos %38 : f32
        %47 = arith.mulf %32, %36 : f32
        %48 = arith.divf %47, %37 : f32
        %49 = arith.mulf %46, %48 : f32
        %50 = arith.subf %45, %49 : f32
        %51 = arith.mulf %40, %50 : f32
        scf.yield %51 : f32
      } else {
        scf.yield %14#1 : f32
      }
      %20 = arith.cmpf oge, %8, %cst_6 : f32
      %21 = arith.andi %18, %20 : i1
      %22 = arith.select %20, %c5_i32, %c0_i32 : i32
      %23 = arith.select %18, %22, %c0_i32 : i32
      %24:3 = scf.if %21 -> (f32, i1, f32) {
        %27:3 = scf.if %18 -> (f32, i1, f32) {
          %28 = arith.mulf %8, %cst_7 : f32
          %29 = math.sqrt %8 : f32
          %30 = arith.mulf %28, %29 : f32
          %31 = arith.divf %30, %cst_4 : f32
          %32 = arith.divf %cst_2, %31 : f32
          %33 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %cst) -> (f32) {
            %45 = arith.index_cast %arg4 : index to i8
            %46 = arith.mulf %arg5, %32 : f32
            %47 = arith.index_castui %45 : i8 to index
            %48 = memref.load %7[%47] : memref<8xf32>
            %49 = arith.addf %46, %48 : f32
            affine.yield %49 : f32
          }
          %34 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %cst) -> (f32) {
            %45 = arith.index_cast %arg4 : index to i8
            %46 = arith.mulf %arg5, %32 : f32
            %47 = arith.index_castui %45 : i8 to index
            %48 = memref.load %6[%47] : memref<8xf32>
            %49 = arith.addf %46, %48 : f32
            affine.yield %49 : f32
          }
          %35 = arith.divf %33, %34 : f32
          %36 = arith.mulf %35, %cst_5 : f32
          %37 = math.sqrt %29 : f32
          %38 = arith.mulf %37, %cst_7 : f32
          %39 = math.exp %31 : f32
          %40 = arith.mulf %38, %39 : f32
          %41 = arith.divf %36, %40 : f32
          %42 = arith.cmpf ogt, %8, %cst_8 : f32
          %43 = arith.xori %42, %true : i1
          %44 = arith.select %42, %41, %19 : f32
          scf.yield %41, %43, %44 : f32, i1, f32
        } else {
          scf.yield %cst, %false, %19 : f32, i1, f32
        }
        scf.yield %27#0, %27#1, %27#2 : f32, i1, f32
      } else {
        scf.yield %cst, %18, %19 : f32, i1, f32
      }
      %25:2 = scf.if %24#1 -> (i1, f32) {
        %27 = arith.mulf %8, %8 : f32
        %28 = arith.mulf %27, %8 : f32
        %29:5 = scf.while (%arg4 = %cst_2, %arg5 = %8, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %8, %arg9 = %cst_2) : (f32, f32, f32, f32, f32, f32) -> (f32, f32, f32, f32, f32) {
          %34 = arith.cmpf ogt, %arg4, %cst_13 : f32
          scf.condition(%34) %arg8, %arg9, %arg5, %arg6, %arg7 : f32, f32, f32, f32, f32
        } do {
        ^bb0(%arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):
          %34 = arith.mulf %arg7, %28 : f32
          %35 = arith.addf %arg8, %cst_2 : f32
          %36 = arith.divf %34, %35 : f32
          %37 = arith.mulf %arg6, %28 : f32
          %38 = arith.addf %35, %cst_2 : f32
          %39 = arith.divf %37, %38 : f32
          %40 = arith.divf %36, %38 : f32
          %41 = arith.addf %arg5, %40 : f32
          %42 = arith.addf %38, %cst_2 : f32
          %43 = arith.divf %39, %42 : f32
          %44 = arith.addf %arg4, %43 : f32
          %45 = arith.divf %40, %41 : f32
          %46 = math.absf %45 : f32
          scf.yield %46, %43, %40, %42, %44, %41 : f32, f32, f32, f32, f32, f32
        }
        %30 = arith.andi %23, %c1_i32 : i32
        %31 = arith.cmpi eq, %30, %c0_i32 : i32
        %32 = arith.cmpi ne, %30, %c0_i32 : i32
        %33 = scf.if %31 -> (f32) {
          %34 = arith.mulf %29#1, %cst_9 : f32
          %35 = arith.mulf %29#0, %cst_10 : f32
          %36 = arith.subf %34, %35 : f32
          scf.yield %36 : f32
        } else {
          scf.yield %24#2 : f32
        }
        scf.yield %32, %33 : i1, f32
      } else {
        scf.yield %false, %24#2 : i1, f32
      }
      %26 = arith.select %25#0, %24#0, %25#1 : f32
      affine.store %26, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
  func.func @calc_airy_aif(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.785398185 : f32
    %cst_0 = arith.constant 1.1920929E-7 : f32
    %cst_1 = arith.constant 0x7FC00000 : f32
    %cst_2 = arith.constant 0x7F800000 : f32
    %false = arith.constant false
    %true = arith.constant true
    %cst_3 = arith.constant 0.258819401 : f32
    %cst_4 = arith.constant 0.355028063 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_5 = arith.constant 8.32033538 : f32
    %cst_6 = arith.constant 2.000000e+00 : f32
    %c5_i32 = arith.constant 5 : i32
    %cst_7 = arith.constant 2.090000e+00 : f32
    %cst_8 = arith.constant 0.564189613 : f32
    %cst_9 = arith.constant 3.000000e+00 : f32
    %cst_10 = arith.constant -2.000000e+00 : f32
    %cst_11 = arith.constant 1.000000e+00 : f32
    %cst_12 = arith.constant -2.090000e+00 : f32
    %cst_13 = arith.constant 1.038920e+02 : f32
    %c0_i8 = arith.constant 0 : i8
    %cst_14 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : f32
    %1 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AGD" : memref<10xf32>
    %2 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AGN" : memref<11xf32>
    %3 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AFD" : memref<9xf32>
    %4 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AFN" : memref<9xf32>
    %5 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E2AD" : memref<8xf32>
    %6 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E2AN" : memref<8xf32>
    %7 = math.absf %arg0 : f32
    %8 = arith.cmpf oeq, %7, %cst_2 : f32
    %9 = arith.extui %8 : i1 to i8
    %10 = arith.cmpi eq, %9, %c0_i8 : i8
    %11 = arith.select %8, %cst_1, %0 : f32
    %12:2 = scf.if %10 -> (i1, f32) {
      %25 = arith.cmpf ogt, %arg0, %cst_13 : f32
      %26 = arith.xori %25, %true : i1
      %27 = arith.select %25, %cst_14, %11 : f32
      scf.yield %26, %27 : i1, f32
    } else {
      scf.yield %false, %11 : i1, f32
    }
    %13 = arith.cmpf olt, %arg0, %cst_12 : f32
    %14 = arith.andi %13, %12#0 : i1
    %15 = arith.xori %13, %true : i1
    %16 = arith.andi %15, %12#0 : i1
    %17 = scf.if %14 -> (f32) {
      %25 = arith.mulf %arg0, %cst_10 : f32
      %26 = arith.negf %arg0 : f32
      %27 = math.sqrt %26 : f32
      %28 = arith.mulf %25, %27 : f32
      %29 = arith.divf %28, %cst_9 : f32
      %30 = arith.divf %cst_11, %29 : f32
      %31 = arith.mulf %30, %30 : f32
      %32 = affine.for %arg1 = 0 to 9 iter_args(%arg2 = %cst_14) -> (f32) {
        %50 = arith.index_cast %arg1 : index to i8
        %51 = arith.mulf %arg2, %31 : f32
        %52 = arith.index_castui %50 : i8 to index
        %53 = memref.load %4[%52] : memref<9xf32>
        %54 = arith.addf %51, %53 : f32
        affine.yield %54 : f32
      }
      %33 = affine.for %arg1 = 0 to 9 iter_args(%arg2 = %cst_14) -> (f32) {
        %50 = arith.index_cast %arg1 : index to i8
        %51 = arith.mulf %arg2, %31 : f32
        %52 = arith.index_castui %50 : i8 to index
        %53 = memref.load %3[%52] : memref<9xf32>
        %54 = arith.addf %51, %53 : f32
        affine.yield %54 : f32
      }
      %34 = affine.for %arg1 = 0 to 11 iter_args(%arg2 = %cst_14) -> (f32) {
        %50 = arith.index_cast %arg1 : index to i8
        %51 = arith.mulf %arg2, %31 : f32
        %52 = arith.index_castui %50 : i8 to index
        %53 = memref.load %2[%52] : memref<11xf32>
        %54 = arith.addf %51, %53 : f32
        affine.yield %54 : f32
      }
      %35 = affine.for %arg1 = 0 to 10 iter_args(%arg2 = %cst_14) -> (f32) {
        %50 = arith.index_cast %arg1 : index to i8
        %51 = arith.mulf %arg2, %31 : f32
        %52 = arith.index_castui %50 : i8 to index
        %53 = memref.load %1[%52] : memref<10xf32>
        %54 = arith.addf %51, %53 : f32
        affine.yield %54 : f32
      }
      %36 = arith.addf %29, %cst : f32
      %37 = math.sqrt %27 : f32
      %38 = arith.divf %cst_8, %37 : f32
      %39 = math.sin %36 : f32
      %40 = arith.mulf %31, %32 : f32
      %41 = arith.divf %40, %33 : f32
      %42 = arith.addf %41, %cst_11 : f32
      %43 = arith.mulf %39, %42 : f32
      %44 = math.cos %36 : f32
      %45 = arith.mulf %30, %34 : f32
      %46 = arith.divf %45, %35 : f32
      %47 = arith.mulf %44, %46 : f32
      %48 = arith.subf %43, %47 : f32
      %49 = arith.mulf %38, %48 : f32
      scf.yield %49 : f32
    } else {
      scf.yield %12#1 : f32
    }
    %18 = arith.cmpf oge, %arg0, %cst_7 : f32
    %19 = arith.andi %16, %18 : i1
    %20 = arith.select %18, %c5_i32, %c0_i32 : i32
    %21 = arith.select %16, %20, %c0_i32 : i32
    %22:3 = scf.if %19 -> (f32, i1, f32) {
      %25:3 = scf.if %16 -> (f32, i1, f32) {
        %26 = arith.mulf %arg0, %cst_6 : f32
        %27 = math.sqrt %arg0 : f32
        %28 = arith.mulf %26, %27 : f32
        %29 = arith.divf %28, %cst_9 : f32
        %30 = arith.divf %cst_11, %29 : f32
        %31 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %cst_14) -> (f32) {
          %43 = arith.index_cast %arg1 : index to i8
          %44 = arith.mulf %arg2, %30 : f32
          %45 = arith.index_castui %43 : i8 to index
          %46 = memref.load %6[%45] : memref<8xf32>
          %47 = arith.addf %44, %46 : f32
          affine.yield %47 : f32
        }
        %32 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %cst_14) -> (f32) {
          %43 = arith.index_cast %arg1 : index to i8
          %44 = arith.mulf %arg2, %30 : f32
          %45 = arith.index_castui %43 : i8 to index
          %46 = memref.load %5[%45] : memref<8xf32>
          %47 = arith.addf %44, %46 : f32
          affine.yield %47 : f32
        }
        %33 = arith.divf %31, %32 : f32
        %34 = arith.mulf %33, %cst_8 : f32
        %35 = math.sqrt %27 : f32
        %36 = arith.mulf %35, %cst_6 : f32
        %37 = math.exp %29 : f32
        %38 = arith.mulf %36, %37 : f32
        %39 = arith.divf %34, %38 : f32
        %40 = arith.cmpf ogt, %arg0, %cst_5 : f32
        %41 = arith.xori %40, %true : i1
        %42 = arith.select %40, %39, %17 : f32
        scf.yield %39, %41, %42 : f32, i1, f32
      } else {
        scf.yield %cst_14, %false, %17 : f32, i1, f32
      }
      scf.yield %25#0, %25#1, %25#2 : f32, i1, f32
    } else {
      scf.yield %cst_14, %16, %17 : f32, i1, f32
    }
    %23:2 = scf.if %22#1 -> (i1, f32) {
      %25 = arith.mulf %arg0, %arg0 : f32
      %26 = arith.mulf %25, %arg0 : f32
      %27:5 = scf.while (%arg1 = %cst_11, %arg2 = %arg0, %arg3 = %cst_11, %arg4 = %cst_11, %arg5 = %arg0, %arg6 = %cst_11) : (f32, f32, f32, f32, f32, f32) -> (f32, f32, f32, f32, f32) {
        %32 = arith.cmpf ogt, %arg1, %cst_0 : f32
        scf.condition(%32) %arg5, %arg6, %arg2, %arg3, %arg4 : f32, f32, f32, f32, f32
      } do {
      ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32):
        %32 = arith.mulf %arg4, %26 : f32
        %33 = arith.addf %arg5, %cst_11 : f32
        %34 = arith.divf %32, %33 : f32
        %35 = arith.mulf %arg3, %26 : f32
        %36 = arith.addf %33, %cst_11 : f32
        %37 = arith.divf %35, %36 : f32
        %38 = arith.divf %34, %36 : f32
        %39 = arith.addf %arg2, %38 : f32
        %40 = arith.addf %36, %cst_11 : f32
        %41 = arith.divf %37, %40 : f32
        %42 = arith.addf %arg1, %41 : f32
        %43 = arith.divf %38, %39 : f32
        %44 = math.absf %43 : f32
        scf.yield %44, %41, %38, %40, %42, %39 : f32, f32, f32, f32, f32, f32
      }
      %28 = arith.andi %21, %c1_i32 : i32
      %29 = arith.cmpi eq, %28, %c0_i32 : i32
      %30 = arith.cmpi ne, %28, %c0_i32 : i32
      %31 = scf.if %29 -> (f32) {
        %32 = arith.mulf %27#1, %cst_4 : f32
        %33 = arith.mulf %27#0, %cst_3 : f32
        %34 = arith.subf %32, %33 : f32
        scf.yield %34 : f32
      } else {
        scf.yield %22#2 : f32
      }
      scf.yield %30, %31 : i1, f32
    } else {
      scf.yield %false, %22#2 : i1, f32
    }
    %24 = arith.select %23#0, %22#0, %23#1 : f32
    return %24 : f32
  }
  func.func @_Z15airy_ai_forwardIfET_S0_(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant 0.785398185 : f32
    %cst_0 = arith.constant 1.1920929E-7 : f32
    %cst_1 = arith.constant 0x7FC00000 : f32
    %cst_2 = arith.constant 0x7F800000 : f32
    %false = arith.constant false
    %true = arith.constant true
    %cst_3 = arith.constant 0.258819401 : f32
    %cst_4 = arith.constant 0.355028063 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_5 = arith.constant 8.32033538 : f32
    %cst_6 = arith.constant 2.000000e+00 : f32
    %c5_i32 = arith.constant 5 : i32
    %cst_7 = arith.constant 2.090000e+00 : f32
    %cst_8 = arith.constant 0.564189613 : f32
    %cst_9 = arith.constant 3.000000e+00 : f32
    %cst_10 = arith.constant -2.000000e+00 : f32
    %cst_11 = arith.constant 1.000000e+00 : f32
    %cst_12 = arith.constant -2.090000e+00 : f32
    %cst_13 = arith.constant 1.038920e+02 : f32
    %c0_i8 = arith.constant 0 : i8
    %cst_14 = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : f32
    %1 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AGD" : memref<10xf32>
    %2 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AGN" : memref<11xf32>
    %3 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AFD" : memref<9xf32>
    %4 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AFN" : memref<9xf32>
    %5 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E2AD" : memref<8xf32>
    %6 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E2AN" : memref<8xf32>
    %7 = math.absf %arg0 : f32
    %8 = arith.cmpf oeq, %7, %cst_2 : f32
    %9 = arith.extui %8 : i1 to i8
    %10 = arith.cmpi eq, %9, %c0_i8 : i8
    %11 = arith.select %8, %cst_1, %0 : f32
    %12:2 = scf.if %10 -> (i1, f32) {
      %25 = arith.cmpf ogt, %arg0, %cst_13 : f32
      %26 = arith.xori %25, %true : i1
      %27 = arith.select %25, %cst_14, %11 : f32
      scf.yield %26, %27 : i1, f32
    } else {
      scf.yield %false, %11 : i1, f32
    }
    %13 = arith.cmpf olt, %arg0, %cst_12 : f32
    %14 = arith.andi %13, %12#0 : i1
    %15 = arith.xori %13, %true : i1
    %16 = arith.andi %15, %12#0 : i1
    %17 = scf.if %14 -> (f32) {
      %25 = arith.mulf %arg0, %cst_10 : f32
      %26 = arith.negf %arg0 : f32
      %27 = math.sqrt %26 : f32
      %28 = arith.mulf %25, %27 : f32
      %29 = arith.divf %28, %cst_9 : f32
      %30 = arith.divf %cst_11, %29 : f32
      %31 = arith.mulf %30, %30 : f32
      %32 = affine.for %arg1 = 0 to 9 iter_args(%arg2 = %cst_14) -> (f32) {
        %50 = arith.index_cast %arg1 : index to i8
        %51 = arith.mulf %arg2, %31 : f32
        %52 = arith.index_castui %50 : i8 to index
        %53 = memref.load %4[%52] : memref<9xf32>
        %54 = arith.addf %51, %53 : f32
        affine.yield %54 : f32
      }
      %33 = affine.for %arg1 = 0 to 9 iter_args(%arg2 = %cst_14) -> (f32) {
        %50 = arith.index_cast %arg1 : index to i8
        %51 = arith.mulf %arg2, %31 : f32
        %52 = arith.index_castui %50 : i8 to index
        %53 = memref.load %3[%52] : memref<9xf32>
        %54 = arith.addf %51, %53 : f32
        affine.yield %54 : f32
      }
      %34 = affine.for %arg1 = 0 to 11 iter_args(%arg2 = %cst_14) -> (f32) {
        %50 = arith.index_cast %arg1 : index to i8
        %51 = arith.mulf %arg2, %31 : f32
        %52 = arith.index_castui %50 : i8 to index
        %53 = memref.load %2[%52] : memref<11xf32>
        %54 = arith.addf %51, %53 : f32
        affine.yield %54 : f32
      }
      %35 = affine.for %arg1 = 0 to 10 iter_args(%arg2 = %cst_14) -> (f32) {
        %50 = arith.index_cast %arg1 : index to i8
        %51 = arith.mulf %arg2, %31 : f32
        %52 = arith.index_castui %50 : i8 to index
        %53 = memref.load %1[%52] : memref<10xf32>
        %54 = arith.addf %51, %53 : f32
        affine.yield %54 : f32
      }
      %36 = arith.addf %29, %cst : f32
      %37 = math.sqrt %27 : f32
      %38 = arith.divf %cst_8, %37 : f32
      %39 = math.sin %36 : f32
      %40 = arith.mulf %31, %32 : f32
      %41 = arith.divf %40, %33 : f32
      %42 = arith.addf %41, %cst_11 : f32
      %43 = arith.mulf %39, %42 : f32
      %44 = math.cos %36 : f32
      %45 = arith.mulf %30, %34 : f32
      %46 = arith.divf %45, %35 : f32
      %47 = arith.mulf %44, %46 : f32
      %48 = arith.subf %43, %47 : f32
      %49 = arith.mulf %38, %48 : f32
      scf.yield %49 : f32
    } else {
      scf.yield %12#1 : f32
    }
    %18 = arith.cmpf oge, %arg0, %cst_7 : f32
    %19 = arith.andi %16, %18 : i1
    %20 = arith.select %18, %c5_i32, %c0_i32 : i32
    %21 = arith.select %16, %20, %c0_i32 : i32
    %22:3 = scf.if %19 -> (f32, i1, f32) {
      %25:3 = scf.if %16 -> (f32, i1, f32) {
        %26 = arith.mulf %arg0, %cst_6 : f32
        %27 = math.sqrt %arg0 : f32
        %28 = arith.mulf %26, %27 : f32
        %29 = arith.divf %28, %cst_9 : f32
        %30 = arith.divf %cst_11, %29 : f32
        %31 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %cst_14) -> (f32) {
          %43 = arith.index_cast %arg1 : index to i8
          %44 = arith.mulf %arg2, %30 : f32
          %45 = arith.index_castui %43 : i8 to index
          %46 = memref.load %6[%45] : memref<8xf32>
          %47 = arith.addf %44, %46 : f32
          affine.yield %47 : f32
        }
        %32 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %cst_14) -> (f32) {
          %43 = arith.index_cast %arg1 : index to i8
          %44 = arith.mulf %arg2, %30 : f32
          %45 = arith.index_castui %43 : i8 to index
          %46 = memref.load %5[%45] : memref<8xf32>
          %47 = arith.addf %44, %46 : f32
          affine.yield %47 : f32
        }
        %33 = arith.divf %31, %32 : f32
        %34 = arith.mulf %33, %cst_8 : f32
        %35 = math.sqrt %27 : f32
        %36 = arith.mulf %35, %cst_6 : f32
        %37 = math.exp %29 : f32
        %38 = arith.mulf %36, %37 : f32
        %39 = arith.divf %34, %38 : f32
        %40 = arith.cmpf ogt, %arg0, %cst_5 : f32
        %41 = arith.xori %40, %true : i1
        %42 = arith.select %40, %39, %17 : f32
        scf.yield %39, %41, %42 : f32, i1, f32
      } else {
        scf.yield %cst_14, %false, %17 : f32, i1, f32
      }
      scf.yield %25#0, %25#1, %25#2 : f32, i1, f32
    } else {
      scf.yield %cst_14, %16, %17 : f32, i1, f32
    }
    %23:2 = scf.if %22#1 -> (i1, f32) {
      %25 = arith.mulf %arg0, %arg0 : f32
      %26 = arith.mulf %25, %arg0 : f32
      %27:5 = scf.while (%arg1 = %cst_11, %arg2 = %arg0, %arg3 = %cst_11, %arg4 = %cst_11, %arg5 = %arg0, %arg6 = %cst_11) : (f32, f32, f32, f32, f32, f32) -> (f32, f32, f32, f32, f32) {
        %32 = arith.cmpf ogt, %arg1, %cst_0 : f32
        scf.condition(%32) %arg5, %arg6, %arg2, %arg3, %arg4 : f32, f32, f32, f32, f32
      } do {
      ^bb0(%arg1: f32, %arg2: f32, %arg3: f32, %arg4: f32, %arg5: f32):
        %32 = arith.mulf %arg4, %26 : f32
        %33 = arith.addf %arg5, %cst_11 : f32
        %34 = arith.divf %32, %33 : f32
        %35 = arith.mulf %arg3, %26 : f32
        %36 = arith.addf %33, %cst_11 : f32
        %37 = arith.divf %35, %36 : f32
        %38 = arith.divf %34, %36 : f32
        %39 = arith.addf %arg2, %38 : f32
        %40 = arith.addf %36, %cst_11 : f32
        %41 = arith.divf %37, %40 : f32
        %42 = arith.addf %arg1, %41 : f32
        %43 = arith.divf %38, %39 : f32
        %44 = math.absf %43 : f32
        scf.yield %44, %41, %38, %40, %42, %39 : f32, f32, f32, f32, f32, f32
      }
      %28 = arith.andi %21, %c1_i32 : i32
      %29 = arith.cmpi eq, %28, %c0_i32 : i32
      %30 = arith.cmpi ne, %28, %c0_i32 : i32
      %31 = scf.if %29 -> (f32) {
        %32 = arith.mulf %27#1, %cst_4 : f32
        %33 = arith.mulf %27#0, %cst_3 : f32
        %34 = arith.subf %32, %33 : f32
        scf.yield %34 : f32
      } else {
        scf.yield %22#2 : f32
      }
      scf.yield %30, %31 : i1, f32
    } else {
      scf.yield %false, %22#2 : i1, f32
    }
    %24 = arith.select %23#0, %22#0, %23#1 : f32
    return %24 : f32
  }
  func.func @_ZSt5isinff(%arg0: f32) -> i8 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant 0x7F800000 : f32
    %0 = math.absf %arg0 : f32
    %1 = arith.cmpf oeq, %0, %cst : f32
    %2 = arith.extui %1 : i1 to i8
    return %2 : i8
  }
  func.func @_ZNSt14numeric_limitsIfE9quiet_NaNEv() -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant 0x7FC00000 : f32
    return %cst : f32
  }
  func.func @_ZSt4sqrtf(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = math.sqrt %arg0 : f32
    return %0 : f32
  }
  func.func @_ZN3c106detail2piIfEET_v() -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant 3.14159274 : f32
    return %cst : f32
  }
  func.func @_ZNSt14numeric_limitsIfE7epsilonEv() -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant 1.1920929E-7 : f32
    return %cst : f32
  }
  func.func @_ZSt3absf(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = math.absf %arg0 : f32
    return %0 : f32
  }
}
