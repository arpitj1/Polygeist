module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global constant @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2RQ" : memref<8xf32> = dense<[620.836487, 256987.25, 0x4C9F4AA7, 2.21511598E+10, 4.74914121E+12, 7.8436961E+14, 8.95222367E+16, 5.32278637E+18]>
  memref.global constant @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2RP" : memref<4xf32> = dense<[0xCE5691E2, 4.52228284E+11, -7.27494227E+13, 3.68295727E+15]>
  memref.global constant @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2QQ" : memref<7xf32> = dense<[74.2373276, 1056.44885, 4986.41064, 9562.31933, 7997.0415, 2826.19287, 336.093597]>
  memref.global constant @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2QP" : memref<8xf32> = dense<[0.0510862581, 4.98213863, 75.8238297, 366.779602, 710.856323, 597.489624, 211.688751, 25.2070198]>
  memref.global constant @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2PQ" : memref<7xf32> = dense<[5.71323151E-4, 0.0688455924, 1.10514235, 5.07386398, 8.39985561, 5.20982838, 1.000000e+00]>
  memref.global constant @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2PP" : memref<7xf32> = dense<[7.62125593E-4, 0.0731397048, 1.12719607, 5.11207962, 8.42404556, 5.21451616, 1.000000e+00]>
  memref.global constant @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2YQ" : memref<8xf32> = dense<[594.301575, 235564.094, 0x4C8C277D, 1.87601326E+10, 3.88231279E+12, 6.20557745E+14, 6.8714108E+16, 3.97270621E+18]>
  memref.global constant @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2YP" : memref<6xf32> = dense<[1.26320474E+9, -6.47355891E+11, 1.14509516E+14, -8.1277027E+15, 2.0243947E+17, -7.788772E+17]>
  memref.global constant @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2QQ" : memref<7xf32> = dense<[74.2373276, 1056.44885, 4986.41064, 9562.31933, 7997.0415, 2826.19287, 336.093597]>
  memref.global constant @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2QP" : memref<8xf32> = dense<[0.0510862581, 4.98213863, 75.8238297, 366.779602, 710.856323, 597.489624, 211.688751, 25.2070198]>
  memref.global constant @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2PQ" : memref<7xf32> = dense<[5.71323151E-4, 0.0688455924, 1.10514235, 5.07386398, 8.39985561, 5.20982838, 1.000000e+00]>
  memref.global constant @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2PP" : memref<7xf32> = dense<[7.62125593E-4, 0.0731397048, 1.12719607, 5.11207962, 8.42404556, 5.21451616, 1.000000e+00]>
  func.func @thrust_bessel_y1_forward_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 5.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0.636619746 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant 2.3561945 : f32
    %cst_4 = arith.constant 0.797884583 : f32
    %true = arith.constant true
    %cst_5 = arith.constant 0x7FC00000 : f32
    %cst_6 = arith.constant 49.2184563 : f32
    %cst_7 = arith.constant 14.6819706 : f32
    %cst_8 = arith.constant 0xFF800000 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    %2 = memref.get_global @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2YQ" : memref<8xf32>
    %3 = memref.get_global @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2YP" : memref<6xf32>
    %4 = memref.get_global @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2QQ" : memref<7xf32>
    %5 = memref.get_global @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2QP" : memref<8xf32>
    %6 = memref.get_global @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2PQ" : memref<7xf32>
    %7 = memref.get_global @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2PP" : memref<7xf32>
    %8 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2RQ" : memref<8xf32>
    %9 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2RP" : memref<4xf32>
    %10 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2QQ" : memref<7xf32>
    %11 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2QP" : memref<8xf32>
    %12 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2PQ" : memref<7xf32>
    %13 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2PP" : memref<7xf32>
    affine.for %arg3 = 0 to %0 {
      %14 = affine.load %arg1[%arg3] : memref<?xf32>
      %15 = arith.cmpf ole, %14, %cst : f32
      %16 = scf.if %15 -> (f32) {
        %17 = arith.cmpf oeq, %14, %cst_0 : f32
        %18 = arith.xori %17, %true : i1
        %19 = arith.select %17, %cst_8, %1 : f32
        %20 = arith.cmpf ole, %14, %cst_0 : f32
        %21 = arith.andi %20, %18 : i1
        %22 = arith.xori %21, %true : i1
        %23 = arith.andi %22, %18 : i1
        %24 = scf.if %23 -> (f32) {
          %25 = arith.mulf %14, %14 : f32
          %26 = affine.for %arg4 = 0 to 6 iter_args(%arg5 = %cst_0) -> (f32) {
            %38 = arith.index_cast %arg4 : index to i8
            %39 = arith.mulf %arg5, %25 : f32
            %40 = arith.index_castui %38 : i8 to index
            %41 = memref.load %3[%40] : memref<6xf32>
            %42 = arith.addf %39, %41 : f32
            affine.yield %42 : f32
          }
          %27 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %cst_0) -> (f32) {
            %38 = arith.index_cast %arg4 : index to i8
            %39 = arith.mulf %arg5, %25 : f32
            %40 = arith.index_castui %38 : i8 to index
            %41 = memref.load %2[%40] : memref<8xf32>
            %42 = arith.addf %39, %41 : f32
            affine.yield %42 : f32
          }
          %28 = arith.divf %26, %27 : f32
          %29 = arith.mulf %14, %28 : f32
          %30 = arith.cmpf olt, %14, %cst_0 : f32
          %31 = scf.if %30 -> (f32) {
            %38 = arith.negf %14 : f32
            %39 = arith.cmpf ole, %38, %cst : f32
            %40 = scf.if %39 -> (f32) {
              %42 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (f32) {
                %50 = arith.index_cast %arg4 : index to i8
                %51 = arith.mulf %arg5, %25 : f32
                %52 = arith.index_castui %50 : i8 to index
                %53 = memref.load %9[%52] : memref<4xf32>
                %54 = arith.addf %51, %53 : f32
                affine.yield %54 : f32
              }
              %43 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %cst_0) -> (f32) {
                %50 = arith.index_cast %arg4 : index to i8
                %51 = arith.mulf %arg5, %25 : f32
                %52 = arith.index_castui %50 : i8 to index
                %53 = memref.load %8[%52] : memref<8xf32>
                %54 = arith.addf %51, %53 : f32
                affine.yield %54 : f32
              }
              %44 = arith.divf %42, %43 : f32
              %45 = arith.mulf %44, %38 : f32
              %46 = arith.subf %25, %cst_7 : f32
              %47 = arith.mulf %45, %46 : f32
              %48 = arith.subf %25, %cst_6 : f32
              %49 = arith.mulf %47, %48 : f32
              scf.yield %49 : f32
            } else {
              %42 = arith.divf %cst, %38 : f32
              %43 = arith.mulf %42, %42 : f32
              %44 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst_0) -> (f32) {
                %60 = arith.index_cast %arg4 : index to i8
                %61 = arith.mulf %arg5, %43 : f32
                %62 = arith.index_castui %60 : i8 to index
                %63 = memref.load %13[%62] : memref<7xf32>
                %64 = arith.addf %61, %63 : f32
                affine.yield %64 : f32
              }
              %45 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst_0) -> (f32) {
                %60 = arith.index_cast %arg4 : index to i8
                %61 = arith.mulf %arg5, %43 : f32
                %62 = arith.index_castui %60 : i8 to index
                %63 = memref.load %12[%62] : memref<7xf32>
                %64 = arith.addf %61, %63 : f32
                affine.yield %64 : f32
              }
              %46 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %cst_0) -> (f32) {
                %60 = arith.index_cast %arg4 : index to i8
                %61 = arith.mulf %arg5, %43 : f32
                %62 = arith.index_castui %60 : i8 to index
                %63 = memref.load %11[%62] : memref<8xf32>
                %64 = arith.addf %61, %63 : f32
                affine.yield %64 : f32
              }
              %47 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst_0) -> (f32) {
                %60 = arith.index_cast %arg4 : index to i8
                %61 = arith.mulf %arg5, %43 : f32
                %62 = arith.index_castui %60 : i8 to index
                %63 = memref.load %10[%62] : memref<7xf32>
                %64 = arith.addf %61, %63 : f32
                affine.yield %64 : f32
              }
              %48 = arith.divf %44, %45 : f32
              %49 = arith.subf %38, %cst_3 : f32
              %50 = math.cos %49 : f32
              %51 = arith.mulf %48, %50 : f32
              %52 = arith.divf %46, %47 : f32
              %53 = arith.mulf %42, %52 : f32
              %54 = math.sin %49 : f32
              %55 = arith.mulf %53, %54 : f32
              %56 = arith.subf %51, %55 : f32
              %57 = arith.mulf %56, %cst_4 : f32
              %58 = math.sqrt %38 : f32
              %59 = arith.divf %57, %58 : f32
              scf.yield %59 : f32
            }
            %41 = arith.negf %40 : f32
            scf.yield %41 : f32
          } else {
            %38 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (f32) {
              %46 = arith.index_cast %arg4 : index to i8
              %47 = arith.mulf %arg5, %25 : f32
              %48 = arith.index_castui %46 : i8 to index
              %49 = memref.load %9[%48] : memref<4xf32>
              %50 = arith.addf %47, %49 : f32
              affine.yield %50 : f32
            }
            %39 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %cst_0) -> (f32) {
              %46 = arith.index_cast %arg4 : index to i8
              %47 = arith.mulf %arg5, %25 : f32
              %48 = arith.index_castui %46 : i8 to index
              %49 = memref.load %8[%48] : memref<8xf32>
              %50 = arith.addf %47, %49 : f32
              affine.yield %50 : f32
            }
            %40 = arith.divf %38, %39 : f32
            %41 = arith.mulf %40, %14 : f32
            %42 = arith.subf %25, %cst_7 : f32
            %43 = arith.mulf %41, %42 : f32
            %44 = arith.subf %25, %cst_6 : f32
            %45 = arith.mulf %43, %44 : f32
            scf.yield %45 : f32
          }
          %32 = math.log %14 : f32
          %33 = arith.mulf %31, %32 : f32
          %34 = arith.divf %cst_2, %14 : f32
          %35 = arith.subf %33, %34 : f32
          %36 = arith.mulf %35, %cst_1 : f32
          %37 = arith.addf %29, %36 : f32
          scf.yield %37 : f32
        } else {
          %25 = arith.select %21, %cst_5, %19 : f32
          scf.yield %25 : f32
        }
        scf.yield %24 : f32
      } else {
        %17 = arith.divf %cst, %14 : f32
        %18 = arith.mulf %17, %17 : f32
        %19 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst_0) -> (f32) {
          %35 = arith.index_cast %arg4 : index to i8
          %36 = arith.mulf %arg5, %18 : f32
          %37 = arith.index_castui %35 : i8 to index
          %38 = memref.load %7[%37] : memref<7xf32>
          %39 = arith.addf %36, %38 : f32
          affine.yield %39 : f32
        }
        %20 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst_0) -> (f32) {
          %35 = arith.index_cast %arg4 : index to i8
          %36 = arith.mulf %arg5, %18 : f32
          %37 = arith.index_castui %35 : i8 to index
          %38 = memref.load %6[%37] : memref<7xf32>
          %39 = arith.addf %36, %38 : f32
          affine.yield %39 : f32
        }
        %21 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %cst_0) -> (f32) {
          %35 = arith.index_cast %arg4 : index to i8
          %36 = arith.mulf %arg5, %18 : f32
          %37 = arith.index_castui %35 : i8 to index
          %38 = memref.load %5[%37] : memref<8xf32>
          %39 = arith.addf %36, %38 : f32
          affine.yield %39 : f32
        }
        %22 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst_0) -> (f32) {
          %35 = arith.index_cast %arg4 : index to i8
          %36 = arith.mulf %arg5, %18 : f32
          %37 = arith.index_castui %35 : i8 to index
          %38 = memref.load %4[%37] : memref<7xf32>
          %39 = arith.addf %36, %38 : f32
          affine.yield %39 : f32
        }
        %23 = arith.divf %19, %20 : f32
        %24 = arith.subf %14, %cst_3 : f32
        %25 = math.sin %24 : f32
        %26 = arith.mulf %23, %25 : f32
        %27 = arith.divf %21, %22 : f32
        %28 = arith.mulf %17, %27 : f32
        %29 = math.cos %24 : f32
        %30 = arith.mulf %28, %29 : f32
        %31 = arith.addf %26, %30 : f32
        %32 = arith.mulf %31, %cst_4 : f32
        %33 = math.sqrt %14 : f32
        %34 = arith.divf %32, %33 : f32
        scf.yield %34 : f32
      }
      affine.store %16, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
  func.func @bessel_y1_forwardf(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 14.6819706 : f32
    %cst_1 = arith.constant 49.2184563 : f32
    %cst_2 = arith.constant 0x7FC00000 : f32
    %true = arith.constant true
    %cst_3 = arith.constant 0.797884583 : f32
    %cst_4 = arith.constant 2.3561945 : f32
    %cst_5 = arith.constant 1.000000e+00 : f32
    %cst_6 = arith.constant 0.636619746 : f32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %cst_8 = arith.constant 5.000000e+00 : f32
    %0 = llvm.mlir.undef : f32
    %1 = memref.get_global @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2YQ" : memref<8xf32>
    %2 = memref.get_global @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2YP" : memref<6xf32>
    %3 = memref.get_global @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2QQ" : memref<7xf32>
    %4 = memref.get_global @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2QP" : memref<8xf32>
    %5 = memref.get_global @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2PQ" : memref<7xf32>
    %6 = memref.get_global @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2PP" : memref<7xf32>
    %7 = arith.cmpf ole, %arg0, %cst_8 : f32
    %8 = scf.if %7 -> (f32) {
      %9 = arith.cmpf oeq, %arg0, %cst_7 : f32
      %10 = arith.xori %9, %true : i1
      %11 = arith.select %9, %cst, %0 : f32
      %12 = arith.cmpf ole, %arg0, %cst_7 : f32
      %13 = arith.andi %12, %10 : i1
      %14 = arith.xori %13, %true : i1
      %15 = arith.andi %14, %10 : i1
      %16 = scf.if %15 -> (f32) {
        %17 = arith.mulf %arg0, %arg0 : f32
        %18 = affine.for %arg1 = 0 to 6 iter_args(%arg2 = %cst_7) -> (f32) {
          %32 = arith.index_cast %arg1 : index to i8
          %33 = arith.mulf %arg2, %17 : f32
          %34 = arith.index_castui %32 : i8 to index
          %35 = memref.load %2[%34] : memref<6xf32>
          %36 = arith.addf %33, %35 : f32
          affine.yield %36 : f32
        }
        %19 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %cst_7) -> (f32) {
          %32 = arith.index_cast %arg1 : index to i8
          %33 = arith.mulf %arg2, %17 : f32
          %34 = arith.index_castui %32 : i8 to index
          %35 = memref.load %1[%34] : memref<8xf32>
          %36 = arith.addf %33, %35 : f32
          affine.yield %36 : f32
        }
        %20 = arith.divf %18, %19 : f32
        %21 = arith.mulf %arg0, %20 : f32
        %22 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2RQ" : memref<8xf32>
        %23 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2RP" : memref<4xf32>
        %24 = arith.cmpf olt, %arg0, %cst_7 : f32
        %25 = scf.if %24 -> (f32) {
          %32 = arith.negf %arg0 : f32
          %33 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2QQ" : memref<7xf32>
          %34 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2QP" : memref<8xf32>
          %35 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2PQ" : memref<7xf32>
          %36 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2PP" : memref<7xf32>
          %37 = arith.cmpf ole, %32, %cst_8 : f32
          %38 = scf.if %37 -> (f32) {
            %40 = affine.for %arg1 = 0 to 4 iter_args(%arg2 = %cst_7) -> (f32) {
              %48 = arith.index_cast %arg1 : index to i8
              %49 = arith.mulf %arg2, %17 : f32
              %50 = arith.index_castui %48 : i8 to index
              %51 = memref.load %23[%50] : memref<4xf32>
              %52 = arith.addf %49, %51 : f32
              affine.yield %52 : f32
            }
            %41 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %cst_7) -> (f32) {
              %48 = arith.index_cast %arg1 : index to i8
              %49 = arith.mulf %arg2, %17 : f32
              %50 = arith.index_castui %48 : i8 to index
              %51 = memref.load %22[%50] : memref<8xf32>
              %52 = arith.addf %49, %51 : f32
              affine.yield %52 : f32
            }
            %42 = arith.divf %40, %41 : f32
            %43 = arith.mulf %42, %32 : f32
            %44 = arith.subf %17, %cst_0 : f32
            %45 = arith.mulf %43, %44 : f32
            %46 = arith.subf %17, %cst_1 : f32
            %47 = arith.mulf %45, %46 : f32
            scf.yield %47 : f32
          } else {
            %40 = arith.divf %cst_8, %32 : f32
            %41 = arith.mulf %40, %40 : f32
            %42 = affine.for %arg1 = 0 to 7 iter_args(%arg2 = %cst_7) -> (f32) {
              %58 = arith.index_cast %arg1 : index to i8
              %59 = arith.mulf %arg2, %41 : f32
              %60 = arith.index_castui %58 : i8 to index
              %61 = memref.load %36[%60] : memref<7xf32>
              %62 = arith.addf %59, %61 : f32
              affine.yield %62 : f32
            }
            %43 = affine.for %arg1 = 0 to 7 iter_args(%arg2 = %cst_7) -> (f32) {
              %58 = arith.index_cast %arg1 : index to i8
              %59 = arith.mulf %arg2, %41 : f32
              %60 = arith.index_castui %58 : i8 to index
              %61 = memref.load %35[%60] : memref<7xf32>
              %62 = arith.addf %59, %61 : f32
              affine.yield %62 : f32
            }
            %44 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %cst_7) -> (f32) {
              %58 = arith.index_cast %arg1 : index to i8
              %59 = arith.mulf %arg2, %41 : f32
              %60 = arith.index_castui %58 : i8 to index
              %61 = memref.load %34[%60] : memref<8xf32>
              %62 = arith.addf %59, %61 : f32
              affine.yield %62 : f32
            }
            %45 = affine.for %arg1 = 0 to 7 iter_args(%arg2 = %cst_7) -> (f32) {
              %58 = arith.index_cast %arg1 : index to i8
              %59 = arith.mulf %arg2, %41 : f32
              %60 = arith.index_castui %58 : i8 to index
              %61 = memref.load %33[%60] : memref<7xf32>
              %62 = arith.addf %59, %61 : f32
              affine.yield %62 : f32
            }
            %46 = arith.divf %42, %43 : f32
            %47 = arith.subf %32, %cst_4 : f32
            %48 = math.cos %47 : f32
            %49 = arith.mulf %46, %48 : f32
            %50 = arith.divf %44, %45 : f32
            %51 = arith.mulf %40, %50 : f32
            %52 = math.sin %47 : f32
            %53 = arith.mulf %51, %52 : f32
            %54 = arith.subf %49, %53 : f32
            %55 = arith.mulf %54, %cst_3 : f32
            %56 = math.sqrt %32 : f32
            %57 = arith.divf %55, %56 : f32
            scf.yield %57 : f32
          }
          %39 = arith.negf %38 : f32
          scf.yield %39 : f32
        } else {
          %32 = affine.for %arg1 = 0 to 4 iter_args(%arg2 = %cst_7) -> (f32) {
            %40 = arith.index_cast %arg1 : index to i8
            %41 = arith.mulf %arg2, %17 : f32
            %42 = arith.index_castui %40 : i8 to index
            %43 = memref.load %23[%42] : memref<4xf32>
            %44 = arith.addf %41, %43 : f32
            affine.yield %44 : f32
          }
          %33 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %cst_7) -> (f32) {
            %40 = arith.index_cast %arg1 : index to i8
            %41 = arith.mulf %arg2, %17 : f32
            %42 = arith.index_castui %40 : i8 to index
            %43 = memref.load %22[%42] : memref<8xf32>
            %44 = arith.addf %41, %43 : f32
            affine.yield %44 : f32
          }
          %34 = arith.divf %32, %33 : f32
          %35 = arith.mulf %34, %arg0 : f32
          %36 = arith.subf %17, %cst_0 : f32
          %37 = arith.mulf %35, %36 : f32
          %38 = arith.subf %17, %cst_1 : f32
          %39 = arith.mulf %37, %38 : f32
          scf.yield %39 : f32
        }
        %26 = math.log %arg0 : f32
        %27 = arith.mulf %25, %26 : f32
        %28 = arith.divf %cst_5, %arg0 : f32
        %29 = arith.subf %27, %28 : f32
        %30 = arith.mulf %29, %cst_6 : f32
        %31 = arith.addf %21, %30 : f32
        scf.yield %31 : f32
      } else {
        %17 = arith.select %13, %cst_2, %11 : f32
        scf.yield %17 : f32
      }
      scf.yield %16 : f32
    } else {
      %9 = arith.divf %cst_8, %arg0 : f32
      %10 = arith.mulf %9, %9 : f32
      %11 = affine.for %arg1 = 0 to 7 iter_args(%arg2 = %cst_7) -> (f32) {
        %27 = arith.index_cast %arg1 : index to i8
        %28 = arith.mulf %arg2, %10 : f32
        %29 = arith.index_castui %27 : i8 to index
        %30 = memref.load %6[%29] : memref<7xf32>
        %31 = arith.addf %28, %30 : f32
        affine.yield %31 : f32
      }
      %12 = affine.for %arg1 = 0 to 7 iter_args(%arg2 = %cst_7) -> (f32) {
        %27 = arith.index_cast %arg1 : index to i8
        %28 = arith.mulf %arg2, %10 : f32
        %29 = arith.index_castui %27 : i8 to index
        %30 = memref.load %5[%29] : memref<7xf32>
        %31 = arith.addf %28, %30 : f32
        affine.yield %31 : f32
      }
      %13 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %cst_7) -> (f32) {
        %27 = arith.index_cast %arg1 : index to i8
        %28 = arith.mulf %arg2, %10 : f32
        %29 = arith.index_castui %27 : i8 to index
        %30 = memref.load %4[%29] : memref<8xf32>
        %31 = arith.addf %28, %30 : f32
        affine.yield %31 : f32
      }
      %14 = affine.for %arg1 = 0 to 7 iter_args(%arg2 = %cst_7) -> (f32) {
        %27 = arith.index_cast %arg1 : index to i8
        %28 = arith.mulf %arg2, %10 : f32
        %29 = arith.index_castui %27 : i8 to index
        %30 = memref.load %3[%29] : memref<7xf32>
        %31 = arith.addf %28, %30 : f32
        affine.yield %31 : f32
      }
      %15 = arith.divf %11, %12 : f32
      %16 = arith.subf %arg0, %cst_4 : f32
      %17 = math.sin %16 : f32
      %18 = arith.mulf %15, %17 : f32
      %19 = arith.divf %13, %14 : f32
      %20 = arith.mulf %9, %19 : f32
      %21 = math.cos %16 : f32
      %22 = arith.mulf %20, %21 : f32
      %23 = arith.addf %18, %22 : f32
      %24 = arith.mulf %23, %cst_3 : f32
      %25 = math.sqrt %arg0 : f32
      %26 = arith.divf %24, %25 : f32
      scf.yield %26 : f32
    }
    return %8 : f32
  }
  func.func @_Z17bessel_y1_forwardIfET_S0_(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 14.6819706 : f32
    %cst_1 = arith.constant 49.2184563 : f32
    %cst_2 = arith.constant 0x7FC00000 : f32
    %true = arith.constant true
    %cst_3 = arith.constant 0.797884583 : f32
    %cst_4 = arith.constant 2.3561945 : f32
    %cst_5 = arith.constant 1.000000e+00 : f32
    %cst_6 = arith.constant 0.636619746 : f32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %cst_8 = arith.constant 5.000000e+00 : f32
    %0 = llvm.mlir.undef : f32
    %1 = memref.get_global @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2YQ" : memref<8xf32>
    %2 = memref.get_global @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2YP" : memref<6xf32>
    %3 = memref.get_global @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2QQ" : memref<7xf32>
    %4 = memref.get_global @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2QP" : memref<8xf32>
    %5 = memref.get_global @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2PQ" : memref<7xf32>
    %6 = memref.get_global @"_Z17bessel_y1_forwardIfET_S0_@static@_ZZ17bessel_y1_forwardIfET_S0_E2PP" : memref<7xf32>
    %7 = arith.cmpf ole, %arg0, %cst_8 : f32
    %8 = scf.if %7 -> (f32) {
      %9 = arith.cmpf oeq, %arg0, %cst_7 : f32
      %10 = arith.xori %9, %true : i1
      %11 = arith.select %9, %cst, %0 : f32
      %12 = arith.cmpf ole, %arg0, %cst_7 : f32
      %13 = arith.andi %12, %10 : i1
      %14 = arith.xori %13, %true : i1
      %15 = arith.andi %14, %10 : i1
      %16 = scf.if %15 -> (f32) {
        %17 = arith.mulf %arg0, %arg0 : f32
        %18 = affine.for %arg1 = 0 to 6 iter_args(%arg2 = %cst_7) -> (f32) {
          %32 = arith.index_cast %arg1 : index to i8
          %33 = arith.mulf %arg2, %17 : f32
          %34 = arith.index_castui %32 : i8 to index
          %35 = memref.load %2[%34] : memref<6xf32>
          %36 = arith.addf %33, %35 : f32
          affine.yield %36 : f32
        }
        %19 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %cst_7) -> (f32) {
          %32 = arith.index_cast %arg1 : index to i8
          %33 = arith.mulf %arg2, %17 : f32
          %34 = arith.index_castui %32 : i8 to index
          %35 = memref.load %1[%34] : memref<8xf32>
          %36 = arith.addf %33, %35 : f32
          affine.yield %36 : f32
        }
        %20 = arith.divf %18, %19 : f32
        %21 = arith.mulf %arg0, %20 : f32
        %22 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2RQ" : memref<8xf32>
        %23 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2RP" : memref<4xf32>
        %24 = arith.cmpf olt, %arg0, %cst_7 : f32
        %25 = scf.if %24 -> (f32) {
          %32 = arith.negf %arg0 : f32
          %33 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2QQ" : memref<7xf32>
          %34 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2QP" : memref<8xf32>
          %35 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2PQ" : memref<7xf32>
          %36 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2PP" : memref<7xf32>
          %37 = arith.cmpf ole, %32, %cst_8 : f32
          %38 = scf.if %37 -> (f32) {
            %40 = affine.for %arg1 = 0 to 4 iter_args(%arg2 = %cst_7) -> (f32) {
              %48 = arith.index_cast %arg1 : index to i8
              %49 = arith.mulf %arg2, %17 : f32
              %50 = arith.index_castui %48 : i8 to index
              %51 = memref.load %23[%50] : memref<4xf32>
              %52 = arith.addf %49, %51 : f32
              affine.yield %52 : f32
            }
            %41 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %cst_7) -> (f32) {
              %48 = arith.index_cast %arg1 : index to i8
              %49 = arith.mulf %arg2, %17 : f32
              %50 = arith.index_castui %48 : i8 to index
              %51 = memref.load %22[%50] : memref<8xf32>
              %52 = arith.addf %49, %51 : f32
              affine.yield %52 : f32
            }
            %42 = arith.divf %40, %41 : f32
            %43 = arith.mulf %42, %32 : f32
            %44 = arith.subf %17, %cst_0 : f32
            %45 = arith.mulf %43, %44 : f32
            %46 = arith.subf %17, %cst_1 : f32
            %47 = arith.mulf %45, %46 : f32
            scf.yield %47 : f32
          } else {
            %40 = arith.divf %cst_8, %32 : f32
            %41 = arith.mulf %40, %40 : f32
            %42 = affine.for %arg1 = 0 to 7 iter_args(%arg2 = %cst_7) -> (f32) {
              %58 = arith.index_cast %arg1 : index to i8
              %59 = arith.mulf %arg2, %41 : f32
              %60 = arith.index_castui %58 : i8 to index
              %61 = memref.load %36[%60] : memref<7xf32>
              %62 = arith.addf %59, %61 : f32
              affine.yield %62 : f32
            }
            %43 = affine.for %arg1 = 0 to 7 iter_args(%arg2 = %cst_7) -> (f32) {
              %58 = arith.index_cast %arg1 : index to i8
              %59 = arith.mulf %arg2, %41 : f32
              %60 = arith.index_castui %58 : i8 to index
              %61 = memref.load %35[%60] : memref<7xf32>
              %62 = arith.addf %59, %61 : f32
              affine.yield %62 : f32
            }
            %44 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %cst_7) -> (f32) {
              %58 = arith.index_cast %arg1 : index to i8
              %59 = arith.mulf %arg2, %41 : f32
              %60 = arith.index_castui %58 : i8 to index
              %61 = memref.load %34[%60] : memref<8xf32>
              %62 = arith.addf %59, %61 : f32
              affine.yield %62 : f32
            }
            %45 = affine.for %arg1 = 0 to 7 iter_args(%arg2 = %cst_7) -> (f32) {
              %58 = arith.index_cast %arg1 : index to i8
              %59 = arith.mulf %arg2, %41 : f32
              %60 = arith.index_castui %58 : i8 to index
              %61 = memref.load %33[%60] : memref<7xf32>
              %62 = arith.addf %59, %61 : f32
              affine.yield %62 : f32
            }
            %46 = arith.divf %42, %43 : f32
            %47 = arith.subf %32, %cst_4 : f32
            %48 = math.cos %47 : f32
            %49 = arith.mulf %46, %48 : f32
            %50 = arith.divf %44, %45 : f32
            %51 = arith.mulf %40, %50 : f32
            %52 = math.sin %47 : f32
            %53 = arith.mulf %51, %52 : f32
            %54 = arith.subf %49, %53 : f32
            %55 = arith.mulf %54, %cst_3 : f32
            %56 = math.sqrt %32 : f32
            %57 = arith.divf %55, %56 : f32
            scf.yield %57 : f32
          }
          %39 = arith.negf %38 : f32
          scf.yield %39 : f32
        } else {
          %32 = affine.for %arg1 = 0 to 4 iter_args(%arg2 = %cst_7) -> (f32) {
            %40 = arith.index_cast %arg1 : index to i8
            %41 = arith.mulf %arg2, %17 : f32
            %42 = arith.index_castui %40 : i8 to index
            %43 = memref.load %23[%42] : memref<4xf32>
            %44 = arith.addf %41, %43 : f32
            affine.yield %44 : f32
          }
          %33 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %cst_7) -> (f32) {
            %40 = arith.index_cast %arg1 : index to i8
            %41 = arith.mulf %arg2, %17 : f32
            %42 = arith.index_castui %40 : i8 to index
            %43 = memref.load %22[%42] : memref<8xf32>
            %44 = arith.addf %41, %43 : f32
            affine.yield %44 : f32
          }
          %34 = arith.divf %32, %33 : f32
          %35 = arith.mulf %34, %arg0 : f32
          %36 = arith.subf %17, %cst_0 : f32
          %37 = arith.mulf %35, %36 : f32
          %38 = arith.subf %17, %cst_1 : f32
          %39 = arith.mulf %37, %38 : f32
          scf.yield %39 : f32
        }
        %26 = math.log %arg0 : f32
        %27 = arith.mulf %25, %26 : f32
        %28 = arith.divf %cst_5, %arg0 : f32
        %29 = arith.subf %27, %28 : f32
        %30 = arith.mulf %29, %cst_6 : f32
        %31 = arith.addf %21, %30 : f32
        scf.yield %31 : f32
      } else {
        %17 = arith.select %13, %cst_2, %11 : f32
        scf.yield %17 : f32
      }
      scf.yield %16 : f32
    } else {
      %9 = arith.divf %cst_8, %arg0 : f32
      %10 = arith.mulf %9, %9 : f32
      %11 = affine.for %arg1 = 0 to 7 iter_args(%arg2 = %cst_7) -> (f32) {
        %27 = arith.index_cast %arg1 : index to i8
        %28 = arith.mulf %arg2, %10 : f32
        %29 = arith.index_castui %27 : i8 to index
        %30 = memref.load %6[%29] : memref<7xf32>
        %31 = arith.addf %28, %30 : f32
        affine.yield %31 : f32
      }
      %12 = affine.for %arg1 = 0 to 7 iter_args(%arg2 = %cst_7) -> (f32) {
        %27 = arith.index_cast %arg1 : index to i8
        %28 = arith.mulf %arg2, %10 : f32
        %29 = arith.index_castui %27 : i8 to index
        %30 = memref.load %5[%29] : memref<7xf32>
        %31 = arith.addf %28, %30 : f32
        affine.yield %31 : f32
      }
      %13 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %cst_7) -> (f32) {
        %27 = arith.index_cast %arg1 : index to i8
        %28 = arith.mulf %arg2, %10 : f32
        %29 = arith.index_castui %27 : i8 to index
        %30 = memref.load %4[%29] : memref<8xf32>
        %31 = arith.addf %28, %30 : f32
        affine.yield %31 : f32
      }
      %14 = affine.for %arg1 = 0 to 7 iter_args(%arg2 = %cst_7) -> (f32) {
        %27 = arith.index_cast %arg1 : index to i8
        %28 = arith.mulf %arg2, %10 : f32
        %29 = arith.index_castui %27 : i8 to index
        %30 = memref.load %3[%29] : memref<7xf32>
        %31 = arith.addf %28, %30 : f32
        affine.yield %31 : f32
      }
      %15 = arith.divf %11, %12 : f32
      %16 = arith.subf %arg0, %cst_4 : f32
      %17 = math.sin %16 : f32
      %18 = arith.mulf %15, %17 : f32
      %19 = arith.divf %13, %14 : f32
      %20 = arith.mulf %9, %19 : f32
      %21 = math.cos %16 : f32
      %22 = arith.mulf %20, %21 : f32
      %23 = arith.addf %18, %22 : f32
      %24 = arith.mulf %23, %cst_3 : f32
      %25 = math.sqrt %arg0 : f32
      %26 = arith.divf %24, %25 : f32
      scf.yield %26 : f32
    }
    return %8 : f32
  }
  func.func @_ZNSt14numeric_limitsIfE8infinityEv() -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant 0x7F800000 : f32
    return %cst : f32
  }
  func.func @_ZNSt14numeric_limitsIfE9quiet_NaNEv() -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant 0x7FC00000 : f32
    return %cst : f32
  }
  func.func @_Z17bessel_j1_forwardIfET_S0_(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %true = arith.constant true
    %cst = arith.constant 0.797884583 : f32
    %cst_0 = arith.constant 2.3561945 : f32
    %cst_1 = arith.constant 49.2184563 : f32
    %cst_2 = arith.constant 14.6819706 : f32
    %cst_3 = arith.constant 5.000000e+00 : f32
    %cst_4 = arith.constant 0.000000e+00 : f32
    %0 = llvm.mlir.undef : f32
    %1 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2RQ" : memref<8xf32>
    %2 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2RP" : memref<4xf32>
    %3 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2QQ" : memref<7xf32>
    %4 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2QP" : memref<8xf32>
    %5 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2PQ" : memref<7xf32>
    %6 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2PP" : memref<7xf32>
    %7 = arith.cmpf olt, %arg0, %cst_4 : f32
    %8 = arith.xori %7, %true : i1
    %9 = scf.if %7 -> (f32) {
      %15 = arith.negf %arg0 : f32
      %16 = func.call @_Z17bessel_j1_forwardIfET_S0_(%15) : (f32) -> f32
      %17 = arith.negf %16 : f32
      scf.yield %17 : f32
    } else {
      scf.yield %0 : f32
    }
    %10 = arith.cmpf ole, %arg0, %cst_3 : f32
    %11 = arith.andi %10, %8 : i1
    %12 = arith.xori %10, %true : i1
    %13 = arith.andi %12, %8 : i1
    %14 = scf.if %13 -> (f32) {
      %15 = arith.divf %cst_3, %arg0 : f32
      %16 = arith.mulf %15, %15 : f32
      %17 = affine.for %arg1 = 0 to 7 iter_args(%arg2 = %cst_4) -> (f32) {
        %33 = arith.index_cast %arg1 : index to i8
        %34 = arith.mulf %arg2, %16 : f32
        %35 = arith.index_castui %33 : i8 to index
        %36 = memref.load %6[%35] : memref<7xf32>
        %37 = arith.addf %34, %36 : f32
        affine.yield %37 : f32
      }
      %18 = affine.for %arg1 = 0 to 7 iter_args(%arg2 = %cst_4) -> (f32) {
        %33 = arith.index_cast %arg1 : index to i8
        %34 = arith.mulf %arg2, %16 : f32
        %35 = arith.index_castui %33 : i8 to index
        %36 = memref.load %5[%35] : memref<7xf32>
        %37 = arith.addf %34, %36 : f32
        affine.yield %37 : f32
      }
      %19 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %cst_4) -> (f32) {
        %33 = arith.index_cast %arg1 : index to i8
        %34 = arith.mulf %arg2, %16 : f32
        %35 = arith.index_castui %33 : i8 to index
        %36 = memref.load %4[%35] : memref<8xf32>
        %37 = arith.addf %34, %36 : f32
        affine.yield %37 : f32
      }
      %20 = affine.for %arg1 = 0 to 7 iter_args(%arg2 = %cst_4) -> (f32) {
        %33 = arith.index_cast %arg1 : index to i8
        %34 = arith.mulf %arg2, %16 : f32
        %35 = arith.index_castui %33 : i8 to index
        %36 = memref.load %3[%35] : memref<7xf32>
        %37 = arith.addf %34, %36 : f32
        affine.yield %37 : f32
      }
      %21 = arith.divf %17, %18 : f32
      %22 = arith.subf %arg0, %cst_0 : f32
      %23 = math.cos %22 : f32
      %24 = arith.mulf %21, %23 : f32
      %25 = arith.divf %19, %20 : f32
      %26 = arith.mulf %15, %25 : f32
      %27 = math.sin %22 : f32
      %28 = arith.mulf %26, %27 : f32
      %29 = arith.subf %24, %28 : f32
      %30 = arith.mulf %29, %cst : f32
      %31 = math.sqrt %arg0 : f32
      %32 = arith.divf %30, %31 : f32
      scf.yield %32 : f32
    } else {
      %15 = scf.if %11 -> (f32) {
        %16 = arith.mulf %arg0, %arg0 : f32
        %17 = affine.for %arg1 = 0 to 4 iter_args(%arg2 = %cst_4) -> (f32) {
          %25 = arith.index_cast %arg1 : index to i8
          %26 = arith.mulf %arg2, %16 : f32
          %27 = arith.index_castui %25 : i8 to index
          %28 = memref.load %2[%27] : memref<4xf32>
          %29 = arith.addf %26, %28 : f32
          affine.yield %29 : f32
        }
        %18 = affine.for %arg1 = 0 to 8 iter_args(%arg2 = %cst_4) -> (f32) {
          %25 = arith.index_cast %arg1 : index to i8
          %26 = arith.mulf %arg2, %16 : f32
          %27 = arith.index_castui %25 : i8 to index
          %28 = memref.load %1[%27] : memref<8xf32>
          %29 = arith.addf %26, %28 : f32
          affine.yield %29 : f32
        }
        %19 = arith.divf %17, %18 : f32
        %20 = arith.mulf %19, %arg0 : f32
        %21 = arith.subf %16, %cst_2 : f32
        %22 = arith.mulf %20, %21 : f32
        %23 = arith.subf %16, %cst_1 : f32
        %24 = arith.mulf %22, %23 : f32
        scf.yield %24 : f32
      } else {
        scf.yield %9 : f32
      }
      scf.yield %15 : f32
    }
    return %14 : f32
  }
  func.func @_ZSt4sqrtf(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = math.sqrt %arg0 : f32
    return %0 : f32
  }
}
