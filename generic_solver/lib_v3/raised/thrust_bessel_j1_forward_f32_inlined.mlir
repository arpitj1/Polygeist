module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global constant @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2RQ" : memref<8xf32> = dense<[620.836487, 256987.25, 0x4C9F4AA7, 2.21511598E+10, 4.74914121E+12, 7.8436961E+14, 8.95222367E+16, 5.32278637E+18]>
  memref.global constant @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2RP" : memref<4xf32> = dense<[0xCE5691E2, 4.52228284E+11, -7.27494227E+13, 3.68295727E+15]>
  memref.global constant @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2QQ" : memref<7xf32> = dense<[74.2373276, 1056.44885, 4986.41064, 9562.31933, 7997.0415, 2826.19287, 336.093597]>
  memref.global constant @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2QP" : memref<8xf32> = dense<[0.0510862581, 4.98213863, 75.8238297, 366.779602, 710.856323, 597.489624, 211.688751, 25.2070198]>
  memref.global constant @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2PQ" : memref<7xf32> = dense<[5.71323151E-4, 0.0688455924, 1.10514235, 5.07386398, 8.39985561, 5.20982838, 1.000000e+00]>
  memref.global constant @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2PP" : memref<7xf32> = dense<[7.62125593E-4, 0.0731397048, 1.12719607, 5.11207962, 8.42404556, 5.21451616, 1.000000e+00]>
  func.func @thrust_bessel_j1_forward_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e+00 : f32
    %cst_1 = arith.constant 14.6819706 : f32
    %cst_2 = arith.constant 49.2184563 : f32
    %cst_3 = arith.constant 2.3561945 : f32
    %cst_4 = arith.constant 0.797884583 : f32
    %true = arith.constant true
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    %2 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2RQ" : memref<8xf32>
    %3 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2RP" : memref<4xf32>
    %4 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2QQ" : memref<7xf32>
    %5 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2QP" : memref<8xf32>
    %6 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2PQ" : memref<7xf32>
    %7 = memref.get_global @"_Z17bessel_j1_forwardIfET_S0_@static@_ZZ17bessel_j1_forwardIfET_S0_E2PP" : memref<7xf32>
    affine.for %arg3 = 0 to %0 {
      %8 = affine.load %arg1[%arg3] : memref<?xf32>
      %9 = arith.cmpf olt, %8, %cst : f32
      %10 = arith.xori %9, %true : i1
      %11 = scf.if %9 -> (f32) {
        %17 = arith.negf %8 : f32
        %18 = arith.cmpf ole, %17, %cst_0 : f32
        %19 = scf.if %18 -> (f32) {
          %21 = arith.mulf %8, %8 : f32
          %22 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst) -> (f32) {
            %30 = arith.index_cast %arg4 : index to i8
            %31 = arith.mulf %arg5, %21 : f32
            %32 = arith.index_castui %30 : i8 to index
            %33 = memref.load %3[%32] : memref<4xf32>
            %34 = arith.addf %31, %33 : f32
            affine.yield %34 : f32
          }
          %23 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %cst) -> (f32) {
            %30 = arith.index_cast %arg4 : index to i8
            %31 = arith.mulf %arg5, %21 : f32
            %32 = arith.index_castui %30 : i8 to index
            %33 = memref.load %2[%32] : memref<8xf32>
            %34 = arith.addf %31, %33 : f32
            affine.yield %34 : f32
          }
          %24 = arith.divf %22, %23 : f32
          %25 = arith.mulf %24, %17 : f32
          %26 = arith.subf %21, %cst_1 : f32
          %27 = arith.mulf %25, %26 : f32
          %28 = arith.subf %21, %cst_2 : f32
          %29 = arith.mulf %27, %28 : f32
          scf.yield %29 : f32
        } else {
          %21 = arith.divf %cst_0, %17 : f32
          %22 = arith.mulf %21, %21 : f32
          %23 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst) -> (f32) {
            %39 = arith.index_cast %arg4 : index to i8
            %40 = arith.mulf %arg5, %22 : f32
            %41 = arith.index_castui %39 : i8 to index
            %42 = memref.load %7[%41] : memref<7xf32>
            %43 = arith.addf %40, %42 : f32
            affine.yield %43 : f32
          }
          %24 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst) -> (f32) {
            %39 = arith.index_cast %arg4 : index to i8
            %40 = arith.mulf %arg5, %22 : f32
            %41 = arith.index_castui %39 : i8 to index
            %42 = memref.load %6[%41] : memref<7xf32>
            %43 = arith.addf %40, %42 : f32
            affine.yield %43 : f32
          }
          %25 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %cst) -> (f32) {
            %39 = arith.index_cast %arg4 : index to i8
            %40 = arith.mulf %arg5, %22 : f32
            %41 = arith.index_castui %39 : i8 to index
            %42 = memref.load %5[%41] : memref<8xf32>
            %43 = arith.addf %40, %42 : f32
            affine.yield %43 : f32
          }
          %26 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst) -> (f32) {
            %39 = arith.index_cast %arg4 : index to i8
            %40 = arith.mulf %arg5, %22 : f32
            %41 = arith.index_castui %39 : i8 to index
            %42 = memref.load %4[%41] : memref<7xf32>
            %43 = arith.addf %40, %42 : f32
            affine.yield %43 : f32
          }
          %27 = arith.divf %23, %24 : f32
          %28 = arith.subf %17, %cst_3 : f32
          %29 = math.cos %28 : f32
          %30 = arith.mulf %27, %29 : f32
          %31 = arith.divf %25, %26 : f32
          %32 = arith.mulf %21, %31 : f32
          %33 = math.sin %28 : f32
          %34 = arith.mulf %32, %33 : f32
          %35 = arith.subf %30, %34 : f32
          %36 = arith.mulf %35, %cst_4 : f32
          %37 = math.sqrt %17 : f32
          %38 = arith.divf %36, %37 : f32
          scf.yield %38 : f32
        }
        %20 = arith.negf %19 : f32
        scf.yield %20 : f32
      } else {
        scf.yield %1 : f32
      }
      %12 = arith.cmpf ole, %8, %cst_0 : f32
      %13 = arith.andi %12, %10 : i1
      %14 = arith.xori %12, %true : i1
      %15 = arith.andi %14, %10 : i1
      %16 = scf.if %15 -> (f32) {
        %17 = arith.divf %cst_0, %8 : f32
        %18 = arith.mulf %17, %17 : f32
        %19 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst) -> (f32) {
          %35 = arith.index_cast %arg4 : index to i8
          %36 = arith.mulf %arg5, %18 : f32
          %37 = arith.index_castui %35 : i8 to index
          %38 = memref.load %7[%37] : memref<7xf32>
          %39 = arith.addf %36, %38 : f32
          affine.yield %39 : f32
        }
        %20 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst) -> (f32) {
          %35 = arith.index_cast %arg4 : index to i8
          %36 = arith.mulf %arg5, %18 : f32
          %37 = arith.index_castui %35 : i8 to index
          %38 = memref.load %6[%37] : memref<7xf32>
          %39 = arith.addf %36, %38 : f32
          affine.yield %39 : f32
        }
        %21 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %cst) -> (f32) {
          %35 = arith.index_cast %arg4 : index to i8
          %36 = arith.mulf %arg5, %18 : f32
          %37 = arith.index_castui %35 : i8 to index
          %38 = memref.load %5[%37] : memref<8xf32>
          %39 = arith.addf %36, %38 : f32
          affine.yield %39 : f32
        }
        %22 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst) -> (f32) {
          %35 = arith.index_cast %arg4 : index to i8
          %36 = arith.mulf %arg5, %18 : f32
          %37 = arith.index_castui %35 : i8 to index
          %38 = memref.load %4[%37] : memref<7xf32>
          %39 = arith.addf %36, %38 : f32
          affine.yield %39 : f32
        }
        %23 = arith.divf %19, %20 : f32
        %24 = arith.subf %8, %cst_3 : f32
        %25 = math.cos %24 : f32
        %26 = arith.mulf %23, %25 : f32
        %27 = arith.divf %21, %22 : f32
        %28 = arith.mulf %17, %27 : f32
        %29 = math.sin %24 : f32
        %30 = arith.mulf %28, %29 : f32
        %31 = arith.subf %26, %30 : f32
        %32 = arith.mulf %31, %cst_4 : f32
        %33 = math.sqrt %8 : f32
        %34 = arith.divf %32, %33 : f32
        scf.yield %34 : f32
      } else {
        %17 = scf.if %13 -> (f32) {
          %18 = arith.mulf %8, %8 : f32
          %19 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst) -> (f32) {
            %27 = arith.index_cast %arg4 : index to i8
            %28 = arith.mulf %arg5, %18 : f32
            %29 = arith.index_castui %27 : i8 to index
            %30 = memref.load %3[%29] : memref<4xf32>
            %31 = arith.addf %28, %30 : f32
            affine.yield %31 : f32
          }
          %20 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %cst) -> (f32) {
            %27 = arith.index_cast %arg4 : index to i8
            %28 = arith.mulf %arg5, %18 : f32
            %29 = arith.index_castui %27 : i8 to index
            %30 = memref.load %2[%29] : memref<8xf32>
            %31 = arith.addf %28, %30 : f32
            affine.yield %31 : f32
          }
          %21 = arith.divf %19, %20 : f32
          %22 = arith.mulf %21, %8 : f32
          %23 = arith.subf %18, %cst_1 : f32
          %24 = arith.mulf %22, %23 : f32
          %25 = arith.subf %18, %cst_2 : f32
          %26 = arith.mulf %24, %25 : f32
          scf.yield %26 : f32
        } else {
          scf.yield %11 : f32
        }
        scf.yield %17 : f32
      }
      affine.store %16, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
}

