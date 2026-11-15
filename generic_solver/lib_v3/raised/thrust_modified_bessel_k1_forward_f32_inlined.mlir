module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global constant @"_Z26modified_bessel_i1_forwardIfET_S0_@static@_ZZ26modified_bessel_i1_forwardIfET_S0_E1B" : memref<25xf32> = dense<[7.51729664E-18, 4.41434819E-18, -4.65030551E-17, -3.20952596E-17, 2.96262902E-16, 3.30820232E-16, -1.8803547E-15, -3.81440324E-15, 1.04202772E-14, 4.272440e-14, -2.10154177E-14, -4.08355099E-13, -7.19855165E-13, 2.03562861E-12, 1.41258073E-11, 3.25260374E-11, -1.89749588E-11, -5.58974367E-10, -3.83538046E-9, -2.63146891E-8, -2.51223611E-7, -3.88256467E-6, -1.10588939E-4, -0.00976109784, 0.778576254]>
  memref.global constant @"_Z26modified_bessel_i1_forwardIfET_S0_@static@_ZZ26modified_bessel_i1_forwardIfET_S0_E1A" : memref<29xf32> = dense<[2.77791403E-18, -2.1114212E-17, 1.55363193E-16, -1.10559695E-15, 7.60068413E-15, -5.04218554E-14, 3.2237933E-13, -1.9839744E-12, 1.17361867E-11, -6.663490e-11, 3.62559038E-10, -1.88724969E-9, 9.38153732E-9, -4.44505908E-8, 2.00329481E-7, -8.568720e-07, 3.47025139E-6, -1.32731639E-5, 4.78156508E-5, -1.61760821E-4, 5.122860e-04, -0.00151357241, 0.0041564228, -0.0105640851, 0.0247264486, -0.0529459827, 0.102643661, -0.176416516, 0.252587199]>
  memref.global constant @"_Z26modified_bessel_k1_forwardIfET_S0_@static@_ZZ26modified_bessel_k1_forwardIfET_S0_E1B" : memref<25xf32> = dense<[-5.75674445E-18, 1.79405085E-17, -5.68946273E-17, 1.83809352E-16, -6.05704737E-16, 2.03870306E-15, -7.01983722E-15, 2.47715446E-14, -8.97670541E-14, 3.34841963E-13, -1.28917395E-12, 5.13963958E-12, -2.1299679E-11, 9.218315E-11, -4.19035473E-10, 2.01504968E-9, -1.03457625E-8, 5.74108405E-8, -3.50196046E-7, 2.40648501E-6, -1.93619799E-5, 1.95215514E-4, -0.00285781687, 0.103923738, 2.72062612]>
  memref.global constant @"_Z26modified_bessel_k1_forwardIfET_S0_@static@_ZZ26modified_bessel_k1_forwardIfET_S0_E1A" : memref<11xf32> = dense<[-7.02386355E-18, -2.42744989E-15, -6.66690173E-13, -1.41148843E-10, -2.2133877E-8, -2.43340605E-6, -1.73028893E-4, -0.0069757239, -0.12261118, -0.353155971, 1.52530026]>
  func.func @thrust_modified_bessel_k1_forward_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 8.000000e+00 : f32
    %true = arith.constant true
    %cst_3 = arith.constant -5.75674445E-18 : f32
    %cst_4 = arith.constant -7.02386355E-18 : f32
    %cst_5 = arith.constant 0x7F800000 : f32
    %cst_6 = arith.constant 0x7FC00000 : f32
    %cst_7 = arith.constant 2.77791403E-18 : f32
    %cst_8 = arith.constant 7.51729664E-18 : f32
    %cst_9 = arith.constant 3.200000e+01 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    %2 = memref.get_global @"_Z26modified_bessel_k1_forwardIfET_S0_@static@_ZZ26modified_bessel_k1_forwardIfET_S0_E1B" : memref<25xf32>
    %3 = memref.get_global @"_Z26modified_bessel_k1_forwardIfET_S0_@static@_ZZ26modified_bessel_k1_forwardIfET_S0_E1A" : memref<11xf32>
    %4 = memref.get_global @"_Z26modified_bessel_i1_forwardIfET_S0_@static@_ZZ26modified_bessel_i1_forwardIfET_S0_E1B" : memref<25xf32>
    %5 = memref.get_global @"_Z26modified_bessel_i1_forwardIfET_S0_@static@_ZZ26modified_bessel_i1_forwardIfET_S0_E1A" : memref<29xf32>
    affine.for %arg3 = 0 to %0 {
      %6 = affine.load %arg1[%arg3] : memref<?xf32>
      %7 = arith.cmpf oeq, %6, %cst : f32
      %8 = arith.xori %7, %true : i1
      %9 = arith.select %7, %cst_5, %1 : f32
      %10 = arith.cmpf olt, %6, %cst : f32
      %11 = arith.andi %8, %10 : i1
      %12 = arith.xori %11, %true : i1
      %13 = arith.andi %12, %8 : i1
      %14 = arith.select %11, %cst_6, %9 : f32
      %15 = arith.cmpf ole, %6, %cst_0 : f32
      %16 = scf.if %13 -> (f32) {
        %17 = scf.if %15 -> (f32) {
          %18 = arith.mulf %6, %6 : f32
          %19 = arith.subf %18, %cst_0 : f32
          %20:3 = affine.for %arg4 = 1 to 11 iter_args(%arg5 = %cst_4, %arg6 = %cst, %arg7 = %cst) -> (f32, f32, f32) {
            %31 = arith.index_cast %arg4 : index to i8
            %32 = arith.mulf %19, %arg5 : f32
            %33 = arith.subf %32, %arg6 : f32
            %34 = arith.index_castui %31 : i8 to index
            %35 = memref.load %3[%34] : memref<11xf32>
            %36 = arith.addf %33, %35 : f32
            affine.yield %36, %arg5, %arg6 : f32, f32, f32
          }
          %21 = arith.mulf %6, %cst_1 : f32
          %22 = math.log %21 : f32
          %23 = math.absf %6 : f32
          %24 = arith.cmpf ole, %23, %cst_2 : f32
          %25 = scf.if %24 -> (f32) {
            %31 = arith.divf %23, %cst_0 : f32
            %32 = arith.subf %31, %cst_0 : f32
            %33:3 = affine.for %arg4 = 1 to 29 iter_args(%arg5 = %cst_7, %arg6 = %cst, %arg7 = %cst) -> (f32, f32, f32) {
              %35 = arith.index_cast %arg4 : index to i8
              %36 = arith.mulf %32, %arg5 : f32
              %37 = arith.subf %36, %arg6 : f32
              %38 = arith.index_castui %35 : i8 to index
              %39 = memref.load %5[%38] : memref<29xf32>
              %40 = arith.addf %37, %39 : f32
              affine.yield %40, %arg5, %arg6 : f32, f32, f32
            }
            %34 = scf.if %10 -> (f32) {
              %35 = arith.subf %33#0, %33#2 : f32
              %36 = arith.mulf %35, %cst_1 : f32
              %37 = arith.mulf %36, %23 : f32
              %38 = math.exp %23 : f32
              %39 = arith.mulf %37, %38 : f32
              %40 = arith.negf %39 : f32
              scf.yield %40 : f32
            } else {
              %35 = arith.subf %33#0, %33#2 : f32
              %36 = arith.mulf %35, %cst_1 : f32
              %37 = arith.mulf %36, %23 : f32
              %38 = math.exp %23 : f32
              %39 = arith.mulf %37, %38 : f32
              scf.yield %39 : f32
            }
            scf.yield %34 : f32
          } else {
            %31 = arith.divf %cst_9, %23 : f32
            %32 = arith.subf %31, %cst_0 : f32
            %33:3 = affine.for %arg4 = 1 to 25 iter_args(%arg5 = %cst_8, %arg6 = %cst, %arg7 = %cst) -> (f32, f32, f32) {
              %35 = arith.index_cast %arg4 : index to i8
              %36 = arith.mulf %32, %arg5 : f32
              %37 = arith.subf %36, %arg6 : f32
              %38 = arith.index_castui %35 : i8 to index
              %39 = memref.load %4[%38] : memref<25xf32>
              %40 = arith.addf %37, %39 : f32
              affine.yield %40, %arg5, %arg6 : f32, f32, f32
            }
            %34 = scf.if %10 -> (f32) {
              %35 = math.exp %23 : f32
              %36 = arith.subf %33#0, %33#2 : f32
              %37 = arith.mulf %36, %cst_1 : f32
              %38 = arith.mulf %35, %37 : f32
              %39 = math.sqrt %23 : f32
              %40 = arith.divf %38, %39 : f32
              %41 = arith.negf %40 : f32
              scf.yield %41 : f32
            } else {
              %35 = math.exp %23 : f32
              %36 = arith.subf %33#0, %33#2 : f32
              %37 = arith.mulf %36, %cst_1 : f32
              %38 = arith.mulf %35, %37 : f32
              %39 = math.sqrt %23 : f32
              %40 = arith.divf %38, %39 : f32
              scf.yield %40 : f32
            }
            scf.yield %34 : f32
          }
          %26 = arith.mulf %22, %25 : f32
          %27 = arith.subf %20#0, %20#2 : f32
          %28 = arith.mulf %27, %cst_1 : f32
          %29 = arith.divf %28, %6 : f32
          %30 = arith.addf %26, %29 : f32
          scf.yield %30 : f32
        } else {
          %18 = arith.divf %cst_2, %6 : f32
          %19 = arith.subf %18, %cst_0 : f32
          %20:3 = affine.for %arg4 = 1 to 25 iter_args(%arg5 = %cst_3, %arg6 = %cst, %arg7 = %cst) -> (f32, f32, f32) {
            %28 = arith.index_cast %arg4 : index to i8
            %29 = arith.mulf %19, %arg5 : f32
            %30 = arith.subf %29, %arg6 : f32
            %31 = arith.index_castui %28 : i8 to index
            %32 = memref.load %2[%31] : memref<25xf32>
            %33 = arith.addf %30, %32 : f32
            affine.yield %33, %arg5, %arg6 : f32, f32, f32
          }
          %21 = arith.negf %6 : f32
          %22 = math.exp %21 : f32
          %23 = arith.subf %20#0, %20#2 : f32
          %24 = arith.mulf %23, %cst_1 : f32
          %25 = arith.mulf %22, %24 : f32
          %26 = math.sqrt %6 : f32
          %27 = arith.divf %25, %26 : f32
          scf.yield %27 : f32
        }
        scf.yield %17 : f32
      } else {
        scf.yield %14 : f32
      }
      affine.store %16, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
}

