module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global constant @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1B" : memref<25xf32> = dense<[-7.23318078E-18, -4.83050441E-18, 4.46562156E-17, 3.46122279E-17, -2.82762388E-16, -3.42548568E-16, 1.77256012E-15, 3.81168087E-15, -9.55484674E-15, -4.15056918E-14, 1.54008615E-14, 3.85277829E-13, 7.18012455E-13, -1.79417852E-12, -1.32158121E-11, -3.14991644E-11, 1.18891468E-11, 4.94060237E-10, 3.39623196E-9, 2.26666899E-8, 2.04891862E-7, 2.89137051E-6, 6.88975852E-5, 0.00336911646, 0.804490387]>
  memref.global constant @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1A" : memref<30xf32> = dense<[-4.41534163E-18, 3.33079461E-17, -2.431280e-16, 1.71539133E-15, -1.16853328E-14, 7.67618526E-14, -4.85644673E-13, 2.95505265E-12, -1.72682632E-11, 9.67580876E-11, -5.18979582E-10, 2.65982369E-9, -1.30002498E-8, 6.04699508E-8, -2.67079372E-7, 1.11738757E-6, -4.41673819E-6, 1.64484482E-5, -5.75419508E-5, 1.88502891E-4, -5.76375576E-4, 0.00163947558, -4.324310e-03, 0.0105464607, -0.0237374157, 0.0493052825, -9.490110e-02, 0.171620905, -0.304682672, 0.676795303]>
  memref.global constant @"_Z33scaled_modified_bessel_k0_forwardIfET_S0_@static@_ZZ33scaled_modified_bessel_k0_forwardIfET_S0_E1B" : memref<25xf32> = dense<[5.30043357E-18, -1.64758049E-17, 5.21039149E-17, -1.67823114E-16, 5.51205579E-16, -1.84859329E-15, 6.34007634E-15, -2.22751335E-14, 8.03289049E-14, -2.98009691E-13, 1.14034053E-12, -4.5145979E-12, 1.85594907E-11, -7.95748953E-11, 3.57739727E-10, -1.69753456E-9, 8.57403393E-9, -4.660490e-08, 2.76681362E-7, -1.83175553E-6, 1.39498134E-5, -1.28495492E-4, 0.0015698839, -3.144810e-02, 2.44030309]>
  memref.global constant @"_Z33scaled_modified_bessel_k0_forwardIfET_S0_@static@_ZZ33scaled_modified_bessel_k0_forwardIfET_S0_E1A" : memref<10xf32> = dense<[1.37446541E-16, 4.25981611E-14, 1.03496951E-11, 1.90451632E-9, 2.53479101E-7, 2.28621211E-5, 0.00126461545, 0.0359799378, 0.344289899, -0.535327375]>
  func.func @thrust_calc_scaled_bessel_k0_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 8.000000e+00 : f32
    %true = arith.constant true
    %cst_3 = arith.constant 5.30043357E-18 : f32
    %cst_4 = arith.constant 1.37446541E-16 : f32
    %cst_5 = arith.constant 0x7F800000 : f32
    %cst_6 = arith.constant 0x7FC00000 : f32
    %cst_7 = arith.constant -4.41534163E-18 : f32
    %cst_8 = arith.constant -7.23318078E-18 : f32
    %cst_9 = arith.constant 3.200000e+01 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    %2 = memref.get_global @"_Z33scaled_modified_bessel_k0_forwardIfET_S0_@static@_ZZ33scaled_modified_bessel_k0_forwardIfET_S0_E1B" : memref<25xf32>
    %3 = memref.get_global @"_Z33scaled_modified_bessel_k0_forwardIfET_S0_@static@_ZZ33scaled_modified_bessel_k0_forwardIfET_S0_E1A" : memref<10xf32>
    %4 = memref.get_global @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1B" : memref<25xf32>
    %5 = memref.get_global @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1A" : memref<30xf32>
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
          %20:3 = affine.for %arg4 = 1 to 10 iter_args(%arg5 = %cst_4, %arg6 = %cst, %arg7 = %cst) -> (f32, f32, f32) {
            %32 = arith.index_cast %arg4 : index to i64
            %33 = arith.mulf %19, %arg5 : f32
            %34 = arith.subf %33, %arg6 : f32
            %35 = arith.index_castui %32 : i64 to index
            %36 = memref.load %3[%35] : memref<10xf32>
            %37 = arith.addf %34, %36 : f32
            affine.yield %37, %arg5, %arg6 : f32, f32, f32
          }
          %21 = arith.subf %20#0, %20#2 : f32
          %22 = arith.mulf %21, %cst_1 : f32
          %23 = arith.mulf %6, %cst_1 : f32
          %24 = math.log %23 : f32
          %25 = math.absf %6 : f32
          %26 = arith.cmpf ole, %25, %cst_2 : f32
          %27 = scf.if %26 -> (f32) {
            %32 = arith.divf %25, %cst_0 : f32
            %33 = arith.subf %32, %cst_0 : f32
            %34:3 = affine.for %arg4 = 1 to 30 iter_args(%arg5 = %cst_7, %arg6 = %cst, %arg7 = %cst) -> (f32, f32, f32) {
              %39 = arith.index_cast %arg4 : index to i8
              %40 = arith.mulf %33, %arg5 : f32
              %41 = arith.subf %40, %arg6 : f32
              %42 = arith.index_castui %39 : i8 to index
              %43 = memref.load %5[%42] : memref<30xf32>
              %44 = arith.addf %41, %43 : f32
              affine.yield %44, %arg5, %arg6 : f32, f32, f32
            }
            %35 = math.exp %25 : f32
            %36 = arith.subf %34#0, %34#2 : f32
            %37 = arith.mulf %36, %cst_1 : f32
            %38 = arith.mulf %35, %37 : f32
            scf.yield %38 : f32
          } else {
            %32 = arith.divf %cst_9, %25 : f32
            %33 = arith.subf %32, %cst_0 : f32
            %34:3 = affine.for %arg4 = 1 to 25 iter_args(%arg5 = %cst_8, %arg6 = %cst, %arg7 = %cst) -> (f32, f32, f32) {
              %41 = arith.index_cast %arg4 : index to i8
              %42 = arith.mulf %33, %arg5 : f32
              %43 = arith.subf %42, %arg6 : f32
              %44 = arith.index_castui %41 : i8 to index
              %45 = memref.load %4[%44] : memref<25xf32>
              %46 = arith.addf %43, %45 : f32
              affine.yield %46, %arg5, %arg6 : f32, f32, f32
            }
            %35 = math.exp %25 : f32
            %36 = arith.subf %34#0, %34#2 : f32
            %37 = arith.mulf %36, %cst_1 : f32
            %38 = arith.mulf %35, %37 : f32
            %39 = math.sqrt %25 : f32
            %40 = arith.divf %38, %39 : f32
            scf.yield %40 : f32
          }
          %28 = arith.mulf %24, %27 : f32
          %29 = arith.subf %22, %28 : f32
          %30 = math.exp %6 : f32
          %31 = arith.mulf %29, %30 : f32
          scf.yield %31 : f32
        } else {
          %18 = arith.divf %cst_2, %6 : f32
          %19 = arith.subf %18, %cst_0 : f32
          %20:3 = affine.for %arg4 = 1 to 25 iter_args(%arg5 = %cst_3, %arg6 = %cst, %arg7 = %cst) -> (f32, f32, f32) {
            %25 = arith.index_cast %arg4 : index to i64
            %26 = arith.mulf %19, %arg5 : f32
            %27 = arith.subf %26, %arg6 : f32
            %28 = arith.index_castui %25 : i64 to index
            %29 = memref.load %2[%28] : memref<25xf32>
            %30 = arith.addf %27, %29 : f32
            affine.yield %30, %arg5, %arg6 : f32, f32, f32
          }
          %21 = arith.subf %20#0, %20#2 : f32
          %22 = arith.mulf %21, %cst_1 : f32
          %23 = math.sqrt %6 : f32
          %24 = arith.divf %22, %23 : f32
          scf.yield %24 : f32
        }
        scf.yield %17 : f32
      } else {
        scf.yield %14 : f32
      }
      affine.store %16, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
  func.func @calc_scaled_bessel_k0f(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.200000e+01 : f32
    %cst_0 = arith.constant -7.23318078E-18 : f32
    %cst_1 = arith.constant -4.41534163E-18 : f32
    %cst_2 = arith.constant 0x7FC00000 : f32
    %cst_3 = arith.constant 0x7F800000 : f32
    %cst_4 = arith.constant 1.37446541E-16 : f32
    %cst_5 = arith.constant 5.30043357E-18 : f32
    %true = arith.constant true
    %cst_6 = arith.constant 8.000000e+00 : f32
    %cst_7 = arith.constant 5.000000e-01 : f32
    %cst_8 = arith.constant 2.000000e+00 : f32
    %cst_9 = arith.constant 0.000000e+00 : f32
    %0 = llvm.mlir.undef : f32
    %1 = memref.get_global @"_Z33scaled_modified_bessel_k0_forwardIfET_S0_@static@_ZZ33scaled_modified_bessel_k0_forwardIfET_S0_E1B" : memref<25xf32>
    %2 = memref.get_global @"_Z33scaled_modified_bessel_k0_forwardIfET_S0_@static@_ZZ33scaled_modified_bessel_k0_forwardIfET_S0_E1A" : memref<10xf32>
    %3 = arith.cmpf oeq, %arg0, %cst_9 : f32
    %4 = arith.xori %3, %true : i1
    %5 = arith.select %3, %cst_3, %0 : f32
    %6 = arith.cmpf olt, %arg0, %cst_9 : f32
    %7 = arith.andi %4, %6 : i1
    %8 = arith.xori %7, %true : i1
    %9 = arith.andi %8, %4 : i1
    %10 = arith.select %7, %cst_2, %5 : f32
    %11 = arith.cmpf ole, %arg0, %cst_8 : f32
    %12 = scf.if %9 -> (f32) {
      %13 = scf.if %11 -> (f32) {
        %14 = arith.mulf %arg0, %arg0 : f32
        %15 = arith.subf %14, %cst_8 : f32
        %16:3 = affine.for %arg1 = 1 to 10 iter_args(%arg2 = %cst_4, %arg3 = %cst_9, %arg4 = %cst_9) -> (f32, f32, f32) {
          %30 = arith.index_cast %arg1 : index to i64
          %31 = arith.mulf %15, %arg2 : f32
          %32 = arith.subf %31, %arg3 : f32
          %33 = arith.index_castui %30 : i64 to index
          %34 = memref.load %2[%33] : memref<10xf32>
          %35 = arith.addf %32, %34 : f32
          affine.yield %35, %arg2, %arg3 : f32, f32, f32
        }
        %17 = arith.subf %16#0, %16#2 : f32
        %18 = arith.mulf %17, %cst_7 : f32
        %19 = arith.mulf %arg0, %cst_7 : f32
        %20 = math.log %19 : f32
        %21 = memref.get_global @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1B" : memref<25xf32>
        %22 = memref.get_global @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1A" : memref<30xf32>
        %23 = math.absf %arg0 : f32
        %24 = arith.cmpf ole, %23, %cst_6 : f32
        %25 = scf.if %24 -> (f32) {
          %30 = arith.divf %23, %cst_8 : f32
          %31 = arith.subf %30, %cst_8 : f32
          %32:3 = affine.for %arg1 = 1 to 30 iter_args(%arg2 = %cst_1, %arg3 = %cst_9, %arg4 = %cst_9) -> (f32, f32, f32) {
            %37 = arith.index_cast %arg1 : index to i8
            %38 = arith.mulf %31, %arg2 : f32
            %39 = arith.subf %38, %arg3 : f32
            %40 = arith.index_castui %37 : i8 to index
            %41 = memref.load %22[%40] : memref<30xf32>
            %42 = arith.addf %39, %41 : f32
            affine.yield %42, %arg2, %arg3 : f32, f32, f32
          }
          %33 = math.exp %23 : f32
          %34 = arith.subf %32#0, %32#2 : f32
          %35 = arith.mulf %34, %cst_7 : f32
          %36 = arith.mulf %33, %35 : f32
          scf.yield %36 : f32
        } else {
          %30 = arith.divf %cst, %23 : f32
          %31 = arith.subf %30, %cst_8 : f32
          %32:3 = affine.for %arg1 = 1 to 25 iter_args(%arg2 = %cst_0, %arg3 = %cst_9, %arg4 = %cst_9) -> (f32, f32, f32) {
            %39 = arith.index_cast %arg1 : index to i8
            %40 = arith.mulf %31, %arg2 : f32
            %41 = arith.subf %40, %arg3 : f32
            %42 = arith.index_castui %39 : i8 to index
            %43 = memref.load %21[%42] : memref<25xf32>
            %44 = arith.addf %41, %43 : f32
            affine.yield %44, %arg2, %arg3 : f32, f32, f32
          }
          %33 = math.exp %23 : f32
          %34 = arith.subf %32#0, %32#2 : f32
          %35 = arith.mulf %34, %cst_7 : f32
          %36 = arith.mulf %33, %35 : f32
          %37 = math.sqrt %23 : f32
          %38 = arith.divf %36, %37 : f32
          scf.yield %38 : f32
        }
        %26 = arith.mulf %20, %25 : f32
        %27 = arith.subf %18, %26 : f32
        %28 = math.exp %arg0 : f32
        %29 = arith.mulf %27, %28 : f32
        scf.yield %29 : f32
      } else {
        %14 = arith.divf %cst_6, %arg0 : f32
        %15 = arith.subf %14, %cst_8 : f32
        %16:3 = affine.for %arg1 = 1 to 25 iter_args(%arg2 = %cst_5, %arg3 = %cst_9, %arg4 = %cst_9) -> (f32, f32, f32) {
          %21 = arith.index_cast %arg1 : index to i64
          %22 = arith.mulf %15, %arg2 : f32
          %23 = arith.subf %22, %arg3 : f32
          %24 = arith.index_castui %21 : i64 to index
          %25 = memref.load %1[%24] : memref<25xf32>
          %26 = arith.addf %23, %25 : f32
          affine.yield %26, %arg2, %arg3 : f32, f32, f32
        }
        %17 = arith.subf %16#0, %16#2 : f32
        %18 = arith.mulf %17, %cst_7 : f32
        %19 = math.sqrt %arg0 : f32
        %20 = arith.divf %18, %19 : f32
        scf.yield %20 : f32
      }
      scf.yield %13 : f32
    } else {
      scf.yield %10 : f32
    }
    return %12 : f32
  }
  func.func @_Z33scaled_modified_bessel_k0_forwardIfET_S0_(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant 3.200000e+01 : f32
    %cst_0 = arith.constant -7.23318078E-18 : f32
    %cst_1 = arith.constant -4.41534163E-18 : f32
    %cst_2 = arith.constant 0x7FC00000 : f32
    %cst_3 = arith.constant 0x7F800000 : f32
    %cst_4 = arith.constant 1.37446541E-16 : f32
    %cst_5 = arith.constant 5.30043357E-18 : f32
    %true = arith.constant true
    %cst_6 = arith.constant 8.000000e+00 : f32
    %cst_7 = arith.constant 5.000000e-01 : f32
    %cst_8 = arith.constant 2.000000e+00 : f32
    %cst_9 = arith.constant 0.000000e+00 : f32
    %0 = llvm.mlir.undef : f32
    %1 = memref.get_global @"_Z33scaled_modified_bessel_k0_forwardIfET_S0_@static@_ZZ33scaled_modified_bessel_k0_forwardIfET_S0_E1B" : memref<25xf32>
    %2 = memref.get_global @"_Z33scaled_modified_bessel_k0_forwardIfET_S0_@static@_ZZ33scaled_modified_bessel_k0_forwardIfET_S0_E1A" : memref<10xf32>
    %3 = arith.cmpf oeq, %arg0, %cst_9 : f32
    %4 = arith.xori %3, %true : i1
    %5 = arith.select %3, %cst_3, %0 : f32
    %6 = arith.cmpf olt, %arg0, %cst_9 : f32
    %7 = arith.andi %4, %6 : i1
    %8 = arith.xori %7, %true : i1
    %9 = arith.andi %8, %4 : i1
    %10 = arith.select %7, %cst_2, %5 : f32
    %11 = arith.cmpf ole, %arg0, %cst_8 : f32
    %12 = scf.if %9 -> (f32) {
      %13 = scf.if %11 -> (f32) {
        %14 = arith.mulf %arg0, %arg0 : f32
        %15 = arith.subf %14, %cst_8 : f32
        %16:3 = affine.for %arg1 = 1 to 10 iter_args(%arg2 = %cst_4, %arg3 = %cst_9, %arg4 = %cst_9) -> (f32, f32, f32) {
          %30 = arith.index_cast %arg1 : index to i64
          %31 = arith.mulf %15, %arg2 : f32
          %32 = arith.subf %31, %arg3 : f32
          %33 = arith.index_castui %30 : i64 to index
          %34 = memref.load %2[%33] : memref<10xf32>
          %35 = arith.addf %32, %34 : f32
          affine.yield %35, %arg2, %arg3 : f32, f32, f32
        }
        %17 = arith.subf %16#0, %16#2 : f32
        %18 = arith.mulf %17, %cst_7 : f32
        %19 = arith.mulf %arg0, %cst_7 : f32
        %20 = math.log %19 : f32
        %21 = memref.get_global @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1B" : memref<25xf32>
        %22 = memref.get_global @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1A" : memref<30xf32>
        %23 = math.absf %arg0 : f32
        %24 = arith.cmpf ole, %23, %cst_6 : f32
        %25 = scf.if %24 -> (f32) {
          %30 = arith.divf %23, %cst_8 : f32
          %31 = arith.subf %30, %cst_8 : f32
          %32:3 = affine.for %arg1 = 1 to 30 iter_args(%arg2 = %cst_1, %arg3 = %cst_9, %arg4 = %cst_9) -> (f32, f32, f32) {
            %37 = arith.index_cast %arg1 : index to i8
            %38 = arith.mulf %31, %arg2 : f32
            %39 = arith.subf %38, %arg3 : f32
            %40 = arith.index_castui %37 : i8 to index
            %41 = memref.load %22[%40] : memref<30xf32>
            %42 = arith.addf %39, %41 : f32
            affine.yield %42, %arg2, %arg3 : f32, f32, f32
          }
          %33 = math.exp %23 : f32
          %34 = arith.subf %32#0, %32#2 : f32
          %35 = arith.mulf %34, %cst_7 : f32
          %36 = arith.mulf %33, %35 : f32
          scf.yield %36 : f32
        } else {
          %30 = arith.divf %cst, %23 : f32
          %31 = arith.subf %30, %cst_8 : f32
          %32:3 = affine.for %arg1 = 1 to 25 iter_args(%arg2 = %cst_0, %arg3 = %cst_9, %arg4 = %cst_9) -> (f32, f32, f32) {
            %39 = arith.index_cast %arg1 : index to i8
            %40 = arith.mulf %31, %arg2 : f32
            %41 = arith.subf %40, %arg3 : f32
            %42 = arith.index_castui %39 : i8 to index
            %43 = memref.load %21[%42] : memref<25xf32>
            %44 = arith.addf %41, %43 : f32
            affine.yield %44, %arg2, %arg3 : f32, f32, f32
          }
          %33 = math.exp %23 : f32
          %34 = arith.subf %32#0, %32#2 : f32
          %35 = arith.mulf %34, %cst_7 : f32
          %36 = arith.mulf %33, %35 : f32
          %37 = math.sqrt %23 : f32
          %38 = arith.divf %36, %37 : f32
          scf.yield %38 : f32
        }
        %26 = arith.mulf %20, %25 : f32
        %27 = arith.subf %18, %26 : f32
        %28 = math.exp %arg0 : f32
        %29 = arith.mulf %27, %28 : f32
        scf.yield %29 : f32
      } else {
        %14 = arith.divf %cst_6, %arg0 : f32
        %15 = arith.subf %14, %cst_8 : f32
        %16:3 = affine.for %arg1 = 1 to 25 iter_args(%arg2 = %cst_5, %arg3 = %cst_9, %arg4 = %cst_9) -> (f32, f32, f32) {
          %21 = arith.index_cast %arg1 : index to i64
          %22 = arith.mulf %15, %arg2 : f32
          %23 = arith.subf %22, %arg3 : f32
          %24 = arith.index_castui %21 : i64 to index
          %25 = memref.load %1[%24] : memref<25xf32>
          %26 = arith.addf %23, %25 : f32
          affine.yield %26, %arg2, %arg3 : f32, f32, f32
        }
        %17 = arith.subf %16#0, %16#2 : f32
        %18 = arith.mulf %17, %cst_7 : f32
        %19 = math.sqrt %arg0 : f32
        %20 = arith.divf %18, %19 : f32
        scf.yield %20 : f32
      }
      scf.yield %13 : f32
    } else {
      scf.yield %10 : f32
    }
    return %12 : f32
  }
  func.func @_ZNSt14numeric_limitsIfE8infinityEv() -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant 0x7F800000 : f32
    return %cst : f32
  }
  func.func @_ZNSt14numeric_limitsIfE9quiet_NaNEv() -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant 0x7FC00000 : f32
    return %cst : f32
  }
  func.func @_Z26modified_bessel_i0_forwardIfET_S0_(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant -4.41534163E-18 : f32
    %cst_0 = arith.constant -7.23318078E-18 : f32
    %cst_1 = arith.constant 3.200000e+01 : f32
    %cst_2 = arith.constant 5.000000e-01 : f32
    %cst_3 = arith.constant 2.000000e+00 : f32
    %cst_4 = arith.constant 8.000000e+00 : f32
    %cst_5 = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1B" : memref<25xf32>
    %1 = memref.get_global @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1A" : memref<30xf32>
    %2 = math.absf %arg0 : f32
    %3 = arith.cmpf ole, %2, %cst_4 : f32
    %4 = scf.if %3 -> (f32) {
      %5 = arith.divf %2, %cst_3 : f32
      %6 = arith.subf %5, %cst_3 : f32
      %7:3 = affine.for %arg1 = 1 to 30 iter_args(%arg2 = %cst, %arg3 = %cst_5, %arg4 = %cst_5) -> (f32, f32, f32) {
        %12 = arith.index_cast %arg1 : index to i8
        %13 = arith.mulf %6, %arg2 : f32
        %14 = arith.subf %13, %arg3 : f32
        %15 = arith.index_castui %12 : i8 to index
        %16 = memref.load %1[%15] : memref<30xf32>
        %17 = arith.addf %14, %16 : f32
        affine.yield %17, %arg2, %arg3 : f32, f32, f32
      }
      %8 = math.exp %2 : f32
      %9 = arith.subf %7#0, %7#2 : f32
      %10 = arith.mulf %9, %cst_2 : f32
      %11 = arith.mulf %8, %10 : f32
      scf.yield %11 : f32
    } else {
      %5 = arith.divf %cst_1, %2 : f32
      %6 = arith.subf %5, %cst_3 : f32
      %7:3 = affine.for %arg1 = 1 to 25 iter_args(%arg2 = %cst_0, %arg3 = %cst_5, %arg4 = %cst_5) -> (f32, f32, f32) {
        %14 = arith.index_cast %arg1 : index to i8
        %15 = arith.mulf %6, %arg2 : f32
        %16 = arith.subf %15, %arg3 : f32
        %17 = arith.index_castui %14 : i8 to index
        %18 = memref.load %0[%17] : memref<25xf32>
        %19 = arith.addf %16, %18 : f32
        affine.yield %19, %arg2, %arg3 : f32, f32, f32
      }
      %8 = math.exp %2 : f32
      %9 = arith.subf %7#0, %7#2 : f32
      %10 = arith.mulf %9, %cst_2 : f32
      %11 = arith.mulf %8, %10 : f32
      %12 = math.sqrt %2 : f32
      %13 = arith.divf %11, %12 : f32
      scf.yield %13 : f32
    }
    return %4 : f32
  }
  func.func @_ZSt4sqrtf(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = math.sqrt %arg0 : f32
    return %0 : f32
  }
  func.func @_ZSt3absf(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = math.absf %arg0 : f32
    return %0 : f32
  }
}
