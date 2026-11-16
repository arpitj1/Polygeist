module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global constant @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1B" : memref<25xf32> = dense<[-7.23318078E-18, -4.83050441E-18, 4.46562156E-17, 3.46122279E-17, -2.82762388E-16, -3.42548568E-16, 1.77256012E-15, 3.81168087E-15, -9.55484674E-15, -4.15056918E-14, 1.54008615E-14, 3.85277829E-13, 7.18012455E-13, -1.79417852E-12, -1.32158121E-11, -3.14991644E-11, 1.18891468E-11, 4.94060237E-10, 3.39623196E-9, 2.26666899E-8, 2.04891862E-7, 2.89137051E-6, 6.88975852E-5, 0.00336911646, 0.804490387]>
  memref.global constant @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1A" : memref<30xf32> = dense<[-4.41534163E-18, 3.33079461E-17, -2.431280e-16, 1.71539133E-15, -1.16853328E-14, 7.67618526E-14, -4.85644673E-13, 2.95505265E-12, -1.72682632E-11, 9.67580876E-11, -5.18979582E-10, 2.65982369E-9, -1.30002498E-8, 6.04699508E-8, -2.67079372E-7, 1.11738757E-6, -4.41673819E-6, 1.64484482E-5, -5.75419508E-5, 1.88502891E-4, -5.76375576E-4, 0.00163947558, -4.324310e-03, 0.0105464607, -0.0237374157, 0.0493052825, -9.490110e-02, 0.171620905, -0.304682672, 0.676795303]>
  memref.global constant @"_Z26modified_bessel_k0_forwardIfET_S0_@static@_ZZ26modified_bessel_k0_forwardIfET_S0_E1B" : memref<25xf32> = dense<[5.30043357E-18, -1.64758049E-17, 5.21039149E-17, -1.67823114E-16, 5.51205579E-16, -1.84859329E-15, 6.34007634E-15, -2.22751335E-14, 8.03289049E-14, -2.98009691E-13, 1.14034053E-12, -4.5145979E-12, 1.85594907E-11, -7.95748953E-11, 3.57739727E-10, -1.69753456E-9, 8.57403393E-9, -4.660490e-08, 2.76681362E-7, -1.83175553E-6, 1.39498134E-5, -1.28495492E-4, 0.0015698839, -3.144810e-02, 2.44030309]>
  memref.global constant @"_Z26modified_bessel_k0_forwardIfET_S0_@static@_ZZ26modified_bessel_k0_forwardIfET_S0_E1A" : memref<10xf32> = dense<[1.37446541E-16, 4.25981611E-14, 1.03496951E-11, 1.90451632E-9, 2.53479101E-7, 2.28621211E-5, 0.00126461545, 0.0359799378, 0.344289899, -0.535327375]>
  func.func @thrust_modified_bessel_k0_forward_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %cst_2 = arith.constant 5.000000e-01 : f64
    %cst_3 = arith.constant 8.000000e+00 : f32
    %true = arith.constant true
    %cst_4 = arith.constant 5.30043357E-18 : f32
    %cst_5 = arith.constant 1.37446541E-16 : f32
    %cst_6 = arith.constant 0x7F800000 : f32
    %cst_7 = arith.constant 0x7FC00000 : f32
    %cst_8 = arith.constant -4.41534163E-18 : f32
    %cst_9 = arith.constant -7.23318078E-18 : f32
    %cst_10 = arith.constant 3.200000e+01 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    %2 = memref.get_global @"_Z26modified_bessel_k0_forwardIfET_S0_@static@_ZZ26modified_bessel_k0_forwardIfET_S0_E1B" : memref<25xf32>
    %3 = memref.get_global @"_Z26modified_bessel_k0_forwardIfET_S0_@static@_ZZ26modified_bessel_k0_forwardIfET_S0_E1A" : memref<10xf32>
    %4 = memref.get_global @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1B" : memref<25xf32>
    %5 = memref.get_global @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1A" : memref<30xf32>
    affine.for %arg3 = 0 to %0 {
      %6 = affine.load %arg1[%arg3] : memref<?xf32>
      %7 = arith.cmpf oeq, %6, %cst : f32
      %8 = arith.xori %7, %true : i1
      %9 = arith.select %7, %cst_6, %1 : f32
      %10 = arith.cmpf olt, %6, %cst : f32
      %11 = arith.andi %8, %10 : i1
      %12 = arith.xori %11, %true : i1
      %13 = arith.andi %12, %8 : i1
      %14 = arith.select %11, %cst_7, %9 : f32
      %15 = arith.cmpf ole, %6, %cst_0 : f32
      %16 = scf.if %13 -> (f32) {
        %17 = scf.if %15 -> (f32) {
          %18 = arith.mulf %6, %6 : f32
          %19 = arith.subf %18, %cst_0 : f32
          %20:3 = affine.for %arg4 = 1 to 10 iter_args(%arg5 = %cst_5, %arg6 = %cst, %arg7 = %cst) -> (f32, f32, f32) {
            %34 = arith.index_cast %arg4 : index to i8
            %35 = arith.mulf %19, %arg5 : f32
            %36 = arith.subf %35, %arg6 : f32
            %37 = arith.index_castui %34 : i8 to index
            %38 = memref.load %3[%37] : memref<10xf32>
            %39 = arith.addf %36, %38 : f32
            affine.yield %39, %arg5, %arg6 : f32, f32, f32
          }
          %21 = arith.subf %20#0, %20#2 : f32
          %22 = arith.mulf %21, %cst_1 : f32
          %23 = arith.extf %22 : f32 to f64
          %24 = arith.extf %6 : f32 to f64
          %25 = arith.mulf %24, %cst_2 : f64
          %26 = math.log %25 : f64
          %27 = math.absf %6 : f32
          %28 = arith.cmpf ole, %27, %cst_3 : f32
          %29 = scf.if %28 -> (f32) {
            %34 = arith.divf %27, %cst_0 : f32
            %35 = arith.subf %34, %cst_0 : f32
            %36:3 = affine.for %arg4 = 1 to 30 iter_args(%arg5 = %cst_8, %arg6 = %cst, %arg7 = %cst) -> (f32, f32, f32) {
              %41 = arith.index_cast %arg4 : index to i8
              %42 = arith.mulf %35, %arg5 : f32
              %43 = arith.subf %42, %arg6 : f32
              %44 = arith.index_castui %41 : i8 to index
              %45 = memref.load %5[%44] : memref<30xf32>
              %46 = arith.addf %43, %45 : f32
              affine.yield %46, %arg5, %arg6 : f32, f32, f32
            }
            %37 = math.exp %27 : f32
            %38 = arith.subf %36#0, %36#2 : f32
            %39 = arith.mulf %38, %cst_1 : f32
            %40 = arith.mulf %37, %39 : f32
            scf.yield %40 : f32
          } else {
            %34 = arith.divf %cst_10, %27 : f32
            %35 = arith.subf %34, %cst_0 : f32
            %36:3 = affine.for %arg4 = 1 to 25 iter_args(%arg5 = %cst_9, %arg6 = %cst, %arg7 = %cst) -> (f32, f32, f32) {
              %43 = arith.index_cast %arg4 : index to i8
              %44 = arith.mulf %35, %arg5 : f32
              %45 = arith.subf %44, %arg6 : f32
              %46 = arith.index_castui %43 : i8 to index
              %47 = memref.load %4[%46] : memref<25xf32>
              %48 = arith.addf %45, %47 : f32
              affine.yield %48, %arg5, %arg6 : f32, f32, f32
            }
            %37 = math.exp %27 : f32
            %38 = arith.subf %36#0, %36#2 : f32
            %39 = arith.mulf %38, %cst_1 : f32
            %40 = arith.mulf %37, %39 : f32
            %41 = math.sqrt %27 : f32
            %42 = arith.divf %40, %41 : f32
            scf.yield %42 : f32
          }
          %30 = arith.extf %29 : f32 to f64
          %31 = arith.mulf %26, %30 : f64
          %32 = arith.subf %23, %31 : f64
          %33 = arith.truncf %32 : f64 to f32
          scf.yield %33 : f32
        } else {
          %18 = arith.divf %cst_3, %6 : f32
          %19 = arith.subf %18, %cst_0 : f32
          %20:3 = affine.for %arg4 = 1 to 25 iter_args(%arg5 = %cst_4, %arg6 = %cst, %arg7 = %cst) -> (f32, f32, f32) {
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

