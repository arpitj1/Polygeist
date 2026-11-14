module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_erfcx_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i8 = arith.constant 0 : i8
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e+01 : f32
    %cst_1 = arith.constant 0.564189613 : f32
    %cst_2 = arith.constant 5.000000e+07 : f64
    %cst_3 = arith.constant 4.500000e+00 : f64
    %cst_4 = arith.constant 2.000000e+00 : f64
    %cst_5 = arith.constant 5.000000e+00 : f32
    %cst_6 = arith.constant 3.750000e+00 : f64
    %cst_7 = arith.constant 4.000000e+02 : f32
    %cst_8 = arith.constant 4.000000e+00 : f32
    %cst_9 = arith.constant -2.670000e+01 : f64
    %cst_10 = arith.constant -6.100000e+00 : f64
    %cst_11 = arith.constant 0.564189612865448 : f64
    %cst_12 = arith.constant 0x7F800000 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    affine.for %arg3 = 0 to %0 {
      %2 = affine.load %arg1[%arg3] : memref<?xf32>
      %3 = arith.cmpf uno, %2, %2 : f32
      %4 = arith.extui %3 : i1 to i8
      %5 = arith.cmpi eq, %4, %c0_i8 : i8
      %6 = arith.select %3, %2, %1 : f32
      %7 = arith.cmpf oge, %2, %cst : f32
      %8 = scf.if %5 -> (f32) {
        %9 = scf.if %7 -> (f32) {
          %10 = arith.cmpf ogt, %2, %cst_0 : f32
          %11 = scf.if %10 -> (f32) {
            %12 = arith.extf %2 : f32 to f64
            %13 = arith.cmpf ogt, %12, %cst_2 : f64
            %14 = scf.if %13 -> (f32) {
              %15 = arith.divf %cst_1, %2 : f32
              scf.yield %15 : f32
            } else {
              %15 = arith.mulf %2, %2 : f32
              %16 = arith.extf %15 : f32 to f64
              %17 = arith.addf %16, %cst_3 : f64
              %18 = arith.mulf %16, %17 : f64
              %19 = arith.addf %18, %cst_4 : f64
              %20 = arith.mulf %19, %cst_11 : f64
              %21 = arith.addf %15, %cst_5 : f32
              %22 = arith.mulf %15, %21 : f32
              %23 = arith.extf %22 : f32 to f64
              %24 = arith.addf %23, %cst_6 : f64
              %25 = arith.mulf %12, %24 : f64
              %26 = arith.divf %20, %25 : f64
              %27 = arith.truncf %26 : f64 to f32
              scf.yield %27 : f32
            }
            scf.yield %14 : f32
          } else {
            %12 = arith.addf %2, %cst_8 : f32
            %13 = arith.divf %cst_7, %12 : f32
            %14 = func.call @_Z10erfcx_y100IfENSt9enable_ifIXsr3stdE19is_floating_point_vIT_EES1_E4typeES1_(%13) : (f32) -> f32
            scf.yield %14 : f32
          }
          scf.yield %11 : f32
        } else {
          %10 = arith.extf %2 : f32 to f64
          %11 = arith.cmpf olt, %10, %cst_9 : f64
          %12 = scf.if %11 -> (f32) {
            scf.yield %cst_12 : f32
          } else {
            %13 = arith.cmpf olt, %10, %cst_10 : f64
            %14 = scf.if %13 -> (f32) {
              %15 = arith.mulf %2, %2 : f32
              %16 = arith.extf %15 : f32 to f64
              %17 = math.exp %16 : f64
              %18 = arith.mulf %17, %cst_4 : f64
              %19 = arith.truncf %18 : f64 to f32
              scf.yield %19 : f32
            } else {
              %15 = arith.mulf %2, %2 : f32
              %16 = arith.extf %15 : f32 to f64
              %17 = math.exp %16 : f64
              %18 = arith.mulf %17, %cst_4 : f64
              %19 = arith.subf %cst_8, %2 : f32
              %20 = arith.divf %cst_7, %19 : f32
              %21 = func.call @_Z10erfcx_y100IfENSt9enable_ifIXsr3stdE19is_floating_point_vIT_EES1_E4typeES1_(%20) : (f32) -> f32
              %22 = arith.extf %21 : f32 to f64
              %23 = arith.subf %18, %22 : f64
              %24 = arith.truncf %23 : f64 to f32
              scf.yield %24 : f32
            }
            scf.yield %14 : f32
          }
          scf.yield %12 : f32
        }
        scf.yield %9 : f32
      } else {
        scf.yield %6 : f32
      }
      affine.store %8, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
  func.func @calc_erfcxf(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0x7F800000 : f32
    %cst_0 = arith.constant 0.564189612865448 : f64
    %cst_1 = arith.constant -6.100000e+00 : f64
    %cst_2 = arith.constant -2.670000e+01 : f64
    %cst_3 = arith.constant 4.000000e+00 : f32
    %cst_4 = arith.constant 4.000000e+02 : f32
    %cst_5 = arith.constant 3.750000e+00 : f64
    %cst_6 = arith.constant 5.000000e+00 : f32
    %cst_7 = arith.constant 2.000000e+00 : f64
    %cst_8 = arith.constant 4.500000e+00 : f64
    %cst_9 = arith.constant 5.000000e+07 : f64
    %cst_10 = arith.constant 0.564189613 : f32
    %cst_11 = arith.constant 5.000000e+01 : f32
    %cst_12 = arith.constant 0.000000e+00 : f32
    %c0_i8 = arith.constant 0 : i8
    %0 = llvm.mlir.undef : f32
    %1 = arith.cmpf uno, %arg0, %arg0 : f32
    %2 = arith.extui %1 : i1 to i8
    %3 = arith.cmpi eq, %2, %c0_i8 : i8
    %4 = arith.select %1, %arg0, %0 : f32
    %5 = arith.cmpf oge, %arg0, %cst_12 : f32
    %6 = scf.if %3 -> (f32) {
      %7 = scf.if %5 -> (f32) {
        %8 = arith.cmpf ogt, %arg0, %cst_11 : f32
        %9 = scf.if %8 -> (f32) {
          %10 = arith.extf %arg0 : f32 to f64
          %11 = arith.cmpf ogt, %10, %cst_9 : f64
          %12 = scf.if %11 -> (f32) {
            %13 = arith.divf %cst_10, %arg0 : f32
            scf.yield %13 : f32
          } else {
            %13 = arith.mulf %arg0, %arg0 : f32
            %14 = arith.extf %13 : f32 to f64
            %15 = arith.addf %14, %cst_8 : f64
            %16 = arith.mulf %14, %15 : f64
            %17 = arith.addf %16, %cst_7 : f64
            %18 = arith.mulf %17, %cst_0 : f64
            %19 = arith.addf %13, %cst_6 : f32
            %20 = arith.mulf %13, %19 : f32
            %21 = arith.extf %20 : f32 to f64
            %22 = arith.addf %21, %cst_5 : f64
            %23 = arith.mulf %10, %22 : f64
            %24 = arith.divf %18, %23 : f64
            %25 = arith.truncf %24 : f64 to f32
            scf.yield %25 : f32
          }
          scf.yield %12 : f32
        } else {
          %10 = arith.addf %arg0, %cst_3 : f32
          %11 = arith.divf %cst_4, %10 : f32
          %12 = func.call @_Z10erfcx_y100IfENSt9enable_ifIXsr3stdE19is_floating_point_vIT_EES1_E4typeES1_(%11) : (f32) -> f32
          scf.yield %12 : f32
        }
        scf.yield %9 : f32
      } else {
        %8 = arith.extf %arg0 : f32 to f64
        %9 = arith.cmpf olt, %8, %cst_2 : f64
        %10 = scf.if %9 -> (f32) {
          scf.yield %cst : f32
        } else {
          %11 = arith.cmpf olt, %8, %cst_1 : f64
          %12 = scf.if %11 -> (f32) {
            %13 = arith.mulf %arg0, %arg0 : f32
            %14 = arith.extf %13 : f32 to f64
            %15 = math.exp %14 : f64
            %16 = arith.mulf %15, %cst_7 : f64
            %17 = arith.truncf %16 : f64 to f32
            scf.yield %17 : f32
          } else {
            %13 = arith.mulf %arg0, %arg0 : f32
            %14 = arith.extf %13 : f32 to f64
            %15 = math.exp %14 : f64
            %16 = arith.mulf %15, %cst_7 : f64
            %17 = arith.subf %cst_3, %arg0 : f32
            %18 = arith.divf %cst_4, %17 : f32
            %19 = func.call @_Z10erfcx_y100IfENSt9enable_ifIXsr3stdE19is_floating_point_vIT_EES1_E4typeES1_(%18) : (f32) -> f32
            %20 = arith.extf %19 : f32 to f64
            %21 = arith.subf %16, %20 : f64
            %22 = arith.truncf %21 : f64 to f32
            scf.yield %22 : f32
          }
          scf.yield %12 : f32
        }
        scf.yield %10 : f32
      }
      scf.yield %7 : f32
    } else {
      scf.yield %4 : f32
    }
    return %6 : f32
  }
  func.func @_Z10calc_erfcxIfENSt9enable_ifIXsr3stdE19is_floating_point_vIT_EES1_E4typeES1_(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant 0x7F800000 : f32
    %cst_0 = arith.constant 0.564189612865448 : f64
    %cst_1 = arith.constant -6.100000e+00 : f64
    %cst_2 = arith.constant -2.670000e+01 : f64
    %cst_3 = arith.constant 4.000000e+00 : f32
    %cst_4 = arith.constant 4.000000e+02 : f32
    %cst_5 = arith.constant 3.750000e+00 : f64
    %cst_6 = arith.constant 5.000000e+00 : f32
    %cst_7 = arith.constant 2.000000e+00 : f64
    %cst_8 = arith.constant 4.500000e+00 : f64
    %cst_9 = arith.constant 5.000000e+07 : f64
    %cst_10 = arith.constant 0.564189613 : f32
    %cst_11 = arith.constant 5.000000e+01 : f32
    %cst_12 = arith.constant 0.000000e+00 : f32
    %c0_i8 = arith.constant 0 : i8
    %0 = llvm.mlir.undef : f32
    %1 = arith.cmpf uno, %arg0, %arg0 : f32
    %2 = arith.extui %1 : i1 to i8
    %3 = arith.cmpi eq, %2, %c0_i8 : i8
    %4 = arith.select %1, %arg0, %0 : f32
    %5 = arith.cmpf oge, %arg0, %cst_12 : f32
    %6 = scf.if %3 -> (f32) {
      %7 = scf.if %5 -> (f32) {
        %8 = arith.cmpf ogt, %arg0, %cst_11 : f32
        %9 = scf.if %8 -> (f32) {
          %10 = arith.extf %arg0 : f32 to f64
          %11 = arith.cmpf ogt, %10, %cst_9 : f64
          %12 = scf.if %11 -> (f32) {
            %13 = arith.divf %cst_10, %arg0 : f32
            scf.yield %13 : f32
          } else {
            %13 = arith.mulf %arg0, %arg0 : f32
            %14 = arith.extf %13 : f32 to f64
            %15 = arith.addf %14, %cst_8 : f64
            %16 = arith.mulf %14, %15 : f64
            %17 = arith.addf %16, %cst_7 : f64
            %18 = arith.mulf %17, %cst_0 : f64
            %19 = arith.addf %13, %cst_6 : f32
            %20 = arith.mulf %13, %19 : f32
            %21 = arith.extf %20 : f32 to f64
            %22 = arith.addf %21, %cst_5 : f64
            %23 = arith.mulf %10, %22 : f64
            %24 = arith.divf %18, %23 : f64
            %25 = arith.truncf %24 : f64 to f32
            scf.yield %25 : f32
          }
          scf.yield %12 : f32
        } else {
          %10 = arith.addf %arg0, %cst_3 : f32
          %11 = arith.divf %cst_4, %10 : f32
          %12 = func.call @_Z10erfcx_y100IfENSt9enable_ifIXsr3stdE19is_floating_point_vIT_EES1_E4typeES1_(%11) : (f32) -> f32
          scf.yield %12 : f32
        }
        scf.yield %9 : f32
      } else {
        %8 = arith.extf %arg0 : f32 to f64
        %9 = arith.cmpf olt, %8, %cst_2 : f64
        %10 = scf.if %9 -> (f32) {
          scf.yield %cst : f32
        } else {
          %11 = arith.cmpf olt, %8, %cst_1 : f64
          %12 = scf.if %11 -> (f32) {
            %13 = arith.mulf %arg0, %arg0 : f32
            %14 = arith.extf %13 : f32 to f64
            %15 = math.exp %14 : f64
            %16 = arith.mulf %15, %cst_7 : f64
            %17 = arith.truncf %16 : f64 to f32
            scf.yield %17 : f32
          } else {
            %13 = arith.mulf %arg0, %arg0 : f32
            %14 = arith.extf %13 : f32 to f64
            %15 = math.exp %14 : f64
            %16 = arith.mulf %15, %cst_7 : f64
            %17 = arith.subf %cst_3, %arg0 : f32
            %18 = arith.divf %cst_4, %17 : f32
            %19 = func.call @_Z10erfcx_y100IfENSt9enable_ifIXsr3stdE19is_floating_point_vIT_EES1_E4typeES1_(%18) : (f32) -> f32
            %20 = arith.extf %19 : f32 to f64
            %21 = arith.subf %16, %20 : f64
            %22 = arith.truncf %21 : f64 to f32
            scf.yield %22 : f32
          }
          scf.yield %12 : f32
        }
        scf.yield %10 : f32
      }
      scf.yield %7 : f32
    } else {
      scf.yield %4 : f32
    }
    return %6 : f32
  }
  func.func @_ZN2at6_isnanIfTnNSt9enable_ifIXsr3stdE19is_floating_point_vIT_EEiE4typeELi0EEEbS2_(%arg0: f32) -> i8 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = arith.cmpf uno, %arg0, %arg0 : f32
    %1 = arith.extui %0 : i1 to i8
    return %1 : i8
  }
  func.func @_Z10erfcx_y100IfENSt9enable_ifIXsr3stdE19is_floating_point_vIT_EES1_E4typeES1_(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %true = arith.constant true
    %cst = arith.constant 2.4873652355555557E-13 : f64
    %cst_0 = arith.constant 7.8191526829368226E-11 : f64
    %cst_1 = arith.constant 1.6514019547822821E-8 : f64
    %cst_2 = arith.constant 2.5153688325245316E-6 : f64
    %cst_3 = arith.constant 2.7951610702682382E-4 : f64
    %cst_4 = arith.constant 0.022000938572830479 : f64
    %cst_5 = arith.constant 0.9777170133588503 : f64
    %cst_6 = arith.constant 1.990000e+02 : f32
    %cst_7 = arith.constant 2.4309291271111114E-13 : f64
    %cst_8 = arith.constant 7.5240468141720145E-11 : f64
    %cst_9 = arith.constant 1.5746923065472183E-8 : f64
    %cst_10 = arith.constant 2.3863447359754924E-6 : f64
    %cst_11 = arith.constant 2.6481403465998478E-4 : f64
    %cst_12 = arith.constant 0.020912536329780369 : f64
    %cst_13 = arith.constant 0.93481333942870792 : f64
    %cst_14 = arith.constant 1.970000e+02 : f32
    %cst_15 = arith.constant 2.3737194737777777E-13 : f64
    %cst_16 = arith.constant 7.2357609075043937E-11 : f64
    %cst_17 = arith.constant 1.5008997042116531E-8 : f64
    %cst_18 = arith.constant 2.2633402747585233E-6 : f64
    %cst_19 = arith.constant 2.5086793128395994E-4 : f64
    %cst_20 = arith.constant 0.019881418399127201 : f64
    %cst_21 = arith.constant 0.89402868170849936 : f64
    %cst_22 = arith.constant 1.950000e+02 : f32
    %cst_23 = arith.constant 2.3158593688888887E-13 : f64
    %cst_24 = arith.constant 6.9543803864694175E-11 : f64
    %cst_25 = arith.constant 1.4299555071870523E-8 : f64
    %cst_26 = arith.constant 2.146124825130639E-6 : f64
    %cst_27 = arith.constant 2.3764237370371255E-4 : f64
    %cst_28 = arith.constant 0.01890463221254756 : f64
    %cst_29 = arith.constant 0.85525144775685125 : f64
    %cst_30 = arith.constant 1.930000e+02 : f32
    %cst_31 = arith.constant 2.2574701262222221E-13 : f64
    %cst_32 = arith.constant 6.6799760083972476E-11 : f64
    %cst_33 = arith.constant 1.3617902941839951E-8 : f64
    %cst_34 = arith.constant 2.0344732868018175E-6 : f64
    %cst_35 = arith.constant 2.2510330592753129E-4 : f64
    %cst_36 = arith.constant 0.017979364149044223 : f64
    %cst_37 = arith.constant 0.8183758104102381 : f64
    %cst_38 = arith.constant 1.910000e+02 : f32
    %cst_39 = arith.constant 2.1986708942222223E-13 : f64
    %cst_40 = arith.constant 6.4126040998066347E-11 : f64
    %cst_41 = arith.constant 1.2963340087354342E-8 : f64
    %cst_42 = arith.constant 1.9281661395543912E-6 : f64
    %cst_43 = arith.constant 2.1321800585063328E-4 : f64
    %cst_44 = arith.constant 0.017102934132652431 : f64
    %cst_45 = arith.constant 0.78330143531283491 : f64
    %cst_46 = arith.constant 1.890000e+02 : f32
    %cst_47 = arith.constant 2.1395783431111112E-13 : f64
    %cst_48 = arith.constant 6.1523068312169082E-11 : f64
    %cst_49 = arith.constant 1.2335161021630225E-8 : f64
    %cst_50 = arith.constant 1.8269894883203348E-6 : f64
    %cst_51 = arith.constant 2.0195505163377912E-4 : f64
    %cst_52 = arith.constant 0.016272790364044783 : f64
    %cst_53 = arith.constant 0.74993321911726252 : f64
    %cst_54 = arith.constant 1.870000e+02 : f32
    %cst_55 = arith.constant 2.0803065333333334E-13 : f64
    %cst_56 = arith.constant 5.8991125287563839E-11 : f64
    %cst_57 = arith.constant 1.1732656736113608E-8 : f64
    %cst_58 = arith.constant 1.7307350969359975E-6 : f64
    %cst_59 = arith.constant 1.9128428784550924E-4 : f64
    %cst_60 = arith.constant 0.015486504187117112 : f64
    %cst_61 = arith.constant 0.71818103808729972 : f64
    %cst_62 = arith.constant 1.850000e+02 : f32
    %cst_63 = arith.constant 2.0209663662222222E-13 : f64
    %cst_64 = arith.constant 5.6530360194925692E-11 : f64
    %cst_65 = arith.constant 1.1155116068018043E-8 : f64
    %cst_66 = arith.constant 1.6392004108230584E-6 : f64
    %cst_67 = arith.constant 1.8117679143520433E-4 : f64
    %cst_68 = arith.constant 0.014741765091365868 : f64
    %cst_69 = arith.constant 0.68795950683174434 : f64
    %cst_70 = arith.constant 1.830000e+02 : f32
    %cst_71 = arith.constant 1.9616655146666667E-13 : f64
    %cst_72 = arith.constant 5.4140790105837517E-11 : f64
    %cst_73 = arith.constant 1.060182703153528E-8 : f64
    %cst_74 = arith.constant 1.5521885688723188E-6 : f64
    %cst_75 = arith.constant 1.7160483760259707E-4 : f64
    %cst_76 = arith.constant 0.014036375850601992 : f64
    %cst_77 = arith.constant 0.65918774689725324 : f64
    %cst_78 = arith.constant 1.810000e+02 : f32
    %cst_79 = arith.constant 1.9025081422222223E-13 : f64
    %cst_80 = arith.constant 5.182230499568071E-11 : f64
    %cst_81 = arith.constant 1.0072078109604152E-8 : f64
    %cst_82 = arith.constant 1.4695084048334055E-6 : f64
    %cst_83 = arith.constant 1.6254186562762076E-4 : f64
    %cst_84 = arith.constant 0.013368247798287032 : f64
    %cst_85 = arith.constant 0.6317891649471572 : f64
    %cst_86 = arith.constant 1.790000e+02 : f32
    %cst_87 = arith.constant 1.8435945564444444E-13 : f64
    %cst_88 = arith.constant 4.9574672127669038E-11 : f64
    %cst_89 = arith.constant 9.5651595032306229E-9 : f64
    %cst_90 = arith.constant 1.3909744385382817E-6 : f64
    %cst_91 = arith.constant 1.5396244472258864E-4 : f64
    %cst_92 = arith.constant 0.01273539623952555 : f64
    %cst_93 = arith.constant 0.60569124025293375 : f64
    %cst_94 = arith.constant 1.770000e+02 : f32
    %cst_95 = arith.constant 1.7850211608888889E-13 : f64
    %cst_96 = arith.constant 4.7397540713124621E-11 : f64
    %cst_97 = arith.constant 9.0803643355106019E-9 : f64
    %cst_98 = arith.constant 1.3164068573095711E-6 : f64
    %cst_99 = arith.constant 1.4584223996665839E-4 : f64
    %cst_100 = arith.constant 0.012135935999503878 : f64
    %cst_101 = arith.constant 0.58082532122519326 : f64
    %cst_102 = arith.constant 1.750000e+02 : f32
    %cst_103 = arith.constant 1.7268801084444443E-13 : f64
    %cst_104 = arith.constant 4.5290446811539653E-11 : f64
    %cst_105 = arith.constant 8.6169898078969316E-9 : f64
    %cst_106 = arith.constant 1.2456314879260905E-6 : f64
    %cst_107 = arith.constant 1.3815797838036652E-4 : f64
    %cst_108 = arith.constant 0.011568077107929736 : f64
    %cst_109 = arith.constant 0.55712643071169299 : f64
    %cst_110 = arith.constant 1.730000e+02 : f32
    %cst_111 = arith.constant 1.669259264E-13 : f64
    %cst_112 = arith.constant 4.3252818449517084E-11 : f64
    %cst_113 = arith.constant 8.1743383063044824E-9 : f64
    %cst_114 = arith.constant 1.1784797595374515E-6 : f64
    %cst_115 = arith.constant 1.3088741519572269E-4 : f64
    %cst_116 = arith.constant 0.011030120618800727 : f64
    %cst_117 = arith.constant 0.53453307979101372 : f64
    %cst_118 = arith.constant 1.710000e+02 : f32
    %cst_119 = arith.constant 1.612241968E-13 : f64
    %cst_120 = arith.constant 4.1283980931872625E-11 : f64
    %cst_121 = arith.constant 7.7517184550568704E-9 : f64
    %cst_122 = arith.constant 1.1147886579371265E-6 : f64
    %cst_123 = arith.constant 1.2400930037494997E-4 : f64
    %cst_124 = arith.constant 0.010520454564612427 : f64
    %cst_125 = arith.constant 0.51298708979209262 : f64
    %cst_126 = arith.constant 1.690000e+02 : f32
    %cst_127 = arith.constant 1.5559069118222223E-13 : f64
    %cst_128 = arith.constant 3.9383162326435753E-11 : f64
    %cst_129 = arith.constant 7.3484461168242223E-9 : f64
    %cst_130 = arith.constant 1.0544006716188967E-6 : f64
    %cst_131 = arith.constant 1.1750334542845235E-4 : f64
    %cst_132 = arith.constant 0.010037550043909497 : f64
    %cst_133 = arith.constant 0.49243342227179843 : f64
    %cst_134 = arith.constant 1.670000e+02 : f32
    %cst_135 = arith.constant 1.5003280712888889E-13 : f64
    %cst_136 = arith.constant 3.7549499088161346E-11 : f64
    %cst_137 = arith.constant 6.9638453369956974E-9 : f64
    %cst_138 = arith.constant 9.9716373005509027E-7 : f64
    %cst_139 = arith.constant 1.1135019058000067E-4 : f64
    %cst_140 = arith.constant 0.0095799574408860465 : f64
    %cst_141 = arith.constant 0.4728200166851233 : f64
    %cst_142 = arith.constant 1.650000e+02 : f32
    %cst_143 = arith.constant 1.4455745872E-13 : f64
    %cst_144 = arith.constant 3.5782041795476564E-11 : f64
    %cst_145 = arith.constant 6.5972492312219962E-9 : f64
    %cst_146 = arith.constant 9.4293113464638627E-7 : f64
    %cst_147 = arith.constant 1.0553137232446167E-4 : f64
    %cst_148 = arith.constant 0.0091463027755548237 : f64
    %cst_149 = arith.constant 0.45409763548534332 : f64
    %cst_150 = arith.constant 1.630000e+02 : f32
    %cst_151 = arith.constant 1.3917107176888888E-13 : f64
    %cst_152 = arith.constant 3.407976098345888E-11 : f64
    %cst_153 = arith.constant 6.24800081507886E-9 : f64
    %cst_154 = arith.constant 8.9156148280219875E-7 : f64
    %cst_155 = arith.constant 1.0002929142066799E-4 : f64
    %cst_156 = arith.constant 0.0087352841828289493 : f64
    %cst_157 = arith.constant 0.43621971639463786 : f64
    %cst_158 = arith.constant 1.610000e+02 : f32
    %cst_159 = arith.constant 1.3387957943111111E-13 : f64
    %cst_160 = arith.constant 3.2441553034347471E-11 : f64
    %cst_161 = arith.constant 5.9154537751083489E-9 : f64
    %cst_162 = arith.constant 8.4291858561783143E-7 : f64
    %cst_163 = arith.constant 9.482718135925016E-5 : f64
    %cst_164 = arith.constant 0.0083456685186950457 : f64
    %cst_165 = arith.constant 0.41914223158913788 : f64
    %cst_166 = arith.constant 1.590000e+02 : f32
    %cst_167 = arith.constant 1.2868841946666668E-13 : f64
    %cst_168 = arith.constant 3.0866246101464866E-11 : f64
    %cst_169 = arith.constant 5.5989731807360405E-9 : f64
    %cst_170 = arith.constant 7.9687137961956192E-7 : f64
    %cst_171 = arith.constant 8.9909077342438249E-5 : f64
    %cst_172 = arith.constant 0.0079762880915029734 : f64
    %cst_173 = arith.constant 0.40282355354616939 : f64
    %cst_174 = arith.constant 1.570000e+02 : f32
    %cst_175 = arith.constant 1.2360253370666666E-13 : f64
    %cst_176 = arith.constant 2.935260605416409E-11 : f64
    %cst_177 = arith.constant 5.297936136838812E-9 : f64
    %cst_178 = arith.constant 7.5329383305171329E-7 : f64
    %cst_179 = arith.constant 8.5259785810004609E-5 : f64
    %cst_180 = arith.constant 0.0076260375162549802 : f64
    %cst_181 = arith.constant 0.38722432730555451 : f64
    %cst_182 = arith.constant 1.550000e+02 : f32
    %cst_183 = arith.constant 1.1862637614222222E-13 : f64
    %cst_184 = arith.constant 2.7899342394100073E-11 : f64
    %cst_185 = arith.constant 5.0117323769745884E-9 : f64
    %cst_186 = arith.constant 7.1206484718062691E-7 : f64
    %cst_187 = arith.constant 8.0864854542670719E-5 : f64
    %cst_188 = arith.constant 0.007293870689646138 : f64
    %cst_189 = arith.constant 0.37230734890119727 : f64
    %cst_190 = arith.constant 1.530000e+02 : f32
    %cst_191 = arith.constant 1.1376390933333332E-13 : f64
    %cst_192 = arith.constant 2.650511414114305E-11 : f64
    %cst_193 = arith.constant 4.7397647975845227E-9 : f64
    %cst_194 = arith.constant 6.7306815308917388E-7 : f64
    %cst_195 = arith.constant 7.6710543371454821E-5 : f64
    %cst_196 = arith.constant 0.0069787978834882688 : f64
    %cst_197 = arith.constant 0.35803744972380175 : f64
    %cst_198 = arith.constant 1.510000e+02 : f32
    %cst_199 = arith.constant 1.0901861383111111E-13 : f64
    %cst_200 = arith.constant 2.5168535651285476E-11 : f64
    %cst_201 = arith.constant 4.4814499336514453E-9 : f64
    %cst_202 = arith.constant 6.3619220443228796E-7 : f64
    %cst_203 = arith.constant 7.2783795518603558E-5 : f64
    %cst_204 = arith.constant 0.0066798829540414004 : f64
    %cst_205 = arith.constant 0.34438138658041334 : f64
    %cst_206 = arith.constant 1.490000e+02 : f32
    %cst_207 = arith.constant 1.0439349811555555E-13 : f64
    %cst_208 = arith.constant 2.3888182347073697E-11 : f64
    %cst_209 = arith.constant 4.2362183765883468E-9 : f64
    %cst_210 = arith.constant 6.013300666188594E-7 : f64
    %cst_211 = arith.constant 6.90722095929424E-5 : f64
    %cst_212 = arith.constant 0.0063962406646798078 : f64
    %cst_213 = arith.constant 0.33130773722152623 : f64
    %cst_214 = arith.constant 1.470000e+02 : f32
    %cst_215 = arith.constant 9.9891109760000006E-14 : f64
    %cst_216 = arith.constant 2.2662596341239295E-11 : f64
    %cst_217 = arith.constant 4.0035151353392377E-9 : f64
    %cst_218 = arith.constant 5.6837930287837742E-7 : f64
    %cst_219 = arith.constant 6.5564012259707638E-5 : f64
    %cst_220 = arith.constant 0.00612703411923391 : f64
    %cst_221 = arith.constant 0.31878680111173319 : f64
    %cst_222 = arith.constant 1.450000e+02 : f32
    %cst_223 = arith.constant 9.5513539182222226E-14 : f64
    %cst_224 = arith.constant 2.149029193044454E-11 : f64
    %cst_225 = arith.constant 3.7827999418960236E-9 : f64
    %cst_226 = arith.constant 5.3724185766200942E-7 : f64
    %cst_227 = arith.constant 6.2248031602197683E-5 : f64
    %cst_228 = arith.constant 0.00587147230327454 : f64
    %cst_229 = arith.constant 0.30679050522528839 : f64
    %cst_230 = arith.constant 1.430000e+02 : f32
    %cst_231 = arith.constant 9.126244261333333E-14 : f64
    %cst_232 = arith.constant 2.036976093701707E-11 : f64
    %cst_233 = arith.constant 3.5735475025851714E-9 : f64
    %cst_234 = arith.constant 5.0782393781744835E-7 : f64
    %cst_235 = arith.constant 5.9113671189913308E-5 : f64
    %cst_236 = arith.constant 0.0056288077305420795 : f64
    %cst_237 = arith.constant 0.29529231465348521 : f64
    %cst_238 = arith.constant 1.410000e+02 : f32
    %cst_239 = arith.constant 8.7139049137777772E-14 : f64
    %cst_240 = arith.constant 1.9299477888083468E-11 : f64
    %cst_241 = arith.constant 3.3752476967570798E-9 : f64
    %cst_242 = arith.constant 4.8003589196494737E-7 : f64
    %cst_243 = arith.constant 5.6150884865255812E-5 : f64
    %cst_244 = arith.constant 0.0053983341916695139 : f64
    %cst_245 = arith.constant 0.28426714781640317 : f64
    %cst_246 = arith.constant 1.390000e+02 : f32
    %cst_247 = arith.constant 8.314418236444444E-14 : f64
    %cst_248 = arith.constant 1.827790501024511E-11 : f64
    %cst_249 = arith.constant 3.1874057245814381E-9 : f64
    %cst_250 = arith.constant 4.5379208848865018E-7 : f64
    %cst_251 = arith.constant 5.3350152258326602E-5 : f64
    %cst_252 = arith.constant 0.0051793846023052642 : f64
    %cst_253 = arith.constant 0.27369129607732345 : f64
    %cst_254 = arith.constant 1.370000e+02 : f32
    %cst_255 = arith.constant 7.9278273368888884E-14 : f64
    %cst_256 = arith.constant 1.7303497025347341E-11 : f64
    %cst_257 = arith.constant 3.0095422058900481E-9 : f64
    %cst_258 = arith.constant 4.2901079254268185E-7 : f64
    %cst_259 = arith.constant 5.0702455036930367E-5 : f64
    %cst_260 = arith.constant 0.0049713289477083785 : f64
    %cst_261 = arith.constant 0.26354234756393613 : f64
    %cst_262 = arith.constant 1.350000e+02 : f32
    %cst_263 = arith.constant 7.5541379822222221E-14 : f64
    %cst_264 = arith.constant 1.6374705736458321E-11 : f64
    %cst_265 = arith.constant 2.8411932320871164E-9 : f64
    %cst_266 = arith.constant 4.0561404245564733E-7 : f64
    %cst_267 = arith.constant 4.8199253896534185E-5 : f64
    %cst_268 = arith.constant 0.0047735723208650032 : f64
    %cst_269 = arith.constant 0.25379911500634267 : f64
    %cst_270 = arith.constant 1.330000e+02 : f32
    %cst_271 = arith.constant 7.1933206364444442E-14 : f64
    %cst_272 = arith.constant 1.5489984390884758E-11 : f64
    %cst_273 = arith.constant 2.6819103733055602E-9 : f64
    %cst_274 = arith.constant 3.8352752590033029E-7 : f64
    %cst_275 = arith.constant 4.5832466292683086E-5 : f64
    %cst_276 = arith.constant 0.0045855530511605784 : f64
    %cst_277 = arith.constant 0.24444156740777434 : f64
    %cst_278 = arith.constant 1.310000e+02 : f32
    %cst_279 = arith.constant 6.845312263111111E-14 : f64
    %cst_280 = arith.constant 1.4647791812837902E-11 : f64
    %cst_281 = arith.constant 2.5312606430853202E-9 : f64
    %cst_282 = arith.constant 3.6268045617760415E-7 : f64
    %cst_283 = arith.constant 4.3594444916224701E-5 : f64
    %cst_284 = arith.constant 0.0044067409206365168 : f64
    %cst_285 = arith.constant 0.23545076536988704 : f64
    %cst_286 = arith.constant 1.290000e+02 : f32
    %cst_287 = arith.constant 6.5100183751111115E-14 : f64
    %cst_288 = arith.constant 1.3846596292818514E-11 : f64
    %cst_289 = arith.constant 2.3888264229264067E-9 : f64
    %cst_290 = arith.constant 3.430054489450281E-7 : f64
    %cst_291 = arith.constant 4.1477956909656896E-5 : f64
    %cst_292 = arith.constant 0.0042366354648628521 : f64
    %cst_293 = arith.constant 0.22680879990043229 : f64
    %cst_294 = arith.constant 1.270000e+02 : f32
    %cst_295 = arith.constant 6.1873153262222224E-14 : f64
    %cst_296 = arith.constant 1.3084879235290859E-11 : f64
    %cst_297 = arith.constant 2.2542053491518681E-9 : f64
    %cst_298 = arith.constant 3.2443839970139921E-7 : f64
    %cst_299 = arith.constant 3.947616382098671E-5 : f64
    %cst_300 = arith.constant 0.0040747643554689588 : f64
    %cst_301 = arith.constant 0.21849873453703333 : f64
    %cst_302 = arith.constant 1.250000e+02 : f32
    %cst_303 = arith.constant 5.8770520160000006E-14 : f64
    %cst_304 = arith.constant 1.2361138551062899E-11 : f64
    %cst_305 = arith.constant 2.1270101645763676E-9 : f64
    %cst_306 = arith.constant 3.0691836231886877E-7 : f64
    %cst_307 = arith.constant 3.7582602289680105E-5 : f64
    %cst_308 = arith.constant 0.0039206818613925654 : f64
    %cst_309 = arith.constant 0.21050455062669335 : f64
    %cst_310 = arith.constant 1.230000e+02 : f32
    %cst_311 = arith.constant 5.5790523093333335E-14 : f64
    %cst_312 = arith.constant 1.1673891799578381E-11 : f64
    %cst_313 = arith.constant 2.0068685374849001E-9 : f64
    %cst_314 = arith.constant 2.9038742889416174E-7 : f64
    %cst_315 = arith.constant 3.5791165457592407E-5 : f64
    %cst_316 = arith.constant 0.0037739673859323599 : f64
    %cst_317 = arith.constant 0.20281109560651886 : f64
    %cst_318 = arith.constant 1.210000e+02 : f32
    %cst_319 = arith.constant 5.2931171733333332E-14 : f64
    %cst_320 = arith.constant 1.1021679075323599E-11 : f64
    %cst_321 = arith.constant 1.8934228504790033E-9 : f64
    %cst_322 = arith.constant 2.7479061117017636E-7 : f64
    %cst_323 = arith.constant 3.4096085096200906E-5 : f64
    %cst_324 = arith.constant 0.0036342240767211326 : f64
    %cst_325 = arith.constant 0.19540403413693969 : f64
    %cst_326 = arith.constant 1.190000e+02 : f32
    %cst_327 = arith.constant 5.019026583111111E-14 : f64
    %cst_328 = arith.constant 1.0403065638343878E-11 : f64
    %cst_329 = arith.constant 1.7863299617388377E-9 : f64
    %cst_330 = arith.constant 2.600757237588632E-7 : f64
    %cst_331 = arith.constant 3.2491914440014268E-5 : f64
    %cst_332 = arith.constant 0.0035010775057740316 : f64
    %cst_333 = arith.constant 0.18826980194443665 : f64
    %cst_334 = arith.constant 1.170000e+02 : f32
    %cst_335 = arith.constant 4.756541809777778E-14 : f64
    %cst_336 = arith.constant 9.8166442942854903E-12 : f64
    %cst_337 = arith.constant 1.6852609412267751E-9 : f64
    %cst_338 = arith.constant 2.4619326937592292E-7 : f64
    %cst_339 = arith.constant 3.0973511714709502E-5 : f64
    %cst_340 = arith.constant 0.0033741744168096998 : f64
    %cst_341 = arith.constant 0.18139556223643702 : f64
    %cst_342 = arith.constant 1.150000e+02 : f32
    %cst_343 = arith.constant 4.5054073102222224E-14 : f64
    %cst_344 = arith.constant 9.2610375235427364E-12 : f64
    %cst_345 = arith.constant 1.5899007843582445E-9 : f64
    %cst_346 = arith.constant 2.3309632627767074E-7 : f64
    %cst_347 = arith.constant 2.9536024347344365E-5 : f64
    %cst_348 = arith.constant 0.0032531815370903066 : f64
    %cst_349 = arith.constant 0.17476916455659369 : f64
    %cst_350 = arith.constant 1.130000e+02 : f32
    %cst_351 = arith.constant 4.2653528977777778E-14 : f64
    %cst_352 = arith.constant 8.734899366193081E-12 : f64
    %cst_353 = arith.constant 1.4999481055996091E-9 : f64
    %cst_354 = arith.constant 2.2074043807045782E-7 : f64
    %cst_355 = arith.constant 2.8174873844911173E-5 : f64
    %cst_356 = arith.constant 0.0031377844510793083 : f64
    %cst_357 = arith.constant 0.16837910595412131 : f64
    %cst_358 = arith.constant 1.110000e+02 : f32
    %cst_359 = arith.constant 4.0360957457777778E-14 : f64
    %cst_360 = arith.constant 8.2369170665974318E-12 : f64
    %cst_361 = arith.constant 1.4151148144240729E-9 : f64
    %cst_362 = arith.constant 2.0908350604346383E-7 : f64
    %cst_363 = arith.constant 2.6885741326534563E-5 : f64
    %cst_364 = arith.constant 0.0030276865332726477 : f64
    %cst_365 = arith.constant 0.16221449434620738 : f64
    %cst_366 = arith.constant 1.090000e+02 : f32
    %cst_367 = arith.constant 3.8173420035555557E-14 : f64
    %cst_368 = arith.constant 7.7658124891046765E-12 : f64
    %cst_369 = arith.constant 1.3351257759815557E-9 : f64
    %cst_370 = arith.constant 1.9808568415654462E-7 : f64
    %cst_371 = arith.constant 2.5664553693768451E-5 : f64
    %cst_372 = arith.constant 0.0029226079376196627 : f64
    %cst_373 = arith.constant 0.15626501395774611 : f64
    %cst_374 = arith.constant 1.070000e+02 : f32
    %cst_375 = arith.constant 3.6087889048888887E-14 : f64
    %cst_376 = arith.constant 7.320343304922983E-12 : f64
    %cst_377 = arith.constant 1.259718458758337E-9 : f64
    %cst_378 = arith.constant 1.8770927679626137E-7 : f64
    %cst_379 = arith.constant 2.4507470422713398E-5 : f64
    %cst_380 = arith.constant 0.0028222846410136237 : f64
    %cst_381 = arith.constant 0.15052089272774619 : f64
    %cst_382 = arith.constant 1.050000e+02 : f32
    %cst_383 = arith.constant 3.4101266222222225E-14 : f64
    %cst_384 = arith.constant 6.8993039665054284E-12 : f64
    %cst_385 = arith.constant 1.1886425714330958E-9 : f64
    %cst_386 = arith.constant 1.7791863939526378E-7 : f64
    %cst_387 = arith.constant 2.3410870961050949E-5 : f64
    %cst_388 = arith.constant 0.0027264675383982442 : f64
    %cst_389 = arith.constant 0.14497287157673799 : f64
    %cst_390 = arith.constant 1.030000e+02 : f32
    %cst_391 = arith.constant 3.2210394506666667E-14 : f64
    %cst_392 = arith.constant 6.5015264753090893E-12 : f64
    %cst_393 = arith.constant 1.1216596910444997E-9 : f64
    %cst_394 = arith.constant 1.6868008199296823E-7 : f64
    %cst_395 = arith.constant 2.2371342712572568E-5 : f64
    %cst_396 = arith.constant 0.0026349215871051762 : f64
    %cst_397 = arith.constant 0.13961217543434562 : f64
    %cst_398 = arith.constant 1.010000e+02 : f32
    %cst_399 = arith.constant 3.0412080142222222E-14 : f64
    %cst_400 = arith.constant 6.1258809536787883E-12 : f64
    %cst_401 = arith.constant 1.0585428844575133E-9 : f64
    %cst_402 = arith.constant 1.5996177579900442E-7 : f64
    %cst_403 = arith.constant 2.1385669591362916E-5 : f64
    %cst_404 = arith.constant 0.0025474249981080823 : f64
    %cst_405 = arith.constant 0.13443048593088697 : f64
    %cst_406 = arith.constant 9.900000e+01 : f32
    %cst_407 = arith.constant 2.8703099555555555E-14 : f64
    %cst_408 = arith.constant 5.7712760311351628E-12 : f64
    %cst_409 = arith.constant 9.9907632506389029E-10 : f64
    %cst_410 = arith.constant 1.5173366280523906E-7 : f64
    %cst_411 = arith.constant 2.0450821127475878E-5 : f64
    %cst_412 = arith.constant 0.0024637684719508858 : f64
    %cst_413 = arith.constant 0.12941991566142438 : f64
    %cst_414 = arith.constant 9.700000e+01 : f32
    %cst_415 = arith.constant 2.708022592E-14 : f64
    %cst_416 = arith.constant 5.4366590583134219E-12 : f64
    %cst_417 = arith.constant 9.4305490646459255E-10 : f64
    %cst_418 = arith.constant 1.439673684773947E-7 : f64
    %cst_419 = arith.constant 1.9563942105711612E-5 : f64
    %cst_420 = arith.constant 0.0023837544771809576 : f64
    %cst_421 = arith.constant 0.12457298393509812 : f64
    %cst_422 = arith.constant 9.500000e+01 : f32
    %cst_423 = arith.constant 2.554022711111111E-14 : f64
    %cst_424 = arith.constant 5.121016156922585E-12 : f64
    %cst_425 = arith.constant 8.9028385488493283E-10 : f64
    %cst_426 = arith.constant 1.3663611754337958E-7 : f64
    %cst_427 = arith.constant 1.8722342718958937E-5 : f64
    %cst_428 = arith.constant 0.0023071965691918691 : f64
    %cst_429 = arith.constant 0.11988259392684095 : f64
    %cst_430 = arith.constant 9.300000e+01 : f32
    %cst_431 = arith.constant 2.4079890062222222E-14 : f64
    %cst_432 = arith.constant 4.8233721206418025E-12 : f64
    %cst_433 = arith.constant 8.4057834180389072E-10 : f64
    %cst_434 = arith.constant 1.2971465288245997E-7 : f64
    %cst_435 = arith.constant 1.7923489217504226E-5 : f64
    %cst_436 = arith.constant 0.0022339187474546421 : f64
    %cst_437 = arith.constant 0.11534201115268805 : f64
    %cst_438 = arith.constant 9.100000e+01 : f32
    %cst_439 = arith.constant 2.2696025653333333E-14 : f64
    %cst_440 = arith.constant 4.5427901763106357E-12 : f64
    %cst_441 = arith.constant 7.937630983149963E-10 : f64
    %cst_442 = arith.constant 1.2317915750735938E-7 : f64
    %cst_443 = arith.constant 1.7164995035719656E-5 : f64
    %cst_444 = arith.constant 0.002163754849190817 : f64
    %cst_445 = arith.constant 0.11094484319386444 : f64
    %cst_446 = arith.constant 8.900000e+01 : f32
    %cst_447 = arith.constant 2.1385479360000001E-14 : f64
    %cst_448 = arith.constant 4.2783716186085925E-12 : f64
    %cst_449 = arith.constant 7.4967203250938422E-10 : f64
    %cst_450 = arith.constant 1.1700717962026153E-7 : f64
    %cst_451 = arith.constant 1.6444612377624982E-5 : f64
    %cst_452 = arith.constant 0.002096547977614873 : f64
    %cst_453 = arith.constant 0.10668502059865094 : f64
    %cst_454 = arith.constant 8.700000e+01 : f32
    %cst_455 = arith.constant 2.0145143075555556E-14 : f64
    %cst_456 = arith.constant 4.0292553276632561E-12 : f64
    %cst_457 = arith.constant 7.0814785110097659E-10 : f64
    %cst_458 = arith.constant 1.1117756071353507E-7 : f64
    %cst_459 = arith.constant 1.5760224242962179E-5 : f64
    %cst_460 = arith.constant 0.0020321499629472857 : f64
    %cst_461 = arith.constant 0.1025567788947009 : f64
    %cst_462 = arith.constant 8.500000e+01 : f32
    %cst_463 = arith.constant 1.8971959040000001E-14 : f64
    %cst_464 = arith.constant 3.7946171850824336E-12 : f64
    %cst_465 = arith.constant 6.690416864001935E-10 : f64
    %cst_466 = arith.constant 1.0567036667675984E-7 : f64
    %cst_467 = arith.constant 1.5109836875625445E-5 : f64
    %cst_468 = arith.constant 0.0019704208544725622 : f64
    %cst_469 = arith.constant 0.098554641648004454 : f64
    %cst_470 = arith.constant 8.300000e+01 : f32
    %cst_471 = arith.constant 1.786293159111111E-14 : f64
    %cst_472 = arith.constant 3.573669397558913E-12 : f64
    %cst_473 = arith.constant 6.3221272959790998E-10 : f64
    %cst_474 = arith.constant 1.0046682186333614E-7 : f64
    %cst_475 = arith.constant 1.4491572616545005E-5 : f64
    %cst_476 = arith.constant 0.0019112284419887304 : f64
    %cst_477 = arith.constant 0.094673404508075487 : f64
    %cst_478 = arith.constant 8.100000e+01 : f32
    %cst_479 = arith.constant 1.6815130613333334E-14 : f64
    %cst_480 = arith.constant 3.3656597366099099E-12 : f64
    %cst_481 = arith.constant 5.9752787125242054E-10 : f64
    %cst_482 = arith.constant 9.5549246062549912E-8 : f64
    %cst_483 = arith.constant 1.390366314342612E-5 : f64
    %cst_484 = arith.constant 0.0018544478050657701 : f64
    %cst_485 = arith.constant 0.090908120182172744 : f64
    %cst_486 = arith.constant 7.900000e+01 : f32
    %cst_487 = arith.constant 1.5825697795555556E-14 : f64
    %cst_488 = arith.constant 3.1698707080033958E-12 : f64
    %cst_489 = arith.constant 5.6486134972616462E-10 : f64
    %cst_490 = arith.constant 9.0900994316429002E-8 : f64
    %cst_491 = arith.constant 1.3344443080089493E-5 : f64
    %cst_492 = arith.constant 0.0017999608886001962 : f64
    %cst_493 = arith.constant 0.087254084284461713 : f64
    %cst_494 = arith.constant 7.700000e+01 : f32
    %cst_495 = arith.constant 1.489185159111111E-14 : f64
    %cst_496 = arith.constant 2.9856186620887555E-12 : f64
    %cst_497 = arith.constant 5.3409440823869462E-10 : f64
    %cst_498 = arith.constant 8.6506399515036437E-8 : f64
    %cst_499 = arith.constant 1.2812343958540764E-5 : f64
    %cst_500 = arith.constant 0.0017476561032212657 : f64
    %cst_501 = arith.constant 0.083706822008980358 : f64
    %cst_502 = arith.constant 7.500000e+01 : f32
    %cst_503 = arith.constant 1.4010889635555555E-14 : f64
    %cst_504 = arith.constant 2.8122528497626899E-12 : f64
    %cst_505 = arith.constant 5.0511496109857112E-10 : f64
    %cst_506 = arith.constant 8.2350717698979039E-8 : f64
    %cst_507 = arith.constant 1.2305888517309891E-5 : f64
    %cst_508 = arith.constant 0.0016974279491709504 : f64
    %cst_509 = arith.constant 0.080262075578094616 : f64
    %cst_510 = arith.constant 7.300000e+01 : f32
    %cst_511 = arith.constant 1.3180196462222222E-14 : f64
    %cst_512 = arith.constant 2.6491544403815725E-12 : f64
    %cst_513 = arith.constant 4.7781726956916479E-10 : f64
    %cst_514 = arith.constant 7.8420075993781542E-8 : f64
    %cst_515 = arith.constant 1.1823685320041301E-5 : f64
    %cst_516 = arith.constant 0.0016491766623447889 : f64
    %cst_517 = arith.constant 0.076915792420819556 : f64
    %cst_518 = arith.constant 7.100000e+01 : f32
    %cst_519 = arith.constant 1.2397238257777777E-14 : f64
    %cst_520 = arith.constant 2.4957355004088569E-12 : f64
    %cst_521 = arith.constant 4.5210162777476488E-10 : f64
    %cst_522 = arith.constant 7.4701423097423176E-8 : f64
    %cst_523 = arith.constant 1.1364423678778208E-5 : f64
    %cst_524 = arith.constant 0.0016028078812438821 : f64
    %cst_525 = arith.constant 0.073664114037944595 : f64
    %cst_526 = arith.constant 6.900000e+01 : f32
    %cst_527 = arith.constant 1.1659571751111111E-14 : f64
    %cst_528 = arith.constant 2.3514379522274418E-12 : f64
    %cst_529 = arith.constant 4.2787405890153386E-10 : f64
    %cst_530 = arith.constant 7.1182482239613507E-8 : f64
    %cst_531 = arith.constant 1.092686886686523E-5 : f64
    %cst_532 = arith.constant 0.001558232333649571 : f64
    %cst_533 = arith.constant 0.070503365513338856 : f64
    %cst_534 = arith.constant 6.700000e+01 : f32
    %cst_535 = arith.constant 1.0964842115555555E-14 : f64
    %cst_536 = arith.constant 2.2157325110542536E-12 : f64
    %cst_537 = arith.constant 4.0504602194811138E-10 : f64
    %cst_538 = arith.constant 6.7851706529363335E-8 : f64
    %cst_539 = arith.constant 1.0509857606888329E-5 : f64
    %cst_540 = arith.constant 0.001515365541891654 : f64
    %cst_541 = arith.constant 0.067430045633130389 : f64
    %cst_542 = arith.constant 6.500000e+01 : f32
    %cst_543 = arith.constant 1.031078448E-14 : f64
    %cst_544 = arith.constant 2.088117611438512E-12 : f64
    %cst_545 = arith.constant 3.8353412915303665E-10 : f64
    %cst_546 = arith.constant 6.4698236605933248E-8 : f64
    %cst_547 = arith.constant 1.0112293819576438E-5 : f64
    %cst_548 = arith.constant 0.0014741275456383132 : f64
    %cst_549 = arith.constant 0.064440817576653292 : f64
    %cst_550 = arith.constant 6.300000e+01 : f32
    %cst_551 = arith.constant 9.6952238400000006E-15 : f64
    %cst_552 = arith.constant 1.9681183310134517E-12 : f64
    %cst_553 = arith.constant 3.6325987418295298E-10 : f64
    %cst_554 = arith.constant 6.1711860507347178E-8 : f64
    %cst_555 = arith.constant 9.7331446201016809E-6 : f64
    %cst_556 = arith.constant 0.0014344426411912014 : f64
    %cst_557 = arith.constant 0.061532500145144775 : f64
    %cst_558 = arith.constant 6.100000e+01 : f32
    %cst_559 = arith.constant 9.1160736711111105E-15 : f64
    %cst_560 = arith.constant 1.8552853109751858E-12 : f64
    %cst_561 = arith.constant 3.4414937110591756E-10 : f64
    %cst_562 = arith.constant 5.8882975670265285E-8 : f64
    %cst_563 = arith.constant 9.3714365487312788E-6 : f64
    %cst_564 = arith.constant 0.0013962391363223647 : f64
    %cst_565 = arith.constant 0.058702059496154084 : f64
    %cst_566 = arith.constant 5.900000e+01 : f32
    %cst_567 = arith.constant 8.5713381688888883E-15 : f64
    %cst_568 = arith.constant 1.7491936862246368E-12 : f64
    %cst_569 = arith.constant 3.2613310410503139E-10 : f64
    %cst_570 = arith.constant 5.6202552975056696E-8 : f64
    %cst_571 = arith.constant 9.0262520233016382E-6 : f64
    %cst_572 = arith.constant 0.001359449119740819 : f64
    %cst_573 = arith.constant 0.055946601353500013 : f64
    %cst_574 = arith.constant 5.700000e+01 : f32
    %cst_575 = arith.constant 8.0591079644444444E-15 : f64
    %cst_576 = arith.constant 1.6494420240828494E-12 : f64
    %cst_577 = arith.constant 3.0914568786634799E-10 : f64
    %cst_578 = arith.constant 5.3662102750396798E-8 : f64
    %cst_579 = arith.constant 8.6967260015007666E-6 : f64
    %cst_580 = arith.constant 0.0013240082443256975 : f64
    %cst_581 = arith.constant 0.053263363664388864 : f64
    %cst_582 = arith.constant 5.500000e+01 : f32
    %cst_583 = arith.constant 7.5775607822222226E-15 : f64
    %cst_584 = arith.constant 1.5556512782814827E-12 : f64
    %cst_585 = arith.constant 2.9312563849675507E-10 : f64
    %cst_586 = arith.constant 5.1253642652551842E-8 : f64
    %cst_587 = arith.constant 8.3820428414568805E-6 : f64
    %cst_588 = arith.constant 0.0012898555233099055 : f64
    %cst_589 = arith.constant 0.050649709676983339 : f64
    %cst_590 = arith.constant 5.300000e+01 : f32
    %cst_591 = arith.constant 7.124958935111111E-15 : f64
    %cst_592 = arith.constant 1.4674637611609885E-12 : f64
    %cst_593 = arith.constant 2.7801515481905746E-10 : f64
    %cst_594 = arith.constant 4.896966733568202E-8 : f64
    %cst_595 = arith.constant 8.081433349636768E-6 : f64
    %cst_596 = arith.constant 0.0012569331386432195 : f64
    %cst_597 = arith.constant 0.048103121413299865 : f64
    %cst_598 = arith.constant 5.100000e+01 : f32
    %cst_599 = arith.constant 6.6996477404444446E-15 : f64
    %cst_600 = arith.constant 1.384542137097712E-12 : f64
    %cst_601 = arith.constant 2.6375990983978426E-10 : f64
    %cst_602 = arith.constant 4.6803119830954462E-8 : f64
    %cst_603 = arith.constant 7.7941720055551918E-6 : f64
    %cst_604 = arith.constant 0.001225186260806753 : f64
    %cst_605 = arith.constant 0.045621193513810471 : f64
    %cst_606 = arith.constant 4.900000e+01 : f32
    %cst_607 = arith.constant 6.3000532853333337E-15 : f64
    %cst_608 = arith.constant 1.3065684400300477E-12 : f64
    %cst_609 = arith.constant 2.5030885216472953E-10 : f64
    %cst_610 = arith.constant 4.4747364553960991E-8 : f64
    %cst_611 = arith.constant 7.5195743532849203E-6 : f64
    %cst_612 = arith.constant 0.0011945628793917271 : f64
    %cst_613 = arith.constant 0.04320162743154022 : f64
    %cst_614 = arith.constant 4.700000e+01 : f32
    %cst_615 = arith.constant 5.9246802364444444E-15 : f64
    %cst_616 = arith.constant 1.2332431172381557E-12 : f64
    %cst_617 = arith.constant 2.3761401711005023E-10 : f64
    %cst_618 = arith.constant 4.2796161861855041E-8 : f64
    %cst_619 = arith.constant 7.2569945502343004E-6 : f64
    %cst_620 = arith.constant 0.0011650136437945675 : f64
    %cst_621 = arith.constant 0.040842225954785961 : f64
    %cst_622 = arith.constant 4.500000e+01 : f32
    %cst_623 = arith.constant 5.5721092871111108E-15 : f64
    %cst_624 = arith.constant 1.1642841011361993E-12 : f64
    %cst_625 = arith.constant 2.2563034723692883E-10 : f64
    %cst_626 = arith.constant 4.0943644083718584E-8 : f64
    %cst_627 = arith.constant 7.0058230641246313E-6 : f64
    %cst_628 = arith.constant 0.0011364917134175419 : f64
    %cst_629 = arith.constant 0.03854088803884051 : f64
    %cst_630 = arith.constant 4.300000e+01 : f32
    %cst_631 = arith.constant 5.240994910222222E-15 : f64
    %cst_632 = arith.constant 1.0994259106646732E-12 : f64
    %cst_633 = arith.constant 2.1431552202133775E-10 : f64
    %cst_634 = arith.constant 3.9184292949913591E-8 : f64
    %cst_635 = arith.constant 6.7654845095518363E-6 : f64
    %cst_636 = arith.constant 0.0011089526167995269 : f64
    %cst_637 = arith.constant 0.036295603928292425 : f64
    %cst_638 = arith.constant 4.100000e+01 : f32
    %cst_639 = arith.constant 4.9300625262222223E-15 : f64
    %cst_640 = arith.constant 1.0384187833037281E-12 : f64
    %cst_641 = arith.constant 2.0362979635817883E-10 : f64
    %cst_642 = arith.constant 3.7512918348533524E-8 : f64
    %cst_643 = arith.constant 6.5354356159553933E-6 : f64
    %cst_644 = arith.constant 0.0010823541191350532 : f64
    %cst_645 = arith.constant 0.034104450552588338 : f64
    %cst_646 = arith.constant 3.900000e+01 : f32
    %cst_647 = arith.constant 4.6381060817777776E-15 : f64
    %cst_648 = arith.constant 9.8102783859889256E-13 : f64
    %cst_649 = arith.constant 1.9353584758781173E-10 : f64
    %cst_650 = arith.constant 3.5924638339521923E-8 : f64
    %cst_651 = arith.constant 6.3151633192414583E-6 : f64
    %cst_652 = arith.constant 0.0010566560976716574 : f64
    %cst_653 = arith.constant 0.031965587178596447 : f64
    %cst_654 = arith.constant 3.700000e+01 : f32
    %cst_655 = arith.constant 4.3639844053333335E-15 : f64
    %cst_656 = arith.constant 9.2703227366365038E-13 : f64
    %cst_657 = arith.constant 1.839986307293409E-10 : f64
    %cst_658 = arith.constant 3.4414860359542718E-8 : f64
    %cst_659 = arith.constant 6.1041829697162056E-6 : f64
    %cst_660 = arith.constant 0.001031820424505735 : f64
    %cst_661 = arith.constant 0.029877251304899308 : f64
    %cst_662 = arith.constant 3.500000e+01 : f32
    %cst_663 = arith.constant 4.106620648888889E-15 : f64
    %cst_664 = arith.constant 8.7622459124842529E-13 : f64
    %cst_665 = arith.constant 1.7498524159268457E-10 : f64
    %cst_666 = arith.constant 3.2979263553246519E-8 : f64
    %cst_667 = arith.constant 5.9020366493792216E-6 : f64
    %cst_668 = arith.constant 0.0010078108563256892 : f64
    %cst_669 = arith.constant 0.027837754783474698 : f64
    %cst_670 = arith.constant 3.300000e+01 : f32
    %cst_671 = arith.constant 3.8649975768888887E-15 : f64
    %cst_672 = arith.constant 8.2840985928785404E-13 : f64
    %cst_673 = arith.constant 1.664647874552963E-10 : f64
    %cst_674 = arith.constant 3.1613782169164832E-8 : f64
    %cst_675 = arith.constant 5.7082915920051847E-6 : f64
    %cst_676 = arith.constant 9.8459293067820123E-4 : f64
    %cst_677 = arith.constant 0.025845480155298518 : f64
    %cst_678 = arith.constant 3.100000e+01 : f32
    %cst_679 = arith.constant 3.6381553564444445E-15 : f64
    %cst_680 = arith.constant 7.8340500472414453E-13 : f64
    %cst_681 = arith.constant 1.5840826497296334E-10 : f64
    %cst_682 = arith.constant 3.0314589961047687E-8 : f64
    %cst_683 = arith.constant 5.5225386998049015E-6 : f64
    %cst_684 = arith.constant 9.6213386835900178E-4 : f64
    %cst_685 = arith.constant 0.02389887718722632 : f64
    %cst_686 = arith.constant 2.900000e+01 : f32
    %cst_687 = arith.constant 3.4251892319999999E-15 : f64
    %cst_688 = arith.constant 7.4103813647499203E-13 : f64
    %cst_689 = arith.constant 1.507884450032973E-10 : f64
    %cst_690 = arith.constant 2.9078085538049375E-8 : f64
    %cst_691 = arith.constant 5.3443911508041167E-6 : f64
    %cst_692 = arith.constant 9.4040248155366783E-4 : f64
    %cst_693 = arith.constant 0.021996459598282742 : f64
    %cst_694 = arith.constant 2.700000e+01 : f32
    %cst_695 = arith.constant 3.2252476000000001E-15 : f64
    %cst_696 = arith.constant 7.011479031104373E-13 : f64
    %cst_697 = arith.constant 1.4357976402809041E-10 : f64
    %cst_698 = arith.constant 2.7900878609710433E-8 : f64
    %cst_699 = arith.constant 5.1734830914104276E-6 : f64
    %cst_700 = arith.constant 9.1936908737673677E-4 : f64
    %cst_701 = arith.constant 0.020136801964214277 : f64
    %cst_702 = arith.constant 2.500000e+01 : f32
    %cst_703 = arith.constant 3.0375273884444443E-15 : f64
    %cst_704 = arith.constant 6.6358287745352708E-13 : f64
    %cst_705 = arith.constant 1.3675822186304616E-10 : f64
    %cst_706 = arith.constant 2.677977707421807E-8 : f64
    %cst_707 = arith.constant 5.0094684089553369E-6 : f64
    %cst_708 = arith.constant 8.990054264789172E-4 : f64
    %cst_709 = arith.constant 0.018318536789842393 : f64
    %cst_710 = arith.constant 2.300000e+01 : f32
    %cst_711 = arith.constant 2.8612737351111112E-15 : f64
    %cst_712 = arith.constant 6.282009758687478E-13 : f64
    %cst_713 = arith.constant 1.3030128534230821E-10 : f64
    %cst_714 = arith.constant 2.571177490088171E-8 : f64
    %cst_715 = arith.constant 4.8520195793001752E-6 : f64
    %cst_716 = arith.constant 8.7928458641241465E-4 : f64
    %cst_717 = arith.constant 0.016540351739394069 : f64
    %cst_718 = arith.constant 2.100000e+01 : f32
    %cst_719 = arith.constant 2.6957764568888888E-15 : f64
    %cst_720 = arith.constant 5.9486890370320262E-13 : f64
    %cst_721 = arith.constant 1.2418779768752298E-10 : f64
    %cst_722 = arith.constant 2.4694040760197315E-8 : f64
    %cst_723 = arith.constant 4.7008265848816866E-6 : f64
    %cst_724 = arith.constant 8.6018092946345943E-4 : f64
    %cst_725 = arith.constant 0.014800987015587536 : f64
    %cst_726 = arith.constant 1.900000e+01 : f32
    %cst_727 = arith.constant 2.5403679644444446E-15 : f64
    %cst_728 = arith.constant 5.634616306755024E-13 : f64
    %cst_729 = arith.constant 1.1839789326602696E-10 : f64
    %cst_730 = arith.constant 2.3723907357214174E-8 : f64
    %cst_731 = arith.constant 4.5555958988457508E-6 : f64
    %cst_732 = arith.constant 8.4167002467906964E-4 : f64
    %cst_733 = arith.constant 0.013099232878814654 : f64
    %cst_734 = arith.constant 1.700000e+01 : f32
    %cst_735 = arith.constant 2.3944209546666666E-15 : f64
    %cst_736 = arith.constant 5.3386189365816881E-13 : f64
    %cst_737 = arith.constant 1.129129174587924E-10 : f64
    %cst_738 = arith.constant 2.2798861426211987E-8 : f64
    %cst_739 = arith.constant 4.4160495311765436E-6 : f64
    %cst_740 = arith.constant 8.2372858383196565E-4 : f64
    %cst_741 = arith.constant 0.011433927298290302 : f64
    %cst_742 = arith.constant 1.500000e+01 : f32
    %cst_743 = arith.constant 2.2573462684444446E-15 : f64
    %cst_744 = arith.constant 5.0595972623692818E-13 : f64
    %cst_745 = arith.constant 1.0771535136565471E-10 : f64
    %cst_746 = arith.constant 2.1916534346907168E-8 : f64
    %cst_747 = arith.constant 4.2819241329736986E-6 : f64
    %cst_748 = arith.constant 8.0633440108342842E-4 : f64
    %cst_749 = arith.constant 0.0098039537275352185 : f64
    %cst_750 = arith.constant 1.300000e+01 : f32
    %cst_751 = arith.constant 2.1285907413333335E-15 : f64
    %cst_752 = arith.constant 4.7965201390613336E-13 : f64
    %cst_753 = arith.constant 1.0278874108587318E-10 : f64
    %cst_754 = arith.constant 2.1074693344544657E-8 : f64
    %cst_755 = arith.constant 4.1529701552622653E-6 : f64
    %cst_756 = arith.constant 7.8946629611881714E-4 : f64
    %cst_757 = arith.constant 0.0082082389970241208 : f64
    %cst_758 = arith.constant 1.100000e+01 : f32
    %cst_759 = arith.constant 2.0076352213333332E-15 : f64
    %cst_760 = arith.constant 4.5484207406017752E-13 : f64
    %cst_761 = arith.constant 9.8117631321709102E-11 : f64
    %cst_762 = arith.constant 2.0271233238288382E-8 : f64
    %cst_763 = arith.constant 4.028951058939944E-6 : f64
    %cst_764 = arith.constant 7.7310406054447453E-4 : f64
    %cst_765 = arith.constant 0.0066457513172673054 : f64
    %cst_766 = arith.constant 9.000000e+00 : f32
    %cst_767 = arith.constant 1.8939926435555556E-15 : f64
    %cst_768 = arith.constant 4.3143925959079665E-13 : f64
    %cst_769 = arith.constant 9.3687503063178996E-11 : f64
    %cst_770 = arith.constant 1.9504168704300469E-8 : f64
    %cst_771 = arith.constant 3.9096425726735704E-6 : f64
    %cst_772 = arith.constant 7.5722840734791656E-4 : f64
    %cst_773 = arith.constant 0.0051154983860031977 : f64
    %cst_774 = arith.constant 7.000000e+00 : f32
    %cst_775 = arith.constant 1.787206146488889E-15 : f64
    %cst_776 = arith.constant 4.0935858517772438E-13 : f64
    %cst_777 = arith.constant 8.948471512241509E-11 : f64
    %cst_778 = arith.constant 1.8771627021793087E-8 : f64
    %cst_779 = arith.constant 3.794831995752824E-6 : f64
    %cst_780 = arith.constant 7.4182092323555509E-4 : f64
    %cst_781 = arith.constant 0.0036165255935630175 : f64
    %cst_782 = arith.constant 5.000000e+00 : f32
    %cst_783 = arith.constant 1.6868473576888889E-15 : f64
    %cst_784 = arith.constant 3.8852037518534289E-13 : f64
    %cst_785 = arith.constant 8.5496449296040329E-11 : f64
    %cst_786 = arith.constant 1.8071841272149201E-8 : f64
    %cst_787 = arith.constant 3.6843175430938994E-6 : f64
    %cst_788 = arith.constant 7.2686402367379993E-4 : f64
    %cst_789 = arith.constant 0.0021479143208285143 : f64
    %cst_790 = arith.constant 3.000000e+00 : f32
    %false = arith.constant false
    %cst_791 = arith.constant 1.5917038551111112E-15 : f64
    %cst_792 = arith.constant 3.6885022360434958E-13 : f64
    %cst_793 = arith.constant 8.1710660047307792E-11 : f64
    %cst_794 = arith.constant 1.7403143962587938E-8 : f64
    %cst_795 = arith.constant 3.5779077297597742E-6 : f64
    %cst_796 = arith.constant 7.1234091047026298E-4 : f64
    %cst_797 = arith.constant 7.0878032454106438E-4 : f64
    %cst_798 = arith.constant 1.000000e+00 : f32
    %cst_799 = arith.constant 2.000000e+00 : f32
    %0 = llvm.mlir.undef : f32
    %1 = arith.fptosi %arg0 : f32 to i32
    cf.switch %1 : i32, [
      default: ^bb2(%true, %0 : i1, f32),
      0: ^bb1(%cst_798, %cst_791, %cst_792, %cst_793, %cst_794, %cst_795, %cst_796, %cst_797 : f32, f64, f64, f64, f64, f64, f64, f64),
      1: ^bb1(%cst_790, %cst_783, %cst_784, %cst_785, %cst_786, %cst_787, %cst_788, %cst_789 : f32, f64, f64, f64, f64, f64, f64, f64),
      2: ^bb1(%cst_782, %cst_775, %cst_776, %cst_777, %cst_778, %cst_779, %cst_780, %cst_781 : f32, f64, f64, f64, f64, f64, f64, f64),
      3: ^bb1(%cst_774, %cst_767, %cst_768, %cst_769, %cst_770, %cst_771, %cst_772, %cst_773 : f32, f64, f64, f64, f64, f64, f64, f64),
      4: ^bb1(%cst_766, %cst_759, %cst_760, %cst_761, %cst_762, %cst_763, %cst_764, %cst_765 : f32, f64, f64, f64, f64, f64, f64, f64),
      5: ^bb1(%cst_758, %cst_751, %cst_752, %cst_753, %cst_754, %cst_755, %cst_756, %cst_757 : f32, f64, f64, f64, f64, f64, f64, f64),
      6: ^bb1(%cst_750, %cst_743, %cst_744, %cst_745, %cst_746, %cst_747, %cst_748, %cst_749 : f32, f64, f64, f64, f64, f64, f64, f64),
      7: ^bb1(%cst_742, %cst_735, %cst_736, %cst_737, %cst_738, %cst_739, %cst_740, %cst_741 : f32, f64, f64, f64, f64, f64, f64, f64),
      8: ^bb1(%cst_734, %cst_727, %cst_728, %cst_729, %cst_730, %cst_731, %cst_732, %cst_733 : f32, f64, f64, f64, f64, f64, f64, f64),
      9: ^bb1(%cst_726, %cst_719, %cst_720, %cst_721, %cst_722, %cst_723, %cst_724, %cst_725 : f32, f64, f64, f64, f64, f64, f64, f64),
      10: ^bb1(%cst_718, %cst_711, %cst_712, %cst_713, %cst_714, %cst_715, %cst_716, %cst_717 : f32, f64, f64, f64, f64, f64, f64, f64),
      11: ^bb1(%cst_710, %cst_703, %cst_704, %cst_705, %cst_706, %cst_707, %cst_708, %cst_709 : f32, f64, f64, f64, f64, f64, f64, f64),
      12: ^bb1(%cst_702, %cst_695, %cst_696, %cst_697, %cst_698, %cst_699, %cst_700, %cst_701 : f32, f64, f64, f64, f64, f64, f64, f64),
      13: ^bb1(%cst_694, %cst_687, %cst_688, %cst_689, %cst_690, %cst_691, %cst_692, %cst_693 : f32, f64, f64, f64, f64, f64, f64, f64),
      14: ^bb1(%cst_686, %cst_679, %cst_680, %cst_681, %cst_682, %cst_683, %cst_684, %cst_685 : f32, f64, f64, f64, f64, f64, f64, f64),
      15: ^bb1(%cst_678, %cst_671, %cst_672, %cst_673, %cst_674, %cst_675, %cst_676, %cst_677 : f32, f64, f64, f64, f64, f64, f64, f64),
      16: ^bb1(%cst_670, %cst_663, %cst_664, %cst_665, %cst_666, %cst_667, %cst_668, %cst_669 : f32, f64, f64, f64, f64, f64, f64, f64),
      17: ^bb1(%cst_662, %cst_655, %cst_656, %cst_657, %cst_658, %cst_659, %cst_660, %cst_661 : f32, f64, f64, f64, f64, f64, f64, f64),
      18: ^bb1(%cst_654, %cst_647, %cst_648, %cst_649, %cst_650, %cst_651, %cst_652, %cst_653 : f32, f64, f64, f64, f64, f64, f64, f64),
      19: ^bb1(%cst_646, %cst_639, %cst_640, %cst_641, %cst_642, %cst_643, %cst_644, %cst_645 : f32, f64, f64, f64, f64, f64, f64, f64),
      20: ^bb1(%cst_638, %cst_631, %cst_632, %cst_633, %cst_634, %cst_635, %cst_636, %cst_637 : f32, f64, f64, f64, f64, f64, f64, f64),
      21: ^bb1(%cst_630, %cst_623, %cst_624, %cst_625, %cst_626, %cst_627, %cst_628, %cst_629 : f32, f64, f64, f64, f64, f64, f64, f64),
      22: ^bb1(%cst_622, %cst_615, %cst_616, %cst_617, %cst_618, %cst_619, %cst_620, %cst_621 : f32, f64, f64, f64, f64, f64, f64, f64),
      23: ^bb1(%cst_614, %cst_607, %cst_608, %cst_609, %cst_610, %cst_611, %cst_612, %cst_613 : f32, f64, f64, f64, f64, f64, f64, f64),
      24: ^bb1(%cst_606, %cst_599, %cst_600, %cst_601, %cst_602, %cst_603, %cst_604, %cst_605 : f32, f64, f64, f64, f64, f64, f64, f64),
      25: ^bb1(%cst_598, %cst_591, %cst_592, %cst_593, %cst_594, %cst_595, %cst_596, %cst_597 : f32, f64, f64, f64, f64, f64, f64, f64),
      26: ^bb1(%cst_590, %cst_583, %cst_584, %cst_585, %cst_586, %cst_587, %cst_588, %cst_589 : f32, f64, f64, f64, f64, f64, f64, f64),
      27: ^bb1(%cst_582, %cst_575, %cst_576, %cst_577, %cst_578, %cst_579, %cst_580, %cst_581 : f32, f64, f64, f64, f64, f64, f64, f64),
      28: ^bb1(%cst_574, %cst_567, %cst_568, %cst_569, %cst_570, %cst_571, %cst_572, %cst_573 : f32, f64, f64, f64, f64, f64, f64, f64),
      29: ^bb1(%cst_566, %cst_559, %cst_560, %cst_561, %cst_562, %cst_563, %cst_564, %cst_565 : f32, f64, f64, f64, f64, f64, f64, f64),
      30: ^bb1(%cst_558, %cst_551, %cst_552, %cst_553, %cst_554, %cst_555, %cst_556, %cst_557 : f32, f64, f64, f64, f64, f64, f64, f64),
      31: ^bb1(%cst_550, %cst_543, %cst_544, %cst_545, %cst_546, %cst_547, %cst_548, %cst_549 : f32, f64, f64, f64, f64, f64, f64, f64),
      32: ^bb1(%cst_542, %cst_535, %cst_536, %cst_537, %cst_538, %cst_539, %cst_540, %cst_541 : f32, f64, f64, f64, f64, f64, f64, f64),
      33: ^bb1(%cst_534, %cst_527, %cst_528, %cst_529, %cst_530, %cst_531, %cst_532, %cst_533 : f32, f64, f64, f64, f64, f64, f64, f64),
      34: ^bb1(%cst_526, %cst_519, %cst_520, %cst_521, %cst_522, %cst_523, %cst_524, %cst_525 : f32, f64, f64, f64, f64, f64, f64, f64),
      35: ^bb1(%cst_518, %cst_511, %cst_512, %cst_513, %cst_514, %cst_515, %cst_516, %cst_517 : f32, f64, f64, f64, f64, f64, f64, f64),
      36: ^bb1(%cst_510, %cst_503, %cst_504, %cst_505, %cst_506, %cst_507, %cst_508, %cst_509 : f32, f64, f64, f64, f64, f64, f64, f64),
      37: ^bb1(%cst_502, %cst_495, %cst_496, %cst_497, %cst_498, %cst_499, %cst_500, %cst_501 : f32, f64, f64, f64, f64, f64, f64, f64),
      38: ^bb1(%cst_494, %cst_487, %cst_488, %cst_489, %cst_490, %cst_491, %cst_492, %cst_493 : f32, f64, f64, f64, f64, f64, f64, f64),
      39: ^bb1(%cst_486, %cst_479, %cst_480, %cst_481, %cst_482, %cst_483, %cst_484, %cst_485 : f32, f64, f64, f64, f64, f64, f64, f64),
      40: ^bb1(%cst_478, %cst_471, %cst_472, %cst_473, %cst_474, %cst_475, %cst_476, %cst_477 : f32, f64, f64, f64, f64, f64, f64, f64),
      41: ^bb1(%cst_470, %cst_463, %cst_464, %cst_465, %cst_466, %cst_467, %cst_468, %cst_469 : f32, f64, f64, f64, f64, f64, f64, f64),
      42: ^bb1(%cst_462, %cst_455, %cst_456, %cst_457, %cst_458, %cst_459, %cst_460, %cst_461 : f32, f64, f64, f64, f64, f64, f64, f64),
      43: ^bb1(%cst_454, %cst_447, %cst_448, %cst_449, %cst_450, %cst_451, %cst_452, %cst_453 : f32, f64, f64, f64, f64, f64, f64, f64),
      44: ^bb1(%cst_446, %cst_439, %cst_440, %cst_441, %cst_442, %cst_443, %cst_444, %cst_445 : f32, f64, f64, f64, f64, f64, f64, f64),
      45: ^bb1(%cst_438, %cst_431, %cst_432, %cst_433, %cst_434, %cst_435, %cst_436, %cst_437 : f32, f64, f64, f64, f64, f64, f64, f64),
      46: ^bb1(%cst_430, %cst_423, %cst_424, %cst_425, %cst_426, %cst_427, %cst_428, %cst_429 : f32, f64, f64, f64, f64, f64, f64, f64),
      47: ^bb1(%cst_422, %cst_415, %cst_416, %cst_417, %cst_418, %cst_419, %cst_420, %cst_421 : f32, f64, f64, f64, f64, f64, f64, f64),
      48: ^bb1(%cst_414, %cst_407, %cst_408, %cst_409, %cst_410, %cst_411, %cst_412, %cst_413 : f32, f64, f64, f64, f64, f64, f64, f64),
      49: ^bb1(%cst_406, %cst_399, %cst_400, %cst_401, %cst_402, %cst_403, %cst_404, %cst_405 : f32, f64, f64, f64, f64, f64, f64, f64),
      50: ^bb1(%cst_398, %cst_391, %cst_392, %cst_393, %cst_394, %cst_395, %cst_396, %cst_397 : f32, f64, f64, f64, f64, f64, f64, f64),
      51: ^bb1(%cst_390, %cst_383, %cst_384, %cst_385, %cst_386, %cst_387, %cst_388, %cst_389 : f32, f64, f64, f64, f64, f64, f64, f64),
      52: ^bb1(%cst_382, %cst_375, %cst_376, %cst_377, %cst_378, %cst_379, %cst_380, %cst_381 : f32, f64, f64, f64, f64, f64, f64, f64),
      53: ^bb1(%cst_374, %cst_367, %cst_368, %cst_369, %cst_370, %cst_371, %cst_372, %cst_373 : f32, f64, f64, f64, f64, f64, f64, f64),
      54: ^bb1(%cst_366, %cst_359, %cst_360, %cst_361, %cst_362, %cst_363, %cst_364, %cst_365 : f32, f64, f64, f64, f64, f64, f64, f64),
      55: ^bb1(%cst_358, %cst_351, %cst_352, %cst_353, %cst_354, %cst_355, %cst_356, %cst_357 : f32, f64, f64, f64, f64, f64, f64, f64),
      56: ^bb1(%cst_350, %cst_343, %cst_344, %cst_345, %cst_346, %cst_347, %cst_348, %cst_349 : f32, f64, f64, f64, f64, f64, f64, f64),
      57: ^bb1(%cst_342, %cst_335, %cst_336, %cst_337, %cst_338, %cst_339, %cst_340, %cst_341 : f32, f64, f64, f64, f64, f64, f64, f64),
      58: ^bb1(%cst_334, %cst_327, %cst_328, %cst_329, %cst_330, %cst_331, %cst_332, %cst_333 : f32, f64, f64, f64, f64, f64, f64, f64),
      59: ^bb1(%cst_326, %cst_319, %cst_320, %cst_321, %cst_322, %cst_323, %cst_324, %cst_325 : f32, f64, f64, f64, f64, f64, f64, f64),
      60: ^bb1(%cst_318, %cst_311, %cst_312, %cst_313, %cst_314, %cst_315, %cst_316, %cst_317 : f32, f64, f64, f64, f64, f64, f64, f64),
      61: ^bb1(%cst_310, %cst_303, %cst_304, %cst_305, %cst_306, %cst_307, %cst_308, %cst_309 : f32, f64, f64, f64, f64, f64, f64, f64),
      62: ^bb1(%cst_302, %cst_295, %cst_296, %cst_297, %cst_298, %cst_299, %cst_300, %cst_301 : f32, f64, f64, f64, f64, f64, f64, f64),
      63: ^bb1(%cst_294, %cst_287, %cst_288, %cst_289, %cst_290, %cst_291, %cst_292, %cst_293 : f32, f64, f64, f64, f64, f64, f64, f64),
      64: ^bb1(%cst_286, %cst_279, %cst_280, %cst_281, %cst_282, %cst_283, %cst_284, %cst_285 : f32, f64, f64, f64, f64, f64, f64, f64),
      65: ^bb1(%cst_278, %cst_271, %cst_272, %cst_273, %cst_274, %cst_275, %cst_276, %cst_277 : f32, f64, f64, f64, f64, f64, f64, f64),
      66: ^bb1(%cst_270, %cst_263, %cst_264, %cst_265, %cst_266, %cst_267, %cst_268, %cst_269 : f32, f64, f64, f64, f64, f64, f64, f64),
      67: ^bb1(%cst_262, %cst_255, %cst_256, %cst_257, %cst_258, %cst_259, %cst_260, %cst_261 : f32, f64, f64, f64, f64, f64, f64, f64),
      68: ^bb1(%cst_254, %cst_247, %cst_248, %cst_249, %cst_250, %cst_251, %cst_252, %cst_253 : f32, f64, f64, f64, f64, f64, f64, f64),
      69: ^bb1(%cst_246, %cst_239, %cst_240, %cst_241, %cst_242, %cst_243, %cst_244, %cst_245 : f32, f64, f64, f64, f64, f64, f64, f64),
      70: ^bb1(%cst_238, %cst_231, %cst_232, %cst_233, %cst_234, %cst_235, %cst_236, %cst_237 : f32, f64, f64, f64, f64, f64, f64, f64),
      71: ^bb1(%cst_230, %cst_223, %cst_224, %cst_225, %cst_226, %cst_227, %cst_228, %cst_229 : f32, f64, f64, f64, f64, f64, f64, f64),
      72: ^bb1(%cst_222, %cst_215, %cst_216, %cst_217, %cst_218, %cst_219, %cst_220, %cst_221 : f32, f64, f64, f64, f64, f64, f64, f64),
      73: ^bb1(%cst_214, %cst_207, %cst_208, %cst_209, %cst_210, %cst_211, %cst_212, %cst_213 : f32, f64, f64, f64, f64, f64, f64, f64),
      74: ^bb1(%cst_206, %cst_199, %cst_200, %cst_201, %cst_202, %cst_203, %cst_204, %cst_205 : f32, f64, f64, f64, f64, f64, f64, f64),
      75: ^bb1(%cst_198, %cst_191, %cst_192, %cst_193, %cst_194, %cst_195, %cst_196, %cst_197 : f32, f64, f64, f64, f64, f64, f64, f64),
      76: ^bb1(%cst_190, %cst_183, %cst_184, %cst_185, %cst_186, %cst_187, %cst_188, %cst_189 : f32, f64, f64, f64, f64, f64, f64, f64),
      77: ^bb1(%cst_182, %cst_175, %cst_176, %cst_177, %cst_178, %cst_179, %cst_180, %cst_181 : f32, f64, f64, f64, f64, f64, f64, f64),
      78: ^bb1(%cst_174, %cst_167, %cst_168, %cst_169, %cst_170, %cst_171, %cst_172, %cst_173 : f32, f64, f64, f64, f64, f64, f64, f64),
      79: ^bb1(%cst_166, %cst_159, %cst_160, %cst_161, %cst_162, %cst_163, %cst_164, %cst_165 : f32, f64, f64, f64, f64, f64, f64, f64),
      80: ^bb1(%cst_158, %cst_151, %cst_152, %cst_153, %cst_154, %cst_155, %cst_156, %cst_157 : f32, f64, f64, f64, f64, f64, f64, f64),
      81: ^bb1(%cst_150, %cst_143, %cst_144, %cst_145, %cst_146, %cst_147, %cst_148, %cst_149 : f32, f64, f64, f64, f64, f64, f64, f64),
      82: ^bb1(%cst_142, %cst_135, %cst_136, %cst_137, %cst_138, %cst_139, %cst_140, %cst_141 : f32, f64, f64, f64, f64, f64, f64, f64),
      83: ^bb1(%cst_134, %cst_127, %cst_128, %cst_129, %cst_130, %cst_131, %cst_132, %cst_133 : f32, f64, f64, f64, f64, f64, f64, f64),
      84: ^bb1(%cst_126, %cst_119, %cst_120, %cst_121, %cst_122, %cst_123, %cst_124, %cst_125 : f32, f64, f64, f64, f64, f64, f64, f64),
      85: ^bb1(%cst_118, %cst_111, %cst_112, %cst_113, %cst_114, %cst_115, %cst_116, %cst_117 : f32, f64, f64, f64, f64, f64, f64, f64),
      86: ^bb1(%cst_110, %cst_103, %cst_104, %cst_105, %cst_106, %cst_107, %cst_108, %cst_109 : f32, f64, f64, f64, f64, f64, f64, f64),
      87: ^bb1(%cst_102, %cst_95, %cst_96, %cst_97, %cst_98, %cst_99, %cst_100, %cst_101 : f32, f64, f64, f64, f64, f64, f64, f64),
      88: ^bb1(%cst_94, %cst_87, %cst_88, %cst_89, %cst_90, %cst_91, %cst_92, %cst_93 : f32, f64, f64, f64, f64, f64, f64, f64),
      89: ^bb1(%cst_86, %cst_79, %cst_80, %cst_81, %cst_82, %cst_83, %cst_84, %cst_85 : f32, f64, f64, f64, f64, f64, f64, f64),
      90: ^bb1(%cst_78, %cst_71, %cst_72, %cst_73, %cst_74, %cst_75, %cst_76, %cst_77 : f32, f64, f64, f64, f64, f64, f64, f64),
      91: ^bb1(%cst_70, %cst_63, %cst_64, %cst_65, %cst_66, %cst_67, %cst_68, %cst_69 : f32, f64, f64, f64, f64, f64, f64, f64),
      92: ^bb1(%cst_62, %cst_55, %cst_56, %cst_57, %cst_58, %cst_59, %cst_60, %cst_61 : f32, f64, f64, f64, f64, f64, f64, f64),
      93: ^bb1(%cst_54, %cst_47, %cst_48, %cst_49, %cst_50, %cst_51, %cst_52, %cst_53 : f32, f64, f64, f64, f64, f64, f64, f64),
      94: ^bb1(%cst_46, %cst_39, %cst_40, %cst_41, %cst_42, %cst_43, %cst_44, %cst_45 : f32, f64, f64, f64, f64, f64, f64, f64),
      95: ^bb1(%cst_38, %cst_31, %cst_32, %cst_33, %cst_34, %cst_35, %cst_36, %cst_37 : f32, f64, f64, f64, f64, f64, f64, f64),
      96: ^bb1(%cst_30, %cst_23, %cst_24, %cst_25, %cst_26, %cst_27, %cst_28, %cst_29 : f32, f64, f64, f64, f64, f64, f64, f64),
      97: ^bb1(%cst_22, %cst_15, %cst_16, %cst_17, %cst_18, %cst_19, %cst_20, %cst_21 : f32, f64, f64, f64, f64, f64, f64, f64),
      98: ^bb1(%cst_14, %cst_7, %cst_8, %cst_9, %cst_10, %cst_11, %cst_12, %cst_13 : f32, f64, f64, f64, f64, f64, f64, f64),
      99: ^bb1(%cst_6, %cst, %cst_0, %cst_1, %cst_2, %cst_3, %cst_4, %cst_5 : f32, f64, f64, f64, f64, f64, f64, f64)
    ]
  ^bb1(%2: f32, %3: f64, %4: f64, %5: f64, %6: f64, %7: f64, %8: f64, %9: f64):  // 100 preds: ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0, ^bb0
    %10 = arith.mulf %arg0, %cst_799 : f32
    %11 = arith.subf %10, %2 : f32
    %12 = arith.extf %11 : f32 to f64
    %13 = arith.mulf %12, %3 : f64
    %14 = arith.addf %13, %4 : f64
    %15 = arith.mulf %14, %12 : f64
    %16 = arith.addf %15, %5 : f64
    %17 = arith.mulf %16, %12 : f64
    %18 = arith.addf %17, %6 : f64
    %19 = arith.mulf %18, %12 : f64
    %20 = arith.addf %19, %7 : f64
    %21 = arith.mulf %20, %12 : f64
    %22 = arith.addf %21, %8 : f64
    %23 = arith.mulf %22, %12 : f64
    %24 = arith.addf %23, %9 : f64
    %25 = arith.truncf %24 : f64 to f32
    cf.br ^bb2(%false, %25 : i1, f32)
  ^bb2(%26: i1, %27: f32):  // 2 preds: ^bb0, ^bb1
    %28 = arith.select %26, %cst_798, %27 : f32
    return %28 : f32
  }
  func.func @_ZNSt14numeric_limitsIfE8infinityEv() -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant 0x7F800000 : f32
    return %cst : f32
  }
  func.func @_ZSt5isnanf(%arg0: f32) -> i8 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = arith.cmpf uno, %arg0, %arg0 : f32
    %1 = arith.extui %0 : i1 to i8
    return %1 : i8
  }
}
