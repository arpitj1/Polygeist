module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global constant @"_Z4zetaIfLb0EET_S0_S0_@static@_ZZ4zetaIfLb0EET_S0_S0_E1A" : memref<12xf64> = dense<[1.200000e+01, -7.200000e+02, 3.024000e+04, -1.209600e+06, 0x4186D73000000000, -1892437580.3183792, 0x423165E990000000, -2950130727918.1641, 116467828143500.67, 0xC33055D6586F1431, 1.8152105401943546E+17, -7.1661652561756672E+18]>
  func.func @thrust_calc_polygamma_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: i32, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2.000000e+00 : f64
    %c11 = arith.constant 11 : index
    %c10 = arith.constant 10 : index
    %c9 = arith.constant 9 : index
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant -1.000000e+00 : f32
    %cst_2 = arith.constant 0x7FC00000 : f32
    %cst_3 = arith.constant 0x7F800000 : f32
    %cst_4 = arith.constant -1.1102230246251565E-16 : f64
    %false = arith.constant false
    %true = arith.constant true
    %cst_5 = arith.constant 9.000000e+00 : f64
    %c9_i32 = arith.constant 9 : i32
    %cst_6 = arith.constant 1.000000e+00 : f64
    %cst_7 = arith.constant 5.000000e-01 : f64
    %cst_8 = arith.constant 0.000000e+00 : f64
    %cst_9 = arith.constant 1.1102230246251565E-16 : f64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.remsi %arg2, %c2_i32 : i32
    %2 = arith.cmpi ne, %1, %c0_i32 : i32
    %3 = arith.select %2, %cst_0, %cst_1 : f32
    %4 = arith.sitofp %arg2 : i32 to f32
    %5 = arith.addf %4, %cst_0 : f32
    %6 = arith.addi %arg2, %c1_i32 : i32
    %7 = arith.sitofp %6 : i32 to f32
    %8 = memref.get_global @"_Z4zetaIfLb0EET_S0_S0_@static@_ZZ4zetaIfLb0EET_S0_S0_E1A" : memref<12xf64>
    %9 = llvm.mlir.undef : f32
    %10 = arith.extf %7 : f32 to f64
    %11 = arith.cmpf oeq, %10, %cst_6 : f64
    %12 = arith.xori %11, %true : i1
    %13 = arith.select %11, %cst_3, %9 : f32
    %14 = arith.cmpf olt, %10, %cst_6 : f64
    %15 = arith.andi %14, %12 : i1
    %16 = arith.xori %15, %true : i1
    %17 = arith.andi %16, %12 : i1
    %18 = arith.select %15, %cst_2, %13 : f32
    %19 = arith.negf %7 : f32
    %20 = arith.extf %19 : f32 to f64
    %21 = arith.subf %10, %cst_6 : f64
    affine.for %arg4 = 0 to %0 {
      %22 = memref.load %arg1[%arg4] : memref<?xf32>
      %23 = func.call @lgammaf(%5) : (f32) -> f32
      %24 = math.exp %23 : f32
      %25 = arith.mulf %3, %24 : f32
      %26 = arith.extf %22 : f32 to f64
      %27 = arith.cmpf ole, %26, %cst_8 : f64
      %28 = arith.andi %17, %27 : i1
      %29:2 = scf.if %28 -> (i1, f32) {
        %32:2 = scf.if %17 -> (i1, f32) {
          %33 = func.call @floorf(%22) : (f32) -> f32
          %34 = arith.cmpf oeq, %22, %33 : f32
          %35:2 = scf.if %34 -> (i1, f32) {
            scf.yield %false, %cst_3 : i1, f32
          } else {
            %36 = func.call @floorf(%7) : (f32) -> f32
            %37 = arith.cmpf une, %7, %36 : f32
            %38 = arith.xori %37, %true : i1
            %39 = arith.select %37, %cst_2, %18 : f32
            scf.yield %38, %39 : i1, f32
          }
          scf.yield %35#0, %35#1 : i1, f32
        } else {
          scf.yield %false, %18 : i1, f32
        }
        scf.yield %32#0, %32#1 : i1, f32
      } else {
        scf.yield %17, %18 : i1, f32
      }
      %30 = scf.if %29#0 -> (f32) {
        %32 = math.powf %22, %19 : f32
        %33 = arith.extf %32 : f32 to f64
        %34:6 = scf.while (%arg5 = %c0_i32, %arg6 = %33, %arg7 = %cst_8, %arg8 = %26, %arg9 = %true, %arg10 = %29#1, %arg11 = %true) : (i32, f64, f64, f64, i1, f32, i1) -> (f64, f64, f64, i1, f32, i32) {
          %36 = arith.cmpi slt, %arg5, %c9_i32 : i32
          %37 = scf.if %36 -> (i1) {
            scf.yield %true : i1
          } else {
            %39 = arith.cmpf ole, %arg8, %cst_5 : f64
            scf.yield %39 : i1
          }
          %38 = arith.andi %37, %arg11 : i1
          scf.condition(%38) %arg6, %arg7, %arg8, %arg9, %arg10, %arg5 : f64, f64, f64, i1, f32, i32
        } do {
        ^bb0(%arg5: f64, %arg6: f64, %arg7: f64, %arg8: i1, %arg9: f32, %arg10: i32):
          %36 = arith.addi %arg10, %c1_i32 : i32
          %37 = arith.addf %arg7, %cst_6 : f64
          %38 = math.powf %37, %20 : f64
          %39 = arith.addf %arg5, %38 : f64
          %40 = arith.mulf %39, %cst_4 : f64
          %41 = arith.cmpf olt, %40, %38 : f64
          %42 = arith.mulf %39, %cst_9 : f64
          %43 = arith.cmpf olt, %38, %42 : f64
          %44 = arith.xori %43, %true : i1
          %45 = arith.andi %44, %arg8 : i1
          %46 = arith.andi %41, %43 : i1
          %47 = arith.xori %41, %true : i1
          %48 = arith.andi %41, %45 : i1
          %49 = arith.andi %47, %arg8 : i1
          %50 = arith.ori %48, %49 : i1
          %51 = arith.andi %41, %44 : i1
          %52 = arith.ori %51, %47 : i1
          %53 = scf.if %46 -> (f32) {
            %54 = arith.truncf %39 : f64 to f32
            scf.yield %54 : f32
          } else {
            scf.yield %arg9 : f32
          }
          scf.yield %36, %39, %38, %37, %50, %53, %52 : i32, f64, f64, f64, i1, f32, i1
        }
        %35 = scf.if %34#3 -> (f32) {
          %36 = arith.mulf %34#1, %34#2 : f64
          %37 = arith.divf %36, %21 : f64
          %38 = arith.addf %34#0, %37 : f64
          %39 = arith.mulf %34#1, %cst_7 : f64
          %40 = arith.subf %38, %39 : f64
          %41 = arith.addf %10, %cst_8 : f64
          %42 = arith.divf %34#1, %34#2 : f64
          %43 = arith.mulf %41, %42 : f64
          %44 = memref.load %8[%c0] : memref<12xf64>
          %45 = arith.divf %43, %44 : f64
          %46 = arith.addf %40, %45 : f64
          %47 = arith.divf %45, %46 : f64
          %48 = math.absf %47 : f64
          %49 = arith.cmpf olt, %48, %cst_9 : f64
          %50 = arith.xori %49, %true : i1
          %51 = arith.select %49, %cst_8, %cst : f64
          %52:12 = scf.if %49 -> (f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1) {
            %55 = arith.truncf %46 : f64 to f32
            scf.yield %46, %50, %55, %48, %46, %51, %42, %41, %50, %55, %50, %false : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
          } else {
            %55 = arith.addf %10, %cst_6 : f64
            %56 = arith.mulf %41, %55 : f64
            %57 = arith.divf %42, %34#2 : f64
            %58 = arith.addf %10, %51 : f64
            %59 = arith.mulf %56, %58 : f64
            %60 = arith.divf %57, %34#2 : f64
            %61 = arith.mulf %59, %60 : f64
            %62 = memref.load %8[%c1] : memref<12xf64>
            %63 = arith.divf %61, %62 : f64
            %64 = arith.addf %46, %63 : f64
            %65 = arith.divf %63, %64 : f64
            %66 = math.absf %65 : f64
            %67 = arith.select %50, %66, %63 : f64
            %68 = arith.cmpf olt, %66, %cst_9 : f64
            %69 = arith.andi %68, %50 : i1
            %70 = arith.xori %69, %true : i1
            %71 = arith.andi %70, %50 : i1
            %72 = scf.if %69 -> (f32) {
              %74 = arith.truncf %64 : f64 to f32
              scf.yield %74 : f32
            } else {
              scf.yield %34#4 : f32
            }
            %73:3 = scf.if %71 -> (f64, f64, f64) {
              %74 = arith.addf %51, %cst_6 : f64
              %75 = arith.addf %10, %74 : f64
              %76 = arith.mulf %59, %75 : f64
              %77 = arith.divf %60, %34#2 : f64
              %78 = arith.addf %74, %cst_6 : f64
              scf.yield %76, %77, %78 : f64, f64, f64
            } else {
              scf.yield %59, %60, %51 : f64, f64, f64
            }
            scf.yield %64, %71, %72, %67, %64, %73#2, %73#1, %73#0, %71, %72, %71, %71 : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
          }
          %53:3 = scf.if %52#11 -> (f64, i1, f32) {
            %55:4 = scf.if %52#10 -> (f64, f64, f64, f64) {
              %69 = arith.addf %10, %52#5 : f64
              %70 = arith.mulf %52#7, %69 : f64
              %71 = arith.divf %52#6, %34#2 : f64
              %72 = arith.mulf %70, %71 : f64
              %73 = memref.load %8[%c2] : memref<12xf64>
              %74 = arith.divf %72, %73 : f64
              %75 = arith.addf %52#4, %74 : f64
              scf.yield %70, %71, %74, %75 : f64, f64, f64, f64
            } else {
              scf.yield %52#7, %52#6, %52#3, %52#4 : f64, f64, f64, f64
            }
            %56 = arith.divf %55#2, %55#3 : f64
            %57 = math.absf %56 : f64
            %58 = arith.cmpf olt, %57, %cst_9 : f64
            %59 = arith.andi %58, %52#10 : i1
            %60 = arith.xori %59, %true : i1
            %61 = arith.andi %60, %52#8 : i1
            %62 = arith.andi %60, %52#10 : i1
            %63 = arith.xori %52#10, %true : i1
            %64 = arith.andi %52#10, %61 : i1
            %65 = arith.andi %63, %52#8 : i1
            %66 = arith.ori %64, %65 : i1
            %67 = scf.if %59 -> (f32) {
              %69 = arith.truncf %55#3 : f64 to f32
              scf.yield %69 : f32
            } else {
              scf.yield %52#9 : f32
            }
            %68:3 = scf.if %62 -> (f64, i1, f32) {
              %69 = arith.addf %52#5, %cst_6 : f64
              %70 = arith.addf %10, %69 : f64
              %71 = arith.mulf %55#0, %70 : f64
              %72 = arith.divf %55#1, %34#2 : f64
              %73 = arith.addf %69, %cst_6 : f64
              %74 = arith.addf %10, %73 : f64
              %75 = arith.mulf %71, %74 : f64
              %76 = arith.divf %72, %34#2 : f64
              %77 = arith.mulf %75, %76 : f64
              %78 = memref.load %8[%c3] : memref<12xf64>
              %79 = arith.divf %77, %78 : f64
              %80 = arith.addf %55#3, %79 : f64
              %81 = arith.divf %79, %80 : f64
              %82 = math.absf %81 : f64
              %83 = arith.cmpf olt, %82, %cst_9 : f64
              %84 = arith.xori %83, %true : i1
              %85 = arith.andi %84, %66 : i1
              %86:12 = scf.if %83 -> (f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1) {
                %88 = arith.truncf %80 : f64 to f32
                scf.yield %80, %85, %88, %82, %80, %73, %76, %75, %85, %88, %84, %false : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
              } else {
                %88 = arith.addf %73, %cst_6 : f64
                %89 = arith.addf %10, %88 : f64
                %90 = arith.mulf %75, %89 : f64
                %91 = arith.divf %76, %34#2 : f64
                %92 = arith.addf %88, %cst_6 : f64
                %93 = arith.addf %10, %92 : f64
                %94 = arith.mulf %90, %93 : f64
                %95 = arith.divf %91, %34#2 : f64
                %96 = arith.mulf %94, %95 : f64
                %97 = memref.load %8[%c4] : memref<12xf64>
                %98 = arith.divf %96, %97 : f64
                %99 = arith.addf %80, %98 : f64
                %100 = arith.divf %98, %99 : f64
                %101 = math.absf %100 : f64
                %102 = arith.select %84, %101, %98 : f64
                %103 = arith.cmpf olt, %101, %cst_9 : f64
                %104 = arith.andi %103, %84 : i1
                %105 = arith.xori %104, %true : i1
                %106 = arith.andi %105, %85 : i1
                %107 = arith.andi %105, %84 : i1
                %108 = arith.andi %84, %106 : i1
                %109 = scf.if %104 -> (f32) {
                  %111 = arith.truncf %99 : f64 to f32
                  scf.yield %111 : f32
                } else {
                  scf.yield %67 : f32
                }
                %110:3 = scf.if %107 -> (f64, f64, f64) {
                  %111 = arith.addf %92, %cst_6 : f64
                  %112 = arith.addf %10, %111 : f64
                  %113 = arith.mulf %94, %112 : f64
                  %114 = arith.divf %95, %34#2 : f64
                  %115 = arith.addf %111, %cst_6 : f64
                  scf.yield %113, %114, %115 : f64, f64, f64
                } else {
                  scf.yield %94, %95, %92 : f64, f64, f64
                }
                scf.yield %99, %108, %109, %102, %99, %110#2, %110#1, %110#0, %108, %109, %107, %107 : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
              }
              %87:3 = scf.if %86#11 -> (f64, i1, f32) {
                %88:4 = scf.if %86#10 -> (f64, f64, f64, f64) {
                  %102 = arith.addf %10, %86#5 : f64
                  %103 = arith.mulf %86#7, %102 : f64
                  %104 = arith.divf %86#6, %34#2 : f64
                  %105 = arith.mulf %103, %104 : f64
                  %106 = memref.load %8[%c5] : memref<12xf64>
                  %107 = arith.divf %105, %106 : f64
                  %108 = arith.addf %86#4, %107 : f64
                  scf.yield %103, %104, %107, %108 : f64, f64, f64, f64
                } else {
                  scf.yield %86#7, %86#6, %86#3, %86#4 : f64, f64, f64, f64
                }
                %89 = arith.divf %88#2, %88#3 : f64
                %90 = math.absf %89 : f64
                %91 = arith.cmpf olt, %90, %cst_9 : f64
                %92 = arith.andi %91, %86#10 : i1
                %93 = arith.xori %92, %true : i1
                %94 = arith.andi %93, %86#8 : i1
                %95 = arith.andi %93, %86#10 : i1
                %96 = arith.xori %86#10, %true : i1
                %97 = arith.andi %86#10, %94 : i1
                %98 = arith.andi %96, %86#8 : i1
                %99 = arith.ori %97, %98 : i1
                %100 = scf.if %92 -> (f32) {
                  %102 = arith.truncf %88#3 : f64 to f32
                  scf.yield %102 : f32
                } else {
                  scf.yield %86#9 : f32
                }
                %101:3 = scf.if %95 -> (f64, i1, f32) {
                  %102 = arith.addf %86#5, %cst_6 : f64
                  %103 = arith.addf %10, %102 : f64
                  %104 = arith.mulf %88#0, %103 : f64
                  %105 = arith.divf %88#1, %34#2 : f64
                  %106 = arith.addf %102, %cst_6 : f64
                  %107 = arith.addf %10, %106 : f64
                  %108 = arith.mulf %104, %107 : f64
                  %109 = arith.divf %105, %34#2 : f64
                  %110 = arith.mulf %108, %109 : f64
                  %111 = memref.load %8[%c6] : memref<12xf64>
                  %112 = arith.divf %110, %111 : f64
                  %113 = arith.addf %88#3, %112 : f64
                  %114 = arith.divf %112, %113 : f64
                  %115 = math.absf %114 : f64
                  %116 = arith.cmpf olt, %115, %cst_9 : f64
                  %117 = arith.xori %116, %true : i1
                  %118 = arith.andi %117, %99 : i1
                  %119:12 = scf.if %116 -> (f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1) {
                    %121 = arith.truncf %113 : f64 to f32
                    scf.yield %113, %118, %121, %115, %113, %106, %109, %108, %118, %121, %117, %false : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
                  } else {
                    %121 = arith.addf %106, %cst_6 : f64
                    %122 = arith.addf %10, %121 : f64
                    %123 = arith.mulf %108, %122 : f64
                    %124 = arith.divf %109, %34#2 : f64
                    %125 = arith.addf %121, %cst_6 : f64
                    %126 = arith.addf %10, %125 : f64
                    %127 = arith.mulf %123, %126 : f64
                    %128 = arith.divf %124, %34#2 : f64
                    %129 = arith.mulf %127, %128 : f64
                    %130 = memref.load %8[%c7] : memref<12xf64>
                    %131 = arith.divf %129, %130 : f64
                    %132 = arith.addf %113, %131 : f64
                    %133 = arith.divf %131, %132 : f64
                    %134 = math.absf %133 : f64
                    %135 = arith.select %117, %134, %131 : f64
                    %136 = arith.cmpf olt, %134, %cst_9 : f64
                    %137 = arith.andi %136, %117 : i1
                    %138 = arith.xori %137, %true : i1
                    %139 = arith.andi %138, %118 : i1
                    %140 = arith.andi %138, %117 : i1
                    %141 = arith.andi %117, %139 : i1
                    %142 = scf.if %137 -> (f32) {
                      %144 = arith.truncf %132 : f64 to f32
                      scf.yield %144 : f32
                    } else {
                      scf.yield %100 : f32
                    }
                    %143:3 = scf.if %140 -> (f64, f64, f64) {
                      %144 = arith.addf %125, %cst_6 : f64
                      %145 = arith.addf %10, %144 : f64
                      %146 = arith.mulf %127, %145 : f64
                      %147 = arith.divf %128, %34#2 : f64
                      %148 = arith.addf %144, %cst_6 : f64
                      scf.yield %146, %147, %148 : f64, f64, f64
                    } else {
                      scf.yield %127, %128, %125 : f64, f64, f64
                    }
                    scf.yield %132, %141, %142, %135, %132, %143#2, %143#1, %143#0, %141, %142, %140, %140 : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
                  }
                  %120:3 = scf.if %119#11 -> (f64, i1, f32) {
                    %121:4 = scf.if %119#10 -> (f64, f64, f64, f64) {
                      %135 = arith.addf %10, %119#5 : f64
                      %136 = arith.mulf %119#7, %135 : f64
                      %137 = arith.divf %119#6, %34#2 : f64
                      %138 = arith.mulf %136, %137 : f64
                      %139 = memref.load %8[%c8] : memref<12xf64>
                      %140 = arith.divf %138, %139 : f64
                      %141 = arith.addf %119#4, %140 : f64
                      scf.yield %136, %137, %140, %141 : f64, f64, f64, f64
                    } else {
                      scf.yield %119#7, %119#6, %119#3, %119#4 : f64, f64, f64, f64
                    }
                    %122 = arith.divf %121#2, %121#3 : f64
                    %123 = math.absf %122 : f64
                    %124 = arith.cmpf olt, %123, %cst_9 : f64
                    %125 = arith.andi %124, %119#10 : i1
                    %126 = arith.xori %125, %true : i1
                    %127 = arith.andi %126, %119#8 : i1
                    %128 = arith.andi %126, %119#10 : i1
                    %129 = arith.xori %119#10, %true : i1
                    %130 = arith.andi %119#10, %127 : i1
                    %131 = arith.andi %129, %119#8 : i1
                    %132 = arith.ori %130, %131 : i1
                    %133 = scf.if %125 -> (f32) {
                      %135 = arith.truncf %121#3 : f64 to f32
                      scf.yield %135 : f32
                    } else {
                      scf.yield %119#9 : f32
                    }
                    %134:3 = scf.if %128 -> (f64, i1, f32) {
                      %135 = arith.addf %119#5, %cst_6 : f64
                      %136 = arith.addf %10, %135 : f64
                      %137 = arith.mulf %121#0, %136 : f64
                      %138 = arith.divf %121#1, %34#2 : f64
                      %139 = arith.addf %135, %cst_6 : f64
                      %140 = arith.addf %10, %139 : f64
                      %141 = arith.mulf %137, %140 : f64
                      %142 = arith.divf %138, %34#2 : f64
                      %143 = arith.mulf %141, %142 : f64
                      %144 = memref.load %8[%c9] : memref<12xf64>
                      %145 = arith.divf %143, %144 : f64
                      %146 = arith.addf %121#3, %145 : f64
                      %147 = arith.divf %145, %146 : f64
                      %148 = math.absf %147 : f64
                      %149 = arith.cmpf olt, %148, %cst_9 : f64
                      %150 = arith.xori %149, %true : i1
                      %151 = arith.andi %150, %132 : i1
                      %152:12 = scf.if %149 -> (f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1) {
                        %154 = arith.truncf %146 : f64 to f32
                        scf.yield %146, %151, %154, %148, %146, %139, %142, %141, %151, %154, %150, %false : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
                      } else {
                        %154 = arith.addf %139, %cst_6 : f64
                        %155 = arith.addf %10, %154 : f64
                        %156 = arith.mulf %141, %155 : f64
                        %157 = arith.divf %142, %34#2 : f64
                        %158 = arith.addf %154, %cst_6 : f64
                        %159 = arith.addf %10, %158 : f64
                        %160 = arith.mulf %156, %159 : f64
                        %161 = arith.divf %157, %34#2 : f64
                        %162 = arith.mulf %160, %161 : f64
                        %163 = memref.load %8[%c10] : memref<12xf64>
                        %164 = arith.divf %162, %163 : f64
                        %165 = arith.addf %146, %164 : f64
                        %166 = arith.divf %164, %165 : f64
                        %167 = math.absf %166 : f64
                        %168 = arith.select %150, %167, %164 : f64
                        %169 = arith.cmpf olt, %167, %cst_9 : f64
                        %170 = arith.andi %169, %150 : i1
                        %171 = arith.xori %170, %true : i1
                        %172 = arith.andi %171, %151 : i1
                        %173 = arith.andi %171, %150 : i1
                        %174 = arith.andi %150, %172 : i1
                        %175 = scf.if %170 -> (f32) {
                          %177 = arith.truncf %165 : f64 to f32
                          scf.yield %177 : f32
                        } else {
                          scf.yield %133 : f32
                        }
                        %176:3 = scf.if %173 -> (f64, f64, f64) {
                          %177 = arith.addf %158, %cst_6 : f64
                          %178 = arith.addf %10, %177 : f64
                          %179 = arith.mulf %160, %178 : f64
                          %180 = arith.divf %161, %34#2 : f64
                          %181 = arith.addf %177, %cst_6 : f64
                          scf.yield %179, %180, %181 : f64, f64, f64
                        } else {
                          scf.yield %160, %161, %158 : f64, f64, f64
                        }
                        scf.yield %165, %174, %175, %168, %165, %176#2, %176#1, %176#0, %174, %175, %173, %173 : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
                      }
                      %153:3 = scf.if %152#11 -> (f64, i1, f32) {
                        %154:2 = scf.if %152#10 -> (f64, f64) {
                          %166 = arith.addf %10, %152#5 : f64
                          %167 = arith.mulf %152#7, %166 : f64
                          %168 = arith.divf %152#6, %34#2 : f64
                          %169 = arith.mulf %167, %168 : f64
                          %170 = memref.load %8[%c11] : memref<12xf64>
                          %171 = arith.divf %169, %170 : f64
                          %172 = arith.addf %152#4, %171 : f64
                          scf.yield %171, %172 : f64, f64
                        } else {
                          scf.yield %152#3, %152#4 : f64, f64
                        }
                        %155 = arith.divf %154#0, %154#1 : f64
                        %156 = math.absf %155 : f64
                        %157 = arith.cmpf olt, %156, %cst_9 : f64
                        %158 = arith.andi %157, %152#10 : i1
                        %159 = arith.xori %158, %true : i1
                        %160 = arith.andi %159, %152#8 : i1
                        %161 = arith.xori %152#10, %true : i1
                        %162 = arith.andi %152#10, %160 : i1
                        %163 = arith.andi %161, %152#8 : i1
                        %164 = arith.ori %162, %163 : i1
                        %165 = scf.if %158 -> (f32) {
                          %166 = arith.truncf %154#1 : f64 to f32
                          scf.yield %166 : f32
                        } else {
                          scf.yield %152#9 : f32
                        }
                        scf.yield %154#1, %164, %165 : f64, i1, f32
                      } else {
                        scf.yield %152#0, %152#1, %152#2 : f64, i1, f32
                      }
                      scf.yield %153#0, %153#1, %153#2 : f64, i1, f32
                    } else {
                      scf.yield %121#3, %132, %133 : f64, i1, f32
                    }
                    scf.yield %134#0, %134#1, %134#2 : f64, i1, f32
                  } else {
                    scf.yield %119#0, %119#1, %119#2 : f64, i1, f32
                  }
                  scf.yield %120#0, %120#1, %120#2 : f64, i1, f32
                } else {
                  scf.yield %88#3, %99, %100 : f64, i1, f32
                }
                scf.yield %101#0, %101#1, %101#2 : f64, i1, f32
              } else {
                scf.yield %86#0, %86#1, %86#2 : f64, i1, f32
              }
              scf.yield %87#0, %87#1, %87#2 : f64, i1, f32
            } else {
              scf.yield %55#3, %66, %67 : f64, i1, f32
            }
            scf.yield %68#0, %68#1, %68#2 : f64, i1, f32
          } else {
            scf.yield %52#0, %52#1, %52#2 : f64, i1, f32
          }
          %54 = scf.if %53#1 -> (f32) {
            %55 = arith.truncf %53#0 : f64 to f32
            scf.yield %55 : f32
          } else {
            scf.yield %53#2 : f32
          }
          scf.yield %54 : f32
        } else {
          scf.yield %34#4 : f32
        }
        scf.yield %35 : f32
      } else {
        scf.yield %29#1 : f32
      }
      %31 = arith.mulf %25, %30 : f32
      memref.store %31, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
  func.func private @lgammaf(f32) -> f32
  func.func private @floorf(f32) -> f32
}

