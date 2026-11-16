module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global constant @"_Z4zetaIfLb0EET_S0_S0_@static@_ZZ4zetaIfLb0EET_S0_S0_E1A" : memref<12xf64> = dense<[1.200000e+01, -7.200000e+02, 3.024000e+04, -1.209600e+06, 0x4186D73000000000, -1892437580.3183792, 0x423165E990000000, -2950130727918.1641, 116467828143500.67, 0xC33055D6586F1431, 1.8152105401943546E+17, -7.1661652561756672E+18]>
  func.func @thrust_calc_zeta_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
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
    %cst_0 = arith.constant 1.1102230246251565E-16 : f64
    %cst_1 = arith.constant 0.000000e+00 : f64
    %cst_2 = arith.constant 5.000000e-01 : f64
    %cst_3 = arith.constant 1.000000e+00 : f64
    %c0_i32 = arith.constant 0 : i32
    %c9_i32 = arith.constant 9 : i32
    %cst_4 = arith.constant 9.000000e+00 : f64
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %cst_5 = arith.constant -1.1102230246251565E-16 : f64
    %cst_6 = arith.constant 0x7F800000 : f32
    %cst_7 = arith.constant 0x7FC00000 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = memref.get_global @"_Z4zetaIfLb0EET_S0_S0_@static@_ZZ4zetaIfLb0EET_S0_S0_E1A" : memref<12xf64>
    %2 = llvm.mlir.undef : f32
    affine.for %arg4 = 0 to %0 {
      %3 = memref.load %arg1[%arg4] : memref<?xf32>
      %4 = memref.load %arg2[%arg4] : memref<?xf32>
      %5 = arith.extf %3 : f32 to f64
      %6 = arith.cmpf oeq, %5, %cst_3 : f64
      %7 = arith.xori %6, %true : i1
      %8 = arith.select %6, %cst_6, %2 : f32
      %9 = arith.cmpf olt, %5, %cst_3 : f64
      %10 = arith.andi %9, %7 : i1
      %11 = arith.xori %10, %true : i1
      %12 = arith.andi %11, %7 : i1
      %13 = arith.select %10, %cst_7, %8 : f32
      %14 = arith.extf %4 : f32 to f64
      %15 = arith.cmpf ole, %14, %cst_1 : f64
      %16 = arith.andi %12, %15 : i1
      %17:2 = scf.if %16 -> (i1, f32) {
        %19:2 = scf.if %12 -> (i1, f32) {
          %20 = func.call @floorf(%4) : (f32) -> f32
          %21 = arith.cmpf oeq, %4, %20 : f32
          %22:2 = scf.if %21 -> (i1, f32) {
            scf.yield %false, %cst_6 : i1, f32
          } else {
            %23 = func.call @floorf(%3) : (f32) -> f32
            %24 = arith.cmpf une, %3, %23 : f32
            %25 = arith.xori %24, %true : i1
            %26 = arith.select %24, %cst_7, %13 : f32
            scf.yield %25, %26 : i1, f32
          }
          scf.yield %22#0, %22#1 : i1, f32
        } else {
          scf.yield %false, %13 : i1, f32
        }
        scf.yield %19#0, %19#1 : i1, f32
      } else {
        scf.yield %12, %13 : i1, f32
      }
      %18 = scf.if %17#0 -> (f32) {
        %19 = arith.negf %3 : f32
        %20 = math.powf %4, %19 : f32
        %21 = arith.extf %20 : f32 to f64
        %22 = arith.extf %19 : f32 to f64
        %23:6 = scf.while (%arg5 = %c0_i32, %arg6 = %21, %arg7 = %cst_1, %arg8 = %14, %arg9 = %true, %arg10 = %17#1, %arg11 = %true) : (i32, f64, f64, f64, i1, f32, i1) -> (f64, f64, f64, i1, f32, i32) {
          %25 = arith.cmpi slt, %arg5, %c9_i32 : i32
          %26 = scf.if %25 -> (i1) {
            scf.yield %true : i1
          } else {
            %28 = arith.cmpf ole, %arg8, %cst_4 : f64
            scf.yield %28 : i1
          }
          %27 = arith.andi %26, %arg11 : i1
          scf.condition(%27) %arg6, %arg7, %arg8, %arg9, %arg10, %arg5 : f64, f64, f64, i1, f32, i32
        } do {
        ^bb0(%arg5: f64, %arg6: f64, %arg7: f64, %arg8: i1, %arg9: f32, %arg10: i32):
          %25 = arith.addi %arg10, %c1_i32 : i32
          %26 = arith.addf %arg7, %cst_3 : f64
          %27 = math.powf %26, %22 : f64
          %28 = arith.addf %arg5, %27 : f64
          %29 = arith.mulf %28, %cst_5 : f64
          %30 = arith.cmpf olt, %29, %27 : f64
          %31 = arith.mulf %28, %cst_0 : f64
          %32 = arith.cmpf olt, %27, %31 : f64
          %33 = arith.xori %32, %true : i1
          %34 = arith.andi %33, %arg8 : i1
          %35 = arith.andi %30, %32 : i1
          %36 = arith.xori %30, %true : i1
          %37 = arith.andi %30, %34 : i1
          %38 = arith.andi %36, %arg8 : i1
          %39 = arith.ori %37, %38 : i1
          %40 = arith.andi %30, %33 : i1
          %41 = arith.ori %40, %36 : i1
          %42 = scf.if %35 -> (f32) {
            %43 = arith.truncf %28 : f64 to f32
            scf.yield %43 : f32
          } else {
            scf.yield %arg9 : f32
          }
          scf.yield %25, %28, %27, %26, %39, %42, %41 : i32, f64, f64, f64, i1, f32, i1
        }
        %24 = scf.if %23#3 -> (f32) {
          %25 = arith.mulf %23#1, %23#2 : f64
          %26 = arith.subf %5, %cst_3 : f64
          %27 = arith.divf %25, %26 : f64
          %28 = arith.addf %23#0, %27 : f64
          %29 = arith.mulf %23#1, %cst_2 : f64
          %30 = arith.subf %28, %29 : f64
          %31 = arith.addf %5, %cst_1 : f64
          %32 = arith.divf %23#1, %23#2 : f64
          %33 = arith.mulf %31, %32 : f64
          %34 = memref.load %1[%c0] : memref<12xf64>
          %35 = arith.divf %33, %34 : f64
          %36 = arith.addf %30, %35 : f64
          %37 = arith.divf %35, %36 : f64
          %38 = math.absf %37 : f64
          %39 = arith.cmpf olt, %38, %cst_0 : f64
          %40 = arith.xori %39, %true : i1
          %41 = arith.select %39, %cst_1, %cst : f64
          %42:12 = scf.if %39 -> (f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1) {
            %45 = arith.truncf %36 : f64 to f32
            scf.yield %36, %40, %45, %38, %36, %41, %32, %31, %40, %45, %40, %false : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
          } else {
            %45 = arith.addf %5, %cst_3 : f64
            %46 = arith.mulf %31, %45 : f64
            %47 = arith.divf %32, %23#2 : f64
            %48 = arith.addf %5, %41 : f64
            %49 = arith.mulf %46, %48 : f64
            %50 = arith.divf %47, %23#2 : f64
            %51 = arith.mulf %49, %50 : f64
            %52 = memref.load %1[%c1] : memref<12xf64>
            %53 = arith.divf %51, %52 : f64
            %54 = arith.addf %36, %53 : f64
            %55 = arith.divf %53, %54 : f64
            %56 = math.absf %55 : f64
            %57 = arith.select %40, %56, %53 : f64
            %58 = arith.cmpf olt, %56, %cst_0 : f64
            %59 = arith.andi %58, %40 : i1
            %60 = arith.xori %59, %true : i1
            %61 = arith.andi %60, %40 : i1
            %62 = scf.if %59 -> (f32) {
              %64 = arith.truncf %54 : f64 to f32
              scf.yield %64 : f32
            } else {
              scf.yield %23#4 : f32
            }
            %63:3 = scf.if %61 -> (f64, f64, f64) {
              %64 = arith.addf %41, %cst_3 : f64
              %65 = arith.addf %5, %64 : f64
              %66 = arith.mulf %49, %65 : f64
              %67 = arith.divf %50, %23#2 : f64
              %68 = arith.addf %64, %cst_3 : f64
              scf.yield %66, %67, %68 : f64, f64, f64
            } else {
              scf.yield %49, %50, %41 : f64, f64, f64
            }
            scf.yield %54, %61, %62, %57, %54, %63#2, %63#1, %63#0, %61, %62, %61, %61 : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
          }
          %43:3 = scf.if %42#11 -> (f64, i1, f32) {
            %45:4 = scf.if %42#10 -> (f64, f64, f64, f64) {
              %59 = arith.addf %5, %42#5 : f64
              %60 = arith.mulf %42#7, %59 : f64
              %61 = arith.divf %42#6, %23#2 : f64
              %62 = arith.mulf %60, %61 : f64
              %63 = memref.load %1[%c2] : memref<12xf64>
              %64 = arith.divf %62, %63 : f64
              %65 = arith.addf %42#4, %64 : f64
              scf.yield %60, %61, %64, %65 : f64, f64, f64, f64
            } else {
              scf.yield %42#7, %42#6, %42#3, %42#4 : f64, f64, f64, f64
            }
            %46 = arith.divf %45#2, %45#3 : f64
            %47 = math.absf %46 : f64
            %48 = arith.cmpf olt, %47, %cst_0 : f64
            %49 = arith.andi %48, %42#10 : i1
            %50 = arith.xori %49, %true : i1
            %51 = arith.andi %50, %42#8 : i1
            %52 = arith.andi %50, %42#10 : i1
            %53 = arith.xori %42#10, %true : i1
            %54 = arith.andi %42#10, %51 : i1
            %55 = arith.andi %53, %42#8 : i1
            %56 = arith.ori %54, %55 : i1
            %57 = scf.if %49 -> (f32) {
              %59 = arith.truncf %45#3 : f64 to f32
              scf.yield %59 : f32
            } else {
              scf.yield %42#9 : f32
            }
            %58:3 = scf.if %52 -> (f64, i1, f32) {
              %59 = arith.addf %42#5, %cst_3 : f64
              %60 = arith.addf %5, %59 : f64
              %61 = arith.mulf %45#0, %60 : f64
              %62 = arith.divf %45#1, %23#2 : f64
              %63 = arith.addf %59, %cst_3 : f64
              %64 = arith.addf %5, %63 : f64
              %65 = arith.mulf %61, %64 : f64
              %66 = arith.divf %62, %23#2 : f64
              %67 = arith.mulf %65, %66 : f64
              %68 = memref.load %1[%c3] : memref<12xf64>
              %69 = arith.divf %67, %68 : f64
              %70 = arith.addf %45#3, %69 : f64
              %71 = arith.divf %69, %70 : f64
              %72 = math.absf %71 : f64
              %73 = arith.cmpf olt, %72, %cst_0 : f64
              %74 = arith.xori %73, %true : i1
              %75 = arith.andi %74, %56 : i1
              %76:12 = scf.if %73 -> (f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1) {
                %78 = arith.truncf %70 : f64 to f32
                scf.yield %70, %75, %78, %72, %70, %63, %66, %65, %75, %78, %74, %false : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
              } else {
                %78 = arith.addf %63, %cst_3 : f64
                %79 = arith.addf %5, %78 : f64
                %80 = arith.mulf %65, %79 : f64
                %81 = arith.divf %66, %23#2 : f64
                %82 = arith.addf %78, %cst_3 : f64
                %83 = arith.addf %5, %82 : f64
                %84 = arith.mulf %80, %83 : f64
                %85 = arith.divf %81, %23#2 : f64
                %86 = arith.mulf %84, %85 : f64
                %87 = memref.load %1[%c4] : memref<12xf64>
                %88 = arith.divf %86, %87 : f64
                %89 = arith.addf %70, %88 : f64
                %90 = arith.divf %88, %89 : f64
                %91 = math.absf %90 : f64
                %92 = arith.select %74, %91, %88 : f64
                %93 = arith.cmpf olt, %91, %cst_0 : f64
                %94 = arith.andi %93, %74 : i1
                %95 = arith.xori %94, %true : i1
                %96 = arith.andi %95, %75 : i1
                %97 = arith.andi %95, %74 : i1
                %98 = arith.andi %74, %96 : i1
                %99 = scf.if %94 -> (f32) {
                  %101 = arith.truncf %89 : f64 to f32
                  scf.yield %101 : f32
                } else {
                  scf.yield %57 : f32
                }
                %100:3 = scf.if %97 -> (f64, f64, f64) {
                  %101 = arith.addf %82, %cst_3 : f64
                  %102 = arith.addf %5, %101 : f64
                  %103 = arith.mulf %84, %102 : f64
                  %104 = arith.divf %85, %23#2 : f64
                  %105 = arith.addf %101, %cst_3 : f64
                  scf.yield %103, %104, %105 : f64, f64, f64
                } else {
                  scf.yield %84, %85, %82 : f64, f64, f64
                }
                scf.yield %89, %98, %99, %92, %89, %100#2, %100#1, %100#0, %98, %99, %97, %97 : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
              }
              %77:3 = scf.if %76#11 -> (f64, i1, f32) {
                %78:4 = scf.if %76#10 -> (f64, f64, f64, f64) {
                  %92 = arith.addf %5, %76#5 : f64
                  %93 = arith.mulf %76#7, %92 : f64
                  %94 = arith.divf %76#6, %23#2 : f64
                  %95 = arith.mulf %93, %94 : f64
                  %96 = memref.load %1[%c5] : memref<12xf64>
                  %97 = arith.divf %95, %96 : f64
                  %98 = arith.addf %76#4, %97 : f64
                  scf.yield %93, %94, %97, %98 : f64, f64, f64, f64
                } else {
                  scf.yield %76#7, %76#6, %76#3, %76#4 : f64, f64, f64, f64
                }
                %79 = arith.divf %78#2, %78#3 : f64
                %80 = math.absf %79 : f64
                %81 = arith.cmpf olt, %80, %cst_0 : f64
                %82 = arith.andi %81, %76#10 : i1
                %83 = arith.xori %82, %true : i1
                %84 = arith.andi %83, %76#8 : i1
                %85 = arith.andi %83, %76#10 : i1
                %86 = arith.xori %76#10, %true : i1
                %87 = arith.andi %76#10, %84 : i1
                %88 = arith.andi %86, %76#8 : i1
                %89 = arith.ori %87, %88 : i1
                %90 = scf.if %82 -> (f32) {
                  %92 = arith.truncf %78#3 : f64 to f32
                  scf.yield %92 : f32
                } else {
                  scf.yield %76#9 : f32
                }
                %91:3 = scf.if %85 -> (f64, i1, f32) {
                  %92 = arith.addf %76#5, %cst_3 : f64
                  %93 = arith.addf %5, %92 : f64
                  %94 = arith.mulf %78#0, %93 : f64
                  %95 = arith.divf %78#1, %23#2 : f64
                  %96 = arith.addf %92, %cst_3 : f64
                  %97 = arith.addf %5, %96 : f64
                  %98 = arith.mulf %94, %97 : f64
                  %99 = arith.divf %95, %23#2 : f64
                  %100 = arith.mulf %98, %99 : f64
                  %101 = memref.load %1[%c6] : memref<12xf64>
                  %102 = arith.divf %100, %101 : f64
                  %103 = arith.addf %78#3, %102 : f64
                  %104 = arith.divf %102, %103 : f64
                  %105 = math.absf %104 : f64
                  %106 = arith.cmpf olt, %105, %cst_0 : f64
                  %107 = arith.xori %106, %true : i1
                  %108 = arith.andi %107, %89 : i1
                  %109:12 = scf.if %106 -> (f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1) {
                    %111 = arith.truncf %103 : f64 to f32
                    scf.yield %103, %108, %111, %105, %103, %96, %99, %98, %108, %111, %107, %false : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
                  } else {
                    %111 = arith.addf %96, %cst_3 : f64
                    %112 = arith.addf %5, %111 : f64
                    %113 = arith.mulf %98, %112 : f64
                    %114 = arith.divf %99, %23#2 : f64
                    %115 = arith.addf %111, %cst_3 : f64
                    %116 = arith.addf %5, %115 : f64
                    %117 = arith.mulf %113, %116 : f64
                    %118 = arith.divf %114, %23#2 : f64
                    %119 = arith.mulf %117, %118 : f64
                    %120 = memref.load %1[%c7] : memref<12xf64>
                    %121 = arith.divf %119, %120 : f64
                    %122 = arith.addf %103, %121 : f64
                    %123 = arith.divf %121, %122 : f64
                    %124 = math.absf %123 : f64
                    %125 = arith.select %107, %124, %121 : f64
                    %126 = arith.cmpf olt, %124, %cst_0 : f64
                    %127 = arith.andi %126, %107 : i1
                    %128 = arith.xori %127, %true : i1
                    %129 = arith.andi %128, %108 : i1
                    %130 = arith.andi %128, %107 : i1
                    %131 = arith.andi %107, %129 : i1
                    %132 = scf.if %127 -> (f32) {
                      %134 = arith.truncf %122 : f64 to f32
                      scf.yield %134 : f32
                    } else {
                      scf.yield %90 : f32
                    }
                    %133:3 = scf.if %130 -> (f64, f64, f64) {
                      %134 = arith.addf %115, %cst_3 : f64
                      %135 = arith.addf %5, %134 : f64
                      %136 = arith.mulf %117, %135 : f64
                      %137 = arith.divf %118, %23#2 : f64
                      %138 = arith.addf %134, %cst_3 : f64
                      scf.yield %136, %137, %138 : f64, f64, f64
                    } else {
                      scf.yield %117, %118, %115 : f64, f64, f64
                    }
                    scf.yield %122, %131, %132, %125, %122, %133#2, %133#1, %133#0, %131, %132, %130, %130 : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
                  }
                  %110:3 = scf.if %109#11 -> (f64, i1, f32) {
                    %111:4 = scf.if %109#10 -> (f64, f64, f64, f64) {
                      %125 = arith.addf %5, %109#5 : f64
                      %126 = arith.mulf %109#7, %125 : f64
                      %127 = arith.divf %109#6, %23#2 : f64
                      %128 = arith.mulf %126, %127 : f64
                      %129 = memref.load %1[%c8] : memref<12xf64>
                      %130 = arith.divf %128, %129 : f64
                      %131 = arith.addf %109#4, %130 : f64
                      scf.yield %126, %127, %130, %131 : f64, f64, f64, f64
                    } else {
                      scf.yield %109#7, %109#6, %109#3, %109#4 : f64, f64, f64, f64
                    }
                    %112 = arith.divf %111#2, %111#3 : f64
                    %113 = math.absf %112 : f64
                    %114 = arith.cmpf olt, %113, %cst_0 : f64
                    %115 = arith.andi %114, %109#10 : i1
                    %116 = arith.xori %115, %true : i1
                    %117 = arith.andi %116, %109#8 : i1
                    %118 = arith.andi %116, %109#10 : i1
                    %119 = arith.xori %109#10, %true : i1
                    %120 = arith.andi %109#10, %117 : i1
                    %121 = arith.andi %119, %109#8 : i1
                    %122 = arith.ori %120, %121 : i1
                    %123 = scf.if %115 -> (f32) {
                      %125 = arith.truncf %111#3 : f64 to f32
                      scf.yield %125 : f32
                    } else {
                      scf.yield %109#9 : f32
                    }
                    %124:3 = scf.if %118 -> (f64, i1, f32) {
                      %125 = arith.addf %109#5, %cst_3 : f64
                      %126 = arith.addf %5, %125 : f64
                      %127 = arith.mulf %111#0, %126 : f64
                      %128 = arith.divf %111#1, %23#2 : f64
                      %129 = arith.addf %125, %cst_3 : f64
                      %130 = arith.addf %5, %129 : f64
                      %131 = arith.mulf %127, %130 : f64
                      %132 = arith.divf %128, %23#2 : f64
                      %133 = arith.mulf %131, %132 : f64
                      %134 = memref.load %1[%c9] : memref<12xf64>
                      %135 = arith.divf %133, %134 : f64
                      %136 = arith.addf %111#3, %135 : f64
                      %137 = arith.divf %135, %136 : f64
                      %138 = math.absf %137 : f64
                      %139 = arith.cmpf olt, %138, %cst_0 : f64
                      %140 = arith.xori %139, %true : i1
                      %141 = arith.andi %140, %122 : i1
                      %142:12 = scf.if %139 -> (f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1) {
                        %144 = arith.truncf %136 : f64 to f32
                        scf.yield %136, %141, %144, %138, %136, %129, %132, %131, %141, %144, %140, %false : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
                      } else {
                        %144 = arith.addf %129, %cst_3 : f64
                        %145 = arith.addf %5, %144 : f64
                        %146 = arith.mulf %131, %145 : f64
                        %147 = arith.divf %132, %23#2 : f64
                        %148 = arith.addf %144, %cst_3 : f64
                        %149 = arith.addf %5, %148 : f64
                        %150 = arith.mulf %146, %149 : f64
                        %151 = arith.divf %147, %23#2 : f64
                        %152 = arith.mulf %150, %151 : f64
                        %153 = memref.load %1[%c10] : memref<12xf64>
                        %154 = arith.divf %152, %153 : f64
                        %155 = arith.addf %136, %154 : f64
                        %156 = arith.divf %154, %155 : f64
                        %157 = math.absf %156 : f64
                        %158 = arith.select %140, %157, %154 : f64
                        %159 = arith.cmpf olt, %157, %cst_0 : f64
                        %160 = arith.andi %159, %140 : i1
                        %161 = arith.xori %160, %true : i1
                        %162 = arith.andi %161, %141 : i1
                        %163 = arith.andi %161, %140 : i1
                        %164 = arith.andi %140, %162 : i1
                        %165 = scf.if %160 -> (f32) {
                          %167 = arith.truncf %155 : f64 to f32
                          scf.yield %167 : f32
                        } else {
                          scf.yield %123 : f32
                        }
                        %166:3 = scf.if %163 -> (f64, f64, f64) {
                          %167 = arith.addf %148, %cst_3 : f64
                          %168 = arith.addf %5, %167 : f64
                          %169 = arith.mulf %150, %168 : f64
                          %170 = arith.divf %151, %23#2 : f64
                          %171 = arith.addf %167, %cst_3 : f64
                          scf.yield %169, %170, %171 : f64, f64, f64
                        } else {
                          scf.yield %150, %151, %148 : f64, f64, f64
                        }
                        scf.yield %155, %164, %165, %158, %155, %166#2, %166#1, %166#0, %164, %165, %163, %163 : f64, i1, f32, f64, f64, f64, f64, f64, i1, f32, i1, i1
                      }
                      %143:3 = scf.if %142#11 -> (f64, i1, f32) {
                        %144:2 = scf.if %142#10 -> (f64, f64) {
                          %156 = arith.addf %5, %142#5 : f64
                          %157 = arith.mulf %142#7, %156 : f64
                          %158 = arith.divf %142#6, %23#2 : f64
                          %159 = arith.mulf %157, %158 : f64
                          %160 = memref.load %1[%c11] : memref<12xf64>
                          %161 = arith.divf %159, %160 : f64
                          %162 = arith.addf %142#4, %161 : f64
                          scf.yield %161, %162 : f64, f64
                        } else {
                          scf.yield %142#3, %142#4 : f64, f64
                        }
                        %145 = arith.divf %144#0, %144#1 : f64
                        %146 = math.absf %145 : f64
                        %147 = arith.cmpf olt, %146, %cst_0 : f64
                        %148 = arith.andi %147, %142#10 : i1
                        %149 = arith.xori %148, %true : i1
                        %150 = arith.andi %149, %142#8 : i1
                        %151 = arith.xori %142#10, %true : i1
                        %152 = arith.andi %142#10, %150 : i1
                        %153 = arith.andi %151, %142#8 : i1
                        %154 = arith.ori %152, %153 : i1
                        %155 = scf.if %148 -> (f32) {
                          %156 = arith.truncf %144#1 : f64 to f32
                          scf.yield %156 : f32
                        } else {
                          scf.yield %142#9 : f32
                        }
                        scf.yield %144#1, %154, %155 : f64, i1, f32
                      } else {
                        scf.yield %142#0, %142#1, %142#2 : f64, i1, f32
                      }
                      scf.yield %143#0, %143#1, %143#2 : f64, i1, f32
                    } else {
                      scf.yield %111#3, %122, %123 : f64, i1, f32
                    }
                    scf.yield %124#0, %124#1, %124#2 : f64, i1, f32
                  } else {
                    scf.yield %109#0, %109#1, %109#2 : f64, i1, f32
                  }
                  scf.yield %110#0, %110#1, %110#2 : f64, i1, f32
                } else {
                  scf.yield %78#3, %89, %90 : f64, i1, f32
                }
                scf.yield %91#0, %91#1, %91#2 : f64, i1, f32
              } else {
                scf.yield %76#0, %76#1, %76#2 : f64, i1, f32
              }
              scf.yield %77#0, %77#1, %77#2 : f64, i1, f32
            } else {
              scf.yield %45#3, %56, %57 : f64, i1, f32
            }
            scf.yield %58#0, %58#1, %58#2 : f64, i1, f32
          } else {
            scf.yield %42#0, %42#1, %42#2 : f64, i1, f32
          }
          %44 = scf.if %43#1 -> (f32) {
            %45 = arith.truncf %43#0 : f64 to f32
            scf.yield %45 : f32
          } else {
            scf.yield %43#2 : f32
          }
          scf.yield %44 : f32
        } else {
          scf.yield %23#4 : f32
        }
        scf.yield %24 : f32
      } else {
        scf.yield %17#1 : f32
      }
      memref.store %18, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
  func.func private @floorf(f32) -> f32
}

